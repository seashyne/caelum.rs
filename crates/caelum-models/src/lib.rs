//! Model implementations: Unified Llama/Qwen2.5 GGUF runner + Native Celestia (MLA + DeepSeekMoE).

pub mod cache;
pub mod celestia;
pub mod llama_qwen;
pub mod sampler;

use caelum_core::{CaelumError, Result};
use caelum_gguf::GgufFile;
use caelum_kernels::Backend;
use caelum_tokenizer::{JsonGrammar, Tokenizer};
use std::time::Instant;

pub use cache::{KvCache, MlaLatentCache};
pub use celestia::{write_synthetic_celestia_gguf, CelestiaConfig, CelestiaModel};
pub use llama_qwen::{LlamaQwenConfig, LlamaQwenModel};
pub use sampler::{Sampler, SamplerConfig};

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub kv_cache_bytes: usize,
}

impl GenerationStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 {
            self.prompt_tokens as f64 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.generated_tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }
}

/// Unified model enum auto-dispatched from GGUF `general.architecture`.
pub enum CaelumModel<'a> {
    LlamaQwen(LlamaQwenModel<'a>),
    Celestia(CelestiaModel<'a>),
}

impl<'a> CaelumModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_len: Option<usize>) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or("llama");
        match arch {
            "celestia" => Ok(Self::Celestia(CelestiaModel::from_gguf(gguf, max_seq_len)?)),
            "llama" | "qwen2" | "qwen3" | "mistral" => {
                Ok(Self::LlamaQwen(LlamaQwenModel::from_gguf(gguf, max_seq_len)?))
            }
            other => Err(CaelumError::InvalidFormat(format!(
                "unsupported GGUF architecture '{other}' (supported: llama, qwen2, mistral, celestia)"
            ))),
        }
    }

    pub fn architecture(&self) -> &str {
        match self {
            Self::LlamaQwen(m) => &m.config.arch,
            Self::Celestia(_) => "celestia",
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::LlamaQwen(m) => m.config.vocab_size,
            Self::Celestia(m) => m.config.vocab_size,
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::LlamaQwen(m) => m.max_seq_len(),
            Self::Celestia(m) => m.max_seq_len(),
        }
    }

    pub fn kv_cache_bytes(&self) -> usize {
        match self {
            Self::LlamaQwen(m) => m.kv_cache_bytes(),
            Self::Celestia(m) => m.kv_cache_bytes(),
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Self::LlamaQwen(m) => m.reset_cache(),
            Self::Celestia(m) => m.reset_cache(),
        }
    }

    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
    ) -> Result<&[f32]> {
        match self {
            Self::LlamaQwen(m) => m.forward_token(backend, token_id, pos),
            Self::Celestia(m) => m.forward_token(backend, token_id, pos),
        }
    }

    /// Runs prefill + autoregressive decode, invoking `on_token(token_id, piece)` for each generated token.
    pub fn generate<F>(
        &mut self,
        backend: &dyn Backend,
        tokenizer: &Tokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler_cfg: &SamplerConfig,
        constrain_json: bool,
        mut on_token: F,
    ) -> Result<(String, GenerationStats)>
    where
        F: FnMut(u32, &str),
    {
        if prompt_tokens.is_empty() {
            return Err(CaelumError::Runtime("prompt_tokens cannot be empty".into()));
        }
        self.reset_cache();

        let max_ctx = self.max_seq_len();
        if prompt_tokens.len() >= max_ctx {
            return Err(CaelumError::Runtime(format!(
                "prompt length {} exceeds max context {max_ctx}",
                prompt_tokens.len()
            )));
        }

        // 1. Prefill prompt tokens
        let t_prefill = Instant::now();
        let mut last_logits_vec = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let logits = self.forward_token(backend, tok, pos)?;
            if pos + 1 == prompt_tokens.len() {
                last_logits_vec = logits.to_vec();
            }
        }
        let prefill_secs = t_prefill.elapsed().as_secs_f64();

        // 2. Autoregressive Decode
        let mut sampler = Sampler::new(sampler_cfg.clone());
        let mut grammar = if constrain_json {
            Some(JsonGrammar::new())
        } else {
            None
        };
        let mut history: Vec<u32> = prompt_tokens.to_vec();
        let mut decoder = tokenizer.stream_decoder(true);
        let mut full_text = String::new();

        let t_decode = Instant::now();
        let mut generated = 0usize;
        let mut pos = prompt_tokens.len();

        while generated < max_new_tokens && pos < max_ctx {
            if let Some(ref g) = grammar {
                g.mask_logits(&mut last_logits_vec, tokenizer);
            }

            let next_tok = sampler.sample(&mut last_logits_vec, &history);
            if tokenizer.is_stop_token(next_tok) {
                break;
            }

            if let Some(ref mut g) = grammar {
                let raw_bytes = tokenizer.token_to_bytes(next_tok, true);
                g.advance_bytes(&raw_bytes);
            }

            let piece = decoder.push(tokenizer, next_tok);
            full_text.push_str(&piece);
            on_token(next_tok, &piece);

            history.push(next_tok);
            generated += 1;

            if let Some(ref g) = grammar {
                if g.is_complete() {
                    break;
                }
            }

            if generated < max_new_tokens && pos < max_ctx {
                let logits = self.forward_token(backend, next_tok, pos)?;
                last_logits_vec.copy_from_slice(logits);
                pos += 1;
            }
        }

        let tail = decoder.flush();
        if !tail.is_empty() {
            full_text.push_str(&tail);
            on_token(tokenizer.eos_token_id, &tail);
        }

        let decode_secs = t_decode.elapsed().as_secs_f64();
        let stats = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            prefill_secs,
            decode_secs,
            kv_cache_bytes: self.kv_cache_bytes(),
        };

        Ok((full_text, stats))
    }
}
