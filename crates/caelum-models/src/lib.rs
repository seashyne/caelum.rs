//! Model implementations: Unified Llama/Qwen2.5 GGUF runner (Batched Prefill GEMM + Q8_0 KV Cache
//! + Automatic Prefix Caching) and Native Celestia (True Compressed MLA + DeepSeekMoE).

pub mod cache;
pub mod celestia;
pub mod deepseek2;
pub mod llama_qwen;
pub mod moe;
pub mod rope;
pub mod sampler;

use caelum_core::{CaelumError, Result};
use caelum_gguf::GgufFile;
use caelum_kernels::Backend;
use caelum_tokenizer::{JsonGrammar, LogitGrammar, Tokenizer};
use std::time::Instant;

pub use cache::{KvCache, KvPrecision, MlaLatentCache};
pub use celestia::{write_synthetic_celestia_gguf, CelestiaConfig, CelestiaModel};
pub use deepseek2::{DeepSeek2Config, DeepSeek2Model};
pub use llama_qwen::{LlamaQwenConfig, LlamaQwenModel};
pub use moe::{ExpertGating, MoeConfig};
pub use rope::{RopeConfig, RopeScalingType, RopeTable};
pub use sampler::{Sampler, SamplerConfig};

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    /// Number of prompt tokens served instantaneously from KV Prefix Cache (0 compute).
    pub cached_prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub kv_cache_bytes: usize,
}

impl GenerationStats {
    /// Tokens actually computed during prefill (excluding prefix cache hits).
    pub fn computed_prefill_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)
    }

    pub fn prefill_tps(&self) -> f64 {
        let n = self.computed_prefill_tokens();
        if self.prefill_secs > 0.0 && n > 0 {
            n as f64 / self.prefill_secs
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

enum ModelInner<'a> {
    LlamaQwen(LlamaQwenModel<'a>),
    DeepSeek2(DeepSeek2Model<'a>),
    Celestia(CelestiaModel<'a>),
}

/// Unified model container with automatic KV Prefix Caching across turns and API requests.
pub struct CaelumModel<'a> {
    inner: ModelInner<'a>,
    cached_tokens: Vec<u32>,
}

impl<'a> CaelumModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_len: Option<usize>) -> Result<Self> {
        Self::from_gguf_with_kv(gguf, max_seq_len, KvPrecision::F32)
    }

    pub fn from_gguf_with_kv(
        gguf: &'a GgufFile,
        max_seq_len: Option<usize>,
        kv_precision: KvPrecision,
    ) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or("llama");
        let inner = match arch {
            "celestia" => ModelInner::Celestia(CelestiaModel::from_gguf(gguf, max_seq_len)?),
            "deepseek2" => ModelInner::DeepSeek2(DeepSeek2Model::from_gguf(gguf, max_seq_len)?),
            "llama" | "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "mistral" => {
                ModelInner::LlamaQwen(LlamaQwenModel::from_gguf_with_kv(
                    gguf,
                    max_seq_len,
                    kv_precision,
                )?)
            }
            other => {
                return Err(CaelumError::InvalidFormat(format!(
                    "unsupported GGUF architecture '{other}' (supported: llama, qwen2, qwen2moe, qwen3, qwen3moe, deepseek2, mistral, celestia)"
                )))
            }
        };
        Ok(Self {
            inner,
            cached_tokens: Vec::new(),
        })
    }

    /// Repacks compatible `Q4_K`/`Q6_K` weights into cache-aligned 8-row SIMD tiles for
    /// maximum AVX-VNNI / AVX2+FMA decode and prefill speed.
    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        match &mut self.inner {
            ModelInner::LlamaQwen(m) => m.optimize_for_backend(backend),
            ModelInner::DeepSeek2(m) => m.optimize_for_backend(backend),
            ModelInner::Celestia(_) => {}
        }
    }

    pub fn packed_bytes(&self) -> usize {
        match &self.inner {
            ModelInner::LlamaQwen(m) => m.packed_bytes(),
            ModelInner::DeepSeek2(m) => m.packed_bytes(),
            ModelInner::Celestia(_) => 0,
        }
    }

    pub fn architecture(&self) -> &str {
        match &self.inner {
            ModelInner::LlamaQwen(m) => &m.config.arch,
            ModelInner::DeepSeek2(_) => "deepseek2",
            ModelInner::Celestia(_) => "celestia",
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            ModelInner::LlamaQwen(m) => m.config.vocab_size,
            ModelInner::DeepSeek2(m) => m.config.vocab_size,
            ModelInner::Celestia(m) => m.config.vocab_size,
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match &self.inner {
            ModelInner::LlamaQwen(m) => m.max_seq_len(),
            ModelInner::DeepSeek2(m) => m.max_seq_len(),
            ModelInner::Celestia(m) => m.max_seq_len(),
        }
    }

    pub fn kv_cache_bytes(&self) -> usize {
        match &self.inner {
            ModelInner::LlamaQwen(m) => m.kv_cache_bytes(),
            ModelInner::DeepSeek2(m) => m.kv_cache_bytes(),
            ModelInner::Celestia(m) => m.kv_cache_bytes(),
        }
    }

    pub fn kv_mode_label(&self) -> &'static str {
        match &self.inner {
            ModelInner::LlamaQwen(m) => match m.kv_precision() {
                KvPrecision::F32 => "GQA-F32",
                KvPrecision::Q8_0 => "GQA-Q8_0",
            },
            ModelInner::DeepSeek2(_) => "MLA-Absorbed",
            ModelInner::Celestia(_) => "MLA-Latent",
        }
    }

    pub fn reset_cache(&mut self) {
        self.cached_tokens.clear();
        match &mut self.inner {
            ModelInner::LlamaQwen(m) => m.reset_cache(),
            ModelInner::DeepSeek2(m) => m.reset_cache(),
            ModelInner::Celestia(m) => m.reset_cache(),
        }
    }

    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32]> {
        match &mut self.inner {
            ModelInner::LlamaQwen(m) => m.forward_token(backend, token_id, pos, compute_logits),
            ModelInner::DeepSeek2(m) => m.forward_token(backend, token_id, pos, compute_logits),
            ModelInner::Celestia(m) => m.forward_token(backend, token_id, pos, compute_logits),
        }
    }

    pub fn forward_batch(
        &mut self,
        backend: &dyn Backend,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        match &mut self.inner {
            ModelInner::LlamaQwen(m) => {
                m.forward_batch(backend, tokens, start_pos, compute_last_logits)
            }
            ModelInner::DeepSeek2(m) => {
                m.forward_batch(backend, tokens, start_pos, compute_last_logits)
            }
            ModelInner::Celestia(m) => {
                let n = tokens.len();
                for (i, &tok) in tokens.iter().enumerate() {
                    let is_last = i + 1 == n && compute_last_logits;
                    m.forward_token(backend, tok, start_pos + i, is_last)?;
                }
                if n > 0 && compute_last_logits {
                    m.forward_token(backend, tokens[n - 1], start_pos + n - 1, true)
                } else {
                    Ok(&[])
                }
            }
        }
    }

    /// Runs prefix-cached batched prefill + autoregressive decode, invoking `on_token(id, piece)`.
    ///
    /// Automatically reuses the longest matching token prefix already stored in the KV cache
    /// (ideal for repeated Lenout system prompts and multi-turn `caelum chat`).
    pub fn generate<F>(
        &mut self,
        backend: &dyn Backend,
        tokenizer: &Tokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler_cfg: &SamplerConfig,
        constrain_json: bool,
        on_token: F,
    ) -> Result<(String, GenerationStats)>
    where
        F: FnMut(u32, &str),
    {
        let mut json_g = if constrain_json {
            Some(JsonGrammar::new())
        } else {
            None
        };
        let grammar_ref: Option<&mut dyn LogitGrammar> = json_g
            .as_mut()
            .map(|g| g as &mut dyn LogitGrammar);
        self.generate_with_grammar(
            backend,
            tokenizer,
            prompt_tokens,
            max_new_tokens,
            sampler_cfg,
            grammar_ref,
            on_token,
        )
    }

    /// Runs prefix-cached batched prefill + autoregressive decode with an optional arbitrary
    /// `LogitGrammar` (e.g., `JsonGrammar` or strict `SchemaGrammar::lenout_proposal(mode)`).
    pub fn generate_with_grammar<F>(
        &mut self,
        backend: &dyn Backend,
        tokenizer: &Tokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler_cfg: &SamplerConfig,
        grammar: Option<&mut dyn LogitGrammar>,
        on_token: F,
    ) -> Result<(String, GenerationStats)>
    where
        F: FnMut(u32, &str),
    {
        self.generate_with_stop(
            backend,
            tokenizer,
            prompt_tokens,
            max_new_tokens,
            sampler_cfg,
            grammar,
            &[],
            on_token,
        )
    }

    /// Runs prefix-cached batched prefill + autoregressive decode with optional `LogitGrammar`
    /// and string `stop_sequences` (used by OpenAI `/v1/chat/completions` & FIM `/v1/completions`).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_stop<F>(
        &mut self,
        backend: &dyn Backend,
        tokenizer: &Tokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler_cfg: &SamplerConfig,
        mut grammar: Option<&mut dyn LogitGrammar>,
        stop_sequences: &[String],
        mut on_token: F,
    ) -> Result<(String, GenerationStats)>
    where
        F: FnMut(u32, &str),
    {
        if prompt_tokens.is_empty() {
            return Err(CaelumError::Runtime("prompt_tokens cannot be empty".into()));
        }

        let max_ctx = self.max_seq_len();
        if prompt_tokens.len() >= max_ctx {
            return Err(CaelumError::Runtime(format!(
                "prompt length {} exceeds max context {max_ctx}",
                prompt_tokens.len()
            )));
        }

        // 1. Automatic KV Prefix Cache match (always recompute at least the final prompt token for logits)
        let mut common = 0usize;
        let max_reuse = self.cached_tokens.len().min(prompt_tokens.len() - 1);
        while common < max_reuse && self.cached_tokens[common] == prompt_tokens[common] {
            common += 1;
        }
        self.cached_tokens.truncate(common);

        // 2. Batched Prefill for uncached suffix `prompt_tokens[common..]`
        let t_prefill = Instant::now();
        let uncached = &prompt_tokens[common..];
        let mut last_logits_vec = match &mut self.inner {
            ModelInner::LlamaQwen(m) => {
                m.forward_batch(backend, uncached, common, true)?.to_vec()
            }
            ModelInner::DeepSeek2(m) => {
                m.forward_batch(backend, uncached, common, true)?.to_vec()
            }
            ModelInner::Celestia(m) => {
                let mut out = Vec::new();
                for (i, &tok) in uncached.iter().enumerate() {
                    let is_last = i + 1 == uncached.len();
                    let logits = m.forward_token(backend, tok, common + i, is_last)?;
                    if is_last {
                        out = logits.to_vec();
                    }
                }
                out
            }
        };
        self.cached_tokens.extend_from_slice(uncached);
        let prefill_secs = t_prefill.elapsed().as_secs_f64();

        // 3. Autoregressive Decode
        let mut sampler = Sampler::new(sampler_cfg.clone());
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
                let raw_bytes = tokenizer.cached_token_bytes(next_tok);
                g.advance_bytes(raw_bytes);
            }

            let piece = decoder.push(tokenizer, next_tok);
            full_text.push_str(&piece);

            let mut hit_stop = false;
            if !stop_sequences.is_empty() {
                for stop in stop_sequences {
                    if !stop.is_empty() && full_text.ends_with(stop.as_str()) {
                        let new_len = full_text.len() - stop.len();
                        full_text.truncate(new_len);
                        hit_stop = true;
                        break;
                    }
                }
            }
            if !hit_stop {
                on_token(next_tok, &piece);
            }

            history.push(next_tok);
            generated += 1;

            if hit_stop {
                break;
            }

            if let Some(ref g) = grammar {
                if g.is_complete() {
                    break;
                }
            }

            if generated < max_new_tokens && pos < max_ctx {
                let logits = self.forward_token(backend, next_tok, pos, true)?;
                last_logits_vec.copy_from_slice(logits);
                self.cached_tokens.push(next_tok);
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
            cached_prompt_tokens: common,
            generated_tokens: generated,
            prefill_secs,
            decode_secs,
            kv_cache_bytes: self.kv_cache_bytes(),
        };

        Ok((full_text, stats))
    }
}
