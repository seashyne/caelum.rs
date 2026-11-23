//! Unified Llama / Qwen2 / Qwen2.5 / Mistral GGUF Transformer runtime.
//!
//! Supports:
//! - Zero-copy `mmap` weights (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K`, `Q6_K`)
//! - Grouped Query Attention (GQA)
//! - Exact `Normal` (Llama) and `NeoX` (Qwen2/Qwen2.5) RoPE with position offset
//! - Optional QKV biases (Qwen2 / Qwen2.5)
//! - Tied or untied `output.weight` (`lm_head`)
//! - Zero-allocation decode hot path with pre-allocated scratch buffers

use crate::cache::KvCache;
use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, DType, Result, TensorView, QK8_0,
    QK_K,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, RopeStyle};

#[derive(Debug, Clone)]
pub struct LlamaQwenConfig {
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rope_style: RopeStyle,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl LlamaQwenConfig {
    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or("llama").to_string();
        let p = &arch;

        let hidden_size = gguf.require_usize(&format!("{p}.embedding_length"))?;
        let num_layers = gguf.require_usize(&format!("{p}.block_count"))?;
        let intermediate_size = gguf.require_usize(&format!("{p}.feed_forward_length"))?;
        let num_heads = gguf.require_usize(&format!("{p}.attention.head_count"))?;
        let num_kv_heads = gguf
            .get_meta(&format!("{p}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(num_heads);
        let head_dim = gguf
            .get_meta(&format!("{p}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size / num_heads);
        let rope_dim = gguf
            .get_meta(&format!("{p}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_theta = gguf
            .get_meta(&format!("{p}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| if arch.starts_with("qwen") { 1_000_000.0 } else { 10_000.0 });
        let rms_norm_eps = gguf
            .get_meta(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);
        let ctx_len = gguf
            .get_meta(&format!("{p}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));

        let embd = gguf.tensor("token_embd.weight")?;
        let (vocab_size, _) = embd.dims_2d()?;

        let rope_style = if arch.starts_with("qwen") {
            RopeStyle::NeoX
        } else {
            RopeStyle::Normal
        };

        Ok(Self {
            arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            rope_style,
            rms_norm_eps,
            max_seq_len,
        })
    }
}

struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    wq: TensorView<'a>,
    wk: TensorView<'a>,
    wv: TensorView<'a>,
    wo: TensorView<'a>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: TensorView<'a>,
    w_up: TensorView<'a>,
    w_down: TensorView<'a>,
}

struct ScratchBuf {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    logits: Vec<f32>,
    xq8: Vec<BlockQ8_0>,
    hbq8: Vec<BlockQ8_0>,
    xq8k: Vec<BlockQ8_K>,
    hbq8k: Vec<BlockQ8_K>,
}

impl ScratchBuf {
    fn new(cfg: &LlamaQwenConfig) -> Self {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        Self {
            x: vec![0.0; cfg.hidden_size],
            xb: vec![0.0; cfg.hidden_size],
            xb2: vec![0.0; q_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; cfg.max_seq_len],
            hb: vec![0.0; cfg.intermediate_size],
            hb2: vec![0.0; cfg.intermediate_size],
            logits: vec![0.0; cfg.vocab_size],
            xq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (cfg.hidden_size / QK8_0).max(1)],
            hbq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (cfg.intermediate_size / QK8_0).max(1)],
            xq8k: vec![BlockQ8_K::default(); (cfg.hidden_size / QK_K).max(1)],
            hbq8k: vec![BlockQ8_K::default(); (cfg.intermediate_size / QK_K).max(1)],
        }
    }
}

pub struct LlamaQwenModel<'a> {
    pub config: LlamaQwenConfig,
    token_embd: TensorView<'a>,
    layers: Vec<LayerWeights<'a>>,
    output_norm: Vec<f32>,
    output_weight: TensorView<'a>,
    cache: KvCache,
    scratch: ScratchBuf,
}

impl<'a> LlamaQwenModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let config = LlamaQwenConfig::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            // Tied word embeddings (common in Qwen2.5-0.5B/1.5B, Llama 3.2 1B)
            token_embd
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let bq = if gguf.has_tensor(&format!("blk.{i}.attn_q.bias")) {
                Some(gguf.tensor(&format!("blk.{i}.attn_q.bias"))?.to_f32_vec()?)
            } else {
                None
            };
            let bk = if gguf.has_tensor(&format!("blk.{i}.attn_k.bias")) {
                Some(gguf.tensor(&format!("blk.{i}.attn_k.bias"))?.to_f32_vec()?)
            } else {
                None
            };
            let bv = if gguf.has_tensor(&format!("blk.{i}.attn_v.bias")) {
                Some(gguf.tensor(&format!("blk.{i}.attn_v.bias"))?.to_f32_vec()?)
            } else {
                None
            };

            layers.push(LayerWeights {
                attn_norm: gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?.to_f32_vec()?,
                wq: gguf.tensor(&format!("blk.{i}.attn_q.weight"))?,
                wk: gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
                wv: gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
                wo: gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
                bq,
                bk,
                bv,
                ffn_norm: gguf.tensor(&format!("blk.{i}.ffn_norm.weight"))?.to_f32_vec()?,
                w_gate: gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        let kv_dim = config.num_kv_heads * config.head_dim;
        let cache = KvCache::new(config.num_layers, config.max_seq_len, kv_dim);
        let scratch = ScratchBuf::new(&config);

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output_weight,
            cache,
            scratch,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.cache.memory_bytes()
    }

    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// Runs one autoregressive step for `token_id` at sequence position `pos` and returns `&[f32]` logits.
    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
    ) -> Result<&[f32]> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let kv_mul = cfg.num_heads / cfg.num_kv_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();
        let can_q8_h = cfg.hidden_size % QK8_0 == 0;
        let can_q8_ffn = cfg.intermediate_size % QK8_0 == 0;
        let can_q8k_h = cfg.hidden_size % QK_K == 0;
        let can_q8k_ffn = cfg.intermediate_size % QK_K == 0;

        // 1. Embedding lookup
        self.token_embd
            .dequantize_row(token_id as usize, &mut self.scratch.x)?;

        // 2. Transformer Decoder Layers
        for l in 0..cfg.num_layers {
            let layer = &self.layers[l];

            // Pre-RMSNorm for Attention
            backend.rms_norm(
                &self.scratch.x,
                &layer.attn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            // Q, K, V projections (share dynamic Q8_K or Q8_0 quantization of xb across Q/K/V)
            if can_q8k_h
                && matches!(layer.wq.dtype, DType::Q4_K | DType::Q6_K)
                && matches!(layer.wk.dtype, DType::Q4_K | DType::Q6_K)
                && matches!(layer.wv.dtype, DType::Q4_K | DType::Q6_K)
            {
                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                backend.matvec_q8k_act(&layer.wq, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.q)?;
                backend.matvec_q8k_act(&layer.wk, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.k)?;
                backend.matvec_q8k_act(&layer.wv, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.v)?;
            } else if can_q8_h
                && matches!(layer.wq.dtype, DType::Q4_0 | DType::Q8_0)
                && matches!(layer.wk.dtype, DType::Q4_0 | DType::Q8_0)
                && matches!(layer.wv.dtype, DType::Q4_0 | DType::Q8_0)
            {
                quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                backend.matvec_q8_act(&layer.wq, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.q)?;
                backend.matvec_q8_act(&layer.wk, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.k)?;
                backend.matvec_q8_act(&layer.wv, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.v)?;
            } else {
                backend.matvec(&layer.wq, &self.scratch.xb, &mut self.scratch.q)?;
                backend.matvec(&layer.wk, &self.scratch.xb, &mut self.scratch.k)?;
                backend.matvec(&layer.wv, &self.scratch.xb, &mut self.scratch.v)?;
            }

            // Optional QKV biases (Qwen2 / Qwen2.5)
            if let Some(bq) = &layer.bq {
                for (q, &b) in self.scratch.q.iter_mut().zip(bq.iter()) {
                    *q += b;
                }
            }
            if let Some(bk) = &layer.bk {
                for (k, &b) in self.scratch.k.iter_mut().zip(bk.iter()) {
                    *k += b;
                }
            }
            if let Some(bv) = &layer.bv {
                for (v, &b) in self.scratch.v.iter_mut().zip(bv.iter()) {
                    *v += b;
                }
            }

            // Apply RoPE at exact sequence position `pos`
            for h in 0..cfg.num_heads {
                let qh = &mut self.scratch.q[h * head_dim..(h + 1) * head_dim];
                backend.apply_rope(qh, pos, cfg.rope_dim, cfg.rope_theta, cfg.rope_style);
            }
            for kh in 0..cfg.num_kv_heads {
                let kh_slice = &mut self.scratch.k[kh * head_dim..(kh + 1) * head_dim];
                backend.apply_rope(kh_slice, pos, cfg.rope_dim, cfg.rope_theta, cfg.rope_style);
            }

            // Store K, V into cache slot `pos`
            self.cache.k_slot_mut(l, pos).copy_from_slice(&self.scratch.k);
            self.cache.v_slot_mut(l, pos).copy_from_slice(&self.scratch.v);

            // Grouped Query Attention over `0..=pos`
            self.scratch.xb2.fill(0.0);
            for h in 0..cfg.num_heads {
                let qh = &self.scratch.q[h * head_dim..(h + 1) * head_dim];
                let kv_h = h / kv_mul;
                let kv_off = kv_h * head_dim;

                for t in 0..=pos {
                    let kt = &self.cache.k_slot(l, t)[kv_off..kv_off + head_dim];
                    self.scratch.att[t] = backend.dot_f32(qh, kt) * inv_sqrt_hd;
                }
                backend.softmax(&mut self.scratch.att[..=pos]);

                let out_h = &mut self.scratch.xb2[h * head_dim..(h + 1) * head_dim];
                for t in 0..=pos {
                    let a = self.scratch.att[t];
                    let vt = &self.cache.v_slot(l, t)[kv_off..kv_off + head_dim];
                    for d in 0..head_dim {
                        out_h[d] += a * vt[d];
                    }
                }
            }

            // Output projection + Residual
            backend.matvec(&layer.wo, &self.scratch.xb2, &mut self.scratch.xb)?;
            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                *x += dx;
            }

            // Pre-RMSNorm for SwiGLU FFN
            backend.rms_norm(
                &self.scratch.x,
                &layer.ffn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            if can_q8k_h
                && matches!(layer.w_gate.dtype, DType::Q4_K | DType::Q6_K)
                && matches!(layer.w_up.dtype, DType::Q4_K | DType::Q6_K)
            {
                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                backend.matvec_q8k_act(&layer.w_gate, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb)?;
                backend.matvec_q8k_act(&layer.w_up, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb2)?;
            } else if can_q8_h
                && matches!(layer.w_gate.dtype, DType::Q4_0 | DType::Q8_0)
                && matches!(layer.w_up.dtype, DType::Q4_0 | DType::Q8_0)
            {
                quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                backend.matvec_q8_act(&layer.w_gate, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.hb)?;
                backend.matvec_q8_act(&layer.w_up, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.hb2)?;
            } else {
                backend.matvec(&layer.w_gate, &self.scratch.xb, &mut self.scratch.hb)?;
                backend.matvec(&layer.w_up, &self.scratch.xb, &mut self.scratch.hb2)?;
            }

            backend.swiglu_inplace(&mut self.scratch.hb, &self.scratch.hb2);

            if can_q8k_ffn && matches!(layer.w_down.dtype, DType::Q4_K | DType::Q6_K) {
                quantize_row_q8_k(&self.scratch.hb, &mut self.scratch.hbq8k);
                backend.matvec_q8k_act(&layer.w_down, &self.scratch.hb, &self.scratch.hbq8k, &mut self.scratch.xb)?;
            } else if can_q8_ffn && matches!(layer.w_down.dtype, DType::Q4_0 | DType::Q8_0) {
                quantize_row_q8_0(&self.scratch.hb, &mut self.scratch.hbq8);
                backend.matvec_q8_act(&layer.w_down, &self.scratch.hb, &self.scratch.hbq8, &mut self.scratch.xb)?;
            } else {
                backend.matvec(&layer.w_down, &self.scratch.hb, &mut self.scratch.xb)?;
            }

            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                *x += dx;
            }
        }

        // 3. Final RMSNorm & LM Head
        backend.rms_norm(
            &self.scratch.x,
            &self.output_norm,
            cfg.rms_norm_eps,
            &mut self.scratch.xb,
        );
        backend.matvec(&self.output_weight, &self.scratch.xb, &mut self.scratch.logits)?;

        Ok(&self.scratch.logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaelumModel, SamplerConfig};
    use caelum_core::{quantize_row_q4_0, BlockQ4_0};
    use caelum_gguf::{GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;
    use caelum_tokenizer::Tokenizer;

    #[test]
    fn test_qwen2_gguf_gqa_bias_tied_embeddings_q4_0() {
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let kv_dim = kv_heads * head_dim;

        // Mini GPT-2 byte-level vocab covering ASCII + ChatML special tokens
        let mut tokens: Vec<GgufValue> = (0u8..=127)
            .map(|b| GgufValue::String((b as char).to_string()))
            .collect();
        let mut token_types: Vec<GgufValue> = vec![GgufValue::I32(1); tokens.len()];
        tokens.push(GgufValue::String("<|im_start|>".into()));
        token_types.push(GgufValue::I32(3));
        tokens.push(GgufValue::String("<|im_end|>".into()));
        token_types.push(GgufValue::I32(3));
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("qwen2".into()));
        w.add_meta("qwen2.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("qwen2.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("qwen2.block_count", GgufValue::U32(2));
        w.add_meta("qwen2.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("qwen2.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("qwen2.rope.freq_base", GgufValue::F32(1_000_000.0));
        w.add_meta("qwen2.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6));
        w.add_meta("qwen2.context_length", GgufValue::U32(256));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

        let make_q4 = |rows: usize, cols: usize, salt: usize| -> Vec<u8> {
            let vals: Vec<f32> = (0..rows * cols)
                .map(|i| (((i + salt) % 23) as f32 - 11.0) * 0.02)
                .collect();
            let mut blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; (rows * cols) / 32];
            for r in 0..rows {
                quantize_row_q4_0(
                    &vals[r * cols..(r + 1) * cols],
                    &mut blocks[r * (cols / 32)..(r + 1) * (cols / 32)],
                );
            }
            unsafe {
                std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 18)
            }
            .to_vec()
        };

        // Tied embeddings: intentionally omit `output.weight`
        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();

        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![h, h], DType::Q4_0, make_q4(h, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4(kv_dim, h, l + 3)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4(kv_dim, h, l + 4)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q.bias"), vec![h], &vec![0.01; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_v.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, h], DType::Q4_0, make_q4(h, h, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![ffn, h], DType::Q4_0, make_q4(ffn, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], DType::Q4_0, make_q4(ffn, h, l + 7)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4(h, ffn, l + 8)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let mut model = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        let backend = CpuBackend::new(Some(2));

        let ids = tokenizer.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", false);
        assert!(ids.contains(&((vocab - 2) as u32))); // <|im_start|>

        let (_, stats) = model
            .generate(
                &backend,
                &tokenizer,
                &ids,
                8,
                &SamplerConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
                false,
                |_, _| {},
            )
            .unwrap();
        assert_eq!(stats.generated_tokens, 8);
    }
}
