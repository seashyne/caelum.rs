//! Unified Transformer GGUF runtime supporting:
//! `llama`, `qwen2`/`qwen2.5`, `qwen2moe`, `qwen3`, `qwen3moe`, `gemma2`, `gemma3`,
//! `phi3`/`phi4`, `starcoder2`, and `mistral`.
//!
//! Features:
//! - Zero-copy `mmap` weights (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`)
//! - Online 8-row cache-aligned weight repacking (`TileQ4Kx8`, `TileQ5Kx8`, `TileQ6Kx8`) with AVX-VNNI
//! - Single-barrier Fused QKV and Fused Gate+Up+SwiGLU decode hot path (zero heap allocations)
//! - 256-token Batched Prefill GEMM (`8×4` / `8×8` micro-tiles, reading weights ONCE per chunk)
//! - GQA + Sliding Window Attention + Logit Soft-Capping + QK-Norm + Dual RoPE (Gemma-3) + YaRN
//! - SwiGLU, GeGLU (Gemma-2/3), GELU MLP + LayerNorm (StarCoder2), Fused QKV/FFN (Phi-3/4)
//! - Sparse Mixture-of-Experts (`qwen3moe` / `qwen2moe`) via shared `moe` engine
//! - `F32` and `Q8_0` quantized KV cache

use crate::cache::{KvCache, KvPrecision};
use crate::moe::{forward_moe_token, ExpertGating, MoeConfig, MoeLayerWeights, MoeScratch};
use crate::rope::{RopeConfig, RopeScalingType, RopeTable};
use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, DType, Result, TensorView, QK8_0,
    QK_K,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, PackedMatrix};

#[inline]
fn is_k_quant(dt: DType) -> bool {
    matches!(dt, DType::Q4_K | DType::Q5_K | DType::Q6_K)
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn geglu_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = gelu_tanh(*g) * u;
    }
}

#[inline]
fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu_tanh(*v);
    }
}

#[inline]
fn apply_norm(
    backend: &dyn Backend,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    if let Some(b) = bias {
        // Full LayerNorm (StarCoder2): (x - mean) / sqrt(var + eps) * w + b
        let n = x.len() as f32;
        let mean = x.iter().copied().sum::<f32>() / n;
        let var = x.iter().map(|&v| {
            let d = v - mean;
            d * d
        }).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..x.len() {
            out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
        }
    } else {
        backend.rms_norm(x, w, eps, out);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SwiGlu,
    GeGlu,
    GeluMlp,
}

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
    pub rope: RopeConfig,
    pub rope_local: Option<RopeConfig>,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub embed_scale: f32,
    pub attn_scale: f32,
    pub attn_logit_softcap: f32,
    pub final_logit_softcap: f32,
    pub sliding_window: Option<usize>,
    pub swa_pattern: usize,
    pub activation: FfnActivation,
    pub moe: Option<MoeConfig>,
}

impl LlamaQwenConfig {
    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or("llama").to_string();
        let p = &arch;

        let hidden_size = gguf.require_usize(&format!("{p}.embedding_length"))?;
        let num_layers = gguf.require_usize(&format!("{p}.block_count"))?;
        let num_heads = gguf.require_usize(&format!("{p}.attention.head_count"))?;
        let num_kv_heads = gguf
            .get_meta(&format!("{p}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(num_heads);
        let head_dim = gguf
            .get_meta(&format!("{p}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size / num_heads);
        let rms_norm_eps = gguf
            .get_meta(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_meta(&format!("{p}.attention.layer_norm_epsilon")))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);
        let ctx_len = gguf
            .get_meta(&format!("{p}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));

        let rope = RopeConfig::from_gguf(gguf, &arch, head_dim, ctx_len);

        // Gemma-3 uses dual RoPE bases: local SWA layers use freq_base_swa (default 10_000),
        // global layers use rope.freq_base (default 1_000_000) with optional linear scaling.
        let rope_local = if arch == "gemma3" {
            let local_base = gguf
                .get_meta(&format!("{p}.rope.freq_base_swa"))
                .and_then(|v| v.as_f32())
                .unwrap_or(10_000.0);
            let mut local_cfg = rope.clone();
            local_cfg.freq_base = local_base;
            local_cfg.scaling_type = RopeScalingType::None;
            local_cfg.scaling_factor = 1.0;
            Some(local_cfg)
        } else {
            None
        };

        let embed_scale = if matches!(arch.as_str(), "gemma" | "gemma2" | "gemma3") {
            (hidden_size as f32).sqrt()
        } else {
            1.0
        };

        let attn_scale = if let Some(s) = gguf
            .get_meta(&format!("{p}.attention.query_pre_attn_scalar"))
            .and_then(|v| v.as_f32())
        {
            1.0 / s.sqrt()
        } else if arch == "gemma2" && hidden_size == 4608 && num_heads == 32 {
            // Gemma-2 27B uses query_pre_attn_scalar = hidden_size / num_heads = 144
            1.0 / ((hidden_size / num_heads) as f32).sqrt()
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        let attn_logit_softcap = gguf
            .get_meta(&format!("{p}.attn_logit_softcapping"))
            .and_then(|v| v.as_f32())
            .unwrap_or(0.0);
        let final_logit_softcap = gguf
            .get_meta(&format!("{p}.final_logit_softcapping"))
            .and_then(|v| v.as_f32())
            .unwrap_or(0.0);

        let sliding_window = gguf
            .get_meta(&format!("{p}.attention.sliding_window"))
            .and_then(|v| v.as_usize())
            .filter(|&w| w > 0);
        let swa_pattern = match arch.as_str() {
            "gemma2" => 2,
            "gemma3" => gguf
                .get_meta(&format!("{p}.attention.sliding_window_pattern"))
                .and_then(|v| v.as_usize())
                .unwrap_or(6)
                .max(1),
            _ => 1,
        };

        let activation = match arch.as_str() {
            "gemma" | "gemma2" | "gemma3" => FfnActivation::GeGlu,
            "starcoder2" => FfnActivation::GeluMlp,
            _ => FfnActivation::SwiGlu,
        };

        let num_experts = gguf
            .get_meta(&format!("{p}.expert_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let intermediate_meta = gguf
            .get_meta(&format!("{p}.feed_forward_length"))
            .and_then(|v| v.as_usize());

        let (intermediate_size, moe) = if num_experts > 0 {
            let num_active = gguf
                .get_meta(&format!("{p}.expert_used_count"))
                .and_then(|v| v.as_usize())
                .unwrap_or(2);
            let expert_ffn_dim = gguf
                .get_meta(&format!("{p}.expert_feed_forward_length"))
                .and_then(|v| v.as_usize())
                .unwrap_or_else(|| {
                    intermediate_meta.unwrap_or(hidden_size) / num_active.max(1)
                });
            let shared_ffn_dim = gguf
                .get_meta(&format!("{p}.expert_shared_feed_forward_length"))
                .and_then(|v| v.as_usize())
                .unwrap_or(expert_ffn_dim);
            let norm_weights = gguf
                .get_meta(&format!("{p}.expert_weights_norm"))
                .and_then(|v| v.as_bool())
                .unwrap_or(arch == "qwen3moe");
            let weights_scale = gguf
                .get_meta(&format!("{p}.expert_weights_scale"))
                .and_then(|v| v.as_f32())
                .unwrap_or(1.0);
            let gating = match gguf
                .get_meta(&format!("{p}.expert_gating_func"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1)
            {
                2 => ExpertGating::Sigmoid,
                _ => ExpertGating::Softmax,
            };
            let ffn = intermediate_meta.unwrap_or(expert_ffn_dim);
            (
                ffn,
                Some(MoeConfig {
                    num_experts,
                    num_active_experts: num_active,
                    expert_ffn_dim,
                    shared_ffn_dim,
                    gating,
                    norm_weights,
                    weights_scale,
                }),
            )
        } else {
            let ffn = gguf.require_usize(&format!("{p}.feed_forward_length"))?;
            (ffn, None)
        };

        let embd = gguf.tensor("token_embd.weight")?;
        let (vocab_size, _) = embd.dims_2d()?;

        Ok(Self {
            arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            rope_local,
            rms_norm_eps,
            max_seq_len,
            embed_scale,
            attn_scale,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window,
            swa_pattern,
            activation,
            moe,
        })
    }

    /// Returns whether layer `l` uses local Sliding Window Attention (and Gemma-3 local RoPE).
    #[inline]
    pub fn layer_sliding_window(&self, l: usize) -> Option<usize> {
        let w = self.sliding_window?;
        match self.arch.as_str() {
            "gemma2" => {
                if l % 2 == 0 {
                    Some(w)
                } else {
                    None
                }
            }
            "gemma3" => {
                if (l + 1) % self.swa_pattern == 0 {
                    None
                } else {
                    Some(w)
                }
            }
            _ => Some(w),
        }
    }
}

enum FfnBlock<'a> {
    Dense {
        w_gate: Option<TensorView<'a>>,
        w_up: TensorView<'a>,
        w_down: TensorView<'a>,
        p_gate: Option<PackedMatrix>,
        p_up: Option<PackedMatrix>,
        p_down: Option<PackedMatrix>,
        b_up: Option<Vec<f32>>,
        b_down: Option<Vec<f32>>,
    },
    Moe(MoeLayerWeights<'a>),
}

struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    attn_norm_bias: Option<Vec<f32>>,
    post_attn_norm: Option<Vec<f32>>,
    wq: TensorView<'a>,
    wk: TensorView<'a>,
    wv: TensorView<'a>,
    wo: TensorView<'a>,
    p_wq: Option<PackedMatrix>,
    p_wk: Option<PackedMatrix>,
    p_wv: Option<PackedMatrix>,
    p_wo: Option<PackedMatrix>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    bo: Option<Vec<f32>>,
    /// Per-head QK-Norm weights (`Qwen3` / `Qwen3MoE` / `Gemma-3`, applied before RoPE).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    ffn_norm_bias: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    ffn: FfnBlock<'a>,
}

struct ScratchBuf {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    head_tmp: Vec<f32>,
    att: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    logits: Vec<f32>,
    xq8: Vec<BlockQ8_0>,
    woq8: Vec<BlockQ8_0>,
    hbq8: Vec<BlockQ8_0>,
    xq8k: Vec<BlockQ8_K>,
    woq8k: Vec<BlockQ8_K>,
    hbq8k: Vec<BlockQ8_K>,
    moe: Option<MoeScratch>,
}

impl ScratchBuf {
    fn new(cfg: &LlamaQwenConfig) -> Self {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let moe = cfg
            .moe
            .as_ref()
            .map(|m| MoeScratch::new(cfg.hidden_size, m));
        Self {
            x: vec![0.0; cfg.hidden_size],
            xb: vec![0.0; cfg.hidden_size],
            xb2: vec![0.0; q_dim.max(cfg.hidden_size)],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            head_tmp: vec![0.0; cfg.head_dim],
            att: vec![0.0; cfg.max_seq_len],
            hb: vec![0.0; cfg.intermediate_size],
            hb2: vec![0.0; cfg.intermediate_size],
            logits: vec![0.0; cfg.vocab_size],
            xq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (cfg.hidden_size / QK8_0).max(1)],
            woq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (q_dim / QK8_0).max(1)],
            hbq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (cfg.intermediate_size / QK8_0).max(1)],
            xq8k: vec![BlockQ8_K::default(); (cfg.hidden_size / QK_K).max(1)],
            woq8k: vec![BlockQ8_K::default(); (q_dim / QK_K).max(1)],
            hbq8k: vec![BlockQ8_K::default(); (cfg.intermediate_size / QK_K).max(1)],
            moe,
        }
    }
}

pub struct LlamaQwenModel<'a> {
    pub config: LlamaQwenConfig,
    token_embd: TensorView<'a>,
    layers: Vec<LayerWeights<'a>>,
    output_norm: Vec<f32>,
    output_norm_bias: Option<Vec<f32>>,
    output_weight: TensorView<'a>,
    rope: RopeTable,
    rope_local: Option<RopeTable>,
    cache: KvCache,
    scratch: ScratchBuf,
    packed_bytes: usize,
}

impl<'a> LlamaQwenModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        Self::from_gguf_with_kv(gguf, max_seq_override, KvPrecision::F32)
    }

    pub fn from_gguf_with_kv(
        gguf: &'a GgufFile,
        max_seq_override: Option<usize>,
        kv_precision: KvPrecision,
    ) -> Result<Self> {
        let config = LlamaQwenConfig::from_gguf(gguf, max_seq_override)?;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let ffn_dim = config.intermediate_size;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_norm_bias = if gguf.has_tensor("output_norm.bias") {
            Some(gguf.tensor("output_norm.bias")?.to_f32_vec()?)
        } else {
            None
        };
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let opt_vec = |name: String| -> Result<Option<Vec<f32>>> {
                if gguf.has_tensor(&name) {
                    Ok(Some(gguf.tensor(&name)?.to_f32_vec()?))
                } else {
                    Ok(None)
                }
            };

            // Separate Q/K/V (Llama/Qwen/Gemma/StarCoder2) OR fused `attn_qkv.weight` (Phi-3/Phi-4)
            let (wq, wk, wv, bq, bk, bv) = if gguf.has_tensor(&format!("blk.{i}.attn_q.weight")) {
                (
                    gguf.tensor(&format!("blk.{i}.attn_q.weight"))?,
                    gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
                    gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
                    opt_vec(format!("blk.{i}.attn_q.bias"))?,
                    opt_vec(format!("blk.{i}.attn_k.bias"))?,
                    opt_vec(format!("blk.{i}.attn_v.bias"))?,
                )
            } else {
                let qkv = gguf.tensor(&format!("blk.{i}.attn_qkv.weight"))?;
                let (bq, bk, bv) = if let Some(b) = opt_vec(format!("blk.{i}.attn_qkv.bias"))? {
                    (
                        Some(b[0..q_dim].to_vec()),
                        Some(b[q_dim..q_dim + kv_dim].to_vec()),
                        Some(b[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec()),
                    )
                } else {
                    (None, None, None)
                };
                (
                    qkv.slice_rows(0, q_dim)?,
                    qkv.slice_rows(q_dim, kv_dim)?,
                    qkv.slice_rows(q_dim + kv_dim, kv_dim)?,
                    bq,
                    bk,
                    bv,
                )
            };

            let attn_norm_bias = opt_vec(format!("blk.{i}.attn_norm.bias"))?;
            let post_attn_norm = opt_vec(format!("blk.{i}.post_attention_norm.weight"))?;
            let bo = opt_vec(format!("blk.{i}.attn_output.bias"))?;
            let q_norm = opt_vec(format!("blk.{i}.attn_q_norm.weight"))?;
            let k_norm = opt_vec(format!("blk.{i}.attn_k_norm.weight"))?;
            let ffn_norm_bias = opt_vec(format!("blk.{i}.ffn_norm.bias"))?;
            let post_ffn_norm = opt_vec(format!("blk.{i}.post_ffw_norm.weight"))?;

            let ffn = if gguf.has_tensor(&format!("blk.{i}.ffn_gate_exps.weight")) {
                FfnBlock::Moe(MoeLayerWeights::from_gguf(gguf, i)?)
            } else {
                let (w_gate, w_up) = if gguf.has_tensor(&format!("blk.{i}.ffn_gate.weight")) {
                    (
                        Some(gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?),
                        gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
                    )
                } else {
                    let up_full = gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?;
                    let (up_rows, _) = up_full.dims_2d()?;
                    if up_rows == 2 * ffn_dim && config.activation != FfnActivation::GeluMlp {
                        // Phi-3 / Phi-4 fused Gate+Up (`[2*ffn, hidden]`)
                        (
                            Some(up_full.slice_rows(0, ffn_dim)?),
                            up_full.slice_rows(ffn_dim, ffn_dim)?,
                        )
                    } else {
                        // StarCoder2 non-gated GELU MLP (`[ffn, hidden]`)
                        (None, up_full)
                    }
                };
                let b_up = opt_vec(format!("blk.{i}.ffn_up.bias"))?;
                let b_down = opt_vec(format!("blk.{i}.ffn_down.bias"))?;
                FfnBlock::Dense {
                    w_gate,
                    w_up,
                    w_down: gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
                    p_gate: None,
                    p_up: None,
                    p_down: None,
                    b_up,
                    b_down,
                }
            };

            layers.push(LayerWeights {
                attn_norm: gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?.to_f32_vec()?,
                attn_norm_bias,
                post_attn_norm,
                wq,
                wk,
                wv,
                wo: gguf.tensor(&format!("blk.{i}.attn_output.weight"))?,
                p_wq: None,
                p_wk: None,
                p_wv: None,
                p_wo: None,
                bq,
                bk,
                bv,
                bo,
                q_norm,
                k_norm,
                ffn_norm: gguf.tensor(&format!("blk.{i}.ffn_norm.weight"))?.to_f32_vec()?,
                ffn_norm_bias,
                post_ffn_norm,
                ffn,
            });
        }

        let cache = KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
            kv_precision,
        );
        let rope = RopeTable::new(config.max_seq_len, &config.rope);
        let rope_local = config
            .rope_local
            .as_ref()
            .map(|r| RopeTable::new(config.max_seq_len, r));
        let scratch = ScratchBuf::new(&config);

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output_norm_bias,
            output_weight,
            rope,
            rope_local,
            cache,
            scratch,
            packed_bytes: 0,
        })
    }

    /// Repacks `Q4_K`, `Q5_K`, and `Q6_K` layer projection matrices into cache-aligned 8-row SIMD
    /// tiles for AVX-VNNI / AVX2+FMA fused decode & tiled prefill GEMM.
    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        let mut total_bytes = 0usize;
        for layer in &mut self.layers {
            layer.p_wq = backend.pack_matrix(&layer.wq);
            layer.p_wk = backend.pack_matrix(&layer.wk);
            layer.p_wv = backend.pack_matrix(&layer.wv);
            layer.p_wo = backend.pack_matrix(&layer.wo);
            for p in [&layer.p_wq, &layer.p_wk, &layer.p_wv, &layer.p_wo]
                .into_iter()
                .flatten()
            {
                total_bytes += p.byte_size();
            }
            match &mut layer.ffn {
                FfnBlock::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    p_gate,
                    p_up,
                    p_down,
                    ..
                } => {
                    *p_gate = w_gate.as_ref().and_then(|wg| backend.pack_matrix(wg));
                    *p_up = backend.pack_matrix(w_up);
                    *p_down = backend.pack_matrix(w_down);
                    for p in [p_gate, p_up, p_down].into_iter().flatten() {
                        total_bytes += p.byte_size();
                    }
                }
                FfnBlock::Moe(m) => {
                    m.optimize_for_backend(backend);
                    for p in [&m.p_shexp_gate, &m.p_shexp_up, &m.p_shexp_down]
                        .into_iter()
                        .flatten()
                    {
                        total_bytes += p.byte_size();
                    }
                }
            }
        }
        self.packed_bytes = total_bytes;
    }

    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.cache.memory_bytes()
    }

    pub fn kv_precision(&self) -> KvPrecision {
        self.cache.precision
    }

    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// Runs one autoregressive step for `token_id` at sequence position `pos`.
    /// Zero heap allocations in the decode hot path.
    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32]> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.num_heads * head_dim;
        let kv_mul = cfg.num_heads / cfg.num_kv_heads;
        let attn_scale = cfg.attn_scale;
        let can_q8_h = cfg.hidden_size % QK8_0 == 0;
        let can_q8_q = q_dim % QK8_0 == 0;
        let can_q8_ffn = cfg.intermediate_size % QK8_0 == 0;
        let can_q8k_h = cfg.hidden_size % QK_K == 0;
        let can_q8k_q = q_dim % QK_K == 0;
        let can_q8k_ffn = cfg.intermediate_size % QK_K == 0;

        // 1. Embedding lookup (+ Gemma-2/3 sqrt(hidden_size) scaling)
        self.token_embd
            .dequantize_row(token_id as usize, &mut self.scratch.x)?;
        if cfg.embed_scale != 1.0 {
            let s = cfg.embed_scale;
            for v in self.scratch.x.iter_mut() {
                *v *= s;
            }
        }

        // 2. Transformer Decoder Layers
        for l in 0..cfg.num_layers {
            let layer = &self.layers[l];
            let swa = cfg.layer_sliding_window(l);
            let rope_tbl = if swa.is_some() {
                self.rope_local.as_ref().unwrap_or(&self.rope)
            } else {
                &self.rope
            };

            // Pre-Norm for Attention (RMSNorm or StarCoder2 LayerNorm)
            apply_norm(
                backend,
                &self.scratch.x,
                &layer.attn_norm,
                layer.attn_norm_bias.as_deref(),
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            // Q, K, V projections: single-barrier Fused Repacked VNNI fast path when packed
            if can_q8k_h
                && layer.p_wq.is_some()
                && layer.p_wk.is_some()
                && layer.p_wv.is_some()
            {
                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                backend.fused_qkv_packed_q8k(
                    layer.p_wq.as_ref().unwrap(),
                    layer.p_wk.as_ref().unwrap(),
                    layer.p_wv.as_ref().unwrap(),
                    &self.scratch.xq8k,
                    &mut self.scratch.q,
                    &mut self.scratch.k,
                    &mut self.scratch.v,
                )?;
            } else if can_q8k_h
                && is_k_quant(layer.wq.dtype)
                && is_k_quant(layer.wk.dtype)
                && is_k_quant(layer.wv.dtype)
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

            // Optional per-head QK-Norm (Qwen3 / Qwen3MoE — strictly BEFORE RoPE)
            if let Some(qn) = &layer.q_norm {
                for h in 0..cfg.num_heads {
                    let slice = &mut self.scratch.q[h * head_dim..(h + 1) * head_dim];
                    backend.rms_norm(slice, qn, cfg.rms_norm_eps, &mut self.scratch.head_tmp);
                    slice.copy_from_slice(&self.scratch.head_tmp);
                }
            }
            if let Some(kn) = &layer.k_norm {
                for kh in 0..cfg.num_kv_heads {
                    let slice = &mut self.scratch.k[kh * head_dim..(kh + 1) * head_dim];
                    backend.rms_norm(slice, kn, cfg.rms_norm_eps, &mut self.scratch.head_tmp);
                    slice.copy_from_slice(&self.scratch.head_tmp);
                }
            }

            // Precomputed RoPE at exact sequence position `pos` (Gemma-3 uses local RoPE on SWA layers)
            for h in 0..cfg.num_heads {
                rope_tbl.apply(&mut self.scratch.q[h * head_dim..(h + 1) * head_dim], pos);
            }
            for kh in 0..cfg.num_kv_heads {
                rope_tbl.apply(&mut self.scratch.k[kh * head_dim..(kh + 1) * head_dim], pos);
            }

            // Store K, V into cache slot `pos`
            self.cache.store(l, pos, &self.scratch.k, &self.scratch.v);

            // Grouped Query Attention over `0..=pos` (with optional SWA + logit soft-capping)
            let softcap = cfg.attn_logit_softcap;
            if pos >= 128 && backend.num_threads() > 1 {
                let q_ref: &[f32] = &self.scratch.q;
                let cache_ref = &self.cache;
                let xb2_addr = self.scratch.xb2.as_mut_ptr() as usize;
                let seq_len = pos + 1;
                backend.parallel_for(cfg.num_heads, &|h_start, h_end| {
                    let mut local_att = vec![0.0f32; seq_len];
                    let xb2_ptr = xb2_addr as *mut f32;
                    for h in h_start..h_end {
                        let qh = &q_ref[h * head_dim..(h + 1) * head_dim];
                        let kv_h = h / kv_mul;
                        let out_h = unsafe {
                            std::slice::from_raw_parts_mut(xb2_ptr.add(h * head_dim), head_dim)
                        };
                        cache_ref.attend_head_ext(
                            backend,
                            l,
                            kv_h,
                            qh,
                            pos,
                            attn_scale,
                            swa,
                            softcap,
                            &mut local_att,
                            out_h,
                        );
                    }
                });
            } else {
                for h in 0..cfg.num_heads {
                    let qh = &self.scratch.q[h * head_dim..(h + 1) * head_dim];
                    let kv_h = h / kv_mul;
                    let out_h = &mut self.scratch.xb2[h * head_dim..(h + 1) * head_dim];
                    self.cache.attend_head_ext(
                        backend,
                        l,
                        kv_h,
                        qh,
                        pos,
                        attn_scale,
                        swa,
                        softcap,
                        &mut self.scratch.att,
                        out_h,
                    );
                }
            }

        // Output projection (+ optional StarCoder2 bias + optional Gemma-2/3 post_attn_norm) + Residual
            if let (Some(pwo), true) = (&layer.p_wo, can_q8k_q) {
                quantize_row_q8_k(&self.scratch.xb2[..q_dim], &mut self.scratch.woq8k);
                backend.matvec_packed_q8k(pwo, &self.scratch.woq8k, &mut self.scratch.xb)?;
            } else if can_q8k_q && is_k_quant(layer.wo.dtype) {
                quantize_row_q8_k(&self.scratch.xb2[..q_dim], &mut self.scratch.woq8k);
                backend.matvec_q8k_act(&layer.wo, &self.scratch.xb2[..q_dim], &self.scratch.woq8k, &mut self.scratch.xb)?;
            } else if can_q8_q && matches!(layer.wo.dtype, DType::Q4_0 | DType::Q8_0) {
                quantize_row_q8_0(&self.scratch.xb2[..q_dim], &mut self.scratch.woq8);
                backend.matvec_q8_act(&layer.wo, &self.scratch.xb2[..q_dim], &self.scratch.woq8, &mut self.scratch.xb)?;
            } else {
                backend.matvec(&layer.wo, &self.scratch.xb2[..q_dim], &mut self.scratch.xb)?;
            }
            if let Some(bo) = &layer.bo {
                for (v, &b) in self.scratch.xb.iter_mut().zip(bo.iter()) {
                    *v += b;
                }
            }
            if let Some(pan) = &layer.post_attn_norm {
                let h_len = cfg.hidden_size;
                let tmp = &mut self.scratch.xb2[..h_len];
                backend.rms_norm(&self.scratch.xb, pan, cfg.rms_norm_eps, tmp);
                for (x, &dx) in self.scratch.x.iter_mut().zip(tmp.iter()) {
                    *x += dx;
                }
            } else {
                for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                    *x += dx;
                }
            }

            // Pre-Norm for FFN (Dense SwiGLU / GeGLU / GELU-MLP or Sparse MoE)
            apply_norm(
                backend,
                &self.scratch.x,
                &layer.ffn_norm,
                layer.ffn_norm_bias.as_deref(),
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            match &layer.ffn {
                FfnBlock::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    p_gate,
                    p_up,
                    p_down,
                    b_up,
                    b_down,
                } => {
                    if let Some(wg) = w_gate {
                        // Gated FFN: SwiGLU (Llama/Qwen/Phi-3) or GeGLU (Gemma-2/3)
                        if cfg.activation == FfnActivation::SwiGlu
                            && can_q8k_h
                            && p_gate.is_some()
                            && p_up.is_some()
                        {
                            quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                            backend.fused_gate_up_swiglu_packed_q8k(
                                p_gate.as_ref().unwrap(),
                                p_up.as_ref().unwrap(),
                                &self.scratch.xq8k,
                                &mut self.scratch.hb,
                            )?;
                        } else {
                            if can_q8k_h && (p_gate.is_some() || (is_k_quant(wg.dtype) && is_k_quant(w_up.dtype))) {
                                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                                if let Some(pg) = p_gate {
                                    backend.matvec_packed_q8k(pg, &self.scratch.xq8k, &mut self.scratch.hb)?;
                                } else {
                                    backend.matvec_q8k_act(wg, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb)?;
                                }
                                if let Some(pu) = p_up {
                                    backend.matvec_packed_q8k(pu, &self.scratch.xq8k, &mut self.scratch.hb2)?;
                                } else {
                                    backend.matvec_q8k_act(w_up, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb2)?;
                                }
                            } else if can_q8_h
                                && matches!(wg.dtype, DType::Q4_0 | DType::Q8_0)
                                && matches!(w_up.dtype, DType::Q4_0 | DType::Q8_0)
                            {
                                quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                                backend.matvec_q8_act(wg, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.hb)?;
                                backend.matvec_q8_act(w_up, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.hb2)?;
                            } else {
                                backend.matvec(wg, &self.scratch.xb, &mut self.scratch.hb)?;
                                backend.matvec(w_up, &self.scratch.xb, &mut self.scratch.hb2)?;
                            }
                            if cfg.activation == FfnActivation::GeGlu {
                                geglu_inplace(&mut self.scratch.hb, &self.scratch.hb2);
                            } else {
                                backend.swiglu_inplace(&mut self.scratch.hb, &self.scratch.hb2);
                            }
                        }
                    } else {
                        // Non-gated GELU MLP (StarCoder2): up + bias -> GELU
                        if let (Some(pu), true) = (p_up, can_q8k_h) {
                            quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                            backend.matvec_packed_q8k(pu, &self.scratch.xq8k, &mut self.scratch.hb)?;
                        } else if can_q8k_h && is_k_quant(w_up.dtype) {
                            quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                            backend.matvec_q8k_act(w_up, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb)?;
                        } else if can_q8_h && matches!(w_up.dtype, DType::Q4_0 | DType::Q8_0) {
                            quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                            backend.matvec_q8_act(w_up, &self.scratch.xb, &self.scratch.xq8, &mut self.scratch.hb)?;
                        } else {
                            backend.matvec(w_up, &self.scratch.xb, &mut self.scratch.hb)?;
                        }
                        if let Some(bu) = b_up {
                            for (v, &b) in self.scratch.hb.iter_mut().zip(bu.iter()) {
                                *v += b;
                            }
                        }
                        gelu_inplace(&mut self.scratch.hb);
                    }

                    if let (Some(pd), true) = (p_down, can_q8k_ffn) {
                        quantize_row_q8_k(&self.scratch.hb, &mut self.scratch.hbq8k);
                        backend.matvec_packed_q8k(pd, &self.scratch.hbq8k, &mut self.scratch.xb)?;
                    } else if can_q8k_ffn && is_k_quant(w_down.dtype) {
                        quantize_row_q8_k(&self.scratch.hb, &mut self.scratch.hbq8k);
                        backend.matvec_q8k_act(w_down, &self.scratch.hb, &self.scratch.hbq8k, &mut self.scratch.xb)?;
                    } else if can_q8_ffn && matches!(w_down.dtype, DType::Q4_0 | DType::Q8_0) {
                        quantize_row_q8_0(&self.scratch.hb, &mut self.scratch.hbq8);
                        backend.matvec_q8_act(w_down, &self.scratch.hb, &self.scratch.hbq8, &mut self.scratch.xb)?;
                    } else {
                        backend.matvec(w_down, &self.scratch.hb, &mut self.scratch.xb)?;
                    }
                    if let Some(bd) = b_down {
                        for (v, &b) in self.scratch.xb.iter_mut().zip(bd.iter()) {
                            *v += b;
                        }
                    }
                }
                FfnBlock::Moe(moe_layer) => {
                    let moe_cfg = cfg.moe.as_ref().expect("MoE layer requires MoeConfig");
                    let moe_scratch = self.scratch.moe.as_mut().expect("MoE scratch initialized");
                    let xq8k_opt = if can_q8k_h {
                        quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                        Some(self.scratch.xq8k.as_slice())
                    } else {
                        None
                    };
                    let xq8_opt = if can_q8_h && xq8k_opt.is_none() {
                        quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                        Some(self.scratch.xq8.as_slice())
                    } else {
                        None
                    };
                    let mut out_tmp = vec![0.0f32; cfg.hidden_size];
                    forward_moe_token(
                        backend,
                        moe_layer,
                        moe_cfg,
                        &self.scratch.xb,
                        xq8k_opt,
                        xq8_opt,
                        moe_scratch,
                        &mut out_tmp,
                    )?;
                    self.scratch.xb.copy_from_slice(&out_tmp);
                }
            }

            if let Some(pfn) = &layer.post_ffn_norm {
                let h_len = cfg.hidden_size;
                let tmp = &mut self.scratch.xb2[..h_len];
                backend.rms_norm(&self.scratch.xb, pfn, cfg.rms_norm_eps, tmp);
                for (x, &dx) in self.scratch.x.iter_mut().zip(tmp.iter()) {
                    *x += dx;
                }
            } else {
                for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                    *x += dx;
                }
            }
        }

        // 3. Final Norm & LM Head (+ optional Gemma-2 final logit soft-capping)
        if compute_logits {
            apply_norm(
                backend,
                &self.scratch.x,
                &self.output_norm,
                self.output_norm_bias.as_deref(),
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );
            if can_q8k_h && is_k_quant(self.output_weight.dtype) {
                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                backend.matvec_q8k_act(
                    &self.output_weight,
                    &self.scratch.xb,
                    &self.scratch.xq8k,
                    &mut self.scratch.logits,
                )?;
            } else if can_q8_h && matches!(self.output_weight.dtype, DType::Q4_0 | DType::Q8_0) {
                quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                backend.matvec_q8_act(
                    &self.output_weight,
                    &self.scratch.xb,
                    &self.scratch.xq8,
                    &mut self.scratch.logits,
                )?;
            } else {
                backend.matvec(&self.output_weight, &self.scratch.xb, &mut self.scratch.logits)?;
            }
            if cfg.final_logit_softcap > 0.0 {
                let cap = cfg.final_logit_softcap;
                let inv_cap = 1.0 / cap;
                for v in self.scratch.logits.iter_mut() {
                    *v = cap * (*v * inv_cap).tanh();
                }
            }
        }

        Ok(&self.scratch.logits)
    }

    /// High-throughput Batched Prefill GEMM (`PREFILL_CHUNK = 256`):
    /// reads each layer's repacked weight matrices ONCE across up to 256 tokens in L1 cache.
    pub fn forward_batch(
        &mut self,
        backend: &dyn Backend,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Ok(&self.scratch.logits);
        }
        if tokens.len() == 1 {
            return self.forward_token(backend, tokens[0], start_pos, compute_last_logits);
        }

        const PREFILL_CHUNK: usize = 256;
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let ffn = cfg.intermediate_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let head_dim = cfg.head_dim;
        let kv_mul = cfg.num_heads / cfg.num_kv_heads;
        let attn_scale = cfg.attn_scale;
        let can_q8k_h = h % QK_K == 0;
        let can_q8k_q = q_dim % QK_K == 0;
        let can_q8k_ffn = ffn % QK_K == 0;
        let can_q8_h = h % QK8_0 == 0;
        let can_q8_ffn = ffn % QK8_0 == 0;
        let has_dense_ffn = self
            .layers
            .iter()
            .any(|l| matches!(l.ffn, FfnBlock::Dense { .. }));

        let max_b = tokens.len().min(PREFILL_CHUNK);
        let mut x_buf = vec![0.0f32; max_b * h];
        let mut xb_buf = vec![0.0f32; max_b * h];
        let mut norm_tmp = vec![0.0f32; h];
        let mut q_buf = vec![0.0f32; max_b * q_dim];
        let mut k_buf = vec![0.0f32; max_b * kv_dim];
        let mut v_buf = vec![0.0f32; max_b * kv_dim];
        let mut attn_out_buf = vec![0.0f32; max_b * q_dim];
        let mut hb_buf = if has_dense_ffn { vec![0.0f32; max_b * ffn] } else { Vec::new() };
        let mut hb2_buf = if has_dense_ffn { vec![0.0f32; max_b * ffn] } else { Vec::new() };
        let mut xq8k_buf = vec![BlockQ8_K::default(); max_b * (h / QK_K).max(1)];
        let mut woq8k_buf = vec![BlockQ8_K::default(); max_b * (q_dim / QK_K).max(1)];
        let mut hbq8k_buf = if has_dense_ffn {
            vec![BlockQ8_K::default(); max_b * (ffn / QK_K).max(1)]
        } else {
            Vec::new()
        };
        let mut xq8_buf = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; max_b * (h / QK8_0).max(1)];
        let mut hbq8_buf = if has_dense_ffn {
            vec![BlockQ8_0 { d: 0, qs: [0; 32] }; max_b * (ffn / QK8_0).max(1)]
        } else {
            Vec::new()
        };
        let mut head_tmp = vec![0.0f32; head_dim];

        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk = &tokens[offset..(offset + PREFILL_CHUNK).min(tokens.len())];
            let b = chunk.len();
            let chunk_pos = start_pos + offset;
            let is_last_chunk = offset + b == tokens.len();

            // 1. Dequantize token embeddings for all `b` tokens in chunk (+ Gemma-2/3 embed scale)
            let x = &mut x_buf[..b * h];
            for (t, &tok) in chunk.iter().enumerate() {
                self.token_embd
                    .dequantize_row(tok as usize, &mut x[t * h..(t + 1) * h])?;
            }
            if cfg.embed_scale != 1.0 {
                let s = cfg.embed_scale;
                for v in x.iter_mut() {
                    *v *= s;
                }
            }

            let xb = &mut xb_buf[..b * h];
            let q = &mut q_buf[..b * q_dim];
            let k = &mut k_buf[..b * kv_dim];
            let v = &mut v_buf[..b * kv_dim];
            let attn_out = &mut attn_out_buf[..b * q_dim];

            // 2. Transformer layers in tiled batched GEMM mode
            for l in 0..cfg.num_layers {
                let layer = &self.layers[l];
                let swa = cfg.layer_sliding_window(l);
                let rope_tbl = if swa.is_some() {
                    self.rope_local.as_ref().unwrap_or(&self.rope)
                } else {
                    &self.rope
                };

                for t in 0..b {
                    apply_norm(
                        backend,
                        &x[t * h..(t + 1) * h],
                        &layer.attn_norm,
                        layer.attn_norm_bias.as_deref(),
                        cfg.rms_norm_eps,
                        &mut xb[t * h..(t + 1) * h],
                    );
                }

                // Batched Q, K, V GEMM (using repacked VNNI 8×4 / 8×8 tiles when available!)
                if can_q8k_h
                    && (layer.p_wq.is_some()
                        || (is_k_quant(layer.wq.dtype)
                            && is_k_quant(layer.wk.dtype)
                            && is_k_quant(layer.wv.dtype)))
                {
                    let nb = h / QK_K;
                    let xq = &mut xq8k_buf[..b * nb];
                    for t in 0..b {
                        quantize_row_q8_k(&xb[t * h..(t + 1) * h], &mut xq[t * nb..(t + 1) * nb]);
                    }
                    if let Some(p) = &layer.p_wq {
                        backend.matmul_packed_q8k(p, xq, b, q)?;
                    } else {
                        backend.matmul_q8k_act(&layer.wq, xb, xq, b, q)?;
                    }
                    if let Some(p) = &layer.p_wk {
                        backend.matmul_packed_q8k(p, xq, b, k)?;
                    } else {
                        backend.matmul_q8k_act(&layer.wk, xb, xq, b, k)?;
                    }
                    if let Some(p) = &layer.p_wv {
                        backend.matmul_packed_q8k(p, xq, b, v)?;
                    } else {
                        backend.matmul_q8k_act(&layer.wv, xb, xq, b, v)?;
                    }
                } else if can_q8_h
                    && matches!(layer.wq.dtype, DType::Q4_0 | DType::Q8_0)
                    && matches!(layer.wk.dtype, DType::Q4_0 | DType::Q8_0)
                    && matches!(layer.wv.dtype, DType::Q4_0 | DType::Q8_0)
                {
                    let nb = h / QK8_0;
                    let xq = &mut xq8_buf[..b * nb];
                    for t in 0..b {
                        quantize_row_q8_0(&xb[t * h..(t + 1) * h], &mut xq[t * nb..(t + 1) * nb]);
                    }
                    backend.matmul_q8_act(&layer.wq, xb, xq, b, q)?;
                    backend.matmul_q8_act(&layer.wk, xb, xq, b, k)?;
                    backend.matmul_q8_act(&layer.wv, xb, xq, b, v)?;
                } else {
                    backend.matmul(&layer.wq, xb, b, q)?;
                    backend.matmul(&layer.wk, xb, b, k)?;
                    backend.matmul(&layer.wv, xb, b, v)?;
                }

                // QKV biases, optional Qwen3 QK-Norm, exact RoPE, and KV cache store
                for t in 0..b {
                    let pos = chunk_pos + t;
                    let qt = &mut q[t * q_dim..(t + 1) * q_dim];
                    let kt = &mut k[t * kv_dim..(t + 1) * kv_dim];
                    let vt = &mut v[t * kv_dim..(t + 1) * kv_dim];

                    if let Some(bq) = &layer.bq {
                        for (qv, &bv) in qt.iter_mut().zip(bq.iter()) {
                            *qv += bv;
                        }
                    }
                    if let Some(bk) = &layer.bk {
                        for (kv, &bv) in kt.iter_mut().zip(bk.iter()) {
                            *kv += bv;
                        }
                    }
                    if let Some(bv) = &layer.bv {
                        for (vv, &bias) in vt.iter_mut().zip(bv.iter()) {
                            *vv += bias;
                        }
                    }

                    if let Some(qn) = &layer.q_norm {
                        for head in 0..cfg.num_heads {
                            let s = &mut qt[head * head_dim..(head + 1) * head_dim];
                            backend.rms_norm(s, qn, cfg.rms_norm_eps, &mut head_tmp);
                            s.copy_from_slice(&head_tmp);
                        }
                    }
                    if let Some(kn) = &layer.k_norm {
                        for kh in 0..cfg.num_kv_heads {
                            let s = &mut kt[kh * head_dim..(kh + 1) * head_dim];
                            backend.rms_norm(s, kn, cfg.rms_norm_eps, &mut head_tmp);
                            s.copy_from_slice(&head_tmp);
                        }
                    }

                    for head in 0..cfg.num_heads {
                        rope_tbl.apply(&mut qt[head * head_dim..(head + 1) * head_dim], pos);
                    }
                    for kh in 0..cfg.num_kv_heads {
                        rope_tbl.apply(&mut kt[kh * head_dim..(kh + 1) * head_dim], pos);
                    }

                    self.cache.store(l, pos, kt, vt);
                }

                // Multi-threaded Causal GQA Attention across all `b * num_heads` query heads
                let total_jobs = b * cfg.num_heads;
                let max_t_len = chunk_pos + b;
                let q_ref: &[f32] = q;
                let cache_ref = &self.cache;
                let attn_addr = attn_out.as_mut_ptr() as usize;
                let num_heads = cfg.num_heads;
                let softcap = cfg.attn_logit_softcap;

                backend.parallel_for(total_jobs, &|j_start, j_end| {
                    let mut local_att = vec![0.0f32; max_t_len];
                    let attn_ptr = attn_addr as *mut f32;
                    for job in j_start..j_end {
                        let t = job / num_heads;
                        let head = job % num_heads;
                        let pos = chunk_pos + t;
                        let q_off = t * q_dim + head * head_dim;
                        let qh = &q_ref[q_off..q_off + head_dim];
                        let kv_h = head / kv_mul;
                        let out_h = unsafe {
                            std::slice::from_raw_parts_mut(attn_ptr.add(q_off), head_dim)
                        };
                        cache_ref.attend_head_ext(
                            backend,
                            l,
                            kv_h,
                            qh,
                            pos,
                            attn_scale,
                            swa,
                            softcap,
                            &mut local_att,
                            out_h,
                        );
                    }
                });

                // Batched Output projection (+ optional bias + optional post_attn_norm) + Residual
                if let (Some(pwo), true) = (&layer.p_wo, can_q8k_q) {
                    let nb = q_dim / QK_K;
                    let wq = &mut woq8k_buf[..b * nb];
                    for t in 0..b {
                        quantize_row_q8_k(
                            &attn_out[t * q_dim..(t + 1) * q_dim],
                            &mut wq[t * nb..(t + 1) * nb],
                        );
                    }
                    backend.matmul_packed_q8k(pwo, wq, b, xb)?;
                } else if can_q8k_q && is_k_quant(layer.wo.dtype) {
                    let nb = q_dim / QK_K;
                    let wq = &mut woq8k_buf[..b * nb];
                    for t in 0..b {
                        quantize_row_q8_k(
                            &attn_out[t * q_dim..(t + 1) * q_dim],
                            &mut wq[t * nb..(t + 1) * nb],
                        );
                    }
                    backend.matmul_q8k_act(&layer.wo, attn_out, wq, b, xb)?;
                } else {
                    backend.matmul(&layer.wo, attn_out, b, xb)?;
                }
                if let Some(bo) = &layer.bo {
                    for t in 0..b {
                        let row = &mut xb[t * h..(t + 1) * h];
                        for (v, &bv) in row.iter_mut().zip(bo.iter()) {
                            *v += bv;
                        }
                    }
                }
                if let Some(pan) = &layer.post_attn_norm {
                    for t in 0..b {
                        backend.rms_norm(
                            &xb[t * h..(t + 1) * h],
                            pan,
                            cfg.rms_norm_eps,
                            &mut norm_tmp,
                        );
                        let x_row = &mut x[t * h..(t + 1) * h];
                        for (xv, &dx) in x_row.iter_mut().zip(norm_tmp.iter()) {
                            *xv += dx;
                        }
                    }
                } else {
                    for (xv, &dx) in x.iter_mut().zip(xb.iter()) {
                        *xv += dx;
                    }
                }

                // Pre-Norm for FFN
                for t in 0..b {
                    apply_norm(
                        backend,
                        &x[t * h..(t + 1) * h],
                        &layer.ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        cfg.rms_norm_eps,
                        &mut xb[t * h..(t + 1) * h],
                    );
                }

                match &layer.ffn {
                    FfnBlock::Dense {
                        w_gate,
                        w_up,
                        w_down,
                        p_gate,
                        p_up,
                        p_down,
                        b_up,
                        b_down,
                    } => {
                        let hb = &mut hb_buf[..b * ffn];
                        let hb2 = &mut hb2_buf[..b * ffn];

                        if let Some(wg) = w_gate {
                            if can_q8k_h
                                && (p_gate.is_some()
                                    || (is_k_quant(wg.dtype) && is_k_quant(w_up.dtype)))
                            {
                                let nb = h / QK_K;
                                let xq = &mut xq8k_buf[..b * nb];
                                for t in 0..b {
                                    quantize_row_q8_k(
                                        &xb[t * h..(t + 1) * h],
                                        &mut xq[t * nb..(t + 1) * nb],
                                    );
                                }
                                if let Some(pg) = p_gate {
                                    backend.matmul_packed_q8k(pg, xq, b, hb)?;
                                } else {
                                    backend.matmul_q8k_act(wg, xb, xq, b, hb)?;
                                }
                                if let Some(pu) = p_up {
                                    backend.matmul_packed_q8k(pu, xq, b, hb2)?;
                                } else {
                                    backend.matmul_q8k_act(w_up, xb, xq, b, hb2)?;
                                }
                            } else if can_q8_h
                                && matches!(wg.dtype, DType::Q4_0 | DType::Q8_0)
                                && matches!(w_up.dtype, DType::Q4_0 | DType::Q8_0)
                            {
                                let nb = h / QK8_0;
                                let xq = &mut xq8_buf[..b * nb];
                                for t in 0..b {
                                    quantize_row_q8_0(
                                        &xb[t * h..(t + 1) * h],
                                        &mut xq[t * nb..(t + 1) * nb],
                                    );
                                }
                                backend.matmul_q8_act(wg, xb, xq, b, hb)?;
                                backend.matmul_q8_act(w_up, xb, xq, b, hb2)?;
                            } else {
                                backend.matmul(wg, xb, b, hb)?;
                                backend.matmul(w_up, xb, b, hb2)?;
                            }

                            if cfg.activation == FfnActivation::GeGlu {
                                geglu_inplace(hb, hb2);
                            } else {
                                backend.swiglu_inplace(hb, hb2);
                            }
                        } else {
                            // Non-gated GELU MLP (StarCoder2)
                            if let (Some(pu), true) = (p_up, can_q8k_h) {
                                let nb = h / QK_K;
                                let xq = &mut xq8k_buf[..b * nb];
                                for t in 0..b {
                                    quantize_row_q8_k(
                                        &xb[t * h..(t + 1) * h],
                                        &mut xq[t * nb..(t + 1) * nb],
                                    );
                                }
                                backend.matmul_packed_q8k(pu, xq, b, hb)?;
                            } else if can_q8k_h && is_k_quant(w_up.dtype) {
                                let nb = h / QK_K;
                                let xq = &mut xq8k_buf[..b * nb];
                                for t in 0..b {
                                    quantize_row_q8_k(
                                        &xb[t * h..(t + 1) * h],
                                        &mut xq[t * nb..(t + 1) * nb],
                                    );
                                }
                                backend.matmul_q8k_act(w_up, xb, xq, b, hb)?;
                            } else if can_q8_h && matches!(w_up.dtype, DType::Q4_0 | DType::Q8_0) {
                                let nb = h / QK8_0;
                                let xq = &mut xq8_buf[..b * nb];
                                for t in 0..b {
                                    quantize_row_q8_0(
                                        &xb[t * h..(t + 1) * h],
                                        &mut xq[t * nb..(t + 1) * nb],
                                    );
                                }
                                backend.matmul_q8_act(w_up, xb, xq, b, hb)?;
                            } else {
                                backend.matmul(w_up, xb, b, hb)?;
                            }
                            if let Some(bu) = b_up {
                                for t in 0..b {
                                    let row = &mut hb[t * ffn..(t + 1) * ffn];
                                    for (v, &bv) in row.iter_mut().zip(bu.iter()) {
                                        *v += bv;
                                    }
                                }
                            }
                            gelu_inplace(hb);
                        }

                        if let (Some(pd), true) = (p_down, can_q8k_ffn) {
                            let nb = ffn / QK_K;
                            let hq = &mut hbq8k_buf[..b * nb];
                            for t in 0..b {
                                quantize_row_q8_k(
                                    &hb[t * ffn..(t + 1) * ffn],
                                    &mut hq[t * nb..(t + 1) * nb],
                                );
                            }
                            backend.matmul_packed_q8k(pd, hq, b, xb)?;
                        } else if can_q8k_ffn && is_k_quant(w_down.dtype) {
                            let nb = ffn / QK_K;
                            let hq = &mut hbq8k_buf[..b * nb];
                            for t in 0..b {
                                quantize_row_q8_k(
                                    &hb[t * ffn..(t + 1) * ffn],
                                    &mut hq[t * nb..(t + 1) * nb],
                                );
                            }
                            backend.matmul_q8k_act(w_down, hb, hq, b, xb)?;
                        } else if can_q8_ffn && matches!(w_down.dtype, DType::Q4_0 | DType::Q8_0) {
                            let nb = ffn / QK8_0;
                            let hq = &mut hbq8_buf[..b * nb];
                            for t in 0..b {
                                quantize_row_q8_0(
                                    &hb[t * ffn..(t + 1) * ffn],
                                    &mut hq[t * nb..(t + 1) * nb],
                                );
                            }
                            backend.matmul_q8_act(w_down, hb, hq, b, xb)?;
                        } else {
                            backend.matmul(w_down, hb, b, xb)?;
                        }
                        if let Some(bd) = b_down {
                            for t in 0..b {
                                let row = &mut xb[t * h..(t + 1) * h];
                                for (v, &bv) in row.iter_mut().zip(bd.iter()) {
                                    *v += bv;
                                }
                            }
                        }
                    }
                    FfnBlock::Moe(moe_layer) => {
                        let moe_cfg = cfg.moe.as_ref().expect("MoE layer requires MoeConfig");
                        let moe_scratch =
                            self.scratch.moe.as_mut().expect("MoE scratch initialized");
                        let nb_k = h / QK_K;
                        let nb_8 = h / QK8_0;
                        let mut tok_out = vec![0.0f32; h];
                        for t in 0..b {
                            let x_tok = &xb[t * h..(t + 1) * h];
                            let xq8k_opt = if can_q8k_h {
                                quantize_row_q8_k(x_tok, &mut xq8k_buf[..nb_k]);
                                Some(&xq8k_buf[..nb_k])
                            } else {
                                None
                            };
                            let xq8_opt = if can_q8_h && xq8k_opt.is_none() {
                                quantize_row_q8_0(x_tok, &mut xq8_buf[..nb_8]);
                                Some(&xq8_buf[..nb_8])
                            } else {
                                None
                            };
                            forward_moe_token(
                                backend,
                                moe_layer,
                                moe_cfg,
                                x_tok,
                                xq8k_opt,
                                xq8_opt,
                                moe_scratch,
                                &mut tok_out,
                            )?;
                            xb[t * h..(t + 1) * h].copy_from_slice(&tok_out);
                        }
                    }
                }

                if let Some(pfn) = &layer.post_ffn_norm {
                    for t in 0..b {
                        backend.rms_norm(
                            &xb[t * h..(t + 1) * h],
                            pfn,
                            cfg.rms_norm_eps,
                            &mut norm_tmp,
                        );
                        let x_row = &mut x[t * h..(t + 1) * h];
                        for (xv, &dx) in x_row.iter_mut().zip(norm_tmp.iter()) {
                            *xv += dx;
                        }
                    }
                } else {
                    for (xv, &dx) in x.iter_mut().zip(xb.iter()) {
                        *xv += dx;
                    }
                }
            }

            // 3. Final Norm & LM Head ONLY for the very last token of the prompt
            if is_last_chunk && compute_last_logits {
                let last_x = &x[(b - 1) * h..b * h];
                apply_norm(
                    backend,
                    last_x,
                    &self.output_norm,
                    self.output_norm_bias.as_deref(),
                    cfg.rms_norm_eps,
                    &mut self.scratch.xb,
                );
                if can_q8k_h && is_k_quant(self.output_weight.dtype) {
                    quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                    backend.matvec_q8k_act(
                        &self.output_weight,
                        &self.scratch.xb,
                        &self.scratch.xq8k,
                        &mut self.scratch.logits,
                    )?;
                } else if can_q8_h && matches!(self.output_weight.dtype, DType::Q4_0 | DType::Q8_0) {
                    quantize_row_q8_0(&self.scratch.xb, &mut self.scratch.xq8);
                    backend.matvec_q8_act(
                        &self.output_weight,
                        &self.scratch.xb,
                        &self.scratch.xq8,
                        &mut self.scratch.logits,
                    )?;
                } else {
                    backend.matvec(
                        &self.output_weight,
                        &self.scratch.xb,
                        &mut self.scratch.logits,
                    )?;
                }
                if cfg.final_logit_softcap > 0.0 {
                    let cap = cfg.final_logit_softcap;
                    let inv_cap = 1.0 / cap;
                    for v in self.scratch.logits.iter_mut() {
                        *v = cap * (*v * inv_cap).tanh();
                    }
                }
            }

            offset += b;
        }

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

    fn make_q4_matrix(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
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
    }

    fn mini_chatml_vocab() -> (Vec<GgufValue>, Vec<GgufValue>) {
        let mut tokens: Vec<GgufValue> = (0u8..=127)
            .map(|b| GgufValue::String((b as char).to_string()))
            .collect();
        let mut token_types: Vec<GgufValue> = vec![GgufValue::I32(1); tokens.len()];
        tokens.push(GgufValue::String("<|im_start|>".into()));
        token_types.push(GgufValue::I32(3));
        tokens.push(GgufValue::String("<|im_end|>".into()));
        token_types.push(GgufValue::I32(3));
        (tokens, token_types)
    }

    #[test]
    fn test_qwen2_gguf_gqa_bias_tied_embeddings_q4_0() {
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let kv_dim = kv_heads * head_dim;

        let (tokens, token_types) = mini_chatml_vocab();
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

        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();

        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 3)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 4)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q.bias"), vec![h], &vec![0.01; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_v.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 7)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4_matrix(h, ffn, l + 8)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let mut model = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        let backend = CpuBackend::new(Some(2));

        let ids = tokenizer.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", false);
        assert!(ids.contains(&((vocab - 2) as u32)));

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
        assert_eq!(stats.cached_prompt_tokens, 0);

        // Verify batched prefill matches sequential token-by-token prefill
        let mut m_seq = LlamaQwenModel::from_gguf(&gguf, Some(64)).unwrap();
        let mut seq_logits = Vec::new();
        for (pos, &tok) in ids.iter().enumerate() {
            let l = m_seq
                .forward_token(&backend, tok, pos, pos + 1 == ids.len())
                .unwrap();
            if pos + 1 == ids.len() {
                seq_logits = l.to_vec();
            }
        }
        let mut m_batch = LlamaQwenModel::from_gguf(&gguf, Some(64)).unwrap();
        let batch_logits = m_batch.forward_batch(&backend, &ids, 0, true).unwrap().to_vec();
        for (a, b) in seq_logits.iter().zip(batch_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Batched prefill mismatch vs sequential: seq={a}, batch={b}"
            );
        }

        // Verify automatic KV Prefix Caching hits on shared prefix
        let ids2 = tokenizer.encode("<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n", false);
        let (_, stats2) = model
            .generate(
                &backend,
                &tokenizer,
                &ids2,
                4,
                &SamplerConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
                false,
                |_, _| {},
            )
            .unwrap();
        assert!(stats2.cached_prompt_tokens >= 7);

        let m_q8 = LlamaQwenModel::from_gguf_with_kv(&gguf, Some(64), KvPrecision::Q8_0).unwrap();
        assert!(m_q8.kv_cache_bytes() <= m_seq.kv_cache_bytes());
    }

    #[test]
    fn test_qwen3_qk_norm_and_qwen3moe_3d_experts() {
        let h = 64usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = 32usize; // Intentionally q_dim (128) != hidden_size (64), just like Qwen3-0.6B!
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let n_exp = 4usize;
        let exp_ffn = 64usize;

        let (tokens, token_types) = mini_chatml_vocab();
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("qwen3moe".into()));
        w.add_meta("qwen3moe.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("qwen3moe.block_count", GgufValue::U32(2));
        w.add_meta("qwen3moe.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("qwen3moe.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("qwen3moe.attention.key_length", GgufValue::U32(head_dim as u32));
        w.add_meta("qwen3moe.expert_count", GgufValue::U32(n_exp as u32));
        w.add_meta("qwen3moe.expert_used_count", GgufValue::U32(2));
        w.add_meta("qwen3moe.expert_feed_forward_length", GgufValue::U32(exp_ffn as u32));
        w.add_meta("qwen3moe.rope.freq_base", GgufValue::F32(1_000_000.0));
        w.add_meta("qwen3moe.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6));
        w.add_meta("qwen3moe.context_length", GgufValue::U32(128));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();

        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![q_dim, h], DType::Q4_0, make_q4_matrix(q_dim, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 3)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 4)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q_norm.weight"), vec![head_dim], &vec![1.1; head_dim]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k_norm.weight"), vec![head_dim], &vec![0.9; head_dim]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], DType::Q4_0, make_q4_matrix(h, q_dim, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_gate_inp.weight"), vec![n_exp, h], DType::Q4_0, make_q4_matrix(n_exp, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_gate_exps.weight"), vec![n_exp, exp_ffn, h], DType::Q4_0, make_q4_matrix(n_exp * exp_ffn, h, l + 7)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up_exps.weight"), vec![n_exp, exp_ffn, h], DType::Q4_0, make_q4_matrix(n_exp * exp_ffn, h, l + 8)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down_exps.weight"), vec![n_exp, h, exp_ffn], DType::Q4_0, make_q4_matrix(n_exp * h, exp_ffn, l + 9)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let backend = CpuBackend::new(Some(2));
        let ids = vec![65u32, 66, 67, 68, 69];

        let mut m_seq = LlamaQwenModel::from_gguf(&gguf, Some(64)).unwrap();
        let mut seq_logits = Vec::new();
        for (pos, &tok) in ids.iter().enumerate() {
            let l = m_seq
                .forward_token(&backend, tok, pos, pos + 1 == ids.len())
                .unwrap();
            if pos + 1 == ids.len() {
                seq_logits = l.to_vec();
            }
        }
        let mut m_batch = LlamaQwenModel::from_gguf(&gguf, Some(64)).unwrap();
        let batch_logits = m_batch.forward_batch(&backend, &ids, 0, true).unwrap().to_vec();
        for (a, b) in seq_logits.iter().zip(batch_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Qwen3MoE batched prefill mismatch vs sequential: seq={a}, batch={b}"
            );
        }
    }

    fn assert_seq_batch_parity(gguf: &GgufFile, label: &str, ids: &[u32]) {
        let backend = CpuBackend::new(Some(2));
        let mut m_seq = LlamaQwenModel::from_gguf(gguf, Some(64)).unwrap();
        let mut seq_logits = Vec::new();
        for (pos, &tok) in ids.iter().enumerate() {
            let l = m_seq
                .forward_token(&backend, tok, pos, pos + 1 == ids.len())
                .unwrap();
            if pos + 1 == ids.len() {
                seq_logits = l.to_vec();
            }
        }
        let mut m_batch = LlamaQwenModel::from_gguf(gguf, Some(64)).unwrap();
        let batch_logits = m_batch.forward_batch(&backend, ids, 0, true).unwrap().to_vec();
        assert!(seq_logits.iter().all(|v| v.is_finite()), "{label}: non-finite logits");
        for (a, b) in seq_logits.iter().zip(batch_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "{label} batched prefill mismatch vs sequential: seq={a}, batch={b}"
            );
        }
    }

    #[test]
    fn test_gemma2_and_gemma3_post_norms_geglu_softcap_dual_rope() {
        for arch in ["gemma2", "gemma3"] {
            let h = 96usize;
            let ffn = 128usize;
            let heads = 4usize;
            let kv_heads = 2usize;
            let head_dim = 16usize; // Intentionally q_dim (64) < hidden_size (96), just like Gemma-2 2B!
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let (tokens, token_types) = mini_chatml_vocab();
            let vocab = tokens.len();

            let mut w = GgufWriter::new();
            w.add_meta("general.architecture", GgufValue::String(arch.into()));
            w.add_meta(&format!("{arch}.embedding_length"), GgufValue::U32(h as u32));
            w.add_meta(&format!("{arch}.feed_forward_length"), GgufValue::U32(ffn as u32));
            w.add_meta(&format!("{arch}.block_count"), GgufValue::U32(3));
            w.add_meta(&format!("{arch}.attention.head_count"), GgufValue::U32(heads as u32));
            w.add_meta(&format!("{arch}.attention.head_count_kv"), GgufValue::U32(kv_heads as u32));
            w.add_meta(&format!("{arch}.attention.key_length"), GgufValue::U32(head_dim as u32));
            w.add_meta(&format!("{arch}.attention.layer_norm_rms_epsilon"), GgufValue::F32(1e-6));
            w.add_meta(&format!("{arch}.attention.sliding_window"), GgufValue::U32(3));
            w.add_meta(&format!("{arch}.context_length"), GgufValue::U32(128));
            if arch == "gemma2" {
                w.add_meta("gemma2.attn_logit_softcapping", GgufValue::F32(50.0));
                w.add_meta("gemma2.final_logit_softcapping", GgufValue::F32(30.0));
            } else {
                w.add_meta("gemma3.attention.sliding_window_pattern", GgufValue::U32(2));
                w.add_meta("gemma3.rope.freq_base", GgufValue::F32(1_000_000.0));
                w.add_meta("gemma3.rope.freq_base_swa", GgufValue::F32(10_000.0));
            }
            w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
            w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
            w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
            w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

            w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
            w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
            for l in 0..3 {
                w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
                w.add_tensor_f32(&format!("blk.{l}.post_attention_norm.weight"), vec![h], &vec![1.05; h]).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![q_dim, h], DType::Q4_0, make_q4_matrix(q_dim, h, l + 2)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 3)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 4)).unwrap();
                if arch == "gemma3" {
                    w.add_tensor_f32(&format!("blk.{l}.attn_q_norm.weight"), vec![head_dim], &vec![1.0; head_dim]).unwrap();
                    w.add_tensor_f32(&format!("blk.{l}.attn_k_norm.weight"), vec![head_dim], &vec![1.0; head_dim]).unwrap();
                }
                w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], DType::Q4_0, make_q4_matrix(h, q_dim, l + 5)).unwrap();
                w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
                w.add_tensor_f32(&format!("blk.{l}.post_ffw_norm.weight"), vec![h], &vec![0.95; h]).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 6)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 7)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4_matrix(h, ffn, l + 8)).unwrap();
            }

            let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
            let _ = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
            assert_seq_batch_parity(&gguf, arch, &[65, 66, 67, 68, 69, 70]);
        }
    }

    #[test]
    fn test_phi3_fused_qkv_and_fused_ffn_up() {
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let (tokens, token_types) = mini_chatml_vocab();
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("phi3".into()));
        w.add_meta("phi3.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("phi3.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("phi3.block_count", GgufValue::U32(2));
        w.add_meta("phi3.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("phi3.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("phi3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5));
        w.add_meta("phi3.context_length", GgufValue::U32(128));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            // Phi-3/Phi-4 fused QKV (`[q_dim + 2*kv_dim, h]`) & fused Gate+Up (`[2*ffn, h]`)
            w.add_tensor_raw(&format!("blk.{l}.attn_qkv.weight"), vec![q_dim + 2 * kv_dim, h], DType::Q4_0, make_q4_matrix(q_dim + 2 * kv_dim, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], DType::Q4_0, make_q4_matrix(h, q_dim, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![2 * ffn, h], DType::Q4_0, make_q4_matrix(2 * ffn, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4_matrix(h, ffn, l + 8)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let _ = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        assert_seq_batch_parity(&gguf, "phi3", &[65, 66, 67, 68, 69]);
    }

    #[test]
    fn test_starcoder2_layernorm_biases_and_gelu_mlp() {
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let kv_dim = kv_heads * head_dim;
        let (tokens, token_types) = mini_chatml_vocab();
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("starcoder2".into()));
        w.add_meta("starcoder2.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("starcoder2.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("starcoder2.block_count", GgufValue::U32(2));
        w.add_meta("starcoder2.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("starcoder2.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("starcoder2.attention.layer_norm_epsilon", GgufValue::F32(1e-5));
        w.add_meta("starcoder2.attention.sliding_window", GgufValue::U32(4));
        w.add_meta("starcoder2.context_length", GgufValue::U32(128));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("output_norm.bias", vec![h], &vec![0.02; h]).unwrap();
        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.bias"), vec![h], &vec![0.01; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 3)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 4)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q.bias"), vec![h], &vec![0.01; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_v.bias"), vec![kv_dim], &vec![0.01; kv_dim]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_output.bias"), vec![h], &vec![-0.01; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.bias"), vec![h], &vec![0.01; h]).unwrap();
            // StarCoder2 non-gated GELU MLP (NO `ffn_gate.weight`!)
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 6)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_up.bias"), vec![ffn], &vec![0.02; ffn]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4_matrix(h, ffn, l + 7)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_down.bias"), vec![h], &vec![-0.02; h]).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let _ = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        assert_seq_batch_parity(&gguf, "starcoder2", &[65, 66, 67, 68, 69, 70]);
    }
}
