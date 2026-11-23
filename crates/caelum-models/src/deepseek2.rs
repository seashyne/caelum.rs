//! Native `deepseek2` GGUF architecture runtime (`DeepSeek-Coder-V2-Lite`, `DeepSeek-V2`,
//! `DeepSeek-V3`, `DeepSeek-R1`):
//! - True Compressed Multi-Head Latent Attention (MLA): caches ONLY `[kv_lora_rank + qk_rope]`
//!   per token (~8.9× smaller KV cache than standard MHA)
//! - Supports BOTH Absorbed MLA (`attn_k_b` + `attn_v_b`) AND Legacy unsplit (`attn_kv_b`)
//! - Supports BOTH Lite (`q_lora_rank = 0`, `attn_q`) AND Full (`attn_q_a` + `attn_q_a_norm` + `attn_q_b`)
//! - Exact YaRN RoPE (Normal pairing, unit cos/sin) + YaRN `mscale^2 / sqrt(dk)` attention scaling
//! - Leading Dense SwiGLU layers + DeepSeekMoE (Top-K Routed Experts + Always-Active Shared Expert)

use crate::moe::{forward_moe_token, ExpertGating, MoeConfig, MoeLayerWeights, MoeScratch};
use crate::rope::{RopeConfig, RopeTable};
use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, CaelumError, DType, Result,
    TensorView, QK8_0, QK_K,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, PackedMatrix};

#[inline]
fn is_k_quant(dt: DType) -> bool {
    matches!(dt, DType::Q4_K | DType::Q5_K | DType::Q6_K)
}

#[derive(Debug, Clone)]
pub struct DeepSeek2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub dense_ffn_size: usize,
    pub num_layers: usize,
    pub leading_dense_layers: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope: RopeConfig,
    pub kq_scale: f32,
    pub moe: MoeConfig,
}

impl DeepSeek2Config {
    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let p = "deepseek2";
        let hidden_size = gguf.require_usize(&format!("{p}.embedding_length"))?;
        let dense_ffn_size = gguf
            .get_meta(&format!("{p}.feed_forward_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size * 4);
        let num_layers = gguf.require_usize(&format!("{p}.block_count"))?;
        let leading_dense_layers = gguf
            .get_meta(&format!("{p}.leading_dense_block_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(1);
        let num_heads = gguf.require_usize(&format!("{p}.attention.head_count"))?;
        let q_lora_rank = gguf
            .get_meta(&format!("{p}.attention.q_lora_rank"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let kv_lora_rank = gguf.require_usize(&format!("{p}.attention.kv_lora_rank"))?;

        let key_length_mla = gguf
            .get_meta(&format!("{p}.attention.key_length_mla"))
            .or_else(|| gguf.get_meta(&format!("{p}.attention.key_length")))
            .and_then(|v| v.as_usize())
            .unwrap_or(192);
        let v_head_dim = gguf
            .get_meta(&format!("{p}.attention.value_length_mla"))
            .or_else(|| gguf.get_meta(&format!("{p}.attention.value_length")))
            .and_then(|v| v.as_usize())
            .unwrap_or(128);
        let qk_rope_head_dim = gguf
            .get_meta(&format!("{p}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(64);
        if key_length_mla <= qk_rope_head_dim {
            return Err(CaelumError::InvalidFormat(format!(
                "deepseek2 key_length_mla ({key_length_mla}) must exceed rope_dim ({qk_rope_head_dim})"
            )));
        }
        let qk_nope_head_dim = key_length_mla - qk_rope_head_dim;

        let rms_norm_eps = gguf
            .get_meta(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ctx_len = gguf
            .get_meta(&format!("{p}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));

        let rope = RopeConfig::from_gguf(gguf, p, qk_rope_head_dim, ctx_len);
        let mscale = rope.yarn_kq_mscale();
        let kq_scale = (mscale * mscale) / (key_length_mla as f32).sqrt();

        let num_experts = gguf.require_usize(&format!("{p}.expert_count"))?;
        let num_active_experts = gguf.require_usize(&format!("{p}.expert_used_count"))?;
        let expert_ffn_dim = gguf.require_usize(&format!("{p}.expert_feed_forward_length"))?;
        let n_shared = gguf
            .get_meta(&format!("{p}.expert_shared_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(2);
        let weights_scale = gguf
            .get_meta(&format!("{p}.expert_weights_scale"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);
        let norm_weights = gguf
            .get_meta(&format!("{p}.expert_weights_norm"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let gating = match gguf
            .get_meta(&format!("{p}.expert_gating_func"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
        {
            2 => ExpertGating::Sigmoid,
            _ => ExpertGating::Softmax,
        };

        let (vocab_size, _) = gguf.tensor("token_embd.weight")?.dims_2d()?;

        Ok(Self {
            vocab_size,
            hidden_size,
            dense_ffn_size,
            num_layers,
            leading_dense_layers,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rms_norm_eps,
            max_seq_len,
            rope,
            kq_scale,
            moe: MoeConfig {
                num_experts,
                num_active_experts,
                expert_ffn_dim,
                shared_ffn_dim: expert_ffn_dim * n_shared,
                gating,
                norm_weights,
                weights_scale,
            },
        })
    }

    #[inline]
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

enum QueryProj<'a> {
    Direct {
        wq: TensorView<'a>,
        p_wq: Option<PackedMatrix>,
    },
    LowRank {
        wq_a: TensorView<'a>,
        q_a_norm: Vec<f32>,
        wq_b: TensorView<'a>,
        p_wq_a: Option<PackedMatrix>,
        p_wq_b: Option<PackedMatrix>,
    },
}

enum MlaKvB<'a> {
    /// Modern absorbed MLA (`blk.{i}.attn_k_b.weight` `[n_head, kv_lora, qk_nope]`
    /// and `blk.{i}.attn_v_b.weight` `[n_head, v_dim, kv_lora]`).
    Absorbed {
        wk_b: TensorView<'a>,
        wv_b: TensorView<'a>,
    },
    /// Legacy unsplit MLA (`blk.{i}.attn_kv_b.weight` `[n_head * (qk_nope + v_dim), kv_lora]`).
    LegacyUnsplit { wkv_b: TensorView<'a> },
}

enum Ds2Ffn<'a> {
    Dense {
        w_gate: TensorView<'a>,
        w_up: TensorView<'a>,
        w_down: TensorView<'a>,
        p_gate: Option<PackedMatrix>,
        p_up: Option<PackedMatrix>,
        p_down: Option<PackedMatrix>,
    },
    Moe(MoeLayerWeights<'a>),
}

struct Ds2Layer<'a> {
    attn_norm: Vec<f32>,
    q_proj: QueryProj<'a>,
    wkv_a_mqa: TensorView<'a>,
    p_wkv_a_mqa: Option<PackedMatrix>,
    kv_a_norm: Vec<f32>,
    kv_b: MlaKvB<'a>,
    wo: TensorView<'a>,
    p_wo: Option<PackedMatrix>,
    ffn_norm: Vec<f32>,
    ffn: Ds2Ffn<'a>,
}

/// True Compressed MLA KV Cache storing ONLY `[kv_cmpr (kv_lora_rank), k_pe (qk_rope_head_dim)]`
/// per token per layer (~8.9× smaller than standard MHA K+V cache!).
struct MlaKvStore {
    max_seq_len: usize,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
    /// Layout: `[num_layers][max_seq_len * kv_lora_rank]`
    kv_cmpr: Vec<Vec<f32>>,
    /// Layout: `[num_layers][max_seq_len * qk_rope_dim]`
    k_pe: Vec<Vec<f32>>,
}

impl MlaKvStore {
    fn new(num_layers: usize, max_seq_len: usize, kv_lora_rank: usize, qk_rope_dim: usize) -> Self {
        Self {
            max_seq_len,
            kv_lora_rank,
            qk_rope_dim,
            kv_cmpr: (0..num_layers)
                .map(|_| vec![0.0f32; max_seq_len * kv_lora_rank])
                .collect(),
            k_pe: (0..num_layers)
                .map(|_| vec![0.0f32; max_seq_len * qk_rope_dim])
                .collect(),
        }
    }

    fn clear(&mut self) {
        for v in &mut self.kv_cmpr {
            v.fill(0.0);
        }
        for v in &mut self.k_pe {
            v.fill(0.0);
        }
    }

    fn memory_bytes(&self) -> usize {
        self.kv_cmpr.len()
            * self.max_seq_len
            * (self.kv_lora_rank + self.qk_rope_dim)
            * std::mem::size_of::<f32>()
    }

    #[inline]
    fn store(&mut self, layer: usize, pos: usize, cmpr: &[f32], pe: &[f32]) {
        let c_off = pos * self.kv_lora_rank;
        self.kv_cmpr[layer][c_off..c_off + self.kv_lora_rank].copy_from_slice(cmpr);
        let p_off = pos * self.qk_rope_dim;
        self.k_pe[layer][p_off..p_off + self.qk_rope_dim].copy_from_slice(pe);
    }

    #[inline]
    fn cmpr_at(&self, layer: usize, pos: usize) -> &[f32] {
        let off = pos * self.kv_lora_rank;
        &self.kv_cmpr[layer][off..off + self.kv_lora_rank]
    }

    #[inline]
    fn pe_at(&self, layer: usize, pos: usize) -> &[f32] {
        let off = pos * self.qk_rope_dim;
        &self.k_pe[layer][off..off + self.qk_rope_dim]
    }
}

struct Ds2Scratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    q_latent: Vec<f32>,
    q_latent_norm: Vec<f32>,
    q: Vec<f32>,
    kv_a: Vec<f32>,
    kv_cmpr: Vec<f32>,
    attn_out: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    logits: Vec<f32>,
    xq8k: Vec<BlockQ8_K>,
    xq8: Vec<BlockQ8_0>,
    woq8k: Vec<BlockQ8_K>,
    hbq8k: Vec<BlockQ8_K>,
    moe: MoeScratch,
}

impl Ds2Scratch {
    fn new(cfg: &DeepSeek2Config) -> Self {
        let q_dim = cfg.num_heads * cfg.qk_head_dim();
        let o_dim = cfg.num_heads * cfg.v_head_dim;
        Self {
            x: vec![0.0; cfg.hidden_size],
            xb: vec![0.0; cfg.hidden_size],
            q_latent: vec![0.0; cfg.q_lora_rank.max(1)],
            q_latent_norm: vec![0.0; cfg.q_lora_rank.max(1)],
            q: vec![0.0; q_dim],
            kv_a: vec![0.0; cfg.kv_lora_rank + cfg.qk_rope_head_dim],
            kv_cmpr: vec![0.0; cfg.kv_lora_rank],
            attn_out: vec![0.0; o_dim],
            hb: vec![0.0; cfg.dense_ffn_size],
            hb2: vec![0.0; cfg.dense_ffn_size],
            logits: vec![0.0; cfg.vocab_size],
            xq8k: vec![BlockQ8_K::default(); (cfg.hidden_size / QK_K).max(1)],
            xq8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (cfg.hidden_size / QK8_0).max(1)],
            woq8k: vec![BlockQ8_K::default(); (o_dim / QK_K).max(1)],
            hbq8k: vec![BlockQ8_K::default(); (cfg.dense_ffn_size / QK_K).max(1)],
            moe: MoeScratch::new(cfg.hidden_size, &cfg.moe),
        }
    }
}

pub struct DeepSeek2Model<'a> {
    pub config: DeepSeek2Config,
    token_embd: TensorView<'a>,
    layers: Vec<Ds2Layer<'a>>,
    output_norm: Vec<f32>,
    output_weight: TensorView<'a>,
    rope: RopeTable,
    cache: MlaKvStore,
    scratch: Ds2Scratch,
    packed_bytes: usize,
}

impl<'a> DeepSeek2Model<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let config = DeepSeek2Config::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let attn_norm = gguf
                .tensor(&format!("blk.{i}.attn_norm.weight"))?
                .to_f32_vec()?;
            let q_proj = if gguf.has_tensor(&format!("blk.{i}.attn_q_a.weight")) {
                QueryProj::LowRank {
                    wq_a: gguf.tensor(&format!("blk.{i}.attn_q_a.weight"))?,
                    q_a_norm: gguf
                        .tensor(&format!("blk.{i}.attn_q_a_norm.weight"))?
                        .to_f32_vec()?,
                    wq_b: gguf.tensor(&format!("blk.{i}.attn_q_b.weight"))?,
                    p_wq_a: None,
                    p_wq_b: None,
                }
            } else {
                QueryProj::Direct {
                    wq: gguf.tensor(&format!("blk.{i}.attn_q.weight"))?,
                    p_wq: None,
                }
            };

            let wkv_a_mqa = gguf.tensor(&format!("blk.{i}.attn_kv_a_mqa.weight"))?;
            let kv_a_norm = gguf
                .tensor(&format!("blk.{i}.attn_kv_a_norm.weight"))?
                .to_f32_vec()?;
            let kv_b = if gguf.has_tensor(&format!("blk.{i}.attn_k_b.weight")) {
                MlaKvB::Absorbed {
                    wk_b: gguf.tensor(&format!("blk.{i}.attn_k_b.weight"))?,
                    wv_b: gguf.tensor(&format!("blk.{i}.attn_v_b.weight"))?,
                }
            } else {
                MlaKvB::LegacyUnsplit {
                    wkv_b: gguf.tensor(&format!("blk.{i}.attn_kv_b.weight"))?,
                }
            };

            let wo = gguf.tensor(&format!("blk.{i}.attn_output.weight"))?;
            let ffn_norm = gguf
                .tensor(&format!("blk.{i}.ffn_norm.weight"))?
                .to_f32_vec()?;
            let ffn = if i < config.leading_dense_layers
                && gguf.has_tensor(&format!("blk.{i}.ffn_gate.weight"))
            {
                Ds2Ffn::Dense {
                    w_gate: gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
                    w_up: gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
                    w_down: gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?,
                    p_gate: None,
                    p_up: None,
                    p_down: None,
                }
            } else {
                Ds2Ffn::Moe(MoeLayerWeights::from_gguf(gguf, i)?)
            };

            layers.push(Ds2Layer {
                attn_norm,
                q_proj,
                wkv_a_mqa,
                p_wkv_a_mqa: None,
                kv_a_norm,
                kv_b,
                wo,
                p_wo: None,
                ffn_norm,
                ffn,
            });
        }

        let rope = RopeTable::new(config.max_seq_len, &config.rope);
        let cache = MlaKvStore::new(
            config.num_layers,
            config.max_seq_len,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
        );
        let scratch = Ds2Scratch::new(&config);

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output_weight,
            rope,
            cache,
            scratch,
            packed_bytes: 0,
        })
    }

    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        let mut total = 0usize;
        for l in &mut self.layers {
            match &mut l.q_proj {
                QueryProj::Direct { wq, p_wq } => {
                    *p_wq = backend.pack_matrix(wq);
                    if let Some(p) = p_wq {
                        total += p.byte_size();
                    }
                }
                QueryProj::LowRank {
                    wq_a,
                    wq_b,
                    p_wq_a,
                    p_wq_b,
                    ..
                } => {
                    *p_wq_a = backend.pack_matrix(wq_a);
                    *p_wq_b = backend.pack_matrix(wq_b);
                    for p in [p_wq_a, p_wq_b].into_iter().flatten() {
                        total += p.byte_size();
                    }
                }
            }
            l.p_wkv_a_mqa = backend.pack_matrix(&l.wkv_a_mqa);
            l.p_wo = backend.pack_matrix(&l.wo);
            for p in [&l.p_wkv_a_mqa, &l.p_wo].into_iter().flatten() {
                total += p.byte_size();
            }
            match &mut l.ffn {
                Ds2Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    p_gate,
                    p_up,
                    p_down,
                } => {
                    *p_gate = backend.pack_matrix(w_gate);
                    *p_up = backend.pack_matrix(w_up);
                    *p_down = backend.pack_matrix(w_down);
                    for p in [p_gate, p_up, p_down].into_iter().flatten() {
                        total += p.byte_size();
                    }
                }
                Ds2Ffn::Moe(m) => {
                    m.optimize_for_backend(backend);
                    for p in [&m.p_shexp_gate, &m.p_shexp_up, &m.p_shexp_down]
                        .into_iter()
                        .flatten()
                    {
                        total += p.byte_size();
                    }
                }
            }
        }
        self.packed_bytes = total;
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

    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32]> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_head = cfg.num_heads;
        let kv_rank = cfg.kv_lora_rank;
        let qk_nope = cfg.qk_nope_head_dim;
        let qk_rope = cfg.qk_rope_head_dim;
        let qk_dim = cfg.qk_head_dim();
        let v_dim = cfg.v_head_dim;
        let o_dim = n_head * v_dim;
        let can_q8k_h = h % QK_K == 0;
        let can_q8_h = h % QK8_0 == 0;
        let can_q8k_o = o_dim % QK_K == 0;
        let can_q8k_ffn = cfg.dense_ffn_size % QK_K == 0;

        // 1. Token embedding
        self.token_embd
            .dequantize_row(token_id as usize, &mut self.scratch.x)?;

        // 2. Layers
        for l in 0..cfg.num_layers {
            let layer = &self.layers[l];

            // Pre-Attention RMSNorm
            backend.rms_norm(
                &self.scratch.x,
                &layer.attn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );
            let xq8k_ready = if can_q8k_h {
                quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                true
            } else {
                false
            };

            // 2a. Query projection (Direct for Lite, or LoRA q_a -> norm -> q_b for V2/V3)
            match &layer.q_proj {
                QueryProj::Direct { wq, p_wq } => {
                    if let (Some(p), true) = (p_wq, xq8k_ready) {
                        backend.matvec_packed_q8k(p, &self.scratch.xq8k, &mut self.scratch.q)?;
                    } else if xq8k_ready && is_k_quant(wq.dtype) {
                        backend.matvec_q8k_act(wq, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.q)?;
                    } else {
                        backend.matvec(wq, &self.scratch.xb, &mut self.scratch.q)?;
                    }
                }
                QueryProj::LowRank {
                    wq_a,
                    q_a_norm,
                    wq_b,
                    p_wq_a,
                    p_wq_b,
                } => {
                    let ql = &mut self.scratch.q_latent[..cfg.q_lora_rank];
                    if let (Some(p), true) = (p_wq_a, xq8k_ready) {
                        backend.matvec_packed_q8k(p, &self.scratch.xq8k, ql)?;
                    } else {
                        backend.matvec(wq_a, &self.scratch.xb, ql)?;
                    }
                    let qln = &mut self.scratch.q_latent_norm[..cfg.q_lora_rank];
                    backend.rms_norm(ql, q_a_norm, cfg.rms_norm_eps, qln);
                    if let (Some(p), true) = (p_wq_b, cfg.q_lora_rank % QK_K == 0) {
                        let mut q_q8k = vec![BlockQ8_K::default(); cfg.q_lora_rank / QK_K];
                        quantize_row_q8_k(qln, &mut q_q8k);
                        backend.matvec_packed_q8k(p, &q_q8k, &mut self.scratch.q)?;
                    } else {
                        backend.matvec(wq_b, qln, &mut self.scratch.q)?;
                    }
                }
            }

            // 2b. Compressed KV + decoupled RoPE key (`wkv_a_mqa @ xb`)
            if let (Some(p), true) = (&layer.p_wkv_a_mqa, xq8k_ready) {
                backend.matvec_packed_q8k(p, &self.scratch.xq8k, &mut self.scratch.kv_a)?;
            } else if xq8k_ready && is_k_quant(layer.wkv_a_mqa.dtype) {
                backend.matvec_q8k_act(
                    &layer.wkv_a_mqa,
                    &self.scratch.xb,
                    &self.scratch.xq8k,
                    &mut self.scratch.kv_a,
                )?;
            } else {
                backend.matvec(&layer.wkv_a_mqa, &self.scratch.xb, &mut self.scratch.kv_a)?;
            }

            backend.rms_norm(
                &self.scratch.kv_a[..kv_rank],
                &layer.kv_a_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.kv_cmpr,
            );
            let k_pe = &mut self.scratch.kv_a[kv_rank..kv_rank + qk_rope];
            self.rope.apply(k_pe, pos);

            // Store compressed latent + RoPE key ONCE in MLA cache!
            self.cache.store(l, pos, &self.scratch.kv_cmpr, k_pe);

            // Apply RoPE to trailing `qk_rope` dims of each query head
            for head in 0..n_head {
                let pe_off = head * qk_dim + qk_nope;
                self.rope
                    .apply(&mut self.scratch.q[pe_off..pe_off + qk_rope], pos);
            }

            // 2c. Multi-head Latent Attention (Absorbed MQA over compressed cache, parallelized by head)
            let seq_len = pos + 1;
            let kq_scale = cfg.kq_scale;
            match &layer.kv_b {
                MlaKvB::Absorbed { wk_b, wv_b } => {
                    let q_ref: &[f32] = &self.scratch.q;
                    let cache_ref = &self.cache;
                    let attn_addr = self.scratch.attn_out.as_mut_ptr() as usize;

                    let attend_heads = |h_start: usize, h_end: usize| {
                        let attn_ptr = attn_addr as *mut f32;
                        let mut q_abs = vec![0.0f32; kv_rank];
                        let mut att = vec![0.0f32; seq_len];
                        let mut ctx_lat = vec![0.0f32; kv_rank];

                        for head in h_start..h_end {
                            let q_head = &q_ref[head * qk_dim..(head + 1) * qk_dim];
                            let q_nope = &q_head[..qk_nope];
                            let q_pe = &q_head[qk_nope..];

                            let wk_h = wk_b.slice_outer(head).expect("valid wk_b head slice");
                            let wv_h = wv_b.slice_outer(head).expect("valid wv_b head slice");

                            // Absorb `wk_b[head]` into `q_nope` -> `[kv_lora_rank]`
                            // Small per-head matvec (`[512, 128]`): runs single-threaded inside head shard
                            matvec_small_local(backend, &wk_h, q_nope, &mut q_abs);

                            for t in 0..seq_len {
                                let s = backend.dot_f32(&q_abs, cache_ref.cmpr_at(l, t))
                                    + backend.dot_f32(q_pe, cache_ref.pe_at(l, t));
                                att[t] = s * kq_scale;
                            }
                            backend.softmax(&mut att);

                            ctx_lat.fill(0.0);
                            for t in 0..seq_len {
                                let w = att[t];
                                let cmpr = cache_ref.cmpr_at(l, t);
                                for j in 0..kv_rank {
                                    ctx_lat[j] += w * cmpr[j];
                                }
                            }

                            let out_h = unsafe {
                                std::slice::from_raw_parts_mut(attn_ptr.add(head * v_dim), v_dim)
                            };
                            matvec_small_local(backend, &wv_h, &ctx_lat, out_h);
                        }
                    };

                    if backend.num_threads() > 1 && n_head >= 4 {
                        backend.parallel_for(n_head, &attend_heads);
                    } else {
                        attend_heads(0, n_head);
                    }
                }
                MlaKvB::LegacyUnsplit { wkv_b } => {
                    // Legacy fallback: decompress each cached latent via `wkv_b`
                    let kv_head_stride = qk_nope + v_dim;
                    let mut all_k = vec![0.0f32; seq_len * n_head * qk_nope];
                    let mut all_v = vec![0.0f32; seq_len * n_head * v_dim];
                    let mut kv_tmp = vec![0.0f32; n_head * kv_head_stride];
                    for t in 0..seq_len {
                        backend.matvec(wkv_b, self.cache.cmpr_at(l, t), &mut kv_tmp)?;
                        for head in 0..n_head {
                            let src = &kv_tmp[head * kv_head_stride..(head + 1) * kv_head_stride];
                            let k_off = (t * n_head + head) * qk_nope;
                            let v_off = (t * n_head + head) * v_dim;
                            all_k[k_off..k_off + qk_nope].copy_from_slice(&src[..qk_nope]);
                            all_v[v_off..v_off + v_dim].copy_from_slice(&src[qk_nope..]);
                        }
                    }
                    let mut att = vec![0.0f32; seq_len];
                    for head in 0..n_head {
                        let q_head = &self.scratch.q[head * qk_dim..(head + 1) * qk_dim];
                        let q_nope = &q_head[..qk_nope];
                        let q_pe = &q_head[qk_nope..];
                        for t in 0..seq_len {
                            let k_off = (t * n_head + head) * qk_nope;
                            let s = backend.dot_f32(q_nope, &all_k[k_off..k_off + qk_nope])
                                + backend.dot_f32(q_pe, self.cache.pe_at(l, t));
                            att[t] = s * kq_scale;
                        }
                        backend.softmax(&mut att);
                        let out_h = &mut self.scratch.attn_out[head * v_dim..(head + 1) * v_dim];
                        out_h.fill(0.0);
                        for t in 0..seq_len {
                            let w = att[t];
                            let v_off = (t * n_head + head) * v_dim;
                            let vh = &all_v[v_off..v_off + v_dim];
                            for j in 0..v_dim {
                                out_h[j] += w * vh[j];
                            }
                        }
                    }
                }
            }

            // 2d. Output projection + Residual
            if let (Some(p), true) = (&layer.p_wo, can_q8k_o) {
                quantize_row_q8_k(&self.scratch.attn_out, &mut self.scratch.woq8k);
                backend.matvec_packed_q8k(p, &self.scratch.woq8k, &mut self.scratch.xb)?;
            } else if can_q8k_o && is_k_quant(layer.wo.dtype) {
                quantize_row_q8_k(&self.scratch.attn_out, &mut self.scratch.woq8k);
                backend.matvec_q8k_act(
                    &layer.wo,
                    &self.scratch.attn_out,
                    &self.scratch.woq8k,
                    &mut self.scratch.xb,
                )?;
            } else {
                backend.matvec(&layer.wo, &self.scratch.attn_out, &mut self.scratch.xb)?;
            }
            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                *x += dx;
            }

            // 2e. Pre-FFN RMSNorm + Dense (leading) or DeepSeekMoE (Routed + Shared)
            backend.rms_norm(
                &self.scratch.x,
                &layer.ffn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            match &layer.ffn {
                Ds2Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                    p_gate,
                    p_up,
                    p_down,
                } => {
                    if let (Some(pg), Some(pu), true) = (p_gate, p_up, can_q8k_h) {
                        quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                        backend.fused_gate_up_swiglu_packed_q8k(
                            pg,
                            pu,
                            &self.scratch.xq8k,
                            &mut self.scratch.hb,
                        )?;
                    } else {
                        if can_q8k_h && is_k_quant(w_gate.dtype) && is_k_quant(w_up.dtype) {
                            quantize_row_q8_k(&self.scratch.xb, &mut self.scratch.xq8k);
                            backend.matvec_q8k_act(w_gate, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb)?;
                            backend.matvec_q8k_act(w_up, &self.scratch.xb, &self.scratch.xq8k, &mut self.scratch.hb2)?;
                        } else {
                            backend.matvec(w_gate, &self.scratch.xb, &mut self.scratch.hb)?;
                            backend.matvec(w_up, &self.scratch.xb, &mut self.scratch.hb2)?;
                        }
                        backend.swiglu_inplace(&mut self.scratch.hb, &self.scratch.hb2);
                    }

                    if let (Some(pd), true) = (p_down, can_q8k_ffn) {
                        quantize_row_q8_k(&self.scratch.hb, &mut self.scratch.hbq8k);
                        backend.matvec_packed_q8k(pd, &self.scratch.hbq8k, &mut self.scratch.xb)?;
                    } else if can_q8k_ffn && is_k_quant(w_down.dtype) {
                        quantize_row_q8_k(&self.scratch.hb, &mut self.scratch.hbq8k);
                        backend.matvec_q8k_act(w_down, &self.scratch.hb, &self.scratch.hbq8k, &mut self.scratch.xb)?;
                    } else {
                        backend.matvec(w_down, &self.scratch.hb, &mut self.scratch.xb)?;
                    }
                }
                Ds2Ffn::Moe(moe_layer) => {
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
                    let mut out_tmp = vec![0.0f32; h];
                    forward_moe_token(
                        backend,
                        moe_layer,
                        &cfg.moe,
                        &self.scratch.xb,
                        xq8k_opt,
                        xq8_opt,
                        &mut self.scratch.moe,
                        &mut out_tmp,
                    )?;
                    self.scratch.xb.copy_from_slice(&out_tmp);
                }
            }

            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                *x += dx;
            }
        }

        // 3. Final RMSNorm & LM Head
        if compute_logits {
            backend.rms_norm(
                &self.scratch.x,
                &self.output_norm,
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
            } else {
                backend.matvec(&self.output_weight, &self.scratch.xb, &mut self.scratch.logits)?;
            }
        }

        Ok(&self.scratch.logits)
    }
}

/// Computes a small per-head matvec `[rows, cols] × [cols]` purely on the current worker thread
/// (avoids nested thread-pool locking when already executing inside `backend.parallel_for`!).
fn matvec_small_local(backend: &dyn Backend, w: &TensorView<'_>, x: &[f32], out: &mut [f32]) {
    let (rows, cols) = w.dims_2d().expect("2D slice");
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let mut row_f32 = vec![0.0f32; cols];
    for (r, dst) in out.iter_mut().enumerate() {
        w.dequantize_row(r, &mut row_f32)
            .expect("valid row dequantization");
        *dst = backend.dot_f32(&row_f32, x);
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

    fn make_q4(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| (((i + salt) % 19) as f32 - 9.0) * 0.02)
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

    #[test]
    fn test_deepseek2_lite_mla_absorbed_and_moe() {
        let h = 64usize;
        let dense_ffn = 128usize;
        let n_head = 4usize;
        let kv_lora = 32usize;
        let qk_nope = 32usize;
        let qk_rope = 32usize;
        let qk_dim = qk_nope + qk_rope;
        let v_dim = 32usize;
        let n_exp = 4usize;
        let exp_ffn = 64usize;
        let n_shared = 1usize;

        let mut tokens: Vec<GgufValue> = (0u8..=127)
            .map(|b| GgufValue::String((b as char).to_string()))
            .collect();
        let mut token_types: Vec<GgufValue> = vec![GgufValue::I32(1); tokens.len()];
        tokens.push(GgufValue::String("<｜end▁of▁sentence｜>".into()));
        token_types.push(GgufValue::I32(3));
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("deepseek2".into()));
        w.add_meta("deepseek2.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("deepseek2.feed_forward_length", GgufValue::U32(dense_ffn as u32));
        w.add_meta("deepseek2.block_count", GgufValue::U32(2));
        w.add_meta("deepseek2.leading_dense_block_count", GgufValue::U32(1));
        w.add_meta("deepseek2.attention.head_count", GgufValue::U32(n_head as u32));
        w.add_meta("deepseek2.attention.kv_lora_rank", GgufValue::U32(kv_lora as u32));
        w.add_meta("deepseek2.attention.key_length_mla", GgufValue::U32(qk_dim as u32));
        w.add_meta("deepseek2.attention.value_length_mla", GgufValue::U32(v_dim as u32));
        w.add_meta("deepseek2.rope.dimension_count", GgufValue::U32(qk_rope as u32));
        w.add_meta("deepseek2.rope.freq_base", GgufValue::F32(10_000.0));
        w.add_meta("deepseek2.rope.scaling.type", GgufValue::String("yarn".into()));
        w.add_meta("deepseek2.rope.scaling.factor", GgufValue::F32(40.0));
        w.add_meta("deepseek2.rope.scaling.original_context_length", GgufValue::U32(4096));
        w.add_meta("deepseek2.rope.scaling.yarn_log_multiplier", GgufValue::F32(0.0707));
        w.add_meta("deepseek2.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6));
        w.add_meta("deepseek2.context_length", GgufValue::U32(128));
        w.add_meta("deepseek2.expert_count", GgufValue::U32(n_exp as u32));
        w.add_meta("deepseek2.expert_used_count", GgufValue::U32(2));
        w.add_meta("deepseek2.expert_feed_forward_length", GgufValue::U32(exp_ffn as u32));
        w.add_meta("deepseek2.expert_shared_count", GgufValue::U32(n_shared as u32));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));

        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();

        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![n_head * qk_dim, h], DType::Q4_0, make_q4(n_head * qk_dim, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_kv_a_mqa.weight"), vec![kv_lora + qk_rope, h], DType::Q4_0, make_q4(kv_lora + qk_rope, h, l + 3)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_kv_a_norm.weight"), vec![kv_lora], &vec![1.0; kv_lora]).unwrap();
            // Absorbed MLA 3D tensors: [n_head, kv_lora, qk_nope] and [n_head, v_dim, kv_lora]
            w.add_tensor_raw(&format!("blk.{l}.attn_k_b.weight"), vec![n_head, kv_lora, qk_nope], DType::Q4_0, make_q4(n_head * kv_lora, qk_nope, l + 4)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v_b.weight"), vec![n_head, v_dim, kv_lora], DType::Q4_0, make_q4(n_head * v_dim, kv_lora, l + 5)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, n_head * v_dim], DType::Q4_0, make_q4(h, n_head * v_dim, l + 6)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();

            if l == 0 {
                // Leading dense layer
                w.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![dense_ffn, h], DType::Q4_0, make_q4(dense_ffn, h, 10)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![dense_ffn, h], DType::Q4_0, make_q4(dense_ffn, h, 11)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, dense_ffn], DType::Q4_0, make_q4(h, dense_ffn, 12)).unwrap();
            } else {
                // DeepSeekMoE layer (routed + shared)
                w.add_tensor_raw(&format!("blk.{l}.ffn_gate_inp.weight"), vec![n_exp, h], DType::Q4_0, make_q4(n_exp, h, 13)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_gate_exps.weight"), vec![n_exp, exp_ffn, h], DType::Q4_0, make_q4(n_exp * exp_ffn, h, 14)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_up_exps.weight"), vec![n_exp, exp_ffn, h], DType::Q4_0, make_q4(n_exp * exp_ffn, h, 15)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_down_exps.weight"), vec![n_exp, h, exp_ffn], DType::Q4_0, make_q4(n_exp * h, exp_ffn, 16)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_gate_shexp.weight"), vec![exp_ffn, h], DType::Q4_0, make_q4(exp_ffn, h, 17)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_up_shexp.weight"), vec![exp_ffn, h], DType::Q4_0, make_q4(exp_ffn, h, 18)).unwrap();
                w.add_tensor_raw(&format!("blk.{l}.ffn_down_shexp.weight"), vec![h, exp_ffn], DType::Q4_0, make_q4(h, exp_ffn, 19)).unwrap();
            }
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let mut model = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        let backend = CpuBackend::new(Some(2));
        model.optimize_for_backend(&backend);

        let ids = tokenizer.encode("User: fn add(a, b)\n\nAssistant:", false);
        let (_, stats) = model
            .generate(
                &backend,
                &tokenizer,
                &ids,
                6,
                &SamplerConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
                false,
                |_, _| {},
            )
            .unwrap();
        assert_eq!(stats.generated_tokens, 6);
        assert_eq!(model.kv_mode_label(), "MLA-Absorbed");
    }
}
