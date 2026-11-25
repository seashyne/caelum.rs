//! Native Gemma 4 dense GGUF runner (`gemma4`):
//! - Per-layer heterogeneous Sliding-Window Attention (`head_dim = 256`, `kv = 8`) and
//!   Global Full Attention (`head_dim = 512`, `kv = 1`, `k_eq_v` optional `attn_v`)
//! - QK-Norm (`attn_q_norm`, `attn_k_norm`) + scaleless Value RMSNorm (`v_norm`)
//! - Gemma 4 attention scaling (`1.0`), proportional RoPE (`rope_freqs.weight`) on global layers,
//!   local `freq_base_swa` RoPE on SWA layers
//! - Pre & Post RMSNorm on Attention and GeGLU FFN + per-layer `layer_output_scale`
//! - `sqrt(hidden_size)` embedding scaling + `final_logit_softcapping` (`30.0`)
//! - Online AVX-VNNI `PackedMatrix` repacking + chunked Batched Prefill

use crate::cache::{KvCache, KvPrecision};
use crate::rope::{RopeConfig, RopeScalingType, RopeTable};
use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, DType, Result, TensorView, QK8_0,
    QK_K,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, PackedMatrix};

const PREFILL_CHUNK: usize = 64;

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
fn rms_norm_unweighted_inplace(x: &mut [f32], eps: f32) {
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_count_kv: Vec<usize>,
    pub is_swa: Vec<bool>,
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub embed_scale: f32,
    pub final_logit_softcap: f32,
    pub rope_full: RopeConfig,
    pub rope_swa: RopeConfig,
}

impl Gemma4Config {
    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let p = "gemma4";
        let hidden_size = gguf.require_usize(&format!("{p}.embedding_length"))?;
        let intermediate_size = gguf.require_usize(&format!("{p}.feed_forward_length"))?;
        let num_layers = gguf.require_usize(&format!("{p}.block_count"))?;
        let num_heads = gguf.require_usize(&format!("{p}.attention.head_count"))?;

        let head_count_kv = gguf
            .get_meta(&format!("{p}.attention.head_count_kv"))
            .and_then(|v| v.as_usize_vec().or_else(|| v.as_usize().map(|n| vec![n; num_layers])))
            .unwrap_or_else(|| vec![num_heads; num_layers]);

        let is_swa = gguf
            .get_meta(&format!("{p}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_bool_vec())
            .unwrap_or_else(|| (0..num_layers).map(|i| (i + 1) % 6 != 0).collect());

        let head_dim_full = gguf
            .get_meta(&format!("{p}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(512);
        let head_dim_swa = gguf
            .get_meta(&format!("{p}.attention.key_length_swa"))
            .and_then(|v| v.as_usize())
            .unwrap_or(256);
        let sliding_window = gguf
            .get_meta(&format!("{p}.attention.sliding_window"))
            .and_then(|v| v.as_usize())
            .unwrap_or(1024);
        let rms_norm_eps = gguf
            .get_meta(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ctx_len = gguf
            .get_meta(&format!("{p}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));
        let final_logit_softcap = gguf
            .get_meta(&format!("{p}.final_logit_softcapping"))
            .and_then(|v| v.as_f32())
            .unwrap_or(30.0);

        let rope_full = RopeConfig::from_gguf(gguf, p, head_dim_full, ctx_len);
        let mut rope_swa = rope_full.clone();
        rope_swa.rope_dim = gguf
            .get_meta(&format!("{p}.rope.dimension_count_swa"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_swa);
        rope_swa.freq_base = gguf
            .get_meta(&format!("{p}.rope.freq_base_swa"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        rope_swa.scaling_type = RopeScalingType::None;
        rope_swa.scaling_factor = 1.0;

        let tok_embd = gguf.tensor("token_embd.weight")?;
        let vocab_size = tok_embd.shape[0];

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            head_count_kv,
            is_swa,
            head_dim_full,
            head_dim_swa,
            sliding_window,
            rms_norm_eps,
            max_seq_len,
            embed_scale: (hidden_size as f32).sqrt(),
            final_logit_softcap,
            rope_full,
            rope_swa,
        })
    }
}

struct Gemma4Layer<'a> {
    is_swa: bool,
    head_dim: usize,
    num_kv_heads: usize,
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    layer_output_scale: f32,

    wq: TensorView<'a>,
    wk: TensorView<'a>,
    wv: Option<TensorView<'a>>,
    wo: TensorView<'a>,
    w_gate: TensorView<'a>,
    w_up: TensorView<'a>,
    w_down: TensorView<'a>,

    p_wq: Option<PackedMatrix>,
    p_wk: Option<PackedMatrix>,
    p_wv: Option<PackedMatrix>,
    p_wo: Option<PackedMatrix>,
    p_gate: Option<PackedMatrix>,
    p_up: Option<PackedMatrix>,
    p_down: Option<PackedMatrix>,

    kv: KvCache,
}

pub struct Gemma4Model<'a> {
    pub config: Gemma4Config,
    token_embd: TensorView<'a>,
    output_norm: Vec<f32>,
    output_weight: TensorView<'a>,
    p_output: Option<PackedMatrix>,
    layers: Vec<Gemma4Layer<'a>>,
    rope_full: RopeTable,
    rope_swa: RopeTable,
    kv_precision: KvPrecision,

    // Scratch buffers
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    att: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    logits: Vec<f32>,
    x_q8k: Vec<BlockQ8_K>,
    x_q8: Vec<BlockQ8_0>,
    hb_q8k: Vec<BlockQ8_K>,
    hb_q8: Vec<BlockQ8_0>,
}

impl<'a> Gemma4Model<'a> {
    pub fn from_gguf_with_kv(
        gguf: &'a GgufFile,
        max_seq_override: Option<usize>,
        kv_precision: KvPrecision,
    ) -> Result<Self> {
        let config = Gemma4Config::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };

        let rope_freqs = if gguf.has_tensor("rope_freqs.weight") {
            Some(gguf.tensor("rope_freqs.weight")?.to_f32_vec()?)
        } else {
            None
        };
        let rope_full = RopeTable::with_freq_factors(
            config.max_seq_len,
            &config.rope_full,
            rope_freqs.as_deref(),
        );
        let rope_swa = RopeTable::new(config.max_seq_len, &config.rope_swa);

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let is_swa = config.is_swa.get(l).copied().unwrap_or(true);
            let head_dim = if is_swa {
                config.head_dim_swa
            } else {
                config.head_dim_full
            };
            let num_kv_heads = config.head_count_kv.get(l).copied().unwrap_or(1);

            let attn_norm = gguf.tensor(&format!("blk.{l}.attn_norm.weight"))?.to_f32_vec()?;
            let post_attn_norm = gguf
                .tensor(&format!("blk.{l}.post_attention_norm.weight"))?
                .to_f32_vec()?;
            let ffn_norm = gguf.tensor(&format!("blk.{l}.ffn_norm.weight"))?.to_f32_vec()?;
            let post_ffw_norm = gguf
                .tensor(&format!("blk.{l}.post_ffw_norm.weight"))?
                .to_f32_vec()?;
            let q_norm = gguf
                .tensor(&format!("blk.{l}.attn_q_norm.weight"))?
                .to_f32_vec()?;
            let k_norm = gguf
                .tensor(&format!("blk.{l}.attn_k_norm.weight"))?
                .to_f32_vec()?;
            let layer_output_scale = if gguf.has_tensor(&format!("blk.{l}.layer_output_scale.weight")) {
                gguf.tensor(&format!("blk.{l}.layer_output_scale.weight"))?
                    .to_f32_vec()?
                    .first()
                    .copied()
                    .unwrap_or(1.0)
            } else {
                1.0
            };

            let wq = gguf.tensor(&format!("blk.{l}.attn_q.weight"))?;
            let wk = gguf.tensor(&format!("blk.{l}.attn_k.weight"))?;
            let wv = if gguf.has_tensor(&format!("blk.{l}.attn_v.weight")) {
                Some(gguf.tensor(&format!("blk.{l}.attn_v.weight"))?)
            } else {
                None
            };
            let wo = gguf.tensor(&format!("blk.{l}.attn_output.weight"))?;
            let w_gate = gguf.tensor(&format!("blk.{l}.ffn_gate.weight"))?;
            let w_up = gguf.tensor(&format!("blk.{l}.ffn_up.weight"))?;
            let w_down = gguf.tensor(&format!("blk.{l}.ffn_down.weight"))?;

            let kv = KvCache::new(1, config.max_seq_len, num_kv_heads, head_dim, kv_precision);

            layers.push(Gemma4Layer {
                is_swa,
                head_dim,
                num_kv_heads,
                attn_norm,
                post_attn_norm,
                ffn_norm,
                post_ffw_norm,
                q_norm,
                k_norm,
                layer_output_scale,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                p_wq: None,
                p_wk: None,
                p_wv: None,
                p_wo: None,
                p_gate: None,
                p_up: None,
                p_down: None,
                kv,
            });
        }

        let h = config.hidden_size;
        let ffn = config.intermediate_size;
        let max_q_dim = config.num_heads * config.head_dim_full.max(config.head_dim_swa);
        let max_kv_dim = config
            .head_count_kv
            .iter()
            .zip(config.is_swa.iter())
            .map(|(&kh, &swa)| kh * if swa { config.head_dim_swa } else { config.head_dim_full })
            .max()
            .unwrap_or(max_q_dim);
        let max_in = h.max(max_q_dim).max(ffn);

        Ok(Self {
            x: vec![0.0; h],
            xb: vec![0.0; h],
            xb2: vec![0.0; h],
            q: vec![0.0; max_q_dim],
            k: vec![0.0; max_kv_dim],
            v: vec![0.0; max_kv_dim],
            attn_out: vec![0.0; max_q_dim],
            att: vec![0.0; config.max_seq_len],
            hb: vec![0.0; ffn],
            hb2: vec![0.0; ffn],
            logits: vec![0.0; config.vocab_size],
            x_q8k: vec![BlockQ8_K::default(); max_in.div_ceil(QK_K)],
            x_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; max_in.div_ceil(QK8_0)],
            hb_q8k: vec![BlockQ8_K::default(); ffn.div_ceil(QK_K)],
            hb_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; ffn.div_ceil(QK8_0)],
            config,
            token_embd,
            output_norm,
            output_weight,
            p_output: None,
            layers,
            rope_full,
            rope_swa,
            kv_precision,
        })
    }

    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        for layer in &mut self.layers {
            layer.p_wq = backend.pack_matrix(&layer.wq);
            layer.p_wk = backend.pack_matrix(&layer.wk);
            layer.p_wv = layer.wv.as_ref().and_then(|w| backend.pack_matrix(w));
            layer.p_wo = backend.pack_matrix(&layer.wo);
            layer.p_gate = backend.pack_matrix(&layer.w_gate);
            layer.p_up = backend.pack_matrix(&layer.w_up);
            layer.p_down = backend.pack_matrix(&layer.w_down);
        }
        self.p_output = backend.pack_matrix(&self.output_weight);
    }

    pub fn packed_bytes(&self) -> usize {
        let mut total = self.p_output.as_ref().map_or(0, PackedMatrix::byte_size);
        for l in &self.layers {
            for p in [
                &l.p_wq, &l.p_wk, &l.p_wv, &l.p_wo, &l.p_gate, &l.p_up, &l.p_down,
            ] {
                total += p.as_ref().map_or(0, PackedMatrix::byte_size);
            }
        }
        total
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_precision(&self) -> KvPrecision {
        self.kv_precision
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.kv.memory_bytes()).sum()
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.kv.clear();
        }
    }

    #[inline]
    fn matvec_fast(
        backend: &dyn Backend,
        w: &TensorView<'_>,
        packed: Option<&PackedMatrix>,
        x: &[f32],
        x_q8k: &[BlockQ8_K],
        x_q8: &[BlockQ8_0],
        out: &mut [f32],
    ) -> Result<()> {
        if let Some(p) = packed {
            backend.matvec_packed_q8k(p, &x_q8k[..x.len() / QK_K], out)
        } else if is_k_quant(w.dtype) && x.len().is_multiple_of(QK_K) {
            backend.matvec_q8k_act(w, x, &x_q8k[..x.len() / QK_K], out)
        } else if matches!(w.dtype, DType::Q4_0 | DType::Q8_0) && x.len().is_multiple_of(QK8_0) {
            backend.matvec_q8_act(w, x, &x_q8[..x.len() / QK8_0], out)
        } else {
            backend.matvec(w, x, out)
        }
    }

    #[inline]
    fn prep_q8(x: &[f32], q8k: &mut [BlockQ8_K], q8: &mut [BlockQ8_0]) {
        if x.len().is_multiple_of(QK_K) {
            quantize_row_q8_k(x, &mut q8k[..x.len() / QK_K]);
        }
        if x.len().is_multiple_of(QK8_0) {
            quantize_row_q8_0(x, &mut q8[..x.len() / QK8_0]);
        }
    }

    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32]> {
        let h = self.config.hidden_size;
        let ffn = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let n_heads = self.config.num_heads;

        self.token_embd.dequantize_row(token_id as usize, &mut self.x)?;
        let es = self.config.embed_scale;
        for v in &mut self.x {
            *v *= es;
        }

        for layer in &mut self.layers {
            let hd = layer.head_dim;
            let n_kv = layer.num_kv_heads;
            let q_dim = n_heads * hd;
            let kv_dim = n_kv * hd;
            let rope = if layer.is_swa {
                &self.rope_swa
            } else {
                &self.rope_full
            };
            let swa = if layer.is_swa {
                Some(self.config.sliding_window)
            } else {
                None
            };

            // 1. Pre-attention RMSNorm + Q/K/V projections
            backend.rms_norm(&self.x, &layer.attn_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);

            Self::matvec_fast(
                backend,
                &layer.wq,
                layer.p_wq.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.q[..q_dim],
            )?;
            Self::matvec_fast(
                backend,
                &layer.wk,
                layer.p_wk.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.k[..kv_dim],
            )?;
            if let Some(ref wv) = layer.wv {
                Self::matvec_fast(
                    backend,
                    wv,
                    layer.p_wv.as_ref(),
                    &self.xb,
                    &self.x_q8k,
                    &self.x_q8,
                    &mut self.v[..kv_dim],
                )?;
            } else {
                // Gemma 4 k_eq_v: V reuses pre-norm K projection
                self.v[..kv_dim].copy_from_slice(&self.k[..kv_dim]);
            }

            // 2. Per-head Q-norm + RoPE, K-norm + RoPE, scaleless V-norm
            for head in 0..n_heads {
                let qh = &mut self.q[head * hd..(head + 1) * hd];
                let mut tmp = [0.0f32; 512];
                backend.rms_norm(qh, &layer.q_norm, eps, &mut tmp[..hd]);
                qh.copy_from_slice(&tmp[..hd]);
                rope.apply(qh, pos);
            }
            for kh in 0..n_kv {
                let kh_vec = &mut self.k[kh * hd..(kh + 1) * hd];
                let mut tmp = [0.0f32; 512];
                backend.rms_norm(kh_vec, &layer.k_norm, eps, &mut tmp[..hd]);
                kh_vec.copy_from_slice(&tmp[..hd]);
                rope.apply(kh_vec, pos);

                let vh_vec = &mut self.v[kh * hd..(kh + 1) * hd];
                rms_norm_unweighted_inplace(vh_vec, eps);
            }

            layer.kv.store(0, pos, &self.k[..kv_dim], &self.v[..kv_dim]);

            // 3. Causal GQA Attention (Gemma 4 uses scale = 1.0)
            let heads_per_kv = (n_heads / n_kv).max(1);
            for head in 0..n_heads {
                let kh = head / heads_per_kv;
                let qh = &self.q[head * hd..(head + 1) * hd];
                let out_h = &mut self.attn_out[head * hd..(head + 1) * hd];
                layer.kv.attend_head_ext(
                    backend,
                    0,
                    kh,
                    qh,
                    pos,
                    1.0,
                    swa,
                    0.0,
                    &mut self.att,
                    out_h,
                );
            }

            // 4. Output projection + Post-Attention RMSNorm + Residual
            Self::prep_q8(&self.attn_out[..q_dim], &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(
                backend,
                &layer.wo,
                layer.p_wo.as_ref(),
                &self.attn_out[..q_dim],
                &self.x_q8k,
                &self.x_q8,
                &mut self.xb,
            )?;
            backend.rms_norm(&self.xb, &layer.post_attn_norm, eps, &mut self.xb2);
            for i in 0..h {
                self.x[i] += self.xb2[i];
            }

            // 5. Pre-FFN RMSNorm + GeGLU FFN + Post-FFN RMSNorm + Residual
            backend.rms_norm(&self.x, &layer.ffn_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(
                backend,
                &layer.w_gate,
                layer.p_gate.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.hb[..ffn],
            )?;
            Self::matvec_fast(
                backend,
                &layer.w_up,
                layer.p_up.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.hb2[..ffn],
            )?;
            for i in 0..ffn {
                self.hb[i] = gelu_tanh(self.hb[i]) * self.hb2[i];
            }
            Self::prep_q8(&self.hb[..ffn], &mut self.hb_q8k, &mut self.hb_q8);
            Self::matvec_fast(
                backend,
                &layer.w_down,
                layer.p_down.as_ref(),
                &self.hb[..ffn],
                &self.hb_q8k,
                &self.hb_q8,
                &mut self.xb,
            )?;
            backend.rms_norm(&self.xb, &layer.post_ffw_norm, eps, &mut self.xb2);
            let scale = layer.layer_output_scale;
            for i in 0..h {
                self.x[i] = (self.x[i] + self.xb2[i]) * scale;
            }
        }

        if compute_logits {
            backend.rms_norm(&self.x, &self.output_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(
                backend,
                &self.output_weight,
                self.p_output.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.logits,
            )?;
            let cap = self.config.final_logit_softcap;
            if cap > 0.0 {
                let inv = 1.0 / cap;
                for v in &mut self.logits {
                    *v = cap * (*v * inv).tanh();
                }
            }
            Ok(&self.logits)
        } else {
            Ok(&[])
        }
    }

    pub fn forward_batch(
        &mut self,
        backend: &dyn Backend,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Ok(&[]);
        }
        if tokens.len() == 1 {
            return self.forward_token(backend, tokens[0], start_pos, compute_last_logits);
        }

        let h = self.config.hidden_size;
        let ffn = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let n_heads = self.config.num_heads;
        let total = tokens.len();

        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(PREFILL_CHUNK);
            let chunk = &tokens[offset..offset + b];
            let chunk_start = start_pos + offset;

            let mut x_batch = vec![0.0f32; b * h];
            let es = self.config.embed_scale;
            for (t, &tok) in chunk.iter().enumerate() {
                let row = &mut x_batch[t * h..(t + 1) * h];
                self.token_embd.dequantize_row(tok as usize, row)?;
                for v in row.iter_mut() {
                    *v *= es;
                }
            }

            let mut xb_batch = vec![0.0f32; b * h];
            let mut xb2_batch = vec![0.0f32; b * h];
            let mut hb_batch = vec![0.0f32; b * ffn];
            let mut hb2_batch = vec![0.0f32; b * ffn];
            let mut x_q8k_b = vec![BlockQ8_K::default(); b * h.div_ceil(QK_K)];
            let mut hb_q8k_b = vec![BlockQ8_K::default(); b * ffn.div_ceil(QK_K)];

            for layer in &mut self.layers {
                let hd = layer.head_dim;
                let n_kv = layer.num_kv_heads;
                let q_dim = n_heads * hd;
                let kv_dim = n_kv * hd;
                let rope = if layer.is_swa {
                    &self.rope_swa
                } else {
                    &self.rope_full
                };
                let swa = if layer.is_swa {
                    Some(self.config.sliding_window)
                } else {
                    None
                };

                for t in 0..b {
                    backend.rms_norm(
                        &x_batch[t * h..(t + 1) * h],
                        &layer.attn_norm,
                        eps,
                        &mut xb_batch[t * h..(t + 1) * h],
                    );
                }
                if h.is_multiple_of(QK_K) {
                    let nb = h / QK_K;
                    for t in 0..b {
                        quantize_row_q8_k(
                            &xb_batch[t * h..(t + 1) * h],
                            &mut x_q8k_b[t * nb..(t + 1) * nb],
                        );
                    }
                }

                let mut q_b = vec![0.0f32; b * q_dim];
                let mut k_b = vec![0.0f32; b * kv_dim];
                let mut v_b = vec![0.0f32; b * kv_dim];
                let mut attn_b = vec![0.0f32; b * q_dim];

                Self::matmul_fast(backend, &layer.wq, layer.p_wq.as_ref(), &xb_batch, &x_q8k_b, b, &mut q_b)?;
                Self::matmul_fast(backend, &layer.wk, layer.p_wk.as_ref(), &xb_batch, &x_q8k_b, b, &mut k_b)?;
                if let Some(ref wv) = layer.wv {
                    Self::matmul_fast(backend, wv, layer.p_wv.as_ref(), &xb_batch, &x_q8k_b, b, &mut v_b)?;
                } else {
                    v_b.copy_from_slice(&k_b);
                }

                let heads_per_kv = (n_heads / n_kv).max(1);
                for t in 0..b {
                    let pos = chunk_start + t;
                    let q_row = &mut q_b[t * q_dim..(t + 1) * q_dim];
                    let k_row = &mut k_b[t * kv_dim..(t + 1) * kv_dim];
                    let v_row = &mut v_b[t * kv_dim..(t + 1) * kv_dim];

                    for head in 0..n_heads {
                        let qh = &mut q_row[head * hd..(head + 1) * hd];
                        let mut tmp = [0.0f32; 512];
                        backend.rms_norm(qh, &layer.q_norm, eps, &mut tmp[..hd]);
                        qh.copy_from_slice(&tmp[..hd]);
                        rope.apply(qh, pos);
                    }
                    for kh in 0..n_kv {
                        let kh_vec = &mut k_row[kh * hd..(kh + 1) * hd];
                        let mut tmp = [0.0f32; 512];
                        backend.rms_norm(kh_vec, &layer.k_norm, eps, &mut tmp[..hd]);
                        kh_vec.copy_from_slice(&tmp[..hd]);
                        rope.apply(kh_vec, pos);

                        let vh_vec = &mut v_row[kh * hd..(kh + 1) * hd];
                        rms_norm_unweighted_inplace(vh_vec, eps);
                    }

                    layer.kv.store(0, pos, k_row, v_row);

                    let out_row = &mut attn_b[t * q_dim..(t + 1) * q_dim];
                    for head in 0..n_heads {
                        let kh = head / heads_per_kv;
                        layer.kv.attend_head_ext(
                            backend,
                            0,
                            kh,
                            &q_row[head * hd..(head + 1) * hd],
                            pos,
                            1.0,
                            swa,
                            0.0,
                            &mut self.att,
                            &mut out_row[head * hd..(head + 1) * hd],
                        );
                    }
                }

                let mut attn_q8k = vec![BlockQ8_K::default(); b * q_dim.div_ceil(QK_K)];
                if q_dim.is_multiple_of(QK_K) {
                    let nb = q_dim / QK_K;
                    for t in 0..b {
                        quantize_row_q8_k(
                            &attn_b[t * q_dim..(t + 1) * q_dim],
                            &mut attn_q8k[t * nb..(t + 1) * nb],
                        );
                    }
                }
                Self::matmul_fast(backend, &layer.wo, layer.p_wo.as_ref(), &attn_b, &attn_q8k, b, &mut xb_batch)?;

                for t in 0..b {
                    backend.rms_norm(
                        &xb_batch[t * h..(t + 1) * h],
                        &layer.post_attn_norm,
                        eps,
                        &mut xb2_batch[t * h..(t + 1) * h],
                    );
                    let xr = &mut x_batch[t * h..(t + 1) * h];
                    let ar = &xb2_batch[t * h..(t + 1) * h];
                    for i in 0..h {
                        xr[i] += ar[i];
                    }
                    backend.rms_norm(xr, &layer.ffn_norm, eps, &mut xb_batch[t * h..(t + 1) * h]);
                }

                if h.is_multiple_of(QK_K) {
                    let nb = h / QK_K;
                    for t in 0..b {
                        quantize_row_q8_k(
                            &xb_batch[t * h..(t + 1) * h],
                            &mut x_q8k_b[t * nb..(t + 1) * nb],
                        );
                    }
                }
                Self::matmul_fast(backend, &layer.w_gate, layer.p_gate.as_ref(), &xb_batch, &x_q8k_b, b, &mut hb_batch)?;
                Self::matmul_fast(backend, &layer.w_up, layer.p_up.as_ref(), &xb_batch, &x_q8k_b, b, &mut hb2_batch)?;
                for i in 0..(b * ffn) {
                    hb_batch[i] = gelu_tanh(hb_batch[i]) * hb2_batch[i];
                }
                if ffn.is_multiple_of(QK_K) {
                    let nb = ffn / QK_K;
                    for t in 0..b {
                        quantize_row_q8_k(
                            &hb_batch[t * ffn..(t + 1) * ffn],
                            &mut hb_q8k_b[t * nb..(t + 1) * nb],
                        );
                    }
                }
                Self::matmul_fast(backend, &layer.w_down, layer.p_down.as_ref(), &hb_batch, &hb_q8k_b, b, &mut xb_batch)?;

                let scale = layer.layer_output_scale;
                for t in 0..b {
                    backend.rms_norm(
                        &xb_batch[t * h..(t + 1) * h],
                        &layer.post_ffw_norm,
                        eps,
                        &mut xb2_batch[t * h..(t + 1) * h],
                    );
                    let xr = &mut x_batch[t * h..(t + 1) * h];
                    let fr = &xb2_batch[t * h..(t + 1) * h];
                    for i in 0..h {
                        xr[i] = (xr[i] + fr[i]) * scale;
                    }
                }
            }

            if offset + b == total {
                self.x.copy_from_slice(&x_batch[(b - 1) * h..b * h]);
            }
            offset += b;
        }

        if compute_last_logits {
            backend.rms_norm(&self.x, &self.output_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(
                backend,
                &self.output_weight,
                self.p_output.as_ref(),
                &self.xb,
                &self.x_q8k,
                &self.x_q8,
                &mut self.logits,
            )?;
            let cap = self.config.final_logit_softcap;
            if cap > 0.0 {
                let inv = 1.0 / cap;
                for v in &mut self.logits {
                    *v = cap * (*v * inv).tanh();
                }
            }
            Ok(&self.logits)
        } else {
            Ok(&[])
        }
    }

    #[inline]
    fn matmul_fast(
        backend: &dyn Backend,
        w: &TensorView<'_>,
        packed: Option<&PackedMatrix>,
        x_batch: &[f32],
        x_batch_q8k: &[BlockQ8_K],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()> {
        let cols = x_batch.len() / batch;
        if let Some(p) = packed {
            backend.matmul_packed_q8k(p, x_batch_q8k, batch, out_batch)
        } else if is_k_quant(w.dtype) && cols.is_multiple_of(QK_K) {
            backend.matmul_q8k_act(w, x_batch, x_batch_q8k, batch, out_batch)
        } else {
            backend.matmul(w, x_batch, batch, out_batch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caelum_gguf::{GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;

    fn synth_vec(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 31 + seed * 17) % 19) as f32 - 9.0) * 0.02)
            .collect()
    }

    #[test]
    fn test_gemma4_hybrid_swa_and_k_eq_v_batch_parity() {
        let vocab = 16usize;
        let h = 32usize;
        let ffn = 64usize;
        let n_heads = 2usize;
        let hd_swa = 8usize;
        let hd_full = 16usize;

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("gemma4".into()));
        w.add_meta("gemma4.block_count", GgufValue::U32(2));
        w.add_meta("gemma4.context_length", GgufValue::U32(32));
        w.add_meta("gemma4.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("gemma4.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("gemma4.attention.head_count", GgufValue::U32(n_heads as u32));
        w.add_meta(
            "gemma4.attention.head_count_kv",
            GgufValue::Array(vec![GgufValue::U32(2), GgufValue::U32(1)]),
        );
        w.add_meta(
            "gemma4.attention.sliding_window_pattern",
            GgufValue::Array(vec![GgufValue::Bool(true), GgufValue::Bool(false)]),
        );
        w.add_meta("gemma4.attention.key_length_swa", GgufValue::U32(hd_swa as u32));
        w.add_meta("gemma4.attention.value_length_swa", GgufValue::U32(hd_swa as u32));
        w.add_meta("gemma4.attention.key_length", GgufValue::U32(hd_full as u32));
        w.add_meta("gemma4.attention.value_length", GgufValue::U32(hd_full as u32));
        w.add_meta("gemma4.attention.sliding_window", GgufValue::U32(8));
        w.add_meta("gemma4.rope.dimension_count", GgufValue::U32(hd_full as u32));
        w.add_meta("gemma4.rope.dimension_count_swa", GgufValue::U32(hd_swa as u32));
        w.add_meta("gemma4.final_logit_softcapping", GgufValue::F32(30.0));

        w.add_tensor_f32("token_embd.weight", vec![vocab, h], &synth_vec(vocab * h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("rope_freqs.weight", vec![hd_full / 2], &vec![1.0; hd_full / 2]).unwrap();

        for (l, &(hd, n_kv, has_v)) in [(hd_swa, 2usize, true), (hd_full, 1usize, false)].iter().enumerate() {
            let qd = n_heads * hd;
            let kvd = n_kv * hd;
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.post_attention_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.post_ffw_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q_norm.weight"), vec![hd], &vec![1.0; hd]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k_norm.weight"), vec![hd], &vec![1.0; hd]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.layer_output_scale.weight"), vec![1], &[0.5]).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_q.weight"), vec![qd, h], &synth_vec(qd * h, l + 2)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.attn_k.weight"), vec![kvd, h], &synth_vec(kvd * h, l + 3)).unwrap();
            if has_v {
                w.add_tensor_f32(&format!("blk.{l}.attn_v.weight"), vec![kvd, h], &synth_vec(kvd * h, l + 4)).unwrap();
            }
            w.add_tensor_f32(&format!("blk.{l}.attn_output.weight"), vec![h, qd], &synth_vec(h * qd, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_gate.weight"), vec![ffn, h], &synth_vec(ffn * h, l + 6)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], &synth_vec(ffn * h, l + 7)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], &synth_vec(h * ffn, l + 8)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let backend = CpuBackend::new(Some(1));
        let mut m1 = Gemma4Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();
        let mut m2 = Gemma4Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();

        let toks = [1u32, 3, 5];
        for (i, &t) in toks.iter().enumerate() {
            m1.forward_token(&backend, t, i, i + 1 == toks.len()).unwrap();
        }
        let l1 = m1.logits.clone();
        let l2 = m2.forward_batch(&backend, &toks, 0, true).unwrap().to_vec();
        for (a, b) in l1.iter().zip(l2.iter()) {
            assert!(a.is_finite() && (a - b).abs() < 1e-4, "gemma4 mismatch {a} vs {b}");
        }
    }
}
