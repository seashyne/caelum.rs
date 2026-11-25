//! Native Qwen 3.5 hybrid Gated DeltaNet + Gated Full Attention GGUF runner (`qwen35`):
//! - 3:1 hybrid layers (`full_attention_interval = 4`):
//!   - **Gated DeltaNet (Linear Attention / SSM)** layers: fused `attn_qkv` + `attn_gate`,
//!     depthwise causal `ssm_conv1d` (kernel 4) + SiLU, L2-normalized Q/K, exact Delta Rule
//!     recurrence (`iq = h % num_k_heads`, `scale = 1/sqrt(d_v)`), gated RMSNorm (`ssm_norm`),
//!     and `ssm_out` projection.
//!   - **Gated Full Attention** layers: fused `[Q, gate]` per-head interleaved `attn_q`,
//!     QK-RMSNorm, Partial NeoX RoPE (`rope_dim = 64`), GQA KV cache, `attn_out * sigmoid(gate)`,
//!     and `attn_output` projection.
//! - SwiGLU FFN (`post_attention_norm` -> `ffn_gate`/`ffn_up`/`ffn_down`)
//! - Online AVX-VNNI `PackedMatrix` repacking + chunked Batched Prefill

use crate::cache::{KvCache, KvPrecision};
use crate::rope::{RopeConfig, RopeTable};
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
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn l2_norm_inplace(v: &mut [f32], eps: f32) {
    let ss: f32 = v.iter().map(|&x| x * x).sum();
    let inv = 1.0 / (ss + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub full_attention_interval: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope: RopeConfig,
    // Gated DeltaNet (SSM) params
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let p = "qwen35";
        let hidden_size = gguf.require_usize(&format!("{p}.embedding_length"))?;
        let intermediate_size = gguf.require_usize(&format!("{p}.feed_forward_length"))?;
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
        let full_attention_interval = gguf
            .get_meta(&format!("{p}.full_attention_interval"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4)
            .max(1);
        let rms_norm_eps = gguf
            .get_meta(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ctx_len = gguf
            .get_meta(&format!("{p}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));
        let rope = RopeConfig::from_gguf(gguf, p, head_dim, ctx_len);

        let ssm_conv_kernel = gguf
            .get_meta(&format!("{p}.ssm.conv_kernel"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4);
        let ssm_state_size = gguf
            .get_meta(&format!("{p}.ssm.state_size"))
            .and_then(|v| v.as_usize())
            .unwrap_or(128);
        let ssm_group_count = gguf
            .get_meta(&format!("{p}.ssm.group_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(16);
        let ssm_time_step_rank = gguf
            .get_meta(&format!("{p}.ssm.time_step_rank"))
            .and_then(|v| v.as_usize())
            .unwrap_or(32);
        let ssm_inner_size = gguf
            .get_meta(&format!("{p}.ssm.inner_size"))
            .and_then(|v| v.as_usize())
            .unwrap_or(hidden_size);

        let tok_embd = gguf.tensor("token_embd.weight")?;
        let vocab_size = tok_embd.shape[0];

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            full_attention_interval,
            rms_norm_eps,
            max_seq_len,
            rope,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
        })
    }

    #[inline]
    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval == 0
    }
}

struct DeltaNetLayer<'a> {
    wqkv: TensorView<'a>,
    wgate: TensorView<'a>,
    w_alpha: TensorView<'a>,
    w_beta: TensorView<'a>,
    w_out: TensorView<'a>,
    p_wqkv: Option<PackedMatrix>,
    p_wgate: Option<PackedMatrix>,
    p_alpha: Option<PackedMatrix>,
    p_beta: Option<PackedMatrix>,
    p_out: Option<PackedMatrix>,

    /// Conv1D weights `[channels, kernel]` (row-major).
    conv_w: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,

    /// Recurrent causal conv history `[kernel - 1, channels]`.
    conv_state: Vec<f32>,
    /// Recurrent DeltaNet state `[num_v_heads, head_v_dim, head_v_dim]` stored transposed `S[j][i]`.
    ssm_state: Vec<f32>,
}

struct FullAttnLayer<'a> {
    wq_gate: TensorView<'a>,
    wk: TensorView<'a>,
    wv: TensorView<'a>,
    wo: TensorView<'a>,
    p_wq_gate: Option<PackedMatrix>,
    p_wk: Option<PackedMatrix>,
    p_wv: Option<PackedMatrix>,
    p_wo: Option<PackedMatrix>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    kv: KvCache,
}

enum MixerLayer<'a> {
    DeltaNet(Box<DeltaNetLayer<'a>>),
    FullAttn(Box<FullAttnLayer<'a>>),
}

struct Qwen35Layer<'a> {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    mixer: MixerLayer<'a>,
    w_gate: TensorView<'a>,
    w_up: TensorView<'a>,
    w_down: TensorView<'a>,
    p_gate: Option<PackedMatrix>,
    p_up: Option<PackedMatrix>,
    p_down: Option<PackedMatrix>,
}

pub struct Qwen35Model<'a> {
    pub config: Qwen35Config,
    token_embd: TensorView<'a>,
    output_norm: Vec<f32>,
    output_weight: TensorView<'a>,
    p_output: Option<PackedMatrix>,
    layers: Vec<Qwen35Layer<'a>>,
    rope: RopeTable,
    kv_precision: KvPrecision,

    // Scratch buffers
    x: Vec<f32>,
    xb: Vec<f32>,
    qg_full: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_gate: Vec<f32>,
    attn_out: Vec<f32>,
    att: Vec<f32>,
    qkv_ssm: Vec<f32>,
    conv_out: Vec<f32>,
    z_ssm: Vec<f32>,
    alpha_buf: Vec<f32>,
    beta_buf: Vec<f32>,
    ssm_y: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    logits: Vec<f32>,
    x_q8k: Vec<BlockQ8_K>,
    x_q8: Vec<BlockQ8_0>,
    hb_q8k: Vec<BlockQ8_K>,
    hb_q8: Vec<BlockQ8_0>,
}

impl<'a> Qwen35Model<'a> {
    pub fn from_gguf_with_kv(
        gguf: &'a GgufFile,
        max_seq_override: Option<usize>,
        kv_precision: KvPrecision,
    ) -> Result<Self> {
        let config = Qwen35Config::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };
        let rope = RopeTable::new(config.max_seq_len, &config.rope);

        let num_k_heads = config.ssm_group_count;
        let head_k_dim = config.ssm_state_size;
        let num_v_heads = config.ssm_time_step_rank;
        let head_v_dim = config.ssm_inner_size / num_v_heads;
        let conv_channels = 2 * num_k_heads * head_k_dim + config.ssm_inner_size;
        let conv_hist = config.ssm_conv_kernel.saturating_sub(1);

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let attn_norm = gguf.tensor(&format!("blk.{l}.attn_norm.weight"))?.to_f32_vec()?;
            let post_attn_norm = gguf
                .tensor(&format!("blk.{l}.post_attention_norm.weight"))?
                .to_f32_vec()?;

            let mixer = if config.is_full_attn(l) || gguf.has_tensor(&format!("blk.{l}.attn_q.weight")) {
                let wq_gate = gguf.tensor(&format!("blk.{l}.attn_q.weight"))?;
                let wk = gguf.tensor(&format!("blk.{l}.attn_k.weight"))?;
                let wv = gguf.tensor(&format!("blk.{l}.attn_v.weight"))?;
                let wo = gguf.tensor(&format!("blk.{l}.attn_output.weight"))?;
                let q_norm = gguf
                    .tensor(&format!("blk.{l}.attn_q_norm.weight"))?
                    .to_f32_vec()?;
                let k_norm = gguf
                    .tensor(&format!("blk.{l}.attn_k_norm.weight"))?
                    .to_f32_vec()?;
                let kv = KvCache::new(
                    1,
                    config.max_seq_len,
                    config.num_kv_heads,
                    config.head_dim,
                    kv_precision,
                );
                MixerLayer::FullAttn(Box::new(FullAttnLayer {
                    wq_gate,
                    wk,
                    wv,
                    wo,
                    p_wq_gate: None,
                    p_wk: None,
                    p_wv: None,
                    p_wo: None,
                    q_norm,
                    k_norm,
                    kv,
                }))
            } else {
                let wqkv = gguf.tensor(&format!("blk.{l}.attn_qkv.weight"))?;
                let wgate = gguf.tensor(&format!("blk.{l}.attn_gate.weight"))?;
                let w_alpha = gguf.tensor(&format!("blk.{l}.ssm_alpha.weight"))?;
                let w_beta = gguf.tensor(&format!("blk.{l}.ssm_beta.weight"))?;
                let w_out = gguf.tensor(&format!("blk.{l}.ssm_out.weight"))?;
                let conv_w = gguf.tensor(&format!("blk.{l}.ssm_conv1d.weight"))?.to_f32_vec()?;
                let ssm_a = gguf.tensor(&format!("blk.{l}.ssm_a"))?.to_f32_vec()?;
                let ssm_dt_bias = gguf.tensor(&format!("blk.{l}.ssm_dt.bias"))?.to_f32_vec()?;
                let ssm_norm = gguf.tensor(&format!("blk.{l}.ssm_norm.weight"))?.to_f32_vec()?;
                MixerLayer::DeltaNet(Box::new(DeltaNetLayer {
                    wqkv,
                    wgate,
                    w_alpha,
                    w_beta,
                    w_out,
                    p_wqkv: None,
                    p_wgate: None,
                    p_alpha: None,
                    p_beta: None,
                    p_out: None,
                    conv_w,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    conv_state: vec![0.0; conv_hist * conv_channels],
                    ssm_state: vec![0.0; num_v_heads * head_v_dim * head_v_dim],
                }))
            };

            let w_gate = gguf.tensor(&format!("blk.{l}.ffn_gate.weight"))?;
            let w_up = gguf.tensor(&format!("blk.{l}.ffn_up.weight"))?;
            let w_down = gguf.tensor(&format!("blk.{l}.ffn_down.weight"))?;

            layers.push(Qwen35Layer {
                attn_norm,
                post_attn_norm,
                mixer,
                w_gate,
                w_up,
                w_down,
                p_gate: None,
                p_up: None,
                p_down: None,
            });
        }

        let h = config.hidden_size;
        let ffn = config.intermediate_size;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let max_in = h.max(q_dim).max(config.ssm_inner_size).max(ffn);

        Ok(Self {
            x: vec![0.0; h],
            xb: vec![0.0; h],
            qg_full: vec![0.0; q_dim * 2],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_gate: vec![0.0; q_dim],
            attn_out: vec![0.0; q_dim],
            att: vec![0.0; config.max_seq_len],
            qkv_ssm: vec![0.0; conv_channels],
            conv_out: vec![0.0; conv_channels],
            z_ssm: vec![0.0; config.ssm_inner_size],
            alpha_buf: vec![0.0; num_v_heads],
            beta_buf: vec![0.0; num_v_heads],
            ssm_y: vec![0.0; config.ssm_inner_size],
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
            rope,
            kv_precision,
        })
    }

    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                MixerLayer::DeltaNet(dn) => {
                    dn.p_wqkv = backend.pack_matrix(&dn.wqkv);
                    dn.p_wgate = backend.pack_matrix(&dn.wgate);
                    dn.p_alpha = backend.pack_matrix(&dn.w_alpha);
                    dn.p_beta = backend.pack_matrix(&dn.w_beta);
                    dn.p_out = backend.pack_matrix(&dn.w_out);
                }
                MixerLayer::FullAttn(fa) => {
                    fa.p_wq_gate = backend.pack_matrix(&fa.wq_gate);
                    fa.p_wk = backend.pack_matrix(&fa.wk);
                    fa.p_wv = backend.pack_matrix(&fa.wv);
                    fa.p_wo = backend.pack_matrix(&fa.wo);
                }
            }
            layer.p_gate = backend.pack_matrix(&layer.w_gate);
            layer.p_up = backend.pack_matrix(&layer.w_up);
            layer.p_down = backend.pack_matrix(&layer.w_down);
        }
        self.p_output = backend.pack_matrix(&self.output_weight);
    }

    pub fn packed_bytes(&self) -> usize {
        let mut total = self.p_output.as_ref().map_or(0, PackedMatrix::byte_size);
        for l in &self.layers {
            match &l.mixer {
                MixerLayer::DeltaNet(dn) => {
                    for p in [&dn.p_wqkv, &dn.p_wgate, &dn.p_alpha, &dn.p_beta, &dn.p_out] {
                        total += p.as_ref().map_or(0, PackedMatrix::byte_size);
                    }
                }
                MixerLayer::FullAttn(fa) => {
                    for p in [&fa.p_wq_gate, &fa.p_wk, &fa.p_wv, &fa.p_wo] {
                        total += p.as_ref().map_or(0, PackedMatrix::byte_size);
                    }
                }
            }
            for p in [&l.p_gate, &l.p_up, &l.p_down] {
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
        self.layers
            .iter()
            .map(|l| match &l.mixer {
                MixerLayer::FullAttn(fa) => fa.kv.memory_bytes(),
                MixerLayer::DeltaNet(dn) => {
                    (dn.conv_state.len() + dn.ssm_state.len()) * std::mem::size_of::<f32>()
                }
            })
            .sum()
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            match &mut l.mixer {
                MixerLayer::FullAttn(fa) => fa.kv.clear(),
                MixerLayer::DeltaNet(dn) => {
                    dn.conv_state.fill(0.0);
                    dn.ssm_state.fill(0.0);
                }
            }
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

    /// Executes one token step of Gated DeltaNet (causal conv1d + SiLU + L2-norm + Delta Rule + gated RMSNorm).
    #[allow(clippy::too_many_arguments)]
    fn step_delta_net_core(
        backend: &dyn Backend,
        dn: &mut DeltaNetLayer<'_>,
        qkv_in: &[f32],
        z_in: &[f32],
        alpha_raw: &[f32],
        beta_raw: &[f32],
        conv_kernel: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        eps: f32,
        conv_out: &mut [f32],
        y_out: &mut [f32],
    ) {
        let channels = qkv_in.len();
        let hist = conv_kernel - 1;

        // 1. Causal depthwise Conv1D + SiLU, updating ring/shift history [hist, channels]
        for c in 0..channels {
            let w_base = c * conv_kernel;
            let mut acc = qkv_in[c] * dn.conv_w[w_base + hist];
            for k in 0..hist {
                acc += dn.conv_state[k * channels + c] * dn.conv_w[w_base + k];
            }
            conv_out[c] = silu(acc);
        }
        if hist > 0 {
            if hist > 1 {
                dn.conv_state.copy_within(channels..hist * channels, 0);
            }
            dn.conv_state[(hist - 1) * channels..hist * channels].copy_from_slice(qkv_in);
        }

        // 2. Split Q, K, V and L2-normalize Q and K per head
        let qk_dim = num_k_heads * head_k_dim;
        let (q_all, kv_rest) = conv_out.split_at_mut(qk_dim);
        let (k_all, v_all) = kv_rest.split_at_mut(qk_dim);

        for kh in 0..num_k_heads {
            l2_norm_inplace(&mut q_all[kh * head_k_dim..(kh + 1) * head_k_dim], eps);
            l2_norm_inplace(&mut k_all[kh * head_k_dim..(kh + 1) * head_k_dim], eps);
        }

        // 3. Per-head Gated Delta Rule recurrence (parallelized across `num_v_heads` = 32 heads!)
        // Exact match with `ggml_compute_forward_gated_delta_net`:
        //   iq = iv % num_k_heads, ik = iv % num_k_heads, scale = 1 / sqrt(head_v_dim)
        let scale = 1.0 / (head_v_dim as f32).sqrt();
        let state_stride = head_v_dim * head_v_dim;
        let ssm_ptr = dn.ssm_state.as_mut_ptr() as usize;
        let y_ptr = y_out.as_mut_ptr() as usize;
        let q_ref: &[f32] = q_all;
        let k_ref: &[f32] = k_all;
        let v_ref: &[f32] = v_all;
        let ssm_a = &dn.ssm_a;
        let ssm_dt_bias = &dn.ssm_dt_bias;
        let ssm_norm = &dn.ssm_norm;

        backend.parallel_for(num_v_heads, &|start_h, end_h| {
            let mut delta = vec![0.0f32; head_v_dim];
            let mut raw_y = vec![0.0f32; head_v_dim];
            let mut norm_y = vec![0.0f32; head_v_dim];

            for iv in start_h..end_h {
                let ik = iv % num_k_heads;
                let q_h = &q_ref[ik * head_k_dim..(ik + 1) * head_k_dim];
                let k_h = &k_ref[ik * head_k_dim..(ik + 1) * head_k_dim];
                let v_h = &v_ref[iv * head_v_dim..(iv + 1) * head_v_dim];
                let z_h = &z_in[iv * head_v_dim..(iv + 1) * head_v_dim];

                let beta = sigmoid(beta_raw[iv]);
                let g = ssm_a[iv] * softplus(alpha_raw[iv] + ssm_dt_bias[iv]);
                let decay = g.exp();

                // SAFETY: Each head `iv` owns a disjoint slice `[iv*state_stride .. (iv+1)*state_stride]`
                // and `[iv*head_v_dim .. (iv+1)*head_v_dim]`.
                let s = unsafe {
                    std::slice::from_raw_parts_mut(
                        (ssm_ptr as *mut f32).add(iv * state_stride),
                        state_stride,
                    )
                };
                let yh = unsafe {
                    std::slice::from_raw_parts_mut(
                        (y_ptr as *mut f32).add(iv * head_v_dim),
                        head_v_dim,
                    )
                };

                // S *= exp(g)
                for val in s.iter_mut() {
                    *val *= decay;
                }

                // delta[j] = (v[j] - dot(S_col_j, k)) * beta
                for j in 0..head_v_dim {
                    let row = &s[j * head_v_dim..(j + 1) * head_v_dim];
                    let pred = backend.dot_f32(row, k_h);
                    delta[j] = (v_h[j] - pred) * beta;
                }

                // S_col_j += delta[j] * k, then y[j] = dot(S_col_j, q) * scale
                for j in 0..head_v_dim {
                    let d = delta[j];
                    let row = &mut s[j * head_v_dim..(j + 1) * head_v_dim];
                    for i in 0..head_v_dim {
                        row[i] += d * k_h[i];
                    }
                    raw_y[j] = backend.dot_f32(row, q_h) * scale;
                }

                // Gated RMSNorm: rms_norm(raw_y, ssm_norm) * silu(z_h)
                backend.rms_norm(&raw_y, ssm_norm, eps, &mut norm_y);
                for j in 0..head_v_dim {
                    yh[j] = norm_y[j] * silu(z_h[j]);
                }
            }
        });
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
        let n_kv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let attn_scale = 1.0 / (hd as f32).sqrt();

        let conv_kernel = self.config.ssm_conv_kernel;
        let num_k_heads = self.config.ssm_group_count;
        let head_k_dim = self.config.ssm_state_size;
        let num_v_heads = self.config.ssm_time_step_rank;
        let head_v_dim = self.config.ssm_inner_size / num_v_heads;

        self.token_embd.dequantize_row(token_id as usize, &mut self.x)?;

        for layer in &mut self.layers {
            backend.rms_norm(&self.x, &layer.attn_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);

            match &mut layer.mixer {
                MixerLayer::DeltaNet(dn) => {
                    Self::matvec_fast(backend, &dn.wqkv, dn.p_wqkv.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.qkv_ssm)?;
                    Self::matvec_fast(backend, &dn.wgate, dn.p_wgate.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.z_ssm)?;
                    Self::matvec_fast(backend, &dn.w_alpha, dn.p_alpha.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.alpha_buf)?;
                    Self::matvec_fast(backend, &dn.w_beta, dn.p_beta.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.beta_buf)?;

                    Self::step_delta_net_core(
                        backend,
                        dn,
                        &self.qkv_ssm,
                        &self.z_ssm,
                        &self.alpha_buf,
                        &self.beta_buf,
                        conv_kernel,
                        num_k_heads,
                        head_k_dim,
                        num_v_heads,
                        head_v_dim,
                        eps,
                        &mut self.conv_out,
                        &mut self.ssm_y,
                    );

                    Self::prep_q8(&self.ssm_y, &mut self.x_q8k, &mut self.x_q8);
                    Self::matvec_fast(backend, &dn.w_out, dn.p_out.as_ref(), &self.ssm_y, &self.x_q8k, &self.x_q8, &mut self.xb)?;
                }
                MixerLayer::FullAttn(fa) => {
                    // Joint [Q, gate] interleaved per head: `[n_heads, 2, head_dim]`
                    Self::matvec_fast(backend, &fa.wq_gate, fa.p_wq_gate.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.qg_full)?;
                    Self::matvec_fast(backend, &fa.wk, fa.p_wk.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.k)?;
                    Self::matvec_fast(backend, &fa.wv, fa.p_wv.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.v)?;

                    // De-interleave Q and gate, apply QK-norm + partial NeoX RoPE (64 dims)
                    for head in 0..n_heads {
                        let q_src = &self.qg_full[head * 2 * hd..(head * 2 + 1) * hd];
                        let g_src = &self.qg_full[(head * 2 + 1) * hd..(head * 2 + 2) * hd];
                        let q_dst = &mut self.q[head * hd..(head + 1) * hd];
                        backend.rms_norm(q_src, &fa.q_norm, eps, q_dst);
                        self.rope.apply(q_dst, pos);
                        self.attn_gate[head * hd..(head + 1) * hd].copy_from_slice(g_src);
                    }
                    for kh in 0..n_kv {
                        let k_dst = &mut self.k[kh * hd..(kh + 1) * hd];
                        let mut tmp = [0.0f32; 512];
                        backend.rms_norm(k_dst, &fa.k_norm, eps, &mut tmp[..hd]);
                        k_dst.copy_from_slice(&tmp[..hd]);
                        self.rope.apply(k_dst, pos);
                    }

                    fa.kv.store(0, pos, &self.k[..kv_dim], &self.v[..kv_dim]);

                    let heads_per_kv = (n_heads / n_kv).max(1);
                    for head in 0..n_heads {
                        let kh = head / heads_per_kv;
                        fa.kv.attend_head(
                            backend,
                            0,
                            kh,
                            &self.q[head * hd..(head + 1) * hd],
                            pos,
                            attn_scale,
                            &mut self.att,
                            &mut self.attn_out[head * hd..(head + 1) * hd],
                        );
                    }

                    // Apply sigmoid attention output gate
                    for i in 0..q_dim {
                        self.attn_out[i] *= sigmoid(self.attn_gate[i]);
                    }

                    Self::prep_q8(&self.attn_out, &mut self.x_q8k, &mut self.x_q8);
                    Self::matvec_fast(backend, &fa.wo, fa.p_wo.as_ref(), &self.attn_out, &self.x_q8k, &self.x_q8, &mut self.xb)?;
                }
            }

            // Attention residual
            for i in 0..h {
                self.x[i] += self.xb[i];
            }

            // Post-attention norm + SwiGLU FFN
            backend.rms_norm(&self.x, &layer.post_attn_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(backend, &layer.w_gate, layer.p_gate.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.hb)?;
            Self::matvec_fast(backend, &layer.w_up, layer.p_up.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.hb2)?;
            backend.swiglu_inplace(&mut self.hb[..ffn], &self.hb2[..ffn]);

            Self::prep_q8(&self.hb, &mut self.hb_q8k, &mut self.hb_q8);
            Self::matvec_fast(backend, &layer.w_down, layer.p_down.as_ref(), &self.hb, &self.hb_q8k, &self.hb_q8, &mut self.xb)?;
            for i in 0..h {
                self.x[i] += self.xb[i];
            }
        }

        if compute_logits {
            backend.rms_norm(&self.x, &self.output_norm, eps, &mut self.xb);
            Self::prep_q8(&self.xb, &mut self.x_q8k, &mut self.x_q8);
            Self::matvec_fast(backend, &self.output_weight, self.p_output.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.logits)?;
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
        let n_kv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let attn_scale = 1.0 / (hd as f32).sqrt();

        let conv_kernel = self.config.ssm_conv_kernel;
        let num_k_heads = self.config.ssm_group_count;
        let head_k_dim = self.config.ssm_state_size;
        let num_v_heads = self.config.ssm_time_step_rank;
        let head_v_dim = self.config.ssm_inner_size / num_v_heads;
        let inner = self.config.ssm_inner_size;
        let conv_ch = 2 * num_k_heads * head_k_dim + inner;

        let total = tokens.len();
        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(PREFILL_CHUNK);
            let chunk = &tokens[offset..offset + b];
            let chunk_start = start_pos + offset;

            let mut x_batch = vec![0.0f32; b * h];
            for (t, &tok) in chunk.iter().enumerate() {
                self.token_embd
                    .dequantize_row(tok as usize, &mut x_batch[t * h..(t + 1) * h])?;
            }

            let mut xb_batch = vec![0.0f32; b * h];
            let mut hb_batch = vec![0.0f32; b * ffn];
            let mut hb2_batch = vec![0.0f32; b * ffn];
            let mut x_q8k_b = vec![BlockQ8_K::default(); b * h.div_ceil(QK_K)];
            let mut hb_q8k_b = vec![BlockQ8_K::default(); b * ffn.div_ceil(QK_K)];

            for layer in &mut self.layers {
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

                match &mut layer.mixer {
                    MixerLayer::DeltaNet(dn) => {
                        let mut qkv_b = vec![0.0f32; b * conv_ch];
                        let mut z_b = vec![0.0f32; b * inner];
                        let mut a_b = vec![0.0f32; b * num_v_heads];
                        let mut beta_b = vec![0.0f32; b * num_v_heads];
                        let mut y_b = vec![0.0f32; b * inner];

                        Self::matmul_fast(backend, &dn.wqkv, dn.p_wqkv.as_ref(), &xb_batch, &x_q8k_b, b, &mut qkv_b)?;
                        Self::matmul_fast(backend, &dn.wgate, dn.p_wgate.as_ref(), &xb_batch, &x_q8k_b, b, &mut z_b)?;
                        Self::matmul_fast(backend, &dn.w_alpha, dn.p_alpha.as_ref(), &xb_batch, &x_q8k_b, b, &mut a_b)?;
                        Self::matmul_fast(backend, &dn.w_beta, dn.p_beta.as_ref(), &xb_batch, &x_q8k_b, b, &mut beta_b)?;

                        for t in 0..b {
                            Self::step_delta_net_core(
                                backend,
                                dn,
                                &qkv_b[t * conv_ch..(t + 1) * conv_ch],
                                &z_b[t * inner..(t + 1) * inner],
                                &a_b[t * num_v_heads..(t + 1) * num_v_heads],
                                &beta_b[t * num_v_heads..(t + 1) * num_v_heads],
                                conv_kernel,
                                num_k_heads,
                                head_k_dim,
                                num_v_heads,
                                head_v_dim,
                                eps,
                                &mut self.conv_out,
                                &mut y_b[t * inner..(t + 1) * inner],
                            );
                        }

                        let mut y_q8k = vec![BlockQ8_K::default(); b * inner.div_ceil(QK_K)];
                        if inner.is_multiple_of(QK_K) {
                            let nb = inner / QK_K;
                            for t in 0..b {
                                quantize_row_q8_k(
                                    &y_b[t * inner..(t + 1) * inner],
                                    &mut y_q8k[t * nb..(t + 1) * nb],
                                );
                            }
                        }
                        Self::matmul_fast(backend, &dn.w_out, dn.p_out.as_ref(), &y_b, &y_q8k, b, &mut xb_batch)?;
                    }
                    MixerLayer::FullAttn(fa) => {
                        let mut qg_b = vec![0.0f32; b * q_dim * 2];
                        let mut k_b = vec![0.0f32; b * kv_dim];
                        let mut v_b = vec![0.0f32; b * kv_dim];
                        let mut attn_b = vec![0.0f32; b * q_dim];

                        Self::matmul_fast(backend, &fa.wq_gate, fa.p_wq_gate.as_ref(), &xb_batch, &x_q8k_b, b, &mut qg_b)?;
                        Self::matmul_fast(backend, &fa.wk, fa.p_wk.as_ref(), &xb_batch, &x_q8k_b, b, &mut k_b)?;
                        Self::matmul_fast(backend, &fa.wv, fa.p_wv.as_ref(), &xb_batch, &x_q8k_b, b, &mut v_b)?;

                        let heads_per_kv = (n_heads / n_kv).max(1);
                        for t in 0..b {
                            let pos = chunk_start + t;
                            let qg_row = &qg_b[t * q_dim * 2..(t + 1) * q_dim * 2];
                            let k_row = &mut k_b[t * kv_dim..(t + 1) * kv_dim];
                            let v_row = &v_b[t * kv_dim..(t + 1) * kv_dim];

                            for head in 0..n_heads {
                                let q_src = &qg_row[head * 2 * hd..(head * 2 + 1) * hd];
                                let g_src = &qg_row[(head * 2 + 1) * hd..(head * 2 + 2) * hd];
                                let q_dst = &mut self.q[head * hd..(head + 1) * hd];
                                backend.rms_norm(q_src, &fa.q_norm, eps, q_dst);
                                self.rope.apply(q_dst, pos);
                                self.attn_gate[head * hd..(head + 1) * hd].copy_from_slice(g_src);
                            }
                            for kh in 0..n_kv {
                                let k_dst = &mut k_row[kh * hd..(kh + 1) * hd];
                                let mut tmp = [0.0f32; 512];
                                backend.rms_norm(k_dst, &fa.k_norm, eps, &mut tmp[..hd]);
                                k_dst.copy_from_slice(&tmp[..hd]);
                                self.rope.apply(k_dst, pos);
                            }

                            fa.kv.store(0, pos, k_row, v_row);

                            let out_row = &mut attn_b[t * q_dim..(t + 1) * q_dim];
                            for head in 0..n_heads {
                                let kh = head / heads_per_kv;
                                fa.kv.attend_head(
                                    backend,
                                    0,
                                    kh,
                                    &self.q[head * hd..(head + 1) * hd],
                                    pos,
                                    attn_scale,
                                    &mut self.att,
                                    &mut out_row[head * hd..(head + 1) * hd],
                                );
                            }
                            for i in 0..q_dim {
                                out_row[i] *= sigmoid(self.attn_gate[i]);
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
                        Self::matmul_fast(backend, &fa.wo, fa.p_wo.as_ref(), &attn_b, &attn_q8k, b, &mut xb_batch)?;
                    }
                }

                for i in 0..(b * h) {
                    x_batch[i] += xb_batch[i];
                }

                for t in 0..b {
                    backend.rms_norm(
                        &x_batch[t * h..(t + 1) * h],
                        &layer.post_attn_norm,
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
                Self::matmul_fast(backend, &layer.w_gate, layer.p_gate.as_ref(), &xb_batch, &x_q8k_b, b, &mut hb_batch)?;
                Self::matmul_fast(backend, &layer.w_up, layer.p_up.as_ref(), &xb_batch, &x_q8k_b, b, &mut hb2_batch)?;
                backend.swiglu_inplace(&mut hb_batch, &hb2_batch);

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
                for i in 0..(b * h) {
                    x_batch[i] += xb_batch[i];
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
            Self::matvec_fast(backend, &self.output_weight, self.p_output.as_ref(), &self.xb, &self.x_q8k, &self.x_q8, &mut self.logits)?;
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
            .map(|i| (((i * 29 + seed * 13) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    #[test]
    fn test_qwen35_hybrid_deltanet_and_gated_full_attn_batch_parity() {
        let vocab = 16usize;
        let h = 32usize;
        let ffn = 64usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let hd = 16usize;
        let ssm_k_heads = 2usize;
        let ssm_v_heads = 4usize;
        let ssm_d = 8usize; // inner = 4 * 8 = 32
        let conv_ch = 2 * ssm_k_heads * ssm_d + ssm_v_heads * ssm_d; // 64
        let conv_k = 4usize;

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("qwen35".into()));
        w.add_meta("qwen35.block_count", GgufValue::U32(2));
        w.add_meta("qwen35.context_length", GgufValue::U32(32));
        w.add_meta("qwen35.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("qwen35.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("qwen35.attention.head_count", GgufValue::U32(n_heads as u32));
        w.add_meta("qwen35.attention.head_count_kv", GgufValue::U32(n_kv as u32));
        w.add_meta("qwen35.attention.key_length", GgufValue::U32(hd as u32));
        w.add_meta("qwen35.attention.value_length", GgufValue::U32(hd as u32));
        w.add_meta("qwen35.full_attention_interval", GgufValue::U32(2));
        w.add_meta("qwen35.rope.dimension_count", GgufValue::U32(8));
        w.add_meta("qwen35.ssm.conv_kernel", GgufValue::U32(conv_k as u32));
        w.add_meta("qwen35.ssm.state_size", GgufValue::U32(ssm_d as u32));
        w.add_meta("qwen35.ssm.group_count", GgufValue::U32(ssm_k_heads as u32));
        w.add_meta("qwen35.ssm.time_step_rank", GgufValue::U32(ssm_v_heads as u32));
        w.add_meta("qwen35.ssm.inner_size", GgufValue::U32((ssm_v_heads * ssm_d) as u32));

        w.add_tensor_f32("token_embd.weight", vec![vocab, h], &synth_vec(vocab * h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("output.weight", vec![vocab, h], &synth_vec(vocab * h, 2)).unwrap();

        // Layer 0: Gated DeltaNet
        w.add_tensor_f32("blk.0.attn_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("blk.0.post_attention_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("blk.0.attn_qkv.weight", vec![conv_ch, h], &synth_vec(conv_ch * h, 3)).unwrap();
        w.add_tensor_f32("blk.0.attn_gate.weight", vec![h, h], &synth_vec(h * h, 4)).unwrap();
        w.add_tensor_f32("blk.0.ssm_alpha.weight", vec![ssm_v_heads, h], &synth_vec(ssm_v_heads * h, 5)).unwrap();
        w.add_tensor_f32("blk.0.ssm_beta.weight", vec![ssm_v_heads, h], &synth_vec(ssm_v_heads * h, 6)).unwrap();
        w.add_tensor_f32("blk.0.ssm_conv1d.weight", vec![conv_ch, conv_k], &synth_vec(conv_ch * conv_k, 7)).unwrap();
        w.add_tensor_f32("blk.0.ssm_a", vec![ssm_v_heads], &vec![-0.05; ssm_v_heads]).unwrap();
        w.add_tensor_f32("blk.0.ssm_dt.bias", vec![ssm_v_heads], &vec![0.1; ssm_v_heads]).unwrap();
        w.add_tensor_f32("blk.0.ssm_norm.weight", vec![ssm_d], &vec![1.0; ssm_d]).unwrap();
        w.add_tensor_f32("blk.0.ssm_out.weight", vec![h, h], &synth_vec(h * h, 8)).unwrap();
        w.add_tensor_f32("blk.0.ffn_gate.weight", vec![ffn, h], &synth_vec(ffn * h, 9)).unwrap();
        w.add_tensor_f32("blk.0.ffn_up.weight", vec![ffn, h], &synth_vec(ffn * h, 10)).unwrap();
        w.add_tensor_f32("blk.0.ffn_down.weight", vec![h, ffn], &synth_vec(h * ffn, 11)).unwrap();

        // Layer 1: Gated Full Attention
        let qd = n_heads * hd;
        let kvd = n_kv * hd;
        w.add_tensor_f32("blk.1.attn_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("blk.1.post_attention_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        w.add_tensor_f32("blk.1.attn_q.weight", vec![qd * 2, h], &synth_vec(qd * 2 * h, 12)).unwrap();
        w.add_tensor_f32("blk.1.attn_k.weight", vec![kvd, h], &synth_vec(kvd * h, 13)).unwrap();
        w.add_tensor_f32("blk.1.attn_v.weight", vec![kvd, h], &synth_vec(kvd * h, 14)).unwrap();
        w.add_tensor_f32("blk.1.attn_output.weight", vec![h, qd], &synth_vec(h * qd, 15)).unwrap();
        w.add_tensor_f32("blk.1.attn_q_norm.weight", vec![hd], &vec![1.0; hd]).unwrap();
        w.add_tensor_f32("blk.1.attn_k_norm.weight", vec![hd], &vec![1.0; hd]).unwrap();
        w.add_tensor_f32("blk.1.ffn_gate.weight", vec![ffn, h], &synth_vec(ffn * h, 16)).unwrap();
        w.add_tensor_f32("blk.1.ffn_up.weight", vec![ffn, h], &synth_vec(ffn * h, 17)).unwrap();
        w.add_tensor_f32("blk.1.ffn_down.weight", vec![h, ffn], &synth_vec(h * ffn, 18)).unwrap();

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let backend = CpuBackend::new(Some(1));
        let mut m1 = Qwen35Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();
        let mut m2 = Qwen35Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();

        let toks = [2u32, 4, 6];
        for (i, &t) in toks.iter().enumerate() {
            m1.forward_token(&backend, t, i, i + 1 == toks.len()).unwrap();
        }
        let l1 = m1.logits.clone();
        let l2 = m2.forward_batch(&backend, &toks, 0, true).unwrap().to_vec();
        for (a, b) in l1.iter().zip(l2.iter()) {
            assert!(a.is_finite() && (a - b).abs() < 1e-4, "qwen35 mismatch {a} vs {b}");
        }
    }
}
