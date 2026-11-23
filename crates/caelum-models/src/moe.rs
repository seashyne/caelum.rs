//! Shared Sparse Mixture-of-Experts (MoE) FFN execution engine supporting:
//! - `Qwen3MoE` (`Qwen3-Coder-30B-A3B`, `Qwen3-235B-A22B`)
//! - `Qwen2MoE` (`Qwen2-57B-A14B`, `Qwen1.5-MoE` with gated shared expert)
//! - `DeepSeek2` (`DeepSeek-Coder-V2-Lite`, `DeepSeek-V2`, `DeepSeek-V3`, `DeepSeek-R1`)
//!
//! Uses zero-copy 3D `TensorView::slice_outer(expert_id)` and shares dynamic `Q8_K`/`Q8_0`
//! input quantization across all active routed experts and optional shared experts.

use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, DType, Result, TensorView, QK8_0,
    QK_K,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, PackedMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGating {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_active_experts: usize,
    pub expert_ffn_dim: usize,
    pub shared_ffn_dim: usize,
    pub gating: ExpertGating,
    pub norm_weights: bool,
    pub weights_scale: f32,
}

pub struct MoeLayerWeights<'a> {
    pub gate_inp: TensorView<'a>,
    /// Optional routing selection bias (`blk.{i}.ffn_exp_probs_b.bias`, DeepSeek-V3/R1).
    pub exp_probs_b: Option<Vec<f32>>,
    /// 3D stacked routed expert weights `[n_expert, ffn_exp, hidden]`.
    pub gate_exps: TensorView<'a>,
    pub up_exps: TensorView<'a>,
    /// 3D stacked routed expert down weights `[n_expert, hidden, ffn_exp]`.
    pub down_exps: TensorView<'a>,
    /// Optional always-active shared expert (`DeepSeek2`, `Qwen2MoE`).
    pub shexp_gate: Option<TensorView<'a>>,
    pub shexp_up: Option<TensorView<'a>>,
    pub shexp_down: Option<TensorView<'a>>,
    /// Optional repacked shared expert matrices for AVX-VNNI speed.
    pub p_shexp_gate: Option<PackedMatrix>,
    pub p_shexp_up: Option<PackedMatrix>,
    pub p_shexp_down: Option<PackedMatrix>,
    /// Optional scalar gate vector for shared expert (`blk.{i}.ffn_gate_inp_shexp.weight`, Qwen2MoE).
    pub shexp_gate_inp: Option<Vec<f32>>,
}

impl<'a> MoeLayerWeights<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, layer_idx: usize) -> Result<Self> {
        let gate_inp = gguf.tensor(&format!("blk.{layer_idx}.ffn_gate_inp.weight"))?;
        let exp_probs_b = if gguf.has_tensor(&format!("blk.{layer_idx}.ffn_exp_probs_b.bias")) {
            Some(
                gguf.tensor(&format!("blk.{layer_idx}.ffn_exp_probs_b.bias"))?
                    .to_f32_vec()?,
            )
        } else {
            None
        };
        let gate_exps = gguf.tensor(&format!("blk.{layer_idx}.ffn_gate_exps.weight"))?;
        let up_exps = gguf.tensor(&format!("blk.{layer_idx}.ffn_up_exps.weight"))?;
        let down_exps = gguf.tensor(&format!("blk.{layer_idx}.ffn_down_exps.weight"))?;

        let (shexp_gate, shexp_up, shexp_down) =
            if gguf.has_tensor(&format!("blk.{layer_idx}.ffn_gate_shexp.weight")) {
                (
                    Some(gguf.tensor(&format!("blk.{layer_idx}.ffn_gate_shexp.weight"))?),
                    Some(gguf.tensor(&format!("blk.{layer_idx}.ffn_up_shexp.weight"))?),
                    Some(gguf.tensor(&format!("blk.{layer_idx}.ffn_down_shexp.weight"))?),
                )
            } else {
                (None, None, None)
            };

        let shexp_gate_inp =
            if gguf.has_tensor(&format!("blk.{layer_idx}.ffn_gate_inp_shexp.weight")) {
                Some(
                    gguf.tensor(&format!("blk.{layer_idx}.ffn_gate_inp_shexp.weight"))?
                        .to_f32_vec()?,
                )
            } else {
                None
            };

        Ok(Self {
            gate_inp,
            exp_probs_b,
            gate_exps,
            up_exps,
            down_exps,
            shexp_gate,
            shexp_up,
            shexp_down,
            p_shexp_gate: None,
            p_shexp_up: None,
            p_shexp_down: None,
            shexp_gate_inp,
        })
    }

    pub fn optimize_for_backend(&mut self, backend: &dyn Backend) {
        if let Some(w) = &self.shexp_gate {
            self.p_shexp_gate = backend.pack_matrix(w);
        }
        if let Some(w) = &self.shexp_up {
            self.p_shexp_up = backend.pack_matrix(w);
        }
        if let Some(w) = &self.shexp_down {
            self.p_shexp_down = backend.pack_matrix(w);
        }
    }
}

pub struct MoeScratch {
    pub router_logits: Vec<f32>,
    pub router_probs: Vec<f32>,
    pub exp_gate: Vec<f32>,
    pub exp_up: Vec<f32>,
    pub exp_out: Vec<f32>,
    pub sh_gate: Vec<f32>,
    pub sh_up: Vec<f32>,
    pub exp_q8k: Vec<BlockQ8_K>,
    pub exp_q8: Vec<BlockQ8_0>,
    pub sh_q8k: Vec<BlockQ8_K>,
    pub sh_q8: Vec<BlockQ8_0>,
}

impl MoeScratch {
    pub fn new(hidden_size: usize, cfg: &MoeConfig) -> Self {
        let effn = cfg.expert_ffn_dim.max(1);
        let sffn = cfg.shared_ffn_dim.max(1);
        Self {
            router_logits: vec![0.0; cfg.num_experts.max(1)],
            router_probs: vec![0.0; cfg.num_experts.max(1)],
            exp_gate: vec![0.0; effn],
            exp_up: vec![0.0; effn],
            exp_out: vec![0.0; hidden_size],
            sh_gate: vec![0.0; sffn],
            sh_up: vec![0.0; sffn],
            exp_q8k: vec![BlockQ8_K::default(); (effn / QK_K).max(1)],
            exp_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (effn / QK8_0).max(1)],
            sh_q8k: vec![BlockQ8_K::default(); (sffn / QK_K).max(1)],
            sh_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (sffn / QK8_0).max(1)],
        }
    }
}

#[inline]
fn is_k_quant(dt: DType) -> bool {
    matches!(dt, DType::Q4_K | DType::Q5_K | DType::Q6_K)
}

#[inline]
fn is_q8_compat(dt: DType) -> bool {
    matches!(dt, DType::Q4_0 | DType::Q8_0)
}

/// Evaluates one MoE layer (routed Top-K experts + optional shared expert) for one token
/// and writes the combined FFN output into `out` (`[hidden_size]`).
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_token(
    backend: &dyn Backend,
    layer: &MoeLayerWeights<'_>,
    cfg: &MoeConfig,
    x_norm: &[f32],
    x_q8k: Option<&[BlockQ8_K]>,
    x_q8: Option<&[BlockQ8_0]>,
    scratch: &mut MoeScratch,
    out: &mut [f32],
) -> Result<()> {
    let n_exp = cfg.num_experts;
    let top_k = cfg.num_active_experts.min(n_exp);
    let effn = cfg.expert_ffn_dim;

    // 1. Router logits -> probabilities (Softmax or Sigmoid)
    backend.matvec(&layer.gate_inp, x_norm, &mut scratch.router_logits[..n_exp])?;
    scratch.router_probs[..n_exp].copy_from_slice(&scratch.router_logits[..n_exp]);
    match cfg.gating {
        ExpertGating::Softmax => backend.softmax(&mut scratch.router_probs[..n_exp]),
        ExpertGating::Sigmoid => {
            for p in &mut scratch.router_probs[..n_exp] {
                *p = 1.0 / (1.0 + (-*p).exp());
            }
        }
    }

    // 2. Top-K expert selection (using optional DeepSeek-V3 bias for selection only)
    let mut ranked: Vec<(usize, f32)> = (0..n_exp)
        .map(|e| {
            let score = if let Some(bias) = &layer.exp_probs_b {
                scratch.router_probs[e] + bias[e]
            } else {
                scratch.router_probs[e]
            };
            (e, score)
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut top_weights = Vec::with_capacity(top_k);
    let mut w_sum = 0.0f32;
    for &(e, _) in ranked.iter().take(top_k) {
        let w = scratch.router_probs[e];
        top_weights.push((e, w));
        w_sum += w;
    }
    if cfg.norm_weights && w_sum > 0.0 {
        let inv = 1.0 / w_sum;
        for (_, w) in &mut top_weights {
            *w *= inv;
        }
    }
    if cfg.weights_scale != 1.0 {
        for (_, w) in &mut top_weights {
            *w *= cfg.weights_scale;
        }
    }

    out.fill(0.0);

    // 3. Execute ONLY the Top-K selected experts (reusing quantized input activation!)
    let can_exp_q8k = effn % QK_K == 0;
    let can_exp_q8 = effn % QK8_0 == 0;

    for &(exp_id, weight) in &top_weights {
        let w_gate = layer.gate_exps.slice_outer(exp_id)?;
        let w_up = layer.up_exps.slice_outer(exp_id)?;
        let w_down = layer.down_exps.slice_outer(exp_id)?;

        if let (Some(xq), true) = (x_q8k, is_k_quant(w_gate.dtype) && is_k_quant(w_up.dtype)) {
            backend.matvec_q8k_act(&w_gate, x_norm, xq, &mut scratch.exp_gate[..effn])?;
            backend.matvec_q8k_act(&w_up, x_norm, xq, &mut scratch.exp_up[..effn])?;
        } else if let (Some(xq), true) = (x_q8, is_q8_compat(w_gate.dtype) && is_q8_compat(w_up.dtype)) {
            backend.matvec_q8_act(&w_gate, x_norm, xq, &mut scratch.exp_gate[..effn])?;
            backend.matvec_q8_act(&w_up, x_norm, xq, &mut scratch.exp_up[..effn])?;
        } else {
            backend.matvec(&w_gate, x_norm, &mut scratch.exp_gate[..effn])?;
            backend.matvec(&w_up, x_norm, &mut scratch.exp_up[..effn])?;
        }

        backend.swiglu_inplace(&mut scratch.exp_gate[..effn], &scratch.exp_up[..effn]);

        if can_exp_q8k && is_k_quant(w_down.dtype) {
            let nb = effn / QK_K;
            quantize_row_q8_k(&scratch.exp_gate[..effn], &mut scratch.exp_q8k[..nb]);
            backend.matvec_q8k_act(
                &w_down,
                &scratch.exp_gate[..effn],
                &scratch.exp_q8k[..nb],
                &mut scratch.exp_out,
            )?;
        } else if can_exp_q8 && is_q8_compat(w_down.dtype) {
            let nb = effn / QK8_0;
            quantize_row_q8_0(&scratch.exp_gate[..effn], &mut scratch.exp_q8[..nb]);
            backend.matvec_q8_act(
                &w_down,
                &scratch.exp_gate[..effn],
                &scratch.exp_q8[..nb],
                &mut scratch.exp_out,
            )?;
        } else {
            backend.matvec(&w_down, &scratch.exp_gate[..effn], &mut scratch.exp_out)?;
        }

        for (o, &e) in out.iter_mut().zip(scratch.exp_out.iter()) {
            *o += weight * e;
        }
    }

    // 4. Optional always-active Shared Expert (`DeepSeek2`, `Qwen2MoE`)
    if let (Some(sg), Some(su), Some(sd)) = (&layer.shexp_gate, &layer.shexp_up, &layer.shexp_down)
    {
        let (sffn, _) = sg.dims_2d()?;
        if scratch.sh_gate.len() < sffn {
            scratch.sh_gate.resize(sffn, 0.0);
            scratch.sh_up.resize(sffn, 0.0);
            scratch.sh_q8k.resize((sffn / QK_K).max(1), BlockQ8_K::default());
            scratch
                .sh_q8
                .resize((sffn / QK8_0).max(1), BlockQ8_0 { d: 0, qs: [0; 32] });
        }

        if let (Some(pg), Some(pu), Some(xq)) = (&layer.p_shexp_gate, &layer.p_shexp_up, x_q8k) {
            backend.fused_gate_up_swiglu_packed_q8k(pg, pu, xq, &mut scratch.sh_gate[..sffn])?;
        } else {
            if let (Some(xq), true) = (x_q8k, is_k_quant(sg.dtype) && is_k_quant(su.dtype)) {
                backend.matvec_q8k_act(sg, x_norm, xq, &mut scratch.sh_gate[..sffn])?;
                backend.matvec_q8k_act(su, x_norm, xq, &mut scratch.sh_up[..sffn])?;
            } else {
                backend.matvec(sg, x_norm, &mut scratch.sh_gate[..sffn])?;
                backend.matvec(su, x_norm, &mut scratch.sh_up[..sffn])?;
            }
            backend.swiglu_inplace(&mut scratch.sh_gate[..sffn], &scratch.sh_up[..sffn]);
        }

        let can_sh_q8k = sffn % QK_K == 0;
        let can_sh_q8 = sffn % QK8_0 == 0;
        if let (Some(pd), true) = (&layer.p_shexp_down, can_sh_q8k) {
            let nb = sffn / QK_K;
            quantize_row_q8_k(&scratch.sh_gate[..sffn], &mut scratch.sh_q8k[..nb]);
            backend.matvec_packed_q8k(pd, &scratch.sh_q8k[..nb], &mut scratch.exp_out)?;
        } else if can_sh_q8k && is_k_quant(sd.dtype) {
            let nb = sffn / QK_K;
            quantize_row_q8_k(&scratch.sh_gate[..sffn], &mut scratch.sh_q8k[..nb]);
            backend.matvec_q8k_act(
                sd,
                &scratch.sh_gate[..sffn],
                &scratch.sh_q8k[..nb],
                &mut scratch.exp_out,
            )?;
        } else if can_sh_q8 && is_q8_compat(sd.dtype) {
            let nb = sffn / QK8_0;
            quantize_row_q8_0(&scratch.sh_gate[..sffn], &mut scratch.sh_q8[..nb]);
            backend.matvec_q8_act(
                sd,
                &scratch.sh_gate[..sffn],
                &scratch.sh_q8[..nb],
                &mut scratch.exp_out,
            )?;
        } else {
            backend.matvec(sd, &scratch.sh_gate[..sffn], &mut scratch.exp_out)?;
        }

        // Qwen2MoE applies an optional sigmoid gate to the shared expert output
        let sh_scale = if let Some(gate_w) = &layer.shexp_gate_inp {
            let dot = backend.dot_f32(gate_w, x_norm);
            1.0 / (1.0 + (-dot).exp())
        } else {
            1.0
        };

        for (o, &s) in out.iter_mut().zip(scratch.exp_out.iter()) {
            *o += sh_scale * s;
        }
    }

    Ok(())
}
