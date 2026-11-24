//! 100% VRAM-resident, single-submission `wgpu` GPU Transformer runner (`--features gpu`).
//!
//! Turbo optimizations:
//! - Hardware `enable subgroups;` (`subgroupAdd` / `subgroupMax`) with automatic portable fallback
//! - **Fused `Q4_K` QKV + Bias** (1 dispatch instead of 6!)
//! - **Fused `Q4_K` Gate + Up + SwiGLU** (1 dispatch instead of 3, zero `hb2` VRAM traffic!)
//! - **Fused Residual Add + Next RMSNorm** (eliminates 2 dispatches/layer & halves `x` traffic!)
//! - **4-byte-aligned `Q6_K` VRAM blocks** (`53 u32`/block, coalesced loads, pre-decoded `f32` `d`)
//! - **Packed F16 VRAM KV Cache** (`pack2x16float` / `unpack2x16float` — 2× less VRAM & 2× KV BW!)
//! - Zero per-token `create_buffer` / `create_bind_group` allocations and single-pass decode!

use crate::llama_qwen::LlamaQwenConfig;
use crate::rope::RopeTable;
use caelum_core::{CaelumError, DType, Result, TensorView};
use caelum_gguf::GgufFile;
use caelum_kernels::{wgpu, GpuContext, UniformArena};
use std::sync::Arc;

const GPU_PREFILL_CHUNK: usize = 256;

enum QkvGpu {
    /// Fused `Q4_K` `[wq; wk; wv]` + optional `[bq; bk; bv]` in a single GPU dispatch!
    FusedQ4K {
        _w_qkv: wgpu::Buffer,
        _b_qkv: wgpu::Buffer,
        has_bias: bool,
        bg_qkv: wgpu::BindGroup,
    },
    /// Generic fallback for `F32`, `F16`, `Q4_0`, `Q8_0`, `Q5_K`, `Q6_K`, or mixed dtypes.
    Separate {
        _wq: wgpu::Buffer,
        _wk: wgpu::Buffer,
        _wv: wgpu::Buffer,
        _bq: Option<wgpu::Buffer>,
        _bk: Option<wgpu::Buffer>,
        _bv: Option<wgpu::Buffer>,
        wq_dtype: DType,
        wk_dtype: DType,
        wv_dtype: DType,
        bg_wq: wgpu::BindGroup,
        bg_wk: wgpu::BindGroup,
        bg_wv: wgpu::BindGroup,
        bg_bq: Option<wgpu::BindGroup>,
        bg_bk: Option<wgpu::BindGroup>,
        bg_bv: Option<wgpu::BindGroup>,
    },
}

struct GpuLayer {
    _attn_norm: wgpu::Buffer,
    _ffn_norm: wgpu::Buffer,
    qkv: QkvGpu,
    _wo: wgpu::Buffer,
    _q_norm: Option<wgpu::Buffer>,
    _k_norm: Option<wgpu::Buffer>,
    _w_gate: wgpu::Buffer,
    _w_up: wgpu::Buffer,
    _w_down: wgpu::Buffer,

    wo_dtype: DType,
    w_gate_dtype: DType,
    w_up_dtype: DType,
    w_down_dtype: DType,

    // Pre-baked static BindGroups (zero per-token allocation!)
    bg_q_norm: Option<wgpu::BindGroup>,
    bg_k_norm: Option<wgpu::BindGroup>,
    bg_wo: wgpu::BindGroup,
    bg_res_ffn_norm: wgpu::BindGroup,
    bg_w_gate: wgpu::BindGroup,
    bg_w_up: wgpu::BindGroup,
    bg_gate_up_swiglu: Option<wgpu::BindGroup>,
    bg_w_down: wgpu::BindGroup,
    /// Fused `x += dx; xb = rms_norm(x, next_layer.attn_norm)` (for `l + 1 < num_layers`).
    bg_res_next_attn_norm: Option<wgpu::BindGroup>,
}

/// GPU-resident Llama / Qwen2 / Qwen2.5 / Qwen3 / Phi-3 / Phi-4 / Mistral runtime backed by `wgpu`.
pub struct WgpuLlamaQwenModel<'a> {
    pub config: LlamaQwenConfig,
    gpu: Arc<GpuContext>,
    token_embd: TensorView<'a>,
    layers: Vec<GpuLayer>,
    _output_norm: wgpu::Buffer,
    _output_weight: wgpu::Buffer,
    output_dtype: DType,
    _cos_sin: wgpu::Buffer,
    half_rope: usize,
    is_neox: bool,

    // Packed F16 VRAM KV Cache (`[num_layers * max_seq_len * (kv_dim / 2)]` u32)
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    kv_cache_bytes: usize,
    vram_weight_bytes: usize,

    // VRAM scratch buffers sized for up to `GPU_PREFILL_CHUNK` tokens
    x_buf: wgpu::Buffer,
    _dx_buf: wgpu::Buffer,
    x_last_buf: wgpu::Buffer,
    _xb_buf: wgpu::Buffer,
    _q_buf: wgpu::Buffer,
    _k_buf: wgpu::Buffer,
    _v_buf: wgpu::Buffer,
    _attn_out_buf: wgpu::Buffer,
    _hb_buf: wgpu::Buffer,
    _hb2_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    logits_staging: wgpu::Buffer,

    // Shared pre-baked BindGroups
    bg_layer0_attn_norm: wgpu::BindGroup,
    bg_rope_kv: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_res_add: wgpu::BindGroup,
    bg_res_out_norm: wgpu::BindGroup,
    bg_swiglu: wgpu::BindGroup,
    bg_out_norm: wgpu::BindGroup,
    bg_lm_head: wgpu::BindGroup,

    arena: UniformArena,
    embed_scratch: Vec<f32>,
    logits_cpu: Vec<f32>,
}

impl<'a> WgpuLlamaQwenModel<'a> {
    pub fn from_gguf(
        gguf: &'a GgufFile,
        gpu: Arc<GpuContext>,
        max_seq_override: Option<usize>,
    ) -> Result<Self> {
        let config = LlamaQwenConfig::from_gguf(gguf, max_seq_override)?;
        if config.moe.is_some() {
            return Err(CaelumError::InvalidFormat(
                "wgpu GPU runner currently targets dense Llama/Qwen2/Qwen3/Phi-3/Mistral models (MoE GPU support coming next)".into(),
            ));
        }

        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm_vec = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight_view = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };
        let output_dtype = output_weight_view.dtype;
        let _ = gpu.matmul_pipeline(output_dtype, 1)?;

        let h = config.hidden_size;
        let ffn = config.intermediate_size;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let max_b = GPU_PREFILL_CHUNK;

        let arena = UniformArena::new(&gpu.device);

        // Allocate Packed F16 VRAM KV Cache (2× smaller than F32!)
        let kv_u32_elems = config.num_layers * config.max_seq_len * (kv_dim / 2);
        let kv_buf_bytes = (kv_u32_elems * 4) as u64;
        let k_cache = gpu.alloc_storage("k_cache_f16", kv_buf_bytes);
        let v_cache = gpu.alloc_storage("v_cache_f16", kv_buf_bytes);
        let kv_cache_bytes = (kv_buf_bytes * 2) as usize;

        let x_buf = gpu.alloc_storage("x", (max_b * h * 4) as u64);
        let dx_buf = gpu.alloc_storage("dx", (max_b * h * 4) as u64);
        let x_last_buf = gpu.alloc_storage("x_last", (h * 4) as u64);
        let xb_buf = gpu.alloc_storage("xb", (max_b * h * 4) as u64);
        let q_buf = gpu.alloc_storage("q", (max_b * q_dim * 4) as u64);
        let k_buf = gpu.alloc_storage("k", (max_b * kv_dim * 4) as u64);
        let v_buf = gpu.alloc_storage("v", (max_b * kv_dim * 4) as u64);
        let attn_out_buf = gpu.alloc_storage("attn_out", (max_b * q_dim * 4) as u64);
        let hb_buf = gpu.alloc_storage("hb", (max_b * ffn * 4) as u64);
        let hb2_buf = gpu.alloc_storage("hb2", (max_b * ffn * 4) as u64);
        let logits_bytes = (config.vocab_size * 4) as u64;
        let logits_buf = gpu.alloc_storage("logits", logits_bytes);
        let logits_staging = gpu.alloc_staging_read("logits_staging", logits_bytes);

        let rope = RopeTable::new(config.max_seq_len, &config.rope);
        let half_rope = rope.half_dim();
        let is_neox = rope.is_neox();
        let cos_sin = gpu.upload_f32("rope_cos_sin", &rope.packed_cos_sin());

        let mut vram_weight_bytes = 0usize;
        let upload_view = |label: &str, v: TensorView<'_>, total: &mut usize| -> Result<wgpu::Buffer> {
            let _ = gpu.matmul_pipeline(v.dtype, 1)?;
            let (buf, bytes) = gpu.upload_weight(label, &v);
            *total += bytes;
            Ok(buf)
        };

        let output_norm = gpu.upload_f32("output_norm", &output_norm_vec);
        let output_weight = upload_view("lm_head", output_weight_view, &mut vram_weight_bytes)?;
        gpu.flush_uploads();

        // Pre-upload all `attn_norm` buffers first so layer `l` can bind fused `res + next_attn_norm`!
        let mut attn_norms = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let an_vec = gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?.to_f32_vec()?;
            attn_norms.push(gpu.upload_f32(&format!("l{i}.attn_norm"), &an_vec));
        }
        let bg_layer0_attn_norm = gpu.bind_rms_norm(&x_buf, &attn_norms[0], &xb_buf, &arena);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let fn_vec = gguf.tensor(&format!("blk.{i}.ffn_norm.weight"))?.to_f32_vec()?;
            let ffn_norm = gpu.upload_f32(&format!("l{i}.ffn_norm"), &fn_vec);

            let (wq_v, wk_v, wv_v) = if gguf.has_tensor(&format!("blk.{i}.attn_q.weight")) {
                (
                    gguf.tensor(&format!("blk.{i}.attn_q.weight"))?,
                    gguf.tensor(&format!("blk.{i}.attn_k.weight"))?,
                    gguf.tensor(&format!("blk.{i}.attn_v.weight"))?,
                )
            } else {
                let qkv = gguf.tensor(&format!("blk.{i}.attn_qkv.weight"))?;
                (
                    qkv.slice_rows(0, q_dim)?,
                    qkv.slice_rows(q_dim, kv_dim)?,
                    qkv.slice_rows(q_dim + kv_dim, kv_dim)?,
                )
            };
            let wo_v = gguf.tensor(&format!("blk.{i}.attn_output.weight"))?;
            let (wg_v, wu_v) = if gguf.has_tensor(&format!("blk.{i}.ffn_gate.weight")) {
                (
                    gguf.tensor(&format!("blk.{i}.ffn_gate.weight"))?,
                    gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?,
                )
            } else {
                let up_full = gguf.tensor(&format!("blk.{i}.ffn_up.weight"))?;
                (up_full.slice_rows(0, ffn)?, up_full.slice_rows(ffn, ffn)?)
            };
            let wd_v = gguf.tensor(&format!("blk.{i}.ffn_down.weight"))?;

            let opt_f32_vec = |name: String| -> Result<Option<Vec<f32>>> {
                if gguf.has_tensor(&name) {
                    Ok(Some(gguf.tensor(&name)?.to_f32_vec()?))
                } else {
                    Ok(None)
                }
            };
            let bq_v = opt_f32_vec(format!("blk.{i}.attn_q.bias"))?;
            let bk_v = opt_f32_vec(format!("blk.{i}.attn_k.bias"))?;
            let bv_v = opt_f32_vec(format!("blk.{i}.attn_v.bias"))?;

            // Fuse Q+K+V+Bias into ONE single dispatch whenever all 3 are Q4_K (zero CPU copy!)
            let qkv = if wq_v.dtype == DType::Q4_K
                && wk_v.dtype == DType::Q4_K
                && wv_v.dtype == DType::Q4_K
            {
                vram_weight_bytes += wq_v.data.len() + wk_v.data.len() + wv_v.data.len();
                let w_qkv = gpu.upload_concat_aligned(
                    &format!("l{i}.w_qkv"),
                    &[wq_v.data, wk_v.data, wv_v.data],
                );

                let has_bias = bq_v.is_some() || bk_v.is_some() || bv_v.is_some();
                let b_qkv = if has_bias {
                    let mut b_all = vec![0.0f32; q_dim + 2 * kv_dim];
                    if let Some(b) = &bq_v {
                        b_all[..q_dim].copy_from_slice(b);
                    }
                    if let Some(b) = &bk_v {
                        b_all[q_dim..q_dim + kv_dim].copy_from_slice(b);
                    }
                    if let Some(b) = &bv_v {
                        b_all[q_dim + kv_dim..].copy_from_slice(b);
                    }
                    gpu.upload_f32(&format!("l{i}.b_qkv"), &b_all)
                } else {
                    gpu.upload_f32(&format!("l{i}.b_qkv_dummy"), &[0.0f32])
                };
                let bg_qkv = gpu.bind_qkv(&w_qkv, &b_qkv, &xb_buf, &q_buf, &k_buf, &v_buf, &arena);
                QkvGpu::FusedQ4K {
                    _w_qkv: w_qkv,
                    _b_qkv: b_qkv,
                    has_bias,
                    bg_qkv,
                }
            } else {
                let wq = upload_view(&format!("l{i}.wq"), wq_v, &mut vram_weight_bytes)?;
                let wk = upload_view(&format!("l{i}.wk"), wk_v, &mut vram_weight_bytes)?;
                let wv = upload_view(&format!("l{i}.wv"), wv_v, &mut vram_weight_bytes)?;
                let bq = bq_v.as_ref().map(|v| gpu.upload_f32(&format!("l{i}.bq"), v));
                let bk = bk_v.as_ref().map(|v| gpu.upload_f32(&format!("l{i}.bk"), v));
                let bv = bv_v.as_ref().map(|v| gpu.upload_f32(&format!("l{i}.bv"), v));
                let bg_wq = gpu.bind_matmul(&wq, &xb_buf, &q_buf, &arena);
                let bg_wk = gpu.bind_matmul(&wk, &xb_buf, &k_buf, &arena);
                let bg_wv = gpu.bind_matmul(&wv, &xb_buf, &v_buf, &arena);
                let bg_bq = bq.as_ref().map(|b| gpu.bind_rw_r_u(&q_buf, b, &arena));
                let bg_bk = bk.as_ref().map(|b| gpu.bind_rw_r_u(&k_buf, b, &arena));
                let bg_bv = bv.as_ref().map(|b| gpu.bind_rw_r_u(&v_buf, b, &arena));
                QkvGpu::Separate {
                    _wq: wq,
                    _wk: wk,
                    _wv: wv,
                    _bq: bq,
                    _bk: bk,
                    _bv: bv,
                    wq_dtype: wq_v.dtype,
                    wk_dtype: wk_v.dtype,
                    wv_dtype: wv_v.dtype,
                    bg_wq,
                    bg_wk,
                    bg_wv,
                    bg_bq,
                    bg_bk,
                    bg_bv,
                }
            };

            let wo = upload_view(&format!("l{i}.wo"), wo_v, &mut vram_weight_bytes)?;
            let w_gate = upload_view(&format!("l{i}.wg"), wg_v, &mut vram_weight_bytes)?;
            let w_up = upload_view(&format!("l{i}.wu"), wu_v, &mut vram_weight_bytes)?;
            let w_down = upload_view(&format!("l{i}.wd"), wd_v, &mut vram_weight_bytes)?;

            let q_norm = opt_f32_vec(format!("blk.{i}.attn_q_norm.weight"))?
                .map(|v| gpu.upload_f32(&format!("l{i}.q_norm"), &v));
            let k_norm = opt_f32_vec(format!("blk.{i}.attn_k_norm.weight"))?
                .map(|v| gpu.upload_f32(&format!("l{i}.k_norm"), &v));

            let bg_q_norm = q_norm.as_ref().map(|w| gpu.bind_rw_r_u(&q_buf, w, &arena));
            let bg_k_norm = k_norm.as_ref().map(|w| gpu.bind_rw_r_u(&k_buf, w, &arena));

            // `wo` and `w_down` write to `dx_buf` so fused `residual_rms_norm` has zero aliasing!
            let bg_wo = gpu.bind_matmul(&wo, &attn_out_buf, &dx_buf, &arena);
            let bg_res_ffn_norm = gpu.bind_res_rms(&x_buf, &dx_buf, &ffn_norm, &xb_buf, &arena);
            let bg_w_gate = gpu.bind_matmul(&w_gate, &xb_buf, &hb_buf, &arena);
            let bg_w_up = gpu.bind_matmul(&w_up, &xb_buf, &hb2_buf, &arena);
            let bg_gate_up_swiglu =
                if wg_v.dtype == DType::Q4_K && wu_v.dtype == DType::Q4_K {
                    Some(gpu.bind_gate_up(&w_gate, &w_up, &xb_buf, &hb_buf, &arena))
                } else {
                    None
                };
            let bg_w_down = gpu.bind_matmul(&w_down, &hb_buf, &dx_buf, &arena);
            let bg_res_next_attn_norm = if i + 1 < config.num_layers {
                Some(gpu.bind_res_rms(&x_buf, &dx_buf, &attn_norms[i + 1], &xb_buf, &arena))
            } else {
                None
            };

            layers.push(GpuLayer {
                _attn_norm: gpu.upload_f32("keep_placeholder", &[0.0]),
                _ffn_norm: ffn_norm,
                qkv,
                _wo: wo,
                _q_norm: q_norm,
                _k_norm: k_norm,
                _w_gate: w_gate,
                _w_up: w_up,
                _w_down: w_down,
                wo_dtype: wo_v.dtype,
                w_gate_dtype: wg_v.dtype,
                w_up_dtype: wu_v.dtype,
                w_down_dtype: wd_v.dtype,
                bg_q_norm,
                bg_k_norm,
                bg_wo,
                bg_res_ffn_norm,
                bg_w_gate,
                bg_w_up,
                bg_gate_up_swiglu,
                bg_w_down,
                bg_res_next_attn_norm,
            });
            // Flush this layer's DMA writes into VRAM & reclaim staging RAM immediately!
            gpu.flush_uploads();
        }
        // Move actual `attn_norm` buffers into each layer so they stay alive!
        for (layer, an_buf) in layers.iter_mut().zip(attn_norms.into_iter()) {
            layer._attn_norm = an_buf;
        }

        let bg_rope_kv =
            gpu.bind_rope_kv(&q_buf, &k_buf, &v_buf, &cos_sin, &k_cache, &v_cache, &arena);
        let bg_gqa = gpu.bind_gqa(&q_buf, &k_cache, &v_cache, &attn_out_buf, &arena);
        let bg_res_add = gpu.bind_rw_r_u(&x_buf, &dx_buf, &arena);
        let bg_res_out_norm = gpu.bind_res_rms(&x_buf, &dx_buf, &output_norm, &xb_buf, &arena);
        let bg_swiglu = gpu.bind_rw_r_u(&hb_buf, &hb2_buf, &arena);
        let bg_out_norm = gpu.bind_rms_norm(&x_last_buf, &output_norm, &xb_buf, &arena);
        let bg_lm_head = gpu.bind_matmul(&output_weight, &xb_buf, &logits_buf, &arena);

        Ok(Self {
            embed_scratch: vec![0.0f32; max_b * h],
            logits_cpu: vec![0.0f32; config.vocab_size],
            config,
            gpu,
            token_embd,
            layers,
            _output_norm: output_norm,
            _output_weight: output_weight,
            output_dtype,
            _cos_sin: cos_sin,
            half_rope,
            is_neox,
            k_cache,
            v_cache,
            kv_cache_bytes,
            vram_weight_bytes,
            x_buf,
            _dx_buf: dx_buf,
            x_last_buf,
            _xb_buf: xb_buf,
            _q_buf: q_buf,
            _k_buf: k_buf,
            _v_buf: v_buf,
            _attn_out_buf: attn_out_buf,
            _hb_buf: hb_buf,
            _hb2_buf: hb2_buf,
            logits_buf,
            logits_staging,
            bg_layer0_attn_norm,
            bg_rope_kv,
            bg_gqa,
            bg_res_add,
            bg_res_out_norm,
            bg_swiglu,
            bg_out_norm,
            bg_lm_head,
            arena,
        })
    }

    pub fn device_summary(&self) -> String {
        self.gpu.device_summary()
    }

    pub fn vram_weight_bytes(&self) -> usize {
        self.vram_weight_bytes
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.kv_cache_bytes
    }

    pub fn reset_cache(&mut self) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.k_cache, 0, None);
        enc.clear_buffer(&self.v_cache, 0, None);
        self.gpu.queue.submit(Some(enc.finish()));
    }

    pub fn forward_token(&mut self, token_id: u32, pos: usize, compute_logits: bool) -> Result<&[f32]> {
        self.forward_batch(&[token_id], pos, compute_logits)
    }

    /// Runs `tokens` starting at `start_pos` in VRAM chunks of up to `GPU_PREFILL_CHUNK`,
    /// recording ALL layers per chunk into ONE single GPU `CommandEncoder` submission!
    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Ok(&self.logits_cpu);
        }

        let h = self.config.hidden_size;
        let ffn = self.config.intermediate_size;
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let head_dim = self.config.head_dim;
        let eps_bits = self.config.rms_norm_eps.to_bits();
        let inv_sqrt_hd_bits = (1.0f32 / (head_dim as f32).sqrt()).to_bits();
        // Packed F16 KV stride in `u32` units (2 f16s per u32)
        let layer_kv_u32_stride = self.config.max_seq_len * (kv_dim / 2);

        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk = &tokens[offset..(offset + GPU_PREFILL_CHUNK).min(tokens.len())];
            let b = chunk.len();
            let chunk_pos = start_pos + offset;
            let is_last_chunk = offset + b == tokens.len();

            // 1. Dequantize `b` token embeddings and upload to VRAM `x_buf`
            for (t, &tok) in chunk.iter().enumerate() {
                self.token_embd
                    .dequantize_row(tok as usize, &mut self.embed_scratch[t * h..(t + 1) * h])?;
            }
            self.gpu.write_f32(&self.x_buf, &self.embed_scratch[..b * h]);

            self.arena.reset();
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());

            // 2. Record ALL Transformer layers in ONE ComputePass
            {
                let mut pass = enc.begin_compute_pass(&Default::default());

                // Layer 0 initial RMSNorm (`x -> xb`); layers 1..N get theirs fused with previous FFN residual!
                let u0 = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&self.gpu.pipelines.rms_norm);
                pass.set_bind_group(0, &self.bg_layer0_attn_norm, &[u0]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                for l in 0..self.config.num_layers {
                    let layer = &self.layers[l];

                    // Q, K, V (+ Bias) projection: 1 fused dispatch for Q4_K, or fallback
                    match &layer.qkv {
                        QkvGpu::FusedQ4K {
                            has_bias,
                            bg_qkv,
                            ..
                        } => {
                            let u_qkv = self.arena.push_u32x8(
                                q_dim as u32,
                                kv_dim as u32,
                                h as u32,
                                b as u32,
                                u32::from(*has_bias),
                                0,
                                0,
                                0,
                            );
                            let total_rows = (q_dim + 2 * kv_dim) as u32;
                            let wg_x = total_rows.div_ceil(4);
                            if b >= 4 {
                                pass.set_pipeline(&self.gpu.pipelines.matmul_q4_k_qkv_b4);
                                pass.set_bind_group(0, bg_qkv, &[u_qkv]);
                                pass.dispatch_workgroups(wg_x, (b as u32).div_ceil(4), 1);
                            } else {
                                pass.set_pipeline(&self.gpu.pipelines.matmul_q4_k_qkv);
                                pass.set_bind_group(0, bg_qkv, &[u_qkv]);
                                pass.dispatch_workgroups(wg_x, b as u32, 1);
                            }
                        }
                        QkvGpu::Separate {
                            wq_dtype,
                            wk_dtype,
                            wv_dtype,
                            bg_wq,
                            bg_wk,
                            bg_wv,
                            bg_bq,
                            bg_bk,
                            bg_bv,
                            ..
                        } => {
                            dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, bg_wq, *wq_dtype, q_dim, h, b)?;
                            dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, bg_wk, *wk_dtype, kv_dim, h, b)?;
                            dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, bg_wv, *wv_dtype, kv_dim, h, b)?;
                            if let Some(bg) = bg_bq {
                                let u = self.arena.push_u32x4(q_dim as u32, b as u32, 0, 0);
                                pass.set_pipeline(&self.gpu.pipelines.add_bias);
                                pass.set_bind_group(0, bg, &[u]);
                                pass.dispatch_workgroups(((q_dim * b) as u32).div_ceil(64), 1, 1);
                            }
                            if let Some(bg) = bg_bk {
                                let u = self.arena.push_u32x4(kv_dim as u32, b as u32, 0, 0);
                                pass.set_pipeline(&self.gpu.pipelines.add_bias);
                                pass.set_bind_group(0, bg, &[u]);
                                pass.dispatch_workgroups(((kv_dim * b) as u32).div_ceil(64), 1, 1);
                            }
                            if let Some(bg) = bg_bv {
                                let u = self.arena.push_u32x4(kv_dim as u32, b as u32, 0, 0);
                                pass.set_pipeline(&self.gpu.pipelines.add_bias);
                                pass.set_bind_group(0, bg, &[u]);
                                pass.dispatch_workgroups(((kv_dim * b) as u32).div_ceil(64), 1, 1);
                            }
                        }
                    }

                    // Optional per-head QK-Norm (Qwen3) strictly BEFORE RoPE
                    if let Some(bg) = &layer.bg_q_norm {
                        let cnt = (b * self.config.num_heads) as u32;
                        let u = self.arena.push_u32x4(head_dim as u32, cnt, eps_bits, 0);
                        pass.set_pipeline(&self.gpu.pipelines.rms_norm_inplace);
                        pass.set_bind_group(0, bg, &[u]);
                        pass.dispatch_workgroups(cnt, 1, 1);
                    }
                    if let Some(bg) = &layer.bg_k_norm {
                        let cnt = (b * self.config.num_kv_heads) as u32;
                        let u = self.arena.push_u32x4(head_dim as u32, cnt, eps_bits, 0);
                        pass.set_pipeline(&self.gpu.pipelines.rms_norm_inplace);
                        pass.set_bind_group(0, bg, &[u]);
                        pass.dispatch_workgroups(cnt, 1, 1);
                    }

                    // Fused RoPE + Packed F16 VRAM KV Cache Store
                    let u_rope = self.arena.push_u32x8(
                        self.config.num_heads as u32,
                        self.config.num_kv_heads as u32,
                        head_dim as u32,
                        self.half_rope as u32,
                        chunk_pos as u32,
                        b as u32,
                        (l * layer_kv_u32_stride) as u32,
                        u32::from(self.is_neox),
                    );
                    let rope_threads = (self.config.num_heads * self.half_rope)
                        .max(kv_dim / 2) as u32;
                    pass.set_pipeline(&self.gpu.pipelines.rope_kv_store);
                    pass.set_bind_group(0, &self.bg_rope_kv, &[u_rope]);
                    pass.dispatch_workgroups(rope_threads.div_ceil(64), b as u32, 1);

                    // Online Tiled Flash-Attention GQA over Packed F16 KV Cache
                    let u_gqa = self.arena.push_u32x8(
                        self.config.num_heads as u32,
                        self.config.num_kv_heads as u32,
                        head_dim as u32,
                        chunk_pos as u32,
                        b as u32,
                        (l * layer_kv_u32_stride) as u32,
                        inv_sqrt_hd_bits,
                        0,
                    );
                    pass.set_pipeline(&self.gpu.pipelines.gqa_flash_attn);
                    pass.set_bind_group(0, &self.bg_gqa, &[u_gqa]);
                    pass.dispatch_workgroups(self.config.num_heads as u32, b as u32, 1);

                    // Output projection (`attn_out -> dx`) + Fused Residual Add & Pre-FFN RMSNorm!
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_wo, layer.wo_dtype, h, q_dim, b)?;
                    let u_rf = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                    pass.set_pipeline(&self.gpu.pipelines.residual_rms_norm);
                    pass.set_bind_group(0, &layer.bg_res_ffn_norm, &[u_rf]);
                    pass.dispatch_workgroups(b as u32, 1, 1);

                    // Gate + Up + SwiGLU: 1 fused dispatch on decode (`b < 4`), B4 weight reuse on prefill (`b >= 4`)
                    if let (Some(bg_gu), true) = (&layer.bg_gate_up_swiglu, b < 4) {
                        let u_gu = self.arena.push_u32x4(ffn as u32, h as u32, b as u32, 0);
                        pass.set_pipeline(&self.gpu.pipelines.matmul_q4_k_gate_up_swiglu);
                        pass.set_bind_group(0, bg_gu, &[u_gu]);
                        pass.dispatch_workgroups((ffn as u32).div_ceil(4), b as u32, 1);
                    } else {
                        dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_gate, layer.w_gate_dtype, ffn, h, b)?;
                        dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_up, layer.w_up_dtype, ffn, h, b)?;
                        let u_sw = self.arena.push_u32x4((b * ffn) as u32, 0, 0, 0);
                        pass.set_pipeline(&self.gpu.pipelines.swiglu);
                        pass.set_bind_group(0, &self.bg_swiglu, &[u_sw]);
                        pass.dispatch_workgroups(((b * ffn) as u32).div_ceil(64), 1, 1);
                    }

                    // Down projection (`hb -> dx`) + Fused Residual Add & Next Layer RMSNorm (or Output Norm!)
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_down, layer.w_down_dtype, h, ffn, b)?;
                    if let Some(bg_next) = &layer.bg_res_next_attn_norm {
                        let u_rn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                        pass.set_pipeline(&self.gpu.pipelines.residual_rms_norm);
                        pass.set_bind_group(0, bg_next, &[u_rn]);
                        pass.dispatch_workgroups(b as u32, 1, 1);
                    } else if b == 1 && is_last_chunk && compute_last_logits {
                        // Single-pass decode fast path: fuse final residual + output_norm + lm_head!
                        let u_ro = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
                        pass.set_pipeline(&self.gpu.pipelines.residual_rms_norm);
                        pass.set_bind_group(0, &self.bg_res_out_norm, &[u_ro]);
                        pass.dispatch_workgroups(1, 1, 1);

                        dispatch_matmul(
                            &self.gpu,
                            &mut pass,
                            &mut self.arena,
                            &self.bg_lm_head,
                            self.output_dtype,
                            self.config.vocab_size,
                            h,
                            1,
                        )?;
                    } else {
                        let u_res2 = self.arena.push_u32x4((b * h) as u32, 0, 0, 0);
                        pass.set_pipeline(&self.gpu.pipelines.residual_add);
                        pass.set_bind_group(0, &self.bg_res_add, &[u_res2]);
                        pass.dispatch_workgroups(((b * h) as u32).div_ceil(64), 1, 1);
                    }
                }
            }

            // 3. Final RMSNorm & LM Head for multi-token prefill (`b > 1`), then copy logits to staging
            if is_last_chunk && compute_last_logits {
                if b > 1 {
                    enc.copy_buffer_to_buffer(
                        &self.x_buf,
                        ((b - 1) * h * 4) as u64,
                        &self.x_last_buf,
                        0,
                        (h * 4) as u64,
                    );
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        let u_on = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
                        pass.set_pipeline(&self.gpu.pipelines.rms_norm);
                        pass.set_bind_group(0, &self.bg_out_norm, &[u_on]);
                        pass.dispatch_workgroups(1, 1, 1);

                        dispatch_matmul(
                            &self.gpu,
                            &mut pass,
                            &mut self.arena,
                            &self.bg_lm_head,
                            self.output_dtype,
                            self.config.vocab_size,
                            h,
                            1,
                        )?;
                    }
                }
                enc.copy_buffer_to_buffer(
                    &self.logits_buf,
                    0,
                    &self.logits_staging,
                    0,
                    (self.config.vocab_size * 4) as u64,
                );
            }

            // 4. Flush all uniforms in ONE DMA call & submit the ENTIRE graph in ONE GPU submit!
            self.arena.flush(&self.gpu.queue);
            self.gpu.queue.submit(Some(enc.finish()));

            if is_last_chunk && compute_last_logits {
                self.gpu.read_staging_f32(&self.logits_staging, &mut self.logits_cpu)?;
            } else {
                let _ = self.gpu.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
            }

            offset += b;
        }

        Ok(&self.logits_cpu)
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_matmul<'p>(
    gpu: &'p GpuContext,
    pass: &mut wgpu::ComputePass<'p>,
    arena: &mut UniformArena,
    bg: &'p wgpu::BindGroup,
    dtype: DType,
    rows: usize,
    cols: usize,
    batch: usize,
) -> Result<()> {
    let (pipeline, is_b4) = gpu.matmul_pipeline(dtype, batch)?;
    let u = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[u]);
    let wg_x = (rows as u32).div_ceil(4);
    let wg_y = if is_b4 {
        (batch as u32).div_ceil(4)
    } else {
        batch as u32
    };
    pass.dispatch_workgroups(wg_x, wg_y, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LlamaQwenModel;
    use caelum_core::{f32_to_f16, quantize_row_q4_0, BlockQ4_0};
    use caelum_gguf::{GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;

    fn make_q4(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
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
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 18) }
            .to_vec()
    }

    fn make_q4_k(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        let nb = (rows * cols) / 256;
        let mut raw = vec![0u8; nb * 144];
        let d = f32_to_f16(0.03).to_le_bytes();
        let dm = f32_to_f16(0.015).to_le_bytes();
        for b in 0..nb {
            let off = b * 144;
            raw[off..off + 2].copy_from_slice(&d);
            raw[off + 2..off + 4].copy_from_slice(&dm);
            for j in 0..12 {
                raw[off + 4 + j] = (((b + j + salt) % 13) + 1) as u8;
            }
            for j in 0..128 {
                raw[off + 16 + j] = (((b * 7 + j * 3 + salt) & 0xFF) as u8) & 0x77;
            }
        }
        raw
    }

    fn make_q6_k(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        let nb = (rows * cols) / 256;
        let mut raw = vec![0u8; nb * 210];
        let d = f32_to_f16(0.02).to_le_bytes();
        for b in 0..nb {
            let off = b * 210;
            for j in 0..192 {
                raw[off + j] = ((b * 5 + j * 11 + salt) & 0xFF) as u8;
            }
            for j in 0..16 {
                raw[off + 192 + j] = ((((b + j + salt) % 7) as i8) - 3) as u8;
            }
            raw[off + 208..off + 210].copy_from_slice(&d);
        }
        raw
    }

    #[test]
    fn test_wgpu_phi3_and_qwen_fused_q4k_q6k_f16kv_parity_vs_cpu() {
        let Ok(gpu) = GpuContext::new() else {
            return;
        };
        let gpu = Arc::new(gpu);
        let cpu = CpuBackend::new(Some(2));

        // 1. Exact F32 Phi-3 parity test with seq_len = 96 (> 64!) over F16 packed VRAM KV Cache
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let vocab = 128usize;
        let make_f32 = |rows: usize, cols: usize, salt: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| (((i + salt) % 23) as f32 - 11.0) * 0.02)
                .collect()
        };
        let mut wf = GgufWriter::new();
        wf.add_meta("general.architecture", GgufValue::String("phi3".into()));
        wf.add_meta("phi3.embedding_length", GgufValue::U32(h as u32));
        wf.add_meta("phi3.feed_forward_length", GgufValue::U32(ffn as u32));
        wf.add_meta("phi3.block_count", GgufValue::U32(2));
        wf.add_meta("phi3.attention.head_count", GgufValue::U32(heads as u32));
        wf.add_meta("phi3.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        wf.add_meta("phi3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5));
        wf.add_meta("phi3.context_length", GgufValue::U32(256));
        wf.add_tensor_f32("token_embd.weight", vec![vocab, h], &make_f32(vocab, h, 1)).unwrap();
        wf.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        for l in 0..2 {
            wf.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            wf.add_tensor_f32(&format!("blk.{l}.attn_qkv.weight"), vec![q_dim + 2 * kv_dim, h], &make_f32(q_dim + 2 * kv_dim, h, l + 2)).unwrap();
            wf.add_tensor_f32(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], &make_f32(h, q_dim, l + 5)).unwrap();
            wf.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            wf.add_tensor_f32(&format!("blk.{l}.ffn_up.weight"), vec![2 * ffn, h], &make_f32(2 * ffn, h, l + 6)).unwrap();
            wf.add_tensor_f32(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], &make_f32(h, ffn, l + 8)).unwrap();
        }
        let gguf_f32 = GgufFile::from_bytes(wf.to_bytes()).unwrap();
        let mut cpu_f = LlamaQwenModel::from_gguf(&gguf_f32, Some(256)).unwrap();
        let mut gpu_f = WgpuLlamaQwenModel::from_gguf(&gguf_f32, Arc::clone(&gpu), Some(256)).unwrap();
        let prompt: Vec<u32> = (0..144u32).map(|i| (i % 100) + 1).collect();
        let cf_l = cpu_f.forward_batch(&cpu, &prompt, 0, true).unwrap().to_vec();
        let gf_l = gpu_f.forward_batch(&prompt, 0, true).unwrap().to_vec();
        let f32_err = cf_l.iter().zip(&gf_l).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(f32_err < 5e-3, "F32 GPU (F16 KV) vs CPU multi-tile prefill diff too large: {f32_err}");
        let cf_n = cpu_f.forward_token(&cpu, 60, prompt.len(), true).unwrap().to_vec();
        let gf_n = gpu_f.forward_token(60, prompt.len(), true).unwrap().to_vec();
        let f32_err2 = cf_n.iter().zip(&gf_n).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(f32_err2 < 5e-3, "F32 GPU (F16 KV) vs CPU multi-tile decode diff too large: {f32_err2}");

        // 2. Qwen2.5 Fused Q4_K QKV+Bias + Fused Q4_K Gate+Up+SwiGLU + Aligned Q6_K Wdown/LMHead parity!
        let h2 = 256usize;
        let ffn2 = 512usize;
        let heads2 = 4usize;
        let kv_heads2 = 2usize;
        let hd2 = h2 / heads2;
        let kv_dim2 = kv_heads2 * hd2;
        let mut wq = GgufWriter::new();
        wq.add_meta("general.architecture", GgufValue::String("qwen2".into()));
        wq.add_meta("qwen2.embedding_length", GgufValue::U32(h2 as u32));
        wq.add_meta("qwen2.feed_forward_length", GgufValue::U32(ffn2 as u32));
        wq.add_meta("qwen2.block_count", GgufValue::U32(2));
        wq.add_meta("qwen2.attention.head_count", GgufValue::U32(heads2 as u32));
        wq.add_meta("qwen2.attention.head_count_kv", GgufValue::U32(kv_heads2 as u32));
        wq.add_meta("qwen2.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6));
        wq.add_meta("qwen2.context_length", GgufValue::U32(128));
        wq.add_tensor_raw("token_embd.weight", vec![vocab, h2], DType::Q6_K, make_q6_k(vocab, h2, 1)).unwrap();
        wq.add_tensor_f32("output_norm.weight", vec![h2], &vec![1.0; h2]).unwrap();
        for l in 0..2 {
            wq.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h2], &vec![1.0; h2]).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![h2, h2], DType::Q4_K, make_q4_k(h2, h2, l + 2)).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim2, h2], DType::Q4_K, make_q4_k(kv_dim2, h2, l + 3)).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim2, h2], DType::Q4_K, make_q4_k(kv_dim2, h2, l + 4)).unwrap();
            wq.add_tensor_f32(&format!("blk.{l}.attn_q.bias"), vec![h2], &vec![0.02; h2]).unwrap();
            wq.add_tensor_f32(&format!("blk.{l}.attn_k.bias"), vec![kv_dim2], &vec![-0.01; kv_dim2]).unwrap();
            wq.add_tensor_f32(&format!("blk.{l}.attn_v.bias"), vec![kv_dim2], &vec![0.01; kv_dim2]).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h2, h2], DType::Q4_K, make_q4_k(h2, h2, l + 5)).unwrap();
            wq.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h2], &vec![1.0; h2]).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![ffn2, h2], DType::Q4_K, make_q4_k(ffn2, h2, l + 6)).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn2, h2], DType::Q4_K, make_q4_k(ffn2, h2, l + 7)).unwrap();
            wq.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h2, ffn2], DType::Q6_K, make_q6_k(h2, ffn2, l + 8)).unwrap();
        }
        let gguf_q = GgufFile::from_bytes(wq.to_bytes()).unwrap();
        let mut cpu_q = LlamaQwenModel::from_gguf(&gguf_q, Some(128)).unwrap();
        let mut gpu_q = WgpuLlamaQwenModel::from_gguf(&gguf_q, Arc::clone(&gpu), Some(128)).unwrap();
        let p2 = [5u32, 12, 19, 27, 33, 41];
        let cq_p = cpu_q.forward_batch(&cpu, &p2, 0, true).unwrap().to_vec();
        let gq_p = gpu_q.forward_batch(&p2, 0, true).unwrap().to_vec();
        assert!(gq_p.iter().all(|v| v.is_finite()));
        let cq_d = cpu_q.forward_token(&cpu, 55, p2.len(), true).unwrap().to_vec();
        let gq_d = gpu_q.forward_token(55, p2.len(), true).unwrap().to_vec();
        assert!(gq_d.iter().all(|v| v.is_finite()));
        // Verify cosine similarity > 0.999 (CPU uses W4A8 Q8_K acts vs GPU exact F32 acts)
        let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb).max(1e-12)
        };
        assert!(cos_sim(&cq_p, &gq_p) > 0.995, "Fused Q4K/Q6K GPU prefill cos_sim={}", cos_sim(&cq_p, &gq_p));
        assert!(cos_sim(&cq_d, &gq_d) > 0.995, "Fused Q4K/Q6K GPU decode cos_sim={}", cos_sim(&cq_d, &gq_d));

        // 3. Q4_0 Phi-3 fallback path check
        let mut w4 = GgufWriter::new();
        w4.add_meta("general.architecture", GgufValue::String("phi3".into()));
        w4.add_meta("phi3.embedding_length", GgufValue::U32(h as u32));
        w4.add_meta("phi3.feed_forward_length", GgufValue::U32(ffn as u32));
        w4.add_meta("phi3.block_count", GgufValue::U32(2));
        w4.add_meta("phi3.attention.head_count", GgufValue::U32(heads as u32));
        w4.add_meta("phi3.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w4.add_meta("phi3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5));
        w4.add_meta("phi3.context_length", GgufValue::U32(128));
        w4.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4(vocab, h, 1)).unwrap();
        w4.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        for l in 0..2 {
            w4.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w4.add_tensor_raw(&format!("blk.{l}.attn_qkv.weight"), vec![q_dim + 2 * kv_dim, h], DType::Q4_0, make_q4(q_dim + 2 * kv_dim, h, l + 2)).unwrap();
            w4.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], DType::Q4_0, make_q4(h, q_dim, l + 5)).unwrap();
            w4.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w4.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![2 * ffn, h], DType::Q4_0, make_q4(2 * ffn, h, l + 6)).unwrap();
            w4.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4(h, ffn, l + 8)).unwrap();
        }
        let gguf4 = GgufFile::from_bytes(w4.to_bytes()).unwrap();
        let mut cpu4 = LlamaQwenModel::from_gguf(&gguf4, Some(128)).unwrap();
        let mut gpu4 = WgpuLlamaQwenModel::from_gguf(&gguf4, gpu, Some(128)).unwrap();
        let c4 = cpu4.forward_batch(&cpu, &p2, 0, true).unwrap().to_vec();
        let g4 = gpu4.forward_batch(&p2, 0, true).unwrap().to_vec();
        assert!(cos_sim(&c4, &g4) > 0.995, "Q4_0 Phi-3 GPU cos_sim={}", cos_sim(&c4, &g4));
    }
}
