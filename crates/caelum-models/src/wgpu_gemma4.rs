//! 100% VRAM-resident, single-submission `wgpu` Vulkan/Metal/DX12 GPU runner for **Gemma 4**
//! (`gemma4`: hybrid SWA/Full heterogeneous heads, `k_eq_v`, QK-norm + scaleless V-norm,
//! proportional RoPE, GeGLU, `layer_output_scale`, and `final_logit_softcapping`).

use crate::gemma4::Gemma4Config;
use crate::rope::RopeTable;
use caelum_core::{DType, Result, TensorView};
use caelum_gguf::GgufFile;
use caelum_kernels::{wgpu, GpuContext, UniformArena};
use std::sync::Arc;

const GPU_PREFILL_CHUNK: usize = 64;

struct GpuGemma4Layer {
    is_swa: bool,
    head_dim: usize,
    num_kv_heads: usize,
    kv_u32_offset: u32,
    layer_output_scale: f32,

    _attn_norm: wgpu::Buffer,
    _post_attn_norm: wgpu::Buffer,
    _ffn_norm: wgpu::Buffer,
    _post_ffw_norm: wgpu::Buffer,
    _q_norm: wgpu::Buffer,
    _k_norm: wgpu::Buffer,
    _wq: wgpu::Buffer,
    _wk: wgpu::Buffer,
    _wv: Option<wgpu::Buffer>,
    _wo: wgpu::Buffer,
    _w_gate: wgpu::Buffer,
    _w_up: wgpu::Buffer,
    _w_down: wgpu::Buffer,

    wq_dt: DType,
    wk_dt: DType,
    wv_dt: Option<DType>,
    wo_dt: DType,
    w_gate_dt: DType,
    w_up_dt: DType,
    w_down_dt: DType,

    bg_attn_norm: wgpu::BindGroup,
    bg_wq: wgpu::BindGroup,
    bg_wk: wgpu::BindGroup,
    bg_wv: Option<wgpu::BindGroup>,
    bg_q_norm: wgpu::BindGroup,
    bg_k_norm: wgpu::BindGroup,
    bg_wo: wgpu::BindGroup,
    bg_post_attn_norm: wgpu::BindGroup,
    bg_res_ffn_norm: wgpu::BindGroup,
    bg_w_gate: wgpu::BindGroup,
    bg_w_up: wgpu::BindGroup,
    bg_w_down: wgpu::BindGroup,
    bg_post_ffw_norm: wgpu::BindGroup,
}

pub struct WgpuGemma4Model<'a> {
    pub config: Gemma4Config,
    gpu: Arc<GpuContext>,
    token_embd: TensorView<'a>,
    layers: Vec<GpuGemma4Layer>,
    _output_norm: wgpu::Buffer,
    _output_weight: wgpu::Buffer,
    output_dt: DType,
    _ones_v_norm: wgpu::Buffer,
    _cos_sin_full: wgpu::Buffer,
    _cos_sin_swa: wgpu::Buffer,
    half_rope_full: usize,
    half_rope_swa: usize,

    // Heterogeneous Packed F16 VRAM KV Cache
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    kv_cache_bytes: usize,
    vram_weight_bytes: usize,

    // Scratch buffers
    x_buf: wgpu::Buffer,
    _dx_buf: wgpu::Buffer,
    x_last_buf: wgpu::Buffer,
    _xb_buf: wgpu::Buffer,
    _q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    _attn_out_buf: wgpu::Buffer,
    _hb_buf: wgpu::Buffer,
    _hb2_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    logits_staging: wgpu::Buffer,

    // Shared BindGroups
    bg_v_norm: wgpu::BindGroup,
    bg_rope_kv_full: wgpu::BindGroup,
    bg_rope_kv_swa: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_geglu: wgpu::BindGroup,
    bg_res_add_scale: wgpu::BindGroup,
    bg_out_norm_x: wgpu::BindGroup,
    bg_out_norm_last: wgpu::BindGroup,
    bg_lm_head: wgpu::BindGroup,

    arena: UniformArena,
    embed_scratch: Vec<f32>,
    logits_cpu: Vec<f32>,
}

impl<'a> WgpuGemma4Model<'a> {
    pub fn from_gguf(
        gguf: &'a GgufFile,
        gpu: Arc<GpuContext>,
        max_seq_override: Option<usize>,
    ) -> Result<Self> {
        let config = Gemma4Config::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm_vec = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight_view = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };
        let output_dt = output_weight_view.dtype;
        let _ = gpu.matmul_pipeline(output_dt, 1)?;

        let h = config.hidden_size;
        let ffn = config.intermediate_size;
        let n_heads = config.num_heads;
        let max_hd = config.head_dim_full.max(config.head_dim_swa);
        let max_q_dim = n_heads * max_hd;
        let max_kv_dim = config
            .head_count_kv
            .iter()
            .zip(config.is_swa.iter())
            .map(|(&kh, &swa)| kh * if swa { config.head_dim_swa } else { config.head_dim_full })
            .max()
            .unwrap_or(max_q_dim);
        let max_b = GPU_PREFILL_CHUNK;

        let arena = UniformArena::new(&gpu.device);

        // Compute per-layer Packed F16 KV offsets (heterogeneous kv_dim!)
        let mut layer_kv_offsets = Vec::with_capacity(config.num_layers);
        let mut total_kv_u32 = 0usize;
        for l in 0..config.num_layers {
            layer_kv_offsets.push(total_kv_u32 as u32);
            let is_swa = config.is_swa.get(l).copied().unwrap_or(true);
            let hd = if is_swa { config.head_dim_swa } else { config.head_dim_full };
            let n_kv = config.head_count_kv.get(l).copied().unwrap_or(1);
            total_kv_u32 += config.max_seq_len * ((n_kv * hd) / 2);
        }
        let kv_buf_bytes = (total_kv_u32 * 4) as u64;
        let k_cache = gpu.alloc_storage("g4_k_cache_f16", kv_buf_bytes);
        let v_cache = gpu.alloc_storage("g4_v_cache_f16", kv_buf_bytes);
        let kv_cache_bytes = (kv_buf_bytes as usize) * 2;

        // Activation scratch buffers
        let x_buf = gpu.alloc_storage("g4_x", (max_b * h * 4) as u64);
        let dx_buf = gpu.alloc_storage("g4_dx", (max_b * h * 4) as u64);
        let x_last_buf = gpu.alloc_storage("g4_x_last", (h * 4) as u64);
        let xb_buf = gpu.alloc_storage("g4_xb", (max_b * h * 4) as u64);
        let q_buf = gpu.alloc_storage("g4_q", (max_b * max_q_dim * 4) as u64);
        let k_buf = gpu.alloc_storage("g4_k", (max_b * max_kv_dim * 4) as u64);
        let v_buf = gpu.alloc_storage("g4_v", (max_b * max_kv_dim * 4) as u64);
        let attn_out_buf = gpu.alloc_storage("g4_attn_out", (max_b * max_q_dim * 4) as u64);
        let hb_buf = gpu.alloc_storage("g4_hb", (max_b * ffn * 4) as u64);
        let hb2_buf = gpu.alloc_storage("g4_hb2", (max_b * ffn * 4) as u64);
        let logits_buf = gpu.alloc_storage("g4_logits", (config.vocab_size * 4) as u64);
        let logits_staging = gpu.alloc_staging_read("g4_logits_stage", (config.vocab_size * 4) as u64);

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
        let half_rope_full = rope_full.half_dim();
        let half_rope_swa = rope_swa.half_dim();
        let cos_sin_full = gpu.upload_f32("g4_cs_full", &rope_full.packed_cos_sin());
        let cos_sin_swa = gpu.upload_f32("g4_cs_swa", &rope_swa.packed_cos_sin());
        let ones_v_norm = gpu.upload_f32("g4_ones_vnorm", &vec![1.0f32; max_hd]);

        let mut vram_weight_bytes = 0usize;
        let output_norm = gpu.upload_f32("g4_out_norm", &output_norm_vec);
        vram_weight_bytes += output_norm_vec.len() * 4;
        let (output_weight, out_b) = gpu.upload_weight("g4_output", &output_weight_view);
        vram_weight_bytes += out_b;
        gpu.flush_uploads();

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let is_swa = config.is_swa.get(l).copied().unwrap_or(true);
            let head_dim = if is_swa { config.head_dim_swa } else { config.head_dim_full };
            let num_kv_heads = config.head_count_kv.get(l).copied().unwrap_or(1);

            let an_v = gguf.tensor(&format!("blk.{l}.attn_norm.weight"))?.to_f32_vec()?;
            let pan_v = gguf.tensor(&format!("blk.{l}.post_attention_norm.weight"))?.to_f32_vec()?;
            let fn_v = gguf.tensor(&format!("blk.{l}.ffn_norm.weight"))?.to_f32_vec()?;
            let pfn_v = gguf.tensor(&format!("blk.{l}.post_ffw_norm.weight"))?.to_f32_vec()?;
            let qn_v = gguf.tensor(&format!("blk.{l}.attn_q_norm.weight"))?.to_f32_vec()?;
            let kn_v = gguf.tensor(&format!("blk.{l}.attn_k_norm.weight"))?.to_f32_vec()?;
            let layer_output_scale = if gguf.has_tensor(&format!("blk.{l}.layer_output_scale.weight")) {
                gguf.tensor(&format!("blk.{l}.layer_output_scale.weight"))?
                    .to_f32_vec()?
                    .first()
                    .copied()
                    .unwrap_or(1.0)
            } else {
                1.0
            };

            let attn_norm = gpu.upload_f32(&format!("g4_l{l}_an"), &an_v);
            let post_attn_norm = gpu.upload_f32(&format!("g4_l{l}_pan"), &pan_v);
            let ffn_norm = gpu.upload_f32(&format!("g4_l{l}_fn"), &fn_v);
            let post_ffw_norm = gpu.upload_f32(&format!("g4_l{l}_pfn"), &pfn_v);
            let q_norm = gpu.upload_f32(&format!("g4_l{l}_qn"), &qn_v);
            let k_norm = gpu.upload_f32(&format!("g4_l{l}_kn"), &kn_v);
            vram_weight_bytes += (an_v.len() + pan_v.len() + fn_v.len() + pfn_v.len() + qn_v.len() + kn_v.len()) * 4;

            let wq_v = gguf.tensor(&format!("blk.{l}.attn_q.weight"))?;
            let wk_v = gguf.tensor(&format!("blk.{l}.attn_k.weight"))?;
            let wv_opt = if gguf.has_tensor(&format!("blk.{l}.attn_v.weight")) {
                Some(gguf.tensor(&format!("blk.{l}.attn_v.weight"))?)
            } else {
                None
            };
            let wo_v = gguf.tensor(&format!("blk.{l}.attn_output.weight"))?;
            let wg_v = gguf.tensor(&format!("blk.{l}.ffn_gate.weight"))?;
            let wu_v = gguf.tensor(&format!("blk.{l}.ffn_up.weight"))?;
            let wd_v = gguf.tensor(&format!("blk.{l}.ffn_down.weight"))?;

            for dt in [wq_v.dtype, wk_v.dtype, wo_v.dtype, wg_v.dtype, wu_v.dtype, wd_v.dtype] {
                let _ = gpu.matmul_pipeline(dt, 1)?;
            }
            if let Some(ref wv) = wv_opt {
                let _ = gpu.matmul_pipeline(wv.dtype, 1)?;
            }

            let (wq, b0) = gpu.upload_weight(&format!("g4_l{l}_wq"), &wq_v);
            let (wk, b1) = gpu.upload_weight(&format!("g4_l{l}_wk"), &wk_v);
            let (wv, wv_dt, bg_wv, b2) = if let Some(ref wv_view) = wv_opt {
                let (buf, bytes) = gpu.upload_weight(&format!("g4_l{l}_wv"), wv_view);
                let bg = gpu.bind_matmul(&buf, &xb_buf, &v_buf, &arena);
                (Some(buf), Some(wv_view.dtype), Some(bg), bytes)
            } else {
                (None, None, None, 0)
            };
            let (wo, b3) = gpu.upload_weight(&format!("g4_l{l}_wo"), &wo_v);
            let (w_gate, b4) = gpu.upload_weight(&format!("g4_l{l}_fg"), &wg_v);
            let (w_up, b5) = gpu.upload_weight(&format!("g4_l{l}_fu"), &wu_v);
            let (w_down, b6) = gpu.upload_weight(&format!("g4_l{l}_fd"), &wd_v);
            vram_weight_bytes += b0 + b1 + b2 + b3 + b4 + b5 + b6;

            let bg_attn_norm = gpu.bind_rms_norm(&x_buf, &attn_norm, &xb_buf, &arena);
            let bg_wq = gpu.bind_matmul(&wq, &xb_buf, &q_buf, &arena);
            let bg_wk = gpu.bind_matmul(&wk, &xb_buf, &k_buf, &arena);
            let bg_q_norm = gpu.bind_rw_r_u(&q_buf, &q_norm, &arena);
            let bg_k_norm = gpu.bind_rw_r_u(&k_buf, &k_norm, &arena);
            let bg_wo = gpu.bind_matmul(&wo, &attn_out_buf, &dx_buf, &arena);
            let bg_post_attn_norm = gpu.bind_rw_r_u(&dx_buf, &post_attn_norm, &arena);
            let bg_res_ffn_norm = gpu.bind_res_rms(&x_buf, &dx_buf, &ffn_norm, &xb_buf, &arena);
            let bg_w_gate = gpu.bind_matmul(&w_gate, &xb_buf, &hb_buf, &arena);
            let bg_w_up = gpu.bind_matmul(&w_up, &xb_buf, &hb2_buf, &arena);
            let bg_w_down = gpu.bind_matmul(&w_down, &hb_buf, &dx_buf, &arena);
            let bg_post_ffw_norm = gpu.bind_rw_r_u(&dx_buf, &post_ffw_norm, &arena);

            // Flush every 2 layers to keep DMA staging RAM tiny on 9.3 GB Q6_K models
            if l % 2 == 1 {
                gpu.flush_uploads();
            }

            layers.push(GpuGemma4Layer {
                is_swa,
                head_dim,
                num_kv_heads,
                kv_u32_offset: layer_kv_offsets[l],
                layer_output_scale,
                _attn_norm: attn_norm,
                _post_attn_norm: post_attn_norm,
                _ffn_norm: ffn_norm,
                _post_ffw_norm: post_ffw_norm,
                _q_norm: q_norm,
                _k_norm: k_norm,
                _wq: wq,
                _wk: wk,
                _wv: wv,
                _wo: wo,
                _w_gate: w_gate,
                _w_up: w_up,
                _w_down: w_down,
                wq_dt: wq_v.dtype,
                wk_dt: wk_v.dtype,
                wv_dt,
                wo_dt: wo_v.dtype,
                w_gate_dt: wg_v.dtype,
                w_up_dt: wu_v.dtype,
                w_down_dt: wd_v.dtype,
                bg_attn_norm,
                bg_wq,
                bg_wk,
                bg_wv,
                bg_q_norm,
                bg_k_norm,
                bg_wo,
                bg_post_attn_norm,
                bg_res_ffn_norm,
                bg_w_gate,
                bg_w_up,
                bg_w_down,
                bg_post_ffw_norm,
            });
        }
        gpu.flush_uploads();

        let bg_v_norm = gpu.bind_rw_r_u(&v_buf, &ones_v_norm, &arena);
        let bg_rope_kv_full = gpu.bind_rope_kv(&q_buf, &k_buf, &v_buf, &cos_sin_full, &k_cache, &v_cache, &arena);
        let bg_rope_kv_swa = gpu.bind_rope_kv(&q_buf, &k_buf, &v_buf, &cos_sin_swa, &k_cache, &v_cache, &arena);
        let bg_gqa = gpu.bind_gqa(&q_buf, &k_cache, &v_cache, &attn_out_buf, &arena);
        let bg_geglu = gpu.bind_rw_r_u(&hb_buf, &hb2_buf, &arena);
        let bg_res_add_scale = gpu.bind_rw_r_u(&x_buf, &dx_buf, &arena);
        let bg_out_norm_x = gpu.bind_rms_norm(&x_buf, &output_norm, &xb_buf, &arena);
        let bg_out_norm_last = gpu.bind_rms_norm(&x_last_buf, &output_norm, &xb_buf, &arena);
        let bg_lm_head = gpu.bind_matmul(&output_weight, &xb_buf, &logits_buf, &arena);

        let mut model = Self {
            embed_scratch: vec![0.0; max_b * h],
            logits_cpu: vec![0.0; config.vocab_size],
            config,
            gpu,
            token_embd,
            layers,
            _output_norm: output_norm,
            _output_weight: output_weight,
            output_dt,
            _ones_v_norm: ones_v_norm,
            _cos_sin_full: cos_sin_full,
            _cos_sin_swa: cos_sin_swa,
            half_rope_full,
            half_rope_swa,
            k_cache,
            v_cache,
            kv_cache_bytes,
            vram_weight_bytes,
            x_buf,
            _dx_buf: dx_buf,
            x_last_buf,
            _xb_buf: xb_buf,
            _q_buf: q_buf,
            k_buf,
            v_buf,
            _attn_out_buf: attn_out_buf,
            _hb_buf: hb_buf,
            _hb2_buf: hb2_buf,
            logits_buf,
            logits_staging,
            bg_v_norm,
            bg_rope_kv_full,
            bg_rope_kv_swa,
            bg_gqa,
            bg_geglu,
            bg_res_add_scale,
            bg_out_norm_x,
            bg_out_norm_last,
            bg_lm_head,
            arena,
        };
        model.reset_cache();
        Ok(model)
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.kv_cache_bytes
    }

    pub fn vram_weight_bytes(&self) -> usize {
        self.vram_weight_bytes
    }

    pub fn reset_cache(&mut self) {
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("g4_clear_kv"),
        });
        enc.clear_buffer(&self.k_cache, 0, None);
        enc.clear_buffer(&self.v_cache, 0, None);
        self.gpu.queue.submit(Some(enc.finish()));
    }

    #[inline]
    fn dispatch_matmul<'p>(
        gpu: &'p GpuContext,
        arena: &mut UniformArena,
        pass: &mut wgpu::ComputePass<'p>,
        dtype: DType,
        bg: &'p wgpu::BindGroup,
        rows: usize,
        cols: usize,
        batch: usize,
    ) -> Result<()> {
        let (pipe, dx, y, dz) = gpu.matmul_dispatch(dtype, rows, batch)?;
        let off = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[off]);
        pass.dispatch_workgroups(dx, y, dz);
        Ok(())
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        let b = tokens.len();
        let h = self.config.hidden_size;
        let ffn = self.config.intermediate_size;
        let n_heads = self.config.num_heads;
        let eps_bits = self.config.rms_norm_eps.to_bits();
        let es = self.config.embed_scale;
        let one_bits = 1.0f32.to_bits();

        for (t, &tok) in tokens.iter().enumerate() {
            let row = &mut self.embed_scratch[t * h..(t + 1) * h];
            self.token_embd.dequantize_row(tok as usize, row)?;
            for v in row.iter_mut() {
                *v *= es;
            }
        }
        self.gpu.write_f32(&self.x_buf, &self.embed_scratch[..b * h]);

        self.arena.reset();
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("g4_forward"),
        });

        let p = &self.gpu.pipelines;

        for layer in &self.layers {
            let hd = layer.head_dim;
            let n_kv = layer.num_kv_heads;
            let q_dim = n_heads * hd;
            let kv_dim = n_kv * hd;
            let half_rope = if layer.is_swa {
                self.half_rope_swa
            } else {
                self.half_rope_full
            };
            let bg_rope_kv = if layer.is_swa {
                &self.bg_rope_kv_swa
            } else {
                &self.bg_rope_kv_full
            };
            let swa = if layer.is_swa {
                self.config.sliding_window as u32
            } else {
                0
            };

            // Pass part 1: Attn RMSNorm + Q, K, (optional V) projections
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                let off_an = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm);
                pass.set_bind_group(0, &layer.bg_attn_norm, &[off_an]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.wq_dt, &layer.bg_wq, q_dim, h, b)?;
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.wk_dt, &layer.bg_wk, kv_dim, h, b)?;
                if let (Some(wv_dt), Some(ref bg_wv)) = (layer.wv_dt, &layer.bg_wv) {
                    Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, wv_dt, bg_wv, kv_dim, h, b)?;
                }
            }

            // Gemma 4 k_eq_v: copy pre-norm K into V buffer on GPU
            if layer.bg_wv.is_none() {
                encoder.copy_buffer_to_buffer(&self.k_buf, 0, &self.v_buf, 0, (b * kv_dim * 4) as u64);
            }

            // Pass part 2: Q/K/V norms -> RoPE+KV -> Flash GQA (scale=1.0) -> wo -> post_attn_norm -> FFN -> post_ffw_norm -> scaled residual
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                let off_qn = self.arena.push_u32x4(hd as u32, (b * n_heads) as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm_inplace);
                pass.set_bind_group(0, &layer.bg_q_norm, &[off_qn]);
                pass.dispatch_workgroups((b * n_heads) as u32, 1, 1);

                let off_kn = self.arena.push_u32x4(hd as u32, (b * n_kv) as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm_inplace);
                pass.set_bind_group(0, &layer.bg_k_norm, &[off_kn]);
                pass.dispatch_workgroups((b * n_kv) as u32, 1, 1);

                let off_vn = self.arena.push_u32x4(hd as u32, (b * n_kv) as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm_inplace);
                pass.set_bind_group(0, &self.bg_v_norm, &[off_vn]);
                pass.dispatch_workgroups((b * n_kv) as u32, 1, 1);

                // NeoX RoPE + Packed F16 KV Store
                let off_rk = self.arena.push_u32x8(
                    n_heads as u32,
                    n_kv as u32,
                    hd as u32,
                    half_rope as u32,
                    start_pos as u32,
                    b as u32,
                    layer.kv_u32_offset,
                    1,
                );
                pass.set_pipeline(&p.rope_kv_store);
                pass.set_bind_group(0, bg_rope_kv, &[off_rk]);
                let rope_threads = (n_heads * half_rope).max(kv_dim / 2);
                pass.dispatch_workgroups(rope_threads.div_ceil(64) as u32, b as u32, 1);

                // Flash GQA Attention (Gemma 4 uses attention_scale = 1.0 + SWA on local layers)
                let off_gqa = self.arena.push_u32x8(
                    n_heads as u32,
                    n_kv as u32,
                    hd as u32,
                    start_pos as u32,
                    b as u32,
                    layer.kv_u32_offset,
                    one_bits,
                    swa,
                );
                pass.set_pipeline(&p.gqa_flash_attn);
                pass.set_bind_group(0, &self.bg_gqa, &[off_gqa]);
                pass.dispatch_workgroups(n_heads as u32, b as u32, 1);

                // Output proj + post_attention_norm in-place on dx_buf
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.wo_dt, &layer.bg_wo, h, q_dim, b)?;
                let off_pan = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm_inplace);
                pass.set_bind_group(0, &layer.bg_post_attn_norm, &[off_pan]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                // Fused x += dx; xb = rms_norm(x, ffn_norm)
                let off_fn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&p.residual_rms_norm);
                pass.set_bind_group(0, &layer.bg_res_ffn_norm, &[off_fn]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                // GeGLU FFN
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_gate_dt, &layer.bg_w_gate, ffn, h, b)?;
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_up_dt, &layer.bg_w_up, ffn, h, b)?;
                let off_ge = self.arena.push_u32x4((b * ffn) as u32, 0, 0, 0);
                pass.set_pipeline(&p.geglu);
                pass.set_bind_group(0, &self.bg_geglu, &[off_ge]);
                pass.dispatch_workgroups((b * ffn).div_ceil(64) as u32, 1, 1);

                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_down_dt, &layer.bg_w_down, h, ffn, b)?;
                let off_pfn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm_inplace);
                pass.set_bind_group(0, &layer.bg_post_ffw_norm, &[off_pfn]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                // x = (x + dx) * layer_output_scale
                let off_rs = self.arena.push_u32x4((b * h) as u32, layer.layer_output_scale.to_bits(), 0, 0);
                pass.set_pipeline(&p.residual_add_scale);
                pass.set_bind_group(0, &self.bg_res_add_scale, &[off_rs]);
                pass.dispatch_workgroups((b * h).div_ceil(64) as u32, 1, 1);
            }
        }

        if compute_last_logits {
            if b > 1 {
                encoder.copy_buffer_to_buffer(
                    &self.x_buf,
                    ((b - 1) * h * 4) as u64,
                    &self.x_last_buf,
                    0,
                    (h * 4) as u64,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("g4_lm_head"),
                    timestamp_writes: None,
                });
                let off_on = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
                pass.set_pipeline(&p.rms_norm);
                pass.set_bind_group(
                    0,
                    if b == 1 { &self.bg_out_norm_x } else { &self.bg_out_norm_last },
                    &[off_on],
                );
                pass.dispatch_workgroups(1, 1, 1);
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, self.output_dt, &self.bg_lm_head, self.config.vocab_size, h, 1)?;
            }
            encoder.copy_buffer_to_buffer(
                &self.logits_buf,
                0,
                &self.logits_staging,
                0,
                (self.config.vocab_size * 4) as u64,
            );
        }

        self.arena.flush(&self.gpu.queue);
        self.gpu.queue.submit(Some(encoder.finish()));

        if compute_last_logits {
            self.gpu.read_staging_f32(&self.logits_staging, &mut self.logits_cpu)?;
            let cap = self.config.final_logit_softcap;
            if cap > 0.0 {
                let inv = 1.0 / cap;
                for v in &mut self.logits_cpu {
                    *v = cap * (*v * inv).tanh();
                }
            }
            Ok(&self.logits_cpu)
        } else {
            Ok(&[])
        }
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32]> {
        self.forward_chunk(&[token_id], pos, compute_logits)
    }

    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Ok(&[]);
        }
        let mut off = 0usize;
        while off < tokens.len() {
            let end = (off + GPU_PREFILL_CHUNK).min(tokens.len());
            let is_last = end == tokens.len();
            self.forward_chunk(&tokens[off..end], start_pos + off, is_last && compute_last_logits)?;
            off = end;
        }
        if compute_last_logits {
            Ok(&self.logits_cpu)
        } else {
            Ok(&[])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvPrecision;
    use crate::gemma4::Gemma4Model;
    use caelum_gguf::{GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;

    fn synth_vec(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 31 + seed * 17) % 19) as f32 - 9.0) * 0.02)
            .collect()
    }

    #[test]
    fn test_wgpu_gemma4_hybrid_swa_and_k_eq_v_parity_vs_cpu() {
        let Ok(gpu) = GpuContext::new() else {
            return;
        };
        let gpu = Arc::new(gpu);

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
        let cpu_backend = CpuBackend::new(Some(1));
        let mut cpu = Gemma4Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();
        let mut gpu_m = WgpuGemma4Model::from_gguf(&gguf, gpu, Some(16)).unwrap();

        let prompt = [1u32, 3, 5];
        let c_pre = cpu.forward_batch(&cpu_backend, &prompt, 0, true).unwrap().to_vec();
        let g_pre = gpu_m.forward_batch(&prompt, 0, true).unwrap().to_vec();
        for (a, b) in c_pre.iter().zip(g_pre.iter()) {
            assert!((a - b).abs() < 3e-3, "gemma4 prefill GPU vs CPU mismatch: {a} vs {b}");
        }
        let c_dec = cpu.forward_token(&cpu_backend, 7, 3, true).unwrap().to_vec();
        let g_dec = gpu_m.forward_token(7, 3, true).unwrap().to_vec();
        for (a, b) in c_dec.iter().zip(g_dec.iter()) {
            assert!((a - b).abs() < 3e-3, "gemma4 decode GPU vs CPU mismatch: {a} vs {b}");
        }
    }
}
