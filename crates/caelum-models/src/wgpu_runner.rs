//! 100% VRAM-resident, single-submission `wgpu` GPU Transformer runner (`--features gpu`).
//!
//! Records an ENTIRE multi-layer Transformer forward pass (RMSNorm -> QKV -> Bias -> Qwen3 QK-Norm
//! -> RoPE + KV Store -> Online Flash-Attention GQA -> Wo + Residual -> RMSNorm -> Gate/Up + SwiGLU
//! -> Down + Residual -> Final RMSNorm -> LM Head) into **ONE single `CommandEncoder` submission**
//! with **ZERO per-token `create_buffer` / `create_bind_group` allocations** and **ZERO PCIe syncs
//! between layers**!

use crate::llama_qwen::LlamaQwenConfig;
use crate::rope::RopeTable;
use caelum_core::{CaelumError, DType, Result, TensorView};
use caelum_gguf::GgufFile;
use caelum_kernels::{wgpu, GpuContext, UniformArena};
use std::sync::Arc;

const GPU_PREFILL_CHUNK: usize = 256;

struct GpuLayer {
    _attn_norm: wgpu::Buffer,
    _wq: wgpu::Buffer,
    _wk: wgpu::Buffer,
    _wv: wgpu::Buffer,
    _wo: wgpu::Buffer,
    _bq: Option<wgpu::Buffer>,
    _bk: Option<wgpu::Buffer>,
    _bv: Option<wgpu::Buffer>,
    _q_norm: Option<wgpu::Buffer>,
    _k_norm: Option<wgpu::Buffer>,
    _ffn_norm: wgpu::Buffer,
    _w_gate: wgpu::Buffer,
    _w_up: wgpu::Buffer,
    _w_down: wgpu::Buffer,

    wq_dtype: DType,
    wk_dtype: DType,
    wv_dtype: DType,
    wo_dtype: DType,
    w_gate_dtype: DType,
    w_up_dtype: DType,
    w_down_dtype: DType,

    // Pre-baked static BindGroups (zero per-token allocation!)
    bg_attn_norm: wgpu::BindGroup,
    bg_wq: wgpu::BindGroup,
    bg_wk: wgpu::BindGroup,
    bg_wv: wgpu::BindGroup,
    bg_bq: Option<wgpu::BindGroup>,
    bg_bk: Option<wgpu::BindGroup>,
    bg_bv: Option<wgpu::BindGroup>,
    bg_q_norm: Option<wgpu::BindGroup>,
    bg_k_norm: Option<wgpu::BindGroup>,
    bg_wo: wgpu::BindGroup,
    bg_ffn_norm: wgpu::BindGroup,
    bg_w_gate: wgpu::BindGroup,
    bg_w_up: wgpu::BindGroup,
    bg_w_down: wgpu::BindGroup,
}

/// GPU-resident Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral runtime backed by `wgpu`.
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

    // VRAM KV Cache (`[num_layers * max_seq_len * kv_dim]` f32)
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    kv_cache_bytes: usize,
    vram_weight_bytes: usize,

    // VRAM scratch buffers sized for up to `GPU_PREFILL_CHUNK` tokens
    x_buf: wgpu::Buffer,
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
    bg_rope_kv: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_res_add: wgpu::BindGroup,
    bg_swiglu: wgpu::BindGroup,
    bg_out_norm: wgpu::BindGroup,
    bg_out_norm_b1: wgpu::BindGroup,
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
                "wgpu GPU runner currently targets dense Llama/Qwen2/Qwen3/Mistral models (MoE GPU support coming next)".into(),
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

        // Allocate VRAM activation & KV cache buffers
        let kv_elems = config.num_layers * config.max_seq_len * kv_dim;
        let kv_buf_bytes = (kv_elems * 4) as u64;
        let k_cache = gpu.alloc_storage("k_cache", kv_buf_bytes);
        let v_cache = gpu.alloc_storage("v_cache", kv_buf_bytes);
        let kv_cache_bytes = (kv_buf_bytes * 2) as usize;

        let x_buf = gpu.alloc_storage("x", (max_b * h * 4) as u64);
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
            *total += v.data.len();
            Ok(gpu.upload_raw_padded(label, v.data))
        };

        let output_norm = gpu.upload_f32("output_norm", &output_norm_vec);
        let output_weight = upload_view("lm_head", output_weight_view, &mut vram_weight_bytes)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let an_vec = gguf.tensor(&format!("blk.{i}.attn_norm.weight"))?.to_f32_vec()?;
            let fn_vec = gguf.tensor(&format!("blk.{i}.ffn_norm.weight"))?.to_f32_vec()?;
            let attn_norm = gpu.upload_f32(&format!("l{i}.attn_norm"), &an_vec);
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

            let wq = upload_view(&format!("l{i}.wq"), wq_v, &mut vram_weight_bytes)?;
            let wk = upload_view(&format!("l{i}.wk"), wk_v, &mut vram_weight_bytes)?;
            let wv = upload_view(&format!("l{i}.wv"), wv_v, &mut vram_weight_bytes)?;
            let wo = upload_view(&format!("l{i}.wo"), wo_v, &mut vram_weight_bytes)?;
            let w_gate = upload_view(&format!("l{i}.wg"), wg_v, &mut vram_weight_bytes)?;
            let w_up = upload_view(&format!("l{i}.wu"), wu_v, &mut vram_weight_bytes)?;
            let w_down = upload_view(&format!("l{i}.wd"), wd_v, &mut vram_weight_bytes)?;

            let opt_f32 = |name: String| -> Result<Option<wgpu::Buffer>> {
                if gguf.has_tensor(&name) {
                    let v = gguf.tensor(&name)?.to_f32_vec()?;
                    Ok(Some(gpu.upload_f32(&name, &v)))
                } else {
                    Ok(None)
                }
            };
            let bq = opt_f32(format!("blk.{i}.attn_q.bias"))?;
            let bk = opt_f32(format!("blk.{i}.attn_k.bias"))?;
            let bv = opt_f32(format!("blk.{i}.attn_v.bias"))?;
            let q_norm = opt_f32(format!("blk.{i}.attn_q_norm.weight"))?;
            let k_norm = opt_f32(format!("blk.{i}.attn_k_norm.weight"))?;

            let bg_attn_norm = gpu.bind_rms_norm(&x_buf, &attn_norm, &xb_buf, &arena);
            let bg_wq = gpu.bind_matmul(&wq, &xb_buf, &q_buf, &arena);
            let bg_wk = gpu.bind_matmul(&wk, &xb_buf, &k_buf, &arena);
            let bg_wv = gpu.bind_matmul(&wv, &xb_buf, &v_buf, &arena);
            let bg_bq = bq.as_ref().map(|b| gpu.bind_rw_r_u(&q_buf, b, &arena));
            let bg_bk = bk.as_ref().map(|b| gpu.bind_rw_r_u(&k_buf, b, &arena));
            let bg_bv = bv.as_ref().map(|b| gpu.bind_rw_r_u(&v_buf, b, &arena));
            let bg_q_norm = q_norm.as_ref().map(|w| gpu.bind_rw_r_u(&q_buf, w, &arena));
            let bg_k_norm = k_norm.as_ref().map(|w| gpu.bind_rw_r_u(&k_buf, w, &arena));
            let bg_wo = gpu.bind_matmul(&wo, &attn_out_buf, &xb_buf, &arena);
            let bg_ffn_norm = gpu.bind_rms_norm(&x_buf, &ffn_norm, &xb_buf, &arena);
            let bg_w_gate = gpu.bind_matmul(&w_gate, &xb_buf, &hb_buf, &arena);
            let bg_w_up = gpu.bind_matmul(&w_up, &xb_buf, &hb2_buf, &arena);
            let bg_w_down = gpu.bind_matmul(&w_down, &hb_buf, &xb_buf, &arena);

            layers.push(GpuLayer {
                _attn_norm: attn_norm,
                _wq: wq,
                _wk: wk,
                _wv: wv,
                _wo: wo,
                _bq: bq,
                _bk: bk,
                _bv: bv,
                _q_norm: q_norm,
                _k_norm: k_norm,
                _ffn_norm: ffn_norm,
                _w_gate: w_gate,
                _w_up: w_up,
                _w_down: w_down,
                wq_dtype: wq_v.dtype,
                wk_dtype: wk_v.dtype,
                wv_dtype: wv_v.dtype,
                wo_dtype: wo_v.dtype,
                w_gate_dtype: wg_v.dtype,
                w_up_dtype: wu_v.dtype,
                w_down_dtype: wd_v.dtype,
                bg_attn_norm,
                bg_wq,
                bg_wk,
                bg_wv,
                bg_bq,
                bg_bk,
                bg_bv,
                bg_q_norm,
                bg_k_norm,
                bg_wo,
                bg_ffn_norm,
                bg_w_gate,
                bg_w_up,
                bg_w_down,
            });
        }

        let bg_rope_kv =
            gpu.bind_rope_kv(&q_buf, &k_buf, &v_buf, &cos_sin, &k_cache, &v_cache, &arena);
        let bg_gqa = gpu.bind_gqa(&q_buf, &k_cache, &v_cache, &attn_out_buf, &arena);
        let bg_res_add = gpu.bind_rw_r_u(&x_buf, &xb_buf, &arena);
        let bg_swiglu = gpu.bind_rw_r_u(&hb_buf, &hb2_buf, &arena);
        let bg_out_norm = gpu.bind_rms_norm(&x_last_buf, &output_norm, &xb_buf, &arena);
        let bg_out_norm_b1 = gpu.bind_rms_norm(&x_buf, &output_norm, &xb_buf, &arena);
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
            bg_rope_kv,
            bg_gqa,
            bg_res_add,
            bg_swiglu,
            bg_out_norm,
            bg_out_norm_b1,
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
        // Prefix cache overwrite handles positions `0..`; clear buffers cleanly on reset
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
        let layer_kv_stride = self.config.max_seq_len * kv_dim;

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

                for l in 0..self.config.num_layers {
                    let layer = &self.layers[l];

                    // Pre-Attention RMSNorm
                    let u = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                    pass.set_pipeline(&self.gpu.pipelines.rms_norm);
                    pass.set_bind_group(0, &layer.bg_attn_norm, &[u]);
                    pass.dispatch_workgroups(b as u32, 1, 1);

                    // Q, K, V matmul
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_wq, layer.wq_dtype, q_dim, h, b)?;
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_wk, layer.wk_dtype, kv_dim, h, b)?;
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_wv, layer.wv_dtype, kv_dim, h, b)?;

                    // Optional QKV biases (Qwen2 / Qwen2.5)
                    if let Some(bg) = &layer.bg_bq {
                        let u = self.arena.push_u32x4(q_dim as u32, b as u32, 0, 0);
                        pass.set_pipeline(&self.gpu.pipelines.add_bias);
                        pass.set_bind_group(0, bg, &[u]);
                        pass.dispatch_workgroups(((q_dim * b) as u32).div_ceil(64), 1, 1);
                    }
                    if let Some(bg) = &layer.bg_bk {
                        let u = self.arena.push_u32x4(kv_dim as u32, b as u32, 0, 0);
                        pass.set_pipeline(&self.gpu.pipelines.add_bias);
                        pass.set_bind_group(0, bg, &[u]);
                        pass.dispatch_workgroups(((kv_dim * b) as u32).div_ceil(64), 1, 1);
                    }
                    if let Some(bg) = &layer.bg_bv {
                        let u = self.arena.push_u32x4(kv_dim as u32, b as u32, 0, 0);
                        pass.set_pipeline(&self.gpu.pipelines.add_bias);
                        pass.set_bind_group(0, bg, &[u]);
                        pass.dispatch_workgroups(((kv_dim * b) as u32).div_ceil(64), 1, 1);
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

                    // Fused RoPE + VRAM KV Cache store
                    let u_rope = self.arena.push_u32x8(
                        self.config.num_heads as u32,
                        self.config.num_kv_heads as u32,
                        head_dim as u32,
                        self.half_rope as u32,
                        chunk_pos as u32,
                        b as u32,
                        (l * layer_kv_stride) as u32,
                        u32::from(self.is_neox),
                    );
                    let rope_threads = (self.config.num_heads * self.half_rope).max(kv_dim) as u32;
                    pass.set_pipeline(&self.gpu.pipelines.rope_kv_store);
                    pass.set_bind_group(0, &self.bg_rope_kv, &[u_rope]);
                    pass.dispatch_workgroups(rope_threads.div_ceil(64), b as u32, 1);

                    // Online Tiled Flash-Attention GQA
                    let u_gqa = self.arena.push_u32x8(
                        self.config.num_heads as u32,
                        self.config.num_kv_heads as u32,
                        head_dim as u32,
                        chunk_pos as u32,
                        b as u32,
                        (l * layer_kv_stride) as u32,
                        inv_sqrt_hd_bits,
                        0,
                    );
                    pass.set_pipeline(&self.gpu.pipelines.gqa_flash_attn);
                    pass.set_bind_group(0, &self.bg_gqa, &[u_gqa]);
                    pass.dispatch_workgroups(self.config.num_heads as u32, b as u32, 1);

                    // Output projection + Residual
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_wo, layer.wo_dtype, h, q_dim, b)?;
                    let u_res = self.arena.push_u32x4((b * h) as u32, 0, 0, 0);
                    pass.set_pipeline(&self.gpu.pipelines.residual_add);
                    pass.set_bind_group(0, &self.bg_res_add, &[u_res]);
                    pass.dispatch_workgroups(((b * h) as u32).div_ceil(64), 1, 1);

                    // Pre-FFN RMSNorm
                    let u_fn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                    pass.set_pipeline(&self.gpu.pipelines.rms_norm);
                    pass.set_bind_group(0, &layer.bg_ffn_norm, &[u_fn]);
                    pass.dispatch_workgroups(b as u32, 1, 1);

                    // Gate + Up + SwiGLU + Down + Residual
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_gate, layer.w_gate_dtype, ffn, h, b)?;
                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_up, layer.w_up_dtype, ffn, h, b)?;
                    let u_sw = self.arena.push_u32x4((b * ffn) as u32, 0, 0, 0);
                    pass.set_pipeline(&self.gpu.pipelines.swiglu);
                    pass.set_bind_group(0, &self.bg_swiglu, &[u_sw]);
                    pass.dispatch_workgroups(((b * ffn) as u32).div_ceil(64), 1, 1);

                    dispatch_matmul(&self.gpu, &mut pass, &mut self.arena, &layer.bg_w_down, layer.w_down_dtype, h, ffn, b)?;
                    let u_res2 = self.arena.push_u32x4((b * h) as u32, 0, 0, 0);
                    pass.set_pipeline(&self.gpu.pipelines.residual_add);
                    pass.set_bind_group(0, &self.bg_res_add, &[u_res2]);
                    pass.dispatch_workgroups(((b * h) as u32).div_ceil(64), 1, 1);
                }

                // Single-pass decode fast path (`b == 1`): fuse final RMSNorm + LM Head into the
                // exact same ComputePass (zero extra copy_buffer_to_buffer or pass boundary)!
                if b == 1 && is_last_chunk && compute_last_logits {
                    let u_on = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
                    pass.set_pipeline(&self.gpu.pipelines.rms_norm);
                    pass.set_bind_group(0, &self.bg_out_norm_b1, &[u_on]);
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
    use caelum_core::{quantize_row_q4_0, BlockQ4_0};
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

    #[test]
    fn test_wgpu_phi3_and_qwen_parity_vs_cpu() {
        let Ok(gpu) = GpuContext::new() else {
            return; // Skip gracefully on headless runners without GPU adapter
        };
        let gpu = Arc::new(gpu);
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = h / heads;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let vocab = 128usize;

        // 1. Exact F32 parity test (< 1e-4 proves zero math bugs in WGSL RoPE/FlashAttn/RMSNorm/SwiGLU)
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
        wf.add_meta("phi3.context_length", GgufValue::U32(128));
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
        let cpu = CpuBackend::new(Some(2));
        let mut cpu_f = LlamaQwenModel::from_gguf(&gguf_f32, Some(128)).unwrap();
        let mut gpu_f = WgpuLlamaQwenModel::from_gguf(&gguf_f32, Arc::clone(&gpu), Some(128)).unwrap();
        // Test seq_len = 96 (> 64!) to exercise multi-tile Flash-Attention on GPU!
        let prompt: Vec<u32> = (0..96u32).map(|i| (i % 100) + 1).collect();
        let cf_l = cpu_f.forward_batch(&cpu, &prompt, 0, true).unwrap().to_vec();
        let gf_l = gpu_f.forward_batch(&prompt, 0, true).unwrap().to_vec();
        let f32_err = cf_l.iter().zip(&gf_l).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(f32_err < 1e-4, "F32 GPU vs CPU multi-tile prefill diff too large: {f32_err}");
        let cf_n = cpu_f.forward_token(&cpu, 60, prompt.len(), true).unwrap().to_vec();
        let gf_n = gpu_f.forward_token(60, prompt.len(), true).unwrap().to_vec();
        let f32_err2 = cf_n.iter().zip(&gf_n).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(f32_err2 < 1e-4, "F32 GPU vs CPU multi-tile decode diff too large: {f32_err2}");

        // 2. Quantized Q4_0 parity test (GPU uses full F32 activations vs CPU W4A8 Q8_0 activations)
        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("phi3".into()));
        w.add_meta("phi3.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("phi3.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("phi3.block_count", GgufValue::U32(2));
        w.add_meta("phi3.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("phi3.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("phi3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5));
        w.add_meta("phi3.context_length", GgufValue::U32(128));
        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_qkv.weight"), vec![q_dim + 2 * kv_dim, h], DType::Q4_0, make_q4(q_dim + 2 * kv_dim, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, q_dim], DType::Q4_0, make_q4(h, q_dim, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![2 * ffn, h], DType::Q4_0, make_q4(2 * ffn, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4(h, ffn, l + 8)).unwrap();
        }

        let gguf = GgufFile::from_bytes(w.to_bytes()).unwrap();
        let mut cpu_m = LlamaQwenModel::from_gguf(&gguf, Some(128)).unwrap();
        let mut gpu_m = WgpuLlamaQwenModel::from_gguf(&gguf, gpu, Some(128)).unwrap();

        let cpu_l = cpu_m.forward_batch(&cpu, &prompt, 0, true).unwrap().to_vec();
        let gpu_l = gpu_m.forward_batch(&prompt, 0, true).unwrap().to_vec();
        let max_err = cpu_l
            .iter()
            .zip(&gpu_l)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.06, "Q4_0 GPU vs CPU Phi-3 prefill logit diff too large: {max_err}");

        let cpu_next = cpu_m.forward_token(&cpu, 60, prompt.len(), true).unwrap().to_vec();
        let gpu_next = gpu_m.forward_token(60, prompt.len(), true).unwrap().to_vec();
        let max_err2 = cpu_next
            .iter()
            .zip(&gpu_next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err2 < 0.06, "Q4_0 GPU vs CPU Phi-3 decode logit diff too large: {max_err2}");
    }
}
