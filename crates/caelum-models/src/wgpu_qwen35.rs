//! 100% VRAM-resident, single-submission `wgpu` Vulkan/Metal/DX12 GPU runner for **Qwen 3.5**
//! (`qwen35`: 3:1 hybrid Gated DeltaNet + Gated Full Attention).

use crate::qwen35::Qwen35Config;
use crate::rope::RopeTable;
use caelum_core::{DType, Result, TensorView};
use caelum_gguf::GgufFile;
use caelum_kernels::{wgpu, GpuContext, UniformArena};
use std::sync::Arc;

const GPU_PREFILL_CHUNK: usize = 64;

struct GpuDeltaNetLayer {
    _wqkv: wgpu::Buffer,
    _wgate: wgpu::Buffer,
    _w_alpha: wgpu::Buffer,
    _w_beta: wgpu::Buffer,
    _w_out: wgpu::Buffer,
    _conv_w: wgpu::Buffer,
    _ssm_params: wgpu::Buffer,
    conv_state: wgpu::Buffer,
    ssm_state: wgpu::Buffer,

    wqkv_dt: DType,
    wgate_dt: DType,
    w_alpha_dt: DType,
    w_beta_dt: DType,
    w_out_dt: DType,

    bg_wqkv: wgpu::BindGroup,
    bg_wgate: wgpu::BindGroup,
    bg_alpha: wgpu::BindGroup,
    bg_beta: wgpu::BindGroup,
    bg_conv: wgpu::BindGroup,
    bg_step: wgpu::BindGroup,
    bg_out: wgpu::BindGroup,
}

struct GpuFullAttnLayer {
    kv_layer_idx: usize,
    _wq_gate: wgpu::Buffer,
    _wk: wgpu::Buffer,
    _wv: wgpu::Buffer,
    _wo: wgpu::Buffer,
    _q_norm: wgpu::Buffer,
    _k_norm: wgpu::Buffer,

    wq_gate_dt: DType,
    wk_dt: DType,
    wv_dt: DType,
    wo_dt: DType,

    bg_wq_gate: wgpu::BindGroup,
    bg_wk: wgpu::BindGroup,
    bg_wv: wgpu::BindGroup,
    bg_q_norm: wgpu::BindGroup,
    bg_k_norm: wgpu::BindGroup,
    bg_wo: wgpu::BindGroup,
}

enum GpuMixer {
    DeltaNet(Box<GpuDeltaNetLayer>),
    FullAttn(Box<GpuFullAttnLayer>),
}

struct GpuQwen35Layer {
    _attn_norm: wgpu::Buffer,
    _post_attn_norm: wgpu::Buffer,
    mixer: GpuMixer,
    _w_gate: wgpu::Buffer,
    _w_up: wgpu::Buffer,
    _w_down: wgpu::Buffer,
    w_gate_dt: DType,
    w_up_dt: DType,
    w_down_dt: DType,

    bg_res_post_norm: wgpu::BindGroup,
    bg_w_gate: wgpu::BindGroup,
    bg_w_up: wgpu::BindGroup,
    bg_gate_up_swiglu: Option<wgpu::BindGroup>,
    bg_w_down: wgpu::BindGroup,
    bg_res_next_attn_norm: Option<wgpu::BindGroup>,
}

pub struct WgpuQwen35Model<'a> {
    pub config: Qwen35Config,
    gpu: Arc<GpuContext>,
    token_embd: TensorView<'a>,
    layers: Vec<GpuQwen35Layer>,
    _output_norm: wgpu::Buffer,
    _output_weight: wgpu::Buffer,
    output_dt: DType,
    _cos_sin: wgpu::Buffer,
    half_rope: usize,

    // Packed F16 KV Cache (only for the full-attention layers!)
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    kv_cache_bytes: usize,
    vram_weight_bytes: usize,
    conv_state_bytes: u64,

    // Scratch buffers
    x_buf: wgpu::Buffer,
    _dx_buf: wgpu::Buffer,
    x_last_buf: wgpu::Buffer,
    _xb_buf: wgpu::Buffer,
    _qg_buf: wgpu::Buffer,
    _q_buf: wgpu::Buffer,
    _k_buf: wgpu::Buffer,
    _v_buf: wgpu::Buffer,
    _attn_gate_buf: wgpu::Buffer,
    _attn_out_buf: wgpu::Buffer,
    _qkv_ssm_buf: wgpu::Buffer,
    _conv_out_buf: wgpu::Buffer,
    _z_ssm_buf: wgpu::Buffer,
    _alpha_buf: wgpu::Buffer,
    _beta_buf: wgpu::Buffer,
    _ssm_y_buf: wgpu::Buffer,
    _hb_buf: wgpu::Buffer,
    _hb2_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    logits_staging: wgpu::Buffer,

    // Shared BindGroups
    bg_layer0_attn_norm: wgpu::BindGroup,
    bg_deinterleave_qg: wgpu::BindGroup,
    bg_rope_kv: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_mul_sigmoid: wgpu::BindGroup,
    bg_l2_norm_qk: wgpu::BindGroup,
    bg_swiglu: wgpu::BindGroup,
    bg_res_add: wgpu::BindGroup,
    bg_res_out_norm: wgpu::BindGroup,
    bg_out_norm: wgpu::BindGroup,
    bg_lm_head: wgpu::BindGroup,

    arena: UniformArena,
    embed_scratch: Vec<f32>,
    logits_cpu: Vec<f32>,
}

impl<'a> WgpuQwen35Model<'a> {
    pub fn from_gguf(
        gguf: &'a GgufFile,
        gpu: Arc<GpuContext>,
        max_seq_override: Option<usize>,
    ) -> Result<Self> {
        let config = Qwen35Config::from_gguf(gguf, max_seq_override)?;
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
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let num_k_heads = config.ssm_group_count;
        let head_k_dim = config.ssm_state_size;
        let num_v_heads = config.ssm_time_step_rank;
        let head_v_dim = config.ssm_inner_size / num_v_heads;
        let inner = config.ssm_inner_size;
        let conv_ch = 2 * num_k_heads * head_k_dim + inner;
        let conv_hist = config.ssm_conv_kernel.saturating_sub(1);
        let max_b = GPU_PREFILL_CHUNK;

        let arena = UniformArena::new(&gpu.device);

        // Count full-attention layers for compact Packed-F16 KV cache allocation
        let num_full_layers = (0..config.num_layers)
            .filter(|&l| config.is_full_attn(l) || gguf.has_tensor(&format!("blk.{l}.attn_q.weight")))
            .count()
            .max(1);
        let kv_u32_elems = num_full_layers * config.max_seq_len * (kv_dim / 2);
        let kv_buf_bytes = (kv_u32_elems * 4) as u64;
        let k_cache = gpu.alloc_storage("q35_k_cache_f16", kv_buf_bytes);
        let v_cache = gpu.alloc_storage("q35_v_cache_f16", kv_buf_bytes);

        let conv_state_bytes = (conv_hist * conv_ch * 4) as u64;
        let ssm_state_bytes = (num_v_heads * head_v_dim * head_v_dim * 4) as u64;
        let mut total_state_bytes = (kv_buf_bytes as usize) * 2;

        // Allocate activation scratch buffers in VRAM
        let x_buf = gpu.alloc_storage("q35_x", (max_b * h * 4) as u64);
        let dx_buf = gpu.alloc_storage("q35_dx", (max_b * h * 4) as u64);
        let x_last_buf = gpu.alloc_storage("q35_x_last", (h * 4) as u64);
        let xb_buf = gpu.alloc_storage("q35_xb", (max_b * h * 4) as u64);
        let qg_buf = gpu.alloc_storage("q35_qg", (max_b * q_dim * 2 * 4) as u64);
        let q_buf = gpu.alloc_storage("q35_q", (max_b * q_dim * 4) as u64);
        let k_buf = gpu.alloc_storage("q35_k", (max_b * kv_dim * 4) as u64);
        let v_buf = gpu.alloc_storage("q35_v", (max_b * kv_dim * 4) as u64);
        let attn_gate_buf = gpu.alloc_storage("q35_attn_gate", (max_b * q_dim * 4) as u64);
        let attn_out_buf = gpu.alloc_storage("q35_attn_out", (max_b * q_dim * 4) as u64);
        let qkv_ssm_buf = gpu.alloc_storage("q35_qkv_ssm", (max_b * conv_ch * 4) as u64);
        let conv_out_buf = gpu.alloc_storage("q35_conv_out", (max_b * conv_ch * 4) as u64);
        let z_ssm_buf = gpu.alloc_storage("q35_z_ssm", (max_b * inner * 4) as u64);
        let alpha_buf = gpu.alloc_storage("q35_alpha", (max_b * num_v_heads * 4) as u64);
        let beta_buf = gpu.alloc_storage("q35_beta", (max_b * num_v_heads * 4) as u64);
        let ssm_y_buf = gpu.alloc_storage("q35_ssm_y", (max_b * inner * 4) as u64);
        let hb_buf = gpu.alloc_storage("q35_hb", (max_b * ffn * 4) as u64);
        let hb2_buf = gpu.alloc_storage("q35_hb2", (max_b * ffn * 4) as u64);
        let logits_buf = gpu.alloc_storage("q35_logits", (config.vocab_size * 4) as u64);
        let logits_staging = gpu.alloc_staging_read("q35_logits_stage", (config.vocab_size * 4) as u64);

        let rope = RopeTable::new(config.max_seq_len, &config.rope);
        let half_rope = rope.half_dim();
        let cos_sin = gpu.upload_f32("q35_cos_sin", &rope.packed_cos_sin());

        let mut vram_weight_bytes = 0usize;
        let output_norm = gpu.upload_f32("q35_out_norm", &output_norm_vec);
        vram_weight_bytes += output_norm_vec.len() * 4;
        let (output_weight, out_b) = gpu.upload_weight("q35_output", &output_weight_view);
        vram_weight_bytes += out_b;
        gpu.flush_uploads();

        // Pre-upload all layer attn_norm weights so layer `l` can fuse `res_add + next_attn_norm`
        let mut attn_norms = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let v = gguf.tensor(&format!("blk.{l}.attn_norm.weight"))?.to_f32_vec()?;
            vram_weight_bytes += v.len() * 4;
            attn_norms.push(gpu.upload_f32(&format!("q35_l{l}_an"), &v));
        }

        let bg_layer0_attn_norm = gpu.bind_rms_norm(&x_buf, &attn_norms[0], &xb_buf, &arena);

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut full_kv_idx = 0usize;

        for l in 0..config.num_layers {
            let post_norm_vec = gguf
                .tensor(&format!("blk.{l}.post_attention_norm.weight"))?
                .to_f32_vec()?;
            let post_attn_norm = gpu.upload_f32(&format!("q35_l{l}_pan"), &post_norm_vec);
            vram_weight_bytes += post_norm_vec.len() * 4;

            let is_full = config.is_full_attn(l) || gguf.has_tensor(&format!("blk.{l}.attn_q.weight"));
            let mixer = if is_full {
                let wq_v = gguf.tensor(&format!("blk.{l}.attn_q.weight"))?;
                let wk_v = gguf.tensor(&format!("blk.{l}.attn_k.weight"))?;
                let wv_v = gguf.tensor(&format!("blk.{l}.attn_v.weight"))?;
                let wo_v = gguf.tensor(&format!("blk.{l}.attn_output.weight"))?;
                let _ = gpu.matmul_pipeline(wq_v.dtype, 1)?;
                let _ = gpu.matmul_pipeline(wk_v.dtype, 1)?;
                let _ = gpu.matmul_pipeline(wv_v.dtype, 1)?;
                let _ = gpu.matmul_pipeline(wo_v.dtype, 1)?;

                let (wq_gate, b0) = gpu.upload_weight(&format!("q35_l{l}_wq"), &wq_v);
                let (wk, b1) = gpu.upload_weight(&format!("q35_l{l}_wk"), &wk_v);
                let (wv, b2) = gpu.upload_weight(&format!("q35_l{l}_wv"), &wv_v);
                let (wo, b3) = gpu.upload_weight(&format!("q35_l{l}_wo"), &wo_v);
                vram_weight_bytes += b0 + b1 + b2 + b3;

                let qn_v = gguf.tensor(&format!("blk.{l}.attn_q_norm.weight"))?.to_f32_vec()?;
                let kn_v = gguf.tensor(&format!("blk.{l}.attn_k_norm.weight"))?.to_f32_vec()?;
                let q_norm = gpu.upload_f32(&format!("q35_l{l}_qn"), &qn_v);
                let k_norm = gpu.upload_f32(&format!("q35_l{l}_kn"), &kn_v);
                vram_weight_bytes += (qn_v.len() + kn_v.len()) * 4;

                let bg_wq_gate = gpu.bind_matmul(&wq_gate, &xb_buf, &qg_buf, &arena);
                let bg_wk = gpu.bind_matmul(&wk, &xb_buf, &k_buf, &arena);
                let bg_wv = gpu.bind_matmul(&wv, &xb_buf, &v_buf, &arena);
                let bg_q_norm = gpu.bind_rw_r_u(&q_buf, &q_norm, &arena);
                let bg_k_norm = gpu.bind_rw_r_u(&k_buf, &k_norm, &arena);
                let bg_wo = gpu.bind_matmul(&wo, &attn_out_buf, &dx_buf, &arena);

                let kv_idx = full_kv_idx;
                full_kv_idx += 1;

                GpuMixer::FullAttn(Box::new(GpuFullAttnLayer {
                    kv_layer_idx: kv_idx,
                    _wq_gate: wq_gate,
                    _wk: wk,
                    _wv: wv,
                    _wo: wo,
                    _q_norm: q_norm,
                    _k_norm: k_norm,
                    wq_gate_dt: wq_v.dtype,
                    wk_dt: wk_v.dtype,
                    wv_dt: wv_v.dtype,
                    wo_dt: wo_v.dtype,
                    bg_wq_gate,
                    bg_wk,
                    bg_wv,
                    bg_q_norm,
                    bg_k_norm,
                    bg_wo,
                }))
            } else {
                let wqkv_v = gguf.tensor(&format!("blk.{l}.attn_qkv.weight"))?;
                let wgate_v = gguf.tensor(&format!("blk.{l}.attn_gate.weight"))?;
                let wa_v = gguf.tensor(&format!("blk.{l}.ssm_alpha.weight"))?;
                let wb_v = gguf.tensor(&format!("blk.{l}.ssm_beta.weight"))?;
                let wout_v = gguf.tensor(&format!("blk.{l}.ssm_out.weight"))?;
                for dt in [wqkv_v.dtype, wgate_v.dtype, wa_v.dtype, wb_v.dtype, wout_v.dtype] {
                    let _ = gpu.matmul_pipeline(dt, 1)?;
                }

                let (wqkv, b0) = gpu.upload_weight(&format!("q35_l{l}_qkv"), &wqkv_v);
                let (wgate, b1) = gpu.upload_weight(&format!("q35_l{l}_z"), &wgate_v);
                let (w_alpha, b2) = gpu.upload_weight(&format!("q35_l{l}_a"), &wa_v);
                let (w_beta, b3) = gpu.upload_weight(&format!("q35_l{l}_b"), &wb_v);
                let (w_out, b4) = gpu.upload_weight(&format!("q35_l{l}_so"), &wout_v);
                vram_weight_bytes += b0 + b1 + b2 + b3 + b4;

                let conv_w_vec = gguf.tensor(&format!("blk.{l}.ssm_conv1d.weight"))?.to_f32_vec()?;
                let ssm_a_vec = gguf.tensor(&format!("blk.{l}.ssm_a"))?.to_f32_vec()?;
                let dt_bias_vec = gguf.tensor(&format!("blk.{l}.ssm_dt.bias"))?.to_f32_vec()?;
                let ssm_norm_vec = gguf.tensor(&format!("blk.{l}.ssm_norm.weight"))?.to_f32_vec()?;

                let mut packed_params = Vec::with_capacity(ssm_a_vec.len() + dt_bias_vec.len() + ssm_norm_vec.len());
                packed_params.extend_from_slice(&ssm_a_vec);
                packed_params.extend_from_slice(&dt_bias_vec);
                packed_params.extend_from_slice(&ssm_norm_vec);

                let conv_w = gpu.upload_f32(&format!("q35_l{l}_cw"), &conv_w_vec);
                let ssm_params = gpu.upload_f32(&format!("q35_l{l}_sp"), &packed_params);
                vram_weight_bytes += (conv_w_vec.len() + packed_params.len()) * 4;

                let conv_state = gpu.alloc_storage(&format!("q35_l{l}_cs"), conv_state_bytes);
                let ssm_state = gpu.alloc_storage(&format!("q35_l{l}_ss"), ssm_state_bytes);
                total_state_bytes += (conv_state_bytes + ssm_state_bytes) as usize;

                let bg_wqkv = gpu.bind_matmul(&wqkv, &xb_buf, &qkv_ssm_buf, &arena);
                let bg_wgate = gpu.bind_matmul(&wgate, &xb_buf, &z_ssm_buf, &arena);
                let bg_alpha = gpu.bind_matmul(&w_alpha, &xb_buf, &alpha_buf, &arena);
                let bg_beta = gpu.bind_matmul(&w_beta, &xb_buf, &beta_buf, &arena);
                let bg_conv = gpu.bind_deltanet_conv(&qkv_ssm_buf, &conv_w, &conv_state, &conv_out_buf, &arena);
                let bg_step = gpu.bind_deltanet_step(
                    &conv_out_buf,
                    &z_ssm_buf,
                    &alpha_buf,
                    &beta_buf,
                    &ssm_params,
                    &ssm_state,
                    &ssm_y_buf,
                    &arena,
                );
                let bg_out = gpu.bind_matmul(&w_out, &ssm_y_buf, &dx_buf, &arena);

                GpuMixer::DeltaNet(Box::new(GpuDeltaNetLayer {
                    _wqkv: wqkv,
                    _wgate: wgate,
                    _w_alpha: w_alpha,
                    _w_beta: w_beta,
                    _w_out: w_out,
                    _conv_w: conv_w,
                    _ssm_params: ssm_params,
                    conv_state,
                    ssm_state,
                    wqkv_dt: wqkv_v.dtype,
                    wgate_dt: wgate_v.dtype,
                    w_alpha_dt: wa_v.dtype,
                    w_beta_dt: wb_v.dtype,
                    w_out_dt: wout_v.dtype,
                    bg_wqkv,
                    bg_wgate,
                    bg_alpha,
                    bg_beta,
                    bg_conv,
                    bg_step,
                    bg_out,
                }))
            };

            let wg_v = gguf.tensor(&format!("blk.{l}.ffn_gate.weight"))?;
            let wu_v = gguf.tensor(&format!("blk.{l}.ffn_up.weight"))?;
            let wd_v = gguf.tensor(&format!("blk.{l}.ffn_down.weight"))?;
            let _ = gpu.matmul_pipeline(wg_v.dtype, 1)?;
            let _ = gpu.matmul_pipeline(wu_v.dtype, 1)?;
            let _ = gpu.matmul_pipeline(wd_v.dtype, 1)?;

            let (w_gate, bg0) = gpu.upload_weight(&format!("q35_l{l}_fg"), &wg_v);
            let (w_up, bg1) = gpu.upload_weight(&format!("q35_l{l}_fu"), &wu_v);
            let (w_down, bg2) = gpu.upload_weight(&format!("q35_l{l}_fd"), &wd_v);
            vram_weight_bytes += bg0 + bg1 + bg2;

            let bg_res_post_norm = gpu.bind_res_rms(&x_buf, &dx_buf, &post_attn_norm, &xb_buf, &arena);
            let bg_w_gate = gpu.bind_matmul(&w_gate, &xb_buf, &hb_buf, &arena);
            let bg_w_up = gpu.bind_matmul(&w_up, &xb_buf, &hb2_buf, &arena);
            let bg_gate_up_swiglu = if wg_v.dtype == DType::Q4_K && wu_v.dtype == DType::Q4_K {
                Some(gpu.bind_gate_up(&w_gate, &w_up, &xb_buf, &hb_buf, &arena))
            } else {
                None
            };
            let bg_w_down = gpu.bind_matmul(&w_down, &hb_buf, &dx_buf, &arena);
            let bg_res_next_attn_norm = if l + 1 < config.num_layers {
                Some(gpu.bind_res_rms(&x_buf, &dx_buf, &attn_norms[l + 1], &xb_buf, &arena))
            } else {
                None
            };

            if l % 4 == 3 {
                gpu.flush_uploads();
            }

            layers.push(GpuQwen35Layer {
                _attn_norm: attn_norms[l].clone(),
                _post_attn_norm: post_attn_norm,
                mixer,
                _w_gate: w_gate,
                _w_up: w_up,
                _w_down: w_down,
                w_gate_dt: wg_v.dtype,
                w_up_dt: wu_v.dtype,
                w_down_dt: wd_v.dtype,
                bg_res_post_norm,
                bg_w_gate,
                bg_w_up,
                bg_gate_up_swiglu,
                bg_w_down,
                bg_res_next_attn_norm,
            });
        }
        gpu.flush_uploads();

        let bg_deinterleave_qg = gpu.bind_deinterleave_qg(&qg_buf, &q_buf, &attn_gate_buf, &arena);
        let bg_rope_kv = gpu.bind_rope_kv(&q_buf, &k_buf, &v_buf, &cos_sin, &k_cache, &v_cache, &arena);
        let bg_gqa = gpu.bind_gqa(&q_buf, &k_cache, &v_cache, &attn_out_buf, &arena);
        let bg_mul_sigmoid = gpu.bind_rw_r_u(&attn_out_buf, &attn_gate_buf, &arena);
        let bg_l2_norm_qk = gpu.bind_rw_r_u(&conv_out_buf, &z_ssm_buf, &arena);
        let bg_swiglu = gpu.bind_rw_r_u(&hb_buf, &hb2_buf, &arena);
        let bg_res_add = gpu.bind_rw_r_u(&x_buf, &dx_buf, &arena);
        let bg_res_out_norm = gpu.bind_res_rms(&x_buf, &dx_buf, &output_norm, &xb_buf, &arena);
        let bg_out_norm = gpu.bind_rms_norm(&x_last_buf, &output_norm, &xb_buf, &arena);
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
            _cos_sin: cos_sin,
            half_rope,
            k_cache,
            v_cache,
            kv_cache_bytes: total_state_bytes,
            vram_weight_bytes,
            conv_state_bytes,
            x_buf,
            _dx_buf: dx_buf,
            x_last_buf,
            _xb_buf: xb_buf,
            _qg_buf: qg_buf,
            _q_buf: q_buf,
            _k_buf: k_buf,
            _v_buf: v_buf,
            _attn_gate_buf: attn_gate_buf,
            _attn_out_buf: attn_out_buf,
            _qkv_ssm_buf: qkv_ssm_buf,
            _conv_out_buf: conv_out_buf,
            _z_ssm_buf: z_ssm_buf,
            _alpha_buf: alpha_buf,
            _beta_buf: beta_buf,
            _ssm_y_buf: ssm_y_buf,
            _hb_buf: hb_buf,
            _hb2_buf: hb2_buf,
            logits_buf,
            logits_staging,
            bg_layer0_attn_norm,
            bg_deinterleave_qg,
            bg_rope_kv,
            bg_gqa,
            bg_mul_sigmoid,
            bg_l2_norm_qk,
            bg_swiglu,
            bg_res_add,
            bg_res_out_norm,
            bg_out_norm,
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
            label: Some("q35_clear_state"),
        });
        enc.clear_buffer(&self.k_cache, 0, None);
        enc.clear_buffer(&self.v_cache, 0, None);
        for l in &self.layers {
            if let GpuMixer::DeltaNet(dn) = &l.mixer {
                if self.conv_state_bytes > 0 {
                    enc.clear_buffer(&dn.conv_state, 0, None);
                }
                enc.clear_buffer(&dn.ssm_state, 0, None);
            }
        }
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
        let (pipe, is_b4) = gpu.matmul_pipeline(dtype, batch)?;
        let off = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[off]);
        let y = if is_b4 { batch.div_ceil(4) } else { batch } as u32;
        let gx = rows.div_ceil(4) as u32;
        let (dx, dz) = if gx > 65_535 {
            (32_768u32, gx.div_ceil(32_768))
        } else {
            (gx, 1u32)
        };
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
        let n_kv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let eps_bits = self.config.rms_norm_eps.to_bits();
        let inv_sqrt_hd_bits = (1.0f32 / (hd as f32).sqrt()).to_bits();

        let num_k_heads = self.config.ssm_group_count;
        let head_k_dim = self.config.ssm_state_size;
        let num_v_heads = self.config.ssm_time_step_rank;
        let head_v_dim = self.config.ssm_inner_size / num_v_heads;
        let inner = self.config.ssm_inner_size;
        let conv_ch = 2 * num_k_heads * head_k_dim + inner;
        let ssm_scale_bits = (1.0f32 / (head_v_dim as f32).sqrt()).to_bits();

        for (t, &tok) in tokens.iter().enumerate() {
            self.token_embd
                .dequantize_row(tok as usize, &mut self.embed_scratch[t * h..(t + 1) * h])?;
        }
        self.gpu.write_f32(&self.x_buf, &self.embed_scratch[..b * h]);

        self.arena.reset();
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q35_forward"),
        });

        let p = &self.gpu.pipelines;
        let kv_layer_stride_u32 = self.config.max_seq_len * (kv_dim / 2);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q35_layers"),
                timestamp_writes: None,
            });

            // Layer 0 initial RMSNorm
            let off0 = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
            pass.set_pipeline(&p.rms_norm);
            pass.set_bind_group(0, &self.bg_layer0_attn_norm, &[off0]);
            pass.dispatch_workgroups(b as u32, 1, 1);

            for (l, layer) in self.layers.iter().enumerate() {
                match &layer.mixer {
                    GpuMixer::DeltaNet(dn) => {
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, dn.wqkv_dt, &dn.bg_wqkv, conv_ch, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, dn.wgate_dt, &dn.bg_wgate, inner, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, dn.w_alpha_dt, &dn.bg_alpha, num_v_heads, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, dn.w_beta_dt, &dn.bg_beta, num_v_heads, h, b)?;

                        // Causal Conv1D + SiLU
                        let off_conv = self.arena.push_u32x4(
                            conv_ch as u32,
                            self.config.ssm_conv_kernel as u32,
                            b as u32,
                            0,
                        );
                        pass.set_pipeline(&p.deltanet_conv_silu);
                        pass.set_bind_group(0, &dn.bg_conv, &[off_conv]);
                        pass.dispatch_workgroups(conv_ch.div_ceil(64) as u32, 1, 1);

                        // L2-norm Q and K heads in CONV_OUT
                        let off_l2 = self.arena.push_u32x4(
                            head_k_dim as u32,
                            (num_k_heads * 2) as u32,
                            conv_ch as u32,
                            eps_bits,
                        );
                        pass.set_pipeline(&p.l2_norm_qk);
                        pass.set_bind_group(0, &self.bg_l2_norm_qk, &[off_l2]);
                        pass.dispatch_workgroups((num_k_heads * 2) as u32, b as u32, 1);

                        // Fused DeltaNet Recurrence + Gated RMSNorm
                        let off_step = self.arena.push_u32x8(
                            num_k_heads as u32,
                            num_v_heads as u32,
                            head_v_dim as u32,
                            conv_ch as u32,
                            b as u32,
                            eps_bits,
                            ssm_scale_bits,
                            0,
                        );
                        pass.set_pipeline(&p.deltanet_step);
                        pass.set_bind_group(0, &dn.bg_step, &[off_step]);
                        pass.dispatch_workgroups(num_v_heads as u32, 1, 1);

                        // Output projection
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, dn.w_out_dt, &dn.bg_out, h, inner, b)?;
                    }
                    GpuMixer::FullAttn(fa) => {
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, fa.wq_gate_dt, &fa.bg_wq_gate, q_dim * 2, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, fa.wk_dt, &fa.bg_wk, kv_dim, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, fa.wv_dt, &fa.bg_wv, kv_dim, h, b)?;

                        // De-interleave [Q, gate]
                        let off_di = self.arena.push_u32x4(n_heads as u32, hd as u32, b as u32, 0);
                        pass.set_pipeline(&p.deinterleave_qg);
                        pass.set_bind_group(0, &self.bg_deinterleave_qg, &[off_di]);
                        pass.dispatch_workgroups((b * q_dim).div_ceil(64) as u32, 1, 1);

                        // QK-RMSNorm
                        let off_qn = self.arena.push_u32x4(hd as u32, (b * n_heads) as u32, eps_bits, 0);
                        pass.set_pipeline(&p.rms_norm_inplace);
                        pass.set_bind_group(0, &fa.bg_q_norm, &[off_qn]);
                        pass.dispatch_workgroups((b * n_heads) as u32, 1, 1);

                        let off_kn = self.arena.push_u32x4(hd as u32, (b * n_kv) as u32, eps_bits, 0);
                        pass.set_pipeline(&p.rms_norm_inplace);
                        pass.set_bind_group(0, &fa.bg_k_norm, &[off_kn]);
                        pass.dispatch_workgroups((b * n_kv) as u32, 1, 1);

                        // Partial NeoX RoPE + Packed F16 KV Store
                        let layer_kv_u32_off = (fa.kv_layer_idx * kv_layer_stride_u32) as u32;
                        let off_rk = self.arena.push_u32x8(
                            n_heads as u32,
                            n_kv as u32,
                            hd as u32,
                            self.half_rope as u32,
                            start_pos as u32,
                            b as u32,
                            layer_kv_u32_off,
                            1, // NeoX
                        );
                        pass.set_pipeline(&p.rope_kv_store);
                        pass.set_bind_group(0, &self.bg_rope_kv, &[off_rk]);
                        let rope_threads = (n_heads * self.half_rope).max(kv_dim / 2);
                        pass.dispatch_workgroups(rope_threads.div_ceil(64) as u32, b as u32, 1);

                        // Flash GQA Attention
                        let off_gqa = self.arena.push_u32x8(
                            n_heads as u32,
                            n_kv as u32,
                            hd as u32,
                            start_pos as u32,
                            b as u32,
                            layer_kv_u32_off,
                            inv_sqrt_hd_bits,
                            0,
                        );
                        pass.set_pipeline(&p.gqa_flash_attn);
                        pass.set_bind_group(0, &self.bg_gqa, &[off_gqa]);
                        pass.dispatch_workgroups(n_heads as u32, b as u32, 1);

                        // Output gate: attn_out *= sigmoid(attn_gate)
                        let off_ms = self.arena.push_u32x4((b * q_dim) as u32, 0, 0, 0);
                        pass.set_pipeline(&p.mul_sigmoid);
                        pass.set_bind_group(0, &self.bg_mul_sigmoid, &[off_ms]);
                        pass.dispatch_workgroups((b * q_dim).div_ceil(64) as u32, 1, 1);

                        // Output projection
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, fa.wo_dt, &fa.bg_wo, h, q_dim, b)?;
                    }
                }

                // Fused Residual Add + Post-Attention RMSNorm
                let off_pn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                pass.set_pipeline(&p.residual_rms_norm);
                pass.set_bind_group(0, &layer.bg_res_post_norm, &[off_pn]);
                pass.dispatch_workgroups(b as u32, 1, 1);

                // SwiGLU FFN
                if b == 1 {
                    if let Some(ref bg_gu) = layer.bg_gate_up_swiglu {
                        let off_gu = self.arena.push_u32x4(ffn as u32, h as u32, 1, 0);
                        pass.set_pipeline(&p.matmul_q4_k_gate_up_swiglu);
                        pass.set_bind_group(0, bg_gu, &[off_gu]);
                        pass.dispatch_workgroups(ffn.div_ceil(4) as u32, 1, 1);
                    } else {
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_gate_dt, &layer.bg_w_gate, ffn, h, b)?;
                        Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_up_dt, &layer.bg_w_up, ffn, h, b)?;
                        let off_sw = self.arena.push_u32x4((b * ffn) as u32, 0, 0, 0);
                        pass.set_pipeline(&p.swiglu);
                        pass.set_bind_group(0, &self.bg_swiglu, &[off_sw]);
                        pass.dispatch_workgroups((b * ffn).div_ceil(64) as u32, 1, 1);
                    }
                } else {
                    Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_gate_dt, &layer.bg_w_gate, ffn, h, b)?;
                    Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_up_dt, &layer.bg_w_up, ffn, h, b)?;
                    let off_sw = self.arena.push_u32x4((b * ffn) as u32, 0, 0, 0);
                    pass.set_pipeline(&p.swiglu);
                    pass.set_bind_group(0, &self.bg_swiglu, &[off_sw]);
                    pass.dispatch_workgroups((b * ffn).div_ceil(64) as u32, 1, 1);
                }

                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, layer.w_down_dt, &layer.bg_w_down, h, ffn, b)?;

                // Fused FFN Residual Add + Next Layer Attn RMSNorm (or Output Norm on last layer)
                if let Some(ref bg_next) = layer.bg_res_next_attn_norm {
                    let off_rn = self.arena.push_u32x4(h as u32, b as u32, eps_bits, 0);
                    pass.set_pipeline(&p.residual_rms_norm);
                    pass.set_bind_group(0, bg_next, &[off_rn]);
                    pass.dispatch_workgroups(b as u32, 1, 1);
                } else if l + 1 == self.layers.len() {
                    if b == 1 && compute_last_logits {
                        let off_ro = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
                        pass.set_pipeline(&p.residual_rms_norm);
                        pass.set_bind_group(0, &self.bg_res_out_norm, &[off_ro]);
                        pass.dispatch_workgroups(1, 1, 1);
                    } else {
                        let off_ra = self.arena.push_u32x4((b * h) as u32, 0, 0, 0);
                        pass.set_pipeline(&p.residual_add);
                        pass.set_bind_group(0, &self.bg_res_add, &[off_ra]);
                        pass.dispatch_workgroups((b * h).div_ceil(64) as u32, 1, 1);
                    }
                }
            }

            if compute_last_logits && b == 1 {
                Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, self.output_dt, &self.bg_lm_head, self.config.vocab_size, h, 1)?;
            }
        }

        if compute_last_logits && b > 1 {
            encoder.copy_buffer_to_buffer(
                &self.x_buf,
                ((b - 1) * h * 4) as u64,
                &self.x_last_buf,
                0,
                (h * 4) as u64,
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q35_lm_head"),
                timestamp_writes: None,
            });
            let off_on = self.arena.push_u32x4(h as u32, 1, eps_bits, 0);
            pass.set_pipeline(&p.rms_norm);
            pass.set_bind_group(0, &self.bg_out_norm, &[off_on]);
            pass.dispatch_workgroups(1, 1, 1);
            Self::dispatch_matmul(&self.gpu, &mut self.arena, &mut pass, self.output_dt, &self.bg_lm_head, self.config.vocab_size, h, 1)?;
        }

        if compute_last_logits {
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
    use crate::qwen35::Qwen35Model;
    use caelum_gguf::{GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;

    fn synth_vec(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 29 + seed * 13) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    #[test]
    fn test_wgpu_qwen35_deltanet_and_full_attn_parity_vs_cpu() {
        let Ok(gpu) = GpuContext::new() else {
            return;
        };
        let gpu = Arc::new(gpu);

        let vocab = 16usize;
        let h = 32usize;
        let ffn = 64usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let hd = 16usize;
        let ssm_k_heads = 2usize;
        let ssm_v_heads = 4usize;
        let ssm_d = 8usize;
        let conv_ch = 2 * ssm_k_heads * ssm_d + ssm_v_heads * ssm_d;
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
        let cpu_backend = CpuBackend::new(Some(1));
        let mut cpu = Qwen35Model::from_gguf_with_kv(&gguf, Some(16), KvPrecision::F32).unwrap();
        let mut gpu_m = WgpuQwen35Model::from_gguf(&gguf, gpu, Some(16)).unwrap();

        let prompt = [2u32, 4, 6];
        let c_pre = cpu.forward_batch(&cpu_backend, &prompt, 0, true).unwrap().to_vec();
        let g_pre = gpu_m.forward_batch(&prompt, 0, true).unwrap().to_vec();
        for (a, b) in c_pre.iter().zip(g_pre.iter()) {
            assert!((a - b).abs() < 2e-3, "qwen35 prefill GPU vs CPU mismatch: {a} vs {b}");
        }
        let c_dec = cpu.forward_token(&cpu_backend, 7, 3, true).unwrap().to_vec();
        let g_dec = gpu_m.forward_token(7, 3, true).unwrap().to_vec();
        for (a, b) in c_dec.iter().zip(g_dec.iter()) {
            assert!((a - b).abs() < 2e-3, "qwen35 decode GPU vs CPU mismatch: {a} vs {b}");
        }
    }
}
