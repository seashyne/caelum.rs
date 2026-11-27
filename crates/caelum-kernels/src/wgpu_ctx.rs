//! Cross-platform `wgpu` Compute Engine (Vulkan / Metal / DX12) for Caelum (`--features gpu`).
//!
//! Designed for **zero per-token allocations and single-submission forward passes**:
//! - Every pipeline uses explicit `BindGroupLayout`s with dynamic uniform offsets (`UniformArena`)
//!   so all per-layer `BindGroup`s are pre-baked ONCE at model load time.
//! - Supports `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q8_0`, `F16`, and `F32` quantized GPU shaders
//!   directly over raw GGUF bytes (zero VRAM expansion).

use crate::wgpu_shaders;
use caelum_core::{f16_to_f32, CaelumError, DType, Result, TensorView};
use std::num::NonZeroU64;
use std::sync::Arc;

pub const UNIFORM_STRIDE: usize = 256;
pub const UNIFORM_ARENA_SLOTS: usize = 2048; // 512 KB (> enough for 64+ layers per submit)

/// CPU-staged dynamic uniform ring buffer uploaded in ONE `queue.write_buffer` per forward pass.
pub struct UniformArena {
    pub buffer: wgpu::Buffer,
    staging: Vec<u8>,
    count: usize,
}

impl UniformArena {
    pub fn new(device: &wgpu::Device) -> Self {
        let size = (UNIFORM_ARENA_SLOTS * UNIFORM_STRIDE) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("caelum-uniform-arena"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            staging: vec![0u8; UNIFORM_ARENA_SLOTS * UNIFORM_STRIDE],
            count: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.count = 0;
    }

    /// Appends up to 256 bytes of uniform parameters and returns its dynamic byte offset.
    #[inline]
    pub fn push_u32x4(&mut self, a: u32, b: u32, c: u32, d: u32) -> u32 {
        let slot = self.count;
        assert!(slot < UNIFORM_ARENA_SLOTS, "UniformArena overflow");
        self.count += 1;
        let off = slot * UNIFORM_STRIDE;
        self.staging[off..off + 4].copy_from_slice(&a.to_le_bytes());
        self.staging[off + 4..off + 8].copy_from_slice(&b.to_le_bytes());
        self.staging[off + 8..off + 12].copy_from_slice(&c.to_le_bytes());
        self.staging[off + 12..off + 16].copy_from_slice(&d.to_le_bytes());
        off as u32
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn push_u32x8(
        &mut self,
        w0: u32,
        w1: u32,
        w2: u32,
        w3: u32,
        w4: u32,
        w5: u32,
        w6: u32,
        w7: u32,
    ) -> u32 {
        let slot = self.count;
        assert!(slot < UNIFORM_ARENA_SLOTS, "UniformArena overflow");
        self.count += 1;
        let off = slot * UNIFORM_STRIDE;
        for (i, w) in [w0, w1, w2, w3, w4, w5, w6, w7].into_iter().enumerate() {
            self.staging[off + i * 4..off + (i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        off as u32
    }

    /// Uploads all recorded uniform slots in a single DMA call before `queue.submit`.
    pub fn flush(&self, queue: &wgpu::Queue) {
        if self.count > 0 {
            let bytes = self.count * UNIFORM_STRIDE;
            queue.write_buffer(&self.buffer, 0, &self.staging[..bytes]);
        }
    }
}

/// Pre-compiled hardware cooperative-matrix (`M=8, N=16, K=16`, `f16×f16+f32`) GEMM pipelines
/// (e.g. Intel Xe2 XMX cores), populated only when the GPU exposes `SPV_KHR_cooperative_matrix`.
pub struct XmxPipelines {
    pub q4_k_m8: wgpu::ComputePipeline,
    pub q4_k_m16: wgpu::ComputePipeline,
    pub q4_k_m32: wgpu::ComputePipeline,
    pub q5_k_m8: wgpu::ComputePipeline,
    pub q5_k_m16: wgpu::ComputePipeline,
    pub q6_k_m8: wgpu::ComputePipeline,
    pub q6_k_m16: wgpu::ComputePipeline,
    pub f16_m16: wgpu::ComputePipeline,
    pub f32_m16: wgpu::ComputePipeline,
}

pub struct GpuPipelines {
    pub bgl_matmul: wgpu::BindGroupLayout,
    pub bgl_qkv: wgpu::BindGroupLayout,
    pub bgl_gate_up: wgpu::BindGroupLayout,
    pub bgl_rms_norm: wgpu::BindGroupLayout,
    pub bgl_res_rms: wgpu::BindGroupLayout,
    pub bgl_rw_r_u: wgpu::BindGroupLayout,
    pub bgl_rope_kv: wgpu::BindGroupLayout,
    pub bgl_rope_kv_slots: wgpu::BindGroupLayout,
    pub bgl_gqa: wgpu::BindGroupLayout,
    pub bgl_gqa_slots: wgpu::BindGroupLayout,
    pub bgl_deinterleave_qg: wgpu::BindGroupLayout,
    pub bgl_deltanet_conv: wgpu::BindGroupLayout,
    pub bgl_deltanet_step: wgpu::BindGroupLayout,

    pub matmul_q4_k: wgpu::ComputePipeline,
    pub matmul_q4_k_b4: wgpu::ComputePipeline,
    pub matmul_q4_k_qkv: wgpu::ComputePipeline,
    pub matmul_q4_k_qkv_b4: wgpu::ComputePipeline,
    pub matmul_q4_k_gate_up_swiglu: wgpu::ComputePipeline,
    pub matmul_q5_k: wgpu::ComputePipeline,
    pub matmul_q6_k: wgpu::ComputePipeline,
    pub matmul_q6_k_b4: wgpu::ComputePipeline,
    pub matmul_q4_0: wgpu::ComputePipeline,
    pub matmul_q8_0: wgpu::ComputePipeline,
    pub matmul_f16: wgpu::ComputePipeline,
    pub matmul_f32: wgpu::ComputePipeline,
    pub xmx: Option<XmxPipelines>,

    pub rms_norm: wgpu::ComputePipeline,
    pub residual_rms_norm: wgpu::ComputePipeline,
    pub rms_norm_inplace: wgpu::ComputePipeline,
    pub add_bias: wgpu::ComputePipeline,
    pub rope_kv_store: wgpu::ComputePipeline,
    pub rope_kv_store_slots: wgpu::ComputePipeline,
    pub gqa_flash_attn: wgpu::ComputePipeline,
    pub gqa_flash_attn_slots: wgpu::ComputePipeline,
    pub swiglu: wgpu::ComputePipeline,
    pub geglu: wgpu::ComputePipeline,
    pub residual_add: wgpu::ComputePipeline,
    pub residual_add_scale: wgpu::ComputePipeline,
    pub mul_sigmoid: wgpu::ComputePipeline,
    pub deinterleave_qg: wgpu::ComputePipeline,
    pub deltanet_conv_silu: wgpu::ComputePipeline,
    pub l2_norm_qk: wgpu::ComputePipeline,
    pub deltanet_step: wgpu::ComputePipeline,
}

/// Shared `wgpu` GPU device, queue, adapter metadata, and pre-compiled WGSL compute pipelines.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub use_subgroups: bool,
    pub has_f16: bool,
    pub has_xmx: bool,
    pub pipelines: GpuPipelines,
}

impl GpuContext {
    /// Initializes the highest-performance GPU adapter (Vulkan / Metal / DX12) and compiles
    /// all Caelum WGSL/SPIR-V compute pipelines once.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| CaelumError::Runtime(format!("no compatible wgpu GPU adapter found: {e}")))?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let adapter_features = adapter.features();
        let use_subgroups = adapter_features.contains(wgpu::Features::SUBGROUP)
            && adapter_info.subgroup_min_size >= 8
            && adapter_info.subgroup_max_size <= 32;
        let has_f16 = adapter_features.contains(wgpu::Features::SHADER_F16);
        let has_xmx = has_f16
            && adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter_features.contains(wgpu::Features::PASSTHROUGH_SHADERS)
            && adapter.cooperative_matrix_properties().iter().any(|p| {
                p.m_size == 8
                    && p.n_size == 16
                    && p.k_size == 16
                    && p.ab_type == wgpu::CooperativeScalarType::F16
                    && p.cr_type == wgpu::CooperativeScalarType::F32
            });

        let mut req_features = wgpu::Features::empty();
        if use_subgroups {
            req_features |= wgpu::Features::SUBGROUP;
        }
        if has_f16 {
            req_features |= wgpu::Features::SHADER_F16;
        }
        if has_xmx {
            req_features |=
                wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::PASSTHROUGH_SHADERS;
        }

        let mut req_limits = wgpu::Limits::default();
        req_limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        req_limits.max_buffer_size = adapter_limits.max_buffer_size;
        req_limits.max_compute_workgroup_storage_size =
            adapter_limits.max_compute_workgroup_storage_size;
        req_limits.max_compute_invocations_per_workgroup =
            adapter_limits.max_compute_invocations_per_workgroup;
        req_limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.max(8);

        let experimental_features = if has_xmx {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("caelum-wgpu-device"),
            required_features: req_features,
            required_limits: req_limits.clone(),
            experimental_features,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| CaelumError::Runtime(format!("failed to create wgpu device: {e}")))?;

        let pipelines = build_pipelines(&device, use_subgroups, has_xmx);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits: req_limits,
            use_subgroups,
            has_f16,
            has_xmx,
            pipelines,
        })
    }

    pub fn device_summary(&self) -> String {
        let sg = if self.use_subgroups {
            format!(
                "subgroups={}-{}",
                self.adapter_info.subgroup_min_size, self.adapter_info.subgroup_max_size
            )
        } else {
            "subgroups=off".to_string()
        };
        let xmx = if self.has_xmx { " | XMX" } else { "" };
        format!(
            "{} ({:?}, {:?}) | {sg}{xmx} | max_buf={:.1} GB",
            self.adapter_info.name,
            self.adapter_info.backend,
            self.adapter_info.device_type,
            self.limits.max_storage_buffer_binding_size as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }

    /// Flushes pending `queue.write_buffer` DMA transfers to VRAM and reclaims staging memory
    /// so loading large 7B–14B+ models never accumulates gigabytes of staging RAM or hits Swap.
    pub fn flush_uploads(&self) {
        self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    /// Streams raw quantized/F32 tensor bytes directly into GPU VRAM in bounded `16 MiB` DMA
    /// chunks (`mapped_at_creation: false`), padding to a 4-byte multiple with minimal RAM usage.
    pub fn upload_raw_padded(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let padded_len = ((data.len() + 3) & !3).max(4);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if data.is_empty() {
            return buf;
        }
        const CHUNK: usize = 16 * 1024 * 1024; // 16 MiB (multiple of 4)
        let aligned_main = data.len() & !3;
        let mut off = 0usize;
        while off < aligned_main {
            let end = (off + CHUNK).min(aligned_main);
            self.queue.write_buffer(&buf, off as u64, &data[off..end]);
            off = end;
            if aligned_main > CHUNK && off < aligned_main {
                self.flush_uploads();
            }
        }
        if aligned_main < data.len() {
            let mut tail = [0u8; 4];
            let rem = data.len() - aligned_main;
            tail[..rem].copy_from_slice(&data[aligned_main..]);
            self.queue.write_buffer(&buf, aligned_main as u64, &tail);
        }
        buf
    }

    /// Uploads concatenated slices directly into ONE VRAM storage buffer with zero intermediate
    /// CPU allocation (used for fused `Q4_K` `[wq; wk; wv]` where each slice is 4-byte aligned).
    pub fn upload_concat_aligned(&self, label: &str, slices: &[&[u8]]) -> wgpu::Buffer {
        let total: usize = slices.iter().map(|s| s.len()).sum();
        let padded_len = ((total + 3) & !3).max(4);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut off = 0usize;
        for s in slices {
            debug_assert_eq!(s.len() % 4, 0);
            if !s.is_empty() {
                self.queue.write_buffer(&buf, off as u64, s);
                off += s.len();
            }
        }
        buf
    }

    /// Streams GGML quantized blocks into 4-byte-aligned GPU VRAM blocks with pre-decoded `f32`
    /// block scale `d`, using bounded ~14 MiB staging chunks (zero write-combining stalls).
    fn upload_blocks_aligned(
        &self,
        label: &str,
        raw: &[u8],
        raw_blk: usize,
        aligned_blk: usize,
        d_at_start: bool,
    ) -> wgpu::Buffer {
        debug_assert_eq!(raw.len() % raw_blk, 0);
        debug_assert_eq!(aligned_blk, raw_blk + 2);
        let n_blocks = raw.len() / raw_blk;
        let out_bytes = (n_blocks * aligned_blk).max(4);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if n_blocks == 0 {
            return buf;
        }
        let chunk_blocks = (14 * 1024 * 1024 / aligned_blk).max(1);
        let cap_blocks = n_blocks.min(chunk_blocks);
        let mut scratch = vec![0u8; cap_blocks * aligned_blk];
        let payload = raw_blk - 2;
        let mut blk_off = 0usize;
        while blk_off < n_blocks {
            let count = (n_blocks - blk_off).min(chunk_blocks);
            for i in 0..count {
                let src = &raw[(blk_off + i) * raw_blk..(blk_off + i + 1) * raw_blk];
                let dst = &mut scratch[i * aligned_blk..(i + 1) * aligned_blk];
                if d_at_start {
                    let d_f32 = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
                    dst[0..4].copy_from_slice(&d_f32.to_le_bytes());
                    dst[4..4 + payload].copy_from_slice(&src[2..2 + payload]);
                } else {
                    dst[0..payload].copy_from_slice(&src[0..payload]);
                    let d_f32 = f16_to_f32(u16::from_le_bytes([src[payload], src[payload + 1]]));
                    dst[payload..payload + 4].copy_from_slice(&d_f32.to_le_bytes());
                }
            }
            self.queue.write_buffer(
                &buf,
                (blk_off * aligned_blk) as u64,
                &scratch[..count * aligned_blk],
            );
            blk_off += count;
            if n_blocks > chunk_blocks && blk_off < n_blocks {
                self.flush_uploads();
            }
        }
        buf
    }

    /// Repacks GGML `Q6_K` (`210` bytes/block) into 4-byte-aligned GPU blocks (`212` bytes = `53 u32`/block).
    pub fn upload_q6k_aligned(&self, label: &str, raw: &[u8]) -> wgpu::Buffer {
        self.upload_blocks_aligned(label, raw, 210, 212, false)
    }

    /// Repacks GGML `Q8_0` (`34` bytes/block) into 4-byte-aligned GPU blocks (`36` bytes = `9 u32`/block).
    pub fn upload_q8_0_aligned(&self, label: &str, raw: &[u8]) -> wgpu::Buffer {
        self.upload_blocks_aligned(label, raw, 34, 36, true)
    }

    /// Repacks GGML `Q4_0` (`18` bytes/block) into 4-byte-aligned GPU blocks (`20` bytes = `5 u32`/block).
    pub fn upload_q4_0_aligned(&self, label: &str, raw: &[u8]) -> wgpu::Buffer {
        self.upload_blocks_aligned(label, raw, 18, 20, true)
    }

    /// Uploads any `TensorView` weight matrix in its optimal 4-byte-aligned VRAM format.
    pub fn upload_weight(&self, label: &str, view: &TensorView<'_>) -> (wgpu::Buffer, usize) {
        match view.dtype {
            DType::Q6_K => {
                let aligned_bytes = (view.data.len() / 210) * 212;
                (self.upload_q6k_aligned(label, view.data), aligned_bytes)
            }
            DType::Q8_0 => {
                let aligned_bytes = (view.data.len() / 34) * 36;
                (self.upload_q8_0_aligned(label, view.data), aligned_bytes)
            }
            DType::Q4_0 => {
                let aligned_bytes = (view.data.len() / 18) * 20;
                (self.upload_q4_0_aligned(label, view.data), aligned_bytes)
            }
            _ => (self.upload_raw_padded(label, view.data), view.data.len()),
        }
    }

    /// Uploads an `f32` slice directly into GPU VRAM (`STORAGE`).
    pub fn upload_f32(&self, label: &str, vals: &[f32]) -> wgpu::Buffer {
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals))
        };
        self.upload_raw_padded(label, bytes)
    }

    /// Allocates an uninitialized GPU VRAM storage buffer (`STORAGE | COPY_SRC | COPY_DST`).
    pub fn alloc_storage(&self, label: &str, byte_size: u64) -> wgpu::Buffer {
        let size = ((byte_size + 3) & !3).max(4);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Allocates a CPU-readable staging buffer (`MAP_READ | COPY_DST`).
    pub fn alloc_staging_read(&self, label: &str, byte_size: u64) -> wgpu::Buffer {
        let size = ((byte_size + 3) & !3).max(4);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Writes `f32` data into a GPU buffer via DMA.
    pub fn write_f32(&self, buf: &wgpu::Buffer, vals: &[f32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals))
        };
        self.queue.write_buffer(buf, 0, bytes);
    }

    /// Writes `u32` data into a GPU buffer via DMA (used for multi-slot continuous batch metadata).
    pub fn write_u32(&self, buf: &wgpu::Buffer, vals: &[u32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals))
        };
        self.queue.write_buffer(buf, 0, bytes);
    }

    /// Maps `staging` after `queue.submit` and copies `dst.len()` floats back to CPU.
    pub fn read_staging_f32(&self, staging: &wgpu::Buffer, dst: &mut [f32]) -> Result<()> {
        let byte_len = std::mem::size_of_val(dst) as u64;
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, 0..byte_len, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| CaelumError::Runtime(format!("wgpu poll error: {e}")))?;
        rx.recv()
            .map_err(|e| CaelumError::Runtime(format!("wgpu map recv error: {e}")))?
            .map_err(|e| CaelumError::Runtime(format!("wgpu map error: {e}")))?;
        {
            let view = staging
                .get_mapped_range(0..byte_len)
                .map_err(|e| CaelumError::Runtime(format!("wgpu mapped range error: {e}")))?;
            let src = unsafe {
                std::slice::from_raw_parts(view.as_ptr() as *const f32, dst.len())
            };
            dst.copy_from_slice(src);
        }
        staging.unmap();
        Ok(())
    }

    /// Returns `(pipeline, wg_x, wg_y, wg_z)` selecting hardware XMX cooperative-matrix GEMM
    /// pipelines when supported (`batch >= 4`, or `batch >= 2` for `Q5_K`) and falling back to
    /// portable WGSL `B4` / `B1` shaders automatically.
    pub fn matmul_dispatch(
        &self,
        dtype: DType,
        rows: usize,
        batch: usize,
    ) -> Result<(&wgpu::ComputePipeline, u32, u32, u32)> {
        if let Some(ref xmx) = self.pipelines.xmx {
            let picked: Option<(&wgpu::ComputePipeline, crate::spv_xmx::XmxTileCfg)> = match dtype {
                DType::Q4_K => {
                    if batch >= 17 {
                        Some((&xmx.q4_k_m32, crate::spv_xmx::CFG_M32_N64_K64))
                    } else if batch >= 9 {
                        Some((&xmx.q4_k_m16, crate::spv_xmx::CFG_M16_N32_K64))
                    } else if batch >= 4 {
                        Some((&xmx.q4_k_m8, crate::spv_xmx::CFG_M8_N32_K128))
                    } else {
                        None
                    }
                }
                DType::Q5_K => {
                    if batch >= 9 {
                        Some((&xmx.q5_k_m16, crate::spv_xmx::CFG_M16_N32_K128))
                    } else if batch >= 2 {
                        Some((&xmx.q5_k_m8, crate::spv_xmx::CFG_M8_N32_K128))
                    } else {
                        None
                    }
                }
                DType::Q6_K => {
                    if batch >= 9 {
                        Some((&xmx.q6_k_m16, crate::spv_xmx::CFG_M16_N32_K128))
                    } else if batch >= 4 {
                        Some((&xmx.q6_k_m8, crate::spv_xmx::CFG_M8_N32_K128))
                    } else {
                        None
                    }
                }
                DType::F16 if batch >= 4 => Some((&xmx.f16_m16, crate::spv_xmx::CFG_M16_N32_K64)),
                DType::F32 if batch >= 4 => Some((&xmx.f32_m16, crate::spv_xmx::CFG_M16_N32_K64)),
                _ => None,
            };
            if let Some((pipe, cfg)) = picked {
                return Ok((
                    pipe,
                    (rows as u32).div_ceil(cfg.n_tile),
                    (batch as u32).div_ceil(cfg.m_tile),
                    1,
                ));
            }
        }
        let (pipeline, is_b4) = self.matmul_pipeline(dtype, batch)?;
        let gx = (rows as u32).div_ceil(4);
        let (wg_x, wg_z) = if gx > 65_535 {
            (32_768u32, gx.div_ceil(32_768))
        } else {
            (gx, 1u32)
        };
        let wg_y = if is_b4 {
            (batch as u32).div_ceil(4)
        } else {
            batch as u32
        };
        Ok((pipeline, wg_x, wg_y, wg_z))
    }

    /// Returns the portable WGSL matmul compute pipeline for `dtype` and `batch` size.
    pub fn matmul_pipeline(&self, dtype: DType, batch: usize) -> Result<(&wgpu::ComputePipeline, bool)> {
        match dtype {
            DType::Q4_K => {
                if batch >= 2 {
                    Ok((&self.pipelines.matmul_q4_k_b4, true))
                } else {
                    Ok((&self.pipelines.matmul_q4_k, false))
                }
            }
            DType::Q5_K => Ok((&self.pipelines.matmul_q5_k, false)),
            DType::Q6_K => {
                if batch >= 2 {
                    Ok((&self.pipelines.matmul_q6_k_b4, true))
                } else {
                    Ok((&self.pipelines.matmul_q6_k, false))
                }
            }
            DType::Q4_0 => Ok((&self.pipelines.matmul_q4_0, false)),
            DType::Q8_0 => Ok((&self.pipelines.matmul_q8_0, false)),
            DType::F16 => Ok((&self.pipelines.matmul_f16, false)),
            DType::F32 => Ok((&self.pipelines.matmul_f32, false)),
            other => Err(CaelumError::UnsupportedDType(format!(
                "wgpu shader not yet implemented for {other}"
            ))),
        }
    }

    pub fn bind_matmul(
        &self,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_matmul,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    pub fn bind_rms_norm(
        &self,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_rms_norm,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    pub fn bind_rw_r_u(
        &self,
        a_rw: &wgpu::Buffer,
        b_ro: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_rw_r_u,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_rw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_ro.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_rope_kv(
        &self,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        cos_sin: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_rope_kv,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cos_sin.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v_cache.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    pub fn bind_gqa(
        &self,
        q: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_gqa,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_qkv(
        &self,
        w_qkv: &wgpu::Buffer,
        b_qkv: &wgpu::Buffer,
        x: &wgpu::Buffer,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_qkv,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_qkv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_qkv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    pub fn bind_gate_up(
        &self,
        w_gate: &wgpu::Buffer,
        w_up: &wgpu::Buffer,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_gate_up,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_gate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    pub fn bind_res_rms(
        &self,
        x_rw: &wgpu::Buffer,
        dx: &wgpu::Buffer,
        w: &wgpu::Buffer,
        out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_res_rms,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_rw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    pub fn bind_deinterleave_qg(
        &self,
        qg: &wgpu::Buffer,
        q: &wgpu::Buffer,
        gate: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_deinterleave_qg,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: qg.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gate.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    pub fn bind_deltanet_conv(
        &self,
        qkv: &wgpu::Buffer,
        conv_w: &wgpu::Buffer,
        conv_state: &wgpu::Buffer,
        conv_out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_deltanet_conv,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: qkv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: conv_w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: conv_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: conv_out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_rope_kv_slots(
        &self,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        cos_sin: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        slot_meta: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_rope_kv_slots,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cos_sin.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: slot_meta.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_gqa_slots(
        &self,
        q: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out: &wgpu::Buffer,
        slot_meta: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_gqa_slots,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: slot_meta.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_deltanet_step(
        &self,
        conv_out: &wgpu::Buffer,
        z: &wgpu::Buffer,
        alpha: &wgpu::Buffer,
        beta: &wgpu::Buffer,
        ssm_params: &wgpu::Buffer,
        ssm_state: &wgpu::Buffer,
        y_out: &wgpu::Buffer,
        arena: &UniformArena,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.bgl_deltanet_step,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: conv_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: alpha.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: beta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ssm_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ssm_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: y_out.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        })
    }

    /// Standalone GPU benchmark helper: uploads `w` and `x_batch` once to VRAM, runs `iters`
    /// dispatches in a single GPU submission, and returns `(elapsed_secs, out_sample)`.
    pub fn bench_matmul(
        &self,
        w: &TensorView<'_>,
        x_batch: &[f32],
        batch: usize,
        iters: usize,
    ) -> Result<(f64, Vec<f32>)> {
        let (rows, cols) = w.dims_2d()?;
        let (pipeline, wg_x, wg_y, wg_z) = self.matmul_dispatch(w.dtype, rows, batch)?;
        let (w_buf, _) = self.upload_weight("bench-w", w);
        let x_buf = self.upload_f32("bench-x", x_batch);
        let out_bytes = (batch * rows * 4) as u64;
        let out_buf = self.alloc_storage("bench-out", out_bytes);
        let staging = self.alloc_staging_read("bench-staging", out_bytes);
        let mut arena = UniformArena::new(&self.device);
        let bg = self.bind_matmul(&w_buf, &x_buf, &out_buf, &arena);
        let u_off = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
        arena.flush(&self.queue);

        // Warmup 1 dispatch so shader JIT / clocks settle
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[u_off]);
                pass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }
            self.queue.submit(Some(enc.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        }

        // Timed batch submission (zero CPU sync overhead between iterations)
        let t0 = std::time::Instant::now();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[u_off]);
            for _ in 0..iters {
                pass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(Some(enc.finish()));
        let mut out = vec![0.0f32; batch * rows];
        self.read_staging_f32(&staging, &mut out)?;
        let elapsed = t0.elapsed().as_secs_f64();
        Ok((elapsed, out))
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, min_bytes: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(min_bytes),
        },
        count: None,
    }
}

fn compile_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn compile_spirv_pipeline(
    device: &wgpu::Device,
    label: &str,
    spirv: Vec<u32>,
    wg_size: u32,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    use std::borrow::Cow;
    let module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some(label),
            entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                name: Cow::Borrowed("main"),
                workgroup_size: (wg_size, 1, 1),
            }]),
            spirv: Some(Cow::Owned(spirv)),
            ..Default::default()
        })
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn build_pipelines(device: &wgpu::Device, use_subgroups: bool, has_xmx: bool) -> GpuPipelines {
    let bgl_matmul = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-matmul"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3, 16),
        ],
    });
    let bgl_qkv = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-qkv"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            uniform_entry(6, 32),
        ],
    });
    let bgl_gate_up = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-gate-up"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4, 16),
        ],
    });
    let bgl_rms_norm = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-rms-norm"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3, 16),
        ],
    });
    let bgl_res_rms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-res-rms"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4, 16),
        ],
    });
    let bgl_rw_r_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-rw-r-u"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, true),
            uniform_entry(2, 16),
        ],
    });
    let bgl_rope_kv = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-rope-kv"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, false),
            uniform_entry(6, 32),
        ],
    });
    let bgl_rope_kv_slots = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-rope-kv-slots"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, true),
            uniform_entry(7, 32),
        ],
    });
    let bgl_gqa = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-gqa"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4, 32),
        ],
    });
    let bgl_gqa_slots = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-gqa-slots"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, true),
            uniform_entry(5, 32),
        ],
    });
    let bgl_deinterleave_qg = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-deinterleave-qg"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3, 16),
        ],
    });
    let bgl_deltanet_conv = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-deltanet-conv"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            uniform_entry(4, 16),
        ],
    });
    let bgl_deltanet_step = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl-deltanet-step"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
            storage_entry(6, false),
            uniform_entry(7, 32),
        ],
    });

    let comp = |label: &str, src: &str, bgl: &wgpu::BindGroupLayout| {
        let specialized = wgpu_shaders::specialize_wgsl(src, use_subgroups);
        compile_pipeline(device, label, &specialized, bgl)
    };

    let matmul_q4_k = comp("matmul_q4_k", wgpu_shaders::SHADER_MATMUL_Q4_K, &bgl_matmul);
    let matmul_q4_k_b4 = comp("matmul_q4_k_b4", wgpu_shaders::SHADER_MATMUL_Q4_K_B4, &bgl_matmul);
    let matmul_q4_k_qkv = comp("matmul_q4_k_qkv", wgpu_shaders::SHADER_MATMUL_Q4_K_QKV, &bgl_qkv);
    let matmul_q4_k_qkv_b4 = comp("matmul_q4_k_qkv_b4", wgpu_shaders::SHADER_MATMUL_Q4_K_QKV_B4, &bgl_qkv);
    let matmul_q4_k_gate_up_swiglu = comp(
        "matmul_q4_k_gate_up_swiglu",
        wgpu_shaders::SHADER_MATMUL_Q4_K_GATE_UP_SWIGLU,
        &bgl_gate_up,
    );
    let matmul_q5_k = comp("matmul_q5_k", wgpu_shaders::SHADER_MATMUL_Q5_K, &bgl_matmul);
    let matmul_q6_k = comp("matmul_q6_k", wgpu_shaders::SHADER_MATMUL_Q6_K, &bgl_matmul);
    let matmul_q6_k_b4 = comp("matmul_q6_k_b4", wgpu_shaders::SHADER_MATMUL_Q6_K_B4, &bgl_matmul);
    let matmul_q4_0 = comp("matmul_q4_0", wgpu_shaders::SHADER_MATMUL_Q4_0, &bgl_matmul);
    let matmul_q8_0 = comp("matmul_q8_0", wgpu_shaders::SHADER_MATMUL_Q8_0, &bgl_matmul);
    let matmul_f16 = comp("matmul_f16", wgpu_shaders::SHADER_MATMUL_F16, &bgl_matmul);
    let matmul_f32 = comp("matmul_f32", wgpu_shaders::SHADER_MATMUL_F32, &bgl_matmul);

    let xmx = if has_xmx {
        use crate::spv_xmx::*;
        let spv_comp = |label: &str, spv: Vec<u32>, cfg: XmxTileCfg| {
            compile_spirv_pipeline(device, label, spv, cfg.wg_size(), &bgl_matmul)
        };
        Some(XmxPipelines {
            q4_k_m8: spv_comp("xmx_q4k_m8", spirv_xmx_q4_k(CFG_M8_N32_K128, true), CFG_M8_N32_K128),
            q4_k_m16: spv_comp("xmx_q4k_m16", spirv_xmx_q4_k(CFG_M16_N32_K64, true), CFG_M16_N32_K64),
            q4_k_m32: spv_comp("xmx_q4k_m32", spirv_xmx_q4_k(CFG_M32_N64_K64, true), CFG_M32_N64_K64),
            q5_k_m8: spv_comp("xmx_q5k_m8", spirv_xmx_q5_k(CFG_M8_N32_K128, true), CFG_M8_N32_K128),
            q5_k_m16: spv_comp("xmx_q5k_m16", spirv_xmx_q5_k(CFG_M16_N32_K128, true), CFG_M16_N32_K128),
            q6_k_m8: spv_comp("xmx_q6k_m8", spirv_xmx_q6_k(CFG_M8_N32_K128, true), CFG_M8_N32_K128),
            q6_k_m16: spv_comp("xmx_q6k_m16", spirv_xmx_q6_k(CFG_M16_N32_K128, true), CFG_M16_N32_K128),
            f16_m16: spv_comp("xmx_f16_m16", spirv_xmx_f16(CFG_M16_N32_K64, true), CFG_M16_N32_K64),
            f32_m16: spv_comp("xmx_f32_m16", spirv_xmx_f32(CFG_M16_N32_K64, true), CFG_M16_N32_K64),
        })
    } else {
        None
    };

    let rms_norm = comp("rms_norm", wgpu_shaders::SHADER_RMS_NORM, &bgl_rms_norm);
    let residual_rms_norm = comp(
        "residual_rms_norm",
        wgpu_shaders::SHADER_RESIDUAL_RMS_NORM,
        &bgl_res_rms,
    );
    let rms_norm_inplace = comp(
        "rms_norm_inplace",
        wgpu_shaders::SHADER_RMS_NORM_INPLACE,
        &bgl_rw_r_u,
    );
    let add_bias = compile_pipeline(device, "add_bias", wgpu_shaders::SHADER_ADD_BIAS, &bgl_rw_r_u);
    let rope_kv_store = compile_pipeline(
        device,
        "rope_kv_store_f16",
        wgpu_shaders::SHADER_ROPE_KV_STORE_F16,
        &bgl_rope_kv,
    );
    let rope_kv_store_slots = compile_pipeline(
        device,
        "rope_kv_store_slots_f16",
        wgpu_shaders::SHADER_ROPE_KV_STORE_SLOTS_F16,
        &bgl_rope_kv_slots,
    );
    let gqa_flash_attn = comp(
        "gqa_flash_attn_f16",
        wgpu_shaders::SHADER_GQA_FLASH_ATTN_F16,
        &bgl_gqa,
    );
    let gqa_flash_attn_slots = comp(
        "gqa_flash_attn_slots_f16",
        wgpu_shaders::SHADER_GQA_FLASH_ATTN_SLOTS_F16,
        &bgl_gqa_slots,
    );
    let swiglu = compile_pipeline(device, "swiglu", wgpu_shaders::SHADER_SWIGLU, &bgl_rw_r_u);
    let geglu = compile_pipeline(device, "geglu", wgpu_shaders::SHADER_GEGLU, &bgl_rw_r_u);
    let residual_add = compile_pipeline(
        device,
        "residual_add",
        wgpu_shaders::SHADER_RESIDUAL_ADD,
        &bgl_rw_r_u,
    );
    let residual_add_scale = compile_pipeline(
        device,
        "residual_add_scale",
        wgpu_shaders::SHADER_RESIDUAL_ADD_SCALE,
        &bgl_rw_r_u,
    );
    let mul_sigmoid = compile_pipeline(
        device,
        "mul_sigmoid",
        wgpu_shaders::SHADER_MUL_SIGMOID,
        &bgl_rw_r_u,
    );
    let deinterleave_qg = compile_pipeline(
        device,
        "deinterleave_qg",
        wgpu_shaders::SHADER_DEINTERLEAVE_QG,
        &bgl_deinterleave_qg,
    );
    let deltanet_conv_silu = compile_pipeline(
        device,
        "deltanet_conv_silu",
        wgpu_shaders::SHADER_DELTANET_CONV_SILU,
        &bgl_deltanet_conv,
    );
    let l2_norm_qk = comp("l2_norm_qk", wgpu_shaders::SHADER_L2_NORM_QK, &bgl_rw_r_u);
    let deltanet_step = compile_pipeline(
        device,
        "deltanet_step",
        wgpu_shaders::SHADER_DELTANET_STEP,
        &bgl_deltanet_step,
    );

    GpuPipelines {
        bgl_matmul,
        bgl_qkv,
        bgl_gate_up,
        bgl_rms_norm,
        bgl_res_rms,
        bgl_rw_r_u,
        bgl_rope_kv,
        bgl_rope_kv_slots,
        bgl_gqa,
        bgl_gqa_slots,
        bgl_deinterleave_qg,
        bgl_deltanet_conv,
        bgl_deltanet_step,
        matmul_q4_k,
        matmul_q4_k_b4,
        matmul_q4_k_qkv,
        matmul_q4_k_qkv_b4,
        matmul_q4_k_gate_up_swiglu,
        matmul_q5_k,
        matmul_q6_k,
        matmul_q6_k_b4,
        matmul_q4_0,
        matmul_q8_0,
        matmul_f16,
        matmul_f32,
        xmx,
        rms_norm,
        residual_rms_norm,
        rms_norm_inplace,
        add_bias,
        rope_kv_store,
        rope_kv_store_slots,
        gqa_flash_attn,
        gqa_flash_attn_slots,
        swiglu,
        geglu,
        residual_add,
        residual_add_scale,
        mul_sigmoid,
        deinterleave_qg,
        deltanet_conv_silu,
        l2_norm_qk,
        deltanet_step,
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::{Backend, CpuBackend};
    use caelum_core::{quantize_row_q4_0, quantize_row_q8_0, BlockQ4_0, BlockQ8_0, TensorBuf};

    #[test]
    fn test_aligned_q4_0_and_q8_0_gpu_parity() {
        let Ok(ctx) = GpuContext::new() else { return; };
        let cpu = CpuBackend::new(Some(2));
        let (rows, cols) = (73usize, 256usize);
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 29) % 23) as f32 - 11.0) * 0.04)
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|i| (((i * 17) % 13) as f32 - 6.0) * 0.08)
            .collect();

        // Q4_0 aligned 20B/block parity
        let mut q4 = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; (rows * cols) / 32];
        for r in 0..rows {
            quantize_row_q4_0(&w_f32[r * cols..(r + 1) * cols], &mut q4[r * (cols / 32)..(r + 1) * (cols / 32)]);
        }
        let q4_bytes = unsafe {
            std::slice::from_raw_parts(q4.as_ptr() as *const u8, q4.len() * std::mem::size_of::<BlockQ4_0>())
        };
        let buf_q4 = TensorBuf::from_raw(q4_bytes.to_vec(), vec![rows, cols], DType::Q4_0).unwrap();
        let (_, gpu_q4) = ctx.bench_matmul(&buf_q4.as_view(), &x, 1, 1).unwrap();
        let mut cpu_q4 = vec![0.0f32; rows];
        cpu.matvec(&buf_q4.as_view(), &x, &mut cpu_q4).unwrap();
        let err_q4 = gpu_q4.iter().zip(&cpu_q4).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err_q4 < 2e-3, "aligned Q4_0 GPU mismatch err={err_q4}");

        // Q8_0 aligned 36B/block parity
        let mut q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (rows * cols) / 32];
        for r in 0..rows {
            quantize_row_q8_0(&w_f32[r * cols..(r + 1) * cols], &mut q8[r * (cols / 32)..(r + 1) * (cols / 32)]);
        }
        let q8_bytes = unsafe {
            std::slice::from_raw_parts(q8.as_ptr() as *const u8, q8.len() * std::mem::size_of::<BlockQ8_0>())
        };
        let buf_q8 = TensorBuf::from_raw(q8_bytes.to_vec(), vec![rows, cols], DType::Q8_0).unwrap();
        let (_, gpu_q8) = ctx.bench_matmul(&buf_q8.as_view(), &x, 1, 1).unwrap();
        let mut cpu_q8 = vec![0.0f32; rows];
        cpu.matvec(&buf_q8.as_view(), &x, &mut cpu_q8).unwrap();
        let err_q8 = gpu_q8.iter().zip(&cpu_q8).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err_q8 < 2e-3, "aligned Q8_0 GPU mismatch err={err_q8}");
    }
}
