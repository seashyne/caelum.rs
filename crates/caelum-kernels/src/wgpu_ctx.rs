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
pub const UNIFORM_ARENA_SLOTS: usize = 1024; // 256 KB (> enough for 64+ layers per submit)

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

pub struct GpuPipelines {
    pub bgl_matmul: wgpu::BindGroupLayout,
    pub bgl_qkv: wgpu::BindGroupLayout,
    pub bgl_gate_up: wgpu::BindGroupLayout,
    pub bgl_rms_norm: wgpu::BindGroupLayout,
    pub bgl_res_rms: wgpu::BindGroupLayout,
    pub bgl_rw_r_u: wgpu::BindGroupLayout,
    pub bgl_rope_kv: wgpu::BindGroupLayout,
    pub bgl_gqa: wgpu::BindGroupLayout,

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

    pub rms_norm: wgpu::ComputePipeline,
    pub residual_rms_norm: wgpu::ComputePipeline,
    pub rms_norm_inplace: wgpu::ComputePipeline,
    pub add_bias: wgpu::ComputePipeline,
    pub rope_kv_store: wgpu::ComputePipeline,
    pub gqa_flash_attn: wgpu::ComputePipeline,
    pub swiglu: wgpu::ComputePipeline,
    pub residual_add: wgpu::ComputePipeline,
}

/// Shared `wgpu` GPU device, queue, adapter metadata, and pre-compiled WGSL compute pipelines.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub use_subgroups: bool,
    pub pipelines: GpuPipelines,
}

impl GpuContext {
    /// Initializes the highest-performance GPU adapter (Vulkan / Metal / DX12) and compiles
    /// all Caelum WGSL compute pipelines once.
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

        let mut req_features = wgpu::Features::empty();
        if use_subgroups {
            req_features |= wgpu::Features::SUBGROUP;
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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("caelum-wgpu-device"),
            required_features: req_features,
            required_limits: req_limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| CaelumError::Runtime(format!("failed to create wgpu device: {e}")))?;

        let pipelines = build_pipelines(&device, use_subgroups);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits: req_limits,
            use_subgroups,
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
        format!(
            "{} ({:?}, {:?}) | {sg} | max_buf={:.1} GB",
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

    /// Repacks GGML `Q6_K` (`210` bytes/block) into 4-byte-aligned GPU blocks (`212` bytes = `53 u32`/block)
    /// with pre-decoded `f32` block scale `d`, streaming in bounded `~13.8 MiB` CPU chunks via
    /// `queue.write_buffer` (fast contiguous DMA, zero write-combining stalls, <16 MiB peak RAM).
    pub fn upload_q6k_aligned(&self, label: &str, raw: &[u8]) -> wgpu::Buffer {
        debug_assert_eq!(raw.len() % 210, 0);
        let n_blocks = raw.len() / 210;
        let out_bytes = (n_blocks * 212).max(4);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if n_blocks == 0 {
            return buf;
        }
        const CHUNK_BLOCKS: usize = 65_536; // 65536 * 212 B = ~13.89 MiB
        let cap_blocks = n_blocks.min(CHUNK_BLOCKS);
        let mut scratch = vec![0u8; cap_blocks * 212];
        let mut blk_off = 0usize;
        while blk_off < n_blocks {
            let count = (n_blocks - blk_off).min(CHUNK_BLOCKS);
            for i in 0..count {
                let src = &raw[(blk_off + i) * 210..(blk_off + i + 1) * 210];
                let dst = &mut scratch[i * 212..(i + 1) * 212];
                dst[0..208].copy_from_slice(&src[0..208]);
                let d_f16 = u16::from_le_bytes([src[208], src[209]]);
                let d_f32 = f16_to_f32(d_f16);
                dst[208..212].copy_from_slice(&d_f32.to_le_bytes());
            }
            self.queue
                .write_buffer(&buf, (blk_off * 212) as u64, &scratch[..count * 212]);
            blk_off += count;
            if n_blocks > CHUNK_BLOCKS && blk_off < n_blocks {
                self.flush_uploads();
            }
        }
        buf
    }

    /// Uploads any `TensorView` weight matrix in its optimal VRAM format (aligned `Q6_K` or raw).
    pub fn upload_weight(&self, label: &str, view: &TensorView<'_>) -> (wgpu::Buffer, usize) {
        if view.dtype == DType::Q6_K {
            let aligned_bytes = (view.data.len() / 210) * 212;
            (self.upload_q6k_aligned(label, view.data), aligned_bytes)
        } else {
            (self.upload_raw_padded(label, view.data), view.data.len())
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

    /// Returns the appropriate GPU matmul compute pipeline for `dtype` and `batch` size.
    pub fn matmul_pipeline(&self, dtype: DType, batch: usize) -> Result<(&wgpu::ComputePipeline, bool)> {
        match dtype {
            DType::Q4_K => {
                if batch >= 4 {
                    Ok((&self.pipelines.matmul_q4_k_b4, true))
                } else {
                    Ok((&self.pipelines.matmul_q4_k, false))
                }
            }
            DType::Q5_K => Ok((&self.pipelines.matmul_q5_k, false)),
            DType::Q6_K => {
                if batch >= 4 {
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
        let (pipeline, is_b4) = self.matmul_pipeline(w.dtype, batch)?;
        let (w_buf, _) = self.upload_weight("bench-w", w);
        let x_buf = self.upload_f32("bench-x", x_batch);
        let out_bytes = (batch * rows * 4) as u64;
        let out_buf = self.alloc_storage("bench-out", out_bytes);
        let staging = self.alloc_staging_read("bench-staging", out_bytes);
        let mut arena = UniformArena::new(&self.device);
        let bg = self.bind_matmul(&w_buf, &x_buf, &out_buf, &arena);
        let u_off = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
        arena.flush(&self.queue);

        let wg_x = (rows as u32).div_ceil(4);
        let wg_y = if is_b4 {
            (batch as u32).div_ceil(4)
        } else {
            batch as u32
        };

        // Warmup 1 dispatch so shader JIT / clocks settle
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[u_off]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
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
                pass.dispatch_workgroups(wg_x, wg_y, 1);
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

fn build_pipelines(device: &wgpu::Device, use_subgroups: bool) -> GpuPipelines {
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
    let gqa_flash_attn = comp(
        "gqa_flash_attn_f16",
        wgpu_shaders::SHADER_GQA_FLASH_ATTN_F16,
        &bgl_gqa,
    );
    let swiglu = compile_pipeline(device, "swiglu", wgpu_shaders::SHADER_SWIGLU, &bgl_rw_r_u);
    let residual_add = compile_pipeline(
        device,
        "residual_add",
        wgpu_shaders::SHADER_RESIDUAL_ADD,
        &bgl_rw_r_u,
    );

    GpuPipelines {
        bgl_matmul,
        bgl_qkv,
        bgl_gate_up,
        bgl_rms_norm,
        bgl_res_rms,
        bgl_rw_r_u,
        bgl_rope_kv,
        bgl_gqa,
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
        rms_norm,
        residual_rms_norm,
        rms_norm_inplace,
        add_bias,
        rope_kv_store,
        gqa_flash_attn,
        swiglu,
        residual_add,
    }
}
