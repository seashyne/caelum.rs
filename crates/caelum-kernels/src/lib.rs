//! Hardware-dispatched CPU SIMD kernels (ARMv9 NEON/DotProd + x86_64 AVX2/FMA + Scalar)
//! and persistent multi-threaded execution pool for Caelum.

pub mod cpu_caps;
pub mod pool;
pub mod repack;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod armv9;

use caelum_core::{
    quantize_row_q8_0, quantize_row_q8_k, BlockQ8_0, BlockQ8_K, CaelumError, DType, Result,
    TensorView, QK8_0, QK_K,
};
pub use cpu_caps::CpuCaps;
pub use pool::ThreadPool;
pub use repack::{PackedMatrix, PackedTiles};

/// RoPE pairing convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Interleaved pairs `(2i, 2i+1)` — used by Llama 2/3, Mistral.
    Normal,
    /// Half-split pairs `(i, i + head_dim/2)` — used by Qwen2, Qwen2.5, Celestia.
    NeoX,
}

/// Pluggable compute backend trait.
///
/// Allows dropping in external micro-kernels (such as Arm KleidiAI via FFI or `wgpu` GPU kernels)
/// without modifying any model code in `caelum-models`.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn num_threads(&self) -> usize;

    /// Computes `out = W * x` where `W` has shape `[out_features, in_features]` and `x` has length `in_features`.
    fn matvec(&self, w: &TensorView<'_>, x: &[f32], out: &mut [f32]) -> Result<()>;

    /// Computes `out = W * x` using pre-quantized `Q8_0` activations when supported (W4A8 / W8A8 fast path).
    fn matvec_q8_act(
        &self,
        w: &TensorView<'_>,
        x_f32: &[f32],
        x_q8: &[BlockQ8_0],
        out: &mut [f32],
    ) -> Result<()>;

    /// Computes `out = W * x` using pre-quantized `Q8_K` super-block activations (SIMD Q4_K / Q6_K fast path).
    fn matvec_q8k_act(
        &self,
        w: &TensorView<'_>,
        x_f32: &[f32],
        x_q8k: &[BlockQ8_K],
        out: &mut [f32],
    ) -> Result<()>;

    /// Batched Prefill GEMM: computes `out_batch[t, r] = dot(W[r], x_batch[t])` for `batch` tokens.
    ///
    /// Both `x_batch` (`[batch, cols]`) and `out_batch` (`[batch, rows]`) are token-major contiguous.
    /// Loads each weight row ONCE from DRAM and reuses it across all `batch` tokens.
    fn matmul(
        &self,
        w: &TensorView<'_>,
        x_batch: &[f32],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()>;

    /// Batched Prefill GEMM reusing pre-quantized `Q8_K` token activations (`[batch * (cols/256)]`).
    fn matmul_q8k_act(
        &self,
        w: &TensorView<'_>,
        x_batch_f32: &[f32],
        x_batch_q8k: &[BlockQ8_K],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()>;

    /// Batched Prefill GEMM reusing pre-quantized `Q8_0` token activations (`[batch * (cols/32)]`).
    fn matmul_q8_act(
        &self,
        w: &TensorView<'_>,
        x_batch_f32: &[f32],
        x_batch_q8: &[BlockQ8_0],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()>;

    /// Dot product of two `Q8_0` block slices (used for quantized KV-cache attention).
    fn dot_q8_0(&self, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32;

    /// Parallelizes an index range `0..total` across the backend's lock-free spin-barrier thread pool.
    fn parallel_for(&self, total: usize, f: &(dyn Fn(usize, usize) + Sync));

    /// RMSNorm: `out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// Applies RoPE in-place to a single head vector at sequence position `pos`.
    fn apply_rope(
        &self,
        vec: &mut [f32],
        pos: usize,
        rope_dim: usize,
        theta: f32,
        style: RopeStyle,
    );

    /// In-place numerically stable softmax.
    fn softmax(&self, x: &mut [f32]);

    /// SwiGLU elementwise fuse: `gate[i] = silu(gate[i]) * up[i]`.
    fn swiglu_inplace(&self, gate: &mut [f32], up: &[f32]);

    /// Dot product of two `f32` slices (used for attention scores).
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Repacks a `Q4_K`/`Q6_K` matrix into cache-aligned 8-row SIMD tiles when beneficial.
    fn pack_matrix(&self, _w: &TensorView<'_>) -> Option<PackedMatrix> {
        None
    }

    /// 8-row repacked GEMV (`Q4_K`/`Q6_K` × `Q8_K`) with zero horizontal reductions.
    fn matvec_packed_q8k(
        &self,
        w: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        out: &mut [f32],
    ) -> Result<()>;

    /// 8×4 / 8×8 register-tiled repacked Prefill GEMM (`Q4_K`/`Q6_K` × `Q8_K`).
    fn matmul_packed_q8k(
        &self,
        w: &PackedMatrix,
        x_batch_q8k: &[BlockQ8_K],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()>;

    /// Fused single-barrier QKV projection over repacked `Q4_K`/`Q6_K` matrices.
    #[allow(clippy::too_many_arguments)]
    fn fused_qkv_packed_q8k(
        &self,
        wq: &PackedMatrix,
        wk: &PackedMatrix,
        wv: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()>;

    /// Fused single-barrier Gate + Up + SwiGLU over repacked `Q4_K`/`Q6_K` matrices.
    fn fused_gate_up_swiglu_packed_q8k(
        &self,
        w_gate: &PackedMatrix,
        w_up: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        out_hb: &mut [f32],
    ) -> Result<()>;
}

/// Primary Caelum CPU backend with runtime SIMD dispatch and persistent worker pool.
pub struct CpuBackend {
    pub caps: CpuCaps,
    pool: ThreadPool,
}

impl CpuBackend {
    pub fn new(num_threads: Option<usize>) -> Self {
        let caps = CpuCaps::detect();
        let threads = num_threads.unwrap_or_else(pool::recommended_thread_count);
        Self {
            caps,
            pool: ThreadPool::new(threads),
        }
    }

    pub fn caps(&self) -> &CpuCaps {
        &self.caps
    }

    #[inline]
    fn row_dot_f32(&self, row: &[u8], x: &[f32], dtype: DType) -> f32 {
        match dtype {
            DType::F32 => {
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 && self.caps.fma {
                    return unsafe { x86::dot_f32_bytes_avx2(row, x) };
                }
                #[cfg(target_arch = "aarch64")]
                if self.caps.neon {
                    return unsafe { armv9::dot_f32_bytes_neon(row, x) };
                }
                scalar::dot_f32_bytes(row, x)
            }
            DType::F16 => scalar::dot_f16_bytes(row, x),
            DType::BF16 => scalar::dot_bf16_bytes(row, x),
            DType::Q4_0 => scalar::dot_q4_0_f32(row, x),
            DType::Q4_1 => scalar::dot_q4_1_f32(row, x),
            DType::Q8_0 => scalar::dot_q8_0_f32(row, x),
            DType::Q4_K => scalar::dot_q4_k_f32(row, x),
            DType::Q5_K => scalar::dot_q5_k_f32(row, x),
            DType::Q6_K => scalar::dot_q6_k_f32(row, x),
        }
    }

    #[inline]
    fn row_dot_q8(&self, row: &[u8], x_f32: &[f32], x_q8: &[BlockQ8_0], dtype: DType) -> f32 {
        match dtype {
            DType::Q4_0 => {
                #[cfg(target_arch = "aarch64")]
                if self.caps.dotprod {
                    return unsafe { armv9::dot_q4_0_q8_0_dotprod(row, x_q8) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 && self.caps.avxvnni {
                    return unsafe { x86::dot_q4_0_q8_0_vnni(row, x_q8) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 {
                    return unsafe { x86::dot_q4_0_q8_0_avx2(row, x_q8) };
                }
                scalar::dot_q4_0_q8_0(row, x_q8)
            }
            DType::Q8_0 => {
                #[cfg(target_arch = "aarch64")]
                if self.caps.dotprod {
                    return unsafe { armv9::dot_q8_0_q8_0_dotprod(row, x_q8) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 {
                    return unsafe { x86::dot_q8_0_q8_0_avx2(row, x_q8) };
                }
                scalar::dot_q8_0_q8_0(row, x_q8)
            }
            _ => self.row_dot_f32(row, x_f32, dtype),
        }
    }

    #[inline]
    fn row_dot_q8k(&self, row: &[u8], x_f32: &[f32], x_q8k: &[BlockQ8_K], dtype: DType) -> f32 {
        match dtype {
            DType::Q4_K => {
                #[cfg(target_arch = "aarch64")]
                if self.caps.dotprod {
                    return unsafe { armv9::dot_q4_k_q8_k_dotprod(row, x_q8k) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 {
                    return unsafe { x86::dot_q4_k_q8_k_avx2(row, x_q8k) };
                }
                scalar::dot_q4_k_q8_k(row, x_q8k)
            }
            DType::Q5_K => {
                #[cfg(target_arch = "aarch64")]
                if self.caps.dotprod {
                    return unsafe { armv9::dot_q5_k_q8_k_dotprod(row, x_q8k) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 {
                    return unsafe { x86::dot_q5_k_q8_k_avx2(row, x_q8k) };
                }
                scalar::dot_q5_k_q8_k(row, x_q8k)
            }
            DType::Q6_K => {
                #[cfg(target_arch = "aarch64")]
                if self.caps.dotprod {
                    return unsafe { armv9::dot_q6_k_q8_k_dotprod(row, x_q8k) };
                }
                #[cfg(target_arch = "x86_64")]
                if self.caps.avx2 {
                    return unsafe { x86::dot_q6_k_q8_k_avx2(row, x_q8k) };
                }
                scalar::dot_q6_k_q8_k(row, x_q8k)
            }
            _ => self.row_dot_f32(row, x_f32, dtype),
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        self.caps.best_kernel_name()
    }

    fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    fn matvec(&self, w: &TensorView<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let (rows, cols) = w.dims_2d()?;
        if x.len() != cols || out.len() != rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("W=[{rows},{cols}], x=[{cols}], out=[{rows}]"),
                actual: format!("x=[{}], out=[{}]", x.len(), out.len()),
            });
        }

        // Dynamic W4A8 / W8A8 / K-quant activation quantization for integer SIMD dot products
        if matches!(w.dtype, DType::Q4_0 | DType::Q8_0) && cols % QK8_0 == 0 {
            let mut x_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; cols / QK8_0];
            quantize_row_q8_0(x, &mut x_q8);
            return self.matvec_q8_act(w, x, &x_q8, out);
        }
        if matches!(w.dtype, DType::Q4_K | DType::Q5_K | DType::Q6_K) && cols % QK_K == 0 {
            let mut x_q8k = vec![BlockQ8_K::default(); cols / QK_K];
            quantize_row_q8_k(x, &mut x_q8k);
            return self.matvec_q8k_act(w, x, &x_q8k, out);
        }

        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;

        if rows >= 64 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_rows(out, |row_start, chunk| {
                for (i, dst) in chunk.iter_mut().enumerate() {
                    let r = row_start + i;
                    let row = &data[r * row_stride..(r + 1) * row_stride];
                    *dst = self.row_dot_f32(row, x, dtype);
                }
            });
        } else {
            for (r, dst) in out.iter_mut().enumerate() {
                let row = &data[r * row_stride..(r + 1) * row_stride];
                *dst = self.row_dot_f32(row, x, dtype);
            }
        }
        Ok(())
    }

    fn matvec_q8_act(
        &self,
        w: &TensorView<'_>,
        x_f32: &[f32],
        x_q8: &[BlockQ8_0],
        out: &mut [f32],
    ) -> Result<()> {
        let (rows, cols) = w.dims_2d()?;
        if x_f32.len() != cols || out.len() != rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("W=[{rows},{cols}], x=[{cols}], out=[{rows}]"),
                actual: format!("x=[{}], out=[{}]", x_f32.len(), out.len()),
            });
        }
        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;

        if rows >= 64 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_rows(out, |row_start, chunk| {
                for (i, dst) in chunk.iter_mut().enumerate() {
                    let r = row_start + i;
                    let row = &data[r * row_stride..(r + 1) * row_stride];
                    *dst = self.row_dot_q8(row, x_f32, x_q8, dtype);
                }
            });
        } else {
            for (r, dst) in out.iter_mut().enumerate() {
                let row = &data[r * row_stride..(r + 1) * row_stride];
                *dst = self.row_dot_q8(row, x_f32, x_q8, dtype);
            }
        }
        Ok(())
    }

    fn matvec_q8k_act(
        &self,
        w: &TensorView<'_>,
        x_f32: &[f32],
        x_q8k: &[BlockQ8_K],
        out: &mut [f32],
    ) -> Result<()> {
        let (rows, cols) = w.dims_2d()?;
        if x_f32.len() != cols || out.len() != rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("W=[{rows},{cols}], x=[{cols}], out=[{rows}]"),
                actual: format!("x=[{}], out=[{}]", x_f32.len(), out.len()),
            });
        }
        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;

        let compute_chunk = |row_start: usize, chunk: &mut [f32]| {
            let mut i = 0usize;
            #[cfg(target_arch = "x86_64")]
            if self.caps.avx2 {
                if dtype == DType::Q4_K {
                    while i + 2 <= chunk.len() {
                        let r = row_start + i;
                        let row0 = &data[r * row_stride..(r + 1) * row_stride];
                        let row1 = &data[(r + 1) * row_stride..(r + 2) * row_stride];
                        let pair = unsafe { x86::dot_q4_k_q8_k_2row_avx2(row0, row1, x_q8k) };
                        chunk[i] = pair[0];
                        chunk[i + 1] = pair[1];
                        i += 2;
                    }
                } else if dtype == DType::Q6_K {
                    while i + 2 <= chunk.len() {
                        let r = row_start + i;
                        let row0 = &data[r * row_stride..(r + 1) * row_stride];
                        let row1 = &data[(r + 1) * row_stride..(r + 2) * row_stride];
                        let pair = unsafe { x86::dot_q6_k_q8_k_2row_avx2(row0, row1, x_q8k) };
                        chunk[i] = pair[0];
                        chunk[i + 1] = pair[1];
                        i += 2;
                    }
                }
            }
            while i < chunk.len() {
                let r = row_start + i;
                let row = &data[r * row_stride..(r + 1) * row_stride];
                chunk[i] = self.row_dot_q8k(row, x_f32, x_q8k, dtype);
                i += 1;
            }
        };

        if rows >= 64 && self.pool.num_threads() > 1 {
            if rows >= 1024 {
                let out_addr = out.as_mut_ptr() as usize;
                self.pool.parallel_for_dynamic(rows, 128, |start, end| {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_addr as *mut f32).add(start),
                            end - start,
                        )
                    };
                    compute_chunk(start, slice);
                });
            } else {
                self.pool.parallel_for_rows(out, compute_chunk);
            }
        } else {
            compute_chunk(0, out);
        }
        Ok(())
    }

    fn matmul(
        &self,
        w: &TensorView<'_>,
        x_batch: &[f32],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()> {
        if batch == 1 {
            return self.matvec(w, x_batch, out_batch);
        }
        let (rows, cols) = w.dims_2d()?;
        if x_batch.len() != batch * cols || out_batch.len() != batch * rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("W=[{rows},{cols}], x=[{batch},{cols}], out=[{batch},{rows}]"),
                actual: format!("x.len={}, out.len={}", x_batch.len(), out_batch.len()),
            });
        }

        if matches!(w.dtype, DType::Q4_K | DType::Q5_K | DType::Q6_K) && cols % QK_K == 0 {
            let nb = cols / QK_K;
            let mut x_q8k = vec![BlockQ8_K::default(); batch * nb];
            for t in 0..batch {
                quantize_row_q8_k(
                    &x_batch[t * cols..(t + 1) * cols],
                    &mut x_q8k[t * nb..(t + 1) * nb],
                );
            }
            return self.matmul_q8k_act(w, x_batch, &x_q8k, batch, out_batch);
        }

        if matches!(w.dtype, DType::Q4_0 | DType::Q8_0) && cols % QK8_0 == 0 {
            let nb = cols / QK8_0;
            let mut x_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; batch * nb];
            for t in 0..batch {
                quantize_row_q8_0(
                    &x_batch[t * cols..(t + 1) * cols],
                    &mut x_q8[t * nb..(t + 1) * nb],
                );
            }
            return self.matmul_q8_act(w, x_batch, &x_q8, batch, out_batch);
        }

        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;
        let out_addr = out_batch.as_mut_ptr() as usize;

        let compute_rows = |r_start: usize, r_end: usize| {
            let out_ptr = out_addr as *mut f32;
            for r in r_start..r_end {
                let row = &data[r * row_stride..(r + 1) * row_stride];
                for t in 0..batch {
                    let xt = &x_batch[t * cols..(t + 1) * cols];
                    let val = self.row_dot_f32(row, xt, dtype);
                    unsafe {
                        *out_ptr.add(t * rows + r) = val;
                    }
                }
            }
        };

        if rows >= 64 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_range(rows, compute_rows);
        } else {
            compute_rows(0, rows);
        }
        Ok(())
    }

    fn matmul_q8k_act(
        &self,
        w: &TensorView<'_>,
        x_batch_f32: &[f32],
        x_batch_q8k: &[BlockQ8_K],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()> {
        if batch == 1 {
            return self.matvec_q8k_act(w, x_batch_f32, x_batch_q8k, out_batch);
        }
        let (rows, cols) = w.dims_2d()?;
        let nb = cols / QK_K;
        if x_batch_f32.len() != batch * cols
            || x_batch_q8k.len() != batch * nb
            || out_batch.len() != batch * rows
        {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("W=[{rows},{cols}], batch={batch}"),
                actual: format!(
                    "x_f32={}, x_q8k={}, out={}",
                    x_batch_f32.len(),
                    x_batch_q8k.len(),
                    out_batch.len()
                ),
            });
        }

        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;
        let out_addr = out_batch.as_mut_ptr() as usize;

        let compute_rows = |r_start: usize, r_end: usize| {
            let out_ptr = out_addr as *mut f32;
            let mut r = r_start;

            // ARMv9 I8MM (SMMLA) 2-row x 2-token hardware matrix tile fast path for Q4_K
            #[cfg(target_arch = "aarch64")]
            if dtype == DType::Q4_K && self.caps.i8mm {
                while r + 2 <= r_end {
                    let row0 = &data[r * row_stride..(r + 1) * row_stride];
                    let row1 = &data[(r + 1) * row_stride..(r + 2) * row_stride];
                    let mut t = 0usize;
                    while t + 2 <= batch {
                        let x0 = &x_batch_q8k[t * nb..(t + 1) * nb];
                        let x1 = &x_batch_q8k[(t + 1) * nb..(t + 2) * nb];
                        let tile = unsafe { armv9::dot_q4_k_q8_k_2x2_i8mm(row0, row1, x0, x1) };
                        unsafe {
                            *out_ptr.add(t * rows + r) = tile[0][0];
                            *out_ptr.add((t + 1) * rows + r) = tile[0][1];
                            *out_ptr.add(t * rows + (r + 1)) = tile[1][0];
                            *out_ptr.add((t + 1) * rows + (r + 1)) = tile[1][1];
                        }
                        t += 2;
                    }
                    if t < batch {
                        let xt_f = &x_batch_f32[t * cols..(t + 1) * cols];
                        let xt_q = &x_batch_q8k[t * nb..(t + 1) * nb];
                        unsafe {
                            *out_ptr.add(t * rows + r) = self.row_dot_q8k(row0, xt_f, xt_q, dtype);
                            *out_ptr.add(t * rows + (r + 1)) =
                                self.row_dot_q8k(row1, xt_f, xt_q, dtype);
                        }
                    }
                    r += 2;
                }
            }

            while r < r_end {
                let row = &data[r * row_stride..(r + 1) * row_stride];
                let mut t = 0usize;

                // 8-token-at-once AVX2 weight-reuse fast path for Q4_K prefill
                #[cfg(target_arch = "x86_64")]
                if dtype == DType::Q4_K && self.caps.avx2 {
                    while t + 8 <= batch {
                        let xs = [
                            &x_batch_q8k[t * nb..(t + 1) * nb],
                            &x_batch_q8k[(t + 1) * nb..(t + 2) * nb],
                            &x_batch_q8k[(t + 2) * nb..(t + 3) * nb],
                            &x_batch_q8k[(t + 3) * nb..(t + 4) * nb],
                            &x_batch_q8k[(t + 4) * nb..(t + 5) * nb],
                            &x_batch_q8k[(t + 5) * nb..(t + 6) * nb],
                            &x_batch_q8k[(t + 6) * nb..(t + 7) * nb],
                            &x_batch_q8k[(t + 7) * nb..(t + 8) * nb],
                        ];
                        let v = unsafe { x86::dot_q4_k_q8_k_batch8_avx2(row, xs) };
                        unsafe {
                            for k in 0..8 {
                                *out_ptr.add((t + k) * rows + r) = v[k];
                            }
                        }
                        t += 8;
                    }
                }

                // 4-token-at-once weight-reuse micro-kernels (AVX2 / ARMv9 DotProd)
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                while t + 4 <= batch {
                    let x0 = &x_batch_q8k[t * nb..(t + 1) * nb];
                    let x1 = &x_batch_q8k[(t + 1) * nb..(t + 2) * nb];
                    let x2 = &x_batch_q8k[(t + 2) * nb..(t + 3) * nb];
                    let x3 = &x_batch_q8k[(t + 3) * nb..(t + 4) * nb];

                    let vals: Option<[f32; 4]> = match dtype {
                        DType::Q4_K => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                if self.caps.avx2 {
                                    Some(unsafe { x86::dot_q4_k_q8_k_batch4_avx2(row, x0, x1, x2, x3) })
                                } else {
                                    None
                                }
                            }
                            #[cfg(target_arch = "aarch64")]
                            {
                                if self.caps.dotprod {
                                    Some(unsafe {
                                        armv9::dot_q4_k_q8_k_batch4_dotprod(row, x0, x1, x2, x3)
                                    })
                                } else {
                                    None
                                }
                            }
                        }
                        DType::Q5_K => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                if self.caps.avx2 {
                                    Some(unsafe { x86::dot_q5_k_q8_k_batch4_avx2(row, x0, x1, x2, x3) })
                                } else {
                                    None
                                }
                            }
                            #[cfg(target_arch = "aarch64")]
                            {
                                if self.caps.dotprod {
                                    Some(unsafe {
                                        armv9::dot_q5_k_q8_k_batch4_dotprod(row, x0, x1, x2, x3)
                                    })
                                } else {
                                    None
                                }
                            }
                        }
                        DType::Q6_K => {
                            #[cfg(target_arch = "x86_64")]
                            {
                                if self.caps.avx2 {
                                    Some(unsafe { x86::dot_q6_k_q8_k_batch4_avx2(row, x0, x1, x2, x3) })
                                } else {
                                    None
                                }
                            }
                            #[cfg(target_arch = "aarch64")]
                            {
                                if self.caps.dotprod {
                                    Some(unsafe {
                                        armv9::dot_q6_k_q8_k_batch4_dotprod(row, x0, x1, x2, x3)
                                    })
                                } else {
                                    None
                                }
                            }
                        }
                        _ => None,
                    };

                    if let Some(v) = vals {
                        unsafe {
                            *out_ptr.add(t * rows + r) = v[0];
                            *out_ptr.add((t + 1) * rows + r) = v[1];
                            *out_ptr.add((t + 2) * rows + r) = v[2];
                            *out_ptr.add((t + 3) * rows + r) = v[3];
                        }
                        t += 4;
                    } else {
                        break;
                    }
                }

                while t < batch {
                    let xt_f = &x_batch_f32[t * cols..(t + 1) * cols];
                    let xt_q = &x_batch_q8k[t * nb..(t + 1) * nb];
                    let val = self.row_dot_q8k(row, xt_f, xt_q, dtype);
                    unsafe {
                        *out_ptr.add(t * rows + r) = val;
                    }
                    t += 1;
                }
                r += 1;
            }
        };

        if rows >= 64 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_range(rows, compute_rows);
        } else {
            compute_rows(0, rows);
        }
        Ok(())
    }

    fn matmul_q8_act(
        &self,
        w: &TensorView<'_>,
        x_batch_f32: &[f32],
        x_batch_q8: &[BlockQ8_0],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()> {
        if batch == 1 {
            return self.matvec_q8_act(w, x_batch_f32, x_batch_q8, out_batch);
        }
        let (rows, cols) = w.dims_2d()?;
        let nb = cols / QK8_0;
        let row_stride = (cols / w.dtype.block_size()) * w.dtype.type_size();
        let data = w.data;
        let dtype = w.dtype;
        let out_addr = out_batch.as_mut_ptr() as usize;

        let compute_rows = |r_start: usize, r_end: usize| {
            let out_ptr = out_addr as *mut f32;
            for r in r_start..r_end {
                let row = &data[r * row_stride..(r + 1) * row_stride];
                for t in 0..batch {
                    let xt_f = &x_batch_f32[t * cols..(t + 1) * cols];
                    let xt_q = &x_batch_q8[t * nb..(t + 1) * nb];
                    let val = self.row_dot_q8(row, xt_f, xt_q, dtype);
                    unsafe {
                        *out_ptr.add(t * rows + r) = val;
                    }
                }
            }
        };

        if rows >= 64 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_range(rows, compute_rows);
        } else {
            compute_rows(0, rows);
        }
        Ok(())
    }

    fn dot_q8_0(&self, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
        let raw_a = unsafe {
            std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * std::mem::size_of::<BlockQ8_0>())
        };
        #[cfg(target_arch = "aarch64")]
        if self.caps.dotprod {
            return unsafe { armv9::dot_q8_0_q8_0_dotprod(raw_a, b) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            return unsafe { x86::dot_q8_0_q8_0_avx2(raw_a, b) };
        }
        scalar::dot_q8_0_q8_0(raw_a, b)
    }

    fn parallel_for(&self, total: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        if total > 1 && self.pool.num_threads() > 1 {
            self.pool.parallel_for_range(total, |s, e| f(s, e));
        } else {
            f(0, total);
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        scalar::rms_norm(x, weight, eps, out);
    }

    fn apply_rope(
        &self,
        vec: &mut [f32],
        pos: usize,
        rope_dim: usize,
        theta: f32,
        style: RopeStyle,
    ) {
        scalar::apply_rope(vec, pos, rope_dim, theta, style);
    }

    fn softmax(&self, x: &mut [f32]) {
        scalar::softmax(x);
    }

    fn swiglu_inplace(&self, gate: &mut [f32], up: &[f32]) {
        scalar::swiglu_inplace(gate, up);
    }

    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && self.caps.fma {
            return unsafe { x86::dot_f32_avx2(a, b) };
        }
        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            return unsafe { armv9::dot_f32_neon(a, b) };
        }
        scalar::dot_f32(a, b)
    }

    fn pack_matrix(&self, w: &TensorView<'_>) -> Option<PackedMatrix> {
        PackedMatrix::try_pack(w, &self.caps, &self.pool)
    }

    fn matvec_packed_q8k(
        &self,
        w: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        out: &mut [f32],
    ) -> Result<()> {
        if x_q8k.len() != w.nb || out.len() != w.rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("Packed=[{},{}], x_nb={}", w.rows, w.cols, w.nb),
                actual: format!("x_nb={}, out={}", x_q8k.len(), out.len()),
            });
        }
        let groups = w.num_groups();
        let out_addr = out.as_mut_ptr() as usize;
        let caps = &self.caps;

        let work = |g_start: usize, g_end: usize| {
            let out_ptr = out_addr as *mut f32;
            for g in g_start..g_end {
                let mut v8 = [0.0f32; 8];
                w.compute_group_1tok(g, x_q8k, caps, &mut v8);
                unsafe {
                    std::ptr::copy_nonoverlapping(v8.as_ptr(), out_ptr.add(g * 8), 8);
                }
            }
        };

        if groups >= 8 && self.pool.num_threads() > 1 {
            let chunk = (groups / (self.pool.num_threads() * 4)).clamp(4, 32);
            self.pool.parallel_for_dynamic(groups, chunk, work);
        } else {
            work(0, groups);
        }
        Ok(())
    }

    fn matmul_packed_q8k(
        &self,
        w: &PackedMatrix,
        x_batch_q8k: &[BlockQ8_K],
        batch: usize,
        out_batch: &mut [f32],
    ) -> Result<()> {
        if batch == 1 {
            return self.matvec_packed_q8k(w, x_batch_q8k, out_batch);
        }
        let rows = w.rows;
        let nb = w.nb;
        if x_batch_q8k.len() != batch * nb || out_batch.len() != batch * rows {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("Packed=[{rows},{}], batch={batch}", w.cols),
                actual: format!("x_q8k={}, out={}", x_batch_q8k.len(), out_batch.len()),
            });
        }

        let groups = w.num_groups();
        let out_addr = out_batch.as_mut_ptr() as usize;
        let caps = &self.caps;
        let is_q6k = matches!(w.tiles, PackedTiles::Q6Kx8(_));

        let work = |g_start: usize, g_end: usize| {
            let out_ptr = out_addr as *mut f32;
            for g in g_start..g_end {
                let r_base = g * 8;
                let mut t = 0usize;

                if is_q6k {
                    while t + 8 <= batch {
                        let xs = [
                            &x_batch_q8k[t * nb..(t + 1) * nb],
                            &x_batch_q8k[(t + 1) * nb..(t + 2) * nb],
                            &x_batch_q8k[(t + 2) * nb..(t + 3) * nb],
                            &x_batch_q8k[(t + 3) * nb..(t + 4) * nb],
                            &x_batch_q8k[(t + 4) * nb..(t + 5) * nb],
                            &x_batch_q8k[(t + 5) * nb..(t + 6) * nb],
                            &x_batch_q8k[(t + 6) * nb..(t + 7) * nb],
                            &x_batch_q8k[(t + 7) * nb..(t + 8) * nb],
                        ];
                        let mut tile_out = [[0.0f32; 8]; 8];
                        w.compute_group_8tok_q6k(g, xs, caps, &mut tile_out);
                        unsafe {
                            for k in 0..8 {
                                std::ptr::copy_nonoverlapping(
                                    tile_out[k].as_ptr(),
                                    out_ptr.add((t + k) * rows + r_base),
                                    8,
                                );
                            }
                        }
                        t += 8;
                    }
                }

                while t + 4 <= batch {
                    let xs = [
                        &x_batch_q8k[t * nb..(t + 1) * nb],
                        &x_batch_q8k[(t + 1) * nb..(t + 2) * nb],
                        &x_batch_q8k[(t + 2) * nb..(t + 3) * nb],
                        &x_batch_q8k[(t + 3) * nb..(t + 4) * nb],
                    ];
                    let mut tile_out = [[0.0f32; 8]; 4];
                    w.compute_group_4tok(g, xs, caps, &mut tile_out);
                    unsafe {
                        for k in 0..4 {
                            std::ptr::copy_nonoverlapping(
                                tile_out[k].as_ptr(),
                                out_ptr.add((t + k) * rows + r_base),
                                8,
                            );
                        }
                    }
                    t += 4;
                }

                while t < batch {
                    let xt = &x_batch_q8k[t * nb..(t + 1) * nb];
                    let mut v8 = [0.0f32; 8];
                    w.compute_group_1tok(g, xt, caps, &mut v8);
                    unsafe {
                        std::ptr::copy_nonoverlapping(v8.as_ptr(), out_ptr.add(t * rows + r_base), 8);
                    }
                    t += 1;
                }
            }
        };

        if groups >= 4 && self.pool.num_threads() > 1 {
            let chunk = (groups / (self.pool.num_threads() * 4)).clamp(2, 16);
            self.pool.parallel_for_dynamic(groups, chunk, work);
        } else {
            work(0, groups);
        }
        Ok(())
    }

    fn fused_qkv_packed_q8k(
        &self,
        wq: &PackedMatrix,
        wk: &PackedMatrix,
        wv: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()> {
        let gq = wq.num_groups();
        let gk = wk.num_groups();
        let gv = wv.num_groups();
        let total = gq + gk + gv;
        let q_addr = q.as_mut_ptr() as usize;
        let k_addr = k.as_mut_ptr() as usize;
        let v_addr = v.as_mut_ptr() as usize;
        let caps = &self.caps;

        let work = |start: usize, end: usize| {
            let q_ptr = q_addr as *mut f32;
            let k_ptr = k_addr as *mut f32;
            let v_ptr = v_addr as *mut f32;
            for idx in start..end {
                let mut v8 = [0.0f32; 8];
                if idx < gq {
                    wq.compute_group_1tok(idx, x_q8k, caps, &mut v8);
                    unsafe {
                        std::ptr::copy_nonoverlapping(v8.as_ptr(), q_ptr.add(idx * 8), 8);
                    }
                } else if idx < gq + gk {
                    let kg = idx - gq;
                    wk.compute_group_1tok(kg, x_q8k, caps, &mut v8);
                    unsafe {
                        std::ptr::copy_nonoverlapping(v8.as_ptr(), k_ptr.add(kg * 8), 8);
                    }
                } else {
                    let vg = idx - gq - gk;
                    wv.compute_group_1tok(vg, x_q8k, caps, &mut v8);
                    unsafe {
                        std::ptr::copy_nonoverlapping(v8.as_ptr(), v_ptr.add(vg * 8), 8);
                    }
                }
            }
        };

        if total >= 8 && self.pool.num_threads() > 1 {
            let chunk = (total / (self.pool.num_threads() * 4)).clamp(4, 24);
            self.pool.parallel_for_dynamic(total, chunk, work);
        } else {
            work(0, total);
        }
        Ok(())
    }

    fn fused_gate_up_swiglu_packed_q8k(
        &self,
        w_gate: &PackedMatrix,
        w_up: &PackedMatrix,
        x_q8k: &[BlockQ8_K],
        out_hb: &mut [f32],
    ) -> Result<()> {
        let groups = w_gate.num_groups();
        let out_addr = out_hb.as_mut_ptr() as usize;
        let caps = &self.caps;

        let work = |g_start: usize, g_end: usize| {
            let out_ptr = out_addr as *mut f32;
            for g in g_start..g_end {
                let mut g8 = [0.0f32; 8];
                let mut u8 = [0.0f32; 8];
                w_gate.compute_group_1tok(g, x_q8k, caps, &mut g8);
                w_up.compute_group_1tok(g, x_q8k, caps, &mut u8);
                for i in 0..8 {
                    let gv = g8[i];
                    let silu = gv / (1.0 + (-gv).exp());
                    g8[i] = silu * u8[i];
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(g8.as_ptr(), out_ptr.add(g * 8), 8);
                }
            }
        };

        if groups >= 8 && self.pool.num_threads() > 1 {
            let chunk = (groups / (self.pool.num_threads() * 4)).clamp(4, 32);
            self.pool.parallel_for_dynamic(groups, chunk, work);
        } else {
            work(0, groups);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caelum_core::{quantize_row_q4_0, BlockQ4_0, TensorBuf};

    #[test]
    fn test_simd_matches_scalar_f32_and_q4_0() {
        let backend = CpuBackend::new(Some(4));
        let rows = 128;
        let cols = 128;
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

        // F32 matvec check
        let buf = TensorBuf::from_f32(&w_f32, vec![rows, cols]).unwrap();
        let mut out_simd = vec![0.0f32; rows];
        backend.matvec(&buf.as_view(), &x, &mut out_simd).unwrap();
        for r in 0..rows {
            let expected = scalar::dot_f32(&w_f32[r * cols..(r + 1) * cols], &x);
            assert!(
                (out_simd[r] - expected).abs() < 1e-4,
                "F32 mismatch at row {r}: simd={}, expected={expected}",
                out_simd[r]
            );
        }

        // Q4_0 x Q8_0 (W4A8) SIMD vs scalar check
        let mut q4_blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; (rows * cols) / 32];
        for r in 0..rows {
            quantize_row_q4_0(
                &w_f32[r * cols..(r + 1) * cols],
                &mut q4_blocks[r * (cols / 32)..(r + 1) * (cols / 32)],
            );
        }
        let raw_q4 = unsafe {
            std::slice::from_raw_parts(q4_blocks.as_ptr() as *const u8, q4_blocks.len() * 18)
        }
        .to_vec();
        let buf_q4 = TensorBuf::from_raw(raw_q4, vec![rows, cols], DType::Q4_0).unwrap();
        let mut out_q4 = vec![0.0f32; rows];
        backend.matvec(&buf_q4.as_view(), &x, &mut out_q4).unwrap();

        let mut x_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; cols / 32];
        quantize_row_q8_0(&x, &mut x_q8);
        for r in 0..rows {
            let expected = scalar::dot_q4_0_q8_0(buf_q4.as_view().row_bytes(r), &x_q8);
            assert!(
                (out_q4[r] - expected).abs() < 1e-4,
                "Q4_0 SIMD mismatch at row {r}: simd={}, scalar={expected}",
                out_q4[r]
            );
        }
    }

    #[test]
    fn test_simd_matches_scalar_q4_k_and_q6_k() {
        use caelum_core::f32_to_f16;
        let backend = CpuBackend::new(Some(4));
        let rows = 64;
        let cols = 512; // 2 super-blocks of 256

        // Build deterministic synthetic Q4_K blocks (144 bytes each)
        let mut raw_q4k = vec![0u8; rows * 2 * 144];
        for (i, b) in raw_q4k.iter_mut().enumerate() {
            *b = ((i * 37 + 13) & 0xFF) as u8;
        }
        // Set clean valid f16 scales d and dmin in each block
        for blk in 0..(rows * 2) {
            let off = blk * 144;
            let d = f32_to_f16(0.05).to_le_bytes();
            let dm = f32_to_f16(0.02).to_le_bytes();
            raw_q4k[off..off + 2].copy_from_slice(&d);
            raw_q4k[off + 2..off + 4].copy_from_slice(&dm);
        }

        // Build deterministic synthetic Q6_K blocks (210 bytes each)
        let mut raw_q6k = vec![0u8; rows * 2 * 210];
        for (i, b) in raw_q6k.iter_mut().enumerate() {
            *b = ((i * 53 + 7) & 0xFF) as u8;
        }
        for blk in 0..(rows * 2) {
            let off = blk * 210;
            let d = f32_to_f16(0.03).to_le_bytes();
            raw_q6k[off + 208..off + 210].copy_from_slice(&d);
        }

        let x: Vec<f32> = (0..cols).map(|i| ((i % 23) as f32 - 11.0) * 0.04).collect();
        let mut x_q8k = vec![BlockQ8_K::default(); cols / QK_K];
        quantize_row_q8_k(&x, &mut x_q8k);

        let buf_q4k = TensorBuf::from_raw(raw_q4k, vec![rows, cols], DType::Q4_K).unwrap();
        let mut out_q4k = vec![0.0f32; rows];
        backend.matvec(&buf_q4k.as_view(), &x, &mut out_q4k).unwrap();
        for r in 0..rows {
            let expected = scalar::dot_q4_k_q8_k(buf_q4k.as_view().row_bytes(r), &x_q8k);
            assert!(
                (out_q4k[r] - expected).abs() < 1e-3,
                "Q4_K SIMD mismatch at row {r}: simd={}, scalar={expected}",
                out_q4k[r]
            );
        }

        let buf_q6k = TensorBuf::from_raw(raw_q6k, vec![rows, cols], DType::Q6_K).unwrap();
        let mut out_q6k = vec![0.0f32; rows];
        backend.matvec(&buf_q6k.as_view(), &x, &mut out_q6k).unwrap();
        for r in 0..rows {
            let expected = scalar::dot_q6_k_q8_k(buf_q6k.as_view().row_bytes(r), &x_q8k);
            assert!(
                (out_q6k[r] - expected).abs() < 1e-3,
                "Q6_K SIMD mismatch at row {r}: simd={}, scalar={expected}",
                out_q6k[r]
            );
        }

        // Batched Prefill GEMM check (batch=11 covers 8-token + 4-token + tail fast paths)
        let batch = 11usize;
        let x_batch: Vec<f32> = (0..batch * cols)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        let mut out_batch_q4k = vec![0.0f32; batch * rows];
        backend
            .matmul(&buf_q4k.as_view(), &x_batch, batch, &mut out_batch_q4k)
            .unwrap();
        let mut out_batch_q6k = vec![0.0f32; batch * rows];
        backend
            .matmul(&buf_q6k.as_view(), &x_batch, batch, &mut out_batch_q6k)
            .unwrap();
        for t in 0..batch {
            let xt = &x_batch[t * cols..(t + 1) * cols];
            let mut ref4 = vec![0.0f32; rows];
            let mut ref6 = vec![0.0f32; rows];
            backend.matvec(&buf_q4k.as_view(), xt, &mut ref4).unwrap();
            backend.matvec(&buf_q6k.as_view(), xt, &mut ref6).unwrap();
            for r in 0..rows {
                assert!(
                    (out_batch_q4k[t * rows + r] - ref4[r]).abs() < 1e-3,
                    "Q4_K matmul mismatch at t={t}, r={r}"
                );
                assert!(
                    (out_batch_q6k[t * rows + r] - ref6[r]).abs() < 1e-3,
                    "Q6_K matmul mismatch at t={t}, r={r}"
                );
            }
        }

        // Verify Q5_K SIMD & batched matmul parity against scalar reference
        let mut raw_q5k = vec![0u8; rows * 2 * 176];
        for (i, b) in raw_q5k.iter_mut().enumerate() {
            *b = ((i * 41 + 19) & 0xFF) as u8;
        }
        for blk in 0..(rows * 2) {
            let off = blk * 176;
            raw_q5k[off..off + 2].copy_from_slice(&f32_to_f16(0.04).to_le_bytes());
            raw_q5k[off + 2..off + 4].copy_from_slice(&f32_to_f16(0.015).to_le_bytes());
        }
        let buf_q5k = TensorBuf::from_raw(raw_q5k, vec![rows, cols], DType::Q5_K).unwrap();
        let mut out_q5k = vec![0.0f32; rows];
        backend.matvec(&buf_q5k.as_view(), &x, &mut out_q5k).unwrap();
        for r in 0..rows {
            let expected = scalar::dot_q5_k_q8_k(buf_q5k.as_view().row_bytes(r), &x_q8k);
            assert!(
                (out_q5k[r] - expected).abs() < 1e-3,
                "Q5_K SIMD mismatch at row {r}: simd={}, scalar={expected}",
                out_q5k[r]
            );
        }
        let mut out_batch_q5k = vec![0.0f32; batch * rows];
        backend
            .matmul(&buf_q5k.as_view(), &x_batch, batch, &mut out_batch_q5k)
            .unwrap();
        for t in 0..batch {
            let xt = &x_batch[t * cols..(t + 1) * cols];
            let mut ref5 = vec![0.0f32; rows];
            backend.matvec(&buf_q5k.as_view(), xt, &mut ref5).unwrap();
            for r in 0..rows {
                assert!(
                    (out_batch_q5k[t * rows + r] - ref5[r]).abs() < 1e-3,
                    "Q5_K matmul mismatch at t={t}, r={r}"
                );
            }
        }

        // Verify PackedMatrix (TileQ4Kx8 & TileQ6Kx8) AVX-VNNI/AVX2 matvec, matmul & fused ops
        let nb = cols / QK_K;
        let mut x_batch_q8k = vec![BlockQ8_K::default(); batch * nb];
        for t in 0..batch {
            quantize_row_q8_k(
                &x_batch[t * cols..(t + 1) * cols],
                &mut x_batch_q8k[t * nb..(t + 1) * nb],
            );
        }

        if let Some(packed4) = backend.pack_matrix(&buf_q4k.as_view()) {
            let mut p_out = vec![0.0f32; rows];
            backend.matvec_packed_q8k(&packed4, &x_q8k, &mut p_out).unwrap();
            for r in 0..rows {
                assert!(
                    (p_out[r] - out_q4k[r]).abs() < 2e-3,
                    "Packed Q4_K matvec mismatch at r={r}: packed={}, raw={}",
                    p_out[r],
                    out_q4k[r]
                );
            }
            let mut pb_out = vec![0.0f32; batch * rows];
            backend
                .matmul_packed_q8k(&packed4, &x_batch_q8k, batch, &mut pb_out)
                .unwrap();
            for idx in 0..batch * rows {
                assert!(
                    (pb_out[idx] - out_batch_q4k[idx]).abs() < 2e-3,
                    "Packed Q4_K matmul mismatch at idx={idx}: packed={}, raw={}",
                    pb_out[idx],
                    out_batch_q4k[idx]
                );
            }
        }

        if let Some(packed6) = backend.pack_matrix(&buf_q6k.as_view()) {
            let mut p_out = vec![0.0f32; rows];
            backend.matvec_packed_q8k(&packed6, &x_q8k, &mut p_out).unwrap();
            for r in 0..rows {
                assert!(
                    (p_out[r] - out_q6k[r]).abs() < 2e-3,
                    "Packed Q6_K matvec mismatch at r={r}: packed={}, raw={}",
                    p_out[r],
                    out_q6k[r]
                );
            }
            let mut pb_out = vec![0.0f32; batch * rows];
            backend
                .matmul_packed_q8k(&packed6, &x_batch_q8k, batch, &mut pb_out)
                .unwrap();
            for idx in 0..batch * rows {
                assert!(
                    (pb_out[idx] - out_batch_q6k[idx]).abs() < 2e-3,
                    "Packed Q6_K matmul mismatch at idx={idx}: packed={}, raw={}",
                    pb_out[idx],
                    out_batch_q6k[idx]
                );
            }
        }
    }
}
