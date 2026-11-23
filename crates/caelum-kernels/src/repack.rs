//! Cache-aligned online weight repacking (`TileQ4Kx8`, `TileQ6Kx8`) and AVX-VNNI / AVX2+FMA
//! 8-row GEMV + 8×4 / 8×8 register-tiled Prefill GEMM kernels.
//!
//! Why repacking beats raw row-major `Q4_K`/`Q6_K`:
//! - Interleaves 8 weight rows in 4-byte groups so each 256-bit `vpdpbusd` computes 8 output
//!   rows in parallel with ZERO horizontal reductions (`hsum256`).
//! - Pre-unpacks 6-bit `Q4_K` scales/mins and 6-bit `Q6_K` quants once at model load.
//! - Uses single-instruction hardware `F16C` (`vcvtph2ps`) to expand 8 `f16` scales at once.
//! - Keeps 8×4 (`Q4_K`) and 8×8 (`Q6_K`) micro-tiles strictly within the 16 x86_64 YMM
//!   registers (zero stack spills) and inside the 48 KB L1 data cache across prompt batches.

use crate::pool::ThreadPool;
use crate::CpuCaps;
#[cfg(target_arch = "x86_64")]
use caelum_core::{quant::get_scale_min_k4, QK_K};
use caelum_core::{BlockQ8_K, DType, TensorView};

/// 8-row × 256-element interleaved `Q4_K` tile (`1,184` bytes, 32-byte aligned, +2.7% vs raw).
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct TileQ4Kx8 {
    /// Super-block `d` (IEEE-754 f16 bits) for the 8 rows.
    pub d_f16: [u16; 8],
    /// Super-block `dmin` (IEEE-754 f16 bits) for the 8 rows.
    pub dmin_f16: [u16; 8],
    /// Unpacked 6-bit sub-block scales `sc[sub][row]` (`sub ∈ 0..8`, `row ∈ 0..8`).
    pub sc: [[u8; 8]; 8],
    /// Unpacked 6-bit sub-block mins `m[sub][row]` (`sub ∈ 0..8`, `row ∈ 0..8`).
    pub m: [[u8; 8]; 8],
    /// Interleaved 4-bit nibbles: `4 chunks × 8 k-steps × 32 bytes`.
    /// Byte `(chunk * 8 + k_step) * 32 + row * 4 + i` = `orig_row[row].qs[chunk * 32 + k_step * 4 + i]`.
    pub qs: [u8; 1024],
}

impl Default for TileQ4Kx8 {
    fn default() -> Self {
        Self {
            d_f16: [0; 8],
            dmin_f16: [0; 8],
            sc: [[0; 8]; 8],
            m: [[0; 8]; 8],
            qs: [0; 1024],
        }
    }
}

/// 8-row × 256-element interleaved `Q6_K` tile with pre-unpacked unsigned 6-bit quants (`0..63`).
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct TileQ6Kx8 {
    /// Super-block `d` (IEEE-754 f16 bits) for the 8 rows.
    pub d_f16: [u16; 8],
    /// Padding to keep `sc` and `q` 32-byte aligned.
    pub _pad: [u16; 8],
    /// Signed `i8` sub-block scales `sc[sub][row]` (`sub ∈ 0..16`, `row ∈ 0..8`).
    pub sc: [[i8; 8]; 16],
    /// Pre-unpacked 6-bit unsigned quants `(q_signed + 32) ∈ 0..=63`:
    /// `16 sub-blocks × 4 k-steps × 32 bytes`.
    /// Byte `(sub * 4 + k_step) * 32 + row * 4 + i` = `q_u6(row, sub * 16 + k_step * 4 + i)`.
    pub q: [u8; 2048],
}

impl Default for TileQ6Kx8 {
    fn default() -> Self {
        Self {
            d_f16: [0; 8],
            _pad: [0; 8],
            sc: [[0; 8]; 16],
            q: [0; 2048],
        }
    }
}

/// Repacked weight matrix stored in cache-aligned 8-row SIMD tiles.
pub enum PackedTiles {
    Q4Kx8(Vec<TileQ4Kx8>),
    Q6Kx8(Vec<TileQ6Kx8>),
}

/// Owned repacked 2D weight matrix (`rows % 8 == 0`, `cols % 256 == 0`).
pub struct PackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nb: usize,
    pub tiles: PackedTiles,
}

impl PackedMatrix {
    /// Attempts to repack a 2D `Q4_K` or `Q6_K` weight matrix for AVX2+FMA / AVX-VNNI execution.
    /// Returns `None` on non-AVX2 platforms or incompatible shapes/dtypes.
    pub fn try_pack(view: &TensorView<'_>, caps: &CpuCaps, pool: &ThreadPool) -> Option<Self> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (view, caps, pool);
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if !(caps.avx2 && caps.fma) {
                return None;
            }
            let (rows, cols) = view.dims_2d().ok()?;
            if rows == 0 || rows % 8 != 0 || cols == 0 || cols % QK_K != 0 {
                return None;
            }
            match view.dtype {
                DType::Q4_K => Some(Self::pack_q4_k(view, rows, cols, pool)),
                DType::Q6_K => Some(Self::pack_q6_k(view, rows, cols, pool)),
                _ => None,
            }
        }
    }

    pub fn dtype(&self) -> DType {
        match self.tiles {
            PackedTiles::Q4Kx8(_) => DType::Q4_K,
            PackedTiles::Q6Kx8(_) => DType::Q6_K,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.rows / 8
    }

    pub fn byte_size(&self) -> usize {
        match &self.tiles {
            PackedTiles::Q4Kx8(v) => v.len() * std::mem::size_of::<TileQ4Kx8>(),
            PackedTiles::Q6Kx8(v) => v.len() * std::mem::size_of::<TileQ6Kx8>(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn pack_q4_k(view: &TensorView<'_>, rows: usize, cols: usize, pool: &ThreadPool) -> Self {
        let nb = cols / QK_K;
        let groups = rows / 8;
        let row_stride = nb * 144;
        let data = view.data;
        let mut tiles = vec![TileQ4Kx8::default(); groups * nb];
        let tiles_addr = tiles.as_mut_ptr() as usize;

        let work = |g_start: usize, g_end: usize| {
            let tiles_ptr = tiles_addr as *mut TileQ4Kx8;
            for g in g_start..g_end {
                let r_base = g * 8;
                for b in 0..nb {
                    let tile = unsafe { &mut *tiles_ptr.add(g * nb + b) };
                    for r in 0..8 {
                        let off = (r_base + r) * row_stride + b * 144;
                        tile.d_f16[r] = u16::from_le_bytes([data[off], data[off + 1]]);
                        tile.dmin_f16[r] = u16::from_le_bytes([data[off + 2], data[off + 3]]);
                        let scales = &data[off + 4..off + 16];
                        for sub in 0..8 {
                            let (sc, m) = get_scale_min_k4(sub, scales);
                            tile.sc[sub][r] = sc;
                            tile.m[sub][r] = m;
                        }
                        let qs = &data[off + 16..off + 144];
                        for chunk in 0..4 {
                            for k_step in 0..8 {
                                let dst_off = (chunk * 8 + k_step) * 32 + r * 4;
                                let src_off = chunk * 32 + k_step * 4;
                                tile.qs[dst_off..dst_off + 4]
                                    .copy_from_slice(&qs[src_off..src_off + 4]);
                            }
                        }
                    }
                }
            }
        };

        if groups >= 16 && pool.num_threads() > 1 {
            pool.parallel_for_range(groups, work);
        } else {
            work(0, groups);
        }

        Self {
            rows,
            cols,
            nb,
            tiles: PackedTiles::Q4Kx8(tiles),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn pack_q6_k(view: &TensorView<'_>, rows: usize, cols: usize, pool: &ThreadPool) -> Self {
        let nb = cols / QK_K;
        let groups = rows / 8;
        let row_stride = nb * 210;
        let data = view.data;
        let mut tiles = vec![TileQ6Kx8::default(); groups * nb];
        let tiles_addr = tiles.as_mut_ptr() as usize;

        let work = |g_start: usize, g_end: usize| {
            let tiles_ptr = tiles_addr as *mut TileQ6Kx8;
            for g in g_start..g_end {
                let r_base = g * 8;
                for b in 0..nb {
                    let tile = unsafe { &mut *tiles_ptr.add(g * nb + b) };
                    let mut tmp_q = [0u8; 256];
                    for r in 0..8 {
                        let off = (r_base + r) * row_stride + b * 210;
                        let ql_all = &data[off..off + 128];
                        let qh_all = &data[off + 128..off + 192];
                        let sc_all = &data[off + 192..off + 208];
                        tile.d_f16[r] = u16::from_le_bytes([data[off + 208], data[off + 209]]);

                        for sub in 0..16 {
                            tile.sc[sub][r] = sc_all[sub] as i8;
                        }
                        for half in 0..2 {
                            let ql = &ql_all[half * 64..(half + 1) * 64];
                            let qh = &qh_all[half * 32..(half + 1) * 32];
                            let base = half * 128;
                            for l in 0..32 {
                                tmp_q[base + l] = (ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4);
                                tmp_q[base + l + 32] =
                                    (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4);
                                tmp_q[base + l + 64] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                                tmp_q[base + l + 96] =
                                    (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                            }
                        }
                        for sub in 0..16 {
                            for k_step in 0..4 {
                                let dst_off = (sub * 4 + k_step) * 32 + r * 4;
                                let src_off = sub * 16 + k_step * 4;
                                tile.q[dst_off..dst_off + 4]
                                    .copy_from_slice(&tmp_q[src_off..src_off + 4]);
                            }
                        }
                    }
                }
            }
        };

        if groups >= 16 && pool.num_threads() > 1 {
            pool.parallel_for_range(groups, work);
        } else {
            work(0, groups);
        }

        Self {
            rows,
            cols,
            nb,
            tiles: PackedTiles::Q6Kx8(tiles),
        }
    }

    /// Computes 8 output rows for group `g` against 1 `Q8_K` activation vector (zero horizontal sums).
    #[inline]
    pub fn compute_group_1tok(&self, g: usize, x: &[BlockQ8_K], caps: &CpuCaps, out8: &mut [f32; 8]) {
        #[cfg(target_arch = "x86_64")]
        {
            match &self.tiles {
                PackedTiles::Q4Kx8(tiles) => {
                    let slice = &tiles[g * self.nb..(g + 1) * self.nb];
                    unsafe {
                        if caps.avxvnni && caps.f16c {
                            x86_packed::gemv_q4k_8x_vnni_f16c(slice, x, out8);
                        } else if caps.f16c {
                            x86_packed::gemv_q4k_8x_avx2_f16c(slice, x, out8);
                        } else {
                            x86_packed::gemv_q4k_8x_avx2(slice, x, out8);
                        }
                    }
                }
                PackedTiles::Q6Kx8(tiles) => {
                    let slice = &tiles[g * self.nb..(g + 1) * self.nb];
                    unsafe {
                        if caps.avxvnni && caps.f16c {
                            x86_packed::gemv_q6k_8x_vnni_f16c(slice, x, out8);
                        } else if caps.f16c {
                            x86_packed::gemv_q6k_8x_avx2_f16c(slice, x, out8);
                        } else {
                            x86_packed::gemv_q6k_8x_avx2(slice, x, out8);
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (g, x, caps, out8);
            unreachable!("PackedMatrix is only constructed on x86_64 AVX2+FMA");
        }
    }

    /// Computes 8 output rows × 4 tokens for group `g` (Prefill GEMM micro-tile).
    #[inline]
    pub fn compute_group_4tok(
        &self,
        g: usize,
        xs: [&[BlockQ8_K]; 4],
        caps: &CpuCaps,
        out: &mut [[f32; 8]; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            match &self.tiles {
                PackedTiles::Q4Kx8(tiles) => {
                    let slice = &tiles[g * self.nb..(g + 1) * self.nb];
                    unsafe {
                        if caps.avxvnni && caps.f16c {
                            x86_packed::gemm_q4k_8x4_vnni_f16c(slice, xs, out);
                        } else if caps.f16c {
                            x86_packed::gemm_q4k_8x4_avx2_f16c(slice, xs, out);
                        } else {
                            x86_packed::gemm_q4k_8x4_avx2(slice, xs, out);
                        }
                    }
                }
                PackedTiles::Q6Kx8(tiles) => {
                    let slice = &tiles[g * self.nb..(g + 1) * self.nb];
                    unsafe {
                        if caps.avxvnni && caps.f16c {
                            x86_packed::gemm_q6k_8x4_vnni_f16c(slice, xs, out);
                        } else if caps.f16c {
                            x86_packed::gemm_q6k_8x4_avx2_f16c(slice, xs, out);
                        } else {
                            x86_packed::gemm_q6k_8x4_avx2(slice, xs, out);
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (g, xs, caps, out);
            unreachable!("PackedMatrix is only constructed on x86_64 AVX2+FMA");
        }
    }

    /// Computes 8 output rows × 8 tokens for `Q6_K` group `g` (8× weight reuse, zero spills).
    #[inline]
    pub fn compute_group_8tok_q6k(
        &self,
        g: usize,
        xs: [&[BlockQ8_K]; 8],
        caps: &CpuCaps,
        out: &mut [[f32; 8]; 8],
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if let PackedTiles::Q6Kx8(tiles) = &self.tiles {
                let slice = &tiles[g * self.nb..(g + 1) * self.nb];
                unsafe {
                    if caps.avxvnni && caps.f16c {
                        x86_packed::gemm_q6k_8x8_vnni_f16c(slice, xs, out);
                    } else if caps.f16c {
                        x86_packed::gemm_q6k_8x8_avx2_f16c(slice, xs, out);
                    } else {
                        for t in 0..2 {
                            let mut sub_out = [[0.0f32; 8]; 4];
                            x86_packed::gemm_q6k_8x4_avx2(
                                slice,
                                [xs[t * 4], xs[t * 4 + 1], xs[t * 4 + 2], xs[t * 4 + 3]],
                                &mut sub_out,
                            );
                            out[t * 4..t * 4 + 4].copy_from_slice(&sub_out);
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (g, xs, caps, out);
            unreachable!("PackedMatrix is only constructed on x86_64 AVX2+FMA");
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_packed {
    use super::{TileQ4Kx8, TileQ6Kx8};
    use caelum_core::{f16_to_f32, BlockQ8_K};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vpdpbusd(mut acc: __m256i, u8_w: __m256i, i8_x: __m256i) -> __m256i {
        core::arch::asm!(
            "{{vex}} vpdpbusd {acc:y}, {w:y}, {x:y}",
            acc = inout(ymm_reg) acc,
            w = in(ymm_reg) u8_w,
            x = in(ymm_reg) i8_x,
            options(pure, nomem, nostack)
        );
        acc
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot4_avx2(acc: __m256i, u8_w: __m256i, i8_x: __m256i, ones_i16: __m256i) -> __m256i {
        let p16 = _mm256_maddubs_epi16(u8_w, i8_x);
        let p32 = _mm256_madd_epi16(p16, ones_i16);
        _mm256_add_epi32(acc, p32)
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn load_f16x8_hw(src: &[u16; 8]) -> __m256 {
        _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr() as *const __m128i))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_f16x8_sw(src: &[u16; 8]) -> __m256 {
        let arr = [
            f16_to_f32(src[0]),
            f16_to_f32(src[1]),
            f16_to_f32(src[2]),
            f16_to_f32(src[3]),
            f16_to_f32(src[4]),
            f16_to_f32(src[5]),
            f16_to_f32(src[6]),
            f16_to_f32(src[7]),
        ];
        _mm256_loadu_ps(arr.as_ptr())
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_u8x8_as_f32(src: &[u8; 8]) -> __m256 {
        let v64 = _mm_loadl_epi64(src.as_ptr() as *const __m128i);
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v64))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_i8x8_as_f32(src: &[i8; 8]) -> __m256 {
        let v64 = _mm_loadl_epi64(src.as_ptr() as *const __m128i);
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v64))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn bcast_i32(ptr: *const i8) -> __m256i {
        _mm256_set1_epi32(std::ptr::read_unaligned(ptr as *const i32))
    }

    // ==================== Q4_K 8-row GEMV (1 token decode) ====================

    macro_rules! impl_gemv_q4k_8x {
        ($fn_name:ident, $features:literal, $use_vnni:expr, $use_f16c:expr) => {
            #[target_feature(enable = $features)]
            pub unsafe fn $fn_name(tiles: &[TileQ4Kx8], x: &[BlockQ8_K], out8: &mut [f32; 8]) {
                let nb = tiles.len();
                let mask_0f = _mm256_set1_epi8(0x0F);
                let ones_i16 = _mm256_set1_epi16(1);
                let mut fsum = _mm256_setzero_ps();

                for b in 0..nb {
                    let tile = tiles.get_unchecked(b);
                    let xb = x.get_unchecked(b);

                    if b + 1 < nb {
                        let next_ptr = (tiles.as_ptr().add(b + 1)) as *const i8;
                        _mm_prefetch(next_ptr, _MM_HINT_T0);
                        _mm_prefetch(next_ptr.add(192), _MM_HINT_T0);
                        _mm_prefetch(next_ptr.add(512), _MM_HINT_T0);
                    }

                    let vd = if $use_f16c {
                        load_f16x8_hw(&tile.d_f16)
                    } else {
                        load_f16x8_sw(&tile.d_f16)
                    };
                    let vdmin = if $use_f16c {
                        load_f16x8_hw(&tile.dmin_f16)
                    } else {
                        load_f16x8_sw(&tile.dmin_f16)
                    };

                    let mut blk_sum = _mm256_setzero_ps();
                    let qs_ptr = tile.qs.as_ptr();
                    let x_ptr = xb.qs.as_ptr();

                    for chunk in 0..4 {
                        let mut acc_lo = _mm256_setzero_si256();
                        let mut acc_hi = _mm256_setzero_si256();
                        let c_qbase = chunk * 8 * 32;
                        let c_xlo = chunk * 64;
                        let c_xhi = chunk * 64 + 32;

                        for k in 0..8 {
                            let qb = _mm256_load_si256(qs_ptr.add(c_qbase + k * 32) as *const __m256i);
                            let q_lo = _mm256_and_si256(qb, mask_0f);
                            let q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_0f);
                            let x_lo = bcast_i32(x_ptr.add(c_xlo + k * 4));
                            let x_hi = bcast_i32(x_ptr.add(c_xhi + k * 4));
                            if $use_vnni {
                                acc_lo = vpdpbusd(acc_lo, q_lo, x_lo);
                                acc_hi = vpdpbusd(acc_hi, q_hi, x_hi);
                            } else {
                                acc_lo = dot4_avx2(acc_lo, q_lo, x_lo, ones_i16);
                                acc_hi = dot4_avx2(acc_hi, q_hi, x_hi, ones_i16);
                            }
                        }

                        let sub_lo = chunk * 2;
                        let sub_hi = chunk * 2 + 1;
                        let dsc_lo = _mm256_mul_ps(vd, load_u8x8_as_f32(&tile.sc[sub_lo]));
                        let dsc_hi = _mm256_mul_ps(vd, load_u8x8_as_f32(&tile.sc[sub_hi]));
                        let dmm_lo = _mm256_mul_ps(vdmin, load_u8x8_as_f32(&tile.m[sub_lo]));
                        let dmm_hi = _mm256_mul_ps(vdmin, load_u8x8_as_f32(&tile.m[sub_hi]));

                        let bs_lo = (*xb.bsums.get_unchecked(sub_lo * 2) as i32
                            + *xb.bsums.get_unchecked(sub_lo * 2 + 1) as i32) as f32;
                        let bs_hi = (*xb.bsums.get_unchecked(sub_hi * 2) as i32
                            + *xb.bsums.get_unchecked(sub_hi * 2 + 1) as i32) as f32;

                        blk_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_lo), dsc_lo, blk_sum);
                        blk_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_hi), dsc_hi, blk_sum);
                        blk_sum = _mm256_fnmadd_ps(_mm256_set1_ps(bs_lo), dmm_lo, blk_sum);
                        blk_sum = _mm256_fnmadd_ps(_mm256_set1_ps(bs_hi), dmm_hi, blk_sum);
                    }

                    fsum = _mm256_fmadd_ps(blk_sum, _mm256_set1_ps(xb.d), fsum);
                }

                _mm256_storeu_ps(out8.as_mut_ptr(), fsum);
            }
        };
    }

    impl_gemv_q4k_8x!(gemv_q4k_8x_vnni_f16c, "avx2,fma,f16c", true, true);
    impl_gemv_q4k_8x!(gemv_q4k_8x_avx2_f16c, "avx2,fma,f16c", false, true);
    impl_gemv_q4k_8x!(gemv_q4k_8x_avx2, "avx2,fma", false, false);

    // ==================== Q4_K 8×4 GEMM (4 tokens prefill) ====================

    macro_rules! impl_gemm_q4k_8x4 {
        ($fn_name:ident, $features:literal, $use_vnni:expr, $use_f16c:expr) => {
            #[target_feature(enable = $features)]
            pub unsafe fn $fn_name(
                tiles: &[TileQ4Kx8],
                xs: [&[BlockQ8_K]; 4],
                out: &mut [[f32; 8]; 4],
            ) {
                let nb = tiles.len();
                let mask_0f = _mm256_set1_epi8(0x0F);
                let ones_i16 = _mm256_set1_epi16(1);
                let mut fsum = [_mm256_setzero_ps(); 4];

                for b in 0..nb {
                    let tile = tiles.get_unchecked(b);
                    let xbs = [
                        xs[0].get_unchecked(b),
                        xs[1].get_unchecked(b),
                        xs[2].get_unchecked(b),
                        xs[3].get_unchecked(b),
                    ];

                    let vd = if $use_f16c {
                        load_f16x8_hw(&tile.d_f16)
                    } else {
                        load_f16x8_sw(&tile.d_f16)
                    };
                    let vdmin = if $use_f16c {
                        load_f16x8_hw(&tile.dmin_f16)
                    } else {
                        load_f16x8_sw(&tile.dmin_f16)
                    };

                    let mut blk_sum = [_mm256_setzero_ps(); 4];
                    let qs_ptr = tile.qs.as_ptr();
                    let xp0 = xbs[0].qs.as_ptr();
                    let xp1 = xbs[1].qs.as_ptr();
                    let xp2 = xbs[2].qs.as_ptr();
                    let xp3 = xbs[3].qs.as_ptr();

                    for chunk in 0..4 {
                        let mut acc_lo0 = _mm256_setzero_si256();
                        let mut acc_lo1 = _mm256_setzero_si256();
                        let mut acc_lo2 = _mm256_setzero_si256();
                        let mut acc_lo3 = _mm256_setzero_si256();
                        let mut acc_hi0 = _mm256_setzero_si256();
                        let mut acc_hi1 = _mm256_setzero_si256();
                        let mut acc_hi2 = _mm256_setzero_si256();
                        let mut acc_hi3 = _mm256_setzero_si256();

                        let c_qbase = chunk * 8 * 32;
                        let c_xlo = chunk * 64;
                        let c_xhi = chunk * 64 + 32;

                        for k in 0..8 {
                            let qb = _mm256_load_si256(qs_ptr.add(c_qbase + k * 32) as *const __m256i);
                            let q_lo = _mm256_and_si256(qb, mask_0f);
                            let q_hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), mask_0f);
                            let olo = c_xlo + k * 4;
                            let ohi = c_xhi + k * 4;

                            if $use_vnni {
                                acc_lo0 = vpdpbusd(acc_lo0, q_lo, bcast_i32(xp0.add(olo)));
                                acc_hi0 = vpdpbusd(acc_hi0, q_hi, bcast_i32(xp0.add(ohi)));
                                acc_lo1 = vpdpbusd(acc_lo1, q_lo, bcast_i32(xp1.add(olo)));
                                acc_hi1 = vpdpbusd(acc_hi1, q_hi, bcast_i32(xp1.add(ohi)));
                                acc_lo2 = vpdpbusd(acc_lo2, q_lo, bcast_i32(xp2.add(olo)));
                                acc_hi2 = vpdpbusd(acc_hi2, q_hi, bcast_i32(xp2.add(ohi)));
                                acc_lo3 = vpdpbusd(acc_lo3, q_lo, bcast_i32(xp3.add(olo)));
                                acc_hi3 = vpdpbusd(acc_hi3, q_hi, bcast_i32(xp3.add(ohi)));
                            } else {
                                acc_lo0 = dot4_avx2(acc_lo0, q_lo, bcast_i32(xp0.add(olo)), ones_i16);
                                acc_hi0 = dot4_avx2(acc_hi0, q_hi, bcast_i32(xp0.add(ohi)), ones_i16);
                                acc_lo1 = dot4_avx2(acc_lo1, q_lo, bcast_i32(xp1.add(olo)), ones_i16);
                                acc_hi1 = dot4_avx2(acc_hi1, q_hi, bcast_i32(xp1.add(ohi)), ones_i16);
                                acc_lo2 = dot4_avx2(acc_lo2, q_lo, bcast_i32(xp2.add(olo)), ones_i16);
                                acc_hi2 = dot4_avx2(acc_hi2, q_hi, bcast_i32(xp2.add(ohi)), ones_i16);
                                acc_lo3 = dot4_avx2(acc_lo3, q_lo, bcast_i32(xp3.add(olo)), ones_i16);
                                acc_hi3 = dot4_avx2(acc_hi3, q_hi, bcast_i32(xp3.add(ohi)), ones_i16);
                            }
                        }

                        let sub_lo = chunk * 2;
                        let sub_hi = chunk * 2 + 1;
                        let dsc_lo = _mm256_mul_ps(vd, load_u8x8_as_f32(&tile.sc[sub_lo]));
                        let dsc_hi = _mm256_mul_ps(vd, load_u8x8_as_f32(&tile.sc[sub_hi]));
                        let dmm_lo = _mm256_mul_ps(vdmin, load_u8x8_as_f32(&tile.m[sub_lo]));
                        let dmm_hi = _mm256_mul_ps(vdmin, load_u8x8_as_f32(&tile.m[sub_hi]));

                        let accs_lo = [acc_lo0, acc_lo1, acc_lo2, acc_lo3];
                        let accs_hi = [acc_hi0, acc_hi1, acc_hi2, acc_hi3];

                        for t in 0..4 {
                            let xb = xbs[t];
                            let bs_lo = (*xb.bsums.get_unchecked(sub_lo * 2) as i32
                                + *xb.bsums.get_unchecked(sub_lo * 2 + 1) as i32) as f32;
                            let bs_hi = (*xb.bsums.get_unchecked(sub_hi * 2) as i32
                                + *xb.bsums.get_unchecked(sub_hi * 2 + 1) as i32) as f32;
                            let mut s = blk_sum[t];
                            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(accs_lo[t]), dsc_lo, s);
                            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(accs_hi[t]), dsc_hi, s);
                            s = _mm256_fnmadd_ps(_mm256_set1_ps(bs_lo), dmm_lo, s);
                            s = _mm256_fnmadd_ps(_mm256_set1_ps(bs_hi), dmm_hi, s);
                            blk_sum[t] = s;
                        }
                    }

                    for t in 0..4 {
                        fsum[t] = _mm256_fmadd_ps(blk_sum[t], _mm256_set1_ps(xbs[t].d), fsum[t]);
                    }
                }

                for t in 0..4 {
                    _mm256_storeu_ps(out[t].as_mut_ptr(), fsum[t]);
                }
            }
        };
    }

    impl_gemm_q4k_8x4!(gemm_q4k_8x4_vnni_f16c, "avx2,fma,f16c", true, true);
    impl_gemm_q4k_8x4!(gemm_q4k_8x4_avx2_f16c, "avx2,fma,f16c", false, true);
    impl_gemm_q4k_8x4!(gemm_q4k_8x4_avx2, "avx2,fma", false, false);

    // ==================== Q6_K 8-row GEMV (1 token decode, ZERO unpacking!) ====================

    macro_rules! impl_gemv_q6k_8x {
        ($fn_name:ident, $features:literal, $use_vnni:expr, $use_f16c:expr) => {
            #[target_feature(enable = $features)]
            pub unsafe fn $fn_name(tiles: &[TileQ6Kx8], x: &[BlockQ8_K], out8: &mut [f32; 8]) {
                let nb = tiles.len();
                let ones_i16 = _mm256_set1_epi16(1);
                let mut fsum = _mm256_setzero_ps();

                for b in 0..nb {
                    let tile = tiles.get_unchecked(b);
                    let xb = x.get_unchecked(b);

                    if b + 1 < nb {
                        let next_ptr = (tiles.as_ptr().add(b + 1)) as *const i8;
                        _mm_prefetch(next_ptr, _MM_HINT_T0);
                        _mm_prefetch(next_ptr.add(256), _MM_HINT_T0);
                    }

                    let vd = if $use_f16c {
                        load_f16x8_hw(&tile.d_f16)
                    } else {
                        load_f16x8_sw(&tile.d_f16)
                    };
                    let vd_dx = _mm256_mul_ps(vd, _mm256_set1_ps(xb.d));

                    let mut blk_sum = _mm256_setzero_ps();
                    let q_ptr = tile.q.as_ptr();
                    let x_ptr = xb.qs.as_ptr();

                    for sub in 0..16 {
                        let mut acc = _mm256_setzero_si256();
                        let s_q = sub * 4 * 32;
                        let s_x = sub * 16;
                        for k in 0..4 {
                            let qv = _mm256_load_si256(q_ptr.add(s_q + k * 32) as *const __m256i);
                            let xv = bcast_i32(x_ptr.add(s_x + k * 4));
                            if $use_vnni {
                                acc = vpdpbusd(acc, qv, xv);
                            } else {
                                acc = dot4_avx2(acc, qv, xv, ones_i16);
                            }
                        }
                        let bias = _mm256_set1_epi32((*xb.bsums.get_unchecked(sub) as i32) * 32);
                        let adj = _mm256_sub_epi32(acc, bias);
                        let sc_f = load_i8x8_as_f32(&tile.sc[sub]);
                        blk_sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(adj), sc_f, blk_sum);
                    }

                    fsum = _mm256_fmadd_ps(blk_sum, vd_dx, fsum);
                }

                _mm256_storeu_ps(out8.as_mut_ptr(), fsum);
            }
        };
    }

    impl_gemv_q6k_8x!(gemv_q6k_8x_vnni_f16c, "avx2,fma,f16c", true, true);
    impl_gemv_q6k_8x!(gemv_q6k_8x_avx2_f16c, "avx2,fma,f16c", false, true);
    impl_gemv_q6k_8x!(gemv_q6k_8x_avx2, "avx2,fma", false, false);

    // ==================== Q6_K 8×4 & 8×8 GEMM (Prefill, ZERO unpacking!) ====================

    macro_rules! impl_gemm_q6k_8x4 {
        ($fn_name:ident, $features:literal, $use_vnni:expr, $use_f16c:expr) => {
            #[target_feature(enable = $features)]
            pub unsafe fn $fn_name(
                tiles: &[TileQ6Kx8],
                xs: [&[BlockQ8_K]; 4],
                out: &mut [[f32; 8]; 4],
            ) {
                let nb = tiles.len();
                let ones_i16 = _mm256_set1_epi16(1);
                let mut fsum = [_mm256_setzero_ps(); 4];

                for b in 0..nb {
                    let tile = tiles.get_unchecked(b);
                    let xbs = [
                        xs[0].get_unchecked(b),
                        xs[1].get_unchecked(b),
                        xs[2].get_unchecked(b),
                        xs[3].get_unchecked(b),
                    ];
                    let vd = if $use_f16c {
                        load_f16x8_hw(&tile.d_f16)
                    } else {
                        load_f16x8_sw(&tile.d_f16)
                    };
                    let mut blk_sum = [_mm256_setzero_ps(); 4];
                    let q_ptr = tile.q.as_ptr();
                    let xp0 = xbs[0].qs.as_ptr();
                    let xp1 = xbs[1].qs.as_ptr();
                    let xp2 = xbs[2].qs.as_ptr();
                    let xp3 = xbs[3].qs.as_ptr();

                    for sub in 0..16 {
                        let mut a0 = _mm256_setzero_si256();
                        let mut a1 = _mm256_setzero_si256();
                        let mut a2 = _mm256_setzero_si256();
                        let mut a3 = _mm256_setzero_si256();
                        let s_q = sub * 4 * 32;
                        let s_x = sub * 16;

                        for k in 0..4 {
                            let qv = _mm256_load_si256(q_ptr.add(s_q + k * 32) as *const __m256i);
                            let ox = s_x + k * 4;
                            if $use_vnni {
                                a0 = vpdpbusd(a0, qv, bcast_i32(xp0.add(ox)));
                                a1 = vpdpbusd(a1, qv, bcast_i32(xp1.add(ox)));
                                a2 = vpdpbusd(a2, qv, bcast_i32(xp2.add(ox)));
                                a3 = vpdpbusd(a3, qv, bcast_i32(xp3.add(ox)));
                            } else {
                                a0 = dot4_avx2(a0, qv, bcast_i32(xp0.add(ox)), ones_i16);
                                a1 = dot4_avx2(a1, qv, bcast_i32(xp1.add(ox)), ones_i16);
                                a2 = dot4_avx2(a2, qv, bcast_i32(xp2.add(ox)), ones_i16);
                                a3 = dot4_avx2(a3, qv, bcast_i32(xp3.add(ox)), ones_i16);
                            }
                        }

                        let sc_f = load_i8x8_as_f32(&tile.sc[sub]);
                        let accs = [a0, a1, a2, a3];
                        for t in 0..4 {
                            let bias = _mm256_set1_epi32((*xbs[t].bsums.get_unchecked(sub) as i32) * 32);
                            let adj = _mm256_sub_epi32(accs[t], bias);
                            blk_sum[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(adj), sc_f, blk_sum[t]);
                        }
                    }

                    for t in 0..4 {
                        let vd_dx = _mm256_mul_ps(vd, _mm256_set1_ps(xbs[t].d));
                        fsum[t] = _mm256_fmadd_ps(blk_sum[t], vd_dx, fsum[t]);
                    }
                }

                for t in 0..4 {
                    _mm256_storeu_ps(out[t].as_mut_ptr(), fsum[t]);
                }
            }
        };
    }

    impl_gemm_q6k_8x4!(gemm_q6k_8x4_vnni_f16c, "avx2,fma,f16c", true, true);
    impl_gemm_q6k_8x4!(gemm_q6k_8x4_avx2_f16c, "avx2,fma,f16c", false, true);
    impl_gemm_q6k_8x4!(gemm_q6k_8x4_avx2, "avx2,fma", false, false);

    macro_rules! impl_gemm_q6k_8x8 {
        ($fn_name:ident, $features:literal, $use_vnni:expr) => {
            #[target_feature(enable = $features)]
            pub unsafe fn $fn_name(
                tiles: &[TileQ6Kx8],
                xs: [&[BlockQ8_K]; 8],
                out: &mut [[f32; 8]; 8],
            ) {
                let nb = tiles.len();
                let ones_i16 = _mm256_set1_epi16(1);
                let mut fsum = [_mm256_setzero_ps(); 8];

                for b in 0..nb {
                    let tile = tiles.get_unchecked(b);
                    let xbs = [
                        xs[0].get_unchecked(b),
                        xs[1].get_unchecked(b),
                        xs[2].get_unchecked(b),
                        xs[3].get_unchecked(b),
                        xs[4].get_unchecked(b),
                        xs[5].get_unchecked(b),
                        xs[6].get_unchecked(b),
                        xs[7].get_unchecked(b),
                    ];
                    let vd = load_f16x8_hw(&tile.d_f16);
                    let mut blk_sum = [_mm256_setzero_ps(); 8];
                    let q_ptr = tile.q.as_ptr();

                    for sub in 0..16 {
                        let mut a0 = _mm256_setzero_si256();
                        let mut a1 = _mm256_setzero_si256();
                        let mut a2 = _mm256_setzero_si256();
                        let mut a3 = _mm256_setzero_si256();
                        let mut a4 = _mm256_setzero_si256();
                        let mut a5 = _mm256_setzero_si256();
                        let mut a6 = _mm256_setzero_si256();
                        let mut a7 = _mm256_setzero_si256();
                        let s_q = sub * 4 * 32;
                        let s_x = sub * 16;

                        for k in 0..4 {
                            let qv = _mm256_load_si256(q_ptr.add(s_q + k * 32) as *const __m256i);
                            let ox = s_x + k * 4;
                            if $use_vnni {
                                a0 = vpdpbusd(a0, qv, bcast_i32(xbs[0].qs.as_ptr().add(ox)));
                                a1 = vpdpbusd(a1, qv, bcast_i32(xbs[1].qs.as_ptr().add(ox)));
                                a2 = vpdpbusd(a2, qv, bcast_i32(xbs[2].qs.as_ptr().add(ox)));
                                a3 = vpdpbusd(a3, qv, bcast_i32(xbs[3].qs.as_ptr().add(ox)));
                                a4 = vpdpbusd(a4, qv, bcast_i32(xbs[4].qs.as_ptr().add(ox)));
                                a5 = vpdpbusd(a5, qv, bcast_i32(xbs[5].qs.as_ptr().add(ox)));
                                a6 = vpdpbusd(a6, qv, bcast_i32(xbs[6].qs.as_ptr().add(ox)));
                                a7 = vpdpbusd(a7, qv, bcast_i32(xbs[7].qs.as_ptr().add(ox)));
                            } else {
                                a0 = dot4_avx2(a0, qv, bcast_i32(xbs[0].qs.as_ptr().add(ox)), ones_i16);
                                a1 = dot4_avx2(a1, qv, bcast_i32(xbs[1].qs.as_ptr().add(ox)), ones_i16);
                                a2 = dot4_avx2(a2, qv, bcast_i32(xbs[2].qs.as_ptr().add(ox)), ones_i16);
                                a3 = dot4_avx2(a3, qv, bcast_i32(xbs[3].qs.as_ptr().add(ox)), ones_i16);
                                a4 = dot4_avx2(a4, qv, bcast_i32(xbs[4].qs.as_ptr().add(ox)), ones_i16);
                                a5 = dot4_avx2(a5, qv, bcast_i32(xbs[5].qs.as_ptr().add(ox)), ones_i16);
                                a6 = dot4_avx2(a6, qv, bcast_i32(xbs[6].qs.as_ptr().add(ox)), ones_i16);
                                a7 = dot4_avx2(a7, qv, bcast_i32(xbs[7].qs.as_ptr().add(ox)), ones_i16);
                            }
                        }

                        let sc_f = load_i8x8_as_f32(&tile.sc[sub]);
                        let accs = [a0, a1, a2, a3, a4, a5, a6, a7];
                        for t in 0..8 {
                            let bias = _mm256_set1_epi32((*xbs[t].bsums.get_unchecked(sub) as i32) * 32);
                            let adj = _mm256_sub_epi32(accs[t], bias);
                            blk_sum[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(adj), sc_f, blk_sum[t]);
                        }
                    }

                    for t in 0..8 {
                        let vd_dx = _mm256_mul_ps(vd, _mm256_set1_ps(xbs[t].d));
                        fsum[t] = _mm256_fmadd_ps(blk_sum[t], vd_dx, fsum[t]);
                    }
                }

                for t in 0..8 {
                    _mm256_storeu_ps(out[t].as_mut_ptr(), fsum[t]);
                }
            }
        };
    }

    impl_gemm_q6k_8x8!(gemm_q6k_8x8_vnni_f16c, "avx2,fma,f16c", true);
    impl_gemm_q6k_8x8!(gemm_q6k_8x8_avx2_f16c, "avx2,fma,f16c", false);
}
