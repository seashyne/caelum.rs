//! x86_64 AVX2 + FMA SIMD kernels for F32 and W4A8/W8A8 quantized dot products.

use caelum_core::{f16_to_f32, quant::get_scale_min_k4, BlockQ8_0, BlockQ8_K};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0x01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// 8-wide AVX2 + FMA dot product on two `f32` slices.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// 8-wide AVX2 + FMA dot product between unaligned little-endian f32 bytes and `x: &[f32]`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_bytes_avx2(raw: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let raw_f32 = raw.as_ptr() as *const f32;

    while i + 16 <= n {
        let vw0 = _mm256_loadu_ps(raw_f32.add(i));
        let vx0 = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(vw0, vx0, acc0);

        let vw1 = _mm256_loadu_ps(raw_f32.add(i + 8));
        let vx1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(vw1, vx1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let vw = _mm256_loadu_ps(raw_f32.add(i));
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(vw, vx, acc0);
        i += 8;
    }

    let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        let b = &raw[i * 4..i * 4 + 4];
        sum += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * *x.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Exact AVX2 W4A8 (`Q4_0 × Q8_0`) dot product using signed 16-bit multiply-add (`_mm256_madd_epi16`).
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q4_0_q8_0_avx2(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mask_low = _mm_set1_epi8(0x0F);
    let eight = _mm_set1_epi8(8);
    let mut sum = 0.0f32;

    for b in 0..nb {
        let off = b * 18;
        let dw = f16_to_f32(u16::from_le_bytes([*raw.get_unchecked(off), *raw.get_unchecked(off + 1)]));
        let dx = f16_to_f32(x.get_unchecked(b).d);

        let qs = _mm_loadu_si128(raw.as_ptr().add(off + 2) as *const __m128i);
        let q_lo_i8 = _mm_sub_epi8(_mm_and_si128(qs, mask_low), eight);
        let q_hi_i8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qs, 4), mask_low), eight);

        let x_ptr = x.get_unchecked(b).qs.as_ptr() as *const __m128i;
        let x_lo_i8 = _mm_loadu_si128(x_ptr);
        let x_hi_i8 = _mm_loadu_si128(x_ptr.add(1));

        let q_lo_16 = _mm256_cvtepi8_epi16(q_lo_i8);
        let q_hi_16 = _mm256_cvtepi8_epi16(q_hi_i8);
        let x_lo_16 = _mm256_cvtepi8_epi16(x_lo_i8);
        let x_hi_16 = _mm256_cvtepi8_epi16(x_hi_i8);

        let p0 = _mm256_madd_epi16(q_lo_16, x_lo_16);
        let p1 = _mm256_madd_epi16(q_hi_16, x_hi_16);
        let isum = hsum256_epi32(_mm256_add_epi32(p0, p1));

        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// Exact AVX2 W8A8 (`Q8_0 × Q8_0`) dot product using signed 16-bit multiply-add (`_mm256_madd_epi16`).
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q8_0_q8_0_avx2(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;

    for b in 0..nb {
        let off = b * 34;
        let dw = f16_to_f32(u16::from_le_bytes([*raw.get_unchecked(off), *raw.get_unchecked(off + 1)]));
        let dx = f16_to_f32(x.get_unchecked(b).d);

        let w_ptr = raw.as_ptr().add(off + 2) as *const __m128i;
        let x_ptr = x.get_unchecked(b).qs.as_ptr() as *const __m128i;

        let w_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr));
        let w_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(w_ptr.add(1)));
        let x_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_ptr));
        let x_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_ptr.add(1)));

        let p0 = _mm256_madd_epi16(w_lo, x_lo);
        let p1 = _mm256_madd_epi16(w_hi, x_hi);
        let isum = hsum256_epi32(_mm256_add_epi32(p0, p1));

        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// Exact AVX2 W4A8 (`Q4_K × Q8_K`) super-block dot product with fused scale multiply-add.
///
/// Uses `_mm256_madd_epi16(p16, set1_epi16(sc))` to fuse scale multiplication into the i32
/// accumulation, performing only ONE horizontal reduction per 256-element super-block.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q4_k_q8_k_avx2(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mask_low = _mm256_set1_epi8(0x0F);
    let mut sumf = 0.0f32;

    for b in 0..nb {
        let off = b * 144;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off),
            *raw.get_unchecked(off + 1),
        ]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off + 2),
            *raw.get_unchecked(off + 3),
        ]));
        let scales = &raw[off + 4..off + 16];
        let qs_ptr = raw.as_ptr().add(off + 16);
        let xb = x.get_unchecked(b);

        let mut acc_q = _mm256_setzero_si256();
        let mut sum_m = 0i32;

        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            sum_m += (m1 as i32)
                * (*xb.bsums.get_unchecked(is * 2) as i32
                    + *xb.bsums.get_unchecked(is * 2 + 1) as i32);
            sum_m += (m2 as i32)
                * (*xb.bsums.get_unchecked((is + 1) * 2) as i32
                    + *xb.bsums.get_unchecked((is + 1) * 2 + 1) as i32);

            let q_bytes = _mm256_loadu_si256(qs_ptr.add(chunk * 32) as *const __m256i);
            let q_lo = _mm256_and_si256(q_bytes, mask_low);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), mask_low);

            let x_lo = _mm256_loadu_si256(xb.qs.as_ptr().add(chunk * 64) as *const __m256i);
            let x_hi = _mm256_loadu_si256(xb.qs.as_ptr().add(chunk * 64 + 32) as *const __m256i);

            let p_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, x_lo), _mm256_set1_epi16(sc1 as i16));
            let p_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, x_hi), _mm256_set1_epi16(sc2 as i16));
            acc_q = _mm256_add_epi32(acc_q, _mm256_add_epi32(p_lo, p_hi));
        }

        let sum_q = hsum256_epi32(acc_q);
        sumf += (dw * xb.d) * (sum_q as f32) - (dmin_w * xb.d) * (sum_m as f32);
    }
    sumf
}

/// 4-token batched AVX2 W4A8 (`Q4_K × Q8_K`) prefill GEMM micro-kernel with fused scale multiply-add.
///
/// Unpacks each `Q4_K` weight super-block ONCE in registers and accumulates 4 vector `i32x8`
/// accumulators across all 4 chunks — reducing horizontal reductions by 8×.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q4_k_q8_k_batch4_avx2(
    raw: &[u8],
    x0: &[BlockQ8_K],
    x1: &[BlockQ8_K],
    x2: &[BlockQ8_K],
    x3: &[BlockQ8_K],
) -> [f32; 4] {
    let nb = x0.len();
    let mask_low = _mm256_set1_epi8(0x0F);
    let mut sumf = [0.0f32; 4];

    for b in 0..nb {
        let off = b * 144;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off),
            *raw.get_unchecked(off + 1),
        ]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off + 2),
            *raw.get_unchecked(off + 3),
        ]));
        let scales = &raw[off + 4..off + 16];
        let qs_ptr = raw.as_ptr().add(off + 16);

        let xbs = [
            x0.get_unchecked(b),
            x1.get_unchecked(b),
            x2.get_unchecked(b),
            x3.get_unchecked(b),
        ];

        let mut acc_q = [_mm256_setzero_si256(); 4];
        let mut sum_m = [0i32; 4];

        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let sc1_v = _mm256_set1_epi16(sc1 as i16);
            let sc2_v = _mm256_set1_epi16(sc2 as i16);
            let m1_i = m1 as i32;
            let m2_i = m2 as i32;

            // Unpack weight nibbles ONCE for all 4 tokens
            let q_bytes = _mm256_loadu_si256(qs_ptr.add(chunk * 32) as *const __m256i);
            let q_lo = _mm256_and_si256(q_bytes, mask_low);
            let q_hi = _mm256_and_si256(_mm256_srli_epi16(q_bytes, 4), mask_low);

            let lo_off = chunk * 64;
            let hi_off = chunk * 64 + 32;

            for t in 0..4 {
                let xb = xbs[t];
                sum_m[t] += m1_i
                    * (*xb.bsums.get_unchecked(is * 2) as i32
                        + *xb.bsums.get_unchecked(is * 2 + 1) as i32);
                sum_m[t] += m2_i
                    * (*xb.bsums.get_unchecked((is + 1) * 2) as i32
                        + *xb.bsums.get_unchecked((is + 1) * 2 + 1) as i32);

                let x_lo = _mm256_loadu_si256(xb.qs.as_ptr().add(lo_off) as *const __m256i);
                let x_hi = _mm256_loadu_si256(xb.qs.as_ptr().add(hi_off) as *const __m256i);

                let p_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, x_lo), sc1_v);
                let p_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, x_hi), sc2_v);
                acc_q[t] = _mm256_add_epi32(acc_q[t], _mm256_add_epi32(p_lo, p_hi));
            }
        }

        for t in 0..4 {
            let dx = xbs[t].d;
            let sq = hsum256_epi32(acc_q[t]);
            sumf[t] += (dw * dx) * (sq as f32) - (dmin_w * dx) * (sum_m[t] as f32);
        }
    }
    sumf
}

/// Exact AVX2 W6A8 (`Q6_K × Q8_K`) super-block dot product with single horizontal reduction per block.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q6_k_q8_k_avx2(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mask_low4 = _mm_set1_epi8(0x0F);
    let mask_low2 = _mm_set1_epi8(0x03);
    let thirty_two = _mm_set1_epi8(32);
    let mut sumf = 0.0f32;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn scaled_dot16_v(q16: __m128i, x_ptr: *const i8, sc: i32) -> __m256i {
        let q_w = _mm256_cvtepi8_epi16(q16);
        let x_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_ptr as *const __m128i));
        _mm256_mullo_epi32(_mm256_madd_epi16(q_w, x_w), _mm256_set1_epi32(sc))
    }

    for b in 0..nb {
        let off = b * 210;
        let ql_base = raw.as_ptr().add(off);
        let qh_base = raw.as_ptr().add(off + 128);
        let sc_base = raw.as_ptr().add(off + 192) as *const i8;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off + 208),
            *raw.get_unchecked(off + 209),
        ]));
        let xb = x.get_unchecked(b);

        let mut acc = _mm256_setzero_si256();
        for half in 0..2 {
            let ql_h = ql_base.add(half * 64);
            let qh_h = qh_base.add(half * 32);
            let sc_h = sc_base.add(half * 8);
            let x_h = xb.qs.as_ptr().add(half * 128);

            for is in 0..2 {
                let l = is * 16;
                let ql0 = _mm_loadu_si128(ql_h.add(l) as *const __m128i);
                let ql1 = _mm_loadu_si128(ql_h.add(l + 32) as *const __m128i);
                let qh = _mm_loadu_si128(qh_h.add(l) as *const __m128i);

                let qh0 = _mm_slli_epi16(_mm_and_si128(qh, mask_low2), 4);
                let qh1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 2), mask_low2), 4);
                let qh2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 4), mask_low2), 4);
                let qh3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 6), mask_low2), 4);

                let q1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql0, mask_low4), qh0), thirty_two);
                let q2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql1, mask_low4), qh1), thirty_two);
                let q3 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_low4), qh2),
                    thirty_two,
                );
                let q4 = _mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low4), qh3),
                    thirty_two,
                );

                let s0 = scaled_dot16_v(q1, x_h.add(l), *sc_h.add(is) as i32);
                let s1 = scaled_dot16_v(q2, x_h.add(l + 32), *sc_h.add(is + 2) as i32);
                let s2 = scaled_dot16_v(q3, x_h.add(l + 64), *sc_h.add(is + 4) as i32);
                let s3 = scaled_dot16_v(q4, x_h.add(l + 96), *sc_h.add(is + 6) as i32);
                acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3)));
            }
        }
        sumf += (dw * xb.d) * (hsum256_epi32(acc) as f32);
    }
    sumf
}

/// 4-token batched AVX2 W6A8 (`Q6_K × Q8_K`) prefill GEMM micro-kernel with vector accumulation.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q6_k_q8_k_batch4_avx2(
    raw: &[u8],
    x0: &[BlockQ8_K],
    x1: &[BlockQ8_K],
    x2: &[BlockQ8_K],
    x3: &[BlockQ8_K],
) -> [f32; 4] {
    let nb = x0.len();
    let mask_low4 = _mm_set1_epi8(0x0F);
    let mask_low2 = _mm_set1_epi8(0x03);
    let thirty_two = _mm_set1_epi8(32);
    let mut sumf = [0.0f32; 4];

    for b in 0..nb {
        let off = b * 210;
        let ql_base = raw.as_ptr().add(off);
        let qh_base = raw.as_ptr().add(off + 128);
        let sc_base = raw.as_ptr().add(off + 192) as *const i8;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off + 208),
            *raw.get_unchecked(off + 209),
        ]));
        let xbs = [
            x0.get_unchecked(b),
            x1.get_unchecked(b),
            x2.get_unchecked(b),
            x3.get_unchecked(b),
        ];

        let mut acc = [_mm256_setzero_si256(); 4];
        for half in 0..2 {
            let ql_h = ql_base.add(half * 64);
            let qh_h = qh_base.add(half * 32);
            let sc_h = sc_base.add(half * 8);

            for is in 0..2 {
                let l = is * 16;
                let ql0 = _mm_loadu_si128(ql_h.add(l) as *const __m128i);
                let ql1 = _mm_loadu_si128(ql_h.add(l + 32) as *const __m128i);
                let qh = _mm_loadu_si128(qh_h.add(l) as *const __m128i);

                let qh0 = _mm_slli_epi16(_mm_and_si128(qh, mask_low2), 4);
                let qh1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 2), mask_low2), 4);
                let qh2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 4), mask_low2), 4);
                let qh3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 6), mask_low2), 4);

                let q1_w = _mm256_cvtepi8_epi16(_mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql0, mask_low4), qh0),
                    thirty_two,
                ));
                let q2_w = _mm256_cvtepi8_epi16(_mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(ql1, mask_low4), qh1),
                    thirty_two,
                ));
                let q3_w = _mm256_cvtepi8_epi16(_mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mask_low4), qh2),
                    thirty_two,
                ));
                let q4_w = _mm256_cvtepi8_epi16(_mm_sub_epi8(
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low4), qh3),
                    thirty_two,
                ));

                let sc0 = _mm256_set1_epi32(*sc_h.add(is) as i32);
                let sc2 = _mm256_set1_epi32(*sc_h.add(is + 2) as i32);
                let sc4 = _mm256_set1_epi32(*sc_h.add(is + 4) as i32);
                let sc6 = _mm256_set1_epi32(*sc_h.add(is + 6) as i32);

                for t in 0..4 {
                    let x_h = xbs[t].qs.as_ptr().add(half * 128);
                    let x1_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_h.add(l) as *const __m128i));
                    let x2_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_h.add(l + 32) as *const __m128i));
                    let x3_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_h.add(l + 64) as *const __m128i));
                    let x4_w = _mm256_cvtepi8_epi16(_mm_loadu_si128(x_h.add(l + 96) as *const __m128i));

                    let p0 = _mm256_mullo_epi32(_mm256_madd_epi16(q1_w, x1_w), sc0);
                    let p1 = _mm256_mullo_epi32(_mm256_madd_epi16(q2_w, x2_w), sc2);
                    let p2 = _mm256_mullo_epi32(_mm256_madd_epi16(q3_w, x3_w), sc4);
                    let p3 = _mm256_mullo_epi32(_mm256_madd_epi16(q4_w, x4_w), sc6);
                    acc[t] = _mm256_add_epi32(acc[t], _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
                }
            }
        }
        for t in 0..4 {
            sumf[t] += (dw * xbs[t].d) * (hsum256_epi32(acc[t]) as f32);
        }
    }
    sumf
}
