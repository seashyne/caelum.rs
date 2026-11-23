//! ARMv9 / AArch64 native SIMD kernels (NEON FMA + ARMv9 `dotprod` SDOT W4A8/W8A8).
//!
//! Designed so Arm KleidiAI `i8mm` / `SME2` tiled kernels can plug directly into the
//! same `Backend` dispatch layer without touching higher-level model crates.

use caelum_core::{f16_to_f32, quant::get_scale_min_k4, BlockQ8_0, BlockQ8_K};
use std::arch::aarch64::*;

/// NEON FMA F32 dot product (4-wide x 2 accumulators).
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 8 <= n {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// NEON FMA dot product on unaligned little-endian F32 bytes and `x: &[f32]`.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_bytes_neon(raw: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let raw_f32 = raw.as_ptr() as *const f32;

    while i + 8 <= n {
        let vw0 = vld1q_f32(raw_f32.add(i));
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, vw0, vx0);

        let vw1 = vld1q_f32(raw_f32.add(i + 4));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, vw1, vx1);
        i += 8;
    }
    while i + 4 <= n {
        let vw = vld1q_f32(raw_f32.add(i));
        let vx = vld1q_f32(x.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, vw, vx);
        i += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        let b = &raw[i * 4..i * 4 + 4];
        sum += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * *x.get_unchecked(i);
        i += 1;
    }
    sum
}

/// ARMv9 native W4A8 (`Q4_0 × Q8_0`) dot product using hardware `SDOT` (`vdotq_s32`).
///
/// Each 32-element block unpacks 16 low nibbles (`0..16`) and 16 high nibbles (`16..32`),
/// centers by `-8`, and executes two single-cycle `SDOT` instructions directly into `i32x4`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_0_q8_0_dotprod(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mask_low = vdupq_n_u8(0x0F);
    let eight = vdupq_n_s8(8);
    let mut sum = 0.0f32;

    for b in 0..nb {
        let off = b * 18;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off),
            *raw.get_unchecked(off + 1),
        ]));
        let dx = f16_to_f32(x.get_unchecked(b).d);

        let qs = vld1q_u8(raw.as_ptr().add(off + 2));
        let q_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, mask_low)), eight);
        let q_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs)), eight);

        let x_ptr = x.get_unchecked(b).qs.as_ptr();
        let x_lo = vld1q_s8(x_ptr);
        let x_hi = vld1q_s8(x_ptr.add(16));

        let acc = vdotq_s32(vdupq_n_s32(0), q_lo, x_lo);
        let acc = vdotq_s32(acc, q_hi, x_hi);
        let isum = vaddvq_s32(acc);

        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// ARMv9 native W8A8 (`Q8_0 × Q8_0`) dot product using hardware `SDOT` (`vdotq_s32`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q8_0_q8_0_dotprod(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;

    for b in 0..nb {
        let off = b * 34;
        let dw = f16_to_f32(u16::from_le_bytes([
            *raw.get_unchecked(off),
            *raw.get_unchecked(off + 1),
        ]));
        let dx = f16_to_f32(x.get_unchecked(b).d);

        let w_ptr = raw.as_ptr().add(off + 2) as *const i8;
        let x_ptr = x.get_unchecked(b).qs.as_ptr();

        let w_lo = vld1q_s8(w_ptr);
        let w_hi = vld1q_s8(w_ptr.add(16));
        let x_lo = vld1q_s8(x_ptr);
        let x_hi = vld1q_s8(x_ptr.add(16));

        let acc = vdotq_s32(vdupq_n_s32(0), w_lo, x_lo);
        let acc = vdotq_s32(acc, w_hi, x_hi);
        let isum = vaddvq_s32(acc);

        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// ARMv9 native W4A8 (`Q4_K × Q8_K`) super-block dot product using hardware `SDOT` (`vdotq_s32`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8_k_dotprod(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mask_low = vdupq_n_u8(0x0F);
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

        let mut sum_q = 0i32;
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

            let q0 = vld1q_u8(qs_ptr.add(chunk * 32));
            let q1 = vld1q_u8(qs_ptr.add(chunk * 32 + 16));

            let q_lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask_low));
            let q_lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask_low));
            let q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
            let q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

            let x_ptr = xb.qs.as_ptr().add(chunk * 64);
            let x_lo0 = vld1q_s8(x_ptr);
            let x_lo1 = vld1q_s8(x_ptr.add(16));
            let x_hi0 = vld1q_s8(x_ptr.add(32));
            let x_hi1 = vld1q_s8(x_ptr.add(48));

            let acc_lo = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q_lo0, x_lo0), q_lo1, x_lo1);
            let acc_hi = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q_hi0, x_hi0), q_hi1, x_hi1);

            sum_q += (sc1 as i32) * vaddvq_s32(acc_lo) + (sc2 as i32) * vaddvq_s32(acc_hi);
        }

        sumf += (dw * xb.d) * (sum_q as f32) - (dmin_w * xb.d) * (sum_m as f32);
    }
    sumf
}

/// ARMv9 native W6A8 (`Q6_K × Q8_K`) super-block dot product using hardware `SDOT` (`vdotq_s32`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q6_k_q8_k_dotprod(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mask_low4 = vdupq_n_u8(0x0F);
    let mask_low2 = vdupq_n_u8(0x03);
    let thirty_two = vdupq_n_s8(32);
    let mut sumf = 0.0f32;

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

        let mut isum = 0i32;
        for half in 0..2 {
            let ql_h = ql_base.add(half * 64);
            let qh_h = qh_base.add(half * 32);
            let sc_h = sc_base.add(half * 8);
            let x_h = xb.qs.as_ptr().add(half * 128);

            for is in 0..2 {
                let l = is * 16;
                let ql0 = vld1q_u8(ql_h.add(l));
                let ql1 = vld1q_u8(ql_h.add(l + 32));
                let qh = vld1q_u8(qh_h.add(l));

                let qh0 = vshlq_n_u8::<4>(vandq_u8(qh, mask_low2));
                let qh1 = vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh), mask_low2));
                let qh2 = vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh), mask_low2));
                let qh3 = vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh));

                let q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_low4), qh0)), thirty_two);
                let q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_low4), qh1)), thirty_two);
                let q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql0), qh2)), thirty_two);
                let q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql1), qh3)), thirty_two);

                let d1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, vld1q_s8(x_h.add(l))));
                let d2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(x_h.add(l + 32))));
                let d3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, vld1q_s8(x_h.add(l + 64))));
                let d4 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q4, vld1q_s8(x_h.add(l + 96))));

                isum += (*sc_h.add(is) as i32) * d1
                    + (*sc_h.add(is + 2) as i32) * d2
                    + (*sc_h.add(is + 4) as i32) * d3
                    + (*sc_h.add(is + 6) as i32) * d4;
            }
        }
        sumf += (dw * xb.d) * (isum as f32);
    }
    sumf
}

/// Stable ARMv9 `I8MM` (`SMMLA`) 2×8 × 8×2 signed int8 matrix multiply-accumulate instruction.
///
/// Computes a 2×2 output tile `[r0t0, r0t1, r1t0, r1t1]` in a single hardware instruction:
/// - `a`: 2 weight rows × 8 int8 elements (`row0[0..8] || row1[0..8]`)
/// - `b`: 2 token cols × 8 int8 elements (`tok0[0..8] || tok1[0..8]`)
#[inline]
#[target_feature(enable = "neon,i8mm")]
unsafe fn smmla_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    acc
}

/// ARMv9 `I8MM` (`SMMLA`) 2-row × 2-token tiled `Q4_K × Q8_K` prefill GEMM micro-kernel.
///
/// Computes `[[dot(r0,t0), dot(r0,t1)], [dot(r1,t0), dot(r1,t1)]]` using hardware `SMMLA`
/// matrix-multiply instructions (2× throughput over `SDOT`, 2× lower register traffic).
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn dot_q4_k_q8_k_2x2_i8mm(
    raw0: &[u8],
    raw1: &[u8],
    x0: &[BlockQ8_K],
    x1: &[BlockQ8_K],
) -> [[f32; 2]; 2] {
    let nb = x0.len();
    let mask_low = vdupq_n_u8(0x0F);
    let mut out = [[0.0f32; 2]; 2];

    #[inline]
    #[target_feature(enable = "neon,i8mm")]
    unsafe fn smmla_32(
        w0_a: int8x16_t,
        w0_b: int8x16_t,
        w1_a: int8x16_t,
        w1_b: int8x16_t,
        x0_ptr: *const i8,
        x1_ptr: *const i8,
    ) -> int32x4_t {
        let x0_a = vld1q_s8(x0_ptr);
        let x0_b = vld1q_s8(x0_ptr.add(16));
        let x1_a = vld1q_s8(x1_ptr);
        let x1_b = vld1q_s8(x1_ptr.add(16));

        let mut acc = vdupq_n_s32(0);
        acc = smmla_s32(
            acc,
            vcombine_s8(vget_low_s8(w0_a), vget_low_s8(w1_a)),
            vcombine_s8(vget_low_s8(x0_a), vget_low_s8(x1_a)),
        );
        acc = smmla_s32(
            acc,
            vcombine_s8(vget_high_s8(w0_a), vget_high_s8(w1_a)),
            vcombine_s8(vget_high_s8(x0_a), vget_high_s8(x1_a)),
        );
        acc = smmla_s32(
            acc,
            vcombine_s8(vget_low_s8(w0_b), vget_low_s8(w1_b)),
            vcombine_s8(vget_low_s8(x0_b), vget_low_s8(x1_b)),
        );
        smmla_s32(
            acc,
            vcombine_s8(vget_high_s8(w0_b), vget_high_s8(w1_b)),
            vcombine_s8(vget_high_s8(x0_b), vget_high_s8(x1_b)),
        )
    }

    for b in 0..nb {
        let off = b * 144;
        let dw0 = f16_to_f32(u16::from_le_bytes([*raw0.get_unchecked(off), *raw0.get_unchecked(off + 1)]));
        let dm0 = f16_to_f32(u16::from_le_bytes([*raw0.get_unchecked(off + 2), *raw0.get_unchecked(off + 3)]));
        let sc_raw0 = &raw0[off + 4..off + 16];
        let qs0 = raw0.as_ptr().add(off + 16);

        let dw1 = f16_to_f32(u16::from_le_bytes([*raw1.get_unchecked(off), *raw1.get_unchecked(off + 1)]));
        let dm1 = f16_to_f32(u16::from_le_bytes([*raw1.get_unchecked(off + 2), *raw1.get_unchecked(off + 3)]));
        let sc_raw1 = &raw1[off + 4..off + 16];
        let qs1 = raw1.as_ptr().add(off + 16);

        let xb0 = x0.get_unchecked(b);
        let xb1 = x1.get_unchecked(b);

        let mut sum_q = [[0i32; 2]; 2];
        let mut sum_m = [[0i32; 2]; 2];

        for chunk in 0..4 {
            let is = chunk * 2;
            let (s0_1, m0_1) = get_scale_min_k4(is, sc_raw0);
            let (s0_2, m0_2) = get_scale_min_k4(is + 1, sc_raw0);
            let (s1_1, m1_1) = get_scale_min_k4(is, sc_raw1);
            let (s1_2, m1_2) = get_scale_min_k4(is + 1, sc_raw1);

            let bs0_1 = *xb0.bsums.get_unchecked(is * 2) as i32 + *xb0.bsums.get_unchecked(is * 2 + 1) as i32;
            let bs0_2 = *xb0.bsums.get_unchecked((is + 1) * 2) as i32 + *xb0.bsums.get_unchecked((is + 1) * 2 + 1) as i32;
            let bs1_1 = *xb1.bsums.get_unchecked(is * 2) as i32 + *xb1.bsums.get_unchecked(is * 2 + 1) as i32;
            let bs1_2 = *xb1.bsums.get_unchecked((is + 1) * 2) as i32 + *xb1.bsums.get_unchecked((is + 1) * 2 + 1) as i32;

            sum_m[0][0] += (m0_1 as i32) * bs0_1 + (m0_2 as i32) * bs0_2;
            sum_m[0][1] += (m0_1 as i32) * bs1_1 + (m0_2 as i32) * bs1_2;
            sum_m[1][0] += (m1_1 as i32) * bs0_1 + (m1_2 as i32) * bs0_2;
            sum_m[1][1] += (m1_1 as i32) * bs1_1 + (m1_2 as i32) * bs1_2;

            let q0a = vld1q_u8(qs0.add(chunk * 32));
            let q0b = vld1q_u8(qs0.add(chunk * 32 + 16));
            let q1a = vld1q_u8(qs1.add(chunk * 32));
            let q1b = vld1q_u8(qs1.add(chunk * 32 + 16));

            let acc_lo = smmla_32(
                vreinterpretq_s8_u8(vandq_u8(q0a, mask_low)),
                vreinterpretq_s8_u8(vandq_u8(q0b, mask_low)),
                vreinterpretq_s8_u8(vandq_u8(q1a, mask_low)),
                vreinterpretq_s8_u8(vandq_u8(q1b, mask_low)),
                xb0.qs.as_ptr().add(chunk * 64),
                xb1.qs.as_ptr().add(chunk * 64),
            );
            let acc_hi = smmla_32(
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0a)),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0b)),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1a)),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1b)),
                xb0.qs.as_ptr().add(chunk * 64 + 32),
                xb1.qs.as_ptr().add(chunk * 64 + 32),
            );

            sum_q[0][0] += (s0_1 as i32) * vgetq_lane_s32::<0>(acc_lo) + (s0_2 as i32) * vgetq_lane_s32::<0>(acc_hi);
            sum_q[0][1] += (s0_1 as i32) * vgetq_lane_s32::<1>(acc_lo) + (s0_2 as i32) * vgetq_lane_s32::<1>(acc_hi);
            sum_q[1][0] += (s1_1 as i32) * vgetq_lane_s32::<2>(acc_lo) + (s1_2 as i32) * vgetq_lane_s32::<2>(acc_hi);
            sum_q[1][1] += (s1_1 as i32) * vgetq_lane_s32::<3>(acc_lo) + (s1_2 as i32) * vgetq_lane_s32::<3>(acc_hi);
        }

        out[0][0] += (dw0 * xb0.d) * (sum_q[0][0] as f32) - (dm0 * xb0.d) * (sum_m[0][0] as f32);
        out[0][1] += (dw0 * xb1.d) * (sum_q[0][1] as f32) - (dm0 * xb1.d) * (sum_m[0][1] as f32);
        out[1][0] += (dw1 * xb0.d) * (sum_q[1][0] as f32) - (dm1 * xb0.d) * (sum_m[1][0] as f32);
        out[1][1] += (dw1 * xb1.d) * (sum_q[1][1] as f32) - (dm1 * xb1.d) * (sum_m[1][1] as f32);
    }
    out
}

/// ARMv9 `dotprod` 4-token batched `Q4_K × Q8_K` prefill GEMM micro-kernel (unpacks weights once).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_q8_k_batch4_dotprod(
    raw: &[u8],
    x0: &[BlockQ8_K],
    x1: &[BlockQ8_K],
    x2: &[BlockQ8_K],
    x3: &[BlockQ8_K],
) -> [f32; 4] {
    let nb = x0.len();
    let mask_low = vdupq_n_u8(0x0F);
    let mut sumf = [0.0f32; 4];

    for b in 0..nb {
        let off = b * 144;
        let dw = f16_to_f32(u16::from_le_bytes([*raw.get_unchecked(off), *raw.get_unchecked(off + 1)]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([*raw.get_unchecked(off + 2), *raw.get_unchecked(off + 3)]));
        let scales = &raw[off + 4..off + 16];
        let qs_ptr = raw.as_ptr().add(off + 16);
        let xbs = [
            x0.get_unchecked(b),
            x1.get_unchecked(b),
            x2.get_unchecked(b),
            x3.get_unchecked(b),
        ];

        let mut sum_q = [0i32; 4];
        let mut sum_m = [0i32; 4];

        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            let q0 = vld1q_u8(qs_ptr.add(chunk * 32));
            let q1 = vld1q_u8(qs_ptr.add(chunk * 32 + 16));
            let q_lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask_low));
            let q_lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask_low));
            let q_hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
            let q_hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

            for t in 0..4 {
                let xb = xbs[t];
                sum_m[t] += (m1 as i32)
                    * (*xb.bsums.get_unchecked(is * 2) as i32
                        + *xb.bsums.get_unchecked(is * 2 + 1) as i32);
                sum_m[t] += (m2 as i32)
                    * (*xb.bsums.get_unchecked((is + 1) * 2) as i32
                        + *xb.bsums.get_unchecked((is + 1) * 2 + 1) as i32);

                let x_ptr = xb.qs.as_ptr().add(chunk * 64);
                let acc_lo = vdotq_s32(
                    vdotq_s32(vdupq_n_s32(0), q_lo0, vld1q_s8(x_ptr)),
                    q_lo1,
                    vld1q_s8(x_ptr.add(16)),
                );
                let acc_hi = vdotq_s32(
                    vdotq_s32(vdupq_n_s32(0), q_hi0, vld1q_s8(x_ptr.add(32))),
                    q_hi1,
                    vld1q_s8(x_ptr.add(48)),
                );
                sum_q[t] += (sc1 as i32) * vaddvq_s32(acc_lo) + (sc2 as i32) * vaddvq_s32(acc_hi);
            }
        }
        for t in 0..4 {
            let dx = xbs[t].d;
            sumf[t] += (dw * dx) * (sum_q[t] as f32) - (dmin_w * dx) * (sum_m[t] as f32);
        }
    }
    sumf
}
