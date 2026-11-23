//! Exact portable scalar reference kernels for all GGML quantization formats and transformer ops.

use crate::RopeStyle;
use caelum_core::{
    bf16_to_f32, f16_to_f32, quant::get_scale_min_k4, BlockQ8_0, BlockQ8_K, QK4_0, QK4_1, QK8_0,
    QK_K,
};

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let n = a.len();
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn dot_f32_bytes(raw: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, &xv) in raw.chunks_exact(4).zip(x.iter()) {
        let w = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum += w * xv;
    }
    sum
}

#[inline]
pub fn dot_f16_bytes(raw: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, &xv) in raw.chunks_exact(2).zip(x.iter()) {
        let w = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        sum += w * xv;
    }
    sum
}

#[inline]
pub fn dot_bf16_bytes(raw: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, &xv) in raw.chunks_exact(2).zip(x.iter()) {
        let w = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        sum += w * xv;
    }
    sum
}

/// Fused Q4_0 x F32 dot product.
pub fn dot_q4_0_f32(raw: &[u8], x: &[f32]) -> f32 {
    let nb = x.len() / QK4_0;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 18;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let qs = &raw[off + 2..off + 18];
        let xb = &x[b * 32..(b + 1) * 32];
        let mut acc = 0.0f32;
        for j in 0..16 {
            let byte = qs[j];
            let v0 = ((byte & 0x0F) as i32 - 8) as f32;
            let v1 = ((byte >> 4) as i32 - 8) as f32;
            acc += v0 * xb[j] + v1 * xb[j + 16];
        }
        sum += acc * d;
    }
    sum
}

/// Fused Q4_0 x Q8_0 integer dot product (W4A8 reference).
pub fn dot_q4_0_q8_0(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 18;
        let dw = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let dx = f16_to_f32(x[b].d);
        let qs = &raw[off + 2..off + 18];
        let xq = &x[b].qs;
        let mut isum = 0i32;
        for j in 0..16 {
            let byte = qs[j];
            let v0 = (byte & 0x0F) as i32 - 8;
            let v1 = (byte >> 4) as i32 - 8;
            isum += v0 * (xq[j] as i32) + v1 * (xq[j + 16] as i32);
        }
        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// Fused Q4_1 x F32 dot product.
pub fn dot_q4_1_f32(raw: &[u8], x: &[f32]) -> f32 {
    let nb = x.len() / QK4_1;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 20;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let qs = &raw[off + 4..off + 20];
        let xb = &x[b * 32..(b + 1) * 32];
        let mut acc_q = 0.0f32;
        let mut acc_x = 0.0f32;
        for j in 0..16 {
            let byte = qs[j];
            let x0 = xb[j];
            let x1 = xb[j + 16];
            acc_q += (byte & 0x0F) as f32 * x0 + (byte >> 4) as f32 * x1;
            acc_x += x0 + x1;
        }
        sum += acc_q * d + acc_x * m;
    }
    sum
}

/// Fused Q8_0 x F32 dot product.
pub fn dot_q8_0_f32(raw: &[u8], x: &[f32]) -> f32 {
    let nb = x.len() / QK8_0;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 34;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let qs = &raw[off + 2..off + 34];
        let xb = &x[b * 32..(b + 1) * 32];
        let mut acc = 0.0f32;
        for j in 0..32 {
            acc += (qs[j] as i8) as f32 * xb[j];
        }
        sum += acc * d;
    }
    sum
}

/// Fused Q8_0 x Q8_0 integer dot product (W8A8 reference).
pub fn dot_q8_0_q8_0(raw: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 34;
        let dw = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let dx = f16_to_f32(x[b].d);
        let qs = &raw[off + 2..off + 34];
        let xq = &x[b].qs;
        let mut isum = 0i32;
        for j in 0..32 {
            isum += (qs[j] as i8 as i32) * (xq[j] as i32);
        }
        sum += (isum as f32) * (dw * dx);
    }
    sum
}

/// Fused Q4_K x F32 dot product (used in Q4_K_M GGUF models).
pub fn dot_q4_k_f32(raw: &[u8], x: &[f32]) -> f32 {
    let nb = x.len() / QK_K;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 144;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let scales = &raw[off + 4..off + 16];
        let qs = &raw[off + 16..off + 144];
        let xb = &x[b * QK_K..(b + 1) * QK_K];

        let mut is = 0usize;
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            let q = &qs[chunk * 32..(chunk + 1) * 32];
            let x_lo = &xb[chunk * 64..chunk * 64 + 32];
            let x_hi = &xb[chunk * 64 + 32..chunk * 64 + 64];

            let mut acc_q1 = 0.0f32;
            let mut acc_x1 = 0.0f32;
            let mut acc_q2 = 0.0f32;
            let mut acc_x2 = 0.0f32;
            for l in 0..32 {
                let byte = q[l];
                acc_q1 += (byte & 0x0F) as f32 * x_lo[l];
                acc_x1 += x_lo[l];
                acc_q2 += (byte >> 4) as f32 * x_hi[l];
                acc_x2 += x_hi[l];
            }
            sum += d1 * acc_q1 - min1 * acc_x1 + d2 * acc_q2 - min2 * acc_x2;
            is += 2;
        }
    }
    sum
}

/// Fused Q6_K x F32 dot product (used in Q4_K_M & Q6_K GGUF models).
pub fn dot_q6_k_f32(raw: &[u8], x: &[f32]) -> f32 {
    let nb = x.len() / QK_K;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let off = b * 210;
        let ql_all = &raw[off..off + 128];
        let qh_all = &raw[off + 128..off + 192];
        let sc_all = &raw[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([raw[off + 208], raw[off + 209]]));
        let xb = &x[b * QK_K..(b + 1) * QK_K];

        let mut block_sum = 0.0f32;
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            let xh = &xb[half * 128..(half + 1) * 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;

                block_sum += (sc[is] as i8 as f32) * (q1 as f32) * xh[l]
                    + (sc[is + 2] as i8 as f32) * (q2 as f32) * xh[l + 32]
                    + (sc[is + 4] as i8 as f32) * (q3 as f32) * xh[l + 64]
                    + (sc[is + 6] as i8 as f32) * (q4 as f32) * xh[l + 96];
            }
        }
        sum += d * block_sum;
    }
    sum
}

/// Fused Q4_K x Q8_K integer dot product (W4A8 super-block reference).
pub fn dot_q4_k_q8_k(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mut sumf = 0.0f32;
    for b in 0..nb {
        let off = b * 144;
        let dw = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let scales = &raw[off + 4..off + 16];
        let qs = &raw[off + 16..off + 144];
        let xb = &x[b];

        let mut sum_q = 0i32;
        let mut sum_m = 0i32;

        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            sum_m += (m1 as i32) * (xb.bsums[is * 2] as i32 + xb.bsums[is * 2 + 1] as i32);
            sum_m +=
                (m2 as i32) * (xb.bsums[(is + 1) * 2] as i32 + xb.bsums[(is + 1) * 2 + 1] as i32);

            let q = &qs[chunk * 32..(chunk + 1) * 32];
            let x_lo = &xb.qs[chunk * 64..chunk * 64 + 32];
            let x_hi = &xb.qs[chunk * 64 + 32..chunk * 64 + 64];

            let mut acc_lo = 0i32;
            let mut acc_hi = 0i32;
            for l in 0..32 {
                let byte = q[l];
                acc_lo += ((byte & 0x0F) as i32) * (x_lo[l] as i32);
                acc_hi += ((byte >> 4) as i32) * (x_hi[l] as i32);
            }
            sum_q += (sc1 as i32) * acc_lo + (sc2 as i32) * acc_hi;
        }
        sumf += (dw * xb.d) * (sum_q as f32) - (dmin_w * xb.d) * (sum_m as f32);
    }
    sumf
}

/// Fused Q6_K x Q8_K integer dot product (W6A8 super-block reference).
pub fn dot_q6_k_q8_k(raw: &[u8], x: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    let mut sumf = 0.0f32;
    for b in 0..nb {
        let off = b * 210;
        let ql_all = &raw[off..off + 128];
        let qh_all = &raw[off + 128..off + 192];
        let sc_all = &raw[off + 192..off + 208];
        let dw = f16_to_f32(u16::from_le_bytes([raw[off + 208], raw[off + 209]]));
        let xb = &x[b];

        let mut isum = 0i32;
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            let xh = &xb.qs[half * 128..(half + 1) * 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;

                isum += (sc[is] as i8 as i32) * (q1 as i32) * (xh[l] as i32)
                    + (sc[is + 2] as i8 as i32) * (q2 as i32) * (xh[l + 32] as i32)
                    + (sc[is + 4] as i8 as i32) * (q3 as i32) * (xh[l + 64] as i32)
                    + (sc[is + 6] as i8 as i32) * (q4 as i32) * (xh[l + 96] as i32);
            }
        }
        sumf += (dw * xb.d) * (isum as f32);
    }
    sumf
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, out.len());
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Applies Rotary Position Embedding at exact sequence position `pos`.
pub fn apply_rope(vec: &mut [f32], pos: usize, rope_dim: usize, theta: f32, style: RopeStyle) {
    let half = rope_dim / 2;
    let pos_f = pos as f32;
    match style {
        RopeStyle::Normal => {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / rope_dim as f32);
                let angle = pos_f * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = vec[2 * i];
                let x1 = vec[2 * i + 1];
                vec[2 * i] = x0 * cos - x1 * sin;
                vec[2 * i + 1] = x0 * sin + x1 * cos;
            }
        }
        RopeStyle::NeoX => {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / rope_dim as f32);
                let angle = pos_f * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = vec[i];
                let x1 = vec[i + half];
                vec[i] = x0 * cos - x1 * sin;
                vec[i + half] = x0 * sin + x1 * cos;
            }
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        let silu = *g / (1.0 + (-*g).exp());
        *g = silu * u;
    }
}
