//! Exact GGML/GGUF quantization block representations and (de)quantization routines.
//!
//! Binary layouts match `llama.cpp` (`ggml-common.h`) byte-for-byte so any real
//! `.gguf` model (`Q4_0`, `Q4_1`, `Q8_0`, `Q4_K_M`, `Q6_K`, `F16`, `BF16`, `F32`)
//! loads with zero copy via `mmap`.

use crate::dtype::{f16_to_f32, f32_to_f16};

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

/// GGML `block_q4_0`: 32 weights in 18 bytes (4.5 bpw).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: u16,        // f16 delta
    pub qs: [u8; 16],  // 32 x 4-bit nibbles (low = 0..16, high = 16..32)
}

/// GGML `block_q4_1`: 32 weights in 20 bytes (5.0 bpw).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: u16,        // f16 delta
    pub m: u16,        // f16 min
    pub qs: [u8; 16],  // 32 x 4-bit nibbles
}

/// GGML `block_q8_0`: 32 weights in 34 bytes (8.5 bpw).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: u16,        // f16 delta
    pub qs: [i8; 32],  // 32 x int8 quants
}

/// GGML `block_q4_K`: 256 weights in 144 bytes (4.5 bpw super-block, used in Q4_K_M).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_K {
    pub d: u16,           // f16 super-block scale for quantized scales
    pub dmin: u16,        // f16 super-block scale for quantized mins
    pub scales: [u8; 12], // 8 x (6-bit scale + 6-bit min)
    pub qs: [u8; 128],    // 256 x 4-bit quants
}

/// GGML `block_q5_K`: 256 weights in 176 bytes (5.5 bpw super-block, used in Q5_K_M / Q5_K_S).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5_K {
    pub d: u16,           // f16 super-block scale for quantized scales
    pub dmin: u16,        // f16 super-block scale for quantized mins
    pub scales: [u8; 12], // 8 x (6-bit scale + 6-bit min)
    pub qh: [u8; 32],     // 256 x 1-bit high bits (8 bits per byte, 2 bits per 64-elem chunk)
    pub qs: [u8; 128],    // 256 x 4-bit low nibbles
}

/// GGML `block_q6_K`: 256 weights in 210 bytes (6.56 bpw super-block, used in Q4_K_M & Q6_K).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6_K {
    pub ql: [u8; 128],   // lower 4 bits of quants
    pub qh: [u8; 64],    // upper 2 bits of quants
    pub scales: [i8; 16], // 16 x int8 scales
    pub d: u16,          // f16 super-block scale
}

/// GGML `block_q8_K`: 256 dynamic activation quants + 16 sub-block sums (used for SIMD Q4_K/Q6_K).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

impl Default for BlockQ8_K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }
    }
}

const _: () = {
    assert!(std::mem::size_of::<BlockQ4_0>() == 18);
    assert!(std::mem::size_of::<BlockQ4_1>() == 20);
    assert!(std::mem::size_of::<BlockQ8_0>() == 34);
    assert!(std::mem::size_of::<BlockQ4_K>() == 144);
    assert!(std::mem::size_of::<BlockQ5_K>() == 176);
    assert!(std::mem::size_of::<BlockQ6_K>() == 210);
};

/// Quantizes an `f32` row (multiple of 256) into `Q8_K` super-blocks for fast SIMD `Q4_K`/`Q6_K` matvec.
pub fn quantize_row_q8_k(src: &[f32], dst: &mut [BlockQ8_K]) {
    debug_assert_eq!(src.len(), dst.len() * QK_K);
    for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(QK_K)) {
        let mut amax = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            block.bsums.fill(0);
            continue;
        }
        let d = amax / 127.0;
        let id = 1.0 / d;
        block.d = d;
        for g in 0..16 {
            let base = g * 16;
            let mut sum = 0i16;
            for l in 0..16 {
                let q = (chunk[base + l] * id).round().clamp(-127.0, 127.0) as i8;
                block.qs[base + l] = q;
                sum += q as i16;
            }
            block.bsums[g] = sum;
        }
    }
}

/// Quantizes an `f32` row into `Q8_0` blocks (used for W4A8/W8A8 dynamic activation quantization).
pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    debug_assert_eq!(src.len(), dst.len() * QK8_0);
    for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(QK8_0)) {
        let mut amax = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        for (i, &v) in chunk.iter().enumerate() {
            qs[i] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
        block.d = f32_to_f16(d);
        block.qs = qs;
    }
}

/// Quantizes an `f32` row into `Q4_0` blocks (exact GGML nibble packing).
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    debug_assert_eq!(src.len(), dst.len() * QK4_0);
    for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(QK4_0)) {
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if a > amax {
                amax = a;
                max_val = v;
            }
        }
        let d = max_val / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = chunk[j] * id;
            let x1 = chunk[j + 16] * id;
            let xi0 = (x0 + 8.5).clamp(0.0, 15.0) as u8;
            let xi1 = (x1 + 8.5).clamp(0.0, 15.0) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        }
        block.d = f32_to_f16(d);
        block.qs = qs;
    }
}

/// Dequantizes a `Q4_0` byte slice into `dst: &mut [f32]`.
pub fn dequantize_row_q4_0(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK4_0;
    debug_assert_eq!(raw.len(), nb * 18);
    for b in 0..nb {
        let off = b * 18;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let qs = &raw[off + 2..off + 18];
        let out = &mut dst[b * 32..(b + 1) * 32];
        for j in 0..16 {
            let byte = qs[j];
            out[j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            out[j + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
}

/// Dequantizes a `Q4_1` byte slice into `dst: &mut [f32]`.
pub fn dequantize_row_q4_1(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK4_1;
    debug_assert_eq!(raw.len(), nb * 20);
    for b in 0..nb {
        let off = b * 20;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let qs = &raw[off + 4..off + 20];
        let out = &mut dst[b * 32..(b + 1) * 32];
        for j in 0..16 {
            let byte = qs[j];
            out[j] = (byte & 0x0F) as f32 * d + m;
            out[j + 16] = (byte >> 4) as f32 * d + m;
        }
    }
}

/// Dequantizes a `Q8_0` byte slice into `dst: &mut [f32]`.
pub fn dequantize_row_q8_0(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK8_0;
    debug_assert_eq!(raw.len(), nb * 34);
    for b in 0..nb {
        let off = b * 34;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let qs = &raw[off + 2..off + 34];
        let out = &mut dst[b * 32..(b + 1) * 32];
        for j in 0..32 {
            out[j] = (qs[j] as i8) as f32 * d;
        }
    }
}

#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes a `Q4_K` byte slice into `dst: &mut [f32]`.
pub fn dequantize_row_q4_k(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK_K;
    debug_assert_eq!(raw.len(), nb * 144);
    for b in 0..nb {
        let off = b * 144;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let scales = &raw[off + 4..off + 16];
        let qs = &raw[off + 16..off + 144];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        let mut is = 0usize;
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;

            let q = &qs[chunk * 32..(chunk + 1) * 32];
            let y_base = chunk * 64;
            for l in 0..32 {
                out[y_base + l] = d1 * (q[l] & 0x0F) as f32 - min1;
                out[y_base + 32 + l] = d2 * (q[l] >> 4) as f32 - min2;
            }
            is += 2;
        }
    }
}

/// Dequantizes a `Q5_K` byte slice into `dst: &mut [f32]` (matches `dequantize_row_q5_K` bit-for-bit).
pub fn dequantize_row_q5_k(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK_K;
    debug_assert_eq!(raw.len(), nb * 176);
    for b in 0..nb {
        let off = b * 176;
        let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([raw[off + 2], raw[off + 3]]));
        let scales = &raw[off + 4..off + 16];
        let qh = &raw[off + 16..off + 48];
        let qs = &raw[off + 48..off + 176];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = min * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = min * m2 as f32;

            let q = &qs[chunk * 32..(chunk + 1) * 32];
            let y_base = chunk * 64;
            for l in 0..32 {
                let hi1 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi2 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[y_base + l] = d1 * ((q[l] & 0x0F) + hi1) as f32 - min1;
                out[y_base + 32 + l] = d2 * ((q[l] >> 4) + hi2) as f32 - min2;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Dequantizes a `Q6_K` byte slice into `dst: &mut [f32]`.
pub fn dequantize_row_q6_k(raw: &[u8], dst: &mut [f32]) {
    let nb = dst.len() / QK_K;
    debug_assert_eq!(raw.len(), nb * 210);
    for b in 0..nb {
        let off = b * 210;
        let ql_all = &raw[off..off + 128];
        let qh_all = &raw[off + 128..off + 192];
        let sc_all = &raw[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([raw[off + 208], raw[off + 209]]));
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            let y = &mut out[half * 128..(half + 1) * 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;

                y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_roundtrip() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 2];
        quantize_row_q8_0(&src, &mut blocks);
        let raw = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 34)
        };
        let mut dst = vec![0.0f32; 64];
        dequantize_row_q8_0(raw, &mut dst);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 0.03, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_q4_0_roundtrip() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; 2];
        quantize_row_q4_0(&src, &mut blocks);
        let raw = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 18)
        };
        let mut dst = vec![0.0f32; 64];
        dequantize_row_q4_0(raw, &mut dst);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 0.25, "a={a}, b={b}");
        }
    }
}
