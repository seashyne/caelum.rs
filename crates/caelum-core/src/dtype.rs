//! Data types matching GGML/GGUF tensor type IDs + exact IEEE-754 F16/BF16 conversion.

use crate::error::{CaelumError, Result};
use std::fmt;

/// Tensor data type matching GGML/GGUF `ggml_type` numeric IDs.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
}

impl DType {
    pub fn from_ggml_id(id: u32) -> Result<Self> {
        match id {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            2 => Ok(DType::Q4_0),
            3 => Ok(DType::Q4_1),
            8 => Ok(DType::Q8_0),
            12 => Ok(DType::Q4_K),
            13 => Ok(DType::Q5_K),
            14 => Ok(DType::Q6_K),
            30 => Ok(DType::BF16),
            other => Err(CaelumError::UnsupportedDType(format!(
                "ggml_type id={other}"
            ))),
        }
    }

    /// Number of logical elements per quantization block.
    #[inline]
    pub const fn block_size(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => 32,
            DType::Q4_K | DType::Q5_K | DType::Q6_K => 256,
        }
    }

    /// Byte size of one quantization block.
    #[inline]
    pub const fn type_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4_0 => 18,
            DType::Q4_1 => 20,
            DType::Q8_0 => 34,
            DType::Q4_K => 144,
            DType::Q5_K => 176,
            DType::Q6_K => 210,
        }
    }

    /// Computes exact byte size required for `num_elements` elements.
    #[inline]
    pub fn byte_size_for(self, num_elements: usize) -> Result<usize> {
        let bs = self.block_size();
        if num_elements % bs != 0 {
            return Err(CaelumError::ShapeMismatch {
                expected: format!("element count divisible by block size {bs} for {self}"),
                actual: num_elements.to_string(),
            });
        }
        Ok((num_elements / bs) * self.type_size())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::Q4_0 => write!(f, "Q4_0"),
            DType::Q4_1 => write!(f, "Q4_1"),
            DType::Q8_0 => write!(f, "Q8_0"),
            DType::Q4_K => write!(f, "Q4_K"),
            DType::Q5_K => write!(f, "Q5_K"),
            DType::Q6_K => write!(f, "Q6_K"),
        }
    }
}

/// Exact IEEE-754 binary16 (`u16`) to `f32` conversion (zero external dependencies).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16 -> normalized f32
        let mut e = 127 - 14;
        let mut f = frac;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (f << 13));
    }
    if exp == 31 {
        // Inf or NaN
        return f32::from_bits(sign | 0x7F80_0000 | (frac << 13));
    }
    f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (frac << 13))
}

/// Exact `f32` to IEEE-754 binary16 (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf / NaN
        return sign | 0x7C00 | if frac != 0 { 0x0200 | ((frac >> 13) as u16) } else { 0 };
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow -> Inf
        return sign | 0x7C00;
    }
    if unbiased < -14 {
        if unbiased < -25 {
            return sign;
        }
        // Subnormal
        let mant = frac | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let half = 1u32 << (shift - 1);
        let rounded = (mant + half + ((mant >> shift) & 1) - 1) >> shift;
        return sign | (rounded as u16);
    }

    let h_exp = ((unbiased + 15) as u16) << 10;
    let lsb = (frac >> 13) & 1;
    let round_bias = 0x0FFF + lsb;
    let rounded_frac = frac + round_bias;
    if (rounded_frac & 0x0080_0000) != 0 {
        // Mantissa overflow increments exponent
        sign | (h_exp + 0x0400)
    } else {
        sign | h_exp | ((rounded_frac >> 13) as u16)
    }
}

/// Exact `bf16` (`u16`) to `f32` conversion.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `f32` to `bf16` (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        for &x in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 3.140625, -42.25, 1e-4, 65504.0] {
            let h = f32_to_f16(x);
            let y = f16_to_f32(h);
            assert!((x - y).abs() <= 1e-2, "x={x}, y={y}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_bf16_roundtrip() {
        for &x in &[0.0f32, 1.0, -2.5, 128.0] {
            let b = f32_to_bf16(x);
            let y = bf16_to_f32(b);
            assert!((x - y).abs() < 1e-2);
        }
    }
}
