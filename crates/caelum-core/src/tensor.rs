//! Zero-copy borrowed tensor view (`TensorView`) and owned tensor buffer (`TensorBuf`).

use crate::dtype::{bf16_to_f32, f16_to_f32, DType};
use crate::error::{CaelumError, Result};
use crate::quant::{
    dequantize_row_q4_0, dequantize_row_q4_1, dequantize_row_q4_k, dequantize_row_q6_k,
    dequantize_row_q8_0,
};

/// A zero-copy read-only view over tensor bytes (typically borrowed from an `mmap`'d GGUF file).
///
/// Shape follows standard row-major convention `[outer_dim, ..., inner_dim]` where `inner_dim`
/// is contiguous in memory (note: GGUF stores dimensions reversed `[ne0, ne1, ...]`, which
/// `caelum-gguf` converts to row-major `[rows, cols]` when constructing `TensorView`).
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub data: &'a [u8],
    pub shape: [usize; 4],
    pub ndim: usize,
    pub dtype: DType,
}

impl<'a> TensorView<'a> {
    pub fn new(data: &'a [u8], dims: &[usize], dtype: DType) -> Result<Self> {
        if dims.is_empty() || dims.len() > 4 {
            return Err(CaelumError::ShapeMismatch {
                expected: "1..=4 dimensions".into(),
                actual: format!("{}", dims.len()),
            });
        }
        let mut shape = [1usize; 4];
        let mut num_elems = 1usize;
        for (i, &d) in dims.iter().enumerate() {
            shape[i] = d;
            num_elems = num_elems.checked_mul(d).ok_or_else(|| {
                CaelumError::InvalidFormat("tensor element count overflow".into())
            })?;
        }
        let expected_bytes = dtype.byte_size_for(num_elems)?;
        if data.len() < expected_bytes {
            return Err(CaelumError::InvalidFormat(format!(
                "tensor buffer too small: need {expected_bytes} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data: &data[..expected_bytes],
            shape,
            ndim: dims.len(),
            dtype,
        })
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    #[inline]
    pub fn num_elements(&self) -> usize {
        self.dims().iter().product()
    }

    /// For a 2D weight matrix `[out_features, in_features]`, returns `(out_features, in_features)`.
    pub fn dims_2d(&self) -> Result<(usize, usize)> {
        if self.ndim != 2 {
            return Err(CaelumError::ShapeMismatch {
                expected: "2D tensor".into(),
                actual: format!("{}D tensor {:?}", self.ndim, self.dims()),
            });
        }
        Ok((self.shape[0], self.shape[1]))
    }

    /// Returns the raw byte slice for row `row_idx` of a 2D `[rows, cols]` tensor.
    #[inline]
    pub fn row_bytes(&self, row_idx: usize) -> &[u8] {
        let cols = self.shape[1];
        let row_bytes = (cols / self.dtype.block_size()) * self.dtype.type_size();
        let start = row_idx * row_bytes;
        &self.data[start..start + row_bytes]
    }

    /// Dequantizes a single row `row_idx` of a 2D `[rows, cols]` tensor into `dst` (length `cols`).
    pub fn dequantize_row(&self, row_idx: usize, dst: &mut [f32]) -> Result<()> {
        let raw = self.row_bytes(row_idx);
        dequantize_slice(raw, self.dtype, dst)
    }

    /// Dequantizes the entire tensor into an owned `Vec<f32>` (useful for 1D norms/biases).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.num_elements()];
        dequantize_slice(self.data, self.dtype, &mut out)?;
        Ok(out)
    }
}

/// Owned tensor buffer (used for in-memory weights, synthetic models, or repacked buffers).
#[derive(Clone, Debug)]
pub struct TensorBuf {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorBuf {
    pub fn from_f32(values: &[f32], shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(CaelumError::ShapeMismatch {
                expected: expected.to_string(),
                actual: values.len().to_string(),
            });
        }
        let mut data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            data,
            shape,
            dtype: DType::F32,
        })
    }

    pub fn from_raw(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let num_elems: usize = shape.iter().product();
        let expected = dtype.byte_size_for(num_elems)?;
        if data.len() != expected {
            return Err(CaelumError::InvalidFormat(format!(
                "raw tensor size mismatch: expected {expected}, got {}",
                data.len()
            )));
        }
        Ok(Self { data, shape, dtype })
    }

    pub fn as_view(&self) -> TensorView<'_> {
        TensorView::new(&self.data, &self.shape, self.dtype)
            .expect("TensorBuf invariants already validated")
    }
}

/// Dequantizes a contiguous slice of elements of `dtype` into `dst`.
pub fn dequantize_slice(raw: &[u8], dtype: DType, dst: &mut [f32]) -> Result<()> {
    match dtype {
        DType::F32 => {
            for (o, chunk) in dst.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        DType::F16 => {
            for (o, chunk) in dst.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        DType::BF16 => {
            for (o, chunk) in dst.iter_mut().zip(raw.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        DType::Q4_0 => dequantize_row_q4_0(raw, dst),
        DType::Q4_1 => dequantize_row_q4_1(raw, dst),
        DType::Q8_0 => dequantize_row_q8_0(raw, dst),
        DType::Q4_K => dequantize_row_q4_k(raw, dst),
        DType::Q6_K => dequantize_row_q6_k(raw, dst),
    }
    Ok(())
}
