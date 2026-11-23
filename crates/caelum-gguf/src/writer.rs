//! GGUF v3 binary writer (for exporting Celestia models, repacking weights, and end-to-end tests).

use crate::reader::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::value::GgufValue;
use caelum_core::{DType, Result};
use std::io::Write;
use std::path::Path;

struct PendingTensor {
    name: String,
    /// Stored in row-major `[outer, ..., inner]`, written reversed as GGML `[ne0, ne1, ...]`.
    row_major_dims: Vec<usize>,
    dtype: DType,
    data: Vec<u8>,
}

pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<PendingTensor>,
    alignment: usize,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
        }
    }

    pub fn add_meta(&mut self, key: impl Into<String>, value: GgufValue) {
        self.metadata.push((key.into(), value));
    }

    pub fn add_tensor_raw(
        &mut self,
        name: impl Into<String>,
        row_major_dims: Vec<usize>,
        dtype: DType,
        data: Vec<u8>,
    ) -> Result<()> {
        let num_elems: usize = row_major_dims.iter().product();
        let expected = dtype.byte_size_for(num_elems)?;
        assert_eq!(
            data.len(),
            expected,
            "tensor byte length mismatch for dims {:?} dtype {}",
            row_major_dims,
            dtype
        );
        self.tensors.push(PendingTensor {
            name: name.into(),
            row_major_dims,
            dtype,
            data,
        });
        Ok(())
    }

    pub fn add_tensor_f32(
        &mut self,
        name: impl Into<String>,
        row_major_dims: Vec<usize>,
        values: &[f32],
    ) -> Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.add_tensor_raw(name, row_major_dims, DType::F32, bytes)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes()); // GGUF v3
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for (k, v) in &self.metadata {
            write_str(&mut out, k);
            out.extend_from_slice(&v.type_id().to_le_bytes());
            write_value(&mut out, v);
        }

        // Compute aligned relative offsets for each tensor
        let mut rel_offset = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(rel_offset);
            rel_offset = align_up(rel_offset + t.data.len(), self.alignment);
        }

        for (t, &off) in self.tensors.iter().zip(offsets.iter()) {
            write_str(&mut out, &t.name);
            out.extend_from_slice(&(t.row_major_dims.len() as u32).to_le_bytes());
            // Reverse row-major to GGML order [ne0, ne1, ...]
            for &d in t.row_major_dims.iter().rev() {
                out.extend_from_slice(&(d as u64).to_le_bytes());
            }
            out.extend_from_slice(&(t.dtype as u32).to_le_bytes());
            out.extend_from_slice(&(off as u64).to_le_bytes());
        }

        // Pad header to alignment
        let data_start = align_up(out.len(), self.alignment);
        out.resize(data_start, 0);

        for (t, &off) in self.tensors.iter().zip(offsets.iter()) {
            let abs_pos = data_start + off;
            if out.len() < abs_pos {
                out.resize(abs_pos, 0);
            }
            out.extend_from_slice(&t.data);
        }

        out
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes();
        let mut f = std::fs::File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }
}

fn align_up(offset: usize, alignment: usize) -> usize {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => out.push(u8::from(*x)),
        GgufValue::String(s) => write_str(out, s),
        GgufValue::Array(arr) => {
            let elem_type = arr.first().map(GgufValue::type_id).unwrap_or(8);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            for item in arr {
                write_value(out, item);
            }
        }
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::GgufFile;

    #[test]
    fn test_gguf_write_and_parse_roundtrip() {
        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("celestia".into()));
        w.add_meta("celestia.embedding_length", GgufValue::U32(64));
        w.add_tensor_f32("test.weight", vec![4, 8], &(0..32).map(|i| i as f32).collect::<Vec<_>>())
            .unwrap();

        let bytes = w.to_bytes();
        let gf = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(gf.version, 3);
        assert_eq!(gf.architecture(), Some("celestia"));
        assert_eq!(gf.require_usize("celestia.embedding_length").unwrap(), 64);

        let tv = gf.tensor("test.weight").unwrap();
        assert_eq!(tv.dims(), &[4, 8]);
        let vals = tv.to_f32_vec().unwrap();
        assert_eq!(vals.len(), 32);
        assert_eq!(vals[31], 31.0);
    }
}
