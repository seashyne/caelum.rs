//! Zero-copy GGUF v2/v3 reader.

use crate::mmap::MappedFile;
use crate::value::GgufValue;
use caelum_core::{CaelumError, DType, Result, TensorView};
use std::collections::BTreeMap;
use std::path::Path;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML-order dimensions `[ne0, ne1, ...]` where `ne0` is contiguous.
    pub ggml_dims: Vec<usize>,
    pub dtype: DType,
    /// Byte offset relative to `GgufFile::data_offsets[shard]`.
    pub offset: usize,
    pub byte_size: usize,
    /// Backing shard index (`0` for single-file GGUF, `0..split_count` for multi-shard split GGUF).
    pub shard: usize,
}

impl TensorInfo {
    /// Standard row-major dimensions `[outer, ..., inner]` matching `TensorView`.
    pub fn row_major_dims(&self) -> Vec<usize> {
        let mut d = self.ggml_dims.clone();
        d.reverse();
        d
    }

    pub fn num_elements(&self) -> usize {
        self.ggml_dims.iter().product()
    }
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub alignment: usize,
    pub data_offset: usize,
    pub split_count: usize,
    data_offsets: Vec<usize>,
    backings: Vec<MappedFile>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(CaelumError::InvalidFormat(format!(
                "unexpected EOF at offset {} (needed {n} bytes, total {})",
                self.pos,
                self.buf.len()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_value(&mut self, val_type: u32) -> Result<GgufValue> {
        match val_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len.min(1 << 20));
                for _ in 0..len {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(CaelumError::InvalidFormat(format!(
                "unknown GGUF metadata type id={other}"
            ))),
        }
    }
}

#[inline]
fn align_up(offset: usize, alignment: usize) -> usize {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

/// Resolves a sibling split-GGUF shard path matching `*-NNNNN-of-MMMMM.gguf` (1-indexed).
fn resolve_split_shard_path(
    base_path: &Path,
    shard_idx: usize,
    split_count: usize,
) -> Option<std::path::PathBuf> {
    let fname = base_path.file_name()?.to_str()?;
    let stem = fname.strip_suffix(".gguf")?;
    // Expected suffix: `-NNNNN-of-MMMMM` (15 ASCII bytes)
    if stem.len() >= 15 {
        let split_pos = stem.len() - 15;
        let suffix = &stem[split_pos..];
        if suffix.starts_with('-')
            && &suffix[6..10] == "-of-"
            && suffix[1..6].bytes().all(|b| b.is_ascii_digit())
            && suffix[10..15].bytes().all(|b| b.is_ascii_digit())
        {
            let prefix = &stem[..split_pos];
            let shard_name = format!("{prefix}-{:05}-of-{:05}.gguf", shard_idx + 1, split_count);
            return Some(base_path.with_file_name(shard_name));
        }
    }
    None
}

type ParsedGgufParts = (
    u32,
    BTreeMap<String, GgufValue>,
    BTreeMap<String, TensorInfo>,
    usize,
    usize,
);

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let first_backing = MappedFile::open(path)?;
        let (version, metadata, tensors, alignment, data_offset) =
            Self::parse_parts(first_backing.as_slice(), 0)?;

        let split_count = metadata
            .get("split.count")
            .and_then(GgufValue::as_usize)
            .unwrap_or(1)
            .max(1);
        let split_no = metadata
            .get("split.no")
            .and_then(GgufValue::as_usize)
            .unwrap_or(0);

        // If user passed shard > 0 (e.g. -00002-of-00002.gguf), transparently pivot to shard 0
        // so full model metadata & tokenizer KVs are loaded properly.
        if split_count > 1 && split_no > 0 {
            if let Some(shard0_path) = resolve_split_shard_path(path, 0, split_count) {
                if shard0_path.exists() {
                    return Self::open(shard0_path);
                }
            }
            return Err(CaelumError::InvalidFormat(format!(
                "GGUF file is split shard {}/{} and shard 1 could not be found beside '{}'",
                split_no + 1,
                split_count,
                path.display()
            )));
        }

        let mut all_tensors = tensors;
        let mut backings = vec![first_backing];
        let mut data_offsets = vec![data_offset];

        if split_count > 1 {
            for shard_idx in 1..split_count {
                let shard_path = resolve_split_shard_path(path, shard_idx, split_count)
                    .ok_or_else(|| {
                        CaelumError::InvalidFormat(format!(
                            "cannot infer split shard {}/{} filename from '{}'",
                            shard_idx + 1,
                            split_count,
                            path.display()
                        ))
                    })?;
                if !shard_path.exists() {
                    return Err(CaelumError::InvalidFormat(format!(
                        "missing GGUF split shard {}/{}: '{}'",
                        shard_idx + 1,
                        split_count,
                        shard_path.display()
                    )));
                }
                let shard_backing = MappedFile::open(&shard_path)?;
                let (_, _, shard_tensors, _, shard_data_offset) =
                    Self::parse_parts(shard_backing.as_slice(), shard_idx)?;
                for (name, info) in shard_tensors {
                    all_tensors.insert(name, info);
                }
                backings.push(shard_backing);
                data_offsets.push(shard_data_offset);
            }
        }

        Ok(Self {
            version,
            metadata,
            tensors: all_tensors,
            alignment,
            data_offset,
            split_count,
            data_offsets,
            backings,
        })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let backing = MappedFile::from_bytes(bytes);
        let (version, metadata, tensors, alignment, data_offset) =
            Self::parse_parts(backing.as_slice(), 0)?;
        Ok(Self {
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
            split_count: 1,
            data_offsets: vec![data_offset],
            backings: vec![backing],
        })
    }

    fn parse_parts(buf: &[u8], shard: usize) -> Result<ParsedGgufParts> {
        let mut cur = Cursor::new(buf);
        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(CaelumError::InvalidFormat(format!(
                "bad GGUF magic: {:?}",
                magic
            )));
        }
        let version = cur.read_u32()?;
        if version != 2 && version != 3 {
            return Err(CaelumError::InvalidFormat(format!(
                "unsupported GGUF version {version} (expected 2 or 3)"
            )));
        }

        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let val_type = cur.read_u32()?;
            let val = cur.read_value(val_type)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut ggml_dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ggml_dims.push(cur.read_u64()? as usize);
            }
            let dtype_id = cur.read_u32()?;
            let dtype = DType::from_ggml_id(dtype_id)?;
            let offset = cur.read_u64()? as usize;
            let num_elems: usize = ggml_dims.iter().product();
            let byte_size = dtype.byte_size_for(num_elems)?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    ggml_dims,
                    dtype,
                    offset,
                    byte_size,
                    shard,
                },
            );
        }

        let data_offset = align_up(cur.pos, alignment);
        Ok((version, metadata, tensors, alignment, data_offset))
    }

    pub fn file_size(&self) -> usize {
        self.backings.iter().map(|b| b.as_slice().len()).sum()
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn require_usize(&self, key: &str) -> Result<usize> {
        self.get_meta(key)
            .and_then(GgufValue::as_usize)
            .ok_or_else(|| CaelumError::MissingMetadata(key.to_string()))
    }

    pub fn require_f32(&self, key: &str) -> Result<f32> {
        self.get_meta(key)
            .and_then(GgufValue::as_f32)
            .ok_or_else(|| CaelumError::MissingMetadata(key.to_string()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Returns a zero-copy `TensorView` borrowing directly from the `mmap`'d shard file.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| CaelumError::MissingTensor(name.to_string()))?;
        let backing = self.backings.get(info.shard).ok_or_else(|| {
            CaelumError::InvalidFormat(format!(
                "tensor '{name}' references invalid shard {}",
                info.shard
            ))
        })?;
        let start = self.data_offsets[info.shard] + info.offset;
        let end = start + info.byte_size;
        if end > backing.len() {
            return Err(CaelumError::InvalidFormat(format!(
                "tensor '{name}' exceeds shard {} bounds ({end} > {})",
                info.shard,
                backing.len()
            )));
        }
        let dims = info.row_major_dims();
        TensorView::new(&backing[start..end], &dims, info.dtype)
    }

    pub fn total_parameters(&self) -> u64 {
        self.tensors.values().map(|t| t.num_elements() as u64).sum()
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::writer::GgufWriter;

    #[test]
    fn test_split_gguf_two_shards_zero_copy_open() {
        let dir = std::env::temp_dir().join(format!("caelum_split_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard1 = dir.join("demo-00001-of-00002.gguf");
        let shard2 = dir.join("demo-00002-of-00002.gguf");

        let mut w1 = GgufWriter::new();
        w1.add_meta("general.architecture", GgufValue::String("qwen2".into()));
        w1.add_meta("split.no", GgufValue::U16(0));
        w1.add_meta("split.count", GgufValue::U16(2));
        w1.add_tensor_f32("t0", vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        w1.write_to_file(&shard1).unwrap();

        let mut w2 = GgufWriter::new();
        w2.add_meta("split.no", GgufValue::U16(1));
        w2.add_meta("split.count", GgufValue::U16(2));
        w2.add_tensor_f32("t1", vec![4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        w2.write_to_file(&shard2).unwrap();

        // Opening shard 1 or shard 2 yields both tensors seamlessly
        for p in [&shard1, &shard2] {
            let gguf = GgufFile::open(p).unwrap();
            assert_eq!(gguf.split_count, 2);
            assert_eq!(gguf.architecture(), Some("qwen2"));
            assert_eq!(gguf.tensor("t0").unwrap().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(gguf.tensor("t1").unwrap().to_f32_vec().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
