//! GGUF metadata value representation supporting all standard GGUF v2/v3 types.

use std::fmt;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array(_) => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|v| v as usize)
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GgufValue::F32(v) => Some(v),
            GgufValue::F64(v) => Some(v as f32),
            GgufValue::U32(v) => Some(v as f32),
            GgufValue::I32(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            GgufValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn as_string_vec(&self) -> Option<Vec<String>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_str()?.to_string());
        }
        Some(out)
    }

    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_f32()?);
        }
        Some(out)
    }

    pub fn as_i32_vec(&self) -> Option<Vec<i32>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match *item {
                GgufValue::I32(v) => out.push(v),
                GgufValue::U32(v) => out.push(v as i32),
                _ => return None,
            }
        }
        Some(out)
    }

    pub fn as_usize_vec(&self) -> Option<Vec<usize>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_usize()?);
        }
        Some(out)
    }

    pub fn as_bool_vec(&self) -> Option<Vec<bool>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_bool()?);
        }
        Some(out)
    }
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufValue::U8(v) => write!(f, "{v}"),
            GgufValue::I8(v) => write!(f, "{v}"),
            GgufValue::U16(v) => write!(f, "{v}"),
            GgufValue::I16(v) => write!(f, "{v}"),
            GgufValue::U32(v) => write!(f, "{v}"),
            GgufValue::I32(v) => write!(f, "{v}"),
            GgufValue::F32(v) => write!(f, "{v}"),
            GgufValue::Bool(v) => write!(f, "{v}"),
            GgufValue::String(s) => {
                if s.len() > 80 {
                    write!(f, "\"{}...\" ({} bytes)", &s[..80], s.len())
                } else {
                    write!(f, "\"{s}\"")
                }
            }
            GgufValue::Array(arr) => write!(f, "[Array len={}]", arr.len()),
            GgufValue::U64(v) => write!(f, "{v}"),
            GgufValue::I64(v) => write!(f, "{v}"),
            GgufValue::F64(v) => write!(f, "{v}"),
        }
    }
}
