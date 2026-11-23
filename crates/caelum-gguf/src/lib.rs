//! Zero-copy GGUF v2/v3 parser, `mmap` backing store, and GGUF v3 writer.

pub mod mmap;
pub mod reader;
pub mod value;
pub mod writer;

pub use mmap::MappedFile;
pub use reader::{GgufFile, TensorInfo};
pub use value::GgufValue;
pub use writer::GgufWriter;
