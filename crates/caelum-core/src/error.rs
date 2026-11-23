use std::fmt;

#[derive(Debug)]
pub enum CaelumError {
    Io(std::io::Error),
    InvalidFormat(String),
    UnsupportedDType(String),
    ShapeMismatch { expected: String, actual: String },
    MissingTensor(String),
    MissingMetadata(String),
    Tokenizer(String),
    Runtime(String),
}

impl fmt::Display for CaelumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaelumError::Io(e) => write!(f, "I/O error: {e}"),
            CaelumError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            CaelumError::UnsupportedDType(msg) => write!(f, "Unsupported dtype: {msg}"),
            CaelumError::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected}, got {actual}")
            }
            CaelumError::MissingTensor(name) => write!(f, "Missing tensor: {name}"),
            CaelumError::MissingMetadata(key) => write!(f, "Missing metadata key: {key}"),
            CaelumError::Tokenizer(msg) => write!(f, "Tokenizer error: {msg}"),
            CaelumError::Runtime(msg) => write!(f, "Runtime error: {msg}"),
        }
    }
}

impl std::error::Error for CaelumError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CaelumError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CaelumError {
    fn from(err: std::io::Error) -> Self {
        CaelumError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, CaelumError>;
