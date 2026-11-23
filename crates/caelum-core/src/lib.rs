//! Core types, exact IEEE-754 half precision, GGML/GGUF quantization layouts, and tensor views.

pub mod dtype;
pub mod error;
pub mod quant;
pub mod tensor;

pub use dtype::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, DType};
pub use error::{CaelumError, Result};
pub use quant::{
    dequantize_row_q4_0, dequantize_row_q4_1, dequantize_row_q4_k, dequantize_row_q6_k,
    dequantize_row_q8_0, quantize_row_q4_0, quantize_row_q8_0, quantize_row_q8_k, BlockQ4_0,
    BlockQ4_1, BlockQ4_K, BlockQ6_K, BlockQ8_0, BlockQ8_K, QK4_0, QK4_1, QK8_0, QK_K,
};
pub use tensor::{TensorBuf, TensorView};
