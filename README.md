# Caelum (`caelum.rs`)

[![CI](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Caelum** (*Latin: "The Chisel / The Sky"*) is a **zero-dependency, pure-Rust, ARMv9-first LLM inference engine** built from scratch to power **Celestia** (MLA + DeepSeekMoE), **Astraea** (Qwen2.5 / Llama / Mistral GGUF), and **Lenout Creative Workspace**.

No Python. No PyTorch. No `llama.cpp`. **Zero external crates.** Compiles to a single **~936 KB** native binary.

---

## Key Features

- **Zero External Dependencies:** Pure Rust standard library (`std`) — builds in seconds and compiles cleanly for `x86_64`, **ARMv9 (`aarch64`)**, and **`wasm32`**.
- **ARMv9-First Hardware Kernels:**
  - `aarch64` **NEON FMA + ARMv9 `dotprod` (`vdotq_s32`) + ARMv9 `i8mm` (`SMMLA` 2×2 tiled matrix multiply)** fused W4A8 (`Q4_0 × Q8_0`, `Q4_K × Q8_K`) and W6A8/W8A8 (`Q6_K × Q8_K`, `Q8_0 × Q8_0`) SIMD kernels.
  - `x86_64` **AVX2 + FMA** fused-scale vector-accumulator kernels (`~119 GFLOP/s` decode matvec, **`~199 GFLOP/s`** batched prefill GEMM).
  - Pluggable `Backend` trait designed for drop-in **Arm KleidiAI (`SME2`)** or `wgpu` GPU backends.
  - Zero-dependency persistent worker thread pool avoiding per-layer thread spawn overhead.
- **High-Efficiency Memory & Prefill Engine (v0.2.0):**
  - **Batched Prefill GEMM:** Reads each layer's quantized weight matrix once across multi-token chunks and skips intermediate `lm_head` projections.
  - **Quantized `Q8_0` KV Cache (`--kv-q8`):** Cuts GQA KV cache RAM & bandwidth by **3.76×** (e.g., `112 MB` → `29.8 MB` at 2048 context) with SIMD integer attention scoring.
  - **Automatic KV Prefix Caching:** Reuses the longest matching token prefix across multi-turn `caelum chat` and repeated `/v1/celestia/propose` API calls with zero recomputation.
- **Zero-Copy GGUF v2/v3 Engine:** POSIX `mmap` loader + GGUF v3 writer supporting exact GGML binary block layouts (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`).
- **Dual Model Architecture Support:**
  - **Standard GGUF Models:** Qwen2 / Qwen2.5 / Qwen2.5-Coder, Llama 2/3, Mistral (GQA, Normal & NeoX RoPE, optional QKV biases, tied/untied embeddings).
  - **Native Celestia Architecture:** True **Compressed Multi-Head Latent Attention (MLA)** caching only `c_t^{KV}` (75–87.5% KV cache memory savings) + **Sparse DeepSeekMoE** (Shared Expert + Top-K Routed Experts).
- **Multilingual BPE & Strict JSON Schema Constrained Decoding (v0.3.0):**
  - Exact Qwen2.5 / GPT-2 byte-level BPE & SentencePiece BPE with incremental UTF-8 `StreamDecoder` (safe for multi-byte Thai & emoji sequences).
  - **Full `SchemaGrammar` Engine** (`caelum-tokenizer::SchemaGrammar`): enforces exact required fields, `Enum` literals, discriminated `OneOf` action unions, 6-digit `HexColor`, `Identifier`, `UnitFloat` (`[0, 1]`), and 3D `Tuple` coordinates at token-sampling time with an `O(1)` 256-byte prefilter.
- **Built-In HTTP/1.1 Server:** OpenAI-compatible `/v1/chat/completions` (supporting real **SSE `"stream": true`**), Lenout-native `/v1/celestia/propose` (with Prefix Caching), and `/health`.

---

## Architecture

```text
caelum/
├── crates/
│   ├── caelum-core/        # IEEE-754 F16/BF16, GGML quant layouts (Q4_0..Q6_K, Q8_K), TensorView
│   ├── caelum-kernels/     # Backend trait, Persistent ThreadPool, ARMv9 NEON/DotProd & AVX2 SIMD
│   ├── caelum-gguf/        # Zero-copy mmap GGUF v2/v3 parser & GGUF v3 writer
│   ├── caelum-tokenizer/   # Qwen2.5/GPT-2 BPE, SentencePiece, StreamDecoder, Chat templates, JSON Grammar
│   ├── caelum-models/      # Unified Llama/Qwen2.5 runner + Native Celestia (MLA + DeepSeekMoE)
│   └── caelum-cli/         # `caelum` CLI + HTTP/1.1 Server + Lenout Action DSL prompter
└── scripts/
    └── verify_lenout_zod.mjs
```

---

## Quickstart

### Build

```bash
cargo build --release
```

### Inspect Hardware & Run Kernel Benchmarks

```bash
./target/release/caelum info
./target/release/caelum bench --dim 2048 --iters 200
```

### Run Celestia (MLA + DeepSeekMoE) End-to-End Demo

```bash
./target/release/caelum demo-celestia
```

### Run Any Real `.gguf` Model (e.g. Qwen2.5-Coder-1.5B-Instruct `Q4_K_M`)

```bash
# Inspect GGUF metadata & quantization breakdown via mmap
./target/release/caelum inspect /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Streaming text generation (add --kv-q8 for 3.76× smaller KV cache RAM)
./target/release/caelum run /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --kv-q8 \
  -p "Explain why ARMv9 SME2 accelerates LLM inference."

# Interactive multi-turn chat REPL (automatic KV Prefix Caching enabled)
./target/release/caelum chat /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --kv-q8

# Constrained JSON Action Proposal (Lenout Action DSL)
./target/release/caelum propose /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --kv-q8 \
  --mode design -p "Create 2 pricing cards horizontally: Basic and Pro"

# Start OpenAI (SSE streaming) + Lenout (Prefix-Cached) HTTP API server
./target/release/caelum serve /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --kv-q8 --port 8000
```

---

## Verified Benchmarks (v0.3.0, CPU-Only, Intel i5-13400F, 8 Threads)

| Benchmark | v0.1.0 | v0.3.0 |
|---|---|---|
| Release binary size (`caelum`) | 760 KB | **937 KB** (0 external crates) |
| `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` (`1.06 GB`) `mmap` load | 32.2 ms | **32.2 ms** |
| `2048×2048` `Q4_K` matvec (`B=1`, decode) | 80.98 GFLOP/s | **118.82 GFLOP/s** (`0.07 ms/op`) |
| `2048×2048` `Q4_K` batched GEMM (`B=16`, prefill) | — | **198.52 GFLOP/s** (`0.68 ms/op`) |
| `Qwen2.5-Coder-1.5B` KV Cache RAM (2048 ctx) | 112.0 MB (`F32`) | **29.8 MB** (`--kv-q8`, **3.76× smaller**) |
| `Qwen2.5-Coder-1.5B` Prefill speed (`341 tok`) | 21.76 tok/s | **53.5–60.4 tok/s** (**~2.7× faster**) |
| `Qwen2.5-Coder-1.5B` Strict `SchemaGrammar` Decode (151k vocab) | 13.47 tok/s *(JSON only)* | **15.34 tok/s** (**100% Zod-valid 5/5 modes**) |
| Repeated `/v1/celestia/propose` Prefix Cache Hit | 0 tok | **311 tok cached** (`<0.4s` TTFT) |

---

## Cross-Compilation (ARMv9 & WebAssembly)

```bash
# Verify ARMv9 / AArch64 (NEON + DotProd SDOT + I8MM SMMLA 2×2 tiled GEMM)
cargo check --workspace --target aarch64-unknown-linux-gnu

# Verify core inference crates for WebAssembly (browser edge deployment)
cargo check --target wasm32-unknown-unknown \
  -p caelum-core -p caelum-kernels -p caelum-gguf \
  -p caelum-tokenizer -p caelum-models
```

---

## Roadmap

- [x] Batched Prefill GEMM (multi-token weight reuse + ARMv9 `i8mm` `SMMLA` 2×2 tiled & `Q4_K`/`Q6_K` `dotprod` kernels)
- [x] Quantized `Q8_0` KV Cache (`--kv-q8`, 3.76× memory reduction) & Automatic KV Prefix Caching
- [x] OpenAI SSE Streaming (`"stream": true`) & Interactive Multi-Turn `caelum chat` REPL
- [x] Full JSON Schema Grammar (`SchemaGrammar`: required keys, enums, `OneOf` unions, hex colors, `[0,1]` floats, 3D tuples)
- [ ] Optional **Arm KleidiAI** FFI backend for ARMv9.2+ `SME2` matrix tiles
- [ ] `wgpu` / Vulkan GPU compute backend

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
