# Caelum (`caelum.rs`)

[![CI](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Caelum** (*Latin: "The Chisel / The Sky"*) is a **zero-dependency, pure-Rust, ARMv9-first LLM inference engine** built from scratch to power **Celestia** (MLA + DeepSeekMoE), **Astraea** (Qwen2.5 / Llama / Mistral GGUF), and **Lenout Creative Workspace**.

No Python. No PyTorch. No `llama.cpp`. **Zero external crates.** Compiles to a single **~760 KB** native binary.

---

## Key Features

- **Zero External Dependencies:** Pure Rust standard library (`std`) — builds in seconds and compiles cleanly for `x86_64`, **ARMv9 (`aarch64`)**, and **`wasm32`**.
- **ARMv9-First Hardware Kernels:**
  - `aarch64` **NEON FMA + ARMv9 `dotprod` (`vdotq_s32`)** fused W4A8 (`Q4_0 × Q8_0`, `Q4_K × Q8_K`) and W6A8/W8A8 (`Q6_K × Q8_K`, `Q8_0 × Q8_0`) SIMD kernels.
  - `x86_64` **AVX2 + FMA** (`_mm256_maddubs_epi16` / `_mm256_madd_epi16` / `_mm256_fmadd_ps`) kernels.
  - Pluggable `Backend` trait designed for drop-in **Arm KleidiAI (`i8mm` / `SME2`)** or `wgpu` GPU backends.
  - Zero-dependency persistent worker thread pool avoiding per-layer thread spawn overhead.
- **Zero-Copy GGUF v2/v3 Engine:** POSIX `mmap` loader + GGUF v3 writer supporting exact GGML binary block layouts (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`).
- **Dual Model Architecture Support:**
  - **Standard GGUF Models:** Qwen2 / Qwen2.5 / Qwen2.5-Coder, Llama 2/3, Mistral (GQA, Normal & NeoX RoPE, optional QKV biases, tied/untied embeddings).
  - **Native Celestia Architecture:** True **Compressed Multi-Head Latent Attention (MLA)** caching only `c_t^{KV}` (75–87.5% KV cache memory savings) + **Sparse DeepSeekMoE** (Shared Expert + Top-K Routed Experts).
- **Multilingual BPE & Constrained JSON Decoding:**
  - Exact Qwen2.5 / GPT-2 byte-level BPE & SentencePiece BPE with incremental UTF-8 `StreamDecoder` (safe for multi-byte Thai & emoji sequences).
  - Zero-allocation **JSON Grammar Constrained Sampler** (`--json` / `propose`) guaranteeing syntactically valid JSON even on small models.
- **Built-In HTTP/1.1 Server:** OpenAI-compatible `/v1/chat/completions`, Lenout-native `/v1/celestia/propose`, and `/health`.

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

# Streaming text generation
./target/release/caelum run /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  -p "Explain why ARMv9 SME2 accelerates LLM inference."

# Constrained JSON Action Proposal (Lenout Action DSL)
./target/release/caelum propose /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --mode design -p "Create 2 pricing cards horizontally: Basic and Pro"

# Start OpenAI + Lenout HTTP API server
./target/release/caelum serve /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --port 8000
```

---

## Verified Benchmarks (CPU-Only, Intel i5-13400F, 8 Threads)

| Benchmark | Result |
|---|---|
| Release binary size (`caelum`) | **760 KB** (0 external crates) |
| `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` (`1.06 GB`) `mmap` load | **32.2 ms** |
| `2048×2048` `Q4_K` (W4A8-K) matvec kernel | **0.10 ms/op** (**80.98 GFLOP/s**, 1.72× faster than F32) |
| `2048×2048` `Q6_K` (W6A8-K) matvec kernel | **0.12 ms/op** (**67.40 GFLOP/s**, 1.43× faster than F32) |
| `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` Decode speed | **~20.1 tok/s** (CPU-only, no GPU) |
| `Qwen2.5-Coder-1.5B` Lenout JSON Proposal (`--json` 151k vocab) | **13.5 tok/s** (100% Zod-valid JSON) |

---

## Cross-Compilation (ARMv9 & WebAssembly)

```bash
# Verify ARMv9 / AArch64 (NEON + DotProd SDOT kernels)
cargo check --workspace --target aarch64-unknown-linux-gnu

# Verify core inference crates for WebAssembly (browser edge deployment)
cargo check --target wasm32-unknown-unknown \
  -p caelum-core -p caelum-kernels -p caelum-gguf \
  -p caelum-tokenizer -p caelum-models
```

---

## Roadmap

- [ ] Batched Prefill GEMM (tiled matrix-matrix multiplication for fast long-context prefill)
- [ ] Optional **Arm KleidiAI** FFI backend for ARMv9 `i8mm` & `SME2` matrix tiles
- [ ] Quantized `Q8_0` KV Cache & Prefix Caching for instant repeated Lenout prompts
- [ ] Full JSON Schema Grammar (constraining field names, enums, and types at token level)
- [ ] `wgpu` / Vulkan GPU compute backend

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
