# Caelum (`caelum.rs`)

[![CI](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Caelum** (*Latin: "The Chisel / The Sky"*) is a **zero-dependency, pure-Rust, ARMv9 & x86 AVX-VNNI LLM inference engine** built from scratch to power **Celestia** (MLA + DeepSeekMoE), **Astraea** (Qwen2.5 / Qwen3 / DeepSeek-V2 / Llama / Mistral GGUF), and **Lenout Creative Workspace**.

No Python. No PyTorch. No `llama.cpp`. **Zero external crates.** Compiles to a single **~1.2 MB** native binary — and **outperforms `llama.cpp b12651`** on both Decode (**`30.14 tok/s`** vs `20.85 tok/s`) and Prefill (**`248.86 tok/s`** vs `183.75–219.46 tok/s`) on Intel Core i5-13400F CPU-only!

---

## Key Features (v0.6.0)

- **Zero External Dependencies:** Pure Rust standard library (`std`) — builds in seconds and compiles cleanly for `x86_64`, **ARMv9 (`aarch64`)**, and **`wasm32`**.
- **Beats `llama.cpp` on CPU (`x86_64` AVX-VNNI + Online Weight Repacking):**
  - **Cache-Aligned 8-Row Repacked Tiles (`TileQ4Kx8`, `TileQ5Kx8`, `TileQ6Kx8`):** Interleaves 8 weight rows in 4-byte SIMD lanes with pre-unpacked scales/quants and hardware `F16C` (`vcvtph2ps`), executing single-instruction **`{vex} vpdpbusd` (AVX-VNNI)** with **zero horizontal reductions** (`hsum256`) and **zero register spills** — hitting **`1.465 TFLOP/s`** (`Q4_Kx8`), **`1.112 TFLOP/s`** (`Q5_Kx8`), and **`1.401 TFLOP/s`** (`Q6_Kx8`) on a 6-core CPU!
  - **Single-Barrier Fused Decode Ops:** Fused QKV projection + Fused Gate+Up+SwiGLU + zero-allocation 32-wide 2-row `Q6_K` `lm_head` cut thread barriers by ~43% and saturate **~96% of physical DRAM bandwidth**.
  - **P-Core Pinning + Lock-Free Dynamic Work-Stealing `ThreadPool`:** Pins workers to physical P-cores first on Linux hybrid CPUs (eliminating E-core stragglers) and balances heterogeneous cores via an atomic cursor, parking on `Condvar` at 0% CPU when idle.
- **ARMv9-First Hardware Kernels (`aarch64`):**
  - `aarch64` **NEON FMA + ARMv9 `dotprod` (`vdotq_s32`) + ARMv9 `i8mm` (`SMMLA` 2×2 tiled matrix multiply)** fused W4A8/W5A8/W6A8/W8A8 (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`) SIMD kernels.
  - Pluggable `Backend` trait designed for drop-in **Arm KleidiAI (`SME2`)** or `wgpu` GPU backends.
- **Broad Modern Model & Coding Architecture Support:**
  - **Dense & MoE GGUF:** `qwen2` (`Qwen2.5-Coder`), **`qwen3`** (per-head **QK-Norm**), **`qwen3moe`** (`Qwen3-Coder-30B-A3B`), **`qwen2moe`**, **`deepseek2`** (`DeepSeek-Coder-V2-Lite` / `DeepSeek-V2/V3/R1` with **True Compressed Absorbed MLA** + **Batched Prefill** + **YaRN RoPE** + **DeepSeekMoE**), `llama` (`Llama 3.x`, `DeepSeek-Coder v1`, `Yi-Coder`, `OpenCoder`, `CodeLlama`), `mistral` (`Codestral`), and native **`celestia`**.
  - **Zero-Copy GGUF v2/v3 Quantizations:** `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, **`Q5_K`**, `Q6_K`.
- **High-Efficiency Memory & Prefix Caching:**
  - **Quantized `Q8_0` KV Cache (`--kv-q8`):** Cuts GQA KV cache RAM & bandwidth by **3.76×** (`112 MB` → `29.8 MB` at 2048 ctx).
  - **True Compressed MLA Latent Cache (`deepseek2` / `celestia`):** Cuts KV cache by **75–89%** by caching only compressed latents.
  - **Automatic KV Prefix Caching:** Reuses the longest matching token prefix across multi-turn `caelum chat`, OpenAI `/v1/chat/completions`, and `/v1/celestia/propose`.
- **Full OpenAI & IDE Coding Server (Zero-Dep RFC-8259 JSON):**
  - `GET /v1/models` (auto-discovery for OpenAI SDK, **Continue.dev**, **Cline**, **OpenWebUI**)
  - `POST /v1/chat/completions` (full multi-turn `messages[]`, **SSE `"stream": true`**, `response_format: {"type":"json_object"}`, `stop` sequences, `prompt_tokens_details.cached_tokens`)
  - `POST /v1/completions` (**IDE Tab Autocomplete / Fill-In-The-Middle FIM** auto-wrapping `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`)
  - `POST /v1/celestia/propose` (Lenout strict `SchemaGrammar` + Prefix Caching) & `GET /health`

---

## Head-to-Head Benchmark vs `llama.cpp b12651`

Measured on the exact same machine (**Intel Core i5-13400F**, 6 P-cores + 4 E-cores, CPU-only, `32.8 GB/s` measured DRAM) and exact same GGUF model (**`Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`**, `1.06 GB`, `1.78B` params):

| Benchmark | `llama.cpp b12651` *(Alder Lake CPU)* | `Caelum v0.4.0` | **`Caelum v0.5.0` *(Pure Rust)*** | Delta vs `llama.cpp` |
| :--- | :---: | :---: | :---: | :---: |
| **Decode 128 tok (`t=6`, P-cores)** | `20.85 ± 0.05 tok/s` | `20.03 tok/s` | **`30.14 ± 0.10 tok/s`** | **+44.6% faster** |
| **Decode 128 tok (`t=10` / `t=12`)** | `20.38 ± 1.85 tok/s` | `19.93 tok/s` | **`29.09 tok/s`** *(t=10)* | **+42.7% faster** |
| **Short Prefill (`t=6`)** | `144.65 tok/s` *(pp32)* | `56.98 tok/s` | **`235.53 tok/s`** *(pp22)* | **+62.8% faster** |
| **Long Prefill (`t=6` / `t=12`)** | `183.75` / `219.46 tok/s` *(pp256)* | `49.09` / `67.12 tok/s` | **`248.86 ± 2.53 tok/s`** *(t=6, pp232)* | **+13% to +35% faster** |
| **Runtime Size** | `~8.5+ MB` *(CLI + shared libs)* | `942 KB` | **`1.2 MB`** *(single binary, 0 crates)* | **~7× smaller** |

> [!NOTE]
> `Caelum v0.5.0` repacks `Q4_K`/`Q6_K` layer weights once at load time (~`420 ms` total load + repack for `817 MB` on 6 P-cores; pass `--no-repack` to stay on zero-copy `mmap`).

---

## Architecture

```text
caelum/
├── crates/
│   ├── caelum-core/        # IEEE-754 F16/BF16, GGML quant layouts (Q4_0..Q6_K, Q5_K, Q8_K), 3D TensorView
│   ├── caelum-kernels/     # Backend trait, P-core Dynamic ThreadPool, TileQ4Kx8/Q6Kx8 AVX-VNNI, ARMv9 NEON/DotProd/I8MM
│   ├── caelum-gguf/        # Zero-copy mmap GGUF v2/v3 parser & GGUF v3 writer
│   ├── caelum-tokenizer/   # Qwen2.5/Qwen3/GPT-2 BPE, SentencePiece, Auto ChatTemplate, SchemaGrammar
│   ├── caelum-models/      # Llama/Qwen2/Qwen3/MoE + DeepSeek2 (Absorbed MLA+YaRN+MoE) + Native Celestia
│   └── caelum-cli/         # `caelum` CLI + HTTP/1.1 SSE Server + Lenout Action DSL prompter
└── scripts/
    ├── verify_lenout_zod.mjs
    └── test_server.sh
```

---

## Quickstart

### Build

```bash
cargo build --release
```

### Run Real Coding Models (`.gguf` — `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `Q4_0`)

```bash
# Inspect GGUF metadata, detected chat template & quantization breakdown
./target/release/caelum inspect /path/to/model.gguf

# Streaming coding generation (auto-detects P-cores & chat template; add --kv-q8 for 3.76× less KV RAM)
./target/release/caelum run /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --kv-q8 \
  -p "Write a lock-free ring buffer in Rust."

# Qwen3 fast direct coding mode (skip <think> reasoning preamble)
./target/release/caelum run /path/to/qwen3-4b-q5_k_m.gguf --template qwen3-nothink --kv-q8 \
  -p "Implement quicksort in Rust."

# Interactive multi-turn chat REPL (automatic KV Prefix Caching enabled)
./target/release/caelum chat /path/to/model.gguf --kv-q8

# Constrained JSON Action Proposal (Lenout Action DSL + strict SchemaGrammar)
./target/release/caelum propose /path/to/model.gguf --kv-q8 \
  --mode design -p "Create 2 pricing cards horizontally: Basic and Pro"

# Start OpenAI (SSE streaming) + Lenout (Prefix-Cached) HTTP API server
./target/release/caelum serve /path/to/model.gguf --kv-q8 --port 8000
```

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

- [x] Online Weight Repacking (`TileQ4Kx8` / `TileQ6Kx8`) + AVX-VNNI (`vpdpbusd`) `8×4`/`8×8` Prefill GEMM + Fused QKV/SwiGLU + P-Core Pinning (**beats `llama.cpp`** on CPU!)
- [x] `Qwen3` (QK-Norm), `Qwen3MoE`, `Qwen2MoE`, `DeepSeek2` (Absorbed MLA + YaRN + DeepSeekMoE), `Q5_K` SIMD, and GGUF auto `--template`
- [x] Quantized `Q8_0` KV Cache (`--kv-q8`, 3.76× memory reduction) & Automatic KV Prefix Caching
- [x] OpenAI SSE Streaming (`"stream": true`), Interactive Multi-Turn `caelum chat` REPL, and strict `SchemaGrammar`
- [ ] Optional **Arm KleidiAI** FFI backend for ARMv9.2+ `SME2` matrix tiles
- [ ] `wgpu` / Vulkan GPU compute backend

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
