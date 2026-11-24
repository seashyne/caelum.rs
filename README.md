# Caelum (`caelum.rs`)

[![CI](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Caelum** (*Latin: "The Chisel / The Sky"*) is a **zero-dependency-by-default, pure-Rust, ARMv9, x86 AVX-VNNI & optional `wgpu` Vulkan/Metal/DX12 GPU LLM inference engine** built from scratch to power **Celestia** (MLA + DeepSeekMoE), **Astraea** (Qwen2.5 / Qwen3 / Gemma-2/3 / Phi-3/4 / StarCoder2 / DeepSeek-V2 / Llama / Mistral GGUF), **Cline / OpenCode** coding agents, and **Lenout Creative Workspace**.

No Python. No PyTorch. No `llama.cpp`. Default build has **zero external crates** in a single **1.3 MB** native binary (outperforming `llama.cpp b12651` on CPU by **+44.6%** decode / **+35–63%** prefill) — plus an optional **`--features gpu`** `wgpu` Vulkan/Metal/DX12 compute backend hitting **`61.15 tok/s` decode** and **`383.18 tok/s` prefill** on an **Intel Arc B580**!

---

## Key Features (v0.7.0)

- **Zero External Dependencies by Default:** Pure Rust standard library (`std`) — builds in ~6s to a **1.3 MB** binary and compiles cleanly for `x86_64`, **ARMv9 (`aarch64`)**, and **`wasm32`**.
- **Optional `wgpu` GPU Compute Backend (`--features gpu`, `--gpu`):**
  - 100% VRAM-resident Transformer runner (Vulkan / Metal / DX12) executing quantized `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q8_0`, `F16`, and `F32` WGSL compute shaders directly over raw GGUF bytes with **single-pass single-submission decode** and **online tiled Flash-Attention GQA**.
  - Verified on **Intel Arc B580 (BMG G21, Vulkan 1.4)**: **`276.2 GB/s`** (`F32`), **`221.8 GB/s` / `788.6 GFLOP/s`** (`Q4_K B=1`), **`1.02–1.22 TFLOP/s`** (`Q4_K`/`Q6_K B=16`), **`61.15 tok/s` decode** & **`383.18 tok/s` prefill** on `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` (**2.93× faster decode** than `llama.cpp` CPU), with `< 1e-4` F32 parity vs CPU.
- **Beats `llama.cpp` on CPU (`x86_64` AVX-VNNI + Online Weight Repacking):**
  - **Cache-Aligned 8-Row Repacked Tiles (`TileQ4Kx8`, `TileQ5Kx8`, `TileQ6Kx8`):** Interleaves 8 weight rows in 4-byte SIMD lanes with pre-unpacked scales/quants and hardware `F16C` (`vcvtph2ps`), executing single-instruction **`{vex} vpdpbusd` (AVX-VNNI)** with **zero horizontal reductions** (`hsum256`) — hitting **`1.465 TFLOP/s`** (`Q4_Kx8`), **`1.112 TFLOP/s`** (`Q5_Kx8`), and **`1.401 TFLOP/s`** (`Q6_Kx8`) on 6 P-cores!
  - **Single-Barrier Fused Decode Ops:** Fused QKV + Fused Gate+Up+SwiGLU + 32-wide 2-row `Q6_K` `lm_head` + P-Core pinning & lock-free dynamic work-stealing `ThreadPool`.
- **ARMv9-First Hardware Kernels (`aarch64`):**
  - `aarch64` **NEON FMA + ARMv9 `dotprod` (`vdotq_s32`) + ARMv9 `i8mm` (`SMMLA` 2×2 tiled matrix multiply)** fused W4A8/W5A8/W6A8/W8A8 (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`) SIMD kernels.
- **11 Modern Model & Coding Architectures:**
  - **Dense & MoE GGUF:** `qwen2` (`Qwen2.5-Coder`), `qwen3` (per-head **QK-Norm**), `qwen3moe` (`Qwen3-Coder-30B-A3B`), `qwen2moe`, `deepseek2` (**True Compressed Absorbed MLA** + **Batched Prefill** + **YaRN RoPE** + **DeepSeekMoE**), **`gemma2`** (`sqrt(h)` embed scale + 4 RMSNorms/layer + **GeGLU** + **Sliding Window Attention** + **Logit Soft-Capping**), **`gemma3`** (QK-Norm + **Dual Local/Global RoPE** + SWA + GeGLU), **`phi3`/`phi4`** (zero-copy sliced fused `attn_qkv` & fused `ffn_up`, CPU + GPU!), **`starcoder2`** (full **LayerNorm** + biases + non-gated **GELU MLP** + FIM), `llama`, `mistral`, and native `celestia`.
  - **Zero-Copy GGUF v2/v3 Quantizations:** `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`.
  - **GGUF Auto Chat Templates:** `chatml`, `qwen3-nothink`, `llama3`, **`gemma`**, **`phi3`**, `deepseek2`, `deepseek-v1`, `mistral`, `celestia`, `raw`.
- **OpenAI & Coding Agent Server (Cline / OpenCode / Continue.dev Ready!):**
  - `GET /v1/models` & `GET /health`
  - `POST /v1/chat/completions`: **Full OpenAI Tool Calling / Function Calling** (`tools[]`, `tool_choice`, `role: "tool"`, `assistant.tool_calls[]`, parallel calls, real-time **SSE `delta.tool_calls`** streaming without XML tag leaks!), multi-turn `messages[]`, `response_format: {"type":"json_object"}`, `stop`, and **Automatic KV Prefix Caching** (`prompt_tokens_details.cached_tokens`).
  - `POST /v1/completions`: **IDE FIM Tab Autocomplete** auto-wrapping Qwen2.5-Coder (`<|fim_prefix|>`), DeepSeek (`<｜fim▁begin｜>`), and **StarCoder2 (`<fim_prefix>`)**.
  - `POST /v1/celestia/propose`: Lenout strict `SchemaGrammar` + Prefix Caching.

---

## Benchmarks (`Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`, `1.06 GB`)

Measured on the exact same machine (**Intel Core i5-13400F** 6 P-cores + **Intel Arc B580** 12 GB GDDR6 Vulkan 1.4):

| Benchmark | `llama.cpp b12651` *(CPU)* | `Caelum CPU` *(Pure Rust, 0 crates)* | **`Caelum --gpu` *(Arc B580 Vulkan)*** | Speedup vs `llama.cpp` |
| :--- | :---: | :---: | :---: | :---: |
| **Decode 128 tok** | `20.85 ± 0.05 tok/s` | **`30.14 ± 0.10 tok/s`** *(+44.6%)* | **`61.15 tok/s`** | **2.93× faster** |
| **Long Prefill** | `183.75` / `219.46 tok/s` *(pp256)* | **`248.86 ± 2.53 tok/s`** *(pp232)* | **`383.18 tok/s`** *(pp161)* | **+75% to +108% faster** |
| **IDE FIM Autocomplete** | — | `41.02 tok/s` | **`105–123 tok/s`** | — |
| **Runtime Binary Size** | `~8.5+ MB` *(CLI + shared libs)* | **`1.3 MB`** *(0 external crates)* | **`5.4 MB`** *(`--features gpu`)* | **~6.5× smaller (CPU)** |

---

## Architecture

```text
caelum/
├── crates/
│   ├── caelum-core/        # IEEE-754 F16/BF16, GGML quant layouts (Q4_0..Q6_K, Q5_K, Q8_K), zero-copy slice_rows/slice_outer
│   ├── caelum-kernels/     # Backend trait, P-core Dynamic ThreadPool, TileQ4Kx8/Q5Kx8/Q6Kx8 AVX-VNNI, ARMv9, optional wgpu WGSL
│   ├── caelum-gguf/        # Zero-copy mmap GGUF v2/v3 parser & GGUF v3 writer
│   ├── caelum-tokenizer/   # Qwen2.5/Qwen3/GPT-2 BPE, SentencePiece, Auto ChatTemplate (incl. Gemma/Phi-3), SchemaGrammar
│   ├── caelum-models/      # Llama/Qwen2/3/MoE, Gemma-2/3, Phi-3/4, StarCoder2, DeepSeek2 (MLA), Celestia + WgpuLlamaQwenModel
│   └── caelum-cli/         # `caelum` CLI + OpenAI Tool-Calling SSE Server (`tools.rs`, `json.rs`) + Lenout prompter
└── scripts/
    ├── verify_lenout_zod.mjs
    └── test_server.sh
```

---

## Quickstart

### Build

```bash
# Default 100% zero-dependency CPU build (1.3 MB native binary, AVX-VNNI / ARMv9)
cargo build --release

# Optional wgpu GPU build (Vulkan / Metal / DX12)
cargo build --release --features gpu
```

### Run & Serve (CPU or `--gpu` for Cline / OpenCode / Continue.dev)

```bash
# Check detected CPU SIMD & Vulkan/Metal/DX12 GPU adapter
./target/release/caelum info

# Run on GPU (--gpu) or CPU (omit --gpu; add --kv-q8 for 3.76× less KV RAM)
./target/release/caelum run /path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu \
  -p "Write a lock-free ring buffer in Rust."

# Start OpenAI Tool-Calling + FIM + SSE server (point Cline / OpenCode to http://127.0.0.1:8000/v1)
./target/release/caelum serve /path/to/model.gguf --gpu --port 8000

# Benchmark CPU AVX-VNNI vs GPU Vulkan kernels (4096x4096)
./target/release/caelum bench --gpu --dim 4096 --iters 40
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

- [x] Online Weight Repacking (`TileQ4Kx8` / `TileQ5Kx8` / `TileQ6Kx8`) + AVX-VNNI (`vpdpbusd`) `8×4`/`8×8` Prefill GEMM + Fused QKV/SwiGLU + P-Core Pinning (**beats `llama.cpp`** on CPU!)
- [x] 11 Model Architectures: `Qwen2/2.5`, `Qwen3` (QK-Norm), `Qwen2/3MoE`, `DeepSeek2` (Absorbed MLA + YaRN + MoE), **`Gemma-2/3`**, **`Phi-3/4`**, **`StarCoder2`**, `Llama`, `Mistral`, `Celestia`
- [x] Optional **`wgpu` GPU Compute Backend** (`--features gpu` / `--gpu`, Vulkan / Metal / DX12, 100% VRAM-resident)
- [x] **OpenAI Tool Calling / Function Calling** (`tools`, `tool_choice`, `role: "tool"`, SSE `delta.tool_calls` for **Cline / OpenCode**) + IDE FIM (`/v1/completions`) + Automatic KV Prefix Caching
- [ ] WGSL `subgroupAdd` + Fused QKV/Bias GPU kernels & Gemma-2/3 / MoE GPU runner
- [ ] Optional **Arm KleidiAI** FFI backend for ARMv9.2+ `SME2` matrix tiles

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
