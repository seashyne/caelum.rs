# Caelum (`caelum.rs`)

[![CI](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/seashyne/caelum.rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Caelum** (*Latin: "The Chisel / The Sky"*) is a **zero-dependency-by-default, pure-Rust, Multi-User Continuous Batching & Ultra-Low RAM LLM inference engine** targeting **ARMv9**, **x86 AVX-VNNI**, and optional **`wgpu` Vulkan/Metal/DX12 GPUs** — powering **Caelum Studio**, **mybot-cli**, **Celestia**, **Astraea**, **Cline / OpenCode**, and **Lenout Creative Workspace**.

No Python. No PyTorch. No `llama.cpp`. Default CPU build has **zero external crates** in a single **~1.4 MB** native binary (beating `llama.cpp b12651` on CPU by **+44.6%** decode / **+35–63%** prefill) — plus an optional **`--features gpu`** `wgpu` Vulkan/Metal/DX12 backend hitting **`128.5 tok/s`** (`1.5B`), **`65.0 tok/s` aggregate across 4 concurrent users** (`7B` Split-GGUF), **`36.1 tok/s`** (`Qwen3.5-9B`), and **`18.6 tok/s`** (`Gemma-4-12B-it Q6_K`) on an **Intel Arc B580** while using only **`117.7 MB` total host RAM (`VmRSS`)**!

---

## Key Features (`v0.13.0`)

- **Native Hardware Cooperative-Matrix XMX (`v0.13.0`, `--features gpu`):**
  - **Pure-Rust SPIR-V 1.5 XMX Generator (`spv_xmx.rs`):** Emits `SPV_KHR_cooperative_matrix` (`M=8, N=16, K=16`, `f16 × f16 + f32`) hardware matrix-engine compute shaders directly through `wgpu 30`'s `PASSTHROUGH_SHADERS` + `EXPERIMENTAL_COOPERATIVE_MATRIX` with **zero extra Cargo dependencies** and automatic portable WGSL fallback on non-XMX GPUs.
  - **Multi-Subgroup SLM-Tiled Quantized XMX GEMM:** Accelerates batched prefill & multi-user continuous batching for **`Q4_K`** (`0.189 ms` / **`2,838 GFLOP/s`**, **2.02× faster** than WGSL `B4`), **`Q5_K`** (`0.214 ms` / **`2,513 GFLOP/s`**, **~4.6× faster**), **`Q6_K`** (`0.193–0.208 ms` / **`2,585–2,784 GFLOP/s`**), **`F16`**, and **`F32`** across `Llama/Qwen`, `Qwen3.5`, and `Gemma 4`.
  - **Xe2 GDDR6 `B=1` Decode Optimizations:** 4-byte-aligned `Q8_0` (`36B = 9 u32`, **4.0× faster** → `225 GB/s`) & `Q4_0` (`20B = 5 u32`, **2.8× faster** → `159 GB/s` / `145 tok/s` on 1.3B `Q4_0`), 2×-unrolled SSA `Q4_K` (`0.044 ms` / `213 GB/s`), 4×-unrolled `F32` (`327 GB/s`), and barrier-free `sg_size == 32` subgroup row reductions.
- **Multi-User Continuous Batching (`v0.11.0+`, `--slots <N>`):**
  - Iteration-level continuous batching scheduler (`scheduler.rs`) + non-blocking multi-threaded HTTP/1.1 server (`GET /health` responds in **1.45 ms** even under 4/4 active GPU load!).
  - **Shared Weights Across All Concurrent Users:** Hardware-batched `decode_slots()` on **both Vulkan GPU (`SHADER_ROPE_KV_STORE_SLOTS_F16` + `SHADER_GQA_FLASH_ATTN_SLOTS_F16` + `B4` GEMM)** and **CPU AVX-VNNI/AVX2/NEON**, reading model weights ONCE per decode step across all active slots.
  - **Per-Slot Automatic KV Prefix Caching** + immediate slot reclamation on EOS, `max_tokens`, `stop`, grammar completion, or client disconnect.
- **Ultra-Low RAM Engine (`v0.12.0`):**
  - **`madvise(MADV_DONTNEED)` after GPU VRAM upload:** Evicts uploaded `mmap` weight pages from Linux page cache immediately — dropping 4-slot 7B GPU `VmRSS` from **`3,680 MB` → `117.7 MB` (31.3× reduction!)**.
  - **Compact Tokenizer + Metadata Release:** Integer-keyed BPE merge table `(u32, u32) -> u32` + single flat byte arena + immediate removal of duplicate `tokenizer.ggml.*` arrays from `gguf.metadata` — cutting private heap `RssAnon` by **2×** (`115.8 MB` → **`58.6 MB`**).
  - **Lazy On-Demand Per-Slot KV Cache + `Q8_0` (default) & `Q4_0` (`--kv-q4`) KV Quantization:** Idle slots allocate **0 bytes**; `--kv-q8` cuts KV RAM by **3.76×** and `--kv-q4` cuts KV RAM by **7.11×** vs `F32` (7B CPU 4-slot idle = **`62.0 MB` `VmRSS`**, active private heap = **`72.1 MB`**).
  - **RAM-Aware Auto-Repack Guard:** Automatically skips CPU weight repacking when `model_bytes > 55%` of `MemAvailable`, running directly from zero-copy `mmap` without swap thrashing.
- **Zero-Copy Multi-Shard Split-GGUF + 13+ Modern Architectures (`v0.9.0–v0.10.0`):**
  - **Auto-Stitched Split-GGUF:** Pass shard 1 (`*-00001-of-0000N.gguf`) and Caelum maps all shards zero-copy with zero file concatenation.
  - **Dense, Hybrid & MoE GGUF:** **`qwen35`** (3:1 hybrid Gated DeltaNet + Gated Full Attention, CPU & 100% VRAM GPU), **`gemma4`** (hybrid SWA + `K == V` sharing + proportional RoPE + 262k vocab 2D row dispatch, CPU & 100% VRAM GPU), `qwen2`/`qwen2.5`, `qwen3` (QK-Norm), `qwen2moe`/`qwen3moe`, `deepseek2` (Absorbed MLA + YaRN + MoE), `gemma2`/`gemma3`, `phi3`/`phi4`, `starcoder2`, `llama`, `mistral`, and native `celestia`.
- **Built-in Web Chat & Interactive Docs (`GET /`, `/chat`, `/docs`):**
  - Embedded directly inside the single binary (zero external files), displaying live Continuous Batching slot telemetry (`slots_active / slots_total`, `queue_depth`), cached prompt tokens, and a live SSE API Playground.
- **OpenAI & Coding Agent Server (Cline / OpenCode / mybot-cli / Continue.dev Ready):**
  - `GET /health` (live batching metrics), `GET /v1/models`, `POST /v1/chat/completions` (multi-turn, OpenAI Tool Calling + SSE `delta.tool_calls`, `json_object`, `stop`, prefix cache), `POST /v1/completions` (IDE FIM autocomplete), `POST /v1/celestia/propose` (Lenout strict `SchemaGrammar`).

---

## Benchmarks (Intel Core i5-13400F + Intel Arc B580 12 GB Vulkan 1.4)

### Throughput

| Model / Scenario | Decode (`tok/s`) | Prefill (`tok/s`) | Highlights |
| :--- | :---: | :---: | :--- |
| **`Qwen2.5-Coder-1.5B` (`Q4_K_M`, GPU)** | **`128.51 tok/s`** | **`535–567 tok/s`** | **6.16× faster** than `llama.cpp` CPU (`20.85 tok/s`) |
| **`Qwen2.5-7B-Instruct` Split-GGUF (1 User, GPU)** | **`46.88 tok/s`** | **`113.91 tok/s`** | Zero-copy 2-shard Split-GGUF |
| **`Qwen2.5-7B-Instruct` (4 Users Continuous Batch, GPU)** | **`65.02 tok/s` agg.** | `~81.4 tok/s` pure decode | 4/4 active slots · `/health` avg **`1.45 ms`** |
| **`Qwen3.5-9B` (`Q4_K_M`, Hybrid DeltaNet, GPU)** | **`36.12 tok/s`** | **`86.75 tok/s`** | **7.3× faster** than CPU |
| **`Gemma-4-12B-it` (`Q6_K`, 262k vocab, GPU)** | **`18.59 tok/s`** | **`65.85 tok/s`** | **5.85× faster** than CPU · 100% VRAM |

### Measured Memory Footprint (`/proc/<pid>/status`, `Qwen2.5-7B` 4.36 GB, 4 Slots, `ctx=2048`)

| Mode | `v0.11.0` (`VmRSS` / Private Heap) | **`v0.12.0` (`VmRSS` / Private Heap)** | Savings |
| :--- | :---: | :---: | :---: |
| **GPU 4-slot (`--gpu --slots 4`)** | `3,679.9 MB` / `115.8 MB` | **`117.7 MB`** / **`58.6 MB`** | **31.3× `VmRSS` · 2.0× Heap** |
| **CPU 4-slot Idle (`--no-repack`)** | `339.0 MB` / `319.4 MB` | **`62.0 MB`** / **`40.6 MB`** | **5.5× `VmRSS` · 7.9× Heap** |
| **CPU 4-slot Active (`Q8_0` default)** | `4,498.4 MB` / `322.8 MB` | `4,281.5 MB` / **`100.1 MB`** | **3.2× Private Heap** |
| **CPU 4-slot Active (`--kv-q4`)** | — | `4,252.7 MB` / **`72.1 MB`** | **4.5× Heap (7.11× vs `F32`)** |

---

## Architecture

```text
caelum/
├── crates/
│   ├── caelum-core/        # F16/BF16, GGML quant layouts (Q4_0..Q6_K, Q8_K), zero-copy TensorView slicing
│   ├── caelum-kernels/     # Backend trait, P-core ThreadPool, TileQ4Kx8/Q5Kx8/Q6Kx8 AVX-VNNI, ARMv9, wgpu WGSL
│   ├── caelum-gguf/        # Zero-copy multi-shard mmap GGUF v2/v3 parser (MADV_DONTNEED) & GGUF writer
│   ├── caelum-tokenizer/   # Compact integer-keyed BPE & SentencePiece, Flat Byte Arena, ChatTemplate, SchemaGrammar
│   ├── caelum-models/      # Multi-slot Llama/Qwen2/3/MoE, Qwen3.5, Gemma2/3/4, Phi3/4, StarCoder2, DeepSeek2, Celestia + wgpu GPU runners
│   └── caelum-cli/         # `caelum` CLI + Iteration-Level Continuous Batching Scheduler + Multi-Threaded HTTP/1.1 Server
├── web/                    # Embedded Web Chat (`/`, `/chat`) & Interactive Docs (`/docs`)
└── scripts/
```

---

## Quickstart

### Build

```bash
# Default 100% zero-dependency CPU build (AVX-VNNI / ARMv9)
cargo build --release

# Optional wgpu GPU build (Vulkan / Metal / DX12)
cargo build --release --features gpu
```

### Run & Serve

```bash
# Inspect hardware & Split-GGUF model
caelum info
caelum inspect /path/to/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

# Start Multi-User Continuous Batching Server on GPU (4 concurrent slots, ~118 MB host RAM)
caelum serve /path/to/model.gguf --gpu --slots 4 --ctx 2048 --port 8000

# Ultra-Low RAM CPU Server (4-bit Q4_0 KV cache + zero-copy mmap)
caelum serve /path/to/model.gguf --kv-q4 --no-repack --slots 4 --ctx 2048 --port 8000

# Open built-in Web Chat & Docs
#   http://127.0.0.1:8000/       (Web Chat with live slot telemetry)
#   http://127.0.0.1:8000/docs   (Docs & Live API Playground)
#   http://127.0.0.1:8000/health (Live slots_active / queue_depth JSON)
```

---

## Cross-Compilation (ARMv9 & WebAssembly)

```bash
cargo check --workspace --target aarch64-unknown-linux-gnu
cargo check --target wasm32-unknown-unknown \
  -p caelum-core -p caelum-kernels -p caelum-gguf \
  -p caelum-tokenizer -p caelum-models
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
