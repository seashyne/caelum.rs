//! Caelum CLI: Pure Rust ARMv9-first LLM Inference Engine for Celestia, Astraea & Lenout.

mod lenout;
mod server;

use caelum_core::{
    f32_to_f16, quantize_row_q4_0, quantize_row_q8_0, BlockQ4_0, BlockQ8_0, DType, Result, TensorBuf,
};
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, CpuBackend, CpuCaps};
use caelum_models::{
    write_synthetic_celestia_gguf, CaelumModel, CelestiaConfig, KvPrecision, SamplerConfig,
};
use caelum_tokenizer::{ChatMessage, ChatRole, ChatTemplate, SchemaGrammar, Tokenizer};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn print_usage() {
    println!(
        r#"Caelum v{} — Zero-Dependency Pure Rust ARMv9-first LLM Inference Engine

USAGE:
  caelum info
  caelum inspect <model.gguf>
  caelum run <model.gguf> -p "<prompt>" [OPTIONS]
  caelum chat <model.gguf> [--system "..."] [--kv-q8] [--ctx 4096] [--threads N]
  caelum propose <model.gguf> --mode <mode> -p "<prompt>" [--kv-q8] [--threads N] [--max-tokens 512]
  caelum serve <model.gguf> [--host 127.0.0.1] [--port 8000] [--kv-q8] [--threads N] [--ctx 2048]
  caelum bench [--dim 2048] [--iters 200] [--threads N]
  caelum demo-celestia [--save <path.gguf>] [--threads N]

OPTIONS:
  -p, --prompt <TEXT>     User prompt text
  --system <TEXT>         Optional system prompt
  --max-tokens <N>        Maximum tokens to generate (default: 256)
  --temp <FLOAT>          Sampling temperature, 0 = greedy (default: 0.7)
  --top-k <N>             Top-K sampling cutoff (default: 40)
  --top-p <FLOAT>         Top-P nucleus sampling cutoff (default: 0.9)
  --ctx <N>               Max KV context length (default: model min 4096)
  --kv-q8                 Use 8-bit Q8_0 quantized KV cache (3.76× less RAM)
  --threads <N>           Worker thread count (default: auto P-core pinning)
  --template <NAME>       Chat template: auto, chatml, qwen3-nothink, llama3,
                          deepseek2, deepseek-v1, mistral, celestia, raw
  --no-repack             Disable online 8-row SIMD weight repacking
  --json                  Enable strict JSON Grammar Constrained Decoding
  --raw                   Disable chat template wrapping (alias for --template raw)
"#,
        env!("CARGO_PKG_VERSION")
    );
}

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("[Caelum Error] {e}");
        std::process::exit(1);
    }
}

fn run_cli() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "info" => cmd_info(),
        "inspect" => {
            let path = args.get(2).ok_or_else(|| {
                caelum_core::CaelumError::Runtime("usage: caelum inspect <model.gguf>".into())
            })?;
            cmd_inspect(path)
        }
        "run" => cmd_run(&args[2..]),
        "chat" => cmd_chat(&args[2..]),
        "propose" => cmd_propose(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "demo-celestia" => cmd_demo_celestia(&args[2..]),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("Unknown command: {other}\n");
            print_usage();
            Ok(())
        }
    }
}

fn cmd_info() -> Result<()> {
    let caps = CpuCaps::detect();
    let backend = CpuBackend::new(None);
    println!("============================================================");
    println!("  CAELUM ENGINE — SYSTEM & HARDWARE PROFILE");
    println!("============================================================");
    println!("  Version:       v{}", env!("CARGO_PKG_VERSION"));
    println!("  Hardware:      {caps}");
    println!("  Default Pool:  {} threads (P-core pinned)", backend.num_threads());
    println!("  Quantization:  F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K");
    println!("  Architectures: llama, qwen2/2.5, qwen2moe, qwen3, qwen3moe, deepseek2 (MLA+MoE), mistral, celestia");
    println!("  Accelerators:  x86 AVX-VNNI/AVX2+FMA+F16C (8x4/8x8 Repacked) | ARMv9 NEON+DotProd+I8MM SMMLA");
    println!("============================================================");
    Ok(())
}

fn resolve_template(gguf: &GgufFile, override_name: Option<&str>, raw_mode: bool) -> Result<ChatTemplate> {
    if raw_mode {
        return Ok(ChatTemplate::Raw);
    }
    match override_name {
        None | Some("auto") => Ok(ChatTemplate::from_gguf(gguf)),
        Some(name) => ChatTemplate::from_name(name).ok_or_else(|| {
            caelum_core::CaelumError::Runtime(format!(
                "unknown --template '{name}' (expected: auto, chatml, qwen3-nothink, llama3, deepseek2, deepseek-v1, mistral, celestia, raw)"
            ))
        }),
    }
}

fn cmd_inspect(path: &str) -> Result<()> {
    let t0 = Instant::now();
    let gguf = GgufFile::open(path)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut dtype_counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in gguf.tensors.values() {
        let entry = dtype_counts.entry(t.dtype.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.byte_size;
    }

    println!("============================================================");
    println!("  CAELUM GGUF INSPECTOR ({elapsed_ms:.2} ms mmap load)");
    println!("============================================================");
    println!("  File:          {path}");
    println!("  Size:          {:.2} MB", gguf.file_size() as f64 / (1024.0 * 1024.0));
    println!("  GGUF Version:  v{}", gguf.version);
    println!("  Architecture:  {}", gguf.architecture().unwrap_or("unknown"));
    println!("  Chat Template: {}", ChatTemplate::from_gguf(&gguf).name());
    println!("  Parameters:    {:.2}M", gguf.total_parameters() as f64 / 1e6);
    println!("  Tensors:       {}", gguf.tensors.len());
    println!("  Metadata KVs:  {}", gguf.metadata.len());
    println!("\n  Quantization Breakdown:");
    for (dt, (count, bytes)) in &dtype_counts {
        println!(
            "    {:6} : {:4} tensors ({:.2} MB)",
            dt,
            count,
            *bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!("\n  Key Metadata:");
    for (k, v) in &gguf.metadata {
        if !k.starts_with("tokenizer.ggml.tokens")
            && !k.starts_with("tokenizer.ggml.merges")
            && !k.starts_with("tokenizer.ggml.scores")
            && !k.starts_with("tokenizer.ggml.token_type")
        {
            println!("    {k} = {v}");
        }
    }
    println!("============================================================");
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<()> {
    let Some(model_path) = args.first() else {
        return Err(caelum_core::CaelumError::Runtime(
            "usage: caelum run <model.gguf> -p \"<prompt>\"".into(),
        ));
    };

    let mut prompt = String::from("Hello! Introduce yourself briefly.");
    let mut system: Option<String> = None;
    let mut max_tokens = 256usize;
    let mut temp = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut max_ctx: Option<usize> = None;
    let mut threads: Option<usize> = None;
    let mut kv_precision = KvPrecision::F32;
    let mut constrain_json = false;
    let mut raw_mode = false;
    let mut template_override: Option<String> = None;
    let mut enable_repack = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    prompt = v.clone();
                }
            }
            "--system" => {
                i += 1;
                system = args.get(i).cloned();
            }
            "--max-tokens" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    max_tokens = v.parse().unwrap_or(256);
                }
            }
            "--temp" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    temp = v.parse().unwrap_or(0.7);
                }
            }
            "--top-k" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    top_k = v.parse().unwrap_or(40);
                }
            }
            "--top-p" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    top_p = v.parse().unwrap_or(0.9);
                }
            }
            "--ctx" => {
                i += 1;
                max_ctx = args.get(i).and_then(|v| v.parse().ok());
            }
            "--kv-q8" => kv_precision = KvPrecision::Q8_0,
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            "--template" => {
                i += 1;
                template_override = args.get(i).cloned();
            }
            "--no-repack" => enable_repack = false,
            "--json" => constrain_json = true,
            "--raw" => raw_mode = true,
            _ => {}
        }
        i += 1;
    }

    let backend = CpuBackend::new(threads);
    let t_load = Instant::now();
    let gguf = GgufFile::open(model_path)?;
    let arch = gguf.architecture().unwrap_or("llama").to_string();
    let tokenizer = if arch == "celestia" && gguf.get_meta("tokenizer.ggml.tokens").is_none() {
        let (v, _) = gguf.tensor("token_embd.weight")?.dims_2d()?;
        Tokenizer::celestia_default(v)
    } else {
        Tokenizer::from_gguf(&gguf)?
    };
    let mut model = CaelumModel::from_gguf_with_kv(&gguf, max_ctx, kv_precision)?;
    if enable_repack {
        model.optimize_for_backend(&backend);
    }
    let template = resolve_template(&gguf, template_override.as_deref(), raw_mode)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[Caelum] Loaded {} ({:.1}M params) in {:.1}ms | {} | {} threads | tmpl={} | packed={:.0}MB | KV {} ({:.1} MB)",
        arch,
        gguf.total_parameters() as f64 / 1e6,
        load_ms,
        backend.name(),
        backend.num_threads(),
        template.name(),
        model.packed_bytes() as f64 / (1024.0 * 1024.0),
        model.kv_mode_label(),
        model.kv_cache_bytes() as f64 / (1024.0 * 1024.0),
    );

    let full_prompt = if matches!(template, ChatTemplate::Raw) {
        prompt
    } else {
        let mut msgs = Vec::new();
        if let Some(sys) = system {
            msgs.push(ChatMessage {
                role: ChatRole::System,
                content: sys,
            });
        }
        msgs.push(ChatMessage {
            role: ChatRole::User,
            content: prompt,
        });
        template.format(&msgs, true)
    };

    let prompt_tokens = tokenizer.encode(&full_prompt, tokenizer.add_bos);
    let sampler_cfg = SamplerConfig {
        temperature: temp,
        top_k,
        top_p,
        ..Default::default()
    };

    let (_, stats) = model.generate(
        &backend,
        &tokenizer,
        &prompt_tokens,
        max_tokens,
        &sampler_cfg,
        constrain_json,
        |_, piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
        },
    )?;

    println!();
    eprintln!(
        "\n[Caelum Stats] Prefill: {} tok (cached={}, {:.2} tok/s) | Decode: {} tok ({:.2} tok/s)",
        stats.prompt_tokens,
        stats.cached_prompt_tokens,
        stats.prefill_tps(),
        stats.generated_tokens,
        stats.decode_tps()
    );
    Ok(())
}

fn cmd_chat(args: &[String]) -> Result<()> {
    let Some(model_path) = args.first() else {
        return Err(caelum_core::CaelumError::Runtime(
            "usage: caelum chat <model.gguf> [--system \"...\"] [--kv-q8]".into(),
        ));
    };

    let mut system: Option<String> = None;
    let mut max_tokens = 512usize;
    let mut temp = 0.6f32;
    let mut max_ctx: Option<usize> = Some(4096);
    let mut threads: Option<usize> = None;
    let mut kv_precision = KvPrecision::Q8_0; // Default Q8_0 for interactive long-context chat
    let mut template_override: Option<String> = None;
    let mut enable_repack = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--system" => {
                i += 1;
                system = args.get(i).cloned();
            }
            "--max-tokens" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    max_tokens = v.parse().unwrap_or(512);
                }
            }
            "--temp" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    temp = v.parse().unwrap_or(0.6);
                }
            }
            "--ctx" => {
                i += 1;
                max_ctx = args.get(i).and_then(|v| v.parse().ok());
            }
            "--kv-f32" => kv_precision = KvPrecision::F32,
            "--kv-q8" => kv_precision = KvPrecision::Q8_0,
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            "--template" => {
                i += 1;
                template_override = args.get(i).cloned();
            }
            "--no-repack" => enable_repack = false,
            _ => {}
        }
        i += 1;
    }

    let backend = CpuBackend::new(threads);
    let gguf = GgufFile::open(model_path)?;
    let arch = gguf.architecture().unwrap_or("llama").to_string();
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let mut model = CaelumModel::from_gguf_with_kv(&gguf, max_ctx, kv_precision)?;
    if enable_repack {
        model.optimize_for_backend(&backend);
    }
    let template = resolve_template(&gguf, template_override.as_deref(), false)?;

    println!("============================================================");
    println!("  CAELUM INTERACTIVE CHAT (Prefix Caching ON)");
    println!(
        "  Model: {arch} ({:.1}M) | {} | KV: {} ({:.1} MB)",
        gguf.total_parameters() as f64 / 1e6,
        backend.name(),
        model.kv_mode_label(),
        model.kv_cache_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("  Commands: /reset (clear history), /quit (exit)");
    println!("============================================================");

    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: sys,
        });
    }

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("\nYou> ");
        let _ = io::stdout().flush();
        let Some(Ok(line)) = lines.next() else {
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/quit" || trimmed == "/exit" {
            break;
        }
        if trimmed == "/reset" {
            messages.retain(|m| m.role == ChatRole::System);
            model.reset_cache();
            println!("[Caelum] Context cleared.");
            continue;
        }

        messages.push(ChatMessage {
            role: ChatRole::User,
            content: trimmed.to_string(),
        });
        let formatted = template.format(&messages, true);
        let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);

        print!("AI> ");
        let _ = io::stdout().flush();
        let (reply, stats) = model.generate(
            &backend,
            &tokenizer,
            &prompt_tokens,
            max_tokens,
            &SamplerConfig {
                temperature: temp,
                ..Default::default()
            },
            false,
            |_, piece| {
                print!("{piece}");
                let _ = io::stdout().flush();
            },
        )?;
        println!(
            "\n  [prefill: {}/{} cached ({:.1} tok/s) | decode: {} tok ({:.1} tok/s)]",
            stats.cached_prompt_tokens,
            stats.prompt_tokens,
            stats.prefill_tps(),
            stats.generated_tokens,
            stats.decode_tps()
        );
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: reply,
        });
    }
    Ok(())
}

fn cmd_propose(args: &[String]) -> Result<()> {
    let Some(model_path) = args.first() else {
        return Err(caelum_core::CaelumError::Runtime(
            "usage: caelum propose <model.gguf> --mode design -p \"<prompt>\"".into(),
        ));
    };

    let mut prompt = String::from("สร้างการ์ดสินค้า 2 ใบเรียงแนวนอน");
    let mut mode = String::from("design");
    let mut max_tokens = 512usize;
    let mut temp = 0.2f32;
    let mut threads: Option<usize> = None;
    let mut kv_precision = KvPrecision::F32;
    let mut template_override: Option<String> = None;
    let mut enable_repack = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    prompt = v.clone();
                }
            }
            "--mode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    mode = v.clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    max_tokens = v.parse().unwrap_or(512);
                }
            }
            "--temp" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    temp = v.parse().unwrap_or(0.2);
                }
            }
            "--kv-q8" => kv_precision = KvPrecision::Q8_0,
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            "--template" => {
                i += 1;
                template_override = args.get(i).cloned();
            }
            "--no-repack" => enable_repack = false,
            _ => {}
        }
        i += 1;
    }

    let backend = CpuBackend::new(threads);
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let mut model = CaelumModel::from_gguf_with_kv(&gguf, Some(2048), kv_precision)?;
    if enable_repack {
        model.optimize_for_backend(&backend);
    }
    let template = resolve_template(&gguf, template_override.as_deref(), false)?;

    let sys = lenout::build_lenout_system_prompt(&mode, "req_caelum_cli", "fp_caelum_cli");
    let formatted = template.format(
        &[
            ChatMessage {
                role: ChatRole::System,
                content: sys,
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ],
        true,
    );
    let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);

    eprintln!(
        "[Caelum Propose] mode={mode} | {} | {} threads | KV {} ({:.1} MB) | Strict SchemaGrammar=ON",
        backend.name(),
        backend.num_threads(),
        model.kv_mode_label(),
        model.kv_cache_bytes() as f64 / (1024.0 * 1024.0)
    );

    let mut schema_grammar = SchemaGrammar::lenout_proposal(&mode);
    let (json_out, stats) = model.generate_with_grammar(
        &backend,
        &tokenizer,
        &prompt_tokens,
        max_tokens,
        &SamplerConfig {
            temperature: temp,
            ..Default::default()
        },
        Some(&mut schema_grammar),
        |_, _| {},
    )?;

    println!("{json_out}");
    eprintln!(
        "[Caelum Stats] Prefill: {} tok (cached={}, {:.2} tok/s) | Decode: {} tok ({:.2} tok/s)",
        stats.prompt_tokens,
        stats.cached_prompt_tokens,
        stats.prefill_tps(),
        stats.generated_tokens,
        stats.decode_tps()
    );
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<()> {
    let Some(model_path) = args.first() else {
        return Err(caelum_core::CaelumError::Runtime(
            "usage: caelum serve <model.gguf> [--host 127.0.0.1] [--port 8000] [--kv-q8]".into(),
        ));
    };
    let mut host = String::from("127.0.0.1");
    let mut port = 8000u16;
    let mut threads: Option<usize> = None;
    let mut max_ctx: Option<usize> = None;
    let mut kv_precision = KvPrecision::F32;
    let mut template_override: Option<String> = None;
    let mut enable_repack = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    host = v.clone();
                }
            }
            "--port" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    port = v.parse().unwrap_or(8000);
                }
            }
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            "--ctx" => {
                i += 1;
                max_ctx = args.get(i).and_then(|v| v.parse().ok());
            }
            "--kv-q8" => kv_precision = KvPrecision::Q8_0,
            "--template" => {
                i += 1;
                template_override = args.get(i).cloned();
            }
            "--no-repack" => enable_repack = false,
            _ => {}
        }
        i += 1;
    }

    server::run_server(
        model_path,
        &host,
        port,
        threads,
        max_ctx,
        kv_precision,
        template_override.as_deref(),
        enable_repack,
    )
}

fn cmd_bench(args: &[String]) -> Result<()> {
    let mut dim = 2048usize;
    let mut iters = 100usize;
    let mut threads: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dim" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    dim = v.parse().unwrap_or(2048);
                }
            }
            "--iters" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    iters = v.parse().unwrap_or(100);
                }
            }
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            _ => {}
        }
        i += 1;
    }

    let backend = CpuBackend::new(threads);
    println!("============================================================");
    println!("  CAELUM KERNEL BENCHMARK ({dim}x{dim} matvec, {iters} iters)");
    println!("  Hardware: {} | {} threads", backend.caps(), backend.num_threads());
    println!("============================================================");

    let w_f32: Vec<f32> = (0..dim * dim)
        .map(|k| ((k % 31) as f32 - 15.0) * 0.01)
        .collect();
    let x: Vec<f32> = (0..dim).map(|k| ((k % 19) as f32 - 9.0) * 0.02).collect();
    let mut out = vec![0.0f32; dim];

    // 1. F32 MatVec
    let buf_f32 = TensorBuf::from_f32(&w_f32, vec![dim, dim])?;
    let t0 = Instant::now();
    for _ in 0..iters {
        backend.matvec(&buf_f32.as_view(), &x, &mut out)?;
    }
    let dt_f32 = t0.elapsed().as_secs_f64();
    let bytes_f32 = (dim * dim * 4) as f64 * iters as f64;
    let flops = (2 * dim * dim) as f64 * iters as f64;
    println!(
        "  F32   (32-bit): {:7.2} ms/op | {:6.2} GB/s | {:6.2} GFLOP/s",
        dt_f32 * 1000.0 / iters as f64,
        bytes_f32 / dt_f32 / 1e9,
        flops / dt_f32 / 1e9
    );

    // 2. Q8_0 x Q8_0 (W8A8)
    let mut q8_blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; (dim * dim) / 32];
    for r in 0..dim {
        quantize_row_q8_0(
            &w_f32[r * dim..(r + 1) * dim],
            &mut q8_blocks[r * (dim / 32)..(r + 1) * (dim / 32)],
        );
    }
    let raw_q8 = unsafe {
        std::slice::from_raw_parts(q8_blocks.as_ptr() as *const u8, q8_blocks.len() * 34)
    }
    .to_vec();
    let buf_q8 = TensorBuf::from_raw(raw_q8, vec![dim, dim], DType::Q8_0)?;
    let t1 = Instant::now();
    for _ in 0..iters {
        backend.matvec(&buf_q8.as_view(), &x, &mut out)?;
    }
    let dt_q8 = t1.elapsed().as_secs_f64();
    let bytes_q8 = buf_q8.data.len() as f64 * iters as f64;
    println!(
        "  Q8_0  (W8A8)  : {:7.2} ms/op | {:6.2} GB/s | {:6.2} GFLOP/s",
        dt_q8 * 1000.0 / iters as f64,
        bytes_q8 / dt_q8 / 1e9,
        flops / dt_q8 / 1e9
    );

    // 3. Q4_0 x Q8_0 (W4A8 — ARMv9 Sweet Spot)
    let mut q4_blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; (dim * dim) / 32];
    for r in 0..dim {
        quantize_row_q4_0(
            &w_f32[r * dim..(r + 1) * dim],
            &mut q4_blocks[r * (dim / 32)..(r + 1) * (dim / 32)],
        );
    }
    let raw_q4 = unsafe {
        std::slice::from_raw_parts(q4_blocks.as_ptr() as *const u8, q4_blocks.len() * 18)
    }
    .to_vec();
    let buf_q4 = TensorBuf::from_raw(raw_q4, vec![dim, dim], DType::Q4_0)?;
    let t2 = Instant::now();
    for _ in 0..iters {
        backend.matvec(&buf_q4.as_view(), &x, &mut out)?;
    }
    let dt_q4 = t2.elapsed().as_secs_f64();
    let bytes_q4 = buf_q4.data.len() as f64 * iters as f64;
    println!(
        "  Q4_0  (W4A8)  : {:7.2} ms/op | {:6.2} GB/s | {:6.2} GFLOP/s  ({:.2}x faster than F32)",
        dt_q4 * 1000.0 / iters as f64,
        bytes_q4 / dt_q4 / 1e9,
        flops / dt_q4 / 1e9,
        dt_f32 / dt_q4
    );

    // 4. Q4_K x Q8_K (Q4_K_M Primary Kernel)
    if dim % 256 == 0 {
        let blocks_total = (dim * dim) / 256;
        let mut raw_q4k = vec![0x35u8; blocks_total * 144];
        let d_bytes = f32_to_f16(0.05).to_le_bytes();
        let dm_bytes = f32_to_f16(0.02).to_le_bytes();
        for blk in 0..blocks_total {
            raw_q4k[blk * 144..blk * 144 + 2].copy_from_slice(&d_bytes);
            raw_q4k[blk * 144 + 2..blk * 144 + 4].copy_from_slice(&dm_bytes);
        }
        let buf_q4k = TensorBuf::from_raw(raw_q4k, vec![dim, dim], DType::Q4_K)?;
        let t3 = Instant::now();
        for _ in 0..iters {
            backend.matvec(&buf_q4k.as_view(), &x, &mut out)?;
        }
        let dt_q4k = t3.elapsed().as_secs_f64();
        let bytes_q4k = buf_q4k.data.len() as f64 * iters as f64;
        println!(
            "  Q4_K  (W4A8-K): {:7.2} ms/op | {:6.2} GB/s | {:6.2} GFLOP/s  ({:.2}x faster than F32)",
            dt_q4k * 1000.0 / iters as f64,
            bytes_q4k / dt_q4k / 1e9,
            flops / dt_q4k / 1e9,
            dt_f32 / dt_q4k
        );

        let mut raw_q6k = vec![0x24u8; blocks_total * 210];
        for blk in 0..blocks_total {
            raw_q6k[blk * 210 + 208..blk * 210 + 210].copy_from_slice(&d_bytes);
        }
        let buf_q6k = TensorBuf::from_raw(raw_q6k, vec![dim, dim], DType::Q6_K)?;
        let t4 = Instant::now();
        for _ in 0..iters {
            backend.matvec(&buf_q6k.as_view(), &x, &mut out)?;
        }
        let dt_q6k = t4.elapsed().as_secs_f64();
        let bytes_q6k = buf_q6k.data.len() as f64 * iters as f64;
        println!(
            "  Q6_K  (W6A8-K): {:7.2} ms/op | {:6.2} GB/s | {:6.2} GFLOP/s  ({:.2}x faster than F32)",
            dt_q6k * 1000.0 / iters as f64,
            bytes_q6k / dt_q6k / 1e9,
            flops / dt_q6k / 1e9,
            dt_f32 / dt_q6k
        );

        // 5. Batched Prefill GEMM benchmark (batch=16 tokens reusing weight loads)
        let batch = 16usize;
        let x_batch: Vec<f32> = (0..batch * dim)
            .map(|k| ((k % 19) as f32 - 9.0) * 0.02)
            .collect();
        let mut out_batch = vec![0.0f32; batch * dim];
        let b_iters = (iters / 4).max(10);
        let t5 = Instant::now();
        for _ in 0..b_iters {
            backend.matmul(&buf_q4k.as_view(), &x_batch, batch, &mut out_batch)?;
        }
        let dt_gemm = t5.elapsed().as_secs_f64();
        let flops_gemm = (2 * dim * dim * batch) as f64 * b_iters as f64;
        println!(
            "  Q4_K GEMM B=16: {:7.2} ms/op | {:6.2} GFLOP/s  ({:.2}x prefill throughput vs B=1)",
            dt_gemm * 1000.0 / b_iters as f64,
            flops_gemm / dt_gemm / 1e9,
            (flops_gemm / dt_gemm) / (flops / dt_q4k)
        );
    }
    println!("============================================================");
    Ok(())
}

fn cmd_demo_celestia(args: &[String]) -> Result<()> {
    let mut save_path: Option<String> = None;
    let mut threads: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--save" => {
                i += 1;
                save_path = args.get(i).cloned();
            }
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok());
            }
            _ => {}
        }
        i += 1;
    }

    let backend = CpuBackend::new(threads);
    let cfg = CelestiaConfig::micro_test();

    println!("============================================================");
    println!("  CAELUM x CELESTIA (MLA + DeepSeekMoE) END-TO-END DEMO");
    println!("============================================================");
    println!("  Hardware:      {} ({} threads)", backend.caps(), backend.num_threads());
    println!(
        "  Celestia Spec: {}L, hidden={}, heads={}, kv_latent={} (MLA KV savings: {:.1}%)",
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.kv_latent_dim,
        cfg.mla_kv_savings_percent()
    );
    println!(
        "  Sparse MoE:    {} routed experts (Top-{}) + {} shared expert | Q4_0 W4A8",
        cfg.num_experts, cfg.num_active_experts, cfg.shared_experts
    );

    let gguf_bytes = write_synthetic_celestia_gguf(&cfg, DType::Q4_0)?;
    if let Some(ref p) = save_path {
        std::fs::write(p, &gguf_bytes)?;
        println!("  Saved Celestia GGUF v3 ({:.1} KB) -> {p}", gguf_bytes.len() as f64 / 1024.0);
    }

    let gguf = if let Some(ref p) = save_path {
        GgufFile::open(p)?
    } else {
        GgufFile::from_bytes(gguf_bytes)?
    };
    let tokenizer = Tokenizer::celestia_default(cfg.vocab_size);
    let mut model = CaelumModel::from_gguf(&gguf, Some(256))?;

    let prompt = "<system>Target: <mode_design></system><user>สร้างการ์ดสินค้า 3 ใบ</user><assistant>";
    let prompt_tokens = tokenizer.encode(prompt, false);

    println!("\n  [Test] Running JSON-Constrained Autoregressive Decode (32 tokens)...");
    let (out, stats) = model.generate(
        &backend,
        &tokenizer,
        &prompt_tokens,
        32,
        &SamplerConfig {
            temperature: 0.4,
            seed: 2026,
            ..Default::default()
        },
        true,
        |_, _| {},
    )?;

    println!("  Prompt Tokens:  {}", stats.prompt_tokens);
    println!("  Prefill Speed:  {:.1} tok/s", stats.prefill_tps());
    println!("  Decode Speed:   {:.1} tok/s ({} tokens)", stats.decode_tps(), stats.generated_tokens);
    println!("  MLA Cache RAM:  {:.1} KB (Compressed Latent Only)", stats.kv_cache_bytes as f64 / 1024.0);
    println!("  JSON-Safe Out:  {}", out.replace('\n', "\\n"));
    println!("============================================================");
    Ok(())
}
