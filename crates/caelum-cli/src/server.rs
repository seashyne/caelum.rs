//! Zero-dependency HTTP/1.1 API Server for Caelum:
//! - `GET  /health`
//! - `POST /v1/chat/completions` (OpenAI-compatible, supports `"stream": true` SSE!)
//! - `POST /v1/celestia/propose` (Lenout Action DSL with JSON Grammar + Automatic Prefix Caching)

use crate::lenout::build_lenout_system_prompt;
use caelum_core::Result;
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, CpuBackend};
use caelum_models::{CaelumModel, KvPrecision, SamplerConfig};
use caelum_tokenizer::{ChatMessage, ChatRole, ChatTemplate, SchemaGrammar, Tokenizer};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

#[allow(clippy::too_many_arguments)]
pub fn run_server(
    model_path: &str,
    host: &str,
    port: u16,
    threads: Option<usize>,
    max_ctx: Option<usize>,
    kv_precision: KvPrecision,
    template_override: Option<&str>,
    enable_repack: bool,
) -> Result<()> {
    let backend = CpuBackend::new(threads);
    println!("[Caelum Server] Loading GGUF via zero-copy mmap: {model_path}");
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
    let template = match template_override {
        None | Some("auto") => ChatTemplate::from_gguf(&gguf),
        Some(name) => ChatTemplate::from_name(name).ok_or_else(|| {
            caelum_core::CaelumError::Runtime(format!("unknown --template '{name}'"))
        })?,
    };

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("============================================================");
    println!("  CAELUM v{} RUST INFERENCE SERVER READY", env!("CARGO_PKG_VERSION"));
    println!("  Listen:   http://{addr}");
    println!(
        "  Model:    {arch} ({:.1}M params) | KV: {} ({:.1} MB)",
        gguf.total_parameters() as f64 / 1e6,
        model.kv_mode_label(),
        model.kv_cache_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("  Hardware: {} ({} threads)", backend.caps(), backend.num_threads());
    println!("  Features: Batched Prefill GEMM + Automatic Prefix Cache + SSE Stream");
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    POST /v1/chat/completions  (stream: true/false)");
    println!("    POST /v1/celestia/propose  (Lenout JSON Grammar + Prefix Cache)");
    println!("============================================================");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(
                    &mut stream,
                    &backend,
                    &mut model,
                    &tokenizer,
                    template,
                    &arch,
                ) {
                    eprintln!("[Caelum Server] Request error: {e}");
                }
            }
            Err(e) => eprintln!("[Caelum Server] Accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    backend: &CpuBackend,
    model: &mut CaelumModel<'_>,
    tokenizer: &Tokenizer,
    template: ChatTemplate,
    arch: &str,
) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body);

    if method == "OPTIONS" {
        return write_response(stream, "204 No Content", "");
    }

    match (method, path) {
        ("GET", "/health") => {
            let now = unix_ts();
            let resp = format!(
                r#"{{"status":"healthy","engine":"caelum-rs","version":"{}","architecture":"{}","kernel":"{}","kv_mode":"{}","threads":{},"timestamp":{}}}"#,
                env!("CARGO_PKG_VERSION"),
                escape_json(arch),
                escape_json(backend.name()),
                model.kv_mode_label(),
                backend.num_threads(),
                now
            );
            write_response(stream, "200 OK", &resp)
        }
        ("POST", "/v1/chat/completions") => {
            let user_prompt = extract_last_json_string(&body_str, "content")
                .unwrap_or_else(|| "Hello".to_string());
            let max_tokens = extract_json_usize(&body_str, "max_tokens").unwrap_or(256);
            let temp = extract_json_f32(&body_str, "temperature").unwrap_or(0.7);
            let is_stream = extract_json_bool(&body_str, "stream").unwrap_or(false);

            let messages = vec![ChatMessage {
                role: ChatRole::User,
                content: user_prompt,
            }];
            let formatted = template.format(&messages, true);
            let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);

            if is_stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                stream.write_all(header.as_bytes())?;
                stream.flush()?;

                let cmpl_id = format!("chatcmpl-caelum-{}", unix_ts());
                let created = unix_ts();
                let model_name = format!("caelum-{arch}");
                let mut write_err = false;

                let _ = model.generate(
                    backend,
                    tokenizer,
                    &prompt_tokens,
                    max_tokens,
                    &SamplerConfig {
                        temperature: temp,
                        ..Default::default()
                    },
                    false,
                    |_, piece| {
                        if write_err || piece.is_empty() {
                            return;
                        }
                        let chunk = format!(
                            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                            cmpl_id,
                            created,
                            escape_json(&model_name),
                            escape_json(piece)
                        );
                        if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                            write_err = true;
                        }
                    },
                )?;

                if !write_err {
                    let final_chunk = format!(
                        "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
                        cmpl_id,
                        created,
                        escape_json(&model_name)
                    );
                    let _ = stream.write_all(final_chunk.as_bytes());
                    let _ = stream.flush();
                }
                return Ok(());
            }

            let (output, stats) = model.generate(
                backend,
                tokenizer,
                &prompt_tokens,
                max_tokens,
                &SamplerConfig {
                    temperature: temp,
                    ..Default::default()
                },
                false,
                |_, _| {},
            )?;

            let resp = format!(
                r#"{{"id":"chatcmpl-caelum-{}","object":"chat.completion","created":{},"model":"caelum-{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"cached_prompt_tokens":{},"completion_tokens":{},"prefill_tps":{:.2},"decode_tps":{:.2}}}}}"#,
                unix_ts(),
                unix_ts(),
                escape_json(arch),
                escape_json(&output),
                stats.prompt_tokens,
                stats.cached_prompt_tokens,
                stats.generated_tokens,
                stats.prefill_tps(),
                stats.decode_tps()
            );
            write_response(stream, "200 OK", &resp)
        }
        ("POST", "/v1/celestia/propose") => {
            let prompt = extract_last_json_string(&body_str, "prompt")
                .unwrap_or_else(|| "Create a design card".to_string());
            let mode = extract_last_json_string(&body_str, "workspace_mode")
                .unwrap_or_else(|| "design".to_string());
            let request_id = extract_last_json_string(&body_str, "request_id")
                .unwrap_or_else(|| "req_caelum".to_string());
            let fp = extract_last_json_string(&body_str, "context_fingerprint")
                .unwrap_or_else(|| "fp_caelum".to_string());

            // Keep static system prompt stable so Automatic Prefix Cache achieves >80% hit rate!
            let sys = build_lenout_system_prompt(&mode, &request_id, &fp);
            let messages = vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: sys,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: prompt,
                },
            ];
            let formatted = template.format(&messages, true);
            let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);

            let mut schema_grammar = SchemaGrammar::lenout_proposal(&mode);
            let (output, stats) = model.generate_with_grammar(
                backend,
                tokenizer,
                &prompt_tokens,
                512,
                &SamplerConfig {
                    temperature: 0.2,
                    ..Default::default()
                },
                Some(&mut schema_grammar),
                |_, _| {},
            )?;

            let trimmed = output.trim();
            let resp = if trimmed.starts_with('{') && trimmed.ends_with('}') {
                format!(
                    r#"{{"ok":true,"engine":"caelum-rs","cached_prompt_tokens":{},"prefill_tps":{:.2},"decode_tps":{:.2},"proposal":{}}}"#,
                    stats.cached_prompt_tokens,
                    stats.prefill_tps(),
                    stats.decode_tps(),
                    trimmed
                )
            } else {
                format!(
                    r#"{{"ok":false,"engine":"caelum-rs","raw_output":"{}","error":{{"code":"incomplete_json","message":"max_tokens reached before JSON object closed"}}}}"#,
                    escape_json(trimmed)
                )
            };
            write_response(stream, "200 OK", &resp)
        }
        _ => write_response(stream, "404 Not Found", r#"{"error":"route not found"}"#),
    }
}

fn write_response(stream: &mut TcpStream, status: &str, body: &str) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn extract_last_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = json.rfind(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let mut chars = after_colon[1..].chars();
    let mut out = String::new();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(esc) = chars.next() {
                match esc {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    other => out.push(other),
                }
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let num_str: String = rest[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let num_str: String = rest[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    num_str.parse().ok()
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let rest = &json[pos + needle.len()..];
    let colon = rest.find(':')?;
    let tail = rest[colon + 1..].trim_start();
    if tail.starts_with("true") {
        Some(true)
    } else if tail.starts_with("false") {
        Some(false)
    } else {
        None
    }
}
