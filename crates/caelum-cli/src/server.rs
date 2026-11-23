//! Zero-dependency OpenAI & IDE-compatible HTTP/1.1 API Server for Caelum:
//! - `GET  /health`
//! - `GET  /v1/models`             (OpenAI SDK / Continue.dev / Cline / OpenWebUI discovery)
//! - `POST /v1/chat/completions`   (Full multi-turn `messages[]`, SSE `"stream": true`,
//!                                  `response_format: json_object`, `stop`, Automatic Prefix Cache)
//! - `POST /v1/completions`        (IDE Tab Autocomplete & Qwen2.5-Coder FIM `suffix`, SSE/JSON)
//! - `POST /v1/celestia/propose`   (Lenout Action DSL with strict `SchemaGrammar` + Prefix Cache)

use crate::json::JsonValue;
use crate::lenout::build_lenout_system_prompt;
use caelum_core::Result;
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, CpuBackend};
use caelum_models::{CaelumModel, KvPrecision, SamplerConfig};
use caelum_tokenizer::{
    ChatMessage, ChatRole, ChatTemplate, JsonGrammar, LogitGrammar, SchemaGrammar, Tokenizer,
};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
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
    let model_id = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("caelum-{arch}"));
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
    println!(
        "  CAELUM v{} RUST INFERENCE SERVER READY",
        env!("CARGO_PKG_VERSION")
    );
    println!("  Listen:   http://{addr}");
    println!(
        "  Model:    {model_id} ({arch}, {:.1}M params) | tmpl={} | packed={:.0}MB | KV: {} ({:.1} MB)",
        gguf.total_parameters() as f64 / 1e6,
        template.name(),
        model.packed_bytes() as f64 / (1024.0 * 1024.0),
        model.kv_mode_label(),
        model.kv_cache_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Hardware: {} ({} threads)",
        backend.caps(),
        backend.num_threads()
    );
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    GET  /v1/models");
    println!("    POST /v1/chat/completions  (multi-turn, SSE, json_object, stop, prefix-cache)");
    println!("    POST /v1/completions       (IDE FIM tab autocomplete, SSE/JSON)");
    println!("    POST /v1/celestia/propose  (Lenout strict SchemaGrammar + prefix-cache)");
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
                    &model_id,
                ) {
                    eprintln!("[Caelum Server] Request error: {e}");
                }
            }
            Err(e) => eprintln!("[Caelum Server] Accept error: {e}"),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_connection(
    stream: &mut TcpStream,
    backend: &CpuBackend,
    model: &mut CaelumModel<'_>,
    tokenizer: &Tokenizer,
    template: ChatTemplate,
    arch: &str,
    model_id: &str,
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
    let path = parts[1].split('?').next().unwrap_or(parts[1]);

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
                r#"{{"status":"healthy","engine":"caelum-rs","version":"{}","model":"{}","architecture":"{}","template":"{}","kernel":"{}","kv_mode":"{}","threads":{},"timestamp":{}}}"#,
                env!("CARGO_PKG_VERSION"),
                escape_json(model_id),
                escape_json(arch),
                template.name(),
                escape_json(backend.name()),
                model.kv_mode_label(),
                backend.num_threads(),
                now
            );
            write_response(stream, "200 OK", &resp)
        }
        ("GET", "/v1/models") => {
            let now = unix_ts();
            let resp = format!(
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","created":{},"owned_by":"caelum-rs","architecture":"{}","context_length":{}}}]}}"#,
                escape_json(model_id),
                now,
                escape_json(arch),
                model.max_seq_len()
            );
            write_response(stream, "200 OK", &resp)
        }
        ("POST", "/v1/chat/completions") => {
            let req = match JsonValue::parse(&body_str) {
                Ok(v) => v,
                Err(e) => {
                    return write_response(
                        stream,
                        "400 Bad Request",
                        &format!(r#"{{"error":{{"message":"invalid JSON: {}"}}}}"#, escape_json(&e)),
                    );
                }
            };

            let messages = parse_chat_messages(&req);
            let max_tokens = req
                .get("max_completion_tokens")
                .or_else(|| req.get("max_tokens"))
                .and_then(JsonValue::as_usize)
                .unwrap_or(512);
            let temp = req
                .get("temperature")
                .and_then(JsonValue::as_f64)
                .map(|v| v as f32)
                .unwrap_or(0.7);
            let top_p = req
                .get("top_p")
                .and_then(JsonValue::as_f64)
                .map(|v| v as f32)
                .unwrap_or(0.9);
            let is_stream = req
                .get("stream")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let stop_seqs = parse_stop_sequences(&req);
            let json_mode = req
                .get("response_format")
                .and_then(|rf| rf.get("type"))
                .and_then(JsonValue::as_str)
                .map(|t| t == "json_object" || t == "json_schema")
                .unwrap_or(false);

            let formatted = template.format(&messages, true);
            let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);
            let sampler_cfg = SamplerConfig {
                temperature: temp,
                top_p,
                ..Default::default()
            };

            let mut json_g = if json_mode {
                Some(JsonGrammar::new())
            } else {
                None
            };
            let grammar_ref: Option<&mut dyn LogitGrammar> =
                json_g.as_mut().map(|g| g as &mut dyn LogitGrammar);

            if is_stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                stream.write_all(header.as_bytes())?;
                stream.flush()?;

                let cmpl_id = format!("chatcmpl-caelum-{}", unix_ts());
                let created = unix_ts();
                let mut write_err = false;

                // Initial role delta per OpenAI SSE specification
                let role_chunk = format!(
                    "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n",
                    cmpl_id,
                    created,
                    escape_json(model_id),
                );
                if stream.write_all(role_chunk.as_bytes()).is_err() || stream.flush().is_err() {
                    write_err = true;
                }

                let (_, stats) = model.generate_with_stop(
                    backend,
                    tokenizer,
                    &prompt_tokens,
                    max_tokens,
                    &sampler_cfg,
                    grammar_ref,
                    &stop_seqs,
                    |_, piece| {
                        if write_err || piece.is_empty() {
                            return;
                        }
                        let chunk = format!(
                            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                            cmpl_id,
                            created,
                            escape_json(model_id),
                            escape_json(piece)
                        );
                        if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                            write_err = true;
                        }
                    },
                )?;

                if !write_err {
                    let final_chunk = format!(
                        "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}}}\n\ndata: [DONE]\n\n",
                        cmpl_id,
                        created,
                        escape_json(model_id),
                        stats.prompt_tokens,
                        stats.generated_tokens,
                        stats.prompt_tokens + stats.generated_tokens,
                        stats.cached_prompt_tokens
                    );
                    let _ = stream.write_all(final_chunk.as_bytes());
                    let _ = stream.flush();
                }
                return Ok(());
            }

            let (output, stats) = model.generate_with_stop(
                backend,
                tokenizer,
                &prompt_tokens,
                max_tokens,
                &sampler_cfg,
                grammar_ref,
                &stop_seqs,
                |_, _| {},
            )?;

            let now = unix_ts();
            let resp = format!(
                r#"{{"id":"chatcmpl-caelum-{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{},"prompt_tokens_details":{{"cached_tokens":{}}},"prefill_tps":{:.2},"decode_tps":{:.2}}}}}"#,
                now,
                now,
                escape_json(model_id),
                escape_json(&output),
                stats.prompt_tokens,
                stats.generated_tokens,
                stats.prompt_tokens + stats.generated_tokens,
                stats.cached_prompt_tokens,
                stats.prefill_tps(),
                stats.decode_tps()
            );
            write_response(stream, "200 OK", &resp)
        }
        ("POST", "/v1/completions") => {
            // Raw completion & IDE Fill-In-The-Middle (FIM) autocomplete endpoint
            let req = match JsonValue::parse(&body_str) {
                Ok(v) => v,
                Err(e) => {
                    return write_response(
                        stream,
                        "400 Bad Request",
                        &format!(r#"{{"error":{{"message":"invalid JSON: {}"}}}}"#, escape_json(&e)),
                    );
                }
            };

            let prompt_raw = req
                .get("prompt")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            let suffix = req.get("suffix").and_then(JsonValue::as_str);
            let max_tokens = req
                .get("max_tokens")
                .and_then(JsonValue::as_usize)
                .unwrap_or(128);
            let temp = req
                .get("temperature")
                .and_then(JsonValue::as_f64)
                .map(|v| v as f32)
                .unwrap_or(0.2);
            let top_p = req
                .get("top_p")
                .and_then(JsonValue::as_f64)
                .map(|v| v as f32)
                .unwrap_or(0.95);
            let is_stream = req
                .get("stream")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let stop_seqs = parse_stop_sequences(&req);

            // Auto-format Qwen2.5-Coder / DeepSeek FIM when `suffix` is supplied
            let effective_prompt = if let Some(suf) = suffix {
                if tokenizer.token_id("<|fim_prefix|>").is_some() {
                    format!("<|fim_prefix|>{prompt_raw}<|fim_suffix|>{suf}<|fim_middle|>")
                } else if tokenizer.token_id("<｜fim▁begin｜>").is_some() {
                    format!("<｜fim▁begin｜>{prompt_raw}<｜fim▁hole｜>{suf}<｜fim▁end｜>")
                } else {
                    prompt_raw
                }
            } else {
                prompt_raw
            };

            let prompt_tokens = tokenizer.encode(&effective_prompt, tokenizer.add_bos);
            let sampler_cfg = SamplerConfig {
                temperature: temp,
                top_p,
                ..Default::default()
            };

            if is_stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                stream.write_all(header.as_bytes())?;
                stream.flush()?;

                let cmpl_id = format!("cmpl-caelum-{}", unix_ts());
                let created = unix_ts();
                let mut write_err = false;

                let _ = model.generate_with_stop(
                    backend,
                    tokenizer,
                    &prompt_tokens,
                    max_tokens,
                    &sampler_cfg,
                    None,
                    &stop_seqs,
                    |_, piece| {
                        if write_err || piece.is_empty() {
                            return;
                        }
                        let chunk = format!(
                            "data: {{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"text\":\"{}\",\"index\":0,\"finish_reason\":null}}]}}\n\n",
                            cmpl_id,
                            created,
                            escape_json(model_id),
                            escape_json(piece)
                        );
                        if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                            write_err = true;
                        }
                    },
                )?;

                if !write_err {
                    let final_chunk = format!(
                        "data: {{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
                        cmpl_id,
                        created,
                        escape_json(model_id)
                    );
                    let _ = stream.write_all(final_chunk.as_bytes());
                    let _ = stream.flush();
                }
                return Ok(());
            }

            let (output, stats) = model.generate_with_stop(
                backend,
                tokenizer,
                &prompt_tokens,
                max_tokens,
                &sampler_cfg,
                None,
                &stop_seqs,
                |_, _| {},
            )?;
            let now = unix_ts();
            let resp = format!(
                r#"{{"id":"cmpl-caelum-{}","object":"text_completion","created":{},"model":"{}","choices":[{{"text":"{}","index":0,"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{},"prompt_tokens_details":{{"cached_tokens":{}}},"prefill_tps":{:.2},"decode_tps":{:.2}}}}}"#,
                now,
                now,
                escape_json(model_id),
                escape_json(&output),
                stats.prompt_tokens,
                stats.generated_tokens,
                stats.prompt_tokens + stats.generated_tokens,
                stats.cached_prompt_tokens,
                stats.prefill_tps(),
                stats.decode_tps()
            );
            write_response(stream, "200 OK", &resp)
        }
        ("POST", "/v1/celestia/propose") => {
            let req = JsonValue::parse(&body_str).unwrap_or(JsonValue::Null);
            let prompt = req
                .get("prompt")
                .and_then(JsonValue::as_str)
                .unwrap_or("Create a design card")
                .to_string();
            let mode = req
                .get("workspace_mode")
                .and_then(JsonValue::as_str)
                .unwrap_or("design")
                .to_string();
            let request_id = req
                .get("request_id")
                .and_then(JsonValue::as_str)
                .unwrap_or("req_caelum")
                .to_string();
            let fp = req
                .get("context_fingerprint")
                .and_then(JsonValue::as_str)
                .unwrap_or("fp_caelum")
                .to_string();
            let max_tokens = req
                .get("max_tokens")
                .and_then(JsonValue::as_usize)
                .unwrap_or(768);

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
                max_tokens,
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

fn parse_chat_messages(req: &JsonValue) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    if let Some(arr) = req.get("messages").and_then(JsonValue::as_array) {
        for item in arr {
            let role = match item.get("role").and_then(JsonValue::as_str).unwrap_or("user") {
                "system" | "developer" => ChatRole::System,
                "assistant" => ChatRole::Assistant,
                _ => ChatRole::User,
            };
            let content = match item.get("content") {
                Some(JsonValue::Str(s)) => s.clone(),
                Some(JsonValue::Array(parts)) => {
                    let mut combined = String::new();
                    for p in parts {
                        if let Some(txt) = p.get("text").and_then(JsonValue::as_str) {
                            combined.push_str(txt);
                        }
                    }
                    combined
                }
                _ => String::new(),
            };
            out.push(ChatMessage { role, content });
        }
    }
    if out.is_empty() {
        out.push(ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        });
    }
    out
}

fn parse_stop_sequences(req: &JsonValue) -> Vec<String> {
    match req.get("stop") {
        Some(JsonValue::Str(s)) => vec![s.clone()],
        Some(JsonValue::Array(arr)) => arr
            .iter()
            .filter_map(JsonValue::as_str)
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
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
