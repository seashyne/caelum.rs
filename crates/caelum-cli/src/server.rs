//! Zero-dependency HTTP/1.1 API Server for Caelum (OpenAI-compatible + Lenout Action Proposal).

use crate::lenout::build_lenout_system_prompt;
use caelum_core::Result;
use caelum_gguf::GgufFile;
use caelum_kernels::{Backend, CpuBackend};
use caelum_models::{CaelumModel, SamplerConfig};
use caelum_tokenizer::{ChatMessage, ChatRole, ChatTemplate, Tokenizer};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn run_server(
    model_path: &str,
    host: &str,
    port: u16,
    threads: Option<usize>,
    max_ctx: Option<usize>,
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
    let mut model = CaelumModel::from_gguf(&gguf, max_ctx)?;
    let template = ChatTemplate::from_architecture(&arch);

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("============================================================");
    println!("  CAELUM RUST INFERENCE SERVER READY");
    println!("  Listen:   http://{addr}");
    println!("  Model:    {arch} ({:.1}M params)", gguf.total_parameters() as f64 / 1e6);
    println!("  Hardware: {} ({} threads)", backend.caps(), backend.num_threads());
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    POST /v1/chat/completions");
    println!("    POST /v1/celestia/propose");
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
                r#"{{"status":"healthy","engine":"caelum-rs","architecture":"{}","kernel":"{}","threads":{},"timestamp":{}}}"#,
                escape_json(arch),
                escape_json(backend.name()),
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

            let messages = vec![ChatMessage {
                role: ChatRole::User,
                content: user_prompt,
            }];
            let formatted = template.format(&messages, true);
            let prompt_tokens = tokenizer.encode(&formatted, tokenizer.add_bos);

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
                r#"{{"id":"chatcmpl-caelum-{}","object":"chat.completion","created":{},"model":"caelum-{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"decode_tps":{:.2}}}}}"#,
                unix_ts(),
                unix_ts(),
                escape_json(arch),
                escape_json(&output),
                stats.prompt_tokens,
                stats.generated_tokens,
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
                .unwrap_or_else(|| format!("req_{}", unix_ts()));
            let fp = extract_last_json_string(&body_str, "context_fingerprint")
                .unwrap_or_else(|| format!("fp_{}", unix_ts()));

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

            // Constrained JSON decoding guarantees valid JSON output from the engine itself!
            let (output, stats) = model.generate(
                backend,
                tokenizer,
                &prompt_tokens,
                512,
                &SamplerConfig {
                    temperature: 0.2,
                    ..Default::default()
                },
                true,
                |_, _| {},
            )?;

            let trimmed = output.trim();
            let resp = if trimmed.starts_with('{') && trimmed.ends_with('}') {
                format!(
                    r#"{{"ok":true,"engine":"caelum-rs","decode_tps":{:.2},"proposal":{}}}"#,
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
        _ => write_response(
            stream,
            "404 Not Found",
            r#"{"error":"route not found"}"#,
        ),
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

/// Lightweight zero-dep extractor for the last occurrence of `"key": "value"` in a JSON payload.
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
