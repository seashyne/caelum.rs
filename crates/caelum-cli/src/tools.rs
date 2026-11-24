//! OpenAI-compatible Tool Calling / Function Calling engine for `/v1/chat/completions`.
//!
//! Enables coding agents such as **Cline**, **OpenCode**, **Continue.dev**, and standard OpenAI
//! SDKs to invoke tools through Caelum in both non-streaming JSON and streaming SSE modes:
//! - Parses `tools[]`, `tool_choice`, `role: "tool"` messages, and `assistant.tool_calls[]` history
//! - Injects native Hermes / Qwen2.5 / Qwen3 `<tools>...</tools>` schemas into the system prompt
//! - Extracts single or parallel `<tool_call>...</tool_call>` blocks (plus fenced/bare JSON
//!   fallbacks validated against declared tool names)
//! - Streams prose text immediately while buffering `<tool_call>` tags cleanly for SSE `delta.tool_calls`

use crate::json::{escape_json_str, JsonValue};
use caelum_tokenizer::{ChatMessage, ChatRole};

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub raw_definition_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ParsedToolCall {
    pub fn to_openai_json(&self) -> String {
        format!(
            r#"{{"id":"{}","type":"function","function":{{"name":"{}","arguments":"{}"}}}}"#,
            escape_json_str(&self.id),
            escape_json_str(&self.name),
            escape_json_str(&self.arguments_json)
        )
    }
}

/// Parses OpenAI `tools` array and `tool_choice` from `/v1/chat/completions` request JSON.
pub fn parse_tools_and_choice(req: &JsonValue) -> (Vec<ToolSpec>, ToolChoice) {
    let mut specs = Vec::new();
    if let Some(arr) = req.get("tools").and_then(JsonValue::as_array) {
        for item in arr {
            let func = item.get("function").unwrap_or(item);
            if let Some(name) = func.get("name").and_then(JsonValue::as_str) {
                if !name.is_empty() {
                    let raw_def = if item.get("function").is_some() {
                        item.to_json_string()
                    } else {
                        format!(r#"{{"type":"function","function":{}}}"#, item.to_json_string())
                    };
                    specs.push(ToolSpec {
                        name: name.to_string(),
                        raw_definition_json: raw_def,
                    });
                }
            }
        }
    }

    if specs.is_empty() {
        return (specs, ToolChoice::None);
    }

    let choice = match req.get("tool_choice") {
        Some(JsonValue::Str(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" | "any" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Some(obj @ JsonValue::Object(_)) => {
            if let Some(name) = obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(JsonValue::as_str)
            {
                ToolChoice::Named(name.to_string())
            } else {
                ToolChoice::Auto
            }
        }
        _ => ToolChoice::Auto,
    };

    (specs, choice)
}

/// Parses OpenAI `messages[]`, including `assistant.tool_calls[]` history and `role: "tool"`
/// responses (rendered in Qwen2.5/Qwen3/Hermes `<tool_call>` / `<tool_response>` format).
pub fn parse_openai_messages(req: &JsonValue) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::new();
    if let Some(arr) = req.get("messages").and_then(JsonValue::as_array) {
        for item in arr {
            let role_str = item
                .get("role")
                .and_then(JsonValue::as_str)
                .unwrap_or("user");

            let text = extract_content_text(item.get("content"));

            if role_str == "tool" {
                // Exact Qwen2.5 / Qwen3 / Hermes `<tool_response>` format (raw content, no double-escape)
                let block = format!("<tool_response>\n{text}\n</tool_response>");
                // Merge consecutive `role: "tool"` results into a single User turn per Qwen/Hermes spec
                if let Some(last) = out.last_mut() {
                    if last.role == ChatRole::User && last.content.contains("<tool_response>") {
                        last.content.push('\n');
                        last.content.push_str(&block);
                        continue;
                    }
                }
                out.push(ChatMessage {
                    role: ChatRole::User,
                    content: block,
                });
                continue;
            }

            let role = match role_str {
                "system" | "developer" => ChatRole::System,
                "assistant" => ChatRole::Assistant,
                _ => ChatRole::User,
            };

            let mut content = text;
            if role == ChatRole::Assistant {
                if let Some(tc_arr) = item.get("tool_calls").and_then(JsonValue::as_array) {
                    for tc in tc_arr {
                        let func = tc.get("function").unwrap_or(tc);
                        let Some(name) = func.get("name").and_then(JsonValue::as_str) else {
                            continue;
                        };
                        let args_rendered = match func.get("arguments") {
                            Some(JsonValue::Str(s)) => {
                                // Render valid JSON objects inline without double-escaping
                                if JsonValue::parse(s).is_ok() {
                                    s.clone()
                                } else {
                                    format!(r#""{}""#, escape_json_str(s))
                                }
                            }
                            Some(v) => v.to_json_string(),
                            None => "{}".to_string(),
                        };
                        if !content.is_empty() && !content.ends_with('\n') {
                            content.push('\n');
                        }
                        content.push_str(&format!(
                            "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
                            escape_json_str(name),
                            args_rendered
                        ));
                    }
                }
            }

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

fn extract_content_text(val: Option<&JsonValue>) -> String {
    match val {
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
    }
}

/// Injects Qwen2.5 / Qwen3 / Hermes `<tools>...</tools>` signatures into the system prompt.
pub fn inject_tools_into_messages(
    messages: &mut Vec<ChatMessage>,
    tools: &[ToolSpec],
    choice: &ToolChoice,
) {
    if tools.is_empty() || *choice == ToolChoice::None {
        return;
    }

    let mut block = String::from(
        "# Tools\n\n\
        You may call one or more functions to assist with the user query.\n\n\
        You are provided with function signatures within <tools></tools> XML tags:\n\
        <tools>\n",
    );
    for t in tools {
        block.push_str(&t.raw_definition_json);
        block.push('\n');
    }
    block.push_str(
        "</tools>\n\n\
        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
        <tool_call>\n\
        {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
        </tool_call>",
    );

    match choice {
        ToolChoice::Required => {
            block.push_str(
                "\n\nIMPORTANT: You MUST invoke at least one function from <tools> using <tool_call>...</tool_call> tags now.",
            );
        }
        ToolChoice::Named(name) => {
            block.push_str(&format!(
                "\n\nIMPORTANT: You MUST invoke the function \"{name}\" using <tool_call>...</tool_call> tags now."
            ));
        }
        _ => {}
    }

    if let Some(first) = messages.first_mut() {
        if first.role == ChatRole::System {
            if !first.content.is_empty() {
                first.content.push_str("\n\n");
            }
            first.content.push_str(&block);
            return;
        }
    }
    messages.insert(
        0,
        ChatMessage {
            role: ChatRole::System,
            content: block,
        },
    );
}

/// Extracts tool calls from raw model output (supports multiple `<tool_call>...</tool_call>`
/// tags, unclosed `<tool_call>{...}` at EOF, fenced JSON blocks, and bare JSON calls).
pub fn extract_tool_calls(
    raw_output: &str,
    tools: &[ToolSpec],
    id_seed: u64,
) -> (Option<String>, Vec<ParsedToolCall>) {
    if tools.is_empty() {
        return (Some(raw_output.to_string()), Vec::new());
    }

    let mut calls = Vec::new();
    let mut prose = String::new();
    let mut cursor = 0usize;
    let mut found_tag = false;

    while let Some(rel_start) = raw_output[cursor..].find("<tool_call>") {
        found_tag = true;
        let abs_start = cursor + rel_start;
        prose.push_str(&raw_output[cursor..abs_start]);
        let body_start = abs_start + "<tool_call>".len();
        if let Some(rel_end) = raw_output[body_start..].find("</tool_call>") {
            let body = raw_output[body_start..body_start + rel_end].trim();
            collect_calls_from_json_str(body, tools, id_seed, &mut calls, false);
            cursor = body_start + rel_end + "</tool_call>".len();
        } else {
            // Unclosed <tool_call> at end of output
            let body = raw_output[body_start..].trim();
            collect_calls_from_json_str(body, tools, id_seed, &mut calls, false);
            cursor = raw_output.len();
            break;
        }
    }

    if found_tag {
        prose.push_str(&raw_output[cursor..]);
        let cleaned = prose.trim();
        let rem = if cleaned.is_empty() {
            None
        } else {
            Some(cleaned.to_string())
        };
        return (rem, calls);
    }

    // Fallback 1: fenced ```json { "name": ..., "arguments": ... } ```
    if let Some((rem, fenced_calls)) = extract_fenced_tool_calls(raw_output, tools, id_seed) {
        if !fenced_calls.is_empty() {
            return (rem, fenced_calls);
        }
    }

    // Fallback 2: bare JSON object/array matching a declared tool name
    let trimmed = raw_output.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        collect_calls_from_json_str(trimmed, tools, id_seed, &mut calls, true);
        if !calls.is_empty() {
            return (None, calls);
        }
    }

    (Some(raw_output.to_string()), Vec::new())
}

fn extract_fenced_tool_calls(
    raw: &str,
    tools: &[ToolSpec],
    id_seed: u64,
) -> Option<(Option<String>, Vec<ParsedToolCall>)> {
    let mut calls = Vec::new();
    let mut prose = String::new();
    let mut cursor = 0usize;

    while let Some(rel_open) = raw[cursor..].find("```") {
        let open = cursor + rel_open;
        let after_ticks = open + 3;
        let Some(nl_rel) = raw[after_ticks..].find('\n') else {
            break;
        };
        let body_start = after_ticks + nl_rel + 1;
        let Some(rel_close) = raw[body_start..].find("```") else {
            break;
        };
        let body = raw[body_start..body_start + rel_close].trim();
        let before_len = calls.len();
        collect_calls_from_json_str(body, tools, id_seed, &mut calls, true);
        if calls.len() > before_len {
            prose.push_str(&raw[cursor..open]);
            cursor = body_start + rel_close + 3;
        } else {
            prose.push_str(&raw[cursor..body_start + rel_close + 3]);
            cursor = body_start + rel_close + 3;
        }
    }

    if calls.is_empty() {
        return None;
    }
    prose.push_str(&raw[cursor..]);
    let rem = prose.trim();
    Some((
        if rem.is_empty() {
            None
        } else {
            Some(rem.to_string())
        },
        calls,
    ))
}

fn collect_calls_from_json_str(
    body: &str,
    tools: &[ToolSpec],
    id_seed: u64,
    out: &mut Vec<ParsedToolCall>,
    require_known_tool: bool,
) {
    let cleaned = body
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let Ok(val) = JsonValue::parse(cleaned) else {
        return;
    };
    match val {
        JsonValue::Object(_) => {
            if let Some(call) =
                json_obj_to_tool_call(&val, tools, id_seed, out.len(), require_known_tool)
            {
                out.push(call);
            }
        }
        JsonValue::Array(items) => {
            for item in items {
                if let Some(call) =
                    json_obj_to_tool_call(&item, tools, id_seed, out.len(), require_known_tool)
                {
                    out.push(call);
                }
            }
        }
        _ => {}
    }
}

fn json_obj_to_tool_call(
    obj: &JsonValue,
    tools: &[ToolSpec],
    id_seed: u64,
    idx: usize,
    require_known_tool: bool,
) -> Option<ParsedToolCall> {
    let name = obj
        .get("name")
        .or_else(|| obj.get("function").and_then(|f| f.get("name")))
        .and_then(JsonValue::as_str)?;
    if name.is_empty() {
        return None;
    }
    if require_known_tool && !tools.iter().any(|t| t.name == name) {
        return None;
    }

    let args_val = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .or_else(|| obj.get("args"))
        .or_else(|| obj.get("function").and_then(|f| f.get("arguments")));

    let arguments_json = match args_val {
        Some(JsonValue::Str(s)) => s.clone(),
        Some(v) => v.to_json_string(),
        None => "{}".to_string(),
    };

    Some(ParsedToolCall {
        id: format!("call_caelum_{id_seed}_{idx}"),
        name: name.to_string(),
        arguments_json,
    })
}

pub enum StreamEvent<'a> {
    Text(&'a str),
    ToolCall(usize, &'a ParsedToolCall),
}

/// Real-time SSE streaming filter when `tools` is active:
/// - Immediately flushes normal prose text (`StreamEvent::Text`) with zero extra latency
/// - Cleanly suppresses & buffers `<tool_call>...</tool_call>` blocks and emits OpenAI
///   `delta.tool_calls` events (`StreamEvent::ToolCall`) the moment each call completes!
pub struct ToolStreamFilter<'a> {
    tools: &'a [ToolSpec],
    id_seed: u64,
    buf: String,
    in_tool_tag: bool,
    emitted_any_text: bool,
    pub emitted_tool_calls: usize,
}

impl<'a> ToolStreamFilter<'a> {
    pub fn new(tools: &'a [ToolSpec], id_seed: u64) -> Self {
        Self {
            tools,
            id_seed,
            buf: String::new(),
            in_tool_tag: false,
            emitted_any_text: false,
            emitted_tool_calls: 0,
        }
    }

    /// Feeds one newly decoded token piece into the streaming filter.
    pub fn push<F>(&mut self, piece: &str, mut on_event: F)
    where
        F: FnMut(StreamEvent<'_>),
    {
        self.buf.push_str(piece);

        loop {
            if self.in_tool_tag {
                if let Some(end_idx) = self.buf.find("</tool_call>") {
                    let body = self.buf[..end_idx].trim().to_string();
                    self.buf.drain(..end_idx + "</tool_call>".len());
                    self.in_tool_tag = false;
                    let mut parsed = Vec::new();
                    collect_calls_from_json_str(&body, self.tools, self.id_seed, &mut parsed, false);
                    for mut call in parsed {
                        let idx = self.emitted_tool_calls;
                        call.id = format!("call_caelum_{}_{}", self.id_seed, idx);
                        on_event(StreamEvent::ToolCall(idx, &call));
                        self.emitted_tool_calls += 1;
                    }
                    continue;
                }
                break;
            }

            if let Some(start_idx) = self.buf.find("<tool_call>") {
                let before = self.buf[..start_idx].to_string();
                if !before.is_empty() && !(before.trim().is_empty() && !self.emitted_any_text) {
                    on_event(StreamEvent::Text(&before));
                    self.emitted_any_text = true;
                }
                self.buf.drain(..start_idx + "<tool_call>".len());
                self.in_tool_tag = true;
                continue;
            }

            // If nothing emitted yet and buffer starts like bare JSON or ```json, hold briefly
            // (up to 4 KB) so fallback JSON tool-calls don't leak as prose.
            let trimmed = self.buf.trim_start();
            if !self.emitted_any_text
                && self.emitted_tool_calls == 0
                && self.buf.len() < 4096
                && (trimmed.is_empty()
                    || trimmed.starts_with('{')
                    || trimmed.starts_with("```")
                    || trimmed.starts_with('<'))
            {
                break;
            }

            // Keep any trailing partial prefix of "<tool_call>" buffered
            let hold = trailing_tag_prefix_len(&self.buf, "<tool_call>");
            let flush_len = self.buf.len() - hold;
            if flush_len > 0 {
                let to_emit = self.buf[..flush_len].to_string();
                self.buf.drain(..flush_len);
                on_event(StreamEvent::Text(&to_emit));
                self.emitted_any_text = true;
            }
            break;
        }
    }

    /// Flushes any buffered content at end of generation, handling unclosed tags or bare/fenced JSON.
    pub fn finish<F>(&mut self, mut on_event: F)
    where
        F: FnMut(StreamEvent<'_>),
    {
        if self.buf.is_empty() {
            return;
        }
        let rem = std::mem::take(&mut self.buf);
        if self.in_tool_tag {
            let mut parsed = Vec::new();
            collect_calls_from_json_str(rem.trim(), self.tools, self.id_seed, &mut parsed, false);
            for mut call in parsed {
                let idx = self.emitted_tool_calls;
                call.id = format!("call_caelum_{}_{}", self.id_seed, idx);
                on_event(StreamEvent::ToolCall(idx, &call));
                self.emitted_tool_calls += 1;
            }
            return;
        }

        let (text_opt, calls) = extract_tool_calls(&rem, self.tools, self.id_seed);
        if !calls.is_empty() {
            if let Some(t) = text_opt {
                if !t.is_empty() {
                    on_event(StreamEvent::Text(&t));
                }
            }
            for mut call in calls {
                let idx = self.emitted_tool_calls;
                call.id = format!("call_caelum_{}_{}", self.id_seed, idx);
                on_event(StreamEvent::ToolCall(idx, &call));
                self.emitted_tool_calls += 1;
            }
        } else if !rem.is_empty() {
            on_event(StreamEvent::Text(&rem));
        }
    }
}

fn trailing_tag_prefix_len(s: &str, tag: &str) -> usize {
    let max_k = s.len().min(tag.len().saturating_sub(1));
    for k in (1..=max_k).rev() {
        if s.is_char_boundary(s.len() - k) && s.ends_with(&tag[..k]) {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tools() -> Vec<ToolSpec> {
        let req = JsonValue::parse(
            r#"{
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file from disk",
                            "parameters": {
                                "type": "object",
                                "properties": { "path": { "type": "string" } },
                                "required": ["path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "description": "Run a shell command",
                            "parameters": {
                                "type": "object",
                                "properties": { "cmd": { "type": "string" } },
                                "required": ["cmd"]
                            }
                        }
                    }
                ],
                "tool_choice": "auto"
            }"#,
        )
        .unwrap();
        let (tools, choice) = parse_tools_and_choice(&req);
        assert_eq!(choice, ToolChoice::Auto);
        tools
    }

    #[test]
    fn test_extract_parallel_tool_calls_and_prose() {
        let tools = sample_tools();
        let raw = "Let me check both files.\n<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}\n</tool_call>\n<tool_call>\n{\"name\":\"run_command\",\"arguments\":{\"cmd\":\"cargo test\"}}\n</tool_call>";
        let (prose, calls) = extract_tool_calls(raw, &tools, 42);
        assert_eq!(prose.as_deref(), Some("Let me check both files."));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments_json, r#"{"path":"Cargo.toml"}"#);
        assert_eq!(calls[1].name, "run_command");
        assert_eq!(calls[1].arguments_json, r#"{"cmd":"cargo test"}"#);
    }

    #[test]
    fn test_streaming_filter_buffers_tags_and_emits_tool_calls() {
        let tools = sample_tools();
        let mut filter = ToolStreamFilter::new(&tools, 99);
        let tokens = [
            "Checking ",
            "file.<to",
            "ol_call>\n",
            "{\"name\": \"read_file\", ",
            "\"arguments\": {\"path\": \"src/main.rs\"}}",
            "\n</tool_",
            "call>",
        ];
        let mut streamed_text = String::new();
        let mut streamed_calls = Vec::new();
        for tok in tokens {
            filter.push(tok, |ev| match ev {
                StreamEvent::Text(t) => streamed_text.push_str(t),
                StreamEvent::ToolCall(idx, c) => streamed_calls.push((idx, c.clone())),
            });
        }
        filter.finish(|ev| match ev {
            StreamEvent::Text(t) => streamed_text.push_str(t),
            StreamEvent::ToolCall(idx, c) => streamed_calls.push((idx, c.clone())),
        });
        assert_eq!(streamed_text, "Checking file.");
        assert_eq!(streamed_calls.len(), 1);
        assert_eq!(streamed_calls[0].0, 0);
        assert_eq!(streamed_calls[0].1.name, "read_file");
        assert_eq!(streamed_calls[0].1.arguments_json, r#"{"path":"src/main.rs"}"#);
    }

    #[test]
    fn test_multiturn_tool_history_parsing() {
        let req = JsonValue::parse(
            r#"{
                "messages": [
                    {"role": "user", "content": "Read Cargo.toml"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"Cargo.toml\"}"}}
                        ]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "version = 0.7.0"}
                ]
            }"#,
        )
        .unwrap();
        let msgs = parse_openai_messages(&req);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, ChatRole::Assistant);
        assert!(msgs[1].content.contains("<tool_call>"));
        assert!(msgs[1].content.contains("read_file"));
        assert_eq!(msgs[2].role, ChatRole::User);
        assert!(msgs[2].content.contains("<tool_response>"));
        assert!(msgs[2].content.contains("version = 0.7.0"));
    }
}
