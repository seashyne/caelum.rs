//! Chat prompt templates with GGUF auto-detection for Qwen2.5/Qwen3 (ChatML),
//! Llama 3, DeepSeek-V2/Coder, Mistral, and Celestia.

use caelum_gguf::GgufFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\ncontent<|im_end|>\n` (Qwen2 / Qwen2.5 / Qwen3 / Yi-Coder / OpenCoder).
    ChatMl,
    /// Qwen3 ChatML with empty `<think>\n\n</think>\n\n` prefill for direct fast coding answers.
    Qwen3NoThink,
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` (Llama 3 / 3.1 / 3.2).
    Llama3,
    /// `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` (Gemma 2 / Gemma 3).
    Gemma,
    /// `<|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n` (Phi-3).
    Phi3,
    /// DeepSeek-V2 / DeepSeek-Coder-V2 (`User: ...\n\nAssistant:`).
    DeepSeekV2,
    /// DeepSeek-Coder v1 (`### Instruction:\n...\n### Response:\n`).
    DeepSeekV1,
    /// Mistral / CodeLlama (`[INST] ... [/INST]`).
    Mistral,
    /// Native Celestia XML tags (`<role>\n...\n</role>\n`).
    Celestia,
    /// Raw prompt text without template wrapping.
    Raw,
}

impl ChatTemplate {
    /// Auto-detects the best chat template by inspecting `tokenizer.chat_template` in GGUF metadata
    /// first, then falling back to `general.architecture`.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().unwrap_or("llama");
        if let Some(tmpl) = gguf
            .get_meta("tokenizer.chat_template")
            .and_then(|v| v.as_str())
        {
            if tmpl.contains("<|im_start|>") {
                return ChatTemplate::ChatMl;
            }
            if tmpl.contains("<|start_header_id|>") {
                return ChatTemplate::Llama3;
            }
            if tmpl.contains("<start_of_turn>") {
                return ChatTemplate::Gemma;
            }
            if tmpl.contains("<|user|>") && tmpl.contains("<|end|>") {
                return ChatTemplate::Phi3;
            }
            if tmpl.contains("### Instruction:") {
                return ChatTemplate::DeepSeekV1;
            }
            if tmpl.contains("<｜begin▁of▁sentence｜>")
                || (tmpl.contains("User:") && tmpl.contains("Assistant:"))
            {
                return ChatTemplate::DeepSeekV2;
            }
            if tmpl.contains("[INST]") {
                return ChatTemplate::Mistral;
            }
        }
        Self::from_architecture(arch)
    }

    pub fn from_architecture(arch: &str) -> Self {
        match arch {
            "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" => ChatTemplate::ChatMl,
            "gemma" | "gemma2" | "gemma3" => ChatTemplate::Gemma,
            "phi3" => ChatTemplate::Phi3,
            "starcoder2" => ChatTemplate::Raw,
            "deepseek2" => ChatTemplate::DeepSeekV2,
            "mistral" => ChatTemplate::Mistral,
            "celestia" => ChatTemplate::Celestia,
            "llama" => ChatTemplate::Llama3,
            _ => ChatTemplate::ChatMl,
        }
    }

    /// Parses a user-supplied `--template <name>` CLI flag.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "chatml" | "qwen" | "qwen2" | "qwen3" | "phi4" => Some(ChatTemplate::ChatMl),
            "qwen3-nothink" | "nothink" => Some(ChatTemplate::Qwen3NoThink),
            "llama3" | "llama" => Some(ChatTemplate::Llama3),
            "gemma" | "gemma2" | "gemma3" => Some(ChatTemplate::Gemma),
            "phi3" | "phi" => Some(ChatTemplate::Phi3),
            "deepseek2" | "deepseek-v2" | "deepseek" => Some(ChatTemplate::DeepSeekV2),
            "deepseek-v1" | "deepseek-coder" => Some(ChatTemplate::DeepSeekV1),
            "mistral" | "codellama" | "codestral" => Some(ChatTemplate::Mistral),
            "celestia" => Some(ChatTemplate::Celestia),
            "raw" | "starcoder2" => Some(ChatTemplate::Raw),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::ChatMl => "chatml",
            ChatTemplate::Qwen3NoThink => "qwen3-nothink",
            ChatTemplate::Llama3 => "llama3",
            ChatTemplate::Gemma => "gemma",
            ChatTemplate::Phi3 => "phi3",
            ChatTemplate::DeepSeekV2 => "deepseek2",
            ChatTemplate::DeepSeekV1 => "deepseek-v1",
            ChatTemplate::Mistral => "mistral",
            ChatTemplate::Celestia => "celestia",
            ChatTemplate::Raw => "raw",
        }
    }

    pub fn format(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        match self {
            ChatTemplate::ChatMl | ChatTemplate::Qwen3NoThink => {
                for m in messages {
                    let role = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|im_start|>assistant\n");
                    if matches!(self, ChatTemplate::Qwen3NoThink) {
                        out.push_str("<think>\n\n</think>\n\n");
                    }
                }
            }
            ChatTemplate::Llama3 => {
                for m in messages {
                    let role = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!(
                        "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.content
                    ));
                }
                if add_generation_prompt {
                    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            }
            ChatTemplate::Gemma => {
                let mut pending_sys: Option<&str> = None;
                for m in messages {
                    match m.role {
                        ChatRole::System => pending_sys = Some(m.content.as_str()),
                        ChatRole::User => {
                            out.push_str("<start_of_turn>user\n");
                            if let Some(sys) = pending_sys.take() {
                                out.push_str(sys);
                                out.push_str("\n\n");
                            }
                            out.push_str(&m.content);
                            out.push_str("<end_of_turn>\n");
                        }
                        ChatRole::Assistant => {
                            out.push_str("<start_of_turn>model\n");
                            out.push_str(&m.content);
                            out.push_str("<end_of_turn>\n");
                        }
                    }
                }
                if add_generation_prompt {
                    out.push_str("<start_of_turn>model\n");
                }
            }
            ChatTemplate::Phi3 => {
                for m in messages {
                    let tag = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!("<|{tag}|>\n{}<|end|>\n", m.content));
                }
                if add_generation_prompt {
                    out.push_str("<|assistant|>\n");
                }
            }
            ChatTemplate::DeepSeekV2 => {
                for m in messages {
                    match m.role {
                        ChatRole::System => {
                            out.push_str(&m.content);
                            out.push_str("\n\n");
                        }
                        ChatRole::User => {
                            out.push_str("User: ");
                            out.push_str(&m.content);
                            out.push_str("\n\n");
                        }
                        ChatRole::Assistant => {
                            out.push_str("Assistant: ");
                            out.push_str(&m.content);
                            out.push_str("<｜end▁of▁sentence｜>");
                        }
                    }
                }
                if add_generation_prompt {
                    out.push_str("Assistant:");
                }
            }
            ChatTemplate::DeepSeekV1 => {
                for m in messages {
                    match m.role {
                        ChatRole::System => {
                            out.push_str(&m.content);
                            out.push('\n');
                        }
                        ChatRole::User => {
                            out.push_str("### Instruction:\n");
                            out.push_str(&m.content);
                            out.push('\n');
                        }
                        ChatRole::Assistant => {
                            out.push_str("### Response:\n");
                            out.push_str(&m.content);
                            out.push_str("\n<|EOT|>\n");
                        }
                    }
                }
                if add_generation_prompt {
                    out.push_str("### Response:\n");
                }
            }
            ChatTemplate::Mistral => {
                let mut pending_sys: Option<&str> = None;
                for m in messages {
                    match m.role {
                        ChatRole::System => pending_sys = Some(m.content.as_str()),
                        ChatRole::User => {
                            out.push_str("[INST] ");
                            if let Some(sys) = pending_sys.take() {
                                out.push_str(sys);
                                out.push_str("\n\n");
                            }
                            out.push_str(&m.content);
                            out.push_str(" [/INST]");
                        }
                        ChatRole::Assistant => {
                            out.push(' ');
                            out.push_str(&m.content);
                            out.push_str("</s> ");
                        }
                    }
                }
            }
            ChatTemplate::Celestia => {
                for m in messages {
                    let role = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!("<{role}>\n{}\n</{role}>\n", m.content));
                }
                if add_generation_prompt {
                    out.push_str("<assistant>\n");
                }
            }
            ChatTemplate::Raw => {
                for m in messages {
                    out.push_str(&m.content);
                }
            }
        }
        out
    }
}
