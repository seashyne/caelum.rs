//! Chat prompt templates for Qwen2.5 (ChatML), Llama 3, and Celestia native tags.

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
    ChatMl,
    Llama3,
    Celestia,
    Raw,
}

impl ChatTemplate {
    pub fn from_architecture(arch: &str) -> Self {
        match arch {
            "qwen2" | "qwen2moe" | "qwen3" => ChatTemplate::ChatMl,
            "celestia" => ChatTemplate::Celestia,
            "llama" => ChatTemplate::Llama3,
            _ => ChatTemplate::ChatMl,
        }
    }

    pub fn format(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut out = String::new();
        match self {
            ChatTemplate::ChatMl => {
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
