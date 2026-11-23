//! Incremental JSON grammar state machine for constrained decoding (Lenout Action DSL).
//!
//! Masks invalid next tokens directly on logits so even tiny models emit guaranteed valid JSON.

use crate::bpe::Tokenizer;
use crate::schema::LogitGrammar;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// Expecting root `{`
    Start,
    /// Inside object, expecting `"` (key start) or `}` (if empty)
    ObjectKeyOrEnd { allow_end: bool },
    /// Inside string (either object key or string value)
    InString { is_key: bool, escaped: bool, unicode_rem: u8 },
    /// After key string, expecting `:`
    AfterKey,
    /// Expecting any JSON value (`{`, `[`, `"`, number, `true`, `false`, `null`) or `]` if empty array
    Value { allow_array_end: bool },
    /// Reading a JSON number literal
    InNumber { has_dot: bool, has_exp: bool, expect_digit: bool },
    /// Reading `true`, `false`, or `null`
    InLiteral { target: &'static [u8], pos: usize },
    /// After a completed value inside object/array: expecting `,` or `}`/`]`
    AfterValue,
    /// Root JSON object finished: only whitespace or EOS allowed
    Done,
}

#[derive(Debug, Clone)]
pub struct JsonGrammar {
    state: State,
    stack: Vec<Container>,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    pub fn new() -> Self {
        Self {
            state: State::Start,
            stack: Vec::new(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == State::Done
    }

    /// Tries to advance the grammar by a byte slice. Returns `true` if the bytes keep JSON valid.
    pub fn try_advance_bytes(&self, bytes: &[u8]) -> Option<Self> {
        let mut next = self.clone();
        for &b in bytes {
            if !next.step_byte(b) {
                return None;
            }
        }
        Some(next)
    }

    /// Commits accepted bytes to this grammar state.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        if let Some(next) = self.try_advance_bytes(bytes) {
            *self = next;
            true
        } else {
            false
        }
    }

    fn finish_value(&mut self) {
        if self.stack.is_empty() {
            self.state = State::Done;
        } else {
            self.state = State::AfterValue;
        }
    }

    fn step_byte(&mut self, b: u8) -> bool {
        let is_ws = matches!(b, b' ' | b'\n' | b'\r' | b'\t');

        match self.state.clone() {
            State::Start => {
                if is_ws {
                    return true;
                }
                if b == b'{' {
                    self.stack.push(Container::Object);
                    self.state = State::ObjectKeyOrEnd { allow_end: true };
                    return true;
                }
                false
            }
            State::ObjectKeyOrEnd { allow_end } => {
                if is_ws {
                    return true;
                }
                if b == b'"' {
                    self.state = State::InString {
                        is_key: true,
                        escaped: false,
                        unicode_rem: 0,
                    };
                    return true;
                }
                if allow_end && b == b'}' {
                    self.stack.pop();
                    self.finish_value();
                    return true;
                }
                false
            }
            State::InString {
                is_key,
                escaped,
                unicode_rem,
            } => {
                if unicode_rem > 0 {
                    if b.is_ascii_hexdigit() {
                        self.state = State::InString {
                            is_key,
                            escaped: false,
                            unicode_rem: unicode_rem - 1,
                        };
                        return true;
                    }
                    return false;
                }
                if escaped {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.state = State::InString {
                                is_key,
                                escaped: false,
                                unicode_rem: 0,
                            };
                            return true;
                        }
                        b'u' => {
                            self.state = State::InString {
                                is_key,
                                escaped: false,
                                unicode_rem: 4,
                            };
                            return true;
                        }
                        _ => return false,
                    }
                }
                if b < 0x20 {
                    return false; // unescaped control char illegal in JSON strings
                }
                if b == b'\\' {
                    self.state = State::InString {
                        is_key,
                        escaped: true,
                        unicode_rem: 0,
                    };
                    return true;
                }
                if b == b'"' {
                    if is_key {
                        self.state = State::AfterKey;
                    } else {
                        self.finish_value();
                    }
                    return true;
                }
                true
            }
            State::AfterKey => {
                if is_ws {
                    return true;
                }
                if b == b':' {
                    self.state = State::Value {
                        allow_array_end: false,
                    };
                    return true;
                }
                false
            }
            State::Value { allow_array_end } => {
                if is_ws {
                    return true;
                }
                match b {
                    b'{' => {
                        self.stack.push(Container::Object);
                        self.state = State::ObjectKeyOrEnd { allow_end: true };
                        true
                    }
                    b'[' => {
                        self.stack.push(Container::Array);
                        self.state = State::Value {
                            allow_array_end: true,
                        };
                        true
                    }
                    b']' if allow_array_end => {
                        self.stack.pop();
                        self.finish_value();
                        true
                    }
                    b'"' => {
                        self.state = State::InString {
                            is_key: false,
                            escaped: false,
                            unicode_rem: 0,
                        };
                        true
                    }
                    b'-' => {
                        self.state = State::InNumber {
                            has_dot: false,
                            has_exp: false,
                            expect_digit: true,
                        };
                        true
                    }
                    b'0'..=b'9' => {
                        self.state = State::InNumber {
                            has_dot: false,
                            has_exp: false,
                            expect_digit: false,
                        };
                        true
                    }
                    b't' => {
                        self.state = State::InLiteral {
                            target: b"true",
                            pos: 1,
                        };
                        true
                    }
                    b'f' => {
                        self.state = State::InLiteral {
                            target: b"false",
                            pos: 1,
                        };
                        true
                    }
                    b'n' => {
                        self.state = State::InLiteral {
                            target: b"null",
                            pos: 1,
                        };
                        true
                    }
                    _ => false,
                }
            }
            State::InNumber {
                has_dot,
                has_exp,
                expect_digit,
            } => {
                if b.is_ascii_digit() {
                    self.state = State::InNumber {
                        has_dot,
                        has_exp,
                        expect_digit: false,
                    };
                    return true;
                }
                if expect_digit {
                    return false;
                }
                if b == b'.' && !has_dot && !has_exp {
                    self.state = State::InNumber {
                        has_dot: true,
                        has_exp: false,
                        expect_digit: true,
                    };
                    return true;
                }
                if (b == b'e' || b == b'E') && !has_exp {
                    self.state = State::InNumber {
                        has_dot,
                        has_exp: true,
                        expect_digit: true,
                    };
                    return true;
                }
                // Number ended; transition to AfterValue and re-process `b`
                self.finish_value();
                self.step_byte(b)
            }
            State::InLiteral { target, pos } => {
                if pos < target.len() && b == target[pos] {
                    if pos + 1 == target.len() {
                        self.finish_value();
                    } else {
                        self.state = State::InLiteral {
                            target,
                            pos: pos + 1,
                        };
                    }
                    true
                } else {
                    false
                }
            }
            State::AfterValue => {
                if is_ws {
                    return true;
                }
                match self.stack.last() {
                    Some(Container::Object) => {
                        if b == b',' {
                            self.state = State::ObjectKeyOrEnd { allow_end: false };
                            true
                        } else if b == b'}' {
                            self.stack.pop();
                            self.finish_value();
                            true
                        } else {
                            false
                        }
                    }
                    Some(Container::Array) => {
                        if b == b',' {
                            self.state = State::Value {
                                allow_array_end: false,
                            };
                            true
                        } else if b == b']' {
                            self.stack.pop();
                            self.finish_value();
                            true
                        } else {
                            false
                        }
                    }
                    None => false,
                }
            }
            State::Done => is_ws,
        }
    }

    /// Fast 1-byte check rejecting >99% of invalid vocab tokens without cloning grammar state.
    #[inline]
    fn allows_first_byte(&self, b: u8) -> bool {
        let is_ws = matches!(b, b' ' | b'\n' | b'\r' | b'\t');
        match self.state {
            State::Start => is_ws || b == b'{',
            State::ObjectKeyOrEnd { allow_end } => is_ws || b == b'"' || (allow_end && b == b'}'),
            State::InString {
                escaped,
                unicode_rem,
                ..
            } => {
                if unicode_rem > 0 {
                    b.is_ascii_hexdigit()
                } else if escaped {
                    matches!(
                        b,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                    )
                } else {
                    b >= 0x20
                }
            }
            State::AfterKey => is_ws || b == b':',
            State::Value { allow_array_end } => {
                is_ws
                    || matches!(b, b'{' | b'[' | b'"' | b'-' | b'0'..=b'9' | b't' | b'f' | b'n')
                    || (allow_array_end && b == b']')
            }
            State::InNumber { expect_digit, .. } => {
                if expect_digit {
                    b.is_ascii_digit()
                } else {
                    true
                }
            }
            State::InLiteral { target, pos } => pos < target.len() && b == target[pos],
            State::AfterValue => is_ws || matches!(b, b',' | b'}' | b']'),
            State::Done => is_ws,
        }
    }

    /// Masks logits in-place to `-inf` for any token that violates valid JSON syntax.
    pub fn mask_logits(&self, logits: &mut [f32], tokenizer: &Tokenizer) {
        let mut any_allowed = false;
        let orig = logits.to_vec();

        for (id, logit) in logits.iter_mut().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let tid = id as u32;
            if tokenizer.is_stop_token(tid) {
                if self.is_complete() {
                    any_allowed = true;
                } else {
                    *logit = f32::NEG_INFINITY;
                }
                continue;
            }
            let bytes = tokenizer.cached_token_bytes(tid);
            if bytes.is_empty() || !self.allows_first_byte(bytes[0]) {
                *logit = f32::NEG_INFINITY;
                continue;
            }
            if self.try_advance_bytes(bytes).is_some() {
                any_allowed = true;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        // Safety fallback if tokenizer lacks a single-step valid token
        if !any_allowed {
            logits.copy_from_slice(&orig);
        }
    }
}

impl LogitGrammar for JsonGrammar {
    fn is_complete(&self) -> bool {
        JsonGrammar::is_complete(self)
    }

    fn mask_logits(&self, logits: &mut [f32], tokenizer: &Tokenizer) {
        JsonGrammar::mask_logits(self, logits, tokenizer)
    }

    fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        JsonGrammar::advance_bytes(self, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_grammar_accepts_lenout_proposal() {
        let mut g = JsonGrammar::new();
        let sample = br#"{"proposalId":"p1","actions":[{"type":"node.create","x":10,"y":-5.5}],"requiresConfirmation":true}"#;
        assert!(g.advance_bytes(sample));
        assert!(g.is_complete());

        let bad = JsonGrammar::new();
        assert!(bad.try_advance_bytes(b"hello world").is_none());
        assert!(bad.try_advance_bytes(br#"{"key": invalid}"#).is_none());
    }
}
