//! Zero-dependency JSON Schema Constrained Decoding Engine (`SchemaGrammar`) for Caelum.
//!
//! Enforces strict JSON Schemas (required fields, exact enums, discriminated `OneOf` unions,
//! 6-digit hex colors, identifier patterns, fixed-size 3D tuples, numbers/integers/booleans)
//! directly at token-sampling time with an `O(1)` 256-byte first-byte prefilter.

use crate::bpe::Tokenizer;
use std::sync::Arc;

/// Unified logit-masking grammar interface implemented by `JsonGrammar` and `SchemaGrammar`.
pub trait LogitGrammar {
    fn is_complete(&self) -> bool;
    fn mask_logits(&self, logits: &mut [f32], tokenizer: &Tokenizer);
    fn advance_bytes(&mut self, bytes: &[u8]) -> bool;
}

/// Composable JSON Schema AST for strict token-level constrained decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonSchema {
    /// Any valid JSON string.
    String,
    /// String restricted to exact allowed values (e.g., `["start", "process", "decision", "end"]`).
    Enum(Vec<String>),
    /// Exact constant string value (e.g., `"flow.createNode"`).
    ConstString(String),
    /// 6-digit hex color string matching `^#[0-9a-fA-F]{6}$`.
    HexColor,
    /// Identifier matching `^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$`.
    Identifier,
    /// JSON number (integer or decimal).
    Number,
    /// JSON integer.
    Integer,
    /// Unit interval float in `[0, 1]` (e.g. `0`, `1`, `0.95` for opacity/roughness/metallic/confidence).
    UnitFloat,
    /// Boolean `true` or `false`.
    Boolean,
    /// Constant boolean (`true` or `false`, e.g., `"requiresConfirmation": true`).
    ConstBool(bool),
    /// Array of items matching `item` with `[min_items, max_items]` bounds.
    Array {
        item: Box<JsonSchema>,
        min_items: usize,
        max_items: Option<usize>,
    },
    /// Fixed-length tuple (e.g., `[Number, Number, Number]` for 3D coordinates).
    Tuple(Vec<JsonSchema>),
    /// Object with ordered required fields (guarantees zero omitted required fields).
    Object(Vec<(String, JsonSchema)>),
    /// Discriminated union: succeeds if any variant schema matches.
    OneOf(Vec<JsonSchema>),
}

/// Internal execution frame for a deterministic/branching JSON Schema cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    /// Expect optional whitespace followed by exact literal bytes.
    WsLiteral { target: Vec<u8>, pos: usize },
    /// Expect exact bytes with NO leading whitespace (used inside strings/literals).
    Exact { target: Vec<u8>, pos: usize },
    /// Expect optional whitespace then a value matching `JsonSchema`.
    Value(JsonSchema),
    /// Inside a general JSON string body (after opening `"`).
    StringBody { escaped: bool, unicode_rem: u8, len: usize, max_len: usize },
    /// Inside a 6-digit hex color `#RRGGBB"` (after opening `"`).
    HexColorBody { pos: u8 },
    /// Inside an identifier string (after opening `"`).
    IdentBody { len: u8 },
    /// Inside a JSON number (after optional whitespace).
    NumberBody {
        started: bool,
        neg_allowed: bool,
        expect_digit: bool,
        leading_zero: bool,
        has_dot: bool,
        has_exp: bool,
        integer_only: bool,
    },
    /// Inside a `[0, 1]` float (`0`, `1`, or `0.<digits>`).
    UnitFloatBody {
        /// 0 = start/ws, 1 = saw '0', 2 = saw '0.', 3 = saw '0.<digit>+', 4 = saw '1'
        stage: u8,
    },
    /// Driving remaining fields of an `Object` (index `idx` is next field to emit).
    ObjectFields {
        fields: Arc<[(String, JsonSchema)]>,
        idx: usize,
    },
    /// Driving elements of an `Array` after `[` or after an element.
    ArrayTail {
        item: Arc<JsonSchema>,
        count: usize,
        min_items: usize,
        max_items: Option<usize>,
        just_opened: bool,
    },
    /// Driving remaining elements of a `Tuple`.
    TupleTail {
        items: Arc<[JsonSchema]>,
        idx: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Cursor {
    /// Stack of pending frames; top of stack (`last()`) is executed next.
    stack: Vec<Frame>,
}

impl Cursor {
    fn from_root(schema: JsonSchema) -> Self {
        Self {
            stack: vec![Frame::Value(schema)],
        }
    }

    fn is_done(&self) -> bool {
        self.stack.is_empty()
    }

    /// Expands non-consuming transitions (Value dispatch, OneOf/Enum branching, empty frames)
    /// into concrete byte-consuming cursors.
    fn expand(self, out: &mut Vec<Cursor>) {
        let mut work = vec![self];
        while let Some(mut cur) = work.pop() {
            let Some(top) = cur.stack.pop() else {
                out.push(cur);
                continue;
            };
            match top {
                Frame::Value(schema) => match schema {
                    JsonSchema::String => {
                        cur.stack.push(Frame::StringBody {
                            escaped: false,
                            unicode_rem: 0,
                            len: 0,
                            max_len: 512,
                        });
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'"'],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::ConstString(s) => {
                        let mut target = Vec::with_capacity(s.len() + 2);
                        target.push(b'"');
                        target.extend_from_slice(s.as_bytes());
                        target.push(b'"');
                        cur.stack.push(Frame::WsLiteral { target, pos: 0 });
                        work.push(cur);
                    }
                    JsonSchema::Enum(variants) => {
                        for v in variants {
                            let mut branch = cur.clone();
                            let mut target = Vec::with_capacity(v.len() + 2);
                            target.push(b'"');
                            target.extend_from_slice(v.as_bytes());
                            target.push(b'"');
                            branch.stack.push(Frame::WsLiteral { target, pos: 0 });
                            work.push(branch);
                        }
                    }
                    JsonSchema::HexColor => {
                        cur.stack.push(Frame::HexColorBody { pos: 0 });
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'"'],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::Identifier => {
                        cur.stack.push(Frame::IdentBody { len: 0 });
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'"'],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::Number => {
                        cur.stack.push(Frame::NumberBody {
                            started: false,
                            neg_allowed: true,
                            expect_digit: true,
                            leading_zero: false,
                            has_dot: false,
                            has_exp: false,
                            integer_only: false,
                        });
                        out.push(cur);
                    }
                    JsonSchema::Integer => {
                        cur.stack.push(Frame::NumberBody {
                            started: false,
                            neg_allowed: true,
                            expect_digit: true,
                            leading_zero: false,
                            has_dot: false,
                            has_exp: false,
                            integer_only: true,
                        });
                        out.push(cur);
                    }
                    JsonSchema::UnitFloat => {
                        cur.stack.push(Frame::UnitFloatBody { stage: 0 });
                        out.push(cur);
                    }
                    JsonSchema::Boolean => {
                        let mut b_true = cur.clone();
                        b_true.stack.push(Frame::WsLiteral {
                            target: b"true".to_vec(),
                            pos: 0,
                        });
                        work.push(b_true);

                        cur.stack.push(Frame::WsLiteral {
                            target: b"false".to_vec(),
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::ConstBool(val) => {
                        cur.stack.push(Frame::WsLiteral {
                            target: if val { b"true".to_vec() } else { b"false".to_vec() },
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::Object(fields) => {
                        if fields.is_empty() {
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b'}'],
                                pos: 0,
                            });
                        } else {
                            cur.stack.push(Frame::ObjectFields {
                                fields: Arc::from(fields),
                                idx: 0,
                            });
                        }
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'{'],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::Array {
                        item,
                        min_items,
                        max_items,
                    } => {
                        cur.stack.push(Frame::ArrayTail {
                            item: Arc::from(*item),
                            count: 0,
                            min_items,
                            max_items,
                            just_opened: true,
                        });
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'['],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::Tuple(items) => {
                        if items.is_empty() {
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b']'],
                                pos: 0,
                            });
                        } else {
                            cur.stack.push(Frame::TupleTail {
                                items: Arc::from(items),
                                idx: 0,
                            });
                        }
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'['],
                            pos: 0,
                        });
                        work.push(cur);
                    }
                    JsonSchema::OneOf(variants) => {
                        for v in variants {
                            let mut branch = cur.clone();
                            branch.stack.push(Frame::Value(v));
                            work.push(branch);
                        }
                    }
                },
                Frame::ObjectFields { fields, idx } => {
                    if idx >= fields.len() {
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b'}'],
                            pos: 0,
                        });
                        work.push(cur);
                    } else {
                        let (key, val_schema) = fields[idx].clone();
                        let next_idx = idx + 1;
                        if next_idx < fields.len() {
                            cur.stack.push(Frame::ObjectFields {
                                fields,
                                idx: next_idx,
                            });
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b','],
                                pos: 0,
                            });
                        } else {
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b'}'],
                                pos: 0,
                            });
                        }
                        cur.stack.push(Frame::Value(val_schema));
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b':'],
                            pos: 0,
                        });
                        let mut key_lit = Vec::with_capacity(key.len() + 2);
                        key_lit.push(b'"');
                        key_lit.extend_from_slice(key.as_bytes());
                        key_lit.push(b'"');
                        cur.stack.push(Frame::WsLiteral {
                            target: key_lit,
                            pos: 0,
                        });
                        work.push(cur);
                    }
                }
                Frame::ArrayTail {
                    item,
                    count,
                    min_items,
                    max_items,
                    just_opened,
                } => {
                    let can_close = count >= min_items;
                    let can_add = max_items.map_or(true, |m| count < m);

                    if can_close {
                        let mut close_branch = cur.clone();
                        close_branch.stack.push(Frame::WsLiteral {
                            target: vec![b']'],
                            pos: 0,
                        });
                        work.push(close_branch);
                    }
                    if can_add {
                        cur.stack.push(Frame::ArrayTail {
                            item: Arc::clone(&item),
                            count: count + 1,
                            min_items,
                            max_items,
                            just_opened: false,
                        });
                        cur.stack.push(Frame::Value((*item).clone()));
                        if !just_opened {
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b','],
                                pos: 0,
                            });
                        }
                        work.push(cur);
                    }
                }
                Frame::TupleTail { items, idx } => {
                    if idx >= items.len() {
                        cur.stack.push(Frame::WsLiteral {
                            target: vec![b']'],
                            pos: 0,
                        });
                        work.push(cur);
                    } else {
                        let item = items[idx].clone();
                        let next_idx = idx + 1;
                        if next_idx < items.len() {
                            cur.stack.push(Frame::TupleTail {
                                items,
                                idx: next_idx,
                            });
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b','],
                                pos: 0,
                            });
                        } else {
                            cur.stack.push(Frame::WsLiteral {
                                target: vec![b']'],
                                pos: 0,
                            });
                        }
                        cur.stack.push(Frame::Value(item));
                        work.push(cur);
                    }
                }
                concrete => {
                    cur.stack.push(concrete);
                    out.push(cur);
                }
            }
        }
    }

    /// Marks which single bytes can legally be accepted next by this concrete cursor.
    fn mark_allowed_bytes(&self, table: &mut [bool; 256]) {
        let mark_ws = |t: &mut [bool; 256]| {
            t[b' ' as usize] = true;
            t[b'\n' as usize] = true;
            t[b'\r' as usize] = true;
            t[b'\t' as usize] = true;
        };
        let Some(top) = self.stack.last() else {
            mark_ws(table);
            return;
        };
        match top {
            Frame::WsLiteral { target, pos } => {
                if *pos == 0 {
                    mark_ws(table);
                }
                if *pos < target.len() {
                    table[target[*pos] as usize] = true;
                }
            }
            Frame::Exact { target, pos } => {
                if *pos < target.len() {
                    table[target[*pos] as usize] = true;
                }
            }
            Frame::StringBody {
                escaped,
                unicode_rem,
                len,
                max_len,
            } => {
                if *unicode_rem > 0 {
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                    for b in b'a'..=b'f' {
                        table[b as usize] = true;
                    }
                    for b in b'A'..=b'F' {
                        table[b as usize] = true;
                    }
                } else if *escaped {
                    for &b in b"\"\\/bfnrtu" {
                        table[b as usize] = true;
                    }
                } else {
                    table[b'"' as usize] = true;
                    if *len < *max_len {
                        for b in 0x20u8..=0xFF {
                            table[b as usize] = true;
                        }
                    }
                }
            }
            Frame::HexColorBody { pos } => match *pos {
                0 => table[b'#' as usize] = true,
                1..=6 => {
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                    for b in b'a'..=b'f' {
                        table[b as usize] = true;
                    }
                    for b in b'A'..=b'F' {
                        table[b as usize] = true;
                    }
                }
                _ => table[b'"' as usize] = true,
            },
            Frame::IdentBody { len } => {
                if *len == 0 {
                    for b in b'a'..=b'z' {
                        table[b as usize] = true;
                    }
                    for b in b'A'..=b'Z' {
                        table[b as usize] = true;
                    }
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                } else {
                    table[b'"' as usize] = true;
                    if *len < 64 {
                        for b in b'a'..=b'z' {
                            table[b as usize] = true;
                        }
                        for b in b'A'..=b'Z' {
                            table[b as usize] = true;
                        }
                        for b in b'0'..=b'9' {
                            table[b as usize] = true;
                        }
                        for &b in b"_.:-" {
                            table[b as usize] = true;
                        }
                    }
                }
            }
            Frame::NumberBody {
                started,
                neg_allowed,
                expect_digit,
                leading_zero,
                has_dot,
                has_exp,
                integer_only,
            } => {
                if !*started {
                    mark_ws(table);
                    if *neg_allowed {
                        table[b'-' as usize] = true;
                    }
                }
                if !*leading_zero {
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                }
                if !*expect_digit {
                    if !*integer_only && !*has_dot && !*has_exp {
                        table[b'.' as usize] = true;
                    }
                    // Number may also terminate here and hand byte to parent frame
                    let mut ended = self.clone();
                    ended.stack.pop();
                    let mut expanded = Vec::new();
                    ended.expand(&mut expanded);
                    for c in expanded {
                        c.mark_allowed_bytes(table);
                    }
                }
            }
            Frame::UnitFloatBody { stage } => match *stage {
                0 => {
                    mark_ws(table);
                    table[b'0' as usize] = true;
                    table[b'1' as usize] = true;
                }
                1 => {
                    table[b'.' as usize] = true;
                    let mut ended = self.clone();
                    ended.stack.pop();
                    let mut expanded = Vec::new();
                    ended.expand(&mut expanded);
                    for c in expanded {
                        c.mark_allowed_bytes(table);
                    }
                }
                2 => {
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                }
                3 => {
                    for b in b'0'..=b'9' {
                        table[b as usize] = true;
                    }
                    let mut ended = self.clone();
                    ended.stack.pop();
                    let mut expanded = Vec::new();
                    ended.expand(&mut expanded);
                    for c in expanded {
                        c.mark_allowed_bytes(table);
                    }
                }
                _ => {
                    let mut ended = self.clone();
                    ended.stack.pop();
                    let mut expanded = Vec::new();
                    ended.expand(&mut expanded);
                    for c in expanded {
                        c.mark_allowed_bytes(table);
                    }
                }
            },
            _ => {}
        }
    }

    /// Consumes a single byte `b`, pushing zero or more surviving expanded cursors into `out`.
    fn step_byte(mut self, b: u8, out: &mut Vec<Cursor>) {
        let is_ws = matches!(b, b' ' | b'\n' | b'\r' | b'\t');
        let Some(top) = self.stack.pop() else {
            if is_ws {
                out.push(self);
            }
            return;
        };

        match top {
            Frame::WsLiteral { target, pos } => {
                if pos == 0 && is_ws {
                    self.stack.push(Frame::WsLiteral { target, pos: 0 });
                    out.push(self);
                } else if pos < target.len() && b == target[pos] {
                    if pos + 1 < target.len() {
                        self.stack.push(Frame::Exact {
                            target,
                            pos: pos + 1,
                        });
                        out.push(self);
                    } else {
                        self.expand(out);
                    }
                }
            }
            Frame::Exact { target, pos } => {
                if pos < target.len() && b == target[pos] {
                    if pos + 1 < target.len() {
                        self.stack.push(Frame::Exact {
                            target,
                            pos: pos + 1,
                        });
                        out.push(self);
                    } else {
                        self.expand(out);
                    }
                }
            }
            Frame::StringBody {
                escaped,
                unicode_rem,
                len,
                max_len,
            } => {
                if unicode_rem > 0 {
                    if b.is_ascii_hexdigit() {
                        self.stack.push(Frame::StringBody {
                            escaped: false,
                            unicode_rem: unicode_rem - 1,
                            len: len + 1,
                            max_len,
                        });
                        out.push(self);
                    }
                } else if escaped {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.stack.push(Frame::StringBody {
                                escaped: false,
                                unicode_rem: 0,
                                len: len + 1,
                                max_len,
                            });
                            out.push(self);
                        }
                        b'u' => {
                            self.stack.push(Frame::StringBody {
                                escaped: false,
                                unicode_rem: 4,
                                len: len + 1,
                                max_len,
                            });
                            out.push(self);
                        }
                        _ => {}
                    }
                } else if b == b'"' {
                    self.expand(out);
                } else if b >= 0x20 && len < max_len {
                    if b == b'\\' {
                        self.stack.push(Frame::StringBody {
                            escaped: true,
                            unicode_rem: 0,
                            len: len + 1,
                            max_len,
                        });
                    } else {
                        self.stack.push(Frame::StringBody {
                            escaped: false,
                            unicode_rem: 0,
                            len: len + 1,
                            max_len,
                        });
                    }
                    out.push(self);
                }
            }
            Frame::HexColorBody { pos } => match pos {
                0 if b == b'#' => {
                    self.stack.push(Frame::HexColorBody { pos: 1 });
                    out.push(self);
                }
                1..=6 if b.is_ascii_hexdigit() => {
                    self.stack.push(Frame::HexColorBody { pos: pos + 1 });
                    out.push(self);
                }
                7 if b == b'"' => {
                    self.expand(out);
                }
                _ => {}
            },
            Frame::IdentBody { len } => {
                if len == 0 {
                    if b.is_ascii_alphanumeric() {
                        self.stack.push(Frame::IdentBody { len: 1 });
                        out.push(self);
                    }
                } else if b == b'"' {
                    self.expand(out);
                } else if len < 64
                    && (b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
                {
                    self.stack.push(Frame::IdentBody { len: len + 1 });
                    out.push(self);
                }
            }
            Frame::NumberBody {
                started,
                neg_allowed,
                expect_digit,
                leading_zero,
                has_dot,
                has_exp,
                integer_only,
            } => {
                if !started && is_ws {
                    self.stack.push(Frame::NumberBody {
                        started: false,
                        neg_allowed,
                        expect_digit,
                        leading_zero: false,
                        has_dot,
                        has_exp,
                        integer_only,
                    });
                    out.push(self);
                } else if !started && neg_allowed && b == b'-' {
                    self.stack.push(Frame::NumberBody {
                        started: true,
                        neg_allowed: false,
                        expect_digit: true,
                        leading_zero: false,
                        has_dot,
                        has_exp,
                        integer_only,
                    });
                    out.push(self);
                } else if !leading_zero && b.is_ascii_digit() {
                    let is_lead_zero = expect_digit && !has_dot && b == b'0';
                    self.stack.push(Frame::NumberBody {
                        started: true,
                        neg_allowed: false,
                        expect_digit: false,
                        leading_zero: is_lead_zero,
                        has_dot,
                        has_exp,
                        integer_only,
                    });
                    out.push(self);
                } else if !expect_digit && !integer_only && !has_dot && !has_exp && b == b'.' {
                    self.stack.push(Frame::NumberBody {
                        started: true,
                        neg_allowed: false,
                        expect_digit: true,
                        leading_zero: false,
                        has_dot: true,
                        has_exp,
                        integer_only,
                    });
                    out.push(self);
                } else if !expect_digit {
                    // Number finished — delegate `b` to parent frame!
                    let mut next_expanded = Vec::new();
                    self.expand(&mut next_expanded);
                    for c in next_expanded {
                        c.step_byte(b, out);
                    }
                }
            }
            Frame::UnitFloatBody { stage } => match stage {
                0 if is_ws => {
                    self.stack.push(Frame::UnitFloatBody { stage: 0 });
                    out.push(self);
                }
                0 if b == b'0' => {
                    self.stack.push(Frame::UnitFloatBody { stage: 1 });
                    out.push(self);
                }
                0 if b == b'1' => {
                    self.stack.push(Frame::UnitFloatBody { stage: 4 });
                    out.push(self);
                }
                1 if b == b'.' => {
                    self.stack.push(Frame::UnitFloatBody { stage: 2 });
                    out.push(self);
                }
                2 | 3 if b.is_ascii_digit() => {
                    self.stack.push(Frame::UnitFloatBody { stage: 3 });
                    out.push(self);
                }
                1 | 3 | 4 => {
                    let mut next_expanded = Vec::new();
                    self.expand(&mut next_expanded);
                    for c in next_expanded {
                        c.step_byte(b, out);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

/// Active JSON Schema Grammar Engine for strict token-level constrained decoding.
#[derive(Debug, Clone)]
pub struct SchemaGrammar {
    cursors: Vec<Cursor>,
}

impl SchemaGrammar {
    pub fn new(schema: JsonSchema) -> Self {
        let mut cursors = Vec::new();
        Cursor::from_root(schema).expand(&mut cursors);
        dedup_cursors(&mut cursors);
        Self { cursors }
    }

    /// Builds a strict Lenout `CelestiaProposal` JSON Schema matching `@alenout/celestia` Zod schemas!
    pub fn lenout_proposal(mode: &str) -> Self {
        let action_schema = match mode {
            "design" | "whiteboard" | "workspace" => JsonSchema::OneOf(vec![
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("node.create".into())),
                    ("ref".into(), JsonSchema::Identifier),
                    (
                        "node".into(),
                        JsonSchema::Object(vec![
                            (
                                "kind".into(),
                                JsonSchema::Enum(vec![
                                    "card".into(),
                                    "shape".into(),
                                    "text".into(),
                                    "note".into(),
                                    "frame".into(),
                                    "container".into(),
                                ]),
                            ),
                            ("name".into(), JsonSchema::String),
                            ("text".into(), JsonSchema::String),
                            ("offsetX".into(), JsonSchema::Number),
                            ("offsetY".into(), JsonSchema::Number),
                            ("width".into(), JsonSchema::Number),
                            ("height".into(), JsonSchema::Number),
                            ("fill".into(), JsonSchema::HexColor),
                            ("radius".into(), JsonSchema::Number),
                        ]),
                    ),
                ]),
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("layout.apply".into())),
                    (
                        "nodeIds".into(),
                        JsonSchema::Array {
                            item: Box::new(JsonSchema::Identifier),
                            min_items: 2,
                            max_items: Some(32),
                        },
                    ),
                    (
                        "direction".into(),
                        JsonSchema::Enum(vec![
                            "horizontal".into(),
                            "vertical".into(),
                            "grid".into(),
                        ]),
                    ),
                    ("gap".into(), JsonSchema::Number),
                    (
                        "align".into(),
                        JsonSchema::Enum(vec![
                            "start".into(),
                            "center".into(),
                            "end".into(),
                            "stretch".into(),
                        ]),
                    ),
                ]),
            ]),
            "flow" => JsonSchema::OneOf(vec![
                JsonSchema::Object(vec![
                    (
                        "type".into(),
                        JsonSchema::ConstString("flow.createNode".into()),
                    ),
                    ("ref".into(), JsonSchema::Identifier),
                    (
                        "nodeType".into(),
                        JsonSchema::Enum(vec![
                            "start".into(),
                            "process".into(),
                            "decision".into(),
                            "end".into(),
                            "custom".into(),
                        ]),
                    ),
                    ("label".into(), JsonSchema::String),
                    ("offsetX".into(), JsonSchema::Number),
                    ("offsetY".into(), JsonSchema::Number),
                ]),
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("flow.connect".into())),
                    ("sourceId".into(), JsonSchema::Identifier),
                    ("targetId".into(), JsonSchema::Identifier),
                    (
                        "style".into(),
                        JsonSchema::Enum(vec![
                            "orthogonal".into(),
                            "straight".into(),
                            "curved".into(),
                        ]),
                    ),
                    (
                        "arrow".into(),
                        JsonSchema::Enum(vec!["end".into(), "both".into(), "none".into()]),
                    ),
                ]),
            ]),
            "draw" => JsonSchema::OneOf(vec![
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("palette.apply".into())),
                    ("name".into(), JsonSchema::String),
                    (
                        "colors".into(),
                        JsonSchema::Array {
                            item: Box::new(JsonSchema::HexColor),
                            min_items: 1,
                            max_items: Some(12),
                        },
                    ),
                ]),
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("brush.select".into())),
                    ("brushId".into(), JsonSchema::Identifier),
                    ("size".into(), JsonSchema::Number),
                    ("opacity".into(), JsonSchema::UnitFloat),
                    ("color".into(), JsonSchema::HexColor),
                ]),
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("layer.create".into())),
                    ("ref".into(), JsonSchema::Identifier),
                    ("name".into(), JsonSchema::String),
                    ("opacity".into(), JsonSchema::UnitFloat),
                    (
                        "blendMode".into(),
                        JsonSchema::Enum(vec![
                            "normal".into(),
                            "multiply".into(),
                            "screen".into(),
                            "overlay".into(),
                        ]),
                    ),
                ]),
            ]),
            "scene3d" => JsonSchema::Object(vec![
                ("type".into(), JsonSchema::ConstString("scene3d.spawn".into())),
                ("ref".into(), JsonSchema::Identifier),
                (
                    "primitive".into(),
                    JsonSchema::Enum(vec![
                        "box".into(),
                        "sphere".into(),
                        "cylinder".into(),
                        "plane".into(),
                        "torus".into(),
                        "cone".into(),
                    ]),
                ),
                ("name".into(), JsonSchema::String),
                (
                    "position".into(),
                    JsonSchema::Tuple(vec![
                        JsonSchema::Number,
                        JsonSchema::Number,
                        JsonSchema::Number,
                    ]),
                ),
                (
                    "rotation".into(),
                    JsonSchema::Tuple(vec![
                        JsonSchema::Number,
                        JsonSchema::Number,
                        JsonSchema::Number,
                    ]),
                ),
                (
                    "scale".into(),
                    JsonSchema::Tuple(vec![
                        JsonSchema::Number,
                        JsonSchema::Number,
                        JsonSchema::Number,
                    ]),
                ),
                ("color".into(), JsonSchema::HexColor),
                ("roughness".into(), JsonSchema::UnitFloat),
                ("metallic".into(), JsonSchema::UnitFloat),
            ]),
            "presentation" => JsonSchema::Object(vec![
                (
                    "type".into(),
                    JsonSchema::ConstString("presentation.createSlide".into()),
                ),
                ("ref".into(), JsonSchema::Identifier),
                ("title".into(), JsonSchema::String),
                (
                    "layout".into(),
                    JsonSchema::Enum(vec![
                        "title".into(),
                        "title-content".into(),
                        "two-column".into(),
                        "blank".into(),
                        "quote".into(),
                    ]),
                ),
                ("background".into(), JsonSchema::HexColor),
            ]),
            "video" => JsonSchema::OneOf(vec![
                JsonSchema::Object(vec![
                    ("type".into(), JsonSchema::ConstString("video.addTrack".into())),
                    ("ref".into(), JsonSchema::Identifier),
                    (
                        "kind".into(),
                        JsonSchema::Enum(vec![
                            "video".into(),
                            "audio".into(),
                            "text".into(),
                            "overlay".into(),
                        ]),
                    ),
                    ("name".into(), JsonSchema::String),
                ]),
                JsonSchema::Object(vec![
                    (
                        "type".into(),
                        JsonSchema::ConstString("video.setKeyframe".into()),
                    ),
                    ("trackId".into(), JsonSchema::Identifier),
                    ("clipId".into(), JsonSchema::Identifier),
                    ("timeMs".into(), JsonSchema::Number),
                    (
                        "property".into(),
                        JsonSchema::Enum(vec![
                            "x".into(),
                            "y".into(),
                            "scale".into(),
                            "opacity".into(),
                            "rotation".into(),
                            "volume".into(),
                        ]),
                    ),
                    ("value".into(), JsonSchema::Number),
                    (
                        "easing".into(),
                        JsonSchema::Enum(vec![
                            "linear".into(),
                            "easeIn".into(),
                            "easeOut".into(),
                            "easeInOut".into(),
                        ]),
                    ),
                ]),
            ]),
            _ => JsonSchema::Object(vec![
                ("type".into(), JsonSchema::ConstString("node.create".into())),
                ("ref".into(), JsonSchema::Identifier),
                (
                    "node".into(),
                    JsonSchema::Object(vec![
                        ("kind".into(), JsonSchema::ConstString("card".into())),
                        ("name".into(), JsonSchema::String),
                        ("offsetX".into(), JsonSchema::Number),
                        ("offsetY".into(), JsonSchema::Number),
                        ("width".into(), JsonSchema::Number),
                        ("height".into(), JsonSchema::Number),
                        ("fill".into(), JsonSchema::HexColor),
                    ]),
                ),
            ]),
        };

        let target_mode = match mode {
            "draw" | "design" | "whiteboard" | "flow" | "dev" | "presentation" | "video"
            | "scene3d" | "workspace" => mode,
            _ => "workspace",
        };

        let root = JsonSchema::Object(vec![
            ("proposalId".into(), JsonSchema::Identifier),
            ("requestId".into(), JsonSchema::Identifier),
            ("contextFingerprint".into(), JsonSchema::Identifier),
            (
                "targetMode".into(),
                JsonSchema::ConstString(target_mode.into()),
            ),
            ("summary".into(), JsonSchema::String),
            (
                "warnings".into(),
                JsonSchema::Array {
                    item: Box::new(JsonSchema::String),
                    min_items: 0,
                    max_items: Some(4),
                },
            ),
            (
                "actions".into(),
                JsonSchema::Array {
                    item: Box::new(action_schema),
                    min_items: 1,
                    max_items: Some(16),
                },
            ),
            ("requiresConfirmation".into(), JsonSchema::ConstBool(true)),
            ("confidence".into(), JsonSchema::UnitFloat),
        ]);

        Self::new(root)
    }

    fn try_advance_bytes(&self, bytes: &[u8]) -> Option<Vec<Cursor>> {
        let mut active = self.cursors.clone();
        let mut next = Vec::new();
        for &b in bytes {
            next.clear();
            for cur in active.drain(..) {
                cur.step_byte(b, &mut next);
            }
            if next.is_empty() {
                return None;
            }
            dedup_cursors(&mut next);
            std::mem::swap(&mut active, &mut next);
        }
        Some(active)
    }

    /// Computes two `O(1)` lookup tables:
    /// 1. `allowed_first`: valid first byte of a token.
    /// 2. `allowed_after_ws`: valid first *non-whitespace* byte when a token begins with whitespace
    ///    (rejects ~60,000 space-prefixed BPE words in O(1) at every JSON punctuation/key boundary!).
    fn allowed_byte_tables(&self) -> ([bool; 256], [bool; 256]) {
        let mut first = [false; 256];
        let mut after_ws = [false; 256];
        for c in &self.cursors {
            c.mark_allowed_bytes(&mut first);
            match c.stack.last() {
                Some(Frame::WsLiteral { target, pos: 0 }) => {
                    if let Some(&b) = target.first() {
                        after_ws[b as usize] = true;
                    }
                }
                Some(Frame::NumberBody { started: false, .. }) => {
                    after_ws[b'-' as usize] = true;
                    for b in b'0'..=b'9' {
                        after_ws[b as usize] = true;
                    }
                }
                Some(Frame::UnitFloatBody { stage: 0 }) => {
                    after_ws[b'0' as usize] = true;
                    after_ws[b'1' as usize] = true;
                }
                Some(Frame::StringBody { .. })
                | Some(Frame::NumberBody {
                    expect_digit: false,
                    ..
                })
                | Some(Frame::UnitFloatBody { stage: 1 | 3 | 4 }) => {
                    // Whitespace may be string content or terminate a number into parent punctuation
                    after_ws.fill(true);
                }
                None => {} // Done: only whitespace allowed, no non-ws byte allowed!
                _ => {
                    // Any other frame does not accept leading ws anyway
                }
            }
        }
        (first, after_ws)
    }

    /// Zero-allocation fast-accept check for common single-cursor string/ident/literal interior tokens.
    #[inline]
    fn fast_accept_bytes(&self, bytes: &[u8]) -> bool {
        if self.cursors.len() != 1 {
            return false;
        }
        let Some(top) = self.cursors[0].stack.last() else {
            return bytes.iter().all(|&b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'));
        };
        match top {
            Frame::StringBody {
                escaped: false,
                unicode_rem: 0,
                len,
                max_len,
            } => {
                *len + bytes.len() <= *max_len
                    && bytes
                        .iter()
                        .all(|&b| b >= 0x20 && b != b'"' && b != b'\\')
            }
            Frame::IdentBody { len } if *len > 0 => {
                (*len as usize) + bytes.len() <= 64
                    && bytes
                        .iter()
                        .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
            }
            Frame::Exact { target, pos } => {
                let rem = &target[*pos..];
                bytes.len() < rem.len() && rem.starts_with(bytes)
            }
            _ => false,
        }
    }
}

impl LogitGrammar for SchemaGrammar {
    fn is_complete(&self) -> bool {
        self.cursors.iter().any(Cursor::is_done)
    }

    fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        if let Some(next) = self.try_advance_bytes(bytes) {
            self.cursors = next;
            true
        } else {
            false
        }
    }

    fn mask_logits(&self, logits: &mut [f32], tokenizer: &Tokenizer) {
        let (allowed_first, allowed_after_ws) = self.allowed_byte_tables();
        let is_done = self.is_complete();
        let mut any_allowed = false;
        let orig = logits.to_vec();

        for (id, logit) in logits.iter_mut().enumerate() {
            if !logit.is_finite() {
                continue;
            }
            let tid = id as u32;
            if tokenizer.is_stop_token(tid) {
                if is_done {
                    any_allowed = true;
                } else {
                    *logit = f32::NEG_INFINITY;
                }
                continue;
            }
            let bytes = tokenizer.cached_token_bytes(tid);
            if bytes.is_empty() || !allowed_first[bytes[0] as usize] {
                *logit = f32::NEG_INFINITY;
                continue;
            }
            // Reject space-prefixed BPE words (e.g. " hello") in O(1) when after-ws byte cannot match
            if matches!(bytes[0], b' ' | b'\n' | b'\r' | b'\t') {
                if let Some(&non_ws) = bytes
                    .iter()
                    .find(|&&b| !matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
                {
                    if !allowed_after_ws[non_ws as usize] {
                        *logit = f32::NEG_INFINITY;
                        continue;
                    }
                }
            }
            if self.fast_accept_bytes(bytes) || self.try_advance_bytes(bytes).is_some() {
                any_allowed = true;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        if !any_allowed {
            logits.copy_from_slice(&orig);
        }
    }
}

fn dedup_cursors(cursors: &mut Vec<Cursor>) {
    let mut unique = Vec::with_capacity(cursors.len());
    for c in cursors.drain(..) {
        if !unique.contains(&c) {
            unique.push(c);
        }
    }
    *cursors = unique;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenout_flow_schema_enforces_required_coords_and_enums() {
        let mut g = SchemaGrammar::lenout_proposal("flow");
        let valid = br#"{"proposalId":"prop_1","requestId":"req_1","contextFingerprint":"fp_1","targetMode":"flow","summary":"Login flow","warnings":[],"actions":[{"type":"flow.createNode","ref":"n1","nodeType":"start","label":"Start","offsetX":100,"offsetY":100},{"type":"flow.connect","sourceId":"n1","targetId":"n2","style":"orthogonal","arrow":"end"}],"requiresConfirmation":true,"confidence":0.95}"#;
        assert!(g.advance_bytes(valid));
        assert!(g.is_complete());

        // Missing offsetX/offsetY must be REJECTED
        let missing_coords = SchemaGrammar::lenout_proposal("flow");
        let bad = br#"{"proposalId":"prop_1","requestId":"req_1","contextFingerprint":"fp_1","targetMode":"flow","summary":"x","warnings":[],"actions":[{"type":"flow.createNode","ref":"n1","nodeType":"start","label":"Start"}]}"#;
        assert!(missing_coords.try_advance_bytes(bad).is_none());

        // Invalid enum arrow="start" must be REJECTED
        let bad_enum = SchemaGrammar::lenout_proposal("flow");
        let bad2 = br#"{"proposalId":"prop_1","requestId":"req_1","contextFingerprint":"fp_1","targetMode":"flow","summary":"x","warnings":[],"actions":[{"type":"flow.connect","sourceId":"n1","targetId":"n2","style":"orthogonal","arrow":"start"}]}"#;
        assert!(bad_enum.try_advance_bytes(bad2).is_none());
    }

    #[test]
    fn test_lenout_scene3d_and_draw_schemas() {
        let mut g3d = SchemaGrammar::lenout_proposal("scene3d");
        let s3d = br##"{"proposalId":"p1","requestId":"r1","contextFingerprint":"f1","targetMode":"scene3d","summary":"3D box","warnings":[],"actions":[{"type":"scene3d.spawn","ref":"b1","primitive":"box","name":"Cube","position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1],"color":"#4f46e5","roughness":0.3,"metallic":0.5}],"requiresConfirmation":true,"confidence":0.9}"##;
        assert!(g3d.advance_bytes(s3d));
        assert!(g3d.is_complete());

        // Invalid 3-char/bad hex color must be REJECTED
        let gdraw = SchemaGrammar::lenout_proposal("draw");
        let bad_hex = br#"{"proposalId":"p1","requestId":"r1","contextFingerprint":"f1","targetMode":"draw","summary":"x","warnings":[],"actions":[{"type":"palette.apply","name":"Neon","colors":["not-a-color"]}]}"#;
        assert!(gdraw.try_advance_bytes(bad_hex).is_none());
    }
}
