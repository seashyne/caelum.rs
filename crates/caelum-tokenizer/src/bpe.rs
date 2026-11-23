//! BPE (GPT-2 / Qwen2.5 / Llama 3) & SentencePiece (Llama 2 / Mistral) tokenizer loaded directly from GGUF.

use caelum_core::{CaelumError, Result};
use caelum_gguf::GgufFile;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// GPT-2 / Qwen2.5 / Llama 3 byte-level BPE with explicit merge ranks.
    ByteLevelBpe,
    /// SentencePiece BPE with token priority scores and `<0xXX>` byte fallback.
    SentencePiece,
    /// Celestia native byte-level + special token layout.
    CelestiaByte,
}

pub struct Tokenizer {
    pub kind: TokenizerKind,
    pub vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    scores: Vec<f32>,
    special_tokens: Vec<(String, u32)>,
    special_ids: HashSet<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub add_bos: bool,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    /// Precomputed decoded raw bytes per token (skip_special=true -> empty for special tokens)
    /// for zero-allocation grammar masking across 150k+ vocabularies.
    cached_normal_bytes: Vec<Vec<u8>>,
}

fn build_gpt2_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    for b in 33u32..=126 {
        bs.push(b);
    }
    for b in 161u32..=172 {
        bs.push(b);
    }
    for b in 174u32..=255 {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (b, c_code) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(c_code).unwrap();
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

impl Tokenizer {
    /// Loads a tokenizer directly from GGUF metadata (`tokenizer.ggml.*`).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .get_meta("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2");

        let vocab = gguf
            .get_meta("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_vec())
            .ok_or_else(|| CaelumError::MissingMetadata("tokenizer.ggml.tokens".into()))?;

        let scores = gguf
            .get_meta("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_vec())
            .unwrap_or_else(|| vec![0.0; vocab.len()]);

        let token_types = gguf
            .get_meta("tokenizer.ggml.token_type")
            .and_then(|v| v.as_i32_vec())
            .unwrap_or_default();

        let merges_raw = gguf
            .get_meta("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_vec())
            .unwrap_or_default();

        let kind = match model {
            "llama" | "spm" => TokenizerKind::SentencePiece,
            "celestia" => TokenizerKind::CelestiaByte,
            _ => TokenizerKind::ByteLevelBpe,
        };

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut special_tokens = Vec::new();
        let mut special_ids = HashSet::new();

        for (idx, tok) in vocab.iter().enumerate() {
            let id = idx as u32;
            token_to_id.insert(tok.clone(), id);
            // GGML token_type: 3 = CONTROL, 4 = USER_DEFINED
            let ttype = token_types.get(idx).copied().unwrap_or(1);
            if ttype == 3 || ttype == 4 {
                if !tok.is_empty() {
                    special_tokens.push((tok.clone(), id));
                    special_ids.insert(id);
                }
            }
        }
        // Match longer special tokens first
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (rank, line) in merges_raw.iter().enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                merges.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let bos_token_id = gguf
            .get_meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let eos_token_id = gguf
            .get_meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(2);
        let add_bos = gguf
            .get_meta("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(matches!(kind, TokenizerKind::SentencePiece));

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();

        let mut tok = Self {
            kind,
            vocab,
            token_to_id,
            merges,
            scores,
            special_tokens,
            special_ids,
            bos_token_id,
            eos_token_id,
            add_bos,
            byte_to_char,
            char_to_byte,
            cached_normal_bytes: Vec::new(),
        };
        tok.rebuild_byte_cache();
        Ok(tok)
    }

    /// Creates a self-contained Celestia byte-level + special token tokenizer (matching `celestia/tokenizer.py`).
    pub fn celestia_default(vocab_size: usize) -> Self {
        let mut vocab = vec![String::new(); vocab_size.max(356)];
        let specials: &[(&str, usize)] = &[
            ("<pad>", 0),
            ("<s>", 1),
            ("</s>", 2),
            ("<action_start>", 3),
            ("<action_end>", 4),
            ("<think>", 5),
            ("</think>", 6),
            ("<system>", 7),
            ("</system>", 8),
            ("<user>", 9),
            ("</user>", 10),
            ("<assistant>", 11),
            ("</assistant>", 12),
            ("<mode_draw>", 20),
            ("<mode_design>", 21),
            ("<mode_whiteboard>", 22),
            ("<mode_flow>", 23),
            ("<mode_dev>", 24),
            ("<mode_presentation>", 25),
            ("<mode_video>", 26),
            ("<mode_scene3d>", 27),
            ("<mode_workspace>", 28),
        ];

        let mut token_to_id = HashMap::new();
        let mut special_tokens = Vec::new();
        let mut special_ids = HashSet::new();

        for &(tok, id) in specials {
            vocab[id] = tok.to_string();
            token_to_id.insert(tok.to_string(), id as u32);
            special_tokens.push((tok.to_string(), id as u32));
            special_ids.insert(id as u32);
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for b in 0u8..=255 {
            let id = 100 + b as usize;
            let s = format!("<0x{b:02X}>");
            vocab[id] = s.clone();
            token_to_id.insert(s, id as u32);
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();
        let mut tok = Self {
            kind: TokenizerKind::CelestiaByte,
            vocab,
            token_to_id,
            merges: HashMap::new(),
            scores: Vec::new(),
            special_tokens,
            special_ids,
            bos_token_id: Some(1),
            eos_token_id: 2,
            add_bos: false,
            byte_to_char,
            char_to_byte,
            cached_normal_bytes: Vec::new(),
        };
        tok.rebuild_byte_cache();
        tok
    }

    fn rebuild_byte_cache(&mut self) {
        let n = self.vocab.len();
        let mut cache = Vec::with_capacity(n);
        for id in 0..n {
            cache.push(self.token_to_bytes(id as u32, true));
        }
        self.cached_normal_bytes = cache;
    }

    /// Zero-allocation slice of decoded raw bytes for a non-special token (empty for special tokens).
    #[inline]
    pub fn cached_token_bytes(&self, id: u32) -> &[u8] {
        self.cached_normal_bytes
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_stop_token(&self, id: u32) -> bool {
        if id == self.eos_token_id {
            return true;
        }
        if let Some(tok) = self.vocab.get(id as usize) {
            matches!(
                tok.as_str(),
                "<|im_end|>"
                    | "<|endoftext|>"
                    | "<|eot_id|>"
                    | "<|end_of_text|>"
                    | "<|EOT|>"
                    | "<｜end▁of▁sentence｜>"
                    | "</s>"
                    | "</assistant>"
            )
        } else {
            false
        }
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// Encodes text into token IDs (recognizing special tokens first).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                out.push(bos);
            }
        }

        let mut i = 0usize;
        let mut segment_start = 0usize;
        while i < text.len() {
            let rest = &text[i..];
            let mut matched = None;
            for (stok, sid) in &self.special_tokens {
                if rest.starts_with(stok.as_str()) {
                    matched = Some((stok.len(), *sid));
                    break;
                }
            }
            if let Some((len, sid)) = matched {
                if i > segment_start {
                    self.encode_normal(&text[segment_start..i], &mut out);
                }
                out.push(sid);
                i += len;
                segment_start = i;
            } else {
                let ch_len = rest.chars().next().map(char::len_utf8).unwrap_or(1);
                i += ch_len;
            }
        }
        if segment_start < text.len() {
            self.encode_normal(&text[segment_start..], &mut out);
        }
        out
    }

    fn encode_normal(&self, text: &str, out: &mut Vec<u32>) {
        match self.kind {
            TokenizerKind::CelestiaByte => {
                for &b in text.as_bytes() {
                    out.push(100 + b as u32);
                }
            }
            TokenizerKind::ByteLevelBpe => self.encode_gpt2_bpe(text, out),
            TokenizerKind::SentencePiece => self.encode_spm(text, out),
        }
    }

    fn encode_gpt2_bpe(&self, text: &str, out: &mut Vec<u32>) {
        // Pre-tokenize into chunks preserving whitespace attached to following words (Qwen/GPT-2 style)
        let chunks = pretokenize_gpt2(text);
        for chunk in chunks {
            let mut symbols: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();

            if !self.merges.is_empty() {
                loop {
                    if symbols.len() < 2 {
                        break;
                    }
                    let mut best_rank = usize::MAX;
                    let mut best_idx = None;
                    for j in 0..(symbols.len() - 1) {
                        if let Some(&rank) =
                            self.merges.get(&(symbols[j].clone(), symbols[j + 1].clone()))
                        {
                            if rank < best_rank {
                                best_rank = rank;
                                best_idx = Some(j);
                            }
                        }
                    }
                    let Some(idx) = best_idx else { break };
                    let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                    symbols[idx] = merged;
                    symbols.remove(idx + 1);
                }
            }

            for sym in symbols {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    out.push(id);
                } else {
                    // Fallback per character
                    for ch in sym.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.token_to_id.get(&s) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn encode_spm(&self, text: &str, out: &mut Vec<u32>) {
        let escaped = format!("▁{}", text.replace(' ', "▁"));
        let mut symbols: Vec<String> = escaped.chars().map(|c| c.to_string()).collect();

        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = None;
            for j in 0..(symbols.len() - 1) {
                let candidate = format!("{}{}", symbols[j], symbols[j + 1]);
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_idx = Some((j, candidate));
                    }
                }
            }
            let Some((idx, merged)) = best_idx else { break };
            symbols[idx] = merged;
            symbols.remove(idx + 1);
        }

        for sym in symbols {
            if let Some(&id) = self.token_to_id.get(&sym) {
                out.push(id);
            } else {
                // Byte fallback `<0xXX>`
                for &b in sym.as_bytes() {
                    let key = format!("<0x{b:02X}>");
                    if let Some(&id) = self.token_to_id.get(&key) {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// Converts a single token ID to raw UTF-8 bytes (preserving multi-byte Thai sequences).
    pub fn token_to_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        if skip_special && self.special_ids.contains(&id) {
            return Vec::new();
        }
        let Some(tok) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
        if self.special_ids.contains(&id) {
            return tok.as_bytes().to_vec();
        }
        match self.kind {
            TokenizerKind::CelestiaByte => {
                if (100..356).contains(&id) {
                    vec![(id - 100) as u8]
                } else {
                    tok.as_bytes().to_vec()
                }
            }
            TokenizerKind::ByteLevelBpe => {
                let mut bytes = Vec::with_capacity(tok.len());
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                bytes
            }
            TokenizerKind::SentencePiece => {
                if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                    if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                        return vec![b];
                    }
                }
                tok.replace('▁', " ").into_bytes()
            }
        }
    }

    /// Decodes a sequence of token IDs into a UTF-8 `String`.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_to_bytes(id, skip_special));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn stream_decoder(&self, skip_special: bool) -> StreamDecoder {
        StreamDecoder {
            pending: Vec::new(),
            skip_special,
        }
    }
}

/// Incremental UTF-8 stream decoder so multi-byte Thai/emoji tokens never split into replacement chars.
pub struct StreamDecoder {
    pending: Vec<u8>,
    skip_special: bool,
}

impl StreamDecoder {
    pub fn push(&mut self, tokenizer: &Tokenizer, token_id: u32) -> String {
        let piece = tokenizer.token_to_bytes(token_id, self.skip_special);
        self.pending.extend_from_slice(&piece);

        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to > 0 {
                        let s = unsafe {
                            std::str::from_utf8_unchecked(&self.pending[..valid_up_to])
                        };
                        out.push_str(s);
                        self.pending.drain(..valid_up_to);
                    }
                    if let Some(err_len) = e.error_len() {
                        // Invalid byte sequence (not just incomplete tail)
                        out.push('\u{FFFD}');
                        self.pending.drain(..err_len);
                    } else {
                        // Incomplete multi-byte UTF-8 sequence — wait for next token
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Exact zero-dependency implementation of Qwen2/Qwen2.5/GPT-4 (`cl100k` / `qwen2`) pre-tokenizer:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn pretokenize_gpt2(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut i = 0usize;
    let len = text.len();

    #[inline]
    fn is_letter(c: char) -> bool {
        c.is_alphabetic()
    }
    #[inline]
    fn is_number(c: char) -> bool {
        c.is_numeric()
    }
    #[inline]
    fn is_newline(c: char) -> bool {
        c == '\n' || c == '\r'
    }

    while i < len {
        let rest = &text[i..];

        // 1. English contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if rest.starts_with('\'') && rest.len() >= 2 {
            let mut matched_len = 0usize;
            for pat in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
                if rest.len() >= pat.len()
                    && rest.is_char_boundary(pat.len())
                    && rest[..pat.len()].eq_ignore_ascii_case(pat)
                {
                    matched_len = pat.len();
                    break;
                }
            }
            if matched_len > 0 {
                chunks.push(&text[i..i + matched_len]);
                i += matched_len;
                continue;
            }
        }

        let mut chars = rest.char_indices();
        let (_, c0) = chars.next().unwrap();

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if is_letter(c0) {
            let mut end = c0.len_utf8();
            for (idx, ch) in chars {
                if is_letter(ch) {
                    end = idx + ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }
        if !is_newline(c0) && !is_number(c0) {
            if let Some((idx1, c1)) = chars.clone().next() {
                if is_letter(c1) {
                    let mut end = idx1 + c1.len_utf8();
                    for (idx, ch) in chars.skip(1) {
                        if is_letter(ch) {
                            end = idx + ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    chunks.push(&text[i..i + end]);
                    i += end;
                    continue;
                }
            }
        }

        // 3. \p{N} (single digit — exact Qwen2/Qwen2.5 behavior)
        if is_number(c0) {
            let end = c0.len_utf8();
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let start_punct = if c0 == ' ' {
            match chars.clone().next() {
                Some((idx1, c1)) if !c1.is_whitespace() && !is_letter(c1) && !is_number(c1) => {
                    Some(idx1)
                }
                _ => None,
            }
        } else if !c0.is_whitespace() {
            Some(0)
        } else {
            None
        };

        if let Some(p_start) = start_punct {
            let mut end = p_start;
            let mut tail_iter = rest[p_start..].char_indices();
            while let Some((off, ch)) = tail_iter.next() {
                if !ch.is_whitespace() && !is_letter(ch) && !is_number(ch) {
                    end = p_start + off + ch.len_utf8();
                } else {
                    if is_newline(ch) {
                        end = p_start + off + ch.len_utf8();
                        for (off2, ch2) in tail_iter {
                            if is_newline(ch2) {
                                end = p_start + off2 + ch2.len_utf8();
                            } else {
                                break;
                            }
                        }
                    }
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 5. Whitespace: `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`
        let mut ws_end = c0.len_utf8();
        let mut last_nl_end = if is_newline(c0) { Some(ws_end) } else { None };
        let mut prev_ws_end = 0usize;
        let mut ws_count = 1usize;
        let mut followed_by_non_ws = false;

        for (idx, ch) in chars {
            if ch.is_whitespace() {
                prev_ws_end = ws_end;
                ws_end = idx + ch.len_utf8();
                ws_count += 1;
                if is_newline(ch) {
                    last_nl_end = Some(ws_end);
                }
            } else {
                followed_by_non_ws = true;
                break;
            }
        }

        let take = if let Some(nl_end) = last_nl_end {
            nl_end
        } else if followed_by_non_ws && ws_count > 1 {
            // `\s+(?!\S)` keeps the last single space attached to the next token
            prev_ws_end
        } else {
            ws_end
        };

        chunks.push(&text[i..i + take]);
        i += take;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celestia_thai_and_special_tokens_roundtrip() {
        let tok = Tokenizer::celestia_default(512);
        let text = "<system>Target: <mode_design></system><user>สร้างการ์ด 3 ใบ</user><assistant><think>คิด</think>";
        let ids = tok.encode(text, false);
        assert_eq!(ids[0], 7); // <system>
        assert!(ids.contains(&21)); // <mode_design>
        assert!(ids.contains(&5)); // <think>

        // Verify streaming decoder preserves Thai UTF-8 seamlessly
        let mut dec = tok.stream_decoder(false);
        let mut rebuilt = String::new();
        for &id in &ids {
            rebuilt.push_str(&dec.push(&tok, id));
        }
        rebuilt.push_str(&dec.flush());
        assert_eq!(rebuilt, text);
    }
}
