//! Zero-Dependency Iteration-Level Continuous Batching Scheduler for Caelum Server.
//!
//! Coordinates concurrent HTTP user requests across `N` independent KV-cache slots:
//! - Admits queued requests into free slots on every decode iteration (no waiting for
//!   other users to finish!).
//! - Matches incoming prompts against per-slot KV prefix caches to maximize zero-compute
//!   prefix reuse across multi-turn chats and shared system prompts.
//! - Batches all concurrently active user slots into ONE `model.decode_slots(...)` pass
//!   per iteration so model weights are read ONCE across all active users (both on Vulkan
//!   `wgpu` GPU and CPU AVX-VNNI/AVX2/NEON).
//! - Immediately reclaims slots upon EOS, `max_tokens`, stop sequence, grammar completion,
//!   or HTTP client disconnect.

use caelum_kernels::Backend;
use caelum_models::{CaelumModel, GenerationStats, Sampler, SamplerConfig};
use caelum_tokenizer::{JsonGrammar, LogitGrammar, SchemaGrammar, StreamDecoder, Tokenizer};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// Optional constrained decoding grammar attached to a scheduled request.
#[derive(Debug, Clone)]
pub enum GrammarKind {
    None,
    Json,
    LenoutSchema(String),
}

impl GrammarKind {
    fn build(&self) -> Option<Box<dyn LogitGrammar>> {
        match self {
            GrammarKind::None => None,
            GrammarKind::Json => Some(Box::new(JsonGrammar::new())),
            GrammarKind::LenoutSchema(mode) => Some(Box::new(SchemaGrammar::lenout_proposal(mode))),
        }
    }
}

/// Incremental stream event sent from the Continuous Batching Scheduler back to an HTTP worker.
#[derive(Debug)]
pub enum StreamItem {
    Token(String),
    Done {
        text: String,
        stats: GenerationStats,
    },
    Error(String),
}

/// Generation job submitted by an HTTP worker thread to the scheduler.
pub struct SchedulerRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampler_cfg: SamplerConfig,
    pub grammar: GrammarKind,
    pub stop_sequences: Vec<String>,
    pub tx: mpsc::Sender<StreamItem>,
}

/// Lock-free atomic telemetry exposed on `GET /health`.
pub struct SchedulerMetrics {
    pub slots_total: AtomicUsize,
    pub slots_active: AtomicUsize,
    pub queue_depth: AtomicUsize,
    pub total_requests: AtomicU64,
    pub total_tokens_generated: AtomicU64,
    pub total_batches: AtomicU64,
}

impl SchedulerMetrics {
    pub fn new(slots_total: usize) -> Self {
        Self {
            slots_total: AtomicUsize::new(slots_total),
            slots_active: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
        }
    }
}

struct ActiveSlot {
    sampler: Sampler,
    grammar: Option<Box<dyn LogitGrammar>>,
    stop_sequences: Vec<String>,
    history: Vec<u32>,
    decoder: StreamDecoder,
    full_text: String,
    tx: mpsc::Sender<StreamItem>,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    max_new_tokens: usize,
    generated: usize,
    prefill_secs: f64,
    decode_start: Instant,
    pending_token: u32,
    next_pos: usize,
}

/// Runs the iteration-level continuous batching loop until all senders are dropped.
pub fn run_scheduler_loop(
    model: &mut CaelumModel<'_>,
    backend: &dyn Backend,
    tokenizer: &Tokenizer,
    rx: mpsc::Receiver<SchedulerRequest>,
    metrics: Arc<SchedulerMetrics>,
) {
    let num_slots = model.num_slots().max(1);
    let max_ctx = model.max_seq_len();
    let is_recurrent = model.is_recurrent();
    metrics.slots_total.store(num_slots, Ordering::Relaxed);

    let mut slot_caches: Vec<Vec<u32>> = vec![Vec::new(); num_slots];
    let mut active: Vec<Option<ActiveSlot>> = (0..num_slots).map(|_| None).collect();
    let mut queue: VecDeque<SchedulerRequest> = VecDeque::new();

    loop {
        let active_count = active.iter().filter(|s| s.is_some()).count();

        // 1. Wait when completely idle (zero CPU burn), otherwise drain newly arrived requests.
        if active_count == 0 && queue.is_empty() {
            match rx.recv() {
                Ok(req) => queue.push_back(req),
                Err(_) => break, // Server shutting down
            }
        }
        while let Ok(req) = rx.try_recv() {
            queue.push_back(req);
        }
        metrics.queue_depth.store(queue.len(), Ordering::Relaxed);

        // 2. Admit queued requests into any free KV slots (selecting best prefix-cache match!).
        while !queue.is_empty() {
            let free_slots: Vec<usize> = (0..num_slots).filter(|&s| active[s].is_none()).collect();
            if free_slots.is_empty() {
                break;
            }
            let req = queue.pop_front().unwrap();
            metrics.queue_depth.store(queue.len(), Ordering::Relaxed);
            metrics.total_requests.fetch_add(1, Ordering::Relaxed);

            if req.prompt_tokens.is_empty() {
                let _ = req
                    .tx
                    .send(StreamItem::Error("prompt_tokens cannot be empty".into()));
                continue;
            }
            if req.prompt_tokens.len() >= max_ctx {
                let _ = req.tx.send(StreamItem::Error(format!(
                    "prompt length {} exceeds max context {max_ctx}",
                    req.prompt_tokens.len()
                )));
                continue;
            }

            // Pick the free slot with the longest reusable KV prefix match
            let mut best_slot = free_slots[0];
            let mut best_common = 0usize;
            for &s in &free_slots {
                let cache = &slot_caches[s];
                let max_reuse = cache.len().min(req.prompt_tokens.len() - 1);
                let mut c = 0usize;
                while c < max_reuse && cache[c] == req.prompt_tokens[c] {
                    c += 1;
                }
                if is_recurrent && c < cache.len() {
                    c = 0;
                }
                if c > best_common {
                    best_common = c;
                    best_slot = s;
                }
            }

            if is_recurrent && best_common < slot_caches[best_slot].len() {
                model.reset_slot(best_slot);
                slot_caches[best_slot].clear();
                best_common = 0;
            } else {
                slot_caches[best_slot].truncate(best_common);
            }

            // Prefill uncached suffix into `best_slot`
            let t_prefill = Instant::now();
            let uncached = &req.prompt_tokens[best_common..];
            let mut logits = match model.prefill_slot(backend, best_slot, uncached, best_common) {
                Ok(l) => l,
                Err(e) => {
                    let _ = req.tx.send(StreamItem::Error(format!("prefill error: {e}")));
                    continue;
                }
            };
            slot_caches[best_slot].extend_from_slice(uncached);
            let prefill_secs = t_prefill.elapsed().as_secs_f64();

            let mut state = ActiveSlot {
                sampler: Sampler::new(req.sampler_cfg),
                grammar: req.grammar.build(),
                stop_sequences: req.stop_sequences,
                history: req.prompt_tokens.clone(),
                decoder: tokenizer.stream_decoder(true),
                full_text: String::new(),
                tx: req.tx,
                prompt_tokens: req.prompt_tokens.len(),
                cached_prompt_tokens: best_common,
                max_new_tokens: req.max_new_tokens,
                generated: 0,
                prefill_secs,
                decode_start: Instant::now(),
                pending_token: 0,
                next_pos: req.prompt_tokens.len(),
            };

            if state.max_new_tokens == 0 {
                finish_slot(state, model.kv_cache_bytes(), tokenizer);
                continue;
            }

            // Sample first token immediately from prefill logits
            let keep_running =
                step_sample_and_emit(&mut state, &mut logits, tokenizer, max_ctx, &metrics);
            if keep_running {
                active[best_slot] = Some(state);
            } else {
                finish_slot(state, model.kv_cache_bytes(), tokenizer);
            }
        }

        // Drain any newly arrived requests before dispatching decode (keeps queue metric fresh)
        while let Ok(req) = rx.try_recv() {
            queue.push_back(req);
        }
        let active_now = active.iter().filter(|s| s.is_some()).count();
        metrics.slots_active.store(active_now, Ordering::Relaxed);
        metrics.queue_depth.store(queue.len(), Ordering::Relaxed);

        if active_now == 0 {
            continue;
        }

        // 3. ONE Batched Continuous Decode Step across ALL active user slots!
        let mut step_items: Vec<(usize, u32, usize)> = Vec::with_capacity(active_now);
        for (s, slot_opt) in active.iter().enumerate() {
            if let Some(st) = slot_opt {
                step_items.push((s, st.pending_token, st.next_pos));
            }
        }

        let batch_logits = match model.decode_slots(backend, &step_items) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("decode error: {e}");
                for slot_opt in active.iter_mut() {
                    if let Some(st) = slot_opt.take() {
                        let _ = st.tx.send(StreamItem::Error(msg.clone()));
                    }
                }
                metrics.slots_active.store(0, Ordering::Relaxed);
                continue;
            }
        };
        metrics.total_batches.fetch_add(1, Ordering::Relaxed);

        // 4. Sample & stream next token for every active slot; reclaim finished/aborted slots immediately!
        let kv_bytes = model.kv_cache_bytes();
        for ((s, fed_tok, fed_pos), mut logits) in step_items.into_iter().zip(batch_logits) {
            slot_caches[s].push(fed_tok);
            let mut st = active[s].take().unwrap();
            st.next_pos = fed_pos + 1;
            let keep_running =
                step_sample_and_emit(&mut st, &mut logits, tokenizer, max_ctx, &metrics);
            if keep_running {
                active[s] = Some(st);
            } else {
                finish_slot(st, kv_bytes, tokenizer);
            }
        }

        let remaining = active.iter().filter(|s| s.is_some()).count();
        metrics.slots_active.store(remaining, Ordering::Relaxed);
    }
}

/// Applies optional grammar mask, samples the next token, streams the decoded UTF-8 piece,
/// and returns `true` if this slot should continue into the next decode iteration.
fn step_sample_and_emit(
    st: &mut ActiveSlot,
    logits: &mut [f32],
    tokenizer: &Tokenizer,
    max_ctx: usize,
    metrics: &SchedulerMetrics,
) -> bool {
    if let Some(ref g) = st.grammar {
        g.mask_logits(logits, tokenizer);
    }

    let next_tok = st.sampler.sample(logits, &st.history);
    if tokenizer.is_stop_token(next_tok) {
        return false;
    }

    if let Some(ref mut g) = st.grammar {
        let raw_bytes = tokenizer.cached_token_bytes(next_tok);
        g.advance_bytes(raw_bytes);
    }

    let piece = st.decoder.push(tokenizer, next_tok);
    st.full_text.push_str(&piece);

    let mut hit_stop = false;
    if !st.stop_sequences.is_empty() {
        for stop in &st.stop_sequences {
            if !stop.is_empty() && st.full_text.ends_with(stop.as_str()) {
                let new_len = st.full_text.len() - stop.len();
                st.full_text.truncate(new_len);
                hit_stop = true;
                break;
            }
        }
    }

    st.history.push(next_tok);
    st.generated += 1;
    metrics.total_tokens_generated.fetch_add(1, Ordering::Relaxed);

    if !hit_stop && !piece.is_empty() && st.tx.send(StreamItem::Token(piece)).is_err() {
        // HTTP client disconnected — abort immediately & reclaim KV slot!
        return false;
    }

    if hit_stop {
        return false;
    }

    if let Some(ref g) = st.grammar {
        if g.is_complete() {
            return false;
        }
    }

    if st.generated >= st.max_new_tokens || st.next_pos >= max_ctx {
        return false;
    }

    st.pending_token = next_tok;
    true
}

fn finish_slot(mut st: ActiveSlot, kv_cache_bytes: usize, _tokenizer: &Tokenizer) {
    let tail = st.decoder.flush();
    if !tail.is_empty() {
        st.full_text.push_str(&tail);
        let _ = st.tx.send(StreamItem::Token(tail));
    }
    let stats = GenerationStats {
        prompt_tokens: st.prompt_tokens,
        cached_prompt_tokens: st.cached_prompt_tokens,
        generated_tokens: st.generated,
        prefill_secs: st.prefill_secs,
        decode_secs: st.decode_start.elapsed().as_secs_f64(),
        kv_cache_bytes,
    };
    let _ = st.tx.send(StreamItem::Done {
        text: st.full_text,
        stats,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use caelum_core::{quantize_row_q4_0, BlockQ4_0, DType};
    use caelum_gguf::{GgufFile, GgufValue, GgufWriter};
    use caelum_kernels::CpuBackend;
    use caelum_models::KvPrecision;

    fn make_q4_matrix(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * (cols / 32) * std::mem::size_of::<BlockQ4_0>());
        for r in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|c| (((r * 13 + c * 7 + salt * 3) % 17) as f32 - 8.0) * 0.04)
                .collect();
            let mut blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; cols / 32];
            quantize_row_q4_0(&row, &mut blocks);
            for b in blocks {
                out.extend_from_slice(&b.d.to_le_bytes());
                out.extend_from_slice(&b.qs);
            }
        }
        out
    }

    fn build_tiny_gguf() -> GgufFile {
        let h = 64usize;
        let ffn = 128usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let kv_dim = kv_heads * (h / heads);
        let mut tokens = Vec::new();
        let mut token_types = Vec::new();
        for b in 0u8..128 {
            tokens.push(GgufValue::String((b as char).to_string()));
            token_types.push(GgufValue::I32(1));
        }
        tokens.push(GgufValue::String("<|endoftext|>".into()));
        token_types.push(GgufValue::I32(3));
        let vocab = tokens.len();

        let mut w = GgufWriter::new();
        w.add_meta("general.architecture", GgufValue::String("qwen2".into()));
        w.add_meta("qwen2.embedding_length", GgufValue::U32(h as u32));
        w.add_meta("qwen2.feed_forward_length", GgufValue::U32(ffn as u32));
        w.add_meta("qwen2.block_count", GgufValue::U32(2));
        w.add_meta("qwen2.attention.head_count", GgufValue::U32(heads as u32));
        w.add_meta("qwen2.attention.head_count_kv", GgufValue::U32(kv_heads as u32));
        w.add_meta("qwen2.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6));
        w.add_meta("qwen2.context_length", GgufValue::U32(64));
        w.add_meta("tokenizer.ggml.model", GgufValue::String("gpt2".into()));
        w.add_meta("tokenizer.ggml.tokens", GgufValue::Array(tokens));
        w.add_meta("tokenizer.ggml.token_type", GgufValue::Array(token_types));
        w.add_meta("tokenizer.ggml.eos_token_id", GgufValue::U32((vocab - 1) as u32));
        w.add_tensor_raw("token_embd.weight", vec![vocab, h], DType::Q4_0, make_q4_matrix(vocab, h, 1)).unwrap();
        w.add_tensor_f32("output_norm.weight", vec![h], &vec![1.0; h]).unwrap();
        for l in 0..2 {
            w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_q.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 2)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_k.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 3)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_v.weight"), vec![kv_dim, h], DType::Q4_0, make_q4_matrix(kv_dim, h, l + 4)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.attn_output.weight"), vec![h, h], DType::Q4_0, make_q4_matrix(h, h, l + 5)).unwrap();
            w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &vec![1.0; h]).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_gate.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 6)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_up.weight"), vec![ffn, h], DType::Q4_0, make_q4_matrix(ffn, h, l + 7)).unwrap();
            w.add_tensor_raw(&format!("blk.{l}.ffn_down.weight"), vec![h, ffn], DType::Q4_0, make_q4_matrix(h, ffn, l + 8)).unwrap();
        }
        GgufFile::from_bytes(w.to_bytes()).unwrap()
    }

    #[test]
    fn test_continuous_scheduler_concurrent_multi_user_parity() {
        let gguf = build_tiny_gguf();
        let backend = CpuBackend::new(Some(2));
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();

        let prompts: Vec<Vec<u32>> = vec![
            vec![65, 66, 67],
            vec![70, 71, 72, 73, 74],
            vec![80, 81],
            vec![65, 66, 67, 68], // Shares prefix with request 0!
        ];
        let greedy = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };

        // Reference outputs from independent single-slot runs
        let mut expected = Vec::new();
        for p in &prompts {
            let mut single = CaelumModel::from_gguf_with_kv(&gguf, Some(64), KvPrecision::F32).unwrap();
            let (txt, _) = single.generate(&backend, &tokenizer, p, 5, &greedy, false, |_, _| {}).unwrap();
            expected.push(txt);
        }

        // Run all 4 requests through a 2-slot continuous batching scheduler (tests batching + queueing + slot reuse + prefix cache!)
        let mut model = CaelumModel::from_gguf_with_kv_slots(&gguf, Some(64), KvPrecision::F32, 2).unwrap();
        let metrics = Arc::new(SchedulerMetrics::new(2));
        let (tx, rx) = mpsc::channel();

        let mut receivers = Vec::new();
        for p in &prompts {
            let (rtx, rrx) = mpsc::channel();
            tx.send(SchedulerRequest {
                prompt_tokens: p.clone(),
                max_new_tokens: 5,
                sampler_cfg: greedy.clone(),
                grammar: GrammarKind::None,
                stop_sequences: Vec::new(),
                tx: rtx,
            })
            .unwrap();
            receivers.push(rrx);
        }
        drop(tx); // Close channel after enqueueing all 4 requests

        run_scheduler_loop(&mut model, &backend, &tokenizer, rx, Arc::clone(&metrics));

        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 4);
        assert!(metrics.total_batches.load(Ordering::Relaxed) >= 4);
        assert_eq!(metrics.slots_active.load(Ordering::Relaxed), 0);

        for (i, rrx) in receivers.into_iter().enumerate() {
            let mut streamed = String::new();
            let mut final_text = None;
            for item in rrx {
                match item {
                    StreamItem::Token(piece) => streamed.push_str(&piece),
                    StreamItem::Done { text, .. } => final_text = Some(text),
                    StreamItem::Error(e) => panic!("unexpected error on req {i}: {e}"),
                }
            }
            let out = final_text.expect("expected Done");
            assert_eq!(out, streamed, "req {i} streamed mismatch");
            assert_eq!(out, expected[i], "req {i} continuous batching parity mismatch");
        }
    }
}
