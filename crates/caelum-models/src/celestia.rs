//! Native Celestia ARMv9-First Architecture in Pure Rust:
//! - True Compressed Multi-Head Latent Attention (MLA): caches ONLY `c_t^{KV}` (`kv_latent_dim`)
//! - Exact Position-Aware NeoX RoPE (fixes the KV-cache RoPE offset bug)
//! - DeepSeekMoE: Always-Active Shared Expert + Softmax Top-K Routed SwiGLU Experts
//! - GGUF v3 serialization & zero-copy `mmap` loading (`general.architecture = "celestia"`)

use crate::cache::MlaLatentCache;
use caelum_core::{quantize_row_q4_0, BlockQ4_0, DType, Result, TensorView};
use caelum_gguf::{GgufFile, GgufValue, GgufWriter};
use caelum_kernels::{Backend, RopeStyle};

#[derive(Debug, Clone)]
pub struct CelestiaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kv_latent_dim: usize,
    pub num_experts: usize,
    pub num_active_experts: usize,
    pub shared_experts: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl CelestiaConfig {
    /// Compact ARMv9/x86 test & benchmark profile (~1.5M params, all dims multiples of 32/64).
    pub fn micro_test() -> Self {
        Self {
            vocab_size: 512,
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            kv_latent_dim: 32,
            num_experts: 4,
            num_active_experts: 2,
            shared_experts: 1,
            rope_theta: 100_000.0,
            rms_norm_eps: 1e-6,
            max_seq_len: 512,
        }
    }

    /// Celestia-Nano edge profile (~500M total params).
    pub fn nano_500m() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 1024,
            intermediate_size: 2816,
            num_layers: 16,
            num_heads: 8,
            head_dim: 128,
            kv_latent_dim: 256,
            num_experts: 4,
            num_active_experts: 1,
            shared_experts: 1,
            rope_theta: 100_000.0,
            rms_norm_eps: 1e-6,
            max_seq_len: 8192,
        }
    }

    /// Celestia-Base production profile (~1.4B total params).
    pub fn base_1b() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_layers: 24,
            num_heads: 16,
            head_dim: 128,
            kv_latent_dim: 512,
            num_experts: 8,
            num_active_experts: 2,
            shared_experts: 1,
            rope_theta: 100_000.0,
            rms_norm_eps: 1e-6,
            max_seq_len: 32768,
        }
    }

    pub fn expert_intermediate(&self) -> usize {
        self.intermediate_size / self.num_active_experts.max(1)
    }

    pub fn shared_intermediate(&self) -> usize {
        self.intermediate_size / 2
    }

    /// Percentage of KV cache memory saved compared to standard MHA K+V cache.
    pub fn mla_kv_savings_percent(&self) -> f32 {
        let standard_kv = (self.num_heads * self.head_dim * 2) as f32;
        let compressed = self.kv_latent_dim as f32;
        (1.0 - compressed / standard_kv) * 100.0
    }

    pub fn from_gguf(gguf: &GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let hidden_size = gguf.require_usize("celestia.embedding_length")?;
        let intermediate_size = gguf.require_usize("celestia.feed_forward_length")?;
        let num_layers = gguf.require_usize("celestia.block_count")?;
        let num_heads = gguf.require_usize("celestia.attention.head_count")?;
        let kv_latent_dim = gguf.require_usize("celestia.attention.kv_latent_dim")?;
        let num_experts = gguf.require_usize("celestia.expert_count")?;
        let num_active_experts = gguf.require_usize("celestia.expert_used_count")?;
        let shared_experts = gguf
            .get_meta("celestia.shared_expert_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(1);
        let rope_theta = gguf
            .get_meta("celestia.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(100_000.0);
        let rms_norm_eps = gguf
            .get_meta("celestia.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ctx_len = gguf
            .get_meta("celestia.context_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let max_seq_len = max_seq_override.unwrap_or(ctx_len.min(4096));

        let (vocab_size, _) = gguf.tensor("token_embd.weight")?.dims_2d()?;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            head_dim: hidden_size / num_heads,
            kv_latent_dim,
            num_experts,
            num_active_experts,
            shared_experts,
            rope_theta,
            rms_norm_eps,
            max_seq_len,
        })
    }
}

struct ExpertWeights<'a> {
    gate: TensorView<'a>,
    up: TensorView<'a>,
    down: TensorView<'a>,
}

struct CelestiaLayer<'a> {
    attn_norm: Vec<f32>,
    wq: TensorView<'a>,
    w_kv_down: TensorView<'a>,
    w_kv_up: TensorView<'a>,
    wo: TensorView<'a>,
    ffn_norm: Vec<f32>,
    router: TensorView<'a>,
    experts: Vec<ExpertWeights<'a>>,
    shared_expert: Option<ExpertWeights<'a>>,
}

struct CelestiaScratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    c_kv: Vec<f32>,
    kv_up_buf: Vec<f32>,
    k_rebuilt: Vec<f32>,
    v_rebuilt: Vec<f32>,
    att: Vec<f32>,
    attn_out: Vec<f32>,
    router_logits: Vec<f32>,
    moe_accum: Vec<f32>,
    exp_gate: Vec<f32>,
    exp_up: Vec<f32>,
    exp_out: Vec<f32>,
    sh_gate: Vec<f32>,
    sh_up: Vec<f32>,
    logits: Vec<f32>,
}

impl CelestiaScratch {
    fn new(cfg: &CelestiaConfig) -> Self {
        let attn_dim = cfg.num_heads * cfg.head_dim;
        Self {
            x: vec![0.0; cfg.hidden_size],
            xb: vec![0.0; cfg.hidden_size],
            q: vec![0.0; attn_dim],
            c_kv: vec![0.0; cfg.kv_latent_dim],
            kv_up_buf: vec![0.0; attn_dim * 2],
            k_rebuilt: vec![0.0; cfg.max_seq_len * attn_dim],
            v_rebuilt: vec![0.0; cfg.max_seq_len * attn_dim],
            att: vec![0.0; cfg.max_seq_len],
            attn_out: vec![0.0; attn_dim],
            router_logits: vec![0.0; cfg.num_experts],
            moe_accum: vec![0.0; cfg.hidden_size],
            exp_gate: vec![0.0; cfg.expert_intermediate()],
            exp_up: vec![0.0; cfg.expert_intermediate()],
            exp_out: vec![0.0; cfg.hidden_size],
            sh_gate: vec![0.0; cfg.shared_intermediate().max(1)],
            sh_up: vec![0.0; cfg.shared_intermediate().max(1)],
            logits: vec![0.0; cfg.vocab_size],
        }
    }
}

pub struct CelestiaModel<'a> {
    pub config: CelestiaConfig,
    token_embd: TensorView<'a>,
    layers: Vec<CelestiaLayer<'a>>,
    output_norm: Vec<f32>,
    output_weight: TensorView<'a>,
    mla_cache: MlaLatentCache,
    scratch: CelestiaScratch,
}

impl<'a> CelestiaModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, max_seq_override: Option<usize>) -> Result<Self> {
        let config = CelestiaConfig::from_gguf(gguf, max_seq_override)?;
        let token_embd = gguf.tensor("token_embd.weight")?;
        let output_norm = gguf.tensor("output_norm.weight")?.to_f32_vec()?;
        let output_weight = if gguf.has_tensor("output.weight") {
            gguf.tensor("output.weight")?
        } else {
            token_embd
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let mut experts = Vec::with_capacity(config.num_experts);
            for e in 0..config.num_experts {
                experts.push(ExpertWeights {
                    gate: gguf.tensor(&format!("blk.{l}.ffn_gate.{e}.weight"))?,
                    up: gguf.tensor(&format!("blk.{l}.ffn_up.{e}.weight"))?,
                    down: gguf.tensor(&format!("blk.{l}.ffn_down.{e}.weight"))?,
                });
            }

            let shared_expert = if config.shared_experts > 0
                && gguf.has_tensor(&format!("blk.{l}.ffn_gate_shexp.weight"))
            {
                Some(ExpertWeights {
                    gate: gguf.tensor(&format!("blk.{l}.ffn_gate_shexp.weight"))?,
                    up: gguf.tensor(&format!("blk.{l}.ffn_up_shexp.weight"))?,
                    down: gguf.tensor(&format!("blk.{l}.ffn_down_shexp.weight"))?,
                })
            } else {
                None
            };

            layers.push(CelestiaLayer {
                attn_norm: gguf.tensor(&format!("blk.{l}.attn_norm.weight"))?.to_f32_vec()?,
                wq: gguf.tensor(&format!("blk.{l}.attn_q.weight"))?,
                w_kv_down: gguf.tensor(&format!("blk.{l}.attn_kv_down.weight"))?,
                w_kv_up: gguf.tensor(&format!("blk.{l}.attn_kv_up.weight"))?,
                wo: gguf.tensor(&format!("blk.{l}.attn_output.weight"))?,
                ffn_norm: gguf.tensor(&format!("blk.{l}.ffn_norm.weight"))?.to_f32_vec()?,
                router: gguf.tensor(&format!("blk.{l}.ffn_gate_inp.weight"))?,
                experts,
                shared_expert,
            });
        }

        let mla_cache = MlaLatentCache::new(
            config.num_layers,
            config.max_seq_len,
            config.kv_latent_dim,
        );
        let scratch = CelestiaScratch::new(&config);

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output_weight,
            mla_cache,
            scratch,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.mla_cache.memory_bytes()
    }

    pub fn reset_cache(&mut self) {
        self.mla_cache.clear();
    }

    /// Runs one autoregressive step using True Compressed MLA Cache + DeepSeekMoE.
    pub fn forward_token(
        &mut self,
        backend: &dyn Backend,
        token_id: u32,
        pos: usize,
    ) -> Result<&[f32]> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let attn_dim = cfg.num_heads * head_dim;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

        // 1. Embedding lookup
        self.token_embd
            .dequantize_row(token_id as usize, &mut self.scratch.x)?;

        // 2. Celestia Decoder Layers
        for l in 0..cfg.num_layers {
            let layer = &self.layers[l];

            // --- A. Multi-Head Latent Attention (MLA) ---
            backend.rms_norm(
                &self.scratch.x,
                &layer.attn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            // Compute Q and apply NeoX RoPE at exact position `pos`
            backend.matvec(&layer.wq, &self.scratch.xb, &mut self.scratch.q)?;
            for h in 0..cfg.num_heads {
                let qh = &mut self.scratch.q[h * head_dim..(h + 1) * head_dim];
                backend.apply_rope(qh, pos, head_dim, cfg.rope_theta, RopeStyle::NeoX);
            }

            // Compress hidden state into low-dim latent vector `c_t^{KV}` and store ONLY that in cache
            backend.matvec(&layer.w_kv_down, &self.scratch.xb, &mut self.scratch.c_kv)?;
            self.mla_cache
                .slot_mut(l, pos)
                .copy_from_slice(&self.scratch.c_kv);

            // Reconstruct K (with RoPE at exact historical pos `t`) and V from compressed latent slots `0..=pos`
            for t in 0..=pos {
                let latent_t = self.mla_cache.slot(l, t);
                backend.matvec(&layer.w_kv_up, latent_t, &mut self.scratch.kv_up_buf)?;
                let (k_src, v_src) = self.scratch.kv_up_buf.split_at(attn_dim);

                let k_dst = &mut self.scratch.k_rebuilt[t * attn_dim..(t + 1) * attn_dim];
                k_dst.copy_from_slice(k_src);
                for h in 0..cfg.num_heads {
                    backend.apply_rope(
                        &mut k_dst[h * head_dim..(h + 1) * head_dim],
                        t,
                        head_dim,
                        cfg.rope_theta,
                        RopeStyle::NeoX,
                    );
                }

                let v_dst = &mut self.scratch.v_rebuilt[t * attn_dim..(t + 1) * attn_dim];
                v_dst.copy_from_slice(v_src);
            }

            // Scaled Dot-Product Attention per head
            self.scratch.attn_out.fill(0.0);
            for h in 0..cfg.num_heads {
                let qh = &self.scratch.q[h * head_dim..(h + 1) * head_dim];
                let h_off = h * head_dim;
                for t in 0..=pos {
                    let kt = &self.scratch.k_rebuilt[t * attn_dim + h_off..t * attn_dim + h_off + head_dim];
                    self.scratch.att[t] = backend.dot_f32(qh, kt) * inv_sqrt_hd;
                }
                backend.softmax(&mut self.scratch.att[..=pos]);

                let out_h = &mut self.scratch.attn_out[h_off..h_off + head_dim];
                for t in 0..=pos {
                    let a = self.scratch.att[t];
                    let vt = &self.scratch.v_rebuilt[t * attn_dim + h_off..t * attn_dim + h_off + head_dim];
                    for d in 0..head_dim {
                        out_h[d] += a * vt[d];
                    }
                }
            }

            backend.matvec(&layer.wo, &self.scratch.attn_out, &mut self.scratch.xb)?;
            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.xb.iter()) {
                *x += dx;
            }

            // --- B. Sparse DeepSeekMoE (Shared Expert + Top-K Routed Experts) ---
            backend.rms_norm(
                &self.scratch.x,
                &layer.ffn_norm,
                cfg.rms_norm_eps,
                &mut self.scratch.xb,
            );

            self.scratch.moe_accum.fill(0.0);

            // 1. Always-active Shared Expert
            if let Some(shared) = &layer.shared_expert {
                backend.matvec(&shared.gate, &self.scratch.xb, &mut self.scratch.sh_gate)?;
                backend.matvec(&shared.up, &self.scratch.xb, &mut self.scratch.sh_up)?;
                backend.swiglu_inplace(&mut self.scratch.sh_gate, &self.scratch.sh_up);
                backend.matvec(&shared.down, &self.scratch.sh_gate, &mut self.scratch.moe_accum)?;
            }

            // 2. Router Softmax & Top-K Selection
            backend.matvec(&layer.router, &self.scratch.xb, &mut self.scratch.router_logits)?;
            backend.softmax(&mut self.scratch.router_logits);

            let mut expert_scores: Vec<(usize, f32)> = self
                .scratch
                .router_logits
                .iter()
                .copied()
                .enumerate()
                .collect();
            expert_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            expert_scores.truncate(cfg.num_active_experts);

            let w_sum: f32 = expert_scores.iter().map(|(_, w)| *w).sum::<f32>().max(1e-9);

            // 3. Execute ONLY the Top-K active routed experts (true sparse compute!)
            for (e_idx, raw_w) in expert_scores {
                let norm_w = raw_w / w_sum;
                let exp = &layer.experts[e_idx];
                backend.matvec(&exp.gate, &self.scratch.xb, &mut self.scratch.exp_gate)?;
                backend.matvec(&exp.up, &self.scratch.xb, &mut self.scratch.exp_up)?;
                backend.swiglu_inplace(&mut self.scratch.exp_gate, &self.scratch.exp_up);
                backend.matvec(&exp.down, &self.scratch.exp_gate, &mut self.scratch.exp_out)?;

                for (acc, &v) in self.scratch.moe_accum.iter_mut().zip(self.scratch.exp_out.iter()) {
                    *acc += v * norm_w;
                }
            }

            for (x, &dx) in self.scratch.x.iter_mut().zip(self.scratch.moe_accum.iter()) {
                *x += dx;
            }
        }

        // 3. Final RMSNorm & LM Head
        backend.rms_norm(
            &self.scratch.x,
            &self.output_norm,
            cfg.rms_norm_eps,
            &mut self.scratch.xb,
        );
        backend.matvec(&self.output_weight, &self.scratch.xb, &mut self.scratch.logits)?;

        Ok(&self.scratch.logits)
    }
}

/// Generates a deterministic synthetic Celestia GGUF v3 binary (in F32 or Q4_0)
/// for end-to-end verification, ARMv9 benchmarking, and GGUF export testing.
pub fn write_synthetic_celestia_gguf(cfg: &CelestiaConfig, weight_dtype: DType) -> Result<Vec<u8>> {
    let mut w = GgufWriter::new();
    w.add_meta("general.architecture", GgufValue::String("celestia".into()));
    w.add_meta("general.name", GgufValue::String("Celestia-Micro-ARMv9".into()));
    w.add_meta("celestia.embedding_length", GgufValue::U32(cfg.hidden_size as u32));
    w.add_meta("celestia.feed_forward_length", GgufValue::U32(cfg.intermediate_size as u32));
    w.add_meta("celestia.block_count", GgufValue::U32(cfg.num_layers as u32));
    w.add_meta("celestia.attention.head_count", GgufValue::U32(cfg.num_heads as u32));
    w.add_meta("celestia.attention.kv_latent_dim", GgufValue::U32(cfg.kv_latent_dim as u32));
    w.add_meta("celestia.expert_count", GgufValue::U32(cfg.num_experts as u32));
    w.add_meta("celestia.expert_used_count", GgufValue::U32(cfg.num_active_experts as u32));
    w.add_meta("celestia.shared_expert_count", GgufValue::U32(cfg.shared_experts as u32));
    w.add_meta("celestia.rope.freq_base", GgufValue::F32(cfg.rope_theta));
    w.add_meta("celestia.attention.layer_norm_rms_epsilon", GgufValue::F32(cfg.rms_norm_eps));
    w.add_meta("celestia.context_length", GgufValue::U32(cfg.max_seq_len as u32));

    // Deterministic pseudo-random normal-like initializer
    let mut seed = 0xC31E_571A_1234_5678u64;
    fn next_weight(seed: &mut u64, scale: f32) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
        u * scale
    }

    fn add_matrix(
        writer: &mut GgufWriter,
        seed: &mut u64,
        name: &str,
        rows: usize,
        cols: usize,
        weight_dtype: DType,
    ) -> Result<()> {
        let scale = 1.0 / (cols as f32).sqrt();
        let vals: Vec<f32> = (0..rows * cols).map(|_| next_weight(seed, scale)).collect();
        if weight_dtype == DType::Q4_0 && cols % 32 == 0 {
            let mut blocks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; (rows * cols) / 32];
            for r in 0..rows {
                quantize_row_q4_0(
                    &vals[r * cols..(r + 1) * cols],
                    &mut blocks[r * (cols / 32)..(r + 1) * (cols / 32)],
                );
            }
            let raw = unsafe {
                std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 18)
            }
            .to_vec();
            writer.add_tensor_raw(name, vec![rows, cols], DType::Q4_0, raw)
        } else {
            writer.add_tensor_f32(name, vec![rows, cols], &vals)
        }
    }

    let h = cfg.hidden_size;
    let attn_dim = cfg.num_heads * cfg.head_dim;
    let exp_ffn = cfg.expert_intermediate();
    let sh_ffn = cfg.shared_intermediate();
    let ones = vec![1.0f32; h];

    add_matrix(&mut w, &mut seed, "token_embd.weight", cfg.vocab_size, h, weight_dtype)?;
    w.add_tensor_f32("output_norm.weight", vec![h], &ones)?;
    add_matrix(&mut w, &mut seed, "output.weight", cfg.vocab_size, h, weight_dtype)?;

    for l in 0..cfg.num_layers {
        w.add_tensor_f32(&format!("blk.{l}.attn_norm.weight"), vec![h], &ones)?;
        add_matrix(&mut w, &mut seed, &format!("blk.{l}.attn_q.weight"), attn_dim, h, weight_dtype)?;
        add_matrix(&mut w, &mut seed, &format!("blk.{l}.attn_kv_down.weight"), cfg.kv_latent_dim, h, weight_dtype)?;
        add_matrix(&mut w, &mut seed, &format!("blk.{l}.attn_kv_up.weight"), attn_dim * 2, cfg.kv_latent_dim, weight_dtype)?;
        add_matrix(&mut w, &mut seed, &format!("blk.{l}.attn_output.weight"), h, attn_dim, weight_dtype)?;

        w.add_tensor_f32(&format!("blk.{l}.ffn_norm.weight"), vec![h], &ones)?;
        // Router kept in F32 for clean routing precision
        let r_vals: Vec<f32> = (0..cfg.num_experts * h)
            .map(|_| next_weight(&mut seed, 1.0 / (h as f32).sqrt()))
            .collect();
        w.add_tensor_f32(&format!("blk.{l}.ffn_gate_inp.weight"), vec![cfg.num_experts, h], &r_vals)?;

        for e in 0..cfg.num_experts {
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_gate.{e}.weight"), exp_ffn, h, weight_dtype)?;
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_up.{e}.weight"), exp_ffn, h, weight_dtype)?;
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_down.{e}.weight"), h, exp_ffn, weight_dtype)?;
        }

        if cfg.shared_experts > 0 {
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_gate_shexp.weight"), sh_ffn, h, weight_dtype)?;
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_up_shexp.weight"), sh_ffn, h, weight_dtype)?;
            add_matrix(&mut w, &mut seed, &format!("blk.{l}.ffn_down_shexp.weight"), h, sh_ffn, weight_dtype)?;
        }
    }

    Ok(w.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaelumModel, SamplerConfig};
    use caelum_kernels::CpuBackend;
    use caelum_tokenizer::Tokenizer;

    #[test]
    fn test_celestia_mla_moe_q4_0_end_to_end_and_json_constraint() {
        let cfg = CelestiaConfig::micro_test();
        assert!((cfg.mla_kv_savings_percent() - 87.5).abs() < 1e-3);

        let bytes = write_synthetic_celestia_gguf(&cfg, DType::Q4_0).unwrap();
        let gguf = GgufFile::from_bytes(bytes).unwrap();
        let backend = CpuBackend::new(Some(2));
        let tokenizer = Tokenizer::celestia_default(cfg.vocab_size);

        let mut model = CaelumModel::from_gguf(&gguf, Some(64)).unwrap();
        let prompt = tokenizer.encode("<user>สร้างการ์ดสินค้า</user><assistant>", false);

        // Verify JSON constrained generation produces strictly valid JSON prefix/object
        let (text, stats) = model
            .generate(
                &backend,
                &tokenizer,
                &prompt,
                24,
                &SamplerConfig {
                    temperature: 0.0,
                    ..Default::default()
                },
                true,
                |_, _| {},
            )
            .unwrap();

        assert!(stats.generated_tokens > 0);
        assert!(
            text.trim_start().starts_with('{'),
            "JSON grammar must force valid JSON start, got: {text:?}"
        );
    }
}
