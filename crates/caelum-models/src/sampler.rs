//! Autoregressive token sampler: Greedy, Temperature, Top-K, Top-P (Nucleus), and Repetition Penalty.

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 42,
        }
    }
}

/// Deterministic Xoshiro256** PRNG (zero external dependencies).
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn from_seed(seed: u64) -> Self {
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0u64; 4];
        for slot in &mut s {
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = x ^ (x >> 31);
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = Rng::from_seed(cfg.seed);
        Self { cfg, rng }
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        // 1. Repetition penalty
        if (self.cfg.repetition_penalty - 1.0).abs() > 1e-6 {
            let window_start = history.len().saturating_sub(64);
            let mut seen = Vec::new();
            for &tok in &history[window_start..] {
                let idx = tok as usize;
                if idx < logits.len() && !seen.contains(&idx) {
                    seen.push(idx);
                    if logits[idx].is_finite() {
                        if logits[idx] > 0.0 {
                            logits[idx] /= self.cfg.repetition_penalty;
                        } else {
                            logits[idx] *= self.cfg.repetition_penalty;
                        }
                    }
                }
            }
        }

        // 2. Greedy decoding
        if self.cfg.temperature <= 0.0 {
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_id = i;
                }
            }
            return best_id as u32;
        }

        // 3. Temperature + Top-K + Top-P
        let inv_temp = 1.0 / self.cfg.temperature;
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.is_finite())
            .map(|(i, &v)| (i, v * inv_temp))
            .collect();

        if candidates.is_empty() {
            return 0;
        }

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        if self.cfg.top_k > 0 && candidates.len() > self.cfg.top_k {
            candidates.truncate(self.cfg.top_k);
        }

        let max_logit = candidates[0].1;
        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
        let mut sum = 0.0f32;
        for (id, val) in candidates {
            let p = (val - max_logit).exp();
            sum += p;
            probs.push((id, p));
        }
        for (_, p) in &mut probs {
            *p /= sum;
        }

        // Nucleus Top-P cutoff
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for &(_, p) in &probs {
                cum += p;
                keep += 1;
                if cum >= self.cfg.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
            let renorm: f32 = probs.iter().map(|(_, p)| *p).sum();
            for (_, p) in &mut probs {
                *p /= renorm;
            }
        }

        let r = self.rng.next_f32();
        let mut cdf = 0.0f32;
        for &(id, p) in &probs {
            cdf += p;
            if r <= cdf {
                return id as u32;
            }
        }
        probs.last().map(|&(id, _)| id as u32).unwrap_or(0)
    }
}
