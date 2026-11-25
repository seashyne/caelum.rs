//! Precomputed RoPE lookup table supporting Normal & NeoX layouts, Linear scaling,
//! and exact YaRN frequency interpolation (Llama 3 / Qwen2.5 / Qwen3 / DeepSeek-V2/V3).

use caelum_gguf::GgufFile;
use caelum_kernels::RopeStyle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    None,
    Linear,
    Yarn,
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub rope_dim: usize,
    pub freq_base: f32,
    pub style: RopeStyle,
    pub scaling_type: RopeScalingType,
    pub scaling_factor: f32,
    pub orig_ctx_len: usize,
    pub attn_factor: f32,
    pub yarn_log_mul: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// If true (MLA architectures like DeepSeek2), keep cos/sin unit magnitude and apply
    /// YaRN `mscale` to the attention logit scale (`kq_scale`) instead.
    pub unit_magnitude: bool,
}

impl RopeConfig {
    pub fn from_gguf(gguf: &GgufFile, arch: &str, head_dim: usize, ctx_len: usize) -> Self {
        let p = arch;
        let rope_dim = gguf
            .get_meta(&format!("{p}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let freq_base = gguf
            .get_meta(&format!("{p}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| {
                if arch.starts_with("qwen") || arch == "gemma3" {
                    1_000_000.0
                } else {
                    10_000.0
                }
            });
        let style = if arch.starts_with("qwen")
            || arch == "celestia"
            || matches!(
                arch,
                "gemma" | "gemma2" | "gemma3" | "gemma4" | "phi3" | "starcoder2"
            )
        {
            RopeStyle::NeoX
        } else {
            RopeStyle::Normal
        };

        let scaling_str = gguf
            .get_meta(&format!("{p}.rope.scaling.type"))
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        let scaling_factor = gguf
            .get_meta(&format!("{p}.rope.scaling.factor"))
            .or_else(|| gguf.get_meta(&format!("{p}.rope.scale_linear")))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);
        let scaling_type = if scaling_factor <= 1.0 && scaling_str != "yarn" {
            RopeScalingType::None
        } else {
            match scaling_str {
                "linear" => RopeScalingType::Linear,
                "yarn" => RopeScalingType::Yarn,
                _ => RopeScalingType::None,
            }
        };

        let orig_ctx_len = gguf
            .get_meta(&format!("{p}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(ctx_len);
        let attn_factor = gguf
            .get_meta(&format!("{p}.rope.scaling.attn_factor"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);
        let yarn_log_mul = gguf
            .get_meta(&format!("{p}.rope.scaling.yarn_log_multiplier"))
            .and_then(|v| v.as_f32())
            .unwrap_or(0.0);

        Self {
            rope_dim,
            freq_base,
            style,
            scaling_type,
            scaling_factor: scaling_factor.max(1.0),
            orig_ctx_len: orig_ctx_len.max(1),
            attn_factor,
            yarn_log_mul,
            beta_fast: 32.0,
            beta_slow: 1.0,
            unit_magnitude: arch == "deepseek2",
        }
    }

    /// Computes DeepSeek2 MLA YaRN `mscale` applied to attention score scaling (`mscale^2 / sqrt(dk)`).
    pub fn yarn_kq_mscale(&self) -> f32 {
        if self.scaling_type == RopeScalingType::Yarn && self.scaling_factor > 1.0 {
            let log_f = self.scaling_factor.ln();
            let log_mul = if self.yarn_log_mul > 0.0 {
                self.yarn_log_mul
            } else {
                0.1
            };
            self.attn_factor * (1.0 + log_mul * log_f)
        } else {
            1.0
        }
    }
}

pub struct RopeTable {
    half_dim: usize,
    style: RopeStyle,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    pub fn new(max_seq_len: usize, cfg: &RopeConfig) -> Self {
        Self::with_freq_factors(max_seq_len, cfg, None)
    }

    /// Builds a RoPE table with optional per-pair frequency divisors (`rope_freqs.weight`,
    /// e.g. Gemma 4 proportional RoPE where unrotated pairs carry factor `~1e30`).
    pub fn with_freq_factors(
        max_seq_len: usize,
        cfg: &RopeConfig,
        freq_factors: Option<&[f32]>,
    ) -> Self {
        let rope_dim = cfg.rope_dim;
        let half_dim = rope_dim / 2;
        let freq_scale = if cfg.scaling_type == RopeScalingType::None {
            1.0f32
        } else {
            1.0 / cfg.scaling_factor
        };

        // Precompute YaRN ramp boundaries matching `ggml_rope_yarn_corr_dims`
        let (low_dim, high_dim, ext_factor, mscale_rope) =
            if cfg.scaling_type == RopeScalingType::Yarn && cfg.scaling_factor > 1.0 {
                let corr_dim = |n_rot: f32| -> f32 {
                    (rope_dim as f32
                        * (cfg.orig_ctx_len as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln())
                        / (2.0 * cfg.freq_base.ln())
                };
                let low = corr_dim(cfg.beta_fast).floor().max(0.0);
                let high = corr_dim(cfg.beta_slow)
                    .ceil()
                    .min((rope_dim.saturating_sub(1)) as f32);
                let ms = if cfg.unit_magnitude {
                    1.0
                } else {
                    cfg.attn_factor * (1.0 + 0.1 * cfg.scaling_factor.ln())
                };
                (low, high, 1.0f32, ms)
            } else {
                (0.0, 0.0, 0.0f32, 1.0f32)
            };

        let mut inv_freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let extrap = 1.0 / cfg.freq_base.powf((2 * i) as f32 / rope_dim as f32);
            let interp = freq_scale * extrap;
            let mut freq = if ext_factor != 0.0 {
                let y = (i as f32 - low_dim) / (high_dim - low_dim).max(0.001);
                let ramp = (1.0 - y.clamp(0.0, 1.0)) * ext_factor;
                interp * (1.0 - ramp) + extrap * ramp
            } else {
                interp
            };
            if let Some(factors) = freq_factors {
                if let Some(&ff) = factors.get(i) {
                    if ff.is_finite() && ff != 0.0 {
                        freq /= ff;
                    }
                }
            }
            inv_freqs.push(freq);
        }

        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            let p = pos as f32;
            let row = pos * half_dim;
            for i in 0..half_dim {
                let (s, c) = (p * inv_freqs[i]).sin_cos();
                cos[row + i] = c * mscale_rope;
                sin[row + i] = s * mscale_rope;
            }
        }

        Self {
            half_dim,
            style: cfg.style,
            cos,
            sin,
        }
    }

    #[inline]
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    #[inline]
    pub fn is_neox(&self) -> bool {
        matches!(self.style, RopeStyle::NeoX)
    }

    /// Packs `[cos[pos, 0..half], sin[pos, 0..half]]` per position for GPU VRAM upload.
    pub fn packed_cos_sin(&self) -> Vec<f32> {
        let half = self.half_dim;
        if half == 0 {
            return Vec::new();
        }
        let max_seq = self.cos.len() / half;
        let mut out = Vec::with_capacity(max_seq * half * 2);
        for pos in 0..max_seq {
            let row = pos * half;
            out.extend_from_slice(&self.cos[row..row + half]);
            out.extend_from_slice(&self.sin[row..row + half]);
        }
        out
    }

    #[inline]
    pub fn apply(&self, vec: &mut [f32], pos: usize) {
        let half = self.half_dim;
        if half == 0 {
            return;
        }
        let row = pos * half;
        let cs = &self.cos[row..row + half];
        let sn = &self.sin[row..row + half];
        match self.style {
            RopeStyle::Normal => {
                for i in 0..half {
                    let c = cs[i];
                    let s = sn[i];
                    let x0 = vec[2 * i];
                    let x1 = vec[2 * i + 1];
                    vec[2 * i] = x0 * c - x1 * s;
                    vec[2 * i + 1] = x0 * s + x1 * c;
                }
            }
            RopeStyle::NeoX => {
                for i in 0..half {
                    let c = cs[i];
                    let s = sn[i];
                    let x0 = vec[i];
                    let x1 = vec[i + half];
                    vec[i] = x0 * c - x1 * s;
                    vec[i + half] = x0 * s + x1 * c;
                }
            }
        }
    }
}
