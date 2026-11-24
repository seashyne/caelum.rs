//! KV Caches: Standard GQA KV Cache (`F32` or `Q8_0` 3.76× compressed) + True Compressed MLA Latent Cache.

use caelum_core::{f16_to_f32, quantize_row_q8_0, BlockQ8_0, QK8_0};
use caelum_kernels::Backend;

/// Precision mode for standard GQA KV Cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// Full 32-bit float KV cache (maximum precision).
    F32,
    /// Block-wise 8-bit `Q8_0` quantized KV cache (3.76× less RAM & memory bandwidth on ARMv9).
    Q8_0,
}

enum KvStorage {
    F32 {
        k: Vec<f32>,
        v: Vec<f32>,
    },
    Q8_0 {
        blocks_per_head: usize,
        k: Vec<BlockQ8_0>,
        v: Vec<BlockQ8_0>,
    },
}

/// Standard Grouped-Query Attention KV Cache supporting both `F32` and `Q8_0` storage.
pub struct KvCache {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub precision: KvPrecision,
    storage: KvStorage,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        precision: KvPrecision,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        // Use Q8_0 only when head_dim is divisible by 32; otherwise fall back cleanly to F32
        let actual_prec = if precision == KvPrecision::Q8_0 && head_dim % QK8_0 == 0 {
            KvPrecision::Q8_0
        } else {
            KvPrecision::F32
        };

        let storage = match actual_prec {
            KvPrecision::F32 => {
                let total = num_layers * max_seq_len * kv_dim;
                KvStorage::F32 {
                    k: vec![0.0; total],
                    v: vec![0.0; total],
                }
            }
            KvPrecision::Q8_0 => {
                let blocks_per_head = head_dim / QK8_0;
                let total_blocks = num_layers * max_seq_len * num_kv_heads * blocks_per_head;
                let zero = BlockQ8_0 { d: 0, qs: [0; 32] };
                KvStorage::Q8_0 {
                    blocks_per_head,
                    k: vec![zero; total_blocks],
                    v: vec![zero; total_blocks],
                }
            }
        };

        Self {
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            precision: actual_prec,
            storage,
        }
    }

    pub fn clear(&mut self) {
        match &mut self.storage {
            KvStorage::F32 { k, v } => {
                k.fill(0.0);
                v.fill(0.0);
            }
            KvStorage::Q8_0 { k, v, .. } => {
                let zero = BlockQ8_0 { d: 0, qs: [0; 32] };
                k.fill(zero);
                v.fill(zero);
            }
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match &self.storage {
            KvStorage::F32 { k, v } => (k.len() + v.len()) * std::mem::size_of::<f32>(),
            KvStorage::Q8_0 { k, v, .. } => (k.len() + v.len()) * std::mem::size_of::<BlockQ8_0>(),
        }
    }

    /// Stores one position's K and V vectors (`length = num_kv_heads * head_dim`) into `(layer, pos)`.
    pub fn store(&mut self, layer: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        let kv_dim = self.num_kv_heads * self.head_dim;
        debug_assert_eq!(k_vec.len(), kv_dim);
        debug_assert_eq!(v_vec.len(), kv_dim);

        match &mut self.storage {
            KvStorage::F32 { k, v } => {
                let off = (layer * self.max_seq_len + pos) * kv_dim;
                k[off..off + kv_dim].copy_from_slice(k_vec);
                v[off..off + kv_dim].copy_from_slice(v_vec);
            }
            KvStorage::Q8_0 {
                blocks_per_head,
                k,
                v,
            } => {
                let blocks_per_pos = self.num_kv_heads * *blocks_per_head;
                let base = (layer * self.max_seq_len + pos) * blocks_per_pos;
                for kh in 0..self.num_kv_heads {
                    let src_off = kh * self.head_dim;
                    let dst_off = base + kh * *blocks_per_head;
                    quantize_row_q8_0(
                        &k_vec[src_off..src_off + self.head_dim],
                        &mut k[dst_off..dst_off + *blocks_per_head],
                    );
                    quantize_row_q8_0(
                        &v_vec[src_off..src_off + self.head_dim],
                        &mut v[dst_off..dst_off + *blocks_per_head],
                    );
                }
            }
        }
    }

    /// Computes causal attention output for a single query head `qh` against cached `0..=max_t`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_head(
        &self,
        backend: &dyn Backend,
        layer: usize,
        kv_head: usize,
        qh: &[f32],
        max_t: usize,
        scale: f32,
        att_buf: &mut [f32],
        out_head: &mut [f32],
    ) {
        self.attend_head_ext(
            backend, layer, kv_head, qh, max_t, scale, None, 0.0, att_buf, out_head,
        );
    }

    /// Extended causal attention supporting Sliding Window Attention (Gemma-2/3, Phi-3, Mistral,
    /// StarCoder2) and Attention Logit Soft-Capping (`softcap * tanh(score / softcap)`, Gemma-2).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_head_ext(
        &self,
        backend: &dyn Backend,
        layer: usize,
        kv_head: usize,
        qh: &[f32],
        max_t: usize,
        scale: f32,
        sliding_window: Option<usize>,
        logit_softcap: f32,
        att_buf: &mut [f32],
        out_head: &mut [f32],
    ) {
        let head_dim = self.head_dim;
        let kv_dim = self.num_kv_heads * head_dim;
        out_head.fill(0.0);
        let t_start = match sliding_window {
            Some(w) if w > 0 => max_t.saturating_sub(w - 1),
            _ => 0,
        };
        let win_len = max_t - t_start + 1;
        let att_slice = &mut att_buf[..win_len];
        let inv_cap = if logit_softcap > 0.0 {
            1.0 / logit_softcap
        } else {
            0.0
        };

        match &self.storage {
            KvStorage::F32 { k, v } => {
                let kv_off = kv_head * head_dim;
                let layer_base = layer * self.max_seq_len * kv_dim;
                for (wi, t) in (t_start..=max_t).enumerate() {
                    let off = layer_base + t * kv_dim + kv_off;
                    let mut score = backend.dot_f32(qh, &k[off..off + head_dim]) * scale;
                    if logit_softcap > 0.0 {
                        score = logit_softcap * (score * inv_cap).tanh();
                    }
                    att_slice[wi] = score;
                }
                backend.softmax(att_slice);

                for (wi, t) in (t_start..=max_t).enumerate() {
                    let a = att_slice[wi];
                    let off = layer_base + t * kv_dim + kv_off;
                    let vt = &v[off..off + head_dim];
                    for d in 0..head_dim {
                        out_head[d] += a * vt[d];
                    }
                }
            }
            KvStorage::Q8_0 {
                blocks_per_head,
                k,
                v,
            } => {
                let bph = *blocks_per_head;
                let blocks_per_pos = self.num_kv_heads * bph;
                let layer_base = layer * self.max_seq_len * blocks_per_pos;
                let h_off = kv_head * bph;

                // Quantize query head once into Q8_0 for fast SIMD integer attention scores
                let mut q_q8 = [BlockQ8_0 { d: 0, qs: [0; 32] }; 8];
                debug_assert!(bph <= 8, "head_dim > 256 unsupported in stack q_q8");
                let q_blocks = &mut q_q8[..bph];
                quantize_row_q8_0(qh, q_blocks);

                for (wi, t) in (t_start..=max_t).enumerate() {
                    let off = layer_base + t * blocks_per_pos + h_off;
                    let mut score = backend.dot_q8_0(q_blocks, &k[off..off + bph]) * scale;
                    if logit_softcap > 0.0 {
                        score = logit_softcap * (score * inv_cap).tanh();
                    }
                    att_slice[wi] = score;
                }
                backend.softmax(att_slice);

                for (wi, t) in (t_start..=max_t).enumerate() {
                    let a = att_slice[wi];
                    let off = layer_base + t * blocks_per_pos + h_off;
                    let v_blocks = &v[off..off + bph];
                    for (bi, blk) in v_blocks.iter().enumerate() {
                        let s = a * f16_to_f32(blk.d);
                        let dst = &mut out_head[bi * QK8_0..(bi + 1) * QK8_0];
                        let qs = blk.qs;
                        for j in 0..QK8_0 {
                            dst[j] += s * (qs[j] as f32);
                        }
                    }
                }
            }
        }
    }
}

/// True Compressed Multi-Head Latent Attention (MLA) KV Cache for Celestia.
///
/// Stores ONLY the compressed latent vector `c_t^{KV}` (`[kv_latent_dim]`) per token per layer,
/// cutting KV cache RAM & memory bandwidth by 75–87.5% on ARMv9 edge devices.
pub struct MlaLatentCache {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub kv_latent_dim: usize,
    latent: Vec<f32>,
}

impl MlaLatentCache {
    pub fn new(num_layers: usize, max_seq_len: usize, kv_latent_dim: usize) -> Self {
        Self {
            num_layers,
            max_seq_len,
            kv_latent_dim,
            latent: vec![0.0; num_layers * max_seq_len * kv_latent_dim],
        }
    }

    pub fn clear(&mut self) {
        self.latent.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        self.latent.len() * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_latent_dim;
        &mut self.latent[off..off + self.kv_latent_dim]
    }

    #[inline]
    pub fn slot(&self, layer: usize, pos: usize) -> &[f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_latent_dim;
        &self.latent[off..off + self.kv_latent_dim]
    }
}
