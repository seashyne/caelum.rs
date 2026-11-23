//! KV Caches: Standard GQA KV Cache + True Compressed MLA Latent Cache (ARMv9 bandwidth saver).

/// Standard Grouped-Query Attention KV Cache (`[layer][pos][kv_head * head_dim]`).
pub struct KvCache {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub kv_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let total = num_layers * max_seq_len * kv_dim;
        Self {
            num_layers,
            max_seq_len,
            kv_dim,
            k: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    pub fn clear(&mut self) {
        self.k.fill(0.0);
        self.v.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn k_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_dim;
        &mut self.k[off..off + self.kv_dim]
    }

    #[inline]
    pub fn v_slot_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_dim;
        &mut self.v[off..off + self.kv_dim]
    }

    #[inline]
    pub fn k_slot(&self, layer: usize, pos: usize) -> &[f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.k[off..off + self.kv_dim]
    }

    #[inline]
    pub fn v_slot(&self, layer: usize, pos: usize) -> &[f32] {
        let off = (layer * self.max_seq_len + pos) * self.kv_dim;
        &self.v[off..off + self.kv_dim]
    }
}

/// True Compressed Multi-Head Latent Attention (MLA) KV Cache for Celestia.
///
/// Stores ONLY the compressed latent vector `c_t^{KV}` (`[kv_latent_dim]`) per token per layer,
/// cutting KV cache RAM & memory bandwidth by 75–87% on ARMv9 edge devices.
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
