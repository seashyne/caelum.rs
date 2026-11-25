//! High-Performance WGSL Compute Shaders for Caelum's `wgpu` GPU Backend (`--features gpu`).
//!
//! Optimized for discrete & integrated Vulkan / Metal / DX12 GPUs (verified on Intel Arc B580):
//! - Hardware `enable subgroups;` (`subgroupAdd` / `subgroupMax`) with automatic portable fallback
//! - 4-byte-aligned `Q6_K` VRAM layout (`53 u32` = 212B/block, pre-decoded `f32` `d`, hardware `extractBits`)
//! - Fused `Q4_K` **QKV + Optional Bias** (1 dispatch instead of 6!)
//! - Fused `Q4_K` **Gate + Up + SwiGLU** (1 dispatch, reads `X` once, 0 `hb2` VRAM traffic!)
//! - Fused **Residual Add + RMSNorm** (1 dispatch instead of 2, halves `X` traffic!)
//! - Packed **F16 VRAM KV Cache** (`pack2x16float` / `unpack2x16float`, 2× less VRAM & 2× KV bandwidth!)
//! - Online Tiled Flash-Attention GQA with subgroup reductions

/// Specializes a WGSL shader template for either hardware `subgroups` (`subgroupAdd`/`subgroupMax`)
/// or portable `workgroup` shared-memory tree reductions.
pub fn specialize_wgsl(template: &str, use_subgroups: bool) -> String {
    // Note: Naga 30 exposes `subgroupAdd`/`subgroupMax` & subgroup builtins directly when
    // `wgpu::Features::SUBGROUP` is enabled on the device (rejecting `enable subgroups;`).
    let sg_enable = "";
    let sg_args = if use_subgroups {
        ", @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_inv: u32"
    } else {
        ""
    };

    let reduce_row_f32 = if use_subgroups {
        r#"
    let sg_sum = subgroupAdd(acc);
    if (sg_inv == 0u) {
        smem[r_local * 4u + (tid / sg_size)] = sg_sum;
    }
    workgroupBarrier();
    var row_sum = 0.0;
    if (tid == 0u) {
        let n_sg = 32u / sg_size;
        for (var k = 0u; k < n_sg; k++) {
            row_sum += smem[r_local * 4u + k];
        }
    }"#
    } else {
        r#"
    let sidx = r_local * 32u + tid;
    smem[sidx] = acc;
    workgroupBarrier();
    if (tid < 16u) { smem[sidx] += smem[sidx + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { smem[sidx] += smem[sidx + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { smem[sidx] += smem[sidx + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { smem[sidx] += smem[sidx + 2u]; }
    workgroupBarrier();
    var row_sum = 0.0;
    if (tid == 0u) {
        row_sum = smem[sidx] + smem[sidx + 1u];
    }"#
    };

    let reduce_row_vec2 = if use_subgroups {
        r#"
    let sg_sum = subgroupAdd(vec2<f32>(acc_g, acc_u));
    if (sg_inv == 0u) {
        smem2[r_local * 4u + (tid / sg_size)] = sg_sum;
    }
    workgroupBarrier();
    var row_sum2 = vec2<f32>(0.0);
    if (tid == 0u) {
        let n_sg = 32u / sg_size;
        for (var k = 0u; k < n_sg; k++) {
            row_sum2 += smem2[r_local * 4u + k];
        }
    }"#
    } else {
        r#"
    let sidx = r_local * 32u + tid;
    smem2[sidx] = vec2<f32>(acc_g, acc_u);
    workgroupBarrier();
    if (tid < 16u) { smem2[sidx] += smem2[sidx + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { smem2[sidx] += smem2[sidx + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { smem2[sidx] += smem2[sidx + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { smem2[sidx] += smem2[sidx + 2u]; }
    workgroupBarrier();
    var row_sum2 = vec2<f32>(0.0);
    if (tid == 0u) {
        row_sum2 = smem2[sidx] + smem2[sidx + 1u];
    }"#
    };

    let reduce_row_vec4 = if use_subgroups {
        r#"
    let sg_sum = subgroupAdd(vec4<f32>(a0, a1, a2, a3));
    if (sg_inv == 0u) {
        smem4[r_local * 4u + (tid / sg_size)] = sg_sum;
    }
    workgroupBarrier();
    var row_sum4 = vec4<f32>(0.0);
    if (tid == 0u) {
        let n_sg = 32u / sg_size;
        for (var k = 0u; k < n_sg; k++) {
            row_sum4 += smem4[r_local * 4u + k];
        }
    }"#
    } else {
        r#"
    let sidx = r_local * 32u + tid;
    smem4[sidx] = vec4<f32>(a0, a1, a2, a3);
    workgroupBarrier();
    if (tid < 16u) { smem4[sidx] += smem4[sidx + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { smem4[sidx] += smem4[sidx + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { smem4[sidx] += smem4[sidx + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { smem4[sidx] += smem4[sidx + 2u]; }
    workgroupBarrier();
    var row_sum4 = vec4<f32>(0.0);
    if (tid == 0u) {
        row_sum4 = smem4[sidx] + smem4[sidx + 1u];
    }"#
    };

    let reduce_wg64_sum = if use_subgroups {
        r#"
    let sg_ss = subgroupAdd(ss);
    if (sg_inv == 0u) { smem[tid / sg_size] = sg_ss; }
    workgroupBarrier();
    var wg_sum = 0.0;
    let n_sg = 64u / sg_size;
    for (var k = 0u; k < n_sg; k++) { wg_sum += smem[k]; }"#
    } else {
        r#"
    smem[tid] = ss;
    workgroupBarrier();
    if (tid < 32u) { smem[tid] += smem[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { smem[tid] += smem[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { smem[tid] += smem[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { smem[tid] += smem[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { smem[tid] += smem[tid + 2u]; }
    workgroupBarrier();
    let wg_sum = smem[0] + smem[1];"#
    };

    let reduce_wg64_max = if use_subgroups {
        r#"
    let sg_mx = subgroupMax(sc);
    if (sg_inv == 0u) { red[tid / sg_size] = sg_mx; }
    workgroupBarrier();
    var tile_max = -1e30;
    let n_sg_m = 64u / sg_size;
    for (var k = 0u; k < n_sg_m; k++) { tile_max = max(tile_max, red[k]); }
    workgroupBarrier();"#
    } else {
        r#"
    red[tid] = sc;
    workgroupBarrier();
    if (tid < 32u) { red[tid] = max(red[tid], red[tid + 32u]); }
    workgroupBarrier();
    if (tid < 16u) { red[tid] = max(red[tid], red[tid + 16u]); }
    workgroupBarrier();
    if (tid < 8u) { red[tid] = max(red[tid], red[tid + 8u]); }
    workgroupBarrier();
    if (tid < 4u) { red[tid] = max(red[tid], red[tid + 4u]); }
    workgroupBarrier();
    if (tid < 2u) { red[tid] = max(red[tid], red[tid + 2u]); }
    workgroupBarrier();
    let tile_max = max(red[0], red[1]);
    workgroupBarrier();"#
    };

    let reduce_wg64_wsum = if use_subgroups {
        r#"
    let sg_ws = subgroupAdd(w);
    if (sg_inv == 0u) { red[tid / sg_size] = sg_ws; }
    workgroupBarrier();
    var tile_sum = 0.0;
    let n_sg_s = 64u / sg_size;
    for (var k = 0u; k < n_sg_s; k++) { tile_sum += red[k]; }"#
    } else {
        r#"
    red[tid] = w;
    workgroupBarrier();
    if (tid < 32u) { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    let tile_sum = red[0] + red[1];"#
    };

    template
        .replace("/*SG_ENABLE*/", sg_enable)
        .replace("/*SG_ARGS*/", sg_args)
        .replace("/*REDUCE_ROW_F32*/", reduce_row_f32)
        .replace("/*REDUCE_ROW_VEC2*/", reduce_row_vec2)
        .replace("/*REDUCE_ROW_VEC4*/", reduce_row_vec4)
        .replace("/*REDUCE_WG64_SUM*/", reduce_wg64_sum)
        .replace("/*REDUCE_WG64_MAX*/", reduce_wg64_max)
        .replace("/*REDUCE_WG64_WSUM*/", reduce_wg64_wsum)
}

/// Quantized `Q4_K × F32` GEMV/GEMM (`36 u32` = `144` bytes per 256-elem super-block).
pub const SHADER_MATMUL_Q4_K: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let row_base = row * nb * 36u;
        let x_tok = tok * P.cols;
        let is_base = chunk * 2u;
        let x_chunk_off = chunk * 64u + k4 * 4u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 36u;
            let dm = unpack2x16float(W[blk]);
            let s0 = W[blk + 1u];
            let s1 = W[blk + 2u];
            let s2 = W[blk + 3u];
            let sm1 = scale_min_k4(is_base, s0, s1, s2);
            let sm2 = scale_min_k4(is_base + 1u, s0, s1, s2);

            let qw = W[blk + 4u + tid];
            let xb = x_tok + b * 256u + x_chunk_off;

            let b0 = qw & 0xFFu;
            let b1 = (qw >> 8u) & 0xFFu;
            let b2 = (qw >> 16u) & 0xFFu;
            let b3 = (qw >> 24u) & 0xFFu;

            let xl0 = X[xb];      let xh0 = X[xb + 32u];
            let xl1 = X[xb + 1u]; let xh1 = X[xb + 33u];
            let xl2 = X[xb + 2u]; let xh2 = X[xb + 34u];
            let xl3 = X[xb + 3u]; let xh3 = X[xb + 35u];

            let sq1 = f32(b0 & 0xFu) * xl0 + f32(b1 & 0xFu) * xl1 + f32(b2 & 0xFu) * xl2 + f32(b3 & 0xFu) * xl3;
            let sx1 = xl0 + xl1 + xl2 + xl3;
            let sq2 = f32(b0 >> 4u) * xh0 + f32(b1 >> 4u) * xh1 + f32(b2 >> 4u) * xh2 + f32(b3 >> 4u) * xh3;
            let sx2 = xh0 + xh1 + xh2 + xh3;

            acc += (dm.x * f32(sm1.x)) * sq1 - (dm.y * f32(sm1.y)) * sx1
                 + (dm.x * f32(sm2.x)) * sq2 - (dm.y * f32(sm2.y)) * sx2;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// Fused `Q4_K` **Q + K + V + Optional Bias** projection (1 dispatch instead of 6!).
/// `W_QKV` is concatenated `[wq; wk; wv]` (`q_dim + 2*kv_dim` rows).
pub const SHADER_MATMUL_Q4_K_QKV: &str = r#"/*SG_ENABLE*/
struct Params {
    q_dim: u32,
    kv_dim: u32,
    cols: u32,
    batch: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> V: array<f32>;
@group(0) @binding(6) var<uniform> P: Params;

var<workgroup> smem: array<f32, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let total_rows = P.q_dim + 2u * P.kv_dim;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var acc = 0.0;
    if (row < total_rows && tok < P.batch) {
        let row_base = row * nb * 36u;
        let x_tok = tok * P.cols;
        let is_base = chunk * 2u;
        let x_chunk_off = chunk * 64u + k4 * 4u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 36u;
            let dm = unpack2x16float(W[blk]);
            let s0 = W[blk + 1u];
            let s1 = W[blk + 2u];
            let s2 = W[blk + 3u];
            let sm1 = scale_min_k4(is_base, s0, s1, s2);
            let sm2 = scale_min_k4(is_base + 1u, s0, s1, s2);

            let qw = W[blk + 4u + tid];
            let xb = x_tok + b * 256u + x_chunk_off;

            let b0 = qw & 0xFFu;
            let b1 = (qw >> 8u) & 0xFFu;
            let b2 = (qw >> 16u) & 0xFFu;
            let b3 = (qw >> 24u) & 0xFFu;

            let xl0 = X[xb];      let xh0 = X[xb + 32u];
            let xl1 = X[xb + 1u]; let xh1 = X[xb + 33u];
            let xl2 = X[xb + 2u]; let xh2 = X[xb + 34u];
            let xl3 = X[xb + 3u]; let xh3 = X[xb + 35u];

            let sq1 = f32(b0 & 0xFu) * xl0 + f32(b1 & 0xFu) * xl1 + f32(b2 & 0xFu) * xl2 + f32(b3 & 0xFu) * xl3;
            let sx1 = xl0 + xl1 + xl2 + xl3;
            let sq2 = f32(b0 >> 4u) * xh0 + f32(b1 >> 4u) * xh1 + f32(b2 >> 4u) * xh2 + f32(b3 >> 4u) * xh3;
            let sx2 = xh0 + xh1 + xh2 + xh3;

            acc += (dm.x * f32(sm1.x)) * sq1 - (dm.y * f32(sm1.y)) * sx1
                 + (dm.x * f32(sm2.x)) * sq2 - (dm.y * f32(sm2.y)) * sx2;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < total_rows && tok < P.batch) {
        var val = row_sum;
        if (P.has_bias != 0u) {
            val += B[row];
        }
        if (row < P.q_dim) {
            Q[tok * P.q_dim + row] = val;
        } else if (row < P.q_dim + P.kv_dim) {
            K[tok * P.kv_dim + (row - P.q_dim)] = val;
        } else {
            V[tok * P.kv_dim + (row - P.q_dim - P.kv_dim)] = val;
        }
    }
}
"#;

/// Fused 4-token `Q4_K` **Q + K + V + Optional Bias** Prefill GEMM.
pub const SHADER_MATMUL_Q4_K_QKV_B4: &str = r#"/*SG_ENABLE*/
struct Params {
    q_dim: u32,
    kv_dim: u32,
    cols: u32,
    batch: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> V: array<f32>;
@group(0) @binding(6) var<uniform> P: Params;

var<workgroup> smem4: array<vec4<f32>, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let t_base = wid.y * 4u;
    let total_rows = P.q_dim + 2u * P.kv_dim;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < total_rows && t_base < P.batch) {
        let row_base = row * nb * 36u;
        let is_base = chunk * 2u;
        let x_chunk_off = chunk * 64u + k4 * 4u;
        let t0 = t_base * P.cols;
        let t1 = min(t_base + 1u, P.batch - 1u) * P.cols;
        let t2 = min(t_base + 2u, P.batch - 1u) * P.cols;
        let t3 = min(t_base + 3u, P.batch - 1u) * P.cols;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 36u;
            let dm = unpack2x16float(W[blk]);
            let s0 = W[blk + 1u];
            let s1 = W[blk + 2u];
            let s2 = W[blk + 3u];
            let sm1 = scale_min_k4(is_base, s0, s1, s2);
            let sm2 = scale_min_k4(is_base + 1u, s0, s1, s2);
            let d1 = dm.x * f32(sm1.x);
            let m1 = dm.y * f32(sm1.y);
            let d2 = dm.x * f32(sm2.x);
            let m2 = dm.y * f32(sm2.y);

            let qw = W[blk + 4u + tid];
            let q0l = f32(qw & 0xFu) * d1 - m1;
            let q1l = f32((qw >> 8u) & 0xFu) * d1 - m1;
            let q2l = f32((qw >> 16u) & 0xFu) * d1 - m1;
            let q3l = f32((qw >> 24u) & 0xFu) * d1 - m1;
            let q0h = f32((qw >> 4u) & 0xFu) * d2 - m2;
            let q1h = f32((qw >> 12u) & 0xFu) * d2 - m2;
            let q2h = f32((qw >> 20u) & 0xFu) * d2 - m2;
            let q3h = f32((qw >> 28u) & 0xFu) * d2 - m2;

            let off = b * 256u + x_chunk_off;
            let b0 = t0 + off;
            a0 += q0l * X[b0] + q1l * X[b0+1u] + q2l * X[b0+2u] + q3l * X[b0+3u]
                + q0h * X[b0+32u] + q1h * X[b0+33u] + q2h * X[b0+34u] + q3h * X[b0+35u];
            let b1 = t1 + off;
            a1 += q0l * X[b1] + q1l * X[b1+1u] + q2l * X[b1+2u] + q3l * X[b1+3u]
                + q0h * X[b1+32u] + q1h * X[b1+33u] + q2h * X[b1+34u] + q3h * X[b1+35u];
            let b2 = t2 + off;
            a2 += q0l * X[b2] + q1l * X[b2+1u] + q2l * X[b2+2u] + q3l * X[b2+3u]
                + q0h * X[b2+32u] + q1h * X[b2+33u] + q2h * X[b2+34u] + q3h * X[b2+35u];
            let b3 = t3 + off;
            a3 += q0l * X[b3] + q1l * X[b3+1u] + q2l * X[b3+2u] + q3l * X[b3+3u]
                + q0h * X[b3+32u] + q1h * X[b3+33u] + q2h * X[b3+34u] + q3h * X[b3+35u];
        }
    }
    /*REDUCE_ROW_VEC4*/
    if (tid == 0u && row < total_rows) {
        let bv = select(0.0, B[row], P.has_bias != 0u);
        let res = row_sum4 + vec4<f32>(bv);
        if (row < P.q_dim) {
            if (t_base < P.batch) { Q[t_base * P.q_dim + row] = res.x; }
            if (t_base + 1u < P.batch) { Q[(t_base + 1u) * P.q_dim + row] = res.y; }
            if (t_base + 2u < P.batch) { Q[(t_base + 2u) * P.q_dim + row] = res.z; }
            if (t_base + 3u < P.batch) { Q[(t_base + 3u) * P.q_dim + row] = res.w; }
        } else if (row < P.q_dim + P.kv_dim) {
            let kr = row - P.q_dim;
            if (t_base < P.batch) { K[t_base * P.kv_dim + kr] = res.x; }
            if (t_base + 1u < P.batch) { K[(t_base + 1u) * P.kv_dim + kr] = res.y; }
            if (t_base + 2u < P.batch) { K[(t_base + 2u) * P.kv_dim + kr] = res.z; }
            if (t_base + 3u < P.batch) { K[(t_base + 3u) * P.kv_dim + kr] = res.w; }
        } else {
            let vr = row - P.q_dim - P.kv_dim;
            if (t_base < P.batch) { V[t_base * P.kv_dim + vr] = res.x; }
            if (t_base + 1u < P.batch) { V[(t_base + 1u) * P.kv_dim + vr] = res.y; }
            if (t_base + 2u < P.batch) { V[(t_base + 2u) * P.kv_dim + vr] = res.z; }
            if (t_base + 3u < P.batch) { V[(t_base + 3u) * P.kv_dim + vr] = res.w; }
        }
    }
}
"#;

/// Fused `Q4_K` **Gate + Up + SwiGLU** (1 dispatch instead of 3, reads `X` once, 0 `hb2` traffic!).
pub const SHADER_MATMUL_Q4_K_GATE_UP_SWIGLU: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> WG: array<u32>;
@group(0) @binding(1) var<storage, read> WU: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

var<workgroup> smem2: array<vec2<f32>, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var acc_g = 0.0;
    var acc_u = 0.0;
    if (row < P.rows && tok < P.batch) {
        let row_base = row * nb * 36u;
        let x_tok = tok * P.cols;
        let is_base = chunk * 2u;
        let x_chunk_off = chunk * 64u + k4 * 4u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 36u;
            let xb = x_tok + b * 256u + x_chunk_off;

            // Load X ONCE into registers and reuse for both Gate and Up!
            let xl0 = X[xb];      let xh0 = X[xb + 32u];
            let xl1 = X[xb + 1u]; let xh1 = X[xb + 33u];
            let xl2 = X[xb + 2u]; let xh2 = X[xb + 34u];
            let xl3 = X[xb + 3u]; let xh3 = X[xb + 35u];
            let sx1 = xl0 + xl1 + xl2 + xl3;
            let sx2 = xh0 + xh1 + xh2 + xh3;

            // Gate projection
            let dmg = unpack2x16float(WG[blk]);
            let sg0 = WG[blk + 1u]; let sg1 = WG[blk + 2u]; let sg2 = WG[blk + 3u];
            let smg1 = scale_min_k4(is_base, sg0, sg1, sg2);
            let smg2 = scale_min_k4(is_base + 1u, sg0, sg1, sg2);
            let qg = WG[blk + 4u + tid];
            let gb0 = qg & 0xFFu; let gb1 = (qg >> 8u) & 0xFFu;
            let gb2 = (qg >> 16u) & 0xFFu; let gb3 = (qg >> 24u) & 0xFFu;
            let gsq1 = f32(gb0 & 0xFu) * xl0 + f32(gb1 & 0xFu) * xl1 + f32(gb2 & 0xFu) * xl2 + f32(gb3 & 0xFu) * xl3;
            let gsq2 = f32(gb0 >> 4u) * xh0 + f32(gb1 >> 4u) * xh1 + f32(gb2 >> 4u) * xh2 + f32(gb3 >> 4u) * xh3;
            acc_g += (dmg.x * f32(smg1.x)) * gsq1 - (dmg.y * f32(smg1.y)) * sx1
                   + (dmg.x * f32(smg2.x)) * gsq2 - (dmg.y * f32(smg2.y)) * sx2;

            // Up projection
            let dmu = unpack2x16float(WU[blk]);
            let su0 = WU[blk + 1u]; let su1 = WU[blk + 2u]; let su2 = WU[blk + 3u];
            let smu1 = scale_min_k4(is_base, su0, su1, su2);
            let smu2 = scale_min_k4(is_base + 1u, su0, su1, su2);
            let qu = WU[blk + 4u + tid];
            let ub0 = qu & 0xFFu; let ub1 = (qu >> 8u) & 0xFFu;
            let ub2 = (qu >> 16u) & 0xFFu; let ub3 = (qu >> 24u) & 0xFFu;
            let usq1 = f32(ub0 & 0xFu) * xl0 + f32(ub1 & 0xFu) * xl1 + f32(ub2 & 0xFu) * xl2 + f32(ub3 & 0xFu) * xl3;
            let usq2 = f32(ub0 >> 4u) * xh0 + f32(ub1 >> 4u) * xh1 + f32(ub2 >> 4u) * xh2 + f32(ub3 >> 4u) * xh3;
            acc_u += (dmu.x * f32(smu1.x)) * usq1 - (dmu.y * f32(smu1.y)) * sx1
                   + (dmu.x * f32(smu2.x)) * usq2 - (dmu.y * f32(smu2.y)) * sx2;
        }
    }
    /*REDUCE_ROW_VEC2*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        let g = row_sum2.x;
        let u = row_sum2.y;
        OUT[tok * P.rows + row] = (g / (1.0 + exp(-g))) * u;
    }
}
"#;

/// 4-token weight-reuse `Q4_K × F32` Prefill GEMM.
pub const SHADER_MATMUL_Q4_K_B4: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem4: array<vec4<f32>, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let t_base = wid.y * 4u;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < P.rows && t_base < P.batch) {
        let row_base = row * nb * 36u;
        let is_base = chunk * 2u;
        let x_chunk_off = chunk * 64u + k4 * 4u;
        let t0 = t_base * P.cols;
        let t1 = min(t_base + 1u, P.batch - 1u) * P.cols;
        let t2 = min(t_base + 2u, P.batch - 1u) * P.cols;
        let t3 = min(t_base + 3u, P.batch - 1u) * P.cols;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 36u;
            let dm = unpack2x16float(W[blk]);
            let s0 = W[blk + 1u];
            let s1 = W[blk + 2u];
            let s2 = W[blk + 3u];
            let sm1 = scale_min_k4(is_base, s0, s1, s2);
            let sm2 = scale_min_k4(is_base + 1u, s0, s1, s2);
            let d1 = dm.x * f32(sm1.x);
            let m1 = dm.y * f32(sm1.y);
            let d2 = dm.x * f32(sm2.x);
            let m2 = dm.y * f32(sm2.y);

            let qw = W[blk + 4u + tid];
            let q0l = f32(qw & 0xFu) * d1 - m1;
            let q1l = f32((qw >> 8u) & 0xFu) * d1 - m1;
            let q2l = f32((qw >> 16u) & 0xFu) * d1 - m1;
            let q3l = f32((qw >> 24u) & 0xFu) * d1 - m1;
            let q0h = f32((qw >> 4u) & 0xFu) * d2 - m2;
            let q1h = f32((qw >> 12u) & 0xFu) * d2 - m2;
            let q2h = f32((qw >> 20u) & 0xFu) * d2 - m2;
            let q3h = f32((qw >> 28u) & 0xFu) * d2 - m2;

            let off = b * 256u + x_chunk_off;
            let b0 = t0 + off;
            a0 += q0l * X[b0] + q1l * X[b0+1u] + q2l * X[b0+2u] + q3l * X[b0+3u]
                + q0h * X[b0+32u] + q1h * X[b0+33u] + q2h * X[b0+34u] + q3h * X[b0+35u];
            let b1 = t1 + off;
            a1 += q0l * X[b1] + q1l * X[b1+1u] + q2l * X[b1+2u] + q3l * X[b1+3u]
                + q0h * X[b1+32u] + q1h * X[b1+33u] + q2h * X[b1+34u] + q3h * X[b1+35u];
            let b2 = t2 + off;
            a2 += q0l * X[b2] + q1l * X[b2+1u] + q2l * X[b2+2u] + q3l * X[b2+3u]
                + q0h * X[b2+32u] + q1h * X[b2+33u] + q2h * X[b2+34u] + q3h * X[b2+35u];
            let b3 = t3 + off;
            a3 += q0l * X[b3] + q1l * X[b3+1u] + q2l * X[b3+2u] + q3l * X[b3+3u]
                + q0h * X[b3+32u] + q1h * X[b3+33u] + q2h * X[b3+34u] + q3h * X[b3+35u];
        }
    }
    /*REDUCE_ROW_VEC4*/
    if (tid == 0u && row < P.rows) {
        if (t_base < P.batch) { OUT[t_base * P.rows + row] = row_sum4.x; }
        if (t_base + 1u < P.batch) { OUT[(t_base + 1u) * P.rows + row] = row_sum4.y; }
        if (t_base + 2u < P.batch) { OUT[(t_base + 2u) * P.rows + row] = row_sum4.z; }
        if (t_base + 3u < P.batch) { OUT[(t_base + 3u) * P.rows + row] = row_sum4.w; }
    }
}
"#;

/// Quantized `Q5_K × F32` GEMV/GEMM (`44 u32` = `176` bytes per 256-elem super-block).
pub const SHADER_MATMUL_Q5_K: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 128>;

fn scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    if (j < 4u) {
        let sh = j * 8u;
        return vec2<u32>((s0 >> sh) & 63u, (s1 >> sh) & 63u);
    } else {
        let k = j - 4u;
        let sh = k * 8u;
        let q8 = (s2 >> sh) & 0xFFu;
        let q0 = (s0 >> sh) & 0xFFu;
        let q4 = (s1 >> sh) & 0xFFu;
        return vec2<u32>((q8 & 0xFu) | ((q0 >> 6u) << 4u), (q8 >> 4u) | ((q4 >> 6u) << 4u));
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let row_base = row * nb * 44u;
        let x_tok = tok * P.cols;
        let is_base = chunk * 2u;
        let u1 = 1u << (chunk * 2u);
        let u2 = u1 << 1u;
        let x_chunk_off = chunk * 64u + k4 * 4u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 44u;
            let dm = unpack2x16float(W[blk]);
            let s0 = W[blk + 1u];
            let s1 = W[blk + 2u];
            let s2 = W[blk + 3u];
            let sm1 = scale_min_k4(is_base, s0, s1, s2);
            let sm2 = scale_min_k4(is_base + 1u, s0, s1, s2);

            let qhw = W[blk + 4u + k4];
            let qw = W[blk + 12u + tid];
            let xb = x_tok + b * 256u + x_chunk_off;

            var sq1 = 0.0;
            var sx1 = 0.0;
            var sq2 = 0.0;
            var sx2 = 0.0;
            for (var j = 0u; j < 4u; j++) {
                let qb = (qw >> (j * 8u)) & 0xFFu;
                let hb = (qhw >> (j * 8u)) & 0xFFu;
                let hi1 = select(0u, 16u, (hb & u1) != 0u);
                let hi2 = select(0u, 16u, (hb & u2) != 0u);
                let xl = X[xb + j];
                let xh = X[xb + 32u + j];
                sq1 += f32((qb & 0xFu) + hi1) * xl;
                sx1 += xl;
                sq2 += f32((qb >> 4u) + hi2) * xh;
                sx2 += xh;
            }
            acc += (dm.x * f32(sm1.x)) * sq1 - (dm.y * f32(sm1.y)) * sx1
                 + (dm.x * f32(sm2.x)) * sq2 - (dm.y * f32(sm2.y)) * sx2;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// **4-Byte-Aligned `Q6_K × F32` GEMV** (`53 u32` = `212` bytes per 256-elem block).
/// Layout per block: `[0..32]` = `ql[128]`, `[32..48]` = `qh[64]`, `[48..52]` = `scales[16]`,
/// `[52]` = `bitcast<u32>(d_f32)`.
/// Each of 32 threads handles 8 weights with 100% aligned `u32` loads, zero `half` loop,
/// hardware `extractBits` signed scale decode, and `subgroupAdd` reduction!
pub const SHADER_MATMUL_Q6_K: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;

    let half = tid >> 4u;          // 0 or 1
    let t16 = tid & 15u;           // 0..15
    let l_base = t16 * 2u;         // 0, 2, ..., 30
    let w_idx = t16 >> 1u;         // 0..7
    let b_sh = (t16 & 1u) * 16u;   // 0 or 16
    let is_sh = (t16 >> 3u) * 8u;  // 0 (l=0..15) or 8 (l=16..31)

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let row_base = row * nb * 53u;
        let x_tok = tok * P.cols + half * 128u + l_base;
        let ql0_off = half * 16u + w_idx;
        let ql1_off = ql0_off + 8u;
        let qh_off = 32u + half * 8u + w_idx;
        let sc0_off = 48u + half * 2u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 53u;
            let d = bitcast<f32>(W[blk + 52u]);
            let sc_w0 = i32(W[blk + sc0_off]);
            let sc_w1 = i32(W[blk + sc0_off + 1u]);
            let s0 = f32(extractBits(sc_w0, is_sh, 8u));
            let s2 = f32(extractBits(sc_w0, is_sh + 16u, 8u));
            let s4 = f32(extractBits(sc_w1, is_sh, 8u));
            let s6 = f32(extractBits(sc_w1, is_sh + 16u, 8u));

            let ql0_p = (W[blk + ql0_off] >> b_sh) & 0xFFFFu;
            let ql1_p = (W[blk + ql1_off] >> b_sh) & 0xFFFFu;
            let qh_p = (W[blk + qh_off] >> b_sh) & 0xFFFFu;

            let ql0a = ql0_p & 0xFFu; let ql0b = ql0_p >> 8u;
            let ql1a = ql1_p & 0xFFu; let ql1b = ql1_p >> 8u;
            let qha = qh_p & 0xFFu;   let qhb = qh_p >> 8u;

            let xb = x_tok + b * 256u;

            let q1a = f32(i32((ql0a & 0xFu) | ((qha & 3u) << 4u)) - 32);
            let q2a = f32(i32((ql1a & 0xFu) | (((qha >> 2u) & 3u) << 4u)) - 32);
            let q3a = f32(i32((ql0a >> 4u) | (((qha >> 4u) & 3u) << 4u)) - 32);
            let q4a = f32(i32((ql1a >> 4u) | (((qha >> 6u) & 3u) << 4u)) - 32);

            let q1b = f32(i32((ql0b & 0xFu) | ((qhb & 3u) << 4u)) - 32);
            let q2b = f32(i32((ql1b & 0xFu) | (((qhb >> 2u) & 3u) << 4u)) - 32);
            let q3b = f32(i32((ql0b >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32);
            let q4b = f32(i32((ql1b >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32);

            let bsum = s0 * (q1a * X[xb] + q1b * X[xb + 1u])
                     + s2 * (q2a * X[xb + 32u] + q2b * X[xb + 33u])
                     + s4 * (q3a * X[xb + 64u] + q3b * X[xb + 65u])
                     + s6 * (q4a * X[xb + 96u] + q4b * X[xb + 97u]);
            acc += d * bsum;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// 4-token weight-reuse **4-Byte-Aligned `Q6_K × F32` Prefill GEMM** (`53 u32`/block).
pub const SHADER_MATMUL_Q6_K_B4: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem4: array<vec4<f32>, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let t_base = wid.y * 4u;
    let nb = P.cols >> 8u;

    let half = tid >> 4u;
    let t16 = tid & 15u;
    let l_base = t16 * 2u;
    let w_idx = t16 >> 1u;
    let b_sh = (t16 & 1u) * 16u;
    let is_sh = (t16 >> 3u) * 8u;

    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < P.rows && t_base < P.batch) {
        let row_base = row * nb * 53u;
        let x_half = half * 128u + l_base;
        let t0 = t_base * P.cols + x_half;
        let t1 = min(t_base + 1u, P.batch - 1u) * P.cols + x_half;
        let t2 = min(t_base + 2u, P.batch - 1u) * P.cols + x_half;
        let t3 = min(t_base + 3u, P.batch - 1u) * P.cols + x_half;

        let ql0_off = half * 16u + w_idx;
        let ql1_off = ql0_off + 8u;
        let qh_off = 32u + half * 8u + w_idx;
        let sc0_off = 48u + half * 2u;

        for (var b = 0u; b < nb; b++) {
            let blk = row_base + b * 53u;
            let d = bitcast<f32>(W[blk + 52u]);
            let sc_w0 = i32(W[blk + sc0_off]);
            let sc_w1 = i32(W[blk + sc0_off + 1u]);
            let s0 = d * f32(extractBits(sc_w0, is_sh, 8u));
            let s2 = d * f32(extractBits(sc_w0, is_sh + 16u, 8u));
            let s4 = d * f32(extractBits(sc_w1, is_sh, 8u));
            let s6 = d * f32(extractBits(sc_w1, is_sh + 16u, 8u));

            let ql0_p = (W[blk + ql0_off] >> b_sh) & 0xFFFFu;
            let ql1_p = (W[blk + ql1_off] >> b_sh) & 0xFFFFu;
            let qh_p = (W[blk + qh_off] >> b_sh) & 0xFFFFu;

            let ql0a = ql0_p & 0xFFu; let ql0b = ql0_p >> 8u;
            let ql1a = ql1_p & 0xFFu; let ql1b = ql1_p >> 8u;
            let qha = qh_p & 0xFFu;   let qhb = qh_p >> 8u;

            let w1a = s0 * f32(i32((ql0a & 0xFu) | ((qha & 3u) << 4u)) - 32);
            let w2a = s2 * f32(i32((ql1a & 0xFu) | (((qha >> 2u) & 3u) << 4u)) - 32);
            let w3a = s4 * f32(i32((ql0a >> 4u) | (((qha >> 4u) & 3u) << 4u)) - 32);
            let w4a = s6 * f32(i32((ql1a >> 4u) | (((qha >> 6u) & 3u) << 4u)) - 32);

            let w1b = s0 * f32(i32((ql0b & 0xFu) | ((qhb & 3u) << 4u)) - 32);
            let w2b = s2 * f32(i32((ql1b & 0xFu) | (((qhb >> 2u) & 3u) << 4u)) - 32);
            let w3b = s4 * f32(i32((ql0b >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32);
            let w4b = s6 * f32(i32((ql1b >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32);

            let boff = b * 256u;
            let x0 = t0 + boff;
            a0 += w1a * X[x0] + w1b * X[x0+1u] + w2a * X[x0+32u] + w2b * X[x0+33u]
                + w3a * X[x0+64u] + w3b * X[x0+65u] + w4a * X[x0+96u] + w4b * X[x0+97u];
            let x1 = t1 + boff;
            a1 += w1a * X[x1] + w1b * X[x1+1u] + w2a * X[x1+32u] + w2b * X[x1+33u]
                + w3a * X[x1+64u] + w3b * X[x1+65u] + w4a * X[x1+96u] + w4b * X[x1+97u];
            let x2 = t2 + boff;
            a2 += w1a * X[x2] + w1b * X[x2+1u] + w2a * X[x2+32u] + w2b * X[x2+33u]
                + w3a * X[x2+64u] + w3b * X[x2+65u] + w4a * X[x2+96u] + w4b * X[x2+97u];
            let x3 = t3 + boff;
            a3 += w1a * X[x3] + w1b * X[x3+1u] + w2a * X[x3+32u] + w2b * X[x3+33u]
                + w3a * X[x3+64u] + w3b * X[x3+65u] + w4a * X[x3+96u] + w4b * X[x3+97u];
        }
    }
    /*REDUCE_ROW_VEC4*/
    if (tid == 0u && row < P.rows) {
        if (t_base < P.batch) { OUT[t_base * P.rows + row] = row_sum4.x; }
        if (t_base + 1u < P.batch) { OUT[(t_base + 1u) * P.rows + row] = row_sum4.y; }
        if (t_base + 2u < P.batch) { OUT[(t_base + 2u) * P.rows + row] = row_sum4.z; }
        if (t_base + 3u < P.batch) { OUT[(t_base + 3u) * P.rows + row] = row_sum4.w; }
    }
}
"#;

/// Quantized `Q4_0 × F32` GEMV/GEMM (`18` bytes per 32-elem block).
pub const SHADER_MATMUL_Q4_0: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

fn load_u8(off: u32) -> u32 { return (W[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn load_f16_even(off: u32) -> f32 {
    let w = W[off >> 2u];
    return unpack2x16float(select(w & 0xFFFFu, w >> 16u, (off & 2u) != 0u)).x;
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 5u;

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let x_tok = tok * P.cols;
        let row_byte = row * nb * 18u;
        for (var b = tid; b < nb; b += 32u) {
            let off = row_byte + b * 18u;
            let d = load_f16_even(off);
            let xb = x_tok + b * 32u;
            var s = 0.0;
            for (var j = 0u; j < 16u; j++) {
                let q = load_u8(off + 2u + j);
                s += f32(i32(q & 0xFu) - 8) * X[xb + j]
                   + f32(i32(q >> 4u) - 8) * X[xb + 16u + j];
            }
            acc += d * s;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// Quantized `Q8_0 × F32` GEMV/GEMM (`34` bytes per 32-elem block).
pub const SHADER_MATMUL_Q8_0: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

fn load_i8(off: u32) -> f32 {
    let v = (W[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
    return f32(select(i32(v), i32(v) - 256, v >= 128u));
}
fn load_f16_even(off: u32) -> f32 {
    let w = W[off >> 2u];
    return unpack2x16float(select(w & 0xFFFFu, w >> 16u, (off & 2u) != 0u)).x;
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 5u;

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let x_tok = tok * P.cols;
        let row_byte = row * nb * 34u;
        for (var b = tid; b < nb; b += 32u) {
            let off = row_byte + b * 34u;
            let d = load_f16_even(off);
            let xb = x_tok + b * 32u;
            var s = 0.0;
            for (var j = 0u; j < 32u; j++) {
                s += load_i8(off + 2u + j) * X[xb + j];
            }
            acc += d * s;
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// Unquantized `F32 × F32` GEMV/GEMM.
pub const SHADER_MATMUL_F32: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let w_off = row * P.cols;
        let x_off = tok * P.cols;
        for (var c = tid; c < P.cols; c += 32u) {
            acc += W[w_off + c] * X[x_off + c];
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// Unquantized `F16 × F32` GEMV/GEMM.
pub const SHADER_MATMUL_F16: &str = r#"/*SG_ENABLE*/
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x & 31u;
    let r_local = lid.x >> 5u;
    let row = (wid.z * 32768u + wid.x) * 4u + r_local;
    let tok = wid.y;
    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let half_cols = P.cols >> 1u;
        let w_off = row * half_cols;
        let x_off = tok * P.cols;
        for (var c = tid; c < half_cols; c += 32u) {
            let v = unpack2x16float(W[w_off + c]);
            acc += v.x * X[x_off + c * 2u] + v.y * X[x_off + c * 2u + 1u];
        }
    }
    /*REDUCE_ROW_F32*/
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = row_sum;
    }
}
"#;

/// Parallel RMSNorm (`out = rms_norm(x, weight, eps)`), 1 workgroup per token.
pub const SHADER_RMS_NORM: &str = r#"/*SG_ENABLE*/
struct Params { dim: u32, count: u32, eps: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= P.count) { return; }
    let base = row * P.dim;

    var ss = 0.0;
    for (var i = tid; i < P.dim; i += 64u) {
        let v = X[base + i];
        ss += v * v;
    }
    /*REDUCE_WG64_SUM*/
    let inv = inverseSqrt(wg_sum / f32(P.dim) + P.eps);
    for (var i = tid; i < P.dim; i += 64u) {
        OUT[base + i] = X[base + i] * inv * W[i];
    }
}
"#;

/// Fused **Residual Add + Next RMSNorm** (`X += DX; OUT = rms_norm(X, W, eps)`).
/// Eliminates 2 dispatches & 2 barriers per Transformer layer and halves `X` VRAM traffic!
pub const SHADER_RESIDUAL_RMS_NORM: &str = r#"/*SG_ENABLE*/
struct Params { dim: u32, count: u32, eps: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> DX: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= P.count) { return; }
    let base = row * P.dim;

    var ss = 0.0;
    for (var i = tid; i < P.dim; i += 64u) {
        let v = X[base + i] + DX[base + i];
        X[base + i] = v;
        ss += v * v;
    }
    /*REDUCE_WG64_SUM*/
    let inv = inverseSqrt(wg_sum / f32(P.dim) + P.eps);
    for (var i = tid; i < P.dim; i += 64u) {
        OUT[base + i] = X[base + i] * inv * W[i];
    }
}
"#;

/// In-place per-head RMSNorm (`Qwen3` QK-Norm without buffer aliasing).
pub const SHADER_RMS_NORM_INPLACE: &str = r#"/*SG_ENABLE*/
struct Params { dim: u32, count: u32, eps: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= P.count) { return; }
    let base = row * P.dim;

    var ss = 0.0;
    for (var i = tid; i < P.dim; i += 64u) {
        let v = X[base + i];
        ss += v * v;
    }
    /*REDUCE_WG64_SUM*/
    let inv = inverseSqrt(wg_sum / f32(P.dim) + P.eps);
    for (var i = tid; i < P.dim; i += 64u) {
        X[base + i] = X[base + i] * inv * W[i];
    }
}
"#;

/// In-place bias addition `X[t * dim + i] += B[i]` (fallback when QKV is not fused Q4_K).
pub const SHADER_ADD_BIAS: &str = r#"
struct Params { dim: u32, batch: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = P.dim * P.batch;
    let idx = gid.x;
    if (idx < total) {
        X[idx] += B[idx % P.dim];
    }
}
"#;

/// Fused Normal/NeoX RoPE (using precomputed LUT) + **Packed F16 VRAM KV Cache Store**
/// (`K_CACHE` / `V_CACHE` stored as `array<u32>` with `pack2x16float` — 2× less VRAM & bandwidth!).
pub const SHADER_ROPE_KV_STORE_F16: &str = r#"
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    half_rope: u32,
    start_pos: u32,
    batch: u32,
    layer_kv_u32_offset: u32,
    is_neox: u32,
}

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> COS_SIN: array<f32>;
@group(0) @binding(4) var<storage, read_write> K_CACHE: array<u32>;
@group(0) @binding(5) var<storage, read_write> V_CACHE: array<u32>;
@group(0) @binding(6) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let tok = gid.y;
    if (tok >= P.batch) { return; }

    let pos = P.start_pos + tok;
    let q_dim = P.num_heads * P.head_dim;
    let kv_dim = P.num_kv_heads * P.head_dim;
    let hd_u32 = P.head_dim >> 1u;
    let kv_dim_u32 = kv_dim >> 1u;
    let lut_stride = P.half_rope * 2u;
    let lut_base = pos * lut_stride;

    // 1. Apply RoPE to Q heads (F32)
    let q_pairs = P.num_heads * P.half_rope;
    if (tid < q_pairs) {
        let h = tid / P.half_rope;
        let i = tid % P.half_rope;
        let c = COS_SIN[lut_base + i];
        let s = COS_SIN[lut_base + P.half_rope + i];
        let h_off = tok * q_dim + h * P.head_dim;
        if (P.is_neox != 0u) {
            let x0 = Q[h_off + i];
            let x1 = Q[h_off + i + P.half_rope];
            Q[h_off + i] = x0 * c - x1 * s;
            Q[h_off + i + P.half_rope] = x0 * s + x1 * c;
        } else {
            let x0 = Q[h_off + 2u * i];
            let x1 = Q[h_off + 2u * i + 1u];
            Q[h_off + 2u * i] = x0 * c - x1 * s;
            Q[h_off + 2u * i + 1u] = x0 * s + x1 * c;
        }
    }

    // 2. Apply RoPE to K heads & store packed F16 K, V into VRAM KV cache (100% race-free!)
    if (P.is_neox != 0u) {
        let half_rope_u32 = P.half_rope >> 1u;
        let k_u32_jobs = P.num_kv_heads * half_rope_u32;
        if (tid < k_u32_jobs) {
            let kh = tid / half_rope_u32;
            let p = tid % half_rope_u32;
            let i0 = p * 2u;
            let i1 = i0 + 1u;
            let c0 = COS_SIN[lut_base + i0];
            let s0 = COS_SIN[lut_base + P.half_rope + i0];
            let c1 = COS_SIN[lut_base + i1];
            let s1 = COS_SIN[lut_base + P.half_rope + i1];

            let kh_off = tok * kv_dim + kh * P.head_dim;
            let x0a = K[kh_off + i0];
            let x1a = K[kh_off + i0 + P.half_rope];
            let r0a = x0a * c0 - x1a * s0;
            let r1a = x0a * s0 + x1a * c0;

            let x0b = K[kh_off + i1];
            let x1b = K[kh_off + i1 + P.half_rope];
            let r0b = x0b * c1 - x1b * s1;
            let r1b = x0b * s1 + x1b * c1;

            let cache_u32 = P.layer_kv_u32_offset + pos * kv_dim_u32 + kh * hd_u32;
            K_CACHE[cache_u32 + p] = pack2x16float(vec2<f32>(r0a, r0b));
            K_CACHE[cache_u32 + p + half_rope_u32] = pack2x16float(vec2<f32>(r1a, r1b));
            V_CACHE[cache_u32 + p] = pack2x16float(vec2<f32>(V[kh_off + i0], V[kh_off + i1]));
            V_CACHE[cache_u32 + p + half_rope_u32] = pack2x16float(vec2<f32>(
                V[kh_off + i0 + P.half_rope],
                V[kh_off + i1 + P.half_rope],
            ));
        }
    } else {
        let k_pairs = P.num_kv_heads * P.half_rope;
        if (tid < k_pairs) {
            let kh = tid / P.half_rope;
            let i = tid % P.half_rope;
            let c = COS_SIN[lut_base + i];
            let s = COS_SIN[lut_base + P.half_rope + i];
            let kh_off = tok * kv_dim + kh * P.head_dim;
            let x0 = K[kh_off + 2u * i];
            let x1 = K[kh_off + 2u * i + 1u];
            let r0 = x0 * c - x1 * s;
            let r1 = x0 * s + x1 * c;
            let cache_u32 = P.layer_kv_u32_offset + pos * kv_dim_u32 + kh * hd_u32;
            K_CACHE[cache_u32 + i] = pack2x16float(vec2<f32>(r0, r1));
            V_CACHE[cache_u32 + i] = pack2x16float(vec2<f32>(V[kh_off + 2u * i], V[kh_off + 2u * i + 1u]));
        }
    }

    // 3. Copy any non-RoPE trailing head dims into packed F16 K/V cache
    let rope_dim = P.half_rope * 2u;
    if (rope_dim < P.head_dim && tid < kv_dim_u32) {
        let kh = tid / hd_u32;
        let d2 = tid % hd_u32;
        if (d2 >= P.half_rope) {
            let src = tok * kv_dim + kh * P.head_dim + d2 * 2u;
            let dst = P.layer_kv_u32_offset + pos * kv_dim_u32 + kh * hd_u32 + d2;
            K_CACHE[dst] = pack2x16float(vec2<f32>(K[src], K[src + 1u]));
            V_CACHE[dst] = pack2x16float(vec2<f32>(V[src], V[src + 1u]));
        }
    }
}
"#;

/// Online Tiled Flash-Attention GQA over **Packed F16 VRAM KV Cache** with Subgroup Reductions!
/// Supports `head_dim <= 512` (including Gemma 4 full layers) and optional `sliding_window`.
pub const SHADER_GQA_FLASH_ATTN_F16: &str = r#"/*SG_ENABLE*/
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    batch: u32,
    layer_kv_u32_offset: u32,
    inv_sqrt_hd: f32,
    sliding_window: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_CACHE: array<u32>;
@group(0) @binding(2) var<storage, read> V_CACHE: array<u32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

var<workgroup> q_sh: array<f32, 512>;
var<workgroup> out_sh: array<f32, 512>;
var<workgroup> scores: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let h = wid.x;
    let tok = wid.y;
    if (h >= P.num_heads || tok >= P.batch) { return; }

    let hd = P.head_dim;
    let hd_u32 = hd >> 1u;
    let q_dim = P.num_heads * hd;
    let kv_dim_u32 = (P.num_kv_heads * hd) >> 1u;
    let kv_mul = P.num_heads / P.num_kv_heads;
    let kv_h = h / kv_mul;
    let pos = P.start_pos + tok;
    let seq_len = pos + 1u;
    let t_start = select(
        0u,
        select(0u, seq_len - P.sliding_window, seq_len > P.sliding_window),
        P.sliding_window > 0u
    );
    let q_off = tok * q_dim + h * hd;

    for (var d = tid; d < hd; d += 64u) {
        q_sh[d] = Q[q_off + d];
        out_sh[d] = 0.0;
    }
    workgroupBarrier();

    var running_max = -1e30;
    var running_sum = 0.0;

    for (var tile_start = t_start; tile_start < seq_len; tile_start += 64u) {
        let tile_end = min(64u, seq_len - tile_start);
        let s = tile_start + tid;
        var sc = -1e30;
        if (s < seq_len) {
            let k_off = P.layer_kv_u32_offset + s * kv_dim_u32 + kv_h * hd_u32;
            var dot_v = 0.0;
            for (var d2 = 0u; d2 < hd_u32; d2++) {
                let kp = unpack2x16float(K_CACHE[k_off + d2]);
                dot_v += q_sh[d2 * 2u] * kp.x + q_sh[d2 * 2u + 1u] * kp.y;
            }
            sc = dot_v * P.inv_sqrt_hd;
        }
        /*REDUCE_WG64_MAX*/

        let new_max = max(running_max, tile_max);
        let rescale = select(1.0, exp(running_max - new_max), tile_start > t_start);
        let w = select(0.0, exp(sc - new_max), tid < tile_end);
        scores[tid] = w;
        /*REDUCE_WG64_WSUM*/

        for (var d2 = tid; d2 < hd_u32; d2 += 64u) {
            var a0 = out_sh[d2 * 2u] * rescale;
            var a1 = out_sh[d2 * 2u + 1u] * rescale;
            for (var k = 0u; k < tile_end; k++) {
                let v_off = P.layer_kv_u32_offset + (tile_start + k) * kv_dim_u32 + kv_h * hd_u32 + d2;
                let vp = unpack2x16float(V_CACHE[v_off]);
                let wk = scores[k];
                a0 += wk * vp.x;
                a1 += wk * vp.y;
            }
            out_sh[d2 * 2u] = a0;
            out_sh[d2 * 2u + 1u] = a1;
        }
        running_sum = running_sum * rescale + tile_sum;
        running_max = new_max;
        workgroupBarrier();
    }

    let inv_sum = 1.0 / max(running_sum, 1e-20);
    for (var d = tid; d < hd; d += 64u) {
        OUT[q_off + d] = out_sh[d] * inv_sum;
    }
}
"#;

/// Fused SwiGLU `GATE[i] = (GATE[i] / (1 + exp(-GATE[i]))) * UP[i]`.
pub const SHADER_SWIGLU: &str = r#"
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> GATE: array<f32>;
@group(0) @binding(1) var<storage, read> UP: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < P.len) {
        let g = GATE[i];
        GATE[i] = (g / (1.0 + exp(-g))) * UP[i];
    }
}
"#;

/// Fused GeGLU (Gemma 2/3/4): `GATE[i] = gelu_tanh(GATE[i]) * UP[i]`.
pub const SHADER_GEGLU: &str = r#"
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> GATE: array<f32>;
@group(0) @binding(1) var<storage, read> UP: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < P.len) {
        let g = GATE[i];
        let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
        let gelu = 0.5 * g * (1.0 + tanh(clamp(inner, -10.0, 10.0)));
        GATE[i] = gelu * UP[i];
    }
}
"#;

/// Residual add `X[i] += DX[i]`.
pub const SHADER_RESIDUAL_ADD: &str = r#"
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> DX: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < P.len) {
        X[i] += DX[i];
    }
}
"#;

/// Scaled Residual add (Gemma 4 `layer_output_scale`): `X[i] = (X[i] + DX[i]) * scale`.
pub const SHADER_RESIDUAL_ADD_SCALE: &str = r#"
struct Params { len: u32, scale: f32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> DX: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < P.len) {
        X[i] = (X[i] + DX[i]) * P.scale;
    }
}
"#;

/// In-place Sigmoid Gate multiply (Qwen 3.5 Full Attention): `OUT[i] *= sigmoid(GATE[i])`.
pub const SHADER_MUL_SIGMOID: &str = r#"
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(1) var<storage, read> GATE: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < P.len) {
        OUT[i] *= 1.0 / (1.0 + exp(-GATE[i]));
    }
}
"#;

/// De-interleaves Qwen 3.5 joint `[n_heads, 2, head_dim]` into contiguous `Q` and `GATE`.
pub const SHADER_DEINTERLEAVE_QG: &str = r#"
struct Params { num_heads: u32, head_dim: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> QG: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> GATE: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_dim = P.num_heads * P.head_dim;
    let total = P.batch * q_dim;
    let idx = gid.x;
    if (idx >= total) { return; }
    let tok = idx / q_dim;
    let rem = idx % q_dim;
    let h = rem / P.head_dim;
    let d = rem % P.head_dim;
    let src_base = tok * (q_dim * 2u) + h * (P.head_dim * 2u) + d;
    Q[idx] = QG[src_base];
    GATE[idx] = QG[src_base + P.head_dim];
}
"#;

/// Qwen 3.5 Gated DeltaNet Causal Depthwise Conv1D (`kernel=4`) + SiLU + Conv State Update.
/// One thread per channel `c in 0..channels`, stepping sequentially through `0..batch`.
pub const SHADER_DELTANET_CONV_SILU: &str = r#"
struct Params { channels: u32, conv_kernel: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read> CONV_W: array<f32>;
@group(0) @binding(2) var<storage, read_write> CONV_STATE: array<f32>;
@group(0) @binding(3) var<storage, read_write> CONV_OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= P.channels) { return; }
    let ch = P.channels;
    let k = P.conv_kernel;
    let w_base = c * k;

    // Load history into registers (supports kernel <= 4)
    var h0 = select(0.0, CONV_STATE[0u * ch + c], k > 1u);
    var h1 = select(0.0, CONV_STATE[1u * ch + c], k > 2u);
    var h2 = select(0.0, CONV_STATE[2u * ch + c], k > 3u);

    let w0 = CONV_W[w_base + 0u];
    let w1 = select(0.0, CONV_W[w_base + 1u], k > 1u);
    let w2 = select(0.0, CONV_W[w_base + 2u], k > 2u);
    let w3 = select(0.0, CONV_W[w_base + 3u], k > 3u);

    for (var t = 0u; t < P.batch; t++) {
        let x = QKV[t * ch + c];
        var acc = 0.0;
        if (k == 4u) {
            acc = h0 * w0 + h1 * w1 + h2 * w2 + x * w3;
            h0 = h1;
            h1 = h2;
            h2 = x;
        } else if (k == 3u) {
            acc = h0 * w0 + h1 * w1 + x * w2;
            h0 = h1;
            h1 = x;
        } else if (k == 2u) {
            acc = h0 * w0 + x * w1;
            h0 = x;
        } else {
            acc = x * w0;
        }
        CONV_OUT[t * ch + c] = acc / (1.0 + exp(-acc));
    }

    if (k > 1u) { CONV_STATE[0u * ch + c] = h0; }
    if (k > 2u) { CONV_STATE[1u * ch + c] = h1; }
    if (k > 3u) { CONV_STATE[2u * ch + c] = h2; }
}
"#;

/// In-place L2 Normalization for Q and K slices inside `CONV_OUT` (`2 * num_k_heads` heads per tok).
pub const SHADER_L2_NORM_QK: &str = r#"/*SG_ENABLE*/
struct Params { head_k_dim: u32, num_qk_heads: u32, channels: u32, eps: f32 }
@group(0) @binding(0) var<storage, read_write> CONV_OUT: array<f32>;
@group(0) @binding(1) var<storage, read> _DUMMY: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let qkh = wid.x;
    let tok = wid.y;
    if (qkh >= P.num_qk_heads) { return; }
    let base = tok * P.channels + qkh * P.head_k_dim;

    var ss = 0.0;
    for (var i = tid; i < P.head_k_dim; i += 64u) {
        let v = CONV_OUT[base + i];
        ss += v * v;
    }
    /*REDUCE_WG64_SUM*/
    let inv = inverseSqrt(wg_sum + P.eps);
    for (var i = tid; i < P.head_k_dim; i += 64u) {
        CONV_OUT[base + i] *= inv;
    }
}
"#;

/// Fused Qwen 3.5 Gated DeltaNet Recurrence + Gated RMSNorm (`ssm_norm * silu(z)`).
/// Dispatched with `(num_v_heads, 1, 1)` workgroups of `128` threads (supports `head_v_dim <= 128`).
/// `SSM_PARAMS` packs `[ssm_a (num_v_heads) | ssm_dt_bias (num_v_heads) | ssm_norm (head_v_dim)]`.
pub const SHADER_DELTANET_STEP: &str = r#"
struct Params {
    num_k_heads: u32,
    num_v_heads: u32,
    head_dim: u32,
    channels: u32,
    batch: u32,
    eps: f32,
    scale: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> CONV_OUT: array<f32>;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> ALPHA: array<f32>;
@group(0) @binding(3) var<storage, read> BETA: array<f32>;
@group(0) @binding(4) var<storage, read> SSM_PARAMS: array<f32>;
@group(0) @binding(5) var<storage, read_write> SSM_STATE: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y_OUT: array<f32>;
@group(0) @binding(7) var<uniform> P: Params;

var<workgroup> q_sh: array<f32, 128>;
var<workgroup> k_sh: array<f32, 128>;
var<workgroup> red_ss: array<f32, 128>;

fn softplus_f32(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    if (x < -20.0) { return exp(x); }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let j = lid.x;
    let iv = wid.x;
    if (iv >= P.num_v_heads) { return; }

    let hd = P.head_dim;
    let ik = iv % P.num_k_heads;
    let qk_dim = P.num_k_heads * hd;
    let inner = P.num_v_heads * hd;
    let state_base = iv * hd * hd + j * hd;

    let ssm_a = SSM_PARAMS[iv];
    let dt_bias = SSM_PARAMS[P.num_v_heads + iv];
    let norm_w = select(0.0, SSM_PARAMS[P.num_v_heads * 2u + j], j < hd);

    for (var t = 0u; t < P.batch; t++) {
        let conv_base = t * P.channels;
        if (j < hd) {
            q_sh[j] = CONV_OUT[conv_base + ik * hd + j];
            k_sh[j] = CONV_OUT[conv_base + qk_dim + ik * hd + j];
        }
        workgroupBarrier();

        let beta = 1.0 / (1.0 + exp(-BETA[t * P.num_v_heads + iv]));
        let g = ssm_a * softplus_f32(ALPHA[t * P.num_v_heads + iv] + dt_bias);
        let decay = exp(g);

        var y_val = 0.0;
        if (j < hd) {
            // 1. Decay state row j and compute prediction dot(S_row_j, k)
            var pred = 0.0;
            for (var i = 0u; i < hd; i++) {
                let s_dec = SSM_STATE[state_base + i] * decay;
                SSM_STATE[state_base + i] = s_dec;
                pred += s_dec * k_sh[i];
            }
            // 2. Delta rule rank-1 update + output dot(S_row_j, q) * scale
            let v_j = CONV_OUT[conv_base + qk_dim * 2u + iv * hd + j];
            let delta = (v_j - pred) * beta;
            var acc_y = 0.0;
            for (var i = 0u; i < hd; i++) {
                let s_new = SSM_STATE[state_base + i] + delta * k_sh[i];
                SSM_STATE[state_base + i] = s_new;
                acc_y += s_new * q_sh[i];
            }
            y_val = acc_y * P.scale;
        }

        // 3. Fused Gated RMSNorm over `hd` elements: rms_norm(y, ssm_norm, eps) * silu(z)
        red_ss[j] = select(0.0, y_val * y_val, j < hd);
        workgroupBarrier();
        if (j < 64u) { red_ss[j] += red_ss[j + 64u]; }
        workgroupBarrier();
        if (j < 32u) { red_ss[j] += red_ss[j + 32u]; }
        workgroupBarrier();
        if (j < 16u) { red_ss[j] += red_ss[j + 16u]; }
        workgroupBarrier();
        if (j < 8u) { red_ss[j] += red_ss[j + 8u]; }
        workgroupBarrier();
        if (j < 4u) { red_ss[j] += red_ss[j + 4u]; }
        workgroupBarrier();
        if (j < 2u) { red_ss[j] += red_ss[j + 2u]; }
        workgroupBarrier();
        let inv_rms = inverseSqrt((red_ss[0] + red_ss[1]) / f32(hd) + P.eps);

        if (j < hd) {
            let z_j = Z[t * inner + iv * hd + j];
            let z_silu = z_j / (1.0 + exp(-z_j));
            Y_OUT[t * inner + iv * hd + j] = (y_val * inv_rms * norm_w) * z_silu;
        }
        workgroupBarrier();
    }
}
"#;

/// Multi-Slot Continuous Batching RoPE + Packed F16 KV Cache Store.
/// `SLOT_META` holds `[slot_kv_u32_base, pos]` pairs per batch token `tok in 0..batch`.
pub const SHADER_ROPE_KV_STORE_SLOTS_F16: &str = r#"
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    half_rope: u32,
    batch: u32,
    layer_kv_u32_offset: u32,
    is_neox: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> COS_SIN: array<f32>;
@group(0) @binding(4) var<storage, read_write> K_CACHE: array<u32>;
@group(0) @binding(5) var<storage, read_write> V_CACHE: array<u32>;
@group(0) @binding(6) var<storage, read> SLOT_META: array<u32>;
@group(0) @binding(7) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let tok = gid.y;
    if (tok >= P.batch) { return; }

    let slot_base = SLOT_META[tok * 2u];
    let pos = SLOT_META[tok * 2u + 1u];
    let kv_layer_base = slot_base + P.layer_kv_u32_offset;

    let q_dim = P.num_heads * P.head_dim;
    let kv_dim = P.num_kv_heads * P.head_dim;
    let hd_u32 = P.head_dim >> 1u;
    let kv_dim_u32 = kv_dim >> 1u;
    let lut_stride = P.half_rope * 2u;
    let lut_base = pos * lut_stride;

    // 1. Apply RoPE to Q heads (F32)
    let q_pairs = P.num_heads * P.half_rope;
    if (tid < q_pairs) {
        let h = tid / P.half_rope;
        let i = tid % P.half_rope;
        let c = COS_SIN[lut_base + i];
        let s = COS_SIN[lut_base + P.half_rope + i];
        let h_off = tok * q_dim + h * P.head_dim;
        if (P.is_neox != 0u) {
            let x0 = Q[h_off + i];
            let x1 = Q[h_off + i + P.half_rope];
            Q[h_off + i] = x0 * c - x1 * s;
            Q[h_off + i + P.half_rope] = x0 * s + x1 * c;
        } else {
            let x0 = Q[h_off + 2u * i];
            let x1 = Q[h_off + 2u * i + 1u];
            Q[h_off + 2u * i] = x0 * c - x1 * s;
            Q[h_off + 2u * i + 1u] = x0 * s + x1 * c;
        }
    }

    // 2. Apply RoPE to K heads & store packed F16 K, V into this token's slot KV cache
    if (P.is_neox != 0u) {
        let half_rope_u32 = P.half_rope >> 1u;
        let k_u32_jobs = P.num_kv_heads * half_rope_u32;
        if (tid < k_u32_jobs) {
            let kh = tid / half_rope_u32;
            let p = tid % half_rope_u32;
            let i0 = p * 2u;
            let i1 = i0 + 1u;
            let c0 = COS_SIN[lut_base + i0];
            let s0 = COS_SIN[lut_base + P.half_rope + i0];
            let c1 = COS_SIN[lut_base + i1];
            let s1 = COS_SIN[lut_base + P.half_rope + i1];

            let kh_off = tok * kv_dim + kh * P.head_dim;
            let x0a = K[kh_off + i0];
            let x1a = K[kh_off + i0 + P.half_rope];
            let r0a = x0a * c0 - x1a * s0;
            let r1a = x0a * s0 + x1a * c0;

            let x0b = K[kh_off + i1];
            let x1b = K[kh_off + i1 + P.half_rope];
            let r0b = x0b * c1 - x1b * s1;
            let r1b = x0b * s1 + x1b * c1;

            let cache_u32 = kv_layer_base + pos * kv_dim_u32 + kh * hd_u32;
            K_CACHE[cache_u32 + p] = pack2x16float(vec2<f32>(r0a, r0b));
            K_CACHE[cache_u32 + p + half_rope_u32] = pack2x16float(vec2<f32>(r1a, r1b));
            V_CACHE[cache_u32 + p] = pack2x16float(vec2<f32>(V[kh_off + i0], V[kh_off + i1]));
            V_CACHE[cache_u32 + p + half_rope_u32] = pack2x16float(vec2<f32>(
                V[kh_off + i0 + P.half_rope],
                V[kh_off + i1 + P.half_rope],
            ));
        }
    } else {
        let k_pairs = P.num_kv_heads * P.half_rope;
        if (tid < k_pairs) {
            let kh = tid / P.half_rope;
            let i = tid % P.half_rope;
            let c = COS_SIN[lut_base + i];
            let s = COS_SIN[lut_base + P.half_rope + i];
            let kh_off = tok * kv_dim + kh * P.head_dim;
            let x0 = K[kh_off + 2u * i];
            let x1 = K[kh_off + 2u * i + 1u];
            let r0 = x0 * c - x1 * s;
            let r1 = x0 * s + x1 * c;
            let cache_u32 = kv_layer_base + pos * kv_dim_u32 + kh * hd_u32;
            K_CACHE[cache_u32 + i] = pack2x16float(vec2<f32>(r0, r1));
            V_CACHE[cache_u32 + i] = pack2x16float(vec2<f32>(V[kh_off + 2u * i], V[kh_off + 2u * i + 1u]));
        }
    }

    // 3. Copy any non-RoPE trailing head dims
    let rope_dim = P.half_rope * 2u;
    if (rope_dim < P.head_dim && tid < kv_dim_u32) {
        let kh = tid / hd_u32;
        let d2 = tid % hd_u32;
        if (d2 >= P.half_rope) {
            let src = tok * kv_dim + kh * P.head_dim + d2 * 2u;
            let dst = kv_layer_base + pos * kv_dim_u32 + kh * hd_u32 + d2;
            K_CACHE[dst] = pack2x16float(vec2<f32>(K[src], K[src + 1u]));
            V_CACHE[dst] = pack2x16float(vec2<f32>(V[src], V[src + 1u]));
        }
    }
}
"#;

/// Multi-Slot Continuous Batching Online Tiled Flash-Attention GQA over Packed F16 KV Cache.
/// Each batch token `tok` independently attends to its own slot `SLOT_META[2*tok]` up to `SLOT_META[2*tok+1]`.
pub const SHADER_GQA_FLASH_ATTN_SLOTS_F16: &str = r#"/*SG_ENABLE*/
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    batch: u32,
    layer_kv_u32_offset: u32,
    inv_sqrt_hd: f32,
    sliding_window: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_CACHE: array<u32>;
@group(0) @binding(2) var<storage, read> V_CACHE: array<u32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<storage, read> SLOT_META: array<u32>;
@group(0) @binding(5) var<uniform> P: Params;

var<workgroup> q_sh: array<f32, 512>;
var<workgroup> out_sh: array<f32, 512>;
var<workgroup> scores: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>/*SG_ARGS*/
) {
    let tid = lid.x;
    let h = wid.x;
    let tok = wid.y;
    if (h >= P.num_heads || tok >= P.batch) { return; }

    let slot_base = SLOT_META[tok * 2u];
    let pos = SLOT_META[tok * 2u + 1u];
    let kv_layer_base = slot_base + P.layer_kv_u32_offset;

    let hd = P.head_dim;
    let hd_u32 = hd >> 1u;
    let q_dim = P.num_heads * hd;
    let kv_dim_u32 = (P.num_kv_heads * hd) >> 1u;
    let kv_mul = P.num_heads / P.num_kv_heads;
    let kv_h = h / kv_mul;
    let seq_len = pos + 1u;
    let t_start = select(
        0u,
        select(0u, seq_len - P.sliding_window, seq_len > P.sliding_window),
        P.sliding_window > 0u
    );
    let q_off = tok * q_dim + h * hd;

    for (var d = tid; d < hd; d += 64u) {
        q_sh[d] = Q[q_off + d];
        out_sh[d] = 0.0;
    }
    workgroupBarrier();

    var running_max = -1e30;
    var running_sum = 0.0;

    for (var tile_start = t_start; tile_start < seq_len; tile_start += 64u) {
        let tile_end = min(64u, seq_len - tile_start);
        let s = tile_start + tid;
        var sc = -1e30;
        if (s < seq_len) {
            let k_off = kv_layer_base + s * kv_dim_u32 + kv_h * hd_u32;
            var dot_v = 0.0;
            for (var d2 = 0u; d2 < hd_u32; d2++) {
                let kp = unpack2x16float(K_CACHE[k_off + d2]);
                dot_v += q_sh[d2 * 2u] * kp.x + q_sh[d2 * 2u + 1u] * kp.y;
            }
            sc = dot_v * P.inv_sqrt_hd;
        }
        /*REDUCE_WG64_MAX*/

        let new_max = max(running_max, tile_max);
        let rescale = select(1.0, exp(running_max - new_max), tile_start > t_start);
        let w = select(0.0, exp(sc - new_max), tid < tile_end);
        scores[tid] = w;
        /*REDUCE_WG64_WSUM*/

        for (var d2 = tid; d2 < hd_u32; d2 += 64u) {
            var a0 = out_sh[d2 * 2u] * rescale;
            var a1 = out_sh[d2 * 2u + 1u] * rescale;
            for (var k = 0u; k < tile_end; k++) {
                let v_off = kv_layer_base + (tile_start + k) * kv_dim_u32 + kv_h * hd_u32 + d2;
                let vp = unpack2x16float(V_CACHE[v_off]);
                let wk = scores[k];
                a0 += wk * vp.x;
                a1 += wk * vp.y;
            }
            out_sh[d2 * 2u] = a0;
            out_sh[d2 * 2u + 1u] = a1;
        }
        running_sum = running_sum * rescale + tile_sum;
        running_max = new_max;
        workgroupBarrier();
    }

    let inv_sum = 1.0 / max(running_sum, 1e-20);
    for (var d = tid; d < hd; d += 64u) {
        OUT[q_off + d] = out_sh[d] * inv_sum;
    }
}
"#;
