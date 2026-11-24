//! WGSL Compute Shaders for Caelum's `wgpu` GPU Backend (Vulkan / Metal / DX12 / WebGPU).
//!
//! Includes:
//! - Coalesced 32-thread-per-row quantized GEMV & 4-token GEMM for `Q4_K`, `Q5_K`, `Q6_K`,
//!   `Q4_0`, `Q8_0`, `F16`, and `F32` (exact GGML bit layouts, zero dequantization to VRAM)
//! - On-GPU embedding lookup (`Q4_K`/`Q5_K`/`Q6_K`/`Q4_0`/`Q8_0`/`F16`/`F32`)
//! - Tree-reduction RMSNorm (full hidden + in-place Qwen3 per-head QK-Norm)
//! - Fused QKV bias + RoPE (Normal/NeoX/YaRN LUT) + VRAM KV Cache store
//! - Online Tiled Flash-Attention GQA (zero global scratch buffer, any context length)
//! - SwiGLU & Residual Add

/// Quantized `Q4_K × F32` GEMV/GEMM (`36 u32` = `144` bytes per 256-elem super-block).
/// Dispatch: `workgroups = (ceil(rows / 4), batch, 1)` for B=1..3, or `matmul_q4_k_b4` for B>=4.
pub const SHADER_MATMUL_Q4_K: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    batch: u32,
    _pad: u32,
}

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
        let sc = (q8 & 0xFu) | ((q0 >> 6u) << 4u);
        let mn = (q8 >> 4u) | ((q4 >> 6u) << 4u);
        return vec2<u32>(sc, mn);
    }
}

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
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

            var sq1 = 0.0;
            var sx1 = 0.0;
            var sq2 = 0.0;
            var sx2 = 0.0;

            let b0 = qw & 0xFFu;
            let b1 = (qw >> 8u) & 0xFFu;
            let b2 = (qw >> 16u) & 0xFFu;
            let b3 = (qw >> 24u) & 0xFFu;

            let xl0 = X[xb];      let xh0 = X[xb + 32u];
            let xl1 = X[xb + 1u]; let xh1 = X[xb + 33u];
            let xl2 = X[xb + 2u]; let xh2 = X[xb + 34u];
            let xl3 = X[xb + 3u]; let xh3 = X[xb + 35u];

            sq1 = f32(b0 & 0xFu) * xl0 + f32(b1 & 0xFu) * xl1 + f32(b2 & 0xFu) * xl2 + f32(b3 & 0xFu) * xl3;
            sx1 = xl0 + xl1 + xl2 + xl3;
            sq2 = f32(b0 >> 4u) * xh0 + f32(b1 >> 4u) * xh1 + f32(b2 >> 4u) * xh2 + f32(b3 >> 4u) * xh3;
            sx2 = xh0 + xh1 + xh2 + xh3;

            acc += (dm.x * f32(sm1.x)) * sq1 - (dm.y * f32(sm1.y)) * sx1
                 + (dm.x * f32(sm2.x)) * sq2 - (dm.y * f32(sm2.y)) * sx2;
        }
    }

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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// 4-token weight-reuse `Q4_K × F32` Prefill GEMM (`workgroups = (ceil(rows/4), ceil(batch/4), 1)`).
pub const SHADER_MATMUL_Q4_K_B4: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    batch: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<vec4<f32>, 128>;

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

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
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

    let sidx = r_local * 32u + tid;
    smem[sidx] = vec4<f32>(a0, a1, a2, a3);
    workgroupBarrier();
    if (tid < 16u) { smem[sidx] += smem[sidx + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { smem[sidx] += smem[sidx + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { smem[sidx] += smem[sidx + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { smem[sidx] += smem[sidx + 2u]; }
    workgroupBarrier();
    if (tid == 0u && row < P.rows) {
        let res = smem[sidx] + smem[sidx + 1u];
        if (t_base < P.batch) { OUT[t_base * P.rows + row] = res.x; }
        if (t_base + 1u < P.batch) { OUT[(t_base + 1u) * P.rows + row] = res.y; }
        if (t_base + 2u < P.batch) { OUT[(t_base + 2u) * P.rows + row] = res.z; }
        if (t_base + 3u < P.batch) { OUT[(t_base + 3u) * P.rows + row] = res.w; }
    }
}
"#;

/// Quantized `Q5_K × F32` GEMV/GEMM (`44 u32` = `176` bytes per 256-elem super-block).
pub const SHADER_MATMUL_Q5_K: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    batch: u32,
    _pad: u32,
}

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

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;
    let chunk = tid >> 3u;
    let k4 = tid & 7u;
    let u1 = 1u << (chunk * 2u);
    let u2 = 2u << (chunk * 2u);

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let row_base = row * nb * 44u;
        let x_tok = tok * P.cols;
        let is_base = chunk * 2u;
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

            for (var i = 0u; i < 4u; i++) {
                let sh = i * 8u;
                let qb = (qw >> sh) & 0xFFu;
                let hb = (qhw >> sh) & 0xFFu;
                let q1 = (qb & 0xFu) + select(0u, 16u, (hb & u1) != 0u);
                let q2 = (qb >> 4u) + select(0u, 16u, (hb & u2) != 0u);
                let xl = X[xb + i];
                let xh = X[xb + 32u + i];
                sq1 += f32(q1) * xl;
                sx1 += xl;
                sq2 += f32(q2) * xh;
                sx2 += xh;
            }

            acc += (dm.x * f32(sm1.x)) * sq1 - (dm.y * f32(sm1.y)) * sx1
                 + (dm.x * f32(sm2.x)) * sq2 - (dm.y * f32(sm2.y)) * sx2;
        }
    }

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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// Quantized `Q6_K × F32` GEMV/GEMM (`210` bytes per 256-elem super-block, raw byte layout).
pub const SHADER_MATMUL_Q6_K: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    batch: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 128>;

fn load_u8(byte_off: u32) -> u32 {
    return (W[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

fn load_i8(byte_off: u32) -> f32 {
    let v = load_u8(byte_off);
    return f32(select(i32(v), i32(v) - 256, v >= 128u));
}

fn load_f16_even(byte_off: u32) -> f32 {
    let w = W[byte_off >> 2u];
    let u = select(w & 0xFFFFu, w >> 16u, (byte_off & 2u) != 0u);
    return unpack2x16float(u).x;
}

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
    let tok = wid.y;
    let nb = P.cols >> 8u;
    let is_idx = tid >> 4u;

    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let x_tok = tok * P.cols;
        let row_byte = row * nb * 210u;

        for (var b = 0u; b < nb; b++) {
            let off = row_byte + b * 210u;
            let d = load_f16_even(off + 208u);
            var bsum = 0.0;

            for (var half = 0u; half < 2u; half++) {
                let ql_h = off + half * 64u;
                let qh_h = off + 128u + half * 32u;
                let sc_h = off + 192u + half * 8u;
                let x_h = x_tok + b * 256u + half * 128u + tid;

                let ql0 = load_u8(ql_h + tid);
                let ql1 = load_u8(ql_h + 32u + tid);
                let qh = load_u8(qh_h + tid);

                let q1 = f32(i32((ql0 & 0xFu) | ((qh & 3u) << 4u)) - 32);
                let q2 = f32(i32((ql1 & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32);
                let q3 = f32(i32((ql0 >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32);
                let q4 = f32(i32((ql1 >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32);

                bsum += load_i8(sc_h + is_idx) * q1 * X[x_h]
                      + load_i8(sc_h + is_idx + 2u) * q2 * X[x_h + 32u]
                      + load_i8(sc_h + is_idx + 4u) * q3 * X[x_h + 64u]
                      + load_i8(sc_h + is_idx + 6u) * q4 * X[x_h + 96u];
            }
            acc += d * bsum;
        }
    }

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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// 4-token weight-reuse `Q6_K × F32` Prefill GEMM (`workgroups = (ceil(rows/4), ceil(batch/4), 1)`).
pub const SHADER_MATMUL_Q6_K_B4: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    batch: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<vec4<f32>, 128>;

fn load_u8(byte_off: u32) -> u32 {
    return (W[byte_off >> 2u] >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

fn load_i8(byte_off: u32) -> f32 {
    let v = load_u8(byte_off);
    return f32(select(i32(v), i32(v) - 256, v >= 128u));
}

fn load_f16_even(byte_off: u32) -> f32 {
    let w = W[byte_off >> 2u];
    let u = select(w & 0xFFFFu, w >> 16u, (byte_off & 2u) != 0u);
    return unpack2x16float(u).x;
}

@compute @workgroup_size(32, 4, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
    let t_base = wid.y * 4u;
    let nb = P.cols >> 8u;
    let is_idx = tid >> 4u;

    var acc = vec4<f32>(0.0);
    if (row < P.rows && t_base < P.batch) {
        let row_byte = row * nb * 210u;
        let t0 = t_base;
        let t1 = min(t_base + 1u, P.batch - 1u);
        let t2 = min(t_base + 2u, P.batch - 1u);
        let t3 = min(t_base + 3u, P.batch - 1u);

        for (var b = 0u; b < nb; b++) {
            let off = row_byte + b * 210u;
            let d = load_f16_even(off + 208u);
            var bsum = vec4<f32>(0.0);

            for (var half = 0u; half < 2u; half++) {
                let ql_h = off + half * 64u;
                let qh_h = off + 128u + half * 32u;
                let sc_h = off + 192u + half * 8u;

                let ql0 = load_u8(ql_h + tid);
                let ql1 = load_u8(ql_h + 32u + tid);
                let qh = load_u8(qh_h + tid);

                let w1 = load_i8(sc_h + is_idx) * f32(i32((ql0 & 0xFu) | ((qh & 3u) << 4u)) - 32);
                let w2 = load_i8(sc_h + is_idx + 2u) * f32(i32((ql1 & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32);
                let w3 = load_i8(sc_h + is_idx + 4u) * f32(i32((ql0 >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32);
                let w4 = load_i8(sc_h + is_idx + 6u) * f32(i32((ql1 >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32);
                let wv = vec4<f32>(w1, w2, w3, w4);

                let xoff = b * 256u + half * 128u + tid;
                let x0 = t0 * P.cols + xoff;
                bsum.x += dot(wv, vec4<f32>(X[x0], X[x0+32u], X[x0+64u], X[x0+96u]));
                let x1 = t1 * P.cols + xoff;
                bsum.y += dot(wv, vec4<f32>(X[x1], X[x1+32u], X[x1+64u], X[x1+96u]));
                let x2 = t2 * P.cols + xoff;
                bsum.z += dot(wv, vec4<f32>(X[x2], X[x2+32u], X[x2+64u], X[x2+96u]));
                let x3 = t3 * P.cols + xoff;
                bsum.w += dot(wv, vec4<f32>(X[x3], X[x3+32u], X[x3+64u], X[x3+96u]));
            }
            acc += d * bsum;
        }
    }

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
    if (tid == 0u && row < P.rows) {
        let res = smem[sidx] + smem[sidx + 1u];
        if (t_base < P.batch) { OUT[t_base * P.rows + row] = res.x; }
        if (t_base + 1u < P.batch) { OUT[(t_base + 1u) * P.rows + row] = res.y; }
        if (t_base + 2u < P.batch) { OUT[(t_base + 2u) * P.rows + row] = res.z; }
        if (t_base + 3u < P.batch) { OUT[(t_base + 3u) * P.rows + row] = res.w; }
    }
}
"#;

/// Quantized `Q4_0 × F32` GEMV/GEMM (`18` bytes per 32-elem block).
pub const SHADER_MATMUL_Q4_0: &str = r#"
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

@compute @workgroup_size(32, 4, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// Quantized `Q8_0 × F32` GEMV/GEMM (`34` bytes per 32-elem block).
pub const SHADER_MATMUL_Q8_0: &str = r#"
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

@compute @workgroup_size(32, 4, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// Unquantized `F32 × F32` GEMV/GEMM.
pub const SHADER_MATMUL_F32: &str = r#"
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

@compute @workgroup_size(32, 4, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
    let tok = wid.y;
    var acc = 0.0;
    if (row < P.rows && tok < P.batch) {
        let w_off = row * P.cols;
        let x_off = tok * P.cols;
        for (var c = tid; c < P.cols; c += 32u) {
            acc += W[w_off + c] * X[x_off + c];
        }
    }
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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// Unquantized `F16 × F32` GEMV/GEMM.
pub const SHADER_MATMUL_F16: &str = r#"
struct Params { rows: u32, cols: u32, batch: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> smem: array<f32, 128>;

@compute @workgroup_size(32, 4, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let r_local = lid.y;
    let row = wid.x * 4u + r_local;
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
    if (tid == 0u && row < P.rows && tok < P.batch) {
        OUT[tok * P.rows + row] = smem[sidx] + smem[sidx + 1u];
    }
}
"#;

/// Parallel tree-reduction RMSNorm (`out = rms_norm(x, weight, eps)`), 1 workgroup per token.
pub const SHADER_RMS_NORM: &str = r#"
struct Params {
    dim: u32,
    count: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
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
    let inv = inverseSqrt((smem[0] + smem[1]) / f32(P.dim) + P.eps);
    for (var i = tid; i < P.dim; i += 64u) {
        OUT[base + i] = X[base + i] * inv * W[i];
    }
}
"#;

/// In-place per-head RMSNorm (used for `Qwen3` / `Qwen3MoE` QK-Norm without buffer aliasing).
pub const SHADER_RMS_NORM_INPLACE: &str = r#"
struct Params {
    dim: u32,
    count: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

var<workgroup> smem: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
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
    let inv = inverseSqrt((smem[0] + smem[1]) / f32(P.dim) + P.eps);
    for (var i = tid; i < P.dim; i += 64u) {
        X[base + i] = X[base + i] * inv * W[i];
    }
}
"#;

/// In-place bias addition `X[t * dim + i] += B[i]`.
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

/// Fused Normal/NeoX RoPE (using precomputed LUT matching CPU YaRN/Linear/Standard RoPE)
/// + KV Cache store directly in VRAM.
pub const SHADER_ROPE_KV_STORE: &str = r#"
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    half_rope: u32,
    start_pos: u32,
    batch: u32,
    layer_kv_offset: u32,
    is_neox: u32,
}

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> COS_SIN: array<f32>;
@group(0) @binding(4) var<storage, read_write> K_CACHE: array<f32>;
@group(0) @binding(5) var<storage, read_write> V_CACHE: array<f32>;
@group(0) @binding(6) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let tok = gid.y;
    if (tok >= P.batch) { return; }

    let pos = P.start_pos + tok;
    let q_dim = P.num_heads * P.head_dim;
    let kv_dim = P.num_kv_heads * P.head_dim;
    let lut_stride = P.half_rope * 2u;
    let lut_base = pos * lut_stride;

    // 1. Apply RoPE to Q heads
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

    // 2. Apply RoPE to K heads and write K & V into VRAM KV cache slot `pos`
    let k_pairs = P.num_kv_heads * P.half_rope;
    if (tid < k_pairs) {
        let kh = tid / P.half_rope;
        let i = tid % P.half_rope;
        let c = COS_SIN[lut_base + i];
        let s = COS_SIN[lut_base + P.half_rope + i];
        let kh_off = tok * kv_dim + kh * P.head_dim;
        let cache_off = P.layer_kv_offset + pos * kv_dim + kh * P.head_dim;
        if (P.is_neox != 0u) {
            let x0 = K[kh_off + i];
            let x1 = K[kh_off + i + P.half_rope];
            let r0 = x0 * c - x1 * s;
            let r1 = x0 * s + x1 * c;
            K[kh_off + i] = r0;
            K[kh_off + i + P.half_rope] = r1;
            K_CACHE[cache_off + i] = r0;
            K_CACHE[cache_off + i + P.half_rope] = r1;
            V_CACHE[cache_off + i] = V[kh_off + i];
            V_CACHE[cache_off + i + P.half_rope] = V[kh_off + i + P.half_rope];
        } else {
            let x0 = K[kh_off + 2u * i];
            let x1 = K[kh_off + 2u * i + 1u];
            let r0 = x0 * c - x1 * s;
            let r1 = x0 * s + x1 * c;
            K[kh_off + 2u * i] = r0;
            K[kh_off + 2u * i + 1u] = r1;
            K_CACHE[cache_off + 2u * i] = r0;
            K_CACHE[cache_off + 2u * i + 1u] = r1;
            V_CACHE[cache_off + 2u * i] = V[kh_off + 2u * i];
            V_CACHE[cache_off + 2u * i + 1u] = V[kh_off + 2u * i + 1u];
        }
    }

    // Copy any non-RoPE trailing head dims into K/V cache (partial RoPE safety)
    let rope_dim = P.half_rope * 2u;
    if (rope_dim < P.head_dim && tid < kv_dim) {
        let kh = tid / P.head_dim;
        let d = tid % P.head_dim;
        if (d >= rope_dim) {
            let src = tok * kv_dim + kh * P.head_dim + d;
            let dst = P.layer_kv_offset + pos * kv_dim + kh * P.head_dim + d;
            K_CACHE[dst] = K[src];
            V_CACHE[dst] = V[src];
        }
    }
}
"#;

/// Online Tiled Flash-Attention GQA on VRAM (0 bytes global scratch, exact causal softmax).
/// Dispatch: `workgroups = (num_heads, batch, 1)`.
pub const SHADER_GQA_FLASH_ATTN: &str = r#"
struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    batch: u32,
    layer_kv_offset: u32,
    inv_sqrt_hd: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_CACHE: array<f32>;
@group(0) @binding(2) var<storage, read> V_CACHE: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

var<workgroup> q_sh: array<f32, 256>;
var<workgroup> out_sh: array<f32, 256>;
var<workgroup> scores: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let h = wid.x;
    let tok = wid.y;
    if (h >= P.num_heads || tok >= P.batch) { return; }

    let hd = P.head_dim;
    let q_dim = P.num_heads * hd;
    let kv_dim = P.num_kv_heads * hd;
    let kv_mul = P.num_heads / P.num_kv_heads;
    let kv_h = h / kv_mul;
    let pos = P.start_pos + tok;
    let seq_len = pos + 1u;
    let q_off = tok * q_dim + h * hd;

    for (var d = tid; d < hd; d += 64u) {
        q_sh[d] = Q[q_off + d];
        out_sh[d] = 0.0;
    }
    workgroupBarrier();

    var running_max = -1e30;
    var running_sum = 0.0;

    for (var tile_start = 0u; tile_start < seq_len; tile_start += 64u) {
        let s = tile_start + tid;
        var sc = -1e30;
        if (s < seq_len) {
            let k_off = P.layer_kv_offset + s * kv_dim + kv_h * hd;
            var dot_v = 0.0;
            for (var d = 0u; d < hd; d++) {
                dot_v += q_sh[d] * K_CACHE[k_off + d];
            }
            sc = dot_v * P.inv_sqrt_hd;
        }
        scores[tid] = sc;
        red[tid] = sc;
        workgroupBarrier();

        // Parallel max over 64 keys
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
        let new_max = max(running_max, tile_max);
        let rescale = select(1.0, exp(running_max - new_max), tile_start > 0u);
        workgroupBarrier();

        let w = select(0.0, exp(scores[tid] - new_max), s < seq_len);
        scores[tid] = w;
        red[tid] = w;
        workgroupBarrier();

        // Parallel sum of weights
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
        let tile_sum = red[0] + red[1];

        let tile_end = min(64u, seq_len - tile_start);
        for (var d = tid; d < hd; d += 64u) {
            var acc = out_sh[d] * rescale;
            for (var k = 0u; k < tile_end; k++) {
                let v_off = P.layer_kv_offset + (tile_start + k) * kv_dim + kv_h * hd + d;
                acc += scores[k] * V_CACHE[v_off];
            }
            out_sh[d] = acc;
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
