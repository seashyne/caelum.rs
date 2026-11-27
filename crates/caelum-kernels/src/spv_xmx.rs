//! Pure-Rust SPIR-V 1.5 shader generator targeting hardware matrix engines
//! (`SPV_KHR_cooperative_matrix`, `M=8, N=16, K=16`, `f16 × f16 + f32`) such as
//! Intel Xe2 XMX cores via `wgpu`'s `PASSTHROUGH_SHADERS` + `EXPERIMENTAL_COOPERATIVE_MATRIX`.
//!
//! Zero external crate dependencies: builds valid SPIR-V binary streams directly in memory
//! using structured control flow (`OpLoopMerge` / `OpSelectionMerge` / `OpPhi`).
//! All generated compute shaders conform strictly to `GpuPipelines::bgl_matmul`
//! (`binding 0: W (ro)`, `binding 1: X (ro)`, `binding 2: OUT (rw)`, `binding 3: Uniform Params`).

use std::collections::HashMap;

/// Tiling configuration for an XMX cooperative-matrix GEMM shader.
///
/// Computes `OUT[batch, rows] = X[batch, cols] * W[rows, cols]^T` using
/// hardware `M=8, N=16, K=16` cooperative matrix operations inside workgroups
/// of `wg_m_sgs * wg_n_sgs` 32-lane subgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XmxTileCfg {
    /// Total output M-rows (`batch` dimension) per workgroup; multiple of `8 * wg_m_sgs`.
    pub m_tile: u32,
    /// Total output N-cols (`rows` dimension of W) per workgroup; multiple of `16 * wg_n_sgs`.
    pub n_tile: u32,
    /// Reduction K-slice per SLM barrier; multiple of 16 (`64` for Q4_K/F16/F32, `128` for Q6_K).
    pub k_tile: u32,
    /// Number of subgroups tiled along M (`1` or `2`).
    pub wg_m_sgs: u32,
    /// Number of subgroups tiled along N (`1`, `2`, or `4`).
    pub wg_n_sgs: u32,
}

impl XmxTileCfg {
    pub const fn wg_size(&self) -> u32 {
        self.wg_m_sgs * self.wg_n_sgs * 32
    }

    pub const fn acc_m(&self) -> u32 {
        (self.m_tile / 8) / self.wg_m_sgs
    }

    pub const fn acc_n(&self) -> u32 {
        (self.n_tile / 16) / self.wg_n_sgs
    }
}

/// Small-batch (`B = 2..8`) XMX config (`64` threads = 2 subgroups, 8x32 tile).
pub const CFG_M8_N32_K64: XmxTileCfg = XmxTileCfg {
    m_tile: 8,
    n_tile: 32,
    k_tile: 64,
    wg_m_sgs: 1,
    wg_n_sgs: 2,
};

/// Medium-batch (`B = 9..16`) XMX config (`128` threads = 4 subgroups, 16x64 tile).
pub const CFG_M16_N64_K64: XmxTileCfg = XmxTileCfg {
    m_tile: 16,
    n_tile: 64,
    k_tile: 64,
    wg_m_sgs: 1,
    wg_n_sgs: 4,
};

/// Medium-batch (`B = 9..16`) narrower N config (`128` threads = 4 subgroups, 16x32 tile, 2x WG parallelism).
pub const CFG_M16_N32_K64: XmxTileCfg = XmxTileCfg {
    m_tile: 16,
    n_tile: 32,
    k_tile: 64,
    wg_m_sgs: 2,
    wg_n_sgs: 2,
};

/// Large-batch / Prefill (`B >= 17`) XMX config (`128` threads = 4 subgroups, 32x64 tile, 4 acc/sg).
pub const CFG_M32_N64_K64: XmxTileCfg = XmxTileCfg {
    m_tile: 32,
    n_tile: 64,
    k_tile: 64,
    wg_m_sgs: 2,
    wg_n_sgs: 2,
};

/// Prefill high-reuse XMX config (`128` threads = 4 subgroups, 64x32 tile, fits cleanly < 23 KB SLM).
pub const CFG_M64_N32_K64: XmxTileCfg = XmxTileCfg {
    m_tile: 64,
    n_tile: 32,
    k_tile: 64,
    wg_m_sgs: 2,
    wg_n_sgs: 2,
};

/// Small-batch 128-K step XMX config (`64` threads = 2 subgroups, 8x32 tile, K=128 = 8 XMX ops per barrier).
pub const CFG_M8_N32_K128: XmxTileCfg = XmxTileCfg {
    m_tile: 8,
    n_tile: 32,
    k_tile: 128,
    wg_m_sgs: 1,
    wg_n_sgs: 2,
};

/// 128-K step XMX config (`128` threads = 4 subgroups, 16x32 tile, K=128 = 8 XMX ops per barrier, ideal for Q6_K!).
pub const CFG_M16_N32_K128: XmxTileCfg = XmxTileCfg {
    m_tile: 16,
    n_tile: 32,
    k_tile: 128,
    wg_m_sgs: 2,
    wg_n_sgs: 2,
};

/// 128-K step prefill XMX config (`128` threads = 4 subgroups, 32x32 tile, 19.6 KB SLM).
pub const CFG_M32_N32_K128: XmxTileCfg = XmxTileCfg {
    m_tile: 32,
    n_tile: 32,
    k_tile: 128,
    wg_m_sgs: 2,
    wg_n_sgs: 2,
};

// SPIR-V 1.5 opcodes & constants used cleanly by name.
#[allow(dead_code)]
mod op {
    pub const EXTENSION: u32 = 10;
    pub const EXT_INST_IMPORT: u32 = 11;
    pub const EXT_INST: u32 = 12;
    pub const MEMORY_MODEL: u32 = 14;
    pub const ENTRY_POINT: u32 = 15;
    pub const EXECUTION_MODE: u32 = 16;
    pub const CAPABILITY: u32 = 17;
    pub const TYPE_VOID: u32 = 19;
    pub const TYPE_BOOL: u32 = 20;
    pub const TYPE_INT: u32 = 21;
    pub const TYPE_FLOAT: u32 = 22;
    pub const TYPE_VECTOR: u32 = 23;
    pub const TYPE_ARRAY: u32 = 28;
    pub const TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const TYPE_STRUCT: u32 = 30;
    pub const TYPE_POINTER: u32 = 32;
    pub const TYPE_FUNCTION: u32 = 33;
    pub const CONSTANT: u32 = 43;
    pub const CONSTANT_NULL: u32 = 46;
    pub const FUNCTION: u32 = 54;
    pub const FUNCTION_END: u32 = 56;
    pub const VARIABLE: u32 = 59;
    pub const LOAD: u32 = 61;
    pub const STORE: u32 = 62;
    pub const ACCESS_CHAIN: u32 = 65;
    pub const DECORATE: u32 = 71;
    pub const MEMBER_DECORATE: u32 = 72;
    pub const COMPOSITE_EXTRACT: u32 = 81;
    pub const CONVERT_F_TO_S: u32 = 110;
    pub const CONVERT_S_TO_F: u32 = 111;
    pub const CONVERT_U_TO_F: u32 = 112;
    pub const F_CONVERT: u32 = 115;
    pub const BITCAST: u32 = 124;
    pub const F_NEGATE: u32 = 127;
    pub const I_ADD: u32 = 128;
    pub const F_ADD: u32 = 129;
    pub const I_SUB: u32 = 130;
    pub const F_SUB: u32 = 131;
    pub const I_MUL: u32 = 132;
    pub const F_MUL: u32 = 133;
    pub const LOGICAL_AND: u32 = 167;
    pub const SELECT: u32 = 169;
    pub const I_NOT_EQUAL: u32 = 171;
    pub const U_LESS_THAN: u32 = 176;
    pub const SHIFT_RIGHT_LOGICAL: u32 = 194;
    pub const SHIFT_LEFT_LOGICAL: u32 = 196;
    pub const BITWISE_OR: u32 = 197;
    pub const BITWISE_AND: u32 = 199;
    pub const BIT_FIELD_S_EXTRACT: u32 = 202;
    pub const BIT_FIELD_U_EXTRACT: u32 = 203;
    pub const CONTROL_BARRIER: u32 = 224;
    pub const PHI: u32 = 245;
    pub const LOOP_MERGE: u32 = 246;
    pub const SELECTION_MERGE: u32 = 247;
    pub const LABEL: u32 = 248;
    pub const BRANCH: u32 = 249;
    pub const BRANCH_CONDITIONAL: u32 = 250;
    pub const RETURN: u32 = 253;
    pub const TYPE_COOPERATIVE_MATRIX_KHR: u32 = 4456;
    pub const COOPERATIVE_MATRIX_LOAD_KHR: u32 = 4457;
    pub const COOPERATIVE_MATRIX_STORE_KHR: u32 = 4458;
    pub const COOPERATIVE_MATRIX_MUL_ADD_KHR: u32 = 4459;
}

/// Minimal zero-dependency SPIR-V 1.5 assembler with type/constant deduplication.
#[allow(dead_code)]
struct Spv {
    next_id: u32,
    caps: Vec<u32>,
    exts: Vec<u32>,
    header: Vec<u32>,
    decs: Vec<u32>,
    types_globals: Vec<u32>,
    fns: Vec<u32>,

    t_void: u32,
    t_bool: u32,
    t_u32: u32,
    t_i32: u32,
    t_f16: u32,
    t_v2f16: u32,
    t_f32: u32,
    t_v2f32: u32,
    t_v3u32: u32,
    t_coop_a: u32,
    t_coop_b: u32,
    t_coop_c: u32,
    c_coop_c_zero: u32,
    glsl_ext: u32,

    u32_consts: HashMap<u32, u32>,
    f16_zero: u32,
    f32_consts: HashMap<u32, u32>,
    i32_consts: HashMap<i32, u32>,
}

#[allow(dead_code)]
impl Spv {
    fn new_xmx(wg_size: u32) -> (Self, u32, u32, u32, u32) {
        let mut s = Self {
            next_id: 1,
            caps: Vec::new(),
            exts: Vec::new(),
            header: Vec::new(),
            decs: Vec::new(),
            types_globals: Vec::new(),
            fns: Vec::new(),
            t_void: 0,
            t_bool: 0,
            t_u32: 0,
            t_i32: 0,
            t_f16: 0,
            t_v2f16: 0,
            t_f32: 0,
            t_v2f32: 0,
            t_v3u32: 0,
            t_coop_a: 0,
            t_coop_b: 0,
            t_coop_c: 0,
            c_coop_c_zero: 0,
            glsl_ext: 0,
            u32_consts: HashMap::new(),
            f16_zero: 0,
            f32_consts: HashMap::new(),
            i32_consts: HashMap::new(),
        };
        // Capabilities: Shader(1), Float16(9), VulkanMemoryModel(5345), CooperativeMatrixKHR(6022)
        for cap in [1u32, 9, 5345, 6022] {
            Self::inst(&mut s.caps, op::CAPABILITY, &[cap]);
        }
        Self::inst_str(&mut s.exts, op::EXTENSION, &[], "SPV_KHR_cooperative_matrix", &[]);
        Self::inst_str(&mut s.exts, op::EXTENSION, &[], "SPV_KHR_vulkan_memory_model", &[]);
        s.glsl_ext = s.id();
        Self::inst_str(&mut s.exts, op::EXT_INST_IMPORT, &[s.glsl_ext], "GLSL.std.450", &[]);
        // OpMemoryModel Logical(0) Vulkan(3)
        Self::inst(&mut s.header, op::MEMORY_MODEL, &[0, 3]);

        s.t_void = s.id();
        s.t_bool = s.id();
        s.t_u32 = s.id();
        s.t_i32 = s.id();
        s.t_f16 = s.id();
        s.t_v2f16 = s.id();
        s.t_f32 = s.id();
        s.t_v2f32 = s.id();
        s.t_v3u32 = s.id();
        let t_fn = s.id();

        Self::inst(&mut s.types_globals, op::TYPE_VOID, &[s.t_void]);
        Self::inst(&mut s.types_globals, op::TYPE_BOOL, &[s.t_bool]);
        Self::inst(&mut s.types_globals, op::TYPE_INT, &[s.t_u32, 32, 0]);
        Self::inst(&mut s.types_globals, op::TYPE_INT, &[s.t_i32, 32, 1]);
        Self::inst(&mut s.types_globals, op::TYPE_FLOAT, &[s.t_f16, 16]);
        Self::inst(&mut s.types_globals, op::TYPE_VECTOR, &[s.t_v2f16, s.t_f16, 2]);
        Self::inst(&mut s.types_globals, op::TYPE_FLOAT, &[s.t_f32, 32]);
        Self::inst(&mut s.types_globals, op::TYPE_VECTOR, &[s.t_v2f32, s.t_f32, 2]);
        Self::inst(&mut s.types_globals, op::TYPE_VECTOR, &[s.t_v3u32, s.t_u32, 3]);
        Self::inst(&mut s.types_globals, op::TYPE_FUNCTION, &[t_fn, s.t_void]);

        s.f16_zero = s.id();
        Self::inst(&mut s.types_globals, op::CONSTANT, &[s.t_f16, s.f16_zero, 0]);

        let c_3 = s.u32_c(3); // Scope::Subgroup
        let c_8 = s.u32_c(8);
        let c_16 = s.u32_c(16);
        let c_0 = s.u32_c(0); // MatrixAKHR
        let c_1 = s.u32_c(1); // MatrixBKHR
        let c_2 = s.u32_c(2); // MatrixAccumulatorKHR

        s.t_coop_a = s.id();
        s.t_coop_b = s.id();
        s.t_coop_c = s.id();
        s.c_coop_c_zero = s.id();
        Self::inst(&mut s.types_globals, op::TYPE_COOPERATIVE_MATRIX_KHR, &[s.t_coop_a, s.t_f16, c_3, c_8, c_16, c_0]);
        Self::inst(&mut s.types_globals, op::TYPE_COOPERATIVE_MATRIX_KHR, &[s.t_coop_b, s.t_f16, c_3, c_16, c_16, c_1]);
        Self::inst(&mut s.types_globals, op::TYPE_COOPERATIVE_MATRIX_KHR, &[s.t_coop_c, s.t_f32, c_3, c_8, c_16, c_2]);
        Self::inst(&mut s.types_globals, op::CONSTANT_NULL, &[s.t_coop_c, s.c_coop_c_zero]);

        let main_fn = s.id();
        let var_lid = s.id();
        let var_wid = s.id();
        Self::inst(&mut s.header, op::EXECUTION_MODE, &[main_fn, 17, wg_size, 1, 1]);

        (s, main_fn, t_fn, var_lid, var_wid)
    }

    fn id(&mut self) -> u32 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    fn u32_c(&mut self, val: u32) -> u32 {
        if let Some(&id) = self.u32_consts.get(&val) {
            return id;
        }
        let id = self.id();
        Self::inst(&mut self.types_globals, op::CONSTANT, &[self.t_u32, id, val]);
        self.u32_consts.insert(val, id);
        id
    }

    fn i32_c(&mut self, val: i32) -> u32 {
        if let Some(&id) = self.i32_consts.get(&val) {
            return id;
        }
        let id = self.id();
        Self::inst(&mut self.types_globals, op::CONSTANT, &[self.t_i32, id, val as u32]);
        self.i32_consts.insert(val, id);
        id
    }

    fn f32_c(&mut self, val: f32) -> u32 {
        let bits = val.to_bits();
        if let Some(&id) = self.f32_consts.get(&bits) {
            return id;
        }
        let id = self.id();
        Self::inst(&mut self.types_globals, op::CONSTANT, &[self.t_f32, id, bits]);
        self.f32_consts.insert(bits, id);
        id
    }

    fn inst(dst: &mut Vec<u32>, opcode: u32, ops: &[u32]) {
        dst.push((((ops.len() + 1) as u32) << 16) | opcode);
        dst.extend_from_slice(ops);
    }

    fn push_str(dst: &mut Vec<u32>, s: &str) {
        let bytes = s.as_bytes();
        let words = (bytes.len() + 4) / 4;
        let base = dst.len();
        dst.resize(base + words, 0);
        let out = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr().add(base) as *mut u8, words * 4)
        };
        out[..bytes.len()].copy_from_slice(bytes);
    }

    fn inst_str(dst: &mut Vec<u32>, opcode: u32, prefix: &[u32], s: &str, suffix: &[u32]) {
        let s_words = (s.as_bytes().len() + 4) / 4;
        let wc = 1 + prefix.len() + s_words + suffix.len();
        dst.push(((wc as u32) << 16) | opcode);
        dst.extend_from_slice(prefix);
        Self::push_str(dst, s);
        dst.extend_from_slice(suffix);
    }

    fn finish(self) -> Vec<u32> {
        let mut out = vec![0x0723_0203, 0x0001_0500, 0, self.next_id, 0];
        out.extend_from_slice(&self.caps);
        out.extend_from_slice(&self.exts);
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.decs);
        out.extend_from_slice(&self.types_globals);
        out.extend_from_slice(&self.fns);
        out
    }

    // Concise SSA emission helpers inside `self.fns`
    fn label(&mut self) -> u32 {
        let l = self.id();
        Self::inst(&mut self.fns, op::LABEL, &[l]);
        l
    }
    fn emit_label(&mut self, l: u32) {
        Self::inst(&mut self.fns, op::LABEL, &[l]);
    }
    fn bin(&mut self, opcode: u32, ty: u32, a: u32, b: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, opcode, &[ty, r, a, b]);
        r
    }
    fn un(&mut self, opcode: u32, ty: u32, a: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, opcode, &[ty, r, a]);
        r
    }
    fn uadd(&mut self, a: u32, b: u32) -> u32 { self.bin(op::I_ADD, self.t_u32, a, b) }
    fn usub(&mut self, a: u32, b: u32) -> u32 { self.bin(op::I_SUB, self.t_u32, a, b) }
    fn umul(&mut self, a: u32, b: u32) -> u32 { self.bin(op::I_MUL, self.t_u32, a, b) }
    fn shr(&mut self, a: u32, b: u32) -> u32 { self.bin(op::SHIFT_RIGHT_LOGICAL, self.t_u32, a, b) }
    fn shl(&mut self, a: u32, b: u32) -> u32 { self.bin(op::SHIFT_LEFT_LOGICAL, self.t_u32, a, b) }
    fn and(&mut self, a: u32, b: u32) -> u32 { self.bin(op::BITWISE_AND, self.t_u32, a, b) }
    fn or(&mut self, a: u32, b: u32) -> u32 { self.bin(op::BITWISE_OR, self.t_u32, a, b) }
    fn ult(&mut self, a: u32, b: u32) -> u32 { self.bin(op::U_LESS_THAN, self.t_bool, a, b) }
    fn land(&mut self, a: u32, b: u32) -> u32 { self.bin(op::LOGICAL_AND, self.t_bool, a, b) }

    fn fadd(&mut self, a: u32, b: u32) -> u32 { self.bin(op::F_ADD, self.t_f32, a, b) }
    fn fsub(&mut self, a: u32, b: u32) -> u32 { self.bin(op::F_SUB, self.t_f32, a, b) }
    fn fmul(&mut self, a: u32, b: u32) -> u32 { self.bin(op::F_MUL, self.t_f32, a, b) }
    fn fneg(&mut self, a: u32) -> u32 { self.un(op::F_NEGATE, self.t_f32, a) }
    fn u2f(&mut self, a: u32) -> u32 { self.un(op::CONVERT_U_TO_F, self.t_f32, a) }
    fn s2f(&mut self, a: u32) -> u32 { self.un(op::CONVERT_S_TO_F, self.t_f32, a) }
    fn f32_to_f16(&mut self, a: u32) -> u32 { self.un(op::F_CONVERT, self.t_f16, a) }
    fn bitcast_f32(&mut self, a: u32) -> u32 { self.un(op::BITCAST, self.t_f32, a) }

    fn uextract(&mut self, base: u32, off: u32, cnt: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, op::BIT_FIELD_U_EXTRACT, &[self.t_u32, r, base, off, cnt]);
        r
    }
    fn sextract(&mut self, base: u32, off: u32, cnt: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, op::BIT_FIELD_S_EXTRACT, &[self.t_i32, r, base, off, cnt]);
        r
    }
    fn unpack2x16f(&mut self, packed: u32) -> (u32, u32) {
        let v = self.id();
        Self::inst(&mut self.fns, op::EXT_INST, &[self.t_v2f32, v, self.glsl_ext, 62, packed]);
        let x = self.id();
        let y = self.id();
        Self::inst(&mut self.fns, op::COMPOSITE_EXTRACT, &[self.t_f32, x, v, 0]);
        Self::inst(&mut self.fns, op::COMPOSITE_EXTRACT, &[self.t_f32, y, v, 1]);
        (x, y)
    }
    fn select(&mut self, ty: u32, cond: u32, t_val: u32, f_val: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, op::SELECT, &[ty, r, cond, t_val, f_val]);
        r
    }
    fn access(&mut self, ptr_ty: u32, base: u32, indices: &[u32]) -> u32 {
        let r = self.id();
        let mut ops = Vec::with_capacity(3 + indices.len());
        ops.push(ptr_ty);
        ops.push(r);
        ops.push(base);
        ops.extend_from_slice(indices);
        Self::inst(&mut self.fns, op::ACCESS_CHAIN, &ops);
        r
    }
    fn load(&mut self, ty: u32, ptr: u32) -> u32 {
        let r = self.id();
        Self::inst(&mut self.fns, op::LOAD, &[ty, r, ptr]);
        r
    }
    fn store(&mut self, ptr: u32, val: u32) {
        Self::inst(&mut self.fns, op::STORE, &[ptr, val]);
    }
    fn wg_barrier(&mut self, c_scope_wg: u32, c_sem_wg: u32) {
        Self::inst(&mut self.fns, op::CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_sem_wg]);
    }
}

/// Shared descriptor + storage + builtin setup matching `bgl_matmul`.
#[allow(dead_code)]
struct MatmulIo {
    var_w: u32,
    var_x: u32,
    var_out: u32,
    var_p: u32,
    var_sh_a: u32,
    var_sh_b: u32,
    var_sh_c: u32,

    tp_ssbo_u32: u32,
    tp_ssbo_f32: u32,
    tp_unif_u32: u32,
    tp_wg_f16: u32,
    tp_wg_f32: u32,

    c_0: u32,
    c_scope_wg: u32,
    c_sem_wg: u32,

    p_rows: u32,
    p_cols: u32,
    p_batch: u32,
    tid: u32,
    wg_m_base: u32,
    wg_n_base: u32,
    sg_m_off: u32,
    sg_n_off: u32,

    k_stride_a: u32,
    k_stride_b: u32,
}

fn setup_matmul_io(
    s: &mut Spv,
    cfg: XmxTileCfg,
    w_is_f32: bool,
    main_fn: u32,
    t_fn: u32,
    var_lid: u32,
    var_wid: u32,
    b_col_major: bool,
) -> MatmulIo {
    let var_w = s.id();
    let var_x = s.id();
    let var_out = s.id();
    let var_p = s.id();
    let var_sh_a = s.id();
    let var_sh_b = s.id();
    let var_sh_c = s.id();

    Spv::inst_str(
        &mut s.header,
        op::ENTRY_POINT,
        &[5, main_fn],
        "main",
        &[var_lid, var_wid, var_w, var_x, var_out, var_p, var_sh_a, var_sh_b, var_sh_c],
    );

    let t_rra_u32 = s.id();
    let t_rra_f32 = s.id();
    let t_blk_u32 = s.id();
    let t_blk_f32 = s.id();
    let t_params = s.id();
    Spv::inst(&mut s.types_globals, op::TYPE_RUNTIME_ARRAY, &[t_rra_u32, s.t_u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_RUNTIME_ARRAY, &[t_rra_f32, s.t_f32]);
    Spv::inst(&mut s.types_globals, op::TYPE_STRUCT, &[t_blk_u32, t_rra_u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_STRUCT, &[t_blk_f32, t_rra_f32]);
    Spv::inst(&mut s.types_globals, op::TYPE_STRUCT, &[t_params, s.t_u32, s.t_u32, s.t_u32, s.t_u32]);

    Spv::inst(&mut s.decs, op::DECORATE, &[t_rra_u32, 6, 4]); // ArrayStride 4
    Spv::inst(&mut s.decs, op::DECORATE, &[t_rra_f32, 6, 4]);
    Spv::inst(&mut s.decs, op::DECORATE, &[t_blk_u32, 2]); // Block
    Spv::inst(&mut s.decs, op::DECORATE, &[t_blk_f32, 2]);
    Spv::inst(&mut s.decs, op::DECORATE, &[t_params, 2]);
    Spv::inst(&mut s.decs, op::MEMBER_DECORATE, &[t_blk_u32, 0, 35, 0]);
    Spv::inst(&mut s.decs, op::MEMBER_DECORATE, &[t_blk_f32, 0, 35, 0]);
    for m in 0..4u32 {
        Spv::inst(&mut s.decs, op::MEMBER_DECORATE, &[t_params, m, 35, m * 4]);
    }

    let k_stride_a = cfg.k_tile + 8;
    let k_stride_b = if b_col_major { cfg.k_tile + 8 } else { cfg.n_tile + 8 };
    let sh_a_len = cfg.m_tile * k_stride_a;
    let sh_b_len = if b_col_major {
        cfg.n_tile * k_stride_b
    } else {
        cfg.k_tile * k_stride_b
    };
    let sh_c_len = cfg.m_tile * cfg.n_tile;

    let c_sh_a = s.u32_c(sh_a_len);
    let c_sh_b = s.u32_c(sh_b_len);
    let c_sh_c = s.u32_c(sh_c_len);

    let tp_in_v3u32 = s.id();
    let tp_ssbo_bu32 = s.id();
    let tp_ssbo_bf32 = s.id();
    let tp_unif_params = s.id();
    let tp_ssbo_u32 = s.id();
    let tp_ssbo_f32 = s.id();
    let tp_unif_u32 = s.id();
    let t_arr_a = s.id();
    let t_arr_b = s.id();
    let t_arr_c = s.id();
    let tp_wg_a = s.id();
    let tp_wg_b = s.id();
    let tp_wg_c = s.id();
    let tp_wg_f16 = s.id();
    let tp_wg_f32 = s.id();

    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_in_v3u32, 1, s.t_v3u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_ssbo_bu32, 12, t_blk_u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_ssbo_bf32, 12, t_blk_f32]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_unif_params, 2, t_params]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_ssbo_u32, 12, s.t_u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_ssbo_f32, 12, s.t_f32]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_unif_u32, 2, s.t_u32]);
    Spv::inst(&mut s.types_globals, op::TYPE_ARRAY, &[t_arr_a, s.t_f16, c_sh_a]);
    Spv::inst(&mut s.types_globals, op::TYPE_ARRAY, &[t_arr_b, s.t_f16, c_sh_b]);
    Spv::inst(&mut s.types_globals, op::TYPE_ARRAY, &[t_arr_c, s.t_f32, c_sh_c]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_wg_a, 4, t_arr_a]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_wg_b, 4, t_arr_b]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_wg_c, 4, t_arr_c]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_wg_f16, 4, s.t_f16]);
    Spv::inst(&mut s.types_globals, op::TYPE_POINTER, &[tp_wg_f32, 4, s.t_f32]);

    // Builtins
    Spv::inst(&mut s.types_globals, op::VARIABLE, &[tp_in_v3u32, var_lid, 1]);
    Spv::inst(&mut s.decs, op::DECORATE, &[var_lid, 11, 27]); // LocalInvocationId
    Spv::inst(&mut s.types_globals, op::VARIABLE, &[tp_in_v3u32, var_wid, 1]);
    Spv::inst(&mut s.decs, op::DECORATE, &[var_wid, 11, 26]); // WorkgroupId

    // Buffers matching bgl_matmul (binding 0=W, 1=X, 2=OUT, 3=Params)
    let w_ty = if w_is_f32 { tp_ssbo_bf32 } else { tp_ssbo_bu32 };
    for (v, ty, sc, b, ro) in [
        (var_w, w_ty, 12u32, 0u32, true),
        (var_x, tp_ssbo_bf32, 12, 1, true),
        (var_out, tp_ssbo_bf32, 12, 2, false),
        (var_p, tp_unif_params, 2, 3, false),
    ] {
        Spv::inst(&mut s.types_globals, op::VARIABLE, &[ty, v, sc]);
        Spv::inst(&mut s.decs, op::DECORATE, &[v, 34, 0]); // DescriptorSet 0
        Spv::inst(&mut s.decs, op::DECORATE, &[v, 33, b]); // Binding b
        if ro {
            Spv::inst(&mut s.decs, op::DECORATE, &[v, 24]); // NonWritable
        }
    }

    Spv::inst(&mut s.types_globals, op::VARIABLE, &[tp_wg_a, var_sh_a, 4]);
    Spv::inst(&mut s.types_globals, op::VARIABLE, &[tp_wg_b, var_sh_b, 4]);
    Spv::inst(&mut s.types_globals, op::VARIABLE, &[tp_wg_c, var_sh_c, 4]);

    let c_0 = s.u32_c(0);
    let c_1 = s.u32_c(1);
    let c_2 = s.u32_c(2);
    let c_scope_wg = s.u32_c(2);
    let c_sem_wg = s.u32_c(264);

    // Function entry
    Spv::inst(&mut s.fns, op::FUNCTION, &[s.t_void, main_fn, 0, t_fn]);
    let _entry = s.label();

    let lid_v = s.load(s.t_v3u32, var_lid);
    let tid = s.id();
    Spv::inst(&mut s.fns, op::COMPOSITE_EXTRACT, &[s.t_u32, tid, lid_v, 0]);
    let wid_v = s.load(s.t_v3u32, var_wid);
    let wid_x = s.id();
    let wid_y = s.id();
    Spv::inst(&mut s.fns, op::COMPOSITE_EXTRACT, &[s.t_u32, wid_x, wid_v, 0]);
    Spv::inst(&mut s.fns, op::COMPOSITE_EXTRACT, &[s.t_u32, wid_y, wid_v, 1]);

    let pr0 = s.access(tp_unif_u32, var_p, &[c_0]);
    let pr1 = s.access(tp_unif_u32, var_p, &[c_1]);
    let pr2 = s.access(tp_unif_u32, var_p, &[c_2]);
    let p_rows = s.load(s.t_u32, pr0);
    let p_cols = s.load(s.t_u32, pr1);
    let p_batch = s.load(s.t_u32, pr2);

    // wid.x tiles N (`rows`), wid.y tiles M (`batch`)
    let c_ntile = s.u32_c(cfg.n_tile);
    let c_mtile = s.u32_c(cfg.m_tile);
    let wg_n_base = s.umul(wid_x, c_ntile);
    let wg_m_base = s.umul(wid_y, c_mtile);

    let c_5 = s.u32_c(5);
    let sg_id = s.shr(tid, c_5);
    let (sg_m_idx, sg_n_idx) = if cfg.wg_m_sgs == 1 {
        (c_0, sg_id)
    } else if cfg.wg_n_sgs == 1 {
        (sg_id, c_0)
    } else {
        let mask_n = s.u32_c(cfg.wg_n_sgs - 1);
        let shift_m = s.u32_c(cfg.wg_n_sgs.trailing_zeros());
        (s.shr(sg_id, shift_m), s.and(sg_id, mask_n))
    };
    let sg_m_span = s.u32_c(cfg.acc_m() * 8);
    let sg_n_span = s.u32_c(cfg.acc_n() * 16);
    let sg_m_off = s.umul(sg_m_idx, sg_m_span);
    let sg_n_off = s.umul(sg_n_idx, sg_n_span);

    MatmulIo {
        var_w,
        var_x,
        var_out,
        var_p,
        var_sh_a,
        var_sh_b,
        var_sh_c,
        tp_ssbo_u32,
        tp_ssbo_f32,
        tp_unif_u32,
        tp_wg_f16,
        tp_wg_f32,
        c_0,
        c_scope_wg,
        c_sem_wg,
        p_rows,
        p_cols,
        p_batch,
        tid,
        wg_m_base,
        wg_n_base,
        sg_m_off,
        sg_n_off,
        k_stride_a,
        k_stride_b,
    }
}

/// Emit `sh_A[m * k_stride_a + kk] = f16(X[b, k + kk])` cooperatively across all WG threads.
fn emit_load_x_to_sh_a(s: &mut Spv, io: &MatmulIo, cfg: XmxTileCfg, k: u32) {
    let wg_size = cfg.wg_size();
    let total = cfg.m_tile * cfg.k_tile;
    let iters = total / wg_size;
    let k_shift = cfg.k_tile.trailing_zeros();
    let k_mask = cfg.k_tile - 1;
    let c_k_shift = s.u32_c(k_shift);
    let c_k_mask = s.u32_c(k_mask);
    let c_kstride_a = s.u32_c(io.k_stride_a);

    for it in 0..iters {
        let idx = if it == 0 {
            io.tid
        } else {
            let off = s.u32_c(it * wg_size);
            s.uadd(io.tid, off)
        };
        let m = s.shr(idx, c_k_shift);
        let kk = s.and(idx, c_k_mask);
        let g_b = s.uadd(io.wg_m_base, m);
        let g_col = s.uadd(k, kk);
        let in_b = s.ult(g_b, io.p_batch);
        let in_k = s.ult(g_col, io.p_cols);
        let valid = s.land(in_b, in_k);

        // Compute clamped index so SSBO load is always in-bounds cleanly.
        let safe_b = s.select(s.t_u32, in_b, g_b, io.c_0);
        let safe_k = s.select(s.t_u32, in_k, g_col, io.c_0);
        let row_off = s.umul(safe_b, io.p_cols);
        let x_idx = s.uadd(row_off, safe_k);
        let px = s.access(io.tp_ssbo_f32, io.var_x, &[io.c_0, x_idx]);
        let x_f32 = s.load(s.t_f32, px);
        let x_f16 = s.f32_to_f16(x_f32);
        let v_f16 = s.select(s.t_f16, valid, x_f16, s.f16_zero);

        let m_off = s.umul(m, c_kstride_a);
        let sh_idx = s.uadd(m_off, kk);
        let psh = s.access(io.tp_wg_f16, io.var_sh_a, &[sh_idx]);
        s.store(psh, v_f16);
    }
}

/// Emit K-substep XMX `8x16x16` accumulate loop (`k_tile / 16` steps) across subgroup accumulators.
fn emit_xmx_k_substeps(
    s: &mut Spv,
    io: &MatmulIo,
    cfg: XmxTileCfg,
    accs: &mut [u32],
    b_col_major: bool,
) {
    let acc_m = cfg.acc_m() as usize;
    let acc_n = cfg.acc_n() as usize;
    let k_steps = cfg.k_tile / 16;
    let c_layout_a = s.u32_c(0); // RowMajorKHR
    let c_layout_b = s.u32_c(if b_col_major { 1 } else { 0 });
    let c_stride_a = s.u32_c(io.k_stride_a);
    let c_stride_b = s.u32_c(io.k_stride_b);

    for ks in 0..k_steps {
        let kk = s.u32_c(ks * 16);
        let mut mat_a = Vec::with_capacity(acc_m);
        for am in 0..acc_m {
            let m_off = if am == 0 {
                io.sg_m_off
            } else {
                let c_am = s.u32_c((am as u32) * 8);
                s.uadd(io.sg_m_off, c_am)
            };
            let a_base = s.umul(m_off, c_stride_a);
            let a_idx = s.uadd(a_base, kk);
            let pa = s.access(io.tp_wg_f16, io.var_sh_a, &[a_idx]);
            let ma = s.id();
            Spv::inst(&mut s.fns, op::COOPERATIVE_MATRIX_LOAD_KHR, &[s.t_coop_a, ma, pa, c_layout_a, c_stride_a]);
            mat_a.push(ma);
        }
        for an in 0..acc_n {
            let n_off = if an == 0 {
                io.sg_n_off
            } else {
                let c_an = s.u32_c((an as u32) * 16);
                s.uadd(io.sg_n_off, c_an)
            };
            let b_idx = if b_col_major {
                let b_base = s.umul(n_off, c_stride_b);
                s.uadd(b_base, kk)
            } else {
                let b_base = s.umul(kk, c_stride_b);
                s.uadd(b_base, n_off)
            };
            let pb = s.access(io.tp_wg_f16, io.var_sh_b, &[b_idx]);
            let mb = s.id();
            Spv::inst(&mut s.fns, op::COOPERATIVE_MATRIX_LOAD_KHR, &[s.t_coop_b, mb, pb, c_layout_b, c_stride_b]);

            for am in 0..acc_m {
                let idx = am * acc_n + an;
                let mc_next = s.id();
                Spv::inst(
                    &mut s.fns,
                    op::COOPERATIVE_MATRIX_MUL_ADD_KHR,
                    &[s.t_coop_c, mc_next, mat_a[am], mb, accs[idx]],
                );
                accs[idx] = mc_next;
            }
        }
    }
}

/// Store subgroup accumulators into `sh_C` and bounds-checked write out to `OUT[batch, rows]`.
fn emit_store_xmx_out(s: &mut Spv, io: &MatmulIo, cfg: XmxTileCfg, accs: &[u32]) {
    let acc_m = cfg.acc_m() as usize;
    let acc_n = cfg.acc_n() as usize;
    let c_row_major = s.u32_c(0);
    let c_ntile = s.u32_c(cfg.n_tile);

    for am in 0..acc_m {
        let m_off = if am == 0 {
            io.sg_m_off
        } else {
            let c_am = s.u32_c((am as u32) * 8);
            s.uadd(io.sg_m_off, c_am)
        };
        let row_base = s.umul(m_off, c_ntile);
        for an in 0..acc_n {
            let n_off = if an == 0 {
                io.sg_n_off
            } else {
                let c_an = s.u32_c((an as u32) * 16);
                s.uadd(io.sg_n_off, c_an)
            };
            let sh_idx = s.uadd(row_base, n_off);
            let pc = s.access(io.tp_wg_f32, io.var_sh_c, &[sh_idx]);
            Spv::inst(
                &mut s.fns,
                op::COOPERATIVE_MATRIX_STORE_KHR,
                &[pc, accs[am * acc_n + an], c_row_major, c_ntile],
            );
        }
    }

    s.wg_barrier(io.c_scope_wg, io.c_sem_wg);

    // Bounds-checked global write from sh_C -> OUT
    let wg_size = cfg.wg_size();
    let total = cfg.m_tile * cfg.n_tile;
    let iters = total / wg_size;
    let n_shift = cfg.n_tile.trailing_zeros();
    let n_mask = cfg.n_tile - 1;
    let c_n_shift = s.u32_c(n_shift);
    let c_n_mask = s.u32_c(n_mask);

    for it in 0..iters {
        let idx = if it == 0 {
            io.tid
        } else {
            let off = s.u32_c(it * wg_size);
            s.uadd(io.tid, off)
        };
        let m = s.shr(idx, c_n_shift);
        let n = s.and(idx, c_n_mask);
        let g_b = s.uadd(io.wg_m_base, m);
        let g_r = s.uadd(io.wg_n_base, n);
        let in_b = s.ult(g_b, io.p_batch);
        let in_r = s.ult(g_r, io.p_rows);
        let valid = s.land(in_b, in_r);

        let then_lbl = s.id();
        let merge_lbl = s.id();
        Spv::inst(&mut s.fns, op::SELECTION_MERGE, &[merge_lbl, 0]);
        Spv::inst(&mut s.fns, op::BRANCH_CONDITIONAL, &[valid, then_lbl, merge_lbl]);

        s.emit_label(then_lbl);
        let psh = s.access(io.tp_wg_f32, io.var_sh_c, &[idx]);
        let val = s.load(s.t_f32, psh);
        let br = s.umul(g_b, io.p_rows);
        let out_idx = s.uadd(br, g_r);
        let pout = s.access(io.tp_ssbo_f32, io.var_out, &[io.c_0, out_idx]);
        s.store(pout, val);
        Spv::inst(&mut s.fns, op::BRANCH, &[merge_lbl]);

        s.emit_label(merge_lbl);
    }

    Spv::inst(&mut s.fns, op::RETURN, &[]);
    Spv::inst(&mut s.fns, op::FUNCTION_END, &[]);
}

/// Store `val` (`f16`) into `sh_B` at logical `(k_local, n_local)` using either
/// `ColumnMajorKHR` (`n * k_stride_b + k`) or `RowMajorKHR` (`k * k_stride_b + n`).
fn emit_store_sh_b(
    s: &mut Spv,
    io: &MatmulIo,
    b_col_major: bool,
    c_kstride_b: u32,
    k_local: u32,
    n_base_or_idx: u32,
    val_f16: u32,
) {
    let sh_idx = if b_col_major {
        s.uadd(n_base_or_idx, k_local)
    } else {
        let row_k = s.umul(k_local, c_kstride_b);
        s.uadd(row_k, n_base_or_idx)
    };
    let p = s.access(io.tp_wg_f16, io.var_sh_b, &[sh_idx]);
    s.store(p, val_f16);
}

/// Shared structured K-loop + SLM load/barrier + XMX compute + Phi patching + output store
/// reused across `Q4_K`, `Q6_K`, `F16`, and `F32` XMX GEMM kernels.
fn emit_xmx_gemm_loop<F>(
    cfg: XmxTileCfg,
    w_is_f32: bool,
    b_col_major: bool,
    mut emit_b_tile: F,
) -> Vec<u32>
where
    F: FnMut(&mut Spv, &MatmulIo, u32),
{
    let wg_size = cfg.wg_size();
    let (mut s, main_fn, t_fn, var_lid, var_wid) = Spv::new_xmx(wg_size);
    let io = setup_matmul_io(&mut s, cfg, w_is_f32, main_fn, t_fn, var_lid, var_wid, b_col_major);

    let c_ktile = s.u32_c(cfg.k_tile);
    let n_accs = (cfg.acc_m() * cfg.acc_n()) as usize;

    let preheader_lbl = s.id();
    Spv::inst(&mut s.fns, op::BRANCH, &[preheader_lbl]);
    s.emit_label(preheader_lbl);

    let header_lbl = s.id();
    let body_lbl = s.id();
    let cont_lbl = s.id();
    let merge_lbl = s.id();
    Spv::inst(&mut s.fns, op::BRANCH, &[header_lbl]);

    s.emit_label(header_lbl);
    let k_phi = s.id();
    let k_next = s.id();
    Spv::inst(&mut s.fns, op::PHI, &[s.t_u32, k_phi, io.c_0, preheader_lbl, k_next, cont_lbl]);

    let mut acc_phis = Vec::with_capacity(n_accs);
    let mut acc_next_ids = Vec::with_capacity(n_accs);
    for _ in 0..n_accs {
        let ap = s.id();
        let an = s.id();
        Spv::inst(&mut s.fns, op::PHI, &[s.t_coop_c, ap, s.c_coop_c_zero, preheader_lbl, an, cont_lbl]);
        acc_phis.push(ap);
        acc_next_ids.push(an);
    }

    let cond = s.ult(k_phi, io.p_cols);
    Spv::inst(&mut s.fns, op::LOOP_MERGE, &[merge_lbl, cont_lbl, 0]);
    Spv::inst(&mut s.fns, op::BRANCH_CONDITIONAL, &[cond, body_lbl, merge_lbl]);

    s.emit_label(body_lbl);
    emit_load_x_to_sh_a(&mut s, &io, cfg, k_phi);
    emit_b_tile(&mut s, &io, k_phi);
    s.wg_barrier(io.c_scope_wg, io.c_sem_wg);

    let mut cur_accs = acc_phis.clone();
    emit_xmx_k_substeps(&mut s, &io, cfg, &mut cur_accs, b_col_major);
    s.wg_barrier(io.c_scope_wg, io.c_sem_wg);
    Spv::inst(&mut s.fns, op::BRANCH, &[cont_lbl]);

    s.emit_label(cont_lbl);
    let k_next_val = s.id();
    Spv::inst(&mut s.fns, op::I_ADD, &[s.t_u32, k_next_val, k_phi, c_ktile]);
    for w in &mut s.fns {
        if *w == k_next {
            *w = k_next_val;
        }
        for (i, &placeholder) in acc_next_ids.iter().enumerate() {
            if *w == placeholder {
                *w = cur_accs[i];
            }
        }
    }
    Spv::inst(&mut s.fns, op::BRANCH, &[header_lbl]);

    s.emit_label(merge_lbl);
    emit_store_xmx_out(&mut s, &io, cfg, &acc_phis);
    s.finish()
}

/// Generate a SPIR-V 1.5 Intel XMX (`8x16x16`) **`Q4_K × F32` GEMM** shader (`k_tile = 64` or `128`).
///
/// Dispatch grid: `wg_x = rows.div_ceil(cfg.n_tile)`, `wg_y = batch.div_ceil(cfg.m_tile)`, `wg_z = 1`.
pub fn spirv_xmx_q4_k(cfg: XmxTileCfg, b_col_major: bool) -> Vec<u32> {
    spirv_xmx_q4_or_q5_k(cfg, b_col_major, false)
}

/// Generate a SPIR-V 1.5 Intel XMX (`8x16x16`) **`Q5_K × F32` GEMM** shader (`k_tile = 64` or `128`).
///
/// Dispatch grid: `wg_x = rows.div_ceil(cfg.n_tile)`, `wg_y = batch.div_ceil(cfg.m_tile)`, `wg_z = 1`.
pub fn spirv_xmx_q5_k(cfg: XmxTileCfg, b_col_major: bool) -> Vec<u32> {
    spirv_xmx_q4_or_q5_k(cfg, b_col_major, true)
}

fn spirv_xmx_q4_or_q5_k(cfg: XmxTileCfg, b_col_major: bool, is_q5k: bool) -> Vec<u32> {
    assert!(cfg.k_tile == 64 || cfg.k_tile == 128, "Q4_K/Q5_K XMX expects k_tile = 64 or 128");
    emit_xmx_gemm_loop(cfg, false, b_col_major, |s, io, k_phi| {
        let wg_size = cfg.wg_size();
        let c_8 = s.u32_c(8);
        let nb = s.shr(io.p_cols, c_8);
        let c_blk_u32 = s.u32_c(if is_q5k { 44 } else { 36 });
        let c_qs_base = s.u32_c(if is_q5k { 12 } else { 4 });
        let c_1 = s.u32_c(1);
        let c_2 = s.u32_c(2);
        let c_3 = s.u32_c(3);
        let c_4 = s.u32_c(4);
        let c_6 = s.u32_c(6);
        let c_7 = s.u32_c(7);
        let c_16 = s.u32_c(16);
        let c_24 = s.u32_c(24);
        let c_32 = s.u32_c(32);
        let c_0x30 = s.u32_c(0x30);
        let c_kstride_b = s.u32_c(io.k_stride_b);
        let row_blk_stride = s.umul(nb, c_blk_u32);

        let blk_idx = s.shr(k_phi, c_8);
        let k_div64 = s.shr(k_phi, c_6);
        let base_chunk = s.and(k_div64, c_3);
        let is_k128 = cfg.k_tile == 128;

        let w_iters = (cfg.n_tile * 8) / wg_size;
        for it in 0..w_iters {
            let u_idx = if it == 0 {
                io.tid
            } else {
                let off = s.u32_c(it * wg_size);
                s.uadd(io.tid, off)
            };
            let n = s.shr(u_idx, c_3);
            let u = s.and(u_idx, c_7);
            let g_r = s.uadd(io.wg_n_base, n);
            let in_r = s.ult(g_r, io.p_rows);
            let safe_r = s.select(s.t_u32, in_r, g_r, io.c_0);

            let r_base = s.umul(safe_r, row_blk_stride);
            let b_off = s.umul(blk_idx, c_blk_u32);
            let blk = s.uadd(r_base, b_off);

            let pdm = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, blk]);
            let blk1 = s.uadd(blk, c_1);
            let blk2 = s.uadd(blk, c_2);
            let blk3 = s.uadd(blk, c_3);
            let ps0 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, blk1]);
            let ps1 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, blk2]);
            let ps2 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, blk3]);

            let dm_raw = s.load(s.t_u32, pdm);
            let (d, min_d) = s.unpack2x16f(dm_raw);
            let s0 = s.load(s.t_u32, ps0);
            let s1 = s.load(s.t_u32, ps1);
            let s2 = s.load(s.t_u32, ps2);

            let qhw = if is_q5k {
                let qh_u = s.uadd(c_4, u);
                let qh_off = s.uadd(blk, qh_u);
                let pqh = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, qh_off]);
                s.load(s.t_u32, pqh)
            } else {
                0
            };

            let k_u_base = s.shl(u, c_2);
            let n_base = if b_col_major { s.umul(n, c_kstride_b) } else { n };

            let num_chunks = if is_k128 { 2u32 } else { 1u32 };
            for ch in 0..num_chunks {
                let c_ch = s.u32_c(ch);
                let chunk_cur = if ch == 0 { base_chunk } else { s.uadd(base_chunk, c_ch) };
                let (sh0, sh1, chunk_u_base, is_hi_chunk) = if is_k128 {
                    let half_b = s.shr(base_chunk, c_1);
                    let is_hi = s.ult(io.c_0, half_b);
                    let sh_a = if ch == 0 { io.c_0 } else { c_16 };
                    let sh_b = if ch == 0 { c_8 } else { c_24 };
                    let chunk_sh3 = s.shl(chunk_cur, c_3);
                    let c_base = s.uadd(c_qs_base, chunk_sh3);
                    (sh_a, sh_b, c_base, is_hi)
                } else {
                    let chunk_mul8 = s.shl(base_chunk, c_3);
                    let c_base = s.uadd(c_qs_base, chunk_mul8);
                    let is_hi = s.ult(c_1, base_chunk);
                    let chunk1 = s.and(base_chunk, c_1);
                    let sh_a = s.shl(chunk1, c_4);
                    let sh_b = s.uadd(sh_a, c_8);
                    (sh_a, sh_b, c_base, is_hi)
                };

                let u_in_blk = s.uadd(chunk_u_base, u);
                let qw_off = s.uadd(blk, u_in_blk);
                let pqw = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, qw_off]);
                let qw = s.load(s.t_u32, pqw);

                let sc0_lo = s.uextract(s0, sh0, c_6);
                let mn0_lo = s.uextract(s1, sh0, c_6);
                let sc1_lo = s.uextract(s0, sh1, c_6);
                let mn1_lo = s.uextract(s1, sh1, c_6);

                let sh0_p2 = s.uadd(sh0, c_2);
                let sh0_p4 = s.uadd(sh0, c_4);
                let s2_e0 = s.uextract(s2, sh0, c_4);
                let s0_sh0 = s.shr(s0, sh0_p2);
                let s0_hi0 = s.and(s0_sh0, c_0x30);
                let sc0_hi = s.or(s2_e0, s0_hi0);
                let s2_m0 = s.uextract(s2, sh0_p4, c_4);
                let s1_sh0 = s.shr(s1, sh0_p2);
                let s1_hi0 = s.and(s1_sh0, c_0x30);
                let mn0_hi = s.or(s2_m0, s1_hi0);

                let sh1_p2 = s.uadd(sh1, c_2);
                let sh1_p4 = s.uadd(sh1, c_4);
                let s2_e1 = s.uextract(s2, sh1, c_4);
                let s0_sh1 = s.shr(s0, sh1_p2);
                let s0_hi1 = s.and(s0_sh1, c_0x30);
                let sc1_hi = s.or(s2_e1, s0_hi1);
                let s2_m1 = s.uextract(s2, sh1_p4, c_4);
                let s1_sh1 = s.shr(s1, sh1_p2);
                let s1_hi1 = s.and(s1_sh1, c_0x30);
                let mn1_hi = s.or(s2_m1, s1_hi1);

                let sc0_u = s.select(s.t_u32, is_hi_chunk, sc0_hi, sc0_lo);
                let mn0_u = s.select(s.t_u32, is_hi_chunk, mn0_hi, mn0_lo);
                let sc1_u = s.select(s.t_u32, is_hi_chunk, sc1_hi, sc1_lo);
                let mn1_u = s.select(s.t_u32, is_hi_chunk, mn1_hi, mn1_lo);

                let sc0_f = s.u2f(sc0_u);
                let mn0_f = s.u2f(mn0_u);
                let sc1_f = s.u2f(sc1_u);
                let mn1_f = s.u2f(mn1_u);
                let w1 = s.fmul(d, sc0_f);
                let m1 = s.fmul(min_d, mn0_f);
                let w2 = s.fmul(d, sc1_f);
                let m2 = s.fmul(min_d, mn1_f);

                let qh_bit_lo = if is_q5k { s.shl(chunk_cur, c_1) } else { 0 };
                let qh_bit_hi = if is_q5k { s.uadd(qh_bit_lo, c_1) } else { 0 };

                let ch_k_off = ch * 64;
                for b_byte in 0..4u32 {
                    let bit_lo = s.u32_c(b_byte * 8);
                    let bit_hi = s.u32_c(b_byte * 8 + 4);
                    let mut q_lo_u = s.uextract(qw, bit_lo, c_4);
                    let mut q_hi_u = s.uextract(qw, bit_hi, c_4);
                    if is_q5k {
                        let hb_lo_pos = s.uadd(bit_lo, qh_bit_lo);
                        let hb_hi_pos = s.uadd(bit_lo, qh_bit_hi);
                        let hb_lo = s.uextract(qhw, hb_lo_pos, c_1);
                        let hb_hi = s.uextract(qhw, hb_hi_pos, c_1);
                        let hi1 = s.shl(hb_lo, c_4);
                        let hi2 = s.shl(hb_hi, c_4);
                        q_lo_u = s.or(q_lo_u, hi1);
                        q_hi_u = s.or(q_hi_u, hi2);
                    }
                    let q_lo = s.u2f(q_lo_u);
                    let q_hi = s.u2f(q_hi_u);
                    let p_lo_f = s.fmul(w1, q_lo);
                    let p_hi_f = s.fmul(w2, q_hi);
                    let v_lo_f32 = s.fsub(p_lo_f, m1);
                    let v_hi_f32 = s.fsub(p_hi_f, m2);
                    let v_lo_f16 = s.f32_to_f16(v_lo_f32);
                    let v_hi_f16 = s.f32_to_f16(v_hi_f32);
                    let v_lo = s.select(s.t_f16, in_r, v_lo_f16, s.f16_zero);
                    let v_hi = s.select(s.t_f16, in_r, v_hi_f16, s.f16_zero);

                    let c_b = s.u32_c(ch_k_off + b_byte);
                    let k_lo = s.uadd(k_u_base, c_b);
                    let k_hi = s.uadd(k_lo, c_32);
                    emit_store_sh_b(s, io, b_col_major, c_kstride_b, k_lo, n_base, v_lo);
                    emit_store_sh_b(s, io, b_col_major, c_kstride_b, k_hi, n_base, v_hi);
                }
            }
        }
    })
}

/// Generate a SPIR-V 1.5 Intel XMX (`8x16x16`) **4-byte-aligned `Q6_K × F32` GEMM** (`k_tile = 128`).
pub fn spirv_xmx_q6_k(cfg: XmxTileCfg, b_col_major: bool) -> Vec<u32> {
    assert_eq!(cfg.k_tile, 128, "Q6_K XMX expects k_tile = 128 (one Q6_K half-block)");
    emit_xmx_gemm_loop(cfg, false, b_col_major, |s, io, k_phi| {
        let wg_size = cfg.wg_size();
        let c_1 = s.u32_c(1);
        let c_2 = s.u32_c(2);
        let c_3 = s.u32_c(3);
        let c_4 = s.u32_c(4);
        let c_6 = s.u32_c(6);
        let c_7 = s.u32_c(7);
        let c_8 = s.u32_c(8);
        let c_16 = s.u32_c(16);
        let c_32 = s.u32_c(32);
        let c_48 = s.u32_c(48);
        let c_52 = s.u32_c(52);
        let c_53 = s.u32_c(53);
        let c_64 = s.u32_c(64);
        let c_96 = s.u32_c(96);
        let c_0xf = s.u32_c(0xF);
        let c_i32_32 = s.i32_c(-32);
        let c_kstride_b = s.u32_c(io.k_stride_b);

        let nb = s.shr(io.p_cols, c_8);
        let row_blk_stride = s.umul(nb, c_53);
        let blk_idx = s.shr(k_phi, c_8);
        let k_div128 = s.shr(k_phi, c_7);
        let half = s.and(k_div128, c_1);

        let ql_half_off = s.shl(half, c_4); // half * 16
        let qh_half_off = s.shl(half, c_3); // half * 8
        let qh_base_blk = s.uadd(c_32, qh_half_off);
        let sc_half_off = s.shl(half, c_1); // half * 2
        let sc_base_blk = s.uadd(c_48, sc_half_off);

        let w_iters = (cfg.n_tile * 8) / wg_size;
        for it in 0..w_iters {
            let u_idx = if it == 0 {
                io.tid
            } else {
                let off = s.u32_c(it * wg_size);
                s.uadd(io.tid, off)
            };
            let n = s.shr(u_idx, c_3);
            let u = s.and(u_idx, c_7);
            let g_r = s.uadd(io.wg_n_base, n);
            let in_r = s.ult(g_r, io.p_rows);
            let safe_r = s.select(s.t_u32, in_r, g_r, io.c_0);

            let r_base = s.umul(safe_r, row_blk_stride);
            let b_off = s.umul(blk_idx, c_53);
            let blk = s.uadd(r_base, b_off);

            let d_idx = s.uadd(blk, c_52);
            let pd = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, d_idx]);
            let d_u = s.load(s.t_u32, pd);
            let d = s.bitcast_f32(d_u);

            let sc0_off = s.uadd(blk, sc_base_blk);
            let sc1_off = s.uadd(sc0_off, c_1);
            let psc0 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, sc0_off]);
            let psc1 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, sc1_off]);
            let sc_u0 = s.load(s.t_u32, psc0);
            let sc_u1 = s.load(s.t_u32, psc1);
            let sc_w0 = s.un(op::BITCAST, s.t_i32, sc_u0);
            let sc_w1 = s.un(op::BITCAST, s.t_i32, sc_u1);

            let u_hi = s.shr(u, c_2); // 0 if l<16, 1 if l>=16
            let is_sh = s.shl(u_hi, c_3); // 0 or 8
            let is_sh16 = s.uadd(is_sh, c_16);
            let s0_i = s.sextract(sc_w0, is_sh, c_8);
            let s2_i = s.sextract(sc_w0, is_sh16, c_8);
            let s4_i = s.sextract(sc_w1, is_sh, c_8);
            let s6_i = s.sextract(sc_w1, is_sh16, c_8);
            let s0_f = s.s2f(s0_i);
            let s2_f = s.s2f(s2_i);
            let s4_f = s.s2f(s4_i);
            let s6_f = s.s2f(s6_i);
            let ds0 = s.fmul(d, s0_f);
            let ds2 = s.fmul(d, s2_f);
            let ds4 = s.fmul(d, s4_f);
            let ds6 = s.fmul(d, s6_f);

            let ql0_in_blk = s.uadd(ql_half_off, u);
            let ql1_in_blk = s.uadd(ql0_in_blk, c_8);
            let qh_in_blk = s.uadd(qh_base_blk, u);
            let ql0_idx = s.uadd(blk, ql0_in_blk);
            let ql1_idx = s.uadd(blk, ql1_in_blk);
            let qh_idx = s.uadd(blk, qh_in_blk);
            let pql0 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, ql0_idx]);
            let pql1 = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, ql1_idx]);
            let pqh = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, qh_idx]);
            let ql0 = s.load(s.t_u32, pql0);
            let ql1 = s.load(s.t_u32, pql1);
            let qh = s.load(s.t_u32, pqh);

            let k_u_base = s.shl(u, c_2);
            let n_base = if b_col_major { s.umul(n, c_kstride_b) } else { n };

            for j in 0..4u32 {
                let sh_j = s.u32_c(j * 8);
                let q0b = s.uextract(ql0, sh_j, c_8);
                let q1b = s.uextract(ql1, sh_j, c_8);
                let qhb = s.uextract(qh, sh_j, c_8);

                let q0_lo = s.and(q0b, c_0xf);
                let q1_lo = s.and(q1b, c_0xf);
                let q0_hi = s.shr(q0b, c_4);
                let q1_hi = s.shr(q1b, c_4);

                let qh0 = s.uextract(qhb, io.c_0, c_2);
                let qh1 = s.uextract(qhb, c_2, c_2);
                let qh2 = s.uextract(qhb, c_4, c_2);
                let qh3 = s.uextract(qhb, c_6, c_2);
                let h0 = s.shl(qh0, c_4);
                let h1 = s.shl(qh1, c_4);
                let h2 = s.shl(qh2, c_4);
                let h3 = s.shl(qh3, c_4);

                let c_j = s.u32_c(j);
                let k0 = s.uadd(k_u_base, c_j);
                let k1 = s.uadd(k0, c_32);
                let k2 = s.uadd(k0, c_64);
                let k3 = s.uadd(k0, c_96);

                for (q_low, h_bits, ds, k_off) in [
                    (q0_lo, h0, ds0, k0),
                    (q1_lo, h1, ds2, k1),
                    (q0_hi, h2, ds4, k2),
                    (q1_hi, h3, ds6, k3),
                ] {
                    let q6_u = s.or(q_low, h_bits);
                    let q6_u_i = s.un(op::BITCAST, s.t_i32, q6_u);
                    let q6_i = s.bin(op::I_ADD, s.t_i32, q6_u_i, c_i32_32);
                    let q6_f = s.s2f(q6_i);
                    let w_f32 = s.fmul(ds, q6_f);
                    let w_f16 = s.f32_to_f16(w_f32);
                    let w_sel = s.select(s.t_f16, in_r, w_f16, s.f16_zero);
                    emit_store_sh_b(s, io, b_col_major, c_kstride_b, k_off, n_base, w_sel);
                }
            }
        }
    })
}

/// Generate a SPIR-V 1.5 Intel XMX (`8x16x16`) **`F16 × F32` GEMM** (`k_tile = 64`).
pub fn spirv_xmx_f16(cfg: XmxTileCfg, b_col_major: bool) -> Vec<u32> {
    assert_eq!(cfg.k_tile, 64, "F16 XMX expects k_tile = 64");
    emit_xmx_gemm_loop(cfg, false, b_col_major, |s, io, k_phi| {
        let wg_size = cfg.wg_size();
        let c_1 = s.u32_c(1);
        let c_5 = s.u32_c(5);
        let c_31 = s.u32_c(31);
        let c_kstride_b = s.u32_c(io.k_stride_b);

        let half_cols = s.shr(io.p_cols, c_1);
        let k_half = s.shr(k_phi, c_1);
        let total_u32 = cfg.n_tile * 32; // 32 u32s = 64 f16s per row
        let w_iters = total_u32 / wg_size;

        for it in 0..w_iters {
            let idx = if it == 0 {
                io.tid
            } else {
                let off = s.u32_c(it * wg_size);
                s.uadd(io.tid, off)
            };
            let n = s.shr(idx, c_5);
            let u = s.and(idx, c_31);
            let k_lo = s.shl(u, c_1);
            let k_hi = s.uadd(k_lo, c_1);
            let g_r = s.uadd(io.wg_n_base, n);
            let g_k_lo = s.uadd(k_phi, k_lo);
            let g_k_hi = s.uadd(k_phi, k_hi);

            let in_r = s.ult(g_r, io.p_rows);
            let in_k_lo = s.ult(g_k_lo, io.p_cols);
            let in_k_hi = s.ult(g_k_hi, io.p_cols);
            let valid_lo = s.land(in_r, in_k_lo);
            let valid_hi = s.land(in_r, in_k_hi);

            let safe_r = s.select(s.t_u32, in_r, g_r, io.c_0);
            let k_u_off = s.uadd(k_half, u);
            let safe_k_u = s.select(s.t_u32, in_k_lo, k_u_off, io.c_0);
            let row_off = s.umul(safe_r, half_cols);
            let w_idx = s.uadd(row_off, safe_k_u);
            let pw = s.access(io.tp_ssbo_u32, io.var_w, &[io.c_0, w_idx]);
            let packed = s.load(s.t_u32, pw);
            let v2h = s.un(op::BITCAST, s.t_v2f16, packed);
            let f0 = s.id();
            let f1 = s.id();
            Spv::inst(&mut s.fns, op::COMPOSITE_EXTRACT, &[s.t_f16, f0, v2h, 0]);
            Spv::inst(&mut s.fns, op::COMPOSITE_EXTRACT, &[s.t_f16, f1, v2h, 1]);

            let v0 = s.select(s.t_f16, valid_lo, f0, s.f16_zero);
            let v1 = s.select(s.t_f16, valid_hi, f1, s.f16_zero);
            let n_base = if b_col_major { s.umul(n, c_kstride_b) } else { n };
            emit_store_sh_b(s, io, b_col_major, c_kstride_b, k_lo, n_base, v0);
            emit_store_sh_b(s, io, b_col_major, c_kstride_b, k_hi, n_base, v1);
        }
    })
}

/// Generate a SPIR-V 1.5 Intel XMX (`8x16x16`) **`F32 × F32` GEMM** (`k_tile = 64`).
pub fn spirv_xmx_f32(cfg: XmxTileCfg, b_col_major: bool) -> Vec<u32> {
    let wg_size = cfg.wg_size();
    let k_shift = cfg.k_tile.trailing_zeros();
    let k_mask = cfg.k_tile - 1;
    emit_xmx_gemm_loop(cfg, true, b_col_major, |s, io, k_phi| {
        let c_k_shift = s.u32_c(k_shift);
        let c_k_mask = s.u32_c(k_mask);
        let c_kstride_b = s.u32_c(io.k_stride_b);
        let total = cfg.n_tile * cfg.k_tile;
        let w_iters = total / wg_size;

        for it in 0..w_iters {
            let idx = if it == 0 {
                io.tid
            } else {
                let off = s.u32_c(it * wg_size);
                s.uadd(io.tid, off)
            };
            let n = s.shr(idx, c_k_shift);
            let kk = s.and(idx, c_k_mask);
            let g_r = s.uadd(io.wg_n_base, n);
            let g_k = s.uadd(k_phi, kk);
            let in_r = s.ult(g_r, io.p_rows);
            let in_k = s.ult(g_k, io.p_cols);
            let valid = s.land(in_r, in_k);

            let safe_r = s.select(s.t_u32, in_r, g_r, io.c_0);
            let safe_k = s.select(s.t_u32, in_k, g_k, io.c_0);
            let row_off = s.umul(safe_r, io.p_cols);
            let w_idx = s.uadd(row_off, safe_k);
            let pw = s.access(io.tp_ssbo_f32, io.var_w, &[io.c_0, w_idx]);
            let wf32 = s.load(s.t_f32, pw);
            let wf = s.f32_to_f16(wf32);
            let vs = s.select(s.t_f16, valid, wf, s.f16_zero);
            let n_base = if b_col_major { s.umul(n, c_kstride_b) } else { n };
            emit_store_sh_b(s, io, b_col_major, c_kstride_b, kk, n_base, vs);
        }
    })
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::wgpu_ctx::GpuContext;
    use caelum_core::{f32_to_f16, DType, TensorBuf};
    use std::borrow::Cow;

    fn make_test_weight(dtype: DType, rows: usize, cols: usize) -> TensorBuf {
        match dtype {
            DType::Q4_K => {
                let blocks = rows * (cols / 256);
                let mut raw = vec![0u8; blocks * 144];
                let d = f32_to_f16(0.05).to_le_bytes();
                let dm = f32_to_f16(0.02).to_le_bytes();
                for blk in 0..blocks {
                    let off = blk * 144;
                    raw[off..off + 2].copy_from_slice(&d);
                    raw[off + 2..off + 4].copy_from_slice(&dm);
                    for j in 4..144 {
                        raw[off + j] = ((blk * 37 + j * 13) & 0xFF) as u8;
                    }
                }
                TensorBuf::from_raw(raw, vec![rows, cols], DType::Q4_K).unwrap()
            }
            DType::Q5_K => {
                let blocks = rows * (cols / 256);
                let mut raw = vec![0u8; blocks * 176];
                let d = f32_to_f16(0.05).to_le_bytes();
                let dm = f32_to_f16(0.02).to_le_bytes();
                for blk in 0..blocks {
                    let off = blk * 176;
                    raw[off..off + 2].copy_from_slice(&d);
                    raw[off + 2..off + 4].copy_from_slice(&dm);
                    for j in 4..176 {
                        raw[off + j] = ((blk * 41 + j * 19) & 0xFF) as u8;
                    }
                }
                TensorBuf::from_raw(raw, vec![rows, cols], DType::Q5_K).unwrap()
            }
            DType::Q6_K => {
                let blocks = rows * (cols / 256);
                let mut raw = vec![0u8; blocks * 210];
                let d = f32_to_f16(0.04).to_le_bytes();
                for blk in 0..blocks {
                    let off = blk * 210;
                    for j in 0..208 {
                        raw[off + j] = ((blk * 29 + j * 17) & 0xFF) as u8;
                    }
                    raw[off + 208..off + 210].copy_from_slice(&d);
                }
                TensorBuf::from_raw(raw, vec![rows, cols], DType::Q6_K).unwrap()
            }
            DType::F16 => {
                let mut raw = Vec::with_capacity(rows * cols * 2);
                for i in 0..rows * cols {
                    let v = (((i * 23) % 19) as f32 - 9.0) * 0.125;
                    raw.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
                TensorBuf::from_raw(raw, vec![rows, cols], DType::F16).unwrap()
            }
            DType::F32 => {
                let data: Vec<f32> = (0..rows * cols)
                    .map(|i| (((i * 23) % 19) as f32 - 9.0) * 0.125)
                    .collect();
                TensorBuf::from_f32(&data, vec![rows, cols]).unwrap()
            }
            _ => unreachable!(),
        }
    }

    fn run_xmx_shader_test(
        ctx: &GpuContext,
        dtype: DType,
        spv_words: Vec<u32>,
        cfg: XmxTileCfg,
        rows: usize,
        cols: usize,
        batch: usize,
        iters: usize,
    ) -> (f64, f32) {
        let module = unsafe {
            ctx.device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("xmx_test"),
                entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: Cow::Borrowed("main"),
                    workgroup_size: (cfg.wg_size(), 1, 1),
                }]),
                spirv: Some(Cow::Owned(spv_words)),
                ..Default::default()
            })
        };
        let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xmx_pl"),
            bind_group_layouts: &[Some(&ctx.pipelines.bgl_matmul)],
            immediate_size: 0,
        });
        let pipe = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("xmx_pipe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buf = make_test_weight(dtype, rows, cols);
        let w_view = buf.as_view();
        let x_f32: Vec<f32> = (0..batch * cols).map(|i| (((i * 13) % 11) as f32 - 5.0) * 0.1).collect();

        let (w_buf, _) = ctx.upload_weight("xmx-w", &w_view);
        let x_buf = ctx.upload_f32("xmx-x", &x_f32);
        let out_bytes = (batch * rows * 4) as u64;
        let out_buf = ctx.alloc_storage("xmx-out", out_bytes);
        let stg = ctx.alloc_staging_read("xmx-stg", out_bytes);
        let mut arena = crate::wgpu_ctx::UniformArena::new(&ctx.device);
        let bg = ctx.bind_matmul(&w_buf, &x_buf, &out_buf, &arena);
        let u_off = arena.push_u32x4(rows as u32, cols as u32, batch as u32, 0);
        arena.flush(&ctx.queue);

        let wg_x = (rows as u32).div_ceil(cfg.n_tile);
        let wg_y = (batch as u32).div_ceil(cfg.m_tile);

        // 5-dispatch warmup so GPU clocks reach steady boost
        {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &bg, &[u_off]);
                for _ in 0..5 {
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
            }
            ctx.queue.submit(Some(enc.finish()));
            let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        }

        let t0 = std::time::Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[u_off]);
            for _ in 0..iters {
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &stg, 0, out_bytes);
        ctx.queue.submit(Some(enc.finish()));
        let mut got = vec![0.0f32; batch * rows];
        ctx.read_staging_f32(&stg, &mut got).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();

        let (_, wgsl_ref) = ctx.bench_matmul(&w_view, &x_f32, batch, 1).unwrap();
        let mut max_err = 0.0f32;
        let mut max_ref = 0.0f32;
        for (a, b) in got.iter().zip(wgsl_ref.iter()) {
            max_err = max_err.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
        let max_rel = max_err / max_ref.max(1.0);
        if max_rel > 5e-3 {
            eprintln!("got[0..4]={:?} ref[0..4]={:?} max_ref={max_ref:.2}", &got[..4], &wgsl_ref[..4]);
        }
        (elapsed / iters as f64, max_rel)
    }

    #[test]
    fn test_xmx_all_dtypes_parity() {
        let Ok(ctx) = GpuContext::new() else { return; };
        if !ctx.has_xmx {
            return;
        }
        // Verify non-aligned shapes across all XMX dtypes & tile configs against WGSL reference
        for (label, dtype, spv, cfg, rows, cols, batch) in [
            ("Q4_K M8N32K128 ", DType::Q4_K, spirv_xmx_q4_k(CFG_M8_N32_K128, true), CFG_M8_N32_K128, 73, 512, 7),
            ("Q4_K M16N32K64 ", DType::Q4_K, spirv_xmx_q4_k(CFG_M16_N32_K64, true), CFG_M16_N32_K64, 73, 512, 13),
            ("Q4_K M32N64K64 ", DType::Q4_K, spirv_xmx_q4_k(CFG_M32_N64_K64, true), CFG_M32_N64_K64, 73, 512, 21),
            ("Q5_K M16N32K128", DType::Q5_K, spirv_xmx_q5_k(CFG_M16_N32_K128, true), CFG_M16_N32_K128, 73, 512, 13),
            ("Q6_K M16N32K128", DType::Q6_K, spirv_xmx_q6_k(CFG_M16_N32_K128, true), CFG_M16_N32_K128, 73, 512, 13),
            ("F16  M16N32K64 ", DType::F16, spirv_xmx_f16(CFG_M16_N32_K64, true), CFG_M16_N32_K64, 73, 128, 13),
            ("F32  M16N32K64 ", DType::F32, spirv_xmx_f32(CFG_M16_N32_K64, true), CFG_M16_N32_K64, 73, 112, 13),
        ] {
            let (_, rel) = run_xmx_shader_test(&ctx, dtype, spv, cfg, rows, cols, batch, 1);
            assert!(rel < 2e-3, "XMX {label} parity mismatch rel={rel}");
        }
    }
}
