//! Runtime CPU architecture & SIMD capability detection (ARMv9 + x86_64).

use std::fmt;

#[derive(Debug, Clone, Default)]
pub struct CpuCaps {
    pub arch: &'static str,
    // x86_64 extensions
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avxvnni: bool,
    pub avx512f: bool,
    // aarch64 / ARMv9 extensions
    pub neon: bool,
    pub dotprod: bool,
    pub fp16: bool,
    pub i8mm: bool,
    pub bf16: bool,
    pub sve2: bool,
}

impl CpuCaps {
    pub fn detect() -> Self {
        #[allow(unused_mut)]
        let mut caps = Self {
            arch: std::env::consts::ARCH,
            ..Default::default()
        };

        #[cfg(target_arch = "x86_64")]
        {
            caps.avx2 = std::arch::is_x86_feature_detected!("avx2");
            caps.fma = std::arch::is_x86_feature_detected!("fma");
            caps.f16c = std::arch::is_x86_feature_detected!("f16c");
            caps.avxvnni = std::arch::is_x86_feature_detected!("avxvnni");
            caps.avx512f = std::arch::is_x86_feature_detected!("avx512f");
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = std::arch::is_aarch64_feature_detected!("neon");
            caps.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            caps.fp16 = std::arch::is_aarch64_feature_detected!("fp16");
            caps.i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
            caps.bf16 = std::arch::is_aarch64_feature_detected!("bf16");
            caps.sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        }

        caps
    }

    pub fn best_kernel_name(&self) -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            if self.sve2 && self.i8mm {
                return "ARMv9 (SVE2 + I8MM + DotProd + NEON)";
            }
            if self.i8mm && self.dotprod {
                return "ARMv9 (I8MM + DotProd + NEON)";
            }
            if self.dotprod {
                return "ARMv8.2+/v9 (DotProd + NEON)";
            }
            if self.neon {
                return "AArch64 (NEON)";
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.avx2 && self.fma && self.avxvnni {
                return "x86_64 (AVX2 + FMA + AVX-VNNI)";
            }
            if self.avx2 && self.fma {
                return "x86_64 (AVX2 + FMA)";
            }
            if self.avx2 {
                return "x86_64 (AVX2)";
            }
        }

        "Scalar Portable Rust"
    }
}

impl fmt::Display for CpuCaps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "arch={} | kernel={}", self.arch, self.best_kernel_name())?;
        let mut flags = Vec::new();
        if self.avx2 { flags.push("avx2"); }
        if self.fma { flags.push("fma"); }
        if self.f16c { flags.push("f16c"); }
        if self.avxvnni { flags.push("avx_vnni"); }
        if self.avx512f { flags.push("avx512f"); }
        if self.neon { flags.push("neon"); }
        if self.dotprod { flags.push("dotprod"); }
        if self.fp16 { flags.push("fp16"); }
        if self.i8mm { flags.push("i8mm"); }
        if self.bf16 { flags.push("bf16"); }
        if self.sve2 { flags.push("sve2"); }
        if !flags.is_empty() {
            write!(f, " | simd=[{}]", flags.join(", "))?;
        }
        Ok(())
    }
}
