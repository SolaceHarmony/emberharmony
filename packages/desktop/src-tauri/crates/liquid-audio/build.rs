// Compile the NEON micro-kernels (csrc/*.c, csrc/*.cpp) on aarch64.
// Sets `cfg(has_bf16_kernel)` / `cfg(has_neon_zoo)` so the Rust FFI is only wired in where
// the kernels were actually built. Runtime feature detection (NeonFeatures) still gates calls.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_bf16_kernel)");
    println!("cargo::rustc-check-cfg=cfg(has_neon_zoo)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "aarch64" {
        return;
    }

    // Original single-file BFMMLA GEMM (kept as the reference kernel).
    println!("cargo::rerun-if-changed=csrc/bf16_gemm.c");
    cc::Build::new()
        .file("csrc/bf16_gemm.c")
        // Arm BFloat16 extension (FEAT_BF16) → vbfmmlaq_f32 / vld1q_bf16.
        .flag("-march=armv8.2-a+bf16")
        .opt_level(3)
        .warnings(false)
        .compile("lfm_bf16_gemm");
    println!("cargo::rustc-cfg=has_bf16_kernel");

    // The NEON "zoo" (csrc/neon_zoo.cpp). Feature-specific opcodes are confined to
    // functions carrying a per-compiler target attribute (see the file header):
    //   * clang exposes the ACLE intrinsics only when the base -march enables the feature,
    //     so clang gets a base march carrying every feature the zoo uses and the in-file
    //     target-attr macros are empty.
    //   * gcc always declares the intrinsics and honours per-function `target("arch=...")`,
    //     so gcc keeps a low base march and each opcode stays isolated to its function.
    println!("cargo::rerun-if-changed=csrc/neon_zoo.cpp");
    let mut zoo = cc::Build::new();
    zoo.file("csrc/neon_zoo.cpp")
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false);
    if zoo.get_compiler().is_like_clang() {
        // v8.3 base gives FCMA; add bf16 + i8mm so every intrinsic the zoo uses is exposed.
        zoo.flag("-march=armv8.3-a+bf16+i8mm");
    } else {
        // gcc: low base. Each per-function `target("arch=…")` attribute must be a SUPERSET of the
        // base, so the base has to stay minimal — raising it to +bf16+i8mm makes the bf16-only
        // functions a subset and triggers "target specific option mismatch" on always_inline libc
        // calls (memcpy). gcc 13's arm_neon.h already declares bfloat16_t/MMLA types at global
        // scope with this base, and the per-function attributes raise the arch where needed.
        zoo.flag("-march=armv8.2-a");
    }
    zoo.compile("lfm_neon_zoo");
    println!("cargo::rustc-cfg=has_neon_zoo");
}
