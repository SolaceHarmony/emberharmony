// Compile the NEON BFMMLA bf16 GEMM micro-kernel (csrc/bf16_gemm.c) on aarch64.
// Sets `cfg(has_bf16_kernel)` so the Rust FFI in src/bf16_gemm.rs is only wired in
// where the kernel was actually built. Runtime FEAT_BF16 detection still gates calls.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_bf16_kernel)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "aarch64" {
        println!("cargo::rerun-if-changed=csrc/bf16_gemm.c");
        cc::Build::new()
            .file("csrc/bf16_gemm.c")
            // Arm BFloat16 extension (FEAT_BF16) → vbfmmlaq_f32 / vld1q_bf16.
            .flag("-march=armv8.2-a+bf16")
            .opt_level(3)
            .warnings(false)
            .compile("lfm_bf16_gemm");
        println!("cargo::rustc-cfg=has_bf16_kernel");
    }
}
