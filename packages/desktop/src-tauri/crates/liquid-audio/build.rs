// Compile the SIMD micro-kernels on aarch64 (NEON) and x86-64 (AVX). Sets `cfg(has_bf16_kernel)`
// / `cfg(has_flashkern_neon)` / `cfg(has_flashkern_x86)` so the Rust FFI is only wired in where a kernel was
// actually built. Runtime feature detection (NeonFeatures / X86Features) still gates calls.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_bf16_kernel)");
    println!("cargo::rustc-check-cfg=cfg(has_flashkern_neon)");
    println!("cargo::rustc-check-cfg=cfg(has_flashkern_x86)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // x86-64 flashkern (csrc/flashkern_x86.cpp) — the Intel/AMD sibling of the NEON flashkern. Every opcode is
    // confined to a per-function `target(...)` attribute (see the file header), so — on BOTH gcc
    // and clang — the TU needs NO global AVX-512 flags. That isolation is the point: global
    // -mavx512* would let clang emit zmm codegen inside the AVX2 fallback microkernel, which then
    // SIGILLs on an AVX2-only CPU that legitimately passed the AVX2 runtime gate. MSVC accepts
    // neither the attributes nor `__builtin_cpu_supports`, so on the MSVC toolchain flashkern is
    // simply not built (has_flashkern_x86 stays unset → `bf16_gemm_available()` is false → the caller
    // takes candle's own f32 path — the same arch-availability contract used off x86-64/aarch64).
    if arch == "x86_64" {
        let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if env == "msvc" {
            println!(
                "cargo::warning=liquid-audio: x86 flashkern kernels not built on the MSVC toolchain \
                 (needs GCC/Clang target attributes); bf16 GEMM uses candle's f32 path here"
            );
            return;
        }
        println!("cargo::rerun-if-changed=csrc/flashkern_x86.cpp");
        cc::Build::new()
            .file("csrc/flashkern_x86.cpp")
            .cpp(true)
            .std("c++17")
            .opt_level(3)
            .warnings(false)
            // No implicit FP contraction: the fixed-order kernels (complex_mul, depthwise3,
            // the FFT butterflies) promise separate roundings, and at -O3 the compiler would
            // otherwise fuse `a*b − c*d` into FMA. Explicit intrinsics/fmaf stay fused.
            .flag("-ffp-contract=off")
            // No global ISA flags: each function's `target(...)` attribute raises the ISA only
            // inside that function, so the AVX2 baseline stays AVX2 and AVX-512 never leaks in.
            .compile("lfm_flashkern_x86");
        println!("cargo::rustc-cfg=has_flashkern_x86");
        build_kcoro(&arch);
        return;
    }

    if arch != "aarch64" {
        return;
    }

    // Accelerate: the sanctioned dispatcher to Apple's matrix units (AMX today, SME on
    // M4+). Linked on macOS for the prefill-tile path (ENGINE_DESIGN.md E4) and its
    // benches — cblas_sgemm is declared where used; the framework is a system library.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-link-lib=framework=Accelerate");
    }

    // kcoro — the Zero-Spin Coroutine Kernel: the engine's dispatch layer (ENGINE_DESIGN.md
    // §2). Vendored under vendor/kcoro (provenance + local patches: vendor/kcoro/PATCHES.md)
    // and built here with the upstream Makefile's flags, so the engine has no machine-local
    // path dependency and carries the park-race fix (patch 0001).
    build_kcoro(&arch);

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

    // The NEON flashkern (csrc/flashkern_neon.cpp). Feature-specific opcodes are confined to
    // functions carrying a per-compiler target attribute (see the file header):
    //   * clang exposes the ACLE intrinsics only when the base -march enables the feature,
    //     so clang gets a base march carrying every feature flashkern uses and the in-file
    //     target-attr macros are empty.
    //   * gcc always declares the intrinsics and honours per-function `target("arch=...")`,
    //     so gcc keeps a low base march and each opcode stays isolated to its function.
    println!("cargo::rerun-if-changed=csrc/flashkern_neon.cpp");
    let mut kern = cc::Build::new();
    kern.file("csrc/flashkern_neon.cpp")
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false)
        // No implicit FP contraction — the fixed-order kernels promise separate roundings
        // (see the x86 build above); explicit intrinsics (vfmaq/vbfmmla/fmaf) stay fused.
        .flag("-ffp-contract=off");
    if kern.get_compiler().is_like_clang() {
        // v8.3 base gives FCMA; add bf16 + i8mm so every intrinsic flashkern uses is exposed.
        kern.flag("-march=armv8.3-a+bf16+i8mm");
    } else {
        // gcc: low base. Each per-function `target("arch=…")` attribute must be a SUPERSET of the
        // base, so the base has to stay minimal — raising it to +bf16+i8mm makes the bf16-only
        // functions a subset and triggers "target specific option mismatch" on always_inline libc
        // calls (memcpy). gcc 13's arm_neon.h already declares bfloat16_t/MMLA types at global
        // scope with this base, and the per-function attributes raise the arch where needed.
        kern.flag("-march=armv8.2-a");
    }
    kern.compile("lfm_flashkern_neon");
    println!("cargo::rustc-cfg=has_flashkern_neon");
}

// kcoro: build the vendored Zero-Spin Coroutine Kernel (vendor/kcoro — provenance and
// local patches in vendor/kcoro/PATCHES.md). Flags mirror the upstream Makefile
// (mk/common.mk + core/Makefile): C11, -O2, pthreads, KC_SCHED=1; the context switch is
// per-arch GNU assembly. Not built on MSVC (pthreads + GNU asm), so cfg(has_kcoro) stays
// unset there and the engine chassis is simply absent — the same arch-availability
// contract as the kernels.
fn build_kcoro(arch: &str) {
    println!("cargo::rustc-check-cfg=cfg(has_kcoro)");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }
    let asm = match arch {
        "aarch64" => "vendor/kcoro/arch/aarch64/kc_ctx_switch.S",
        "x86_64" => "vendor/kcoro/arch/x86_64/kc_ctx_switch.S",
        _ => return,
    };
    println!("cargo::rerun-if-changed=vendor/kcoro");
    // Same source list as the upstream core/Makefile SRCS (kc_task.c is intentionally
    // not part of the library build there either).
    let srcs = [
        "kc_chan.c",
        "kc_actor.c",
        "kc_cancel.c",
        "kc_sched.c",
        "kcoro_core.c",
        "kc_scope.c",
        "kc_select.c",
        "kc_zcopy.c",
        "kc_runtime_config.c",
        "kc_bench.c",
        "kc_dispatch.c",
    ];
    let mut b = cc::Build::new();
    for s in srcs {
        b.file(format!("vendor/kcoro/core/src/{s}"));
    }
    b.file(asm)
        .include("vendor/kcoro/include")
        .std("c11")
        .opt_level(2)
        .flag("-pthread")
        .define("_GNU_SOURCE", None)
        .define("KC_SCHED", "1")
        .warnings(false)
        .compile("kcoro");
    println!("cargo::rustc-cfg=has_kcoro");

    // The resident native decode engine (csrc/flashkern_engine.cpp): the persistent
    // C++ team — coordinator + workers as kcoro coroutines, stage kernels called
    // in-image, Rust only at the request/park rim. Needs kcoro (above) and links
    // against the flashkern kernel TU compiled elsewhere in this build.
    println!("cargo::rustc-check-cfg=cfg(has_native_engine)");
    println!("cargo::rerun-if-changed=csrc/flashkern_engine.cpp");
    cc::Build::new()
        .file("csrc/flashkern_engine.cpp")
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false)
        // Same TU policy as the kernels: the rounding ladders promise separate
        // roundings — no implicit FP contraction.
        .flag("-ffp-contract=off")
        .flag("-pthread")
        .include("vendor/kcoro/include")
        .compile("lfm_flashkern_engine");
    println!("cargo::rustc-cfg=has_native_engine");
}
