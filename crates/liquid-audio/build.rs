// Compile the liquid-audio native kernels. kcoro itself is built by the sibling
// `kcoro-sys` crate; this build script owns only liquid-audio C/C++ sources.
//
// kcoro + the native engine are the SUBSTRATE, not an optional acceleration:
// there is no degraded build and no custom cfg flags ("did the kernel
// compile?" flags were fallback logic in disguise — if a kernel can't build,
// the BUILD fails, loudly). Arch-specific Rust code gates on plain
// #[cfg(target_arch)]; everything else is unconditional.
fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if !matches!(arch.as_str(), "aarch64" | "x86_64") {
        panic!(
            "liquid-audio: unsupported target arch '{arch}' — the kcoro native \
             engine requires aarch64 or x86_64 (GCC/Clang)"
        );
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        panic!(
            "liquid-audio: MSVC toolchain unsupported — the flashkern kernels \
             and kcoro engine need GCC/Clang target attributes. Build with the \
             GNU toolchain on Windows, or on macOS/Linux."
        );
    }

    // Accelerate is a macOS fact, not an aarch64 fact: the Mimi kernel calls
    // cblas under __APPLE__ on BOTH arches (review P1: an x86_64-apple-darwin
    // link dies on cblas_sgemm$NEWLAPACK with this directive inside the arm
    // branch only).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-link-lib=framework=Accelerate");
        println!("cargo::rustc-link-lib=framework=Security");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo::rustc-link-lib=bcrypt");
    }
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix")
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
    {
        println!("cargo::rustc-link-lib=m");
    }

    println!("cargo::rerun-if-changed=native/src/engine/flashkern_engine.cpp");
    println!("cargo::rerun-if-changed=native/src/model/lfm_model.cpp");
    println!("cargo::rerun-if-changed=native/include/flashkern_conv.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_depth.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_fft.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_gemm.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_math.h");
    println!("cargo::rerun-if-changed=native/include/lfm_model.h");
    println!("cargo::rerun-if-changed=native/include/lfm_model_plan.h");
    println!("cargo::rerun-if-changed=../kcoro-sys/vendor/kcoro_arena/include");
    // C++23, not a style choice: this TU includes kcoro headers, and C++23 is
    // the FIRST standard that requires <stdatomic.h> to work in C++ and expose
    // ::atomic_int (gcc 13 implements it only under -std=c++23; c++20 is not
    // enough). libc++/clang provides the typedefs at c++17 as an extension —
    // which is why macOS was green while the ubuntu-gcc leg was red: libc++ vs
    // libstdc++, not Apple vs Linux. All native C++ in this crate stays on the
    // same std for consistency.
    cc::Build::new()
        .file("native/src/engine/flashkern_engine.cpp")
        .file("native/src/model/lfm_model.cpp")
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(false)
        .flag("-ffp-contract=off")
        .flag("-pthread")
        .include("native/include")
        .include("native/vendor")
        .include("../kcoro-sys/vendor/kcoro_arena/include")
        .compile("lfm_flashkern_engine");

    // Private Rust-kcoro/native docking leaf. The C++ translation unit owns the
    // ring atomics and expected-value doorbells; the C anchor makes header
    // compatibility and all layout assertions part of every build.
    println!("cargo::rerun-if-changed=native/include/lfm_kernel_bridge.h");
    println!("cargo::rerun-if-changed=native/src/runtime/kernel_bridge.cpp");
    println!("cargo::rerun-if-changed=native/src/runtime/kernel_protocol_c.c");
    cc::Build::new()
        .file("native/src/runtime/kernel_bridge.cpp")
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(true)
        .warnings_into_errors(true)
        .flag("-pthread")
        .include("native/include")
        .include("../kcoro-sys/vendor/kcoro_arena/include")
        .compile("lfm_kernel_bridge");
    cc::Build::new()
        .file("native/src/runtime/kernel_protocol_c.c")
        .std("c11")
        .warnings(true)
        .warnings_into_errors(true)
        .include("native/include")
        .compile("lfm_kernel_protocol_c");

    // Snapshotable ChaCha20 CSPRNG state/refill. Apple entropy enters through a
    // tiny architecture assembly thunk to SecRandomCopyBytes; every hot draw is
    // expanded by the assembly block kernel added to the architecture archive.
    println!("cargo::rerun-if-changed=native/include/flashkern_prng.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_rope.h");
    println!("cargo::rerun-if-changed=native/include/flashkern_sampler.h");
    println!("cargo::rerun-if-changed=native/src/engine/flashkern_prng.cpp");
    cc::Build::new()
        .file("native/src/engine/flashkern_prng.cpp")
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(true)
        .warnings_into_errors(true)
        .include("native/include")
        .compile("lfm_flashkern_prng");

    // Flashkern's engine archive consumes the architecture kernels below. Keep
    // the provider after the consumer so GNU ld sees its symbols while they are
    // unresolved; Apple ld happens to tolerate the opposite order.
    if arch == "x86_64" {
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_x86.cpp");
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_prng.S");
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_rope.S");
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_math.S");
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_sampler.S");
        cc::Build::new()
            .file("native/kernels/x86_64/flashkern_x86.cpp")
            .file("native/kernels/x86_64/flashkern_prng.S")
            .file("native/kernels/x86_64/flashkern_rope.S")
            .file("native/kernels/x86_64/flashkern_math.S")
            .file("native/kernels/x86_64/flashkern_sampler.S")
            .cpp(true)
            .std("c++23")
            .opt_level(3)
            .warnings(false)
            .flag("-ffp-contract=off")
            .compile("lfm_flashkern_x86");
    } else {
        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_neon.cpp");
        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_prng.S");
        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_rope.S");
        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_math.S");
        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_sampler.S");
        let mut kern = cc::Build::new();
        kern.file("native/kernels/aarch64/flashkern_neon.cpp")
            .file("native/kernels/aarch64/flashkern_prng.S")
            .file("native/kernels/aarch64/flashkern_rope.S")
            .file("native/kernels/aarch64/flashkern_math.S")
            .file("native/kernels/aarch64/flashkern_sampler.S")
            .cpp(true)
            .std("c++23")
            .opt_level(3)
            .warnings(false)
            .flag("-ffp-contract=off");
        if kern.get_compiler().is_like_clang() {
            kern.flag("-march=armv8.3-a+bf16+i8mm");
        } else {
            kern.flag("-march=armv8.2-a");
        }
        kern.compile("lfm_flashkern_neon");
    }

    // The native Mimi decode kernel (docs/MIMI_PORT.md): five active units;
    // mimi_kv.cpp stays parked (the streaming path owns a RotatingKvCache port
    // inside mimi_transformer.cpp). -ffp-contract=off is LOAD-BEARING here:
    // the scalar parity siblings are only oracles of the Rust reference if
    // clang can't contract a*b+c into fma (rustc never does).
    println!("cargo::rerun-if-changed=native/src/mimi");
    cc::Build::new()
        .files([
            "native/src/mimi/mimi_quant.cpp",
            "native/src/mimi/mimi_conv.cpp",
            "native/src/mimi/mimi_seanet.cpp",
            "native/src/mimi/mimi_transformer.cpp",
            "native/src/mimi/mimi_decode.cpp",
        ])
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(false)
        .flag("-ffp-contract=off")
        .include("native/include")
        .include("native/src/mimi")
        .compile("lfm_mimi");

    // Native checkpoint ownership: whole safetensors shards are read directly
    // into one aligned resident image, then exposed as immutable tensor views.
    // Build this archive after Mimi because Mimi's from-file constructor calls
    // into it (static archive consumers must precede providers on GNU ld).
    println!("cargo::rerun-if-changed=native/src/io/safetensors.cpp");
    println!("cargo::rerun-if-changed=native/include/lfm_safetensors.h");
    println!("cargo::rerun-if-changed=native/vendor/nlohmann");
    cc::Build::new()
        .file("native/src/io/safetensors.cpp")
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(false)
        .include("native/include")
        .include("native/vendor")
        .compile("lfm_safetensors");
}
