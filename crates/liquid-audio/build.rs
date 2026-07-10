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

    if arch == "x86_64" {
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_x86.cpp");
        cc::Build::new()
            .file("native/kernels/x86_64/flashkern_x86.cpp")
            .cpp(true)
            .std("c++23")
            .opt_level(3)
            .warnings(false)
            .flag("-ffp-contract=off")
            .compile("lfm_flashkern_x86");
    } else {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            println!("cargo::rustc-link-lib=framework=Accelerate");
        }

        println!("cargo::rerun-if-changed=native/reference/bf16_gemm.c");
        cc::Build::new()
            .file("native/reference/bf16_gemm.c")
            .flag("-march=armv8.2-a+bf16")
            .opt_level(3)
            .warnings(false)
            .compile("lfm_bf16_gemm");

        println!("cargo::rerun-if-changed=native/kernels/aarch64/flashkern_neon.cpp");
        let mut kern = cc::Build::new();
        kern.file("native/kernels/aarch64/flashkern_neon.cpp")
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

    println!("cargo::rerun-if-changed=native/src/engine/flashkern_engine.cpp");
    println!("cargo::rerun-if-changed=../kcoro-sys/vendor/kcoro/include");
    // C++23, not a style choice: this TU includes kcoro headers, and C++23 is
    // the FIRST standard that requires <stdatomic.h> to work in C++ and expose
    // ::atomic_int (gcc 13 implements it only under -std=c++23; c++20 is not
    // enough). libc++/clang provides the typedefs at c++17 as an extension —
    // which is why macOS was green while the ubuntu-gcc leg was red: libc++ vs
    // libstdc++, not Apple vs Linux. All native C++ in this crate stays on the
    // same std for consistency.
    cc::Build::new()
        .file("native/src/engine/flashkern_engine.cpp")
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(false)
        .flag("-ffp-contract=off")
        .flag("-pthread")
        .include("../kcoro-sys/vendor/kcoro/include")
        .compile("lfm_flashkern_engine");
}
