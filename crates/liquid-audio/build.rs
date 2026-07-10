// Compile the liquid-audio native kernels. kcoro itself is built by the sibling
// `kcoro-sys` crate; this build script owns only liquid-audio C/C++ sources and
// the cfg flags that gate their Rust FFI bridges.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_bf16_kernel)");
    println!("cargo::rustc-check-cfg=cfg(has_flashkern_neon)");
    println!("cargo::rustc-check-cfg=cfg(has_flashkern_x86)");
    println!("cargo::rustc-check-cfg=cfg(has_kcoro)");
    println!("cargo::rustc-check-cfg=cfg(has_native_engine)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if arch == "x86_64" {
        let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if env == "msvc" {
            println!(
                "cargo::warning=liquid-audio: x86 flashkern kernels not built on the MSVC toolchain \
                 (needs GCC/Clang target attributes); bf16 GEMM uses candle's f32 path here"
            );
            return;
        }
        println!("cargo::rerun-if-changed=native/kernels/x86_64/flashkern_x86.cpp");
        cc::Build::new()
            .file("native/kernels/x86_64/flashkern_x86.cpp")
            .cpp(true)
            .std("c++23")
            .opt_level(3)
            .warnings(false)
            .flag("-ffp-contract=off")
            .compile("lfm_flashkern_x86");
        println!("cargo::rustc-cfg=has_flashkern_x86");
        build_native_engine(&arch);
        return;
    }

    if arch != "aarch64" {
        return;
    }

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
    println!("cargo::rustc-cfg=has_bf16_kernel");

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
    println!("cargo::rustc-cfg=has_flashkern_neon");

    build_native_engine(&arch);
}

fn build_native_engine(arch: &str) {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }
    if !matches!(arch, "aarch64" | "x86_64") {
        return;
    }

    println!("cargo::rustc-cfg=has_kcoro");
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
    println!("cargo::rustc-cfg=has_native_engine");
}
