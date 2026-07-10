fn main() {
    println!("cargo::rerun-if-changed=vendor/kcoro");

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let asm = match arch.as_str() {
        "aarch64" => "vendor/kcoro/arch/aarch64/kc_ctx_switch.S",
        "x86_64" => "vendor/kcoro/arch/x86_64/kc_ctx_switch.S",
        _ => return,
    };

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

    let mut build = cc::Build::new();
    for src in srcs {
        build.file(format!("vendor/kcoro/core/src/{src}"));
    }
    build
        .file(asm)
        .include("vendor/kcoro/include")
        .std("c11")
        .opt_level(2)
        .flag("-pthread")
        .define("_GNU_SOURCE", None)
        .define("KC_SCHED", "1")
        .warnings(false)
        .compile("kcoro");
}
