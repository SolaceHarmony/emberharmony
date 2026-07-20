fn main() {
    println!("cargo::rerun-if-changed=vendor/kcoro_arena");

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }

    let srcs = [
        "kc_continuation.c",
        "kc_runtime.c",
        "kc_service.c",
        "kc_doorbell.c",
        "kc_team.c",
        "kc_fixed_scope.c",
        "kc_deadline.c",
    ];

    let mut core = cc::Build::new();
    for src in srcs {
        core.file(format!("vendor/kcoro_arena/core/src/{src}"));
    }
    core.include("vendor/kcoro_arena/include")
        .include("vendor/kcoro_arena/core/src")
        .std("c11")
        .opt_level(2)
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .compile("kcoro_arena_core");

    cc::Build::new()
        .file("vendor/kcoro_arena/port/posix.c")
        .include("vendor/kcoro_arena/include")
        .std("c11")
        .opt_level(2)
        .flag("-pthread")
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .compile("kcoro_arena_port_posix");
}
