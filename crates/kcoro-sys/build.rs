fn main() {
    println!("cargo::rerun-if-changed=vendor/kcoro_arena");

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }

    let srcs = [
        "kcoro_stackless.c",
        "kc_runtime.c",
        "kc_collective.c",
        "kc_doorbell.c",
        "kc_team.c",
        "kc_op.c",
        "kc_ticket.c",
        "koro_sched_stackless.c",
        "kc_chan_stackless.c",
        "kc_desc.c",
        "kc_timer.c",
        "kc_scope.c",
        "kc_actor.c",
        "kc_cancel.c",
        "kc_admin.c",
        "kc_wal.c",
        "kc_checkpoint.c",
        "kc_durable.c",
        "kc_workflow.c",
        "kc_shared.c",
        "kc_transport.c",
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
