#![cfg(target_os = "macos")]

use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

#[repr(C)]
struct RuntimeSnapshot {
    active: usize,
    queued: usize,
    running: usize,
    dormant: usize,
    wake_requests: u64,
    resumes: u64,
    workers: u32,
    accepting: u32,
    started: u32,
    stop_requested: u32,
}

unsafe extern "C" {
    fn lfm_engine_new_status(workers: i32, out_status: *mut i32) -> *mut c_void;
    fn lfm_engine_free(engine: *mut c_void);
    fn lfm_internal_engine_stackless_runtime_for_test(
        engine: *mut c_void,
        runtime: *mut RuntimeSnapshot,
        bridge: *mut Ticket,
    ) -> i32;
}

#[test]
fn production_engine_mounts_lanes_and_bridge_on_one_bounded_pool() {
    let _ = liquid_audio::NativeVoiceSampling::default();
    unsafe { libc::alarm(10) };
    let mut status = 0;
    let engine = unsafe { lfm_engine_new_status(4, &mut status) };
    assert!(
        !engine.is_null(),
        "production engine creation failed: {status}"
    );

    let mut runtime = RuntimeSnapshot {
        active: 0,
        queued: 0,
        running: 0,
        dormant: 0,
        wake_requests: 0,
        resumes: 0,
        workers: 0,
        accepting: 0,
        started: 0,
        stop_requested: 0,
    };
    let mut bridge = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_internal_engine_stackless_runtime_for_test(engine, &mut runtime, &mut bridge)
        },
        0
    );
    assert_eq!(runtime.workers, 4);
    assert_eq!(runtime.started, 1);
    assert_eq!(runtime.accepting, 1);
    assert!(
        runtime.active >= 7,
        "four lanes plus bridge/route/supervisor must be logical frames: {}",
        runtime.active
    );
    assert_ne!(bridge.runtime_epoch, 0);
    assert_ne!(bridge.sequence, 0);
    assert_ne!(bridge.generation, 0);
    assert_eq!(bridge.kind, 8, "bridge return address must be CONTROL");

    unsafe { lfm_engine_free(engine) };
    unsafe { libc::alarm(0) };
}
