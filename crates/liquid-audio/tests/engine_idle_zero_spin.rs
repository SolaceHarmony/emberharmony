//! Zero-spin idle verification for the resident native decode engine.
//!
//! Companion to kcoro's `tests/test_zero_spin_idle.c` (the PR-122 oracle,
//! measured real on Darwin): the engine's lane team must be as silent parked
//! at the doorbell as bare kc_sched workers are parked on the ready queue.
//! Between token passes every lane and the SQ dispatcher sit in kcoro_arena's
//! expected-value wait-word adapter. SQ, command, and fence waits park immediately;
//! none of the paths poll.
//!
//! An integration test so the process contains ONLY this test's threads —
//! the getrusage(RUSAGE_SELF) delta is attributable to the lane team, not to
//! parallel unit tests. Gated on aarch64 macOS (getrusage semantics); the
//! engine itself is unconditional — the substrate builds or the build fails.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use liquid_audio as _;

unsafe extern "C" {
    fn lfm_bf16_gemm_available() -> i32;
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_request_stop(engine: *mut c_void);
    fn lfm_engine_free(engine: *mut c_void);
    fn lfm_engine_lanes(engine: *mut c_void) -> u32;
    fn lfm_engine_mlp(
        engine: *mut c_void,
        input: *const u16,
        norm: *const u16,
        w1: *const u16,
        w3: *const u16,
        w2: *const u16,
        output: *mut u16,
        hidden: usize,
        ffn: usize,
        epsilon: f32,
        lanes: usize,
    ) -> i32;
}

struct Engine(*mut c_void);

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            lfm_engine_request_stop(self.0);
            lfm_engine_free(self.0);
        }
    }
}

/// Process CPU time (all threads, user+system) in milliseconds.
fn proc_cpu_ms() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    assert_eq!(rc, 0, "getrusage failed");
    (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) as f64 * 1000.0
        + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) as f64 / 1000.0
}

fn idle_window_pct(window: Duration) -> f64 {
    let cpu0 = proc_cpu_ms();
    let wall0 = Instant::now();
    std::thread::sleep(window);
    let cpu1 = proc_cpu_ms();
    100.0 * (cpu1 - cpu0) / (wall0.elapsed().as_secs_f64() * 1000.0)
}

#[test]
fn engine_lanes_are_silent_at_idle() {
    // The audited eight-lane baseline is about 0.002-0.005%. Keep enough CI
    // headroom for the test harness while still detecting repeated wake/poll work.
    const IDLE_MAX_PCT: f64 = 0.1;

    let engine = Engine(unsafe { lfm_engine_new(8) });
    assert!(!engine.0.is_null(), "native engine failed to initialize");
    let lanes = unsafe { lfm_engine_lanes(engine.0) } as usize;
    assert!(lanes >= 2, "expected a real lane team, got {lanes}");

    // Let every lane reach its first doorbell park.
    std::thread::sleep(Duration::from_millis(300));

    let cold = idle_window_pct(Duration::from_secs(1));
    eprintln!("cold idle ({lanes} lanes parked): {cold:.3}% process CPU");
    assert!(
        cold < IDLE_MAX_PCT,
        "engine burns {cold:.3}% CPU while idle (limit {IDLE_MAX_PCT}%) — a lane is spinning, not parked"
    );

    // Ring the doorbell through a real typed numerical pass, then prove the team
    // re-parks instead of lingering hot. No callback-only probe exists in production.
    assert_ne!(unsafe { lfm_bf16_gemm_available() }, 0, "typed native MLP unavailable");
    let width = lanes;
    let x = vec![0u16; width];
    let norm = vec![0x3f80u16; width];
    let matrix = vec![0u16; width * width];
    let mut out = vec![1u16; width];
    assert_eq!(
        unsafe {
            lfm_engine_mlp(
                engine.0,
                x.as_ptr(),
                norm.as_ptr(),
                matrix.as_ptr(),
                matrix.as_ptr(),
                matrix.as_ptr(),
                out.as_mut_ptr(),
                width,
                width,
                1e-5,
                lanes,
            )
        },
        0,
        "typed MLP pass refused the idle probe"
    );
    assert_eq!(out, x);
    std::thread::sleep(Duration::from_millis(300));

    let reparked = idle_window_pct(Duration::from_secs(1));
    eprintln!("post-pass idle: {reparked:.3}% process CPU");
    assert!(
        reparked < IDLE_MAX_PCT,
        "engine burns {reparked:.3}% CPU after a pass (limit {IDLE_MAX_PCT}%) — the team did not re-park"
    );
}
