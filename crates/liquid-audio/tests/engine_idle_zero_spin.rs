//! Zero-spin idle verification for the resident native decode engine.
//!
//! Companion to kcoro's `tests/test_zero_spin_idle.c` (the PR-122 oracle,
//! measured real on Darwin): the engine's lane team must be as silent parked
//! at the doorbell as bare kc_sched workers are parked on the ready queue.
//! Between token passes every lane sits in an untimed `pthread_cond_wait`
//! (kcoro patch 0005); the bounded ~100µs fence spin must have expired into a
//! park, never a poll.
//!
//! An integration test so the process contains ONLY this test's threads —
//! the getrusage(RUSAGE_SELF) delta is attributable to the lane team, not to
//! parallel unit tests. Gated on aarch64 macOS, where the engine build cfgs
//! (has_kcoro + has_native_engine + has_flashkern_neon) always hold; absent
//! engine here is a build regression, so the test panics rather than skips.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::time::{Duration, Instant};

use liquid_audio::flashkern::native_engine::process_engine;

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
    const IDLE_MAX_PCT: f64 = 0.5;

    let engine = process_engine()
        .expect("native engine required on aarch64 macOS — kcoro link/init regressed");
    let lanes = engine.lanes_total();
    assert!(lanes >= 2, "expected a real lane team, got {lanes}");

    // Let every lane reach its first doorbell park.
    std::thread::sleep(Duration::from_millis(300));

    let cold = idle_window_pct(Duration::from_secs(1));
    eprintln!("cold idle ({lanes} lanes parked): {cold:.3}% process CPU");
    assert!(
        cold < IDLE_MAX_PCT,
        "engine burns {cold:.3}% CPU while idle (limit {IDLE_MAX_PCT}%) — a lane is spinning, not parked"
    );

    // Ring the doorbell once — a full REQ_CALL pass with a no-op program —
    // then prove the team re-parks instead of lingering hot.
    let ran = engine.run_lanes(|_lane| {});
    assert!(ran, "run_lanes refused a trivial program");
    std::thread::sleep(Duration::from_millis(300));

    let reparked = idle_window_pct(Duration::from_secs(1));
    eprintln!("post-pass idle: {reparked:.3}% process CPU");
    assert!(
        reparked < IDLE_MAX_PCT,
        "engine burns {reparked:.3}% CPU after a pass (limit {IDLE_MAX_PCT}%) — the team did not re-park"
    );
}
