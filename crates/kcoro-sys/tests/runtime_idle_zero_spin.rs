//! Native worker silence gate recovered from the pre-prune zero-spin oracle.
//!
//! This integration-test process owns one bounded kcoro pool. CPU accounting
//! is read from those native worker threads themselves, so Rust executor or
//! harness activity cannot make a spinning coroutine look dormant.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

const ABI: u32 = 1;
const IDLE_MAX_PCT: f64 = 0.5;

#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

type Step = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type Complete = unsafe extern "C" fn(*mut c_void, *const Ticket);

#[repr(C)]
struct ContConfig {
    size: u32,
    abi_version: u32,
    step: Option<Step>,
    argument: *mut c_void,
    frame_size: usize,
    worker_mask: u64,
    completion: Option<Complete>,
    completion_context: *mut c_void,
}

struct Edge {
    entered: AtomicBool,
    completed: AtomicBool,
    lock: Mutex<()>,
    changed: Condvar,
}

impl Edge {
    fn wait(&self, predicate: impl Fn() -> bool, message: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut guard = self.lock.lock().unwrap();
        while !predicate() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "{message}");
            guard = self.changed.wait_timeout(guard, remaining).unwrap().0;
        }
    }
}

unsafe extern "C" fn step(raw: *mut c_void) -> *mut c_void {
    let edge = unsafe { &*koro_cont_argument(raw).cast::<Edge>() };
    match unsafe { koro_cont_state_get(raw) } {
        0 => {
            edge.entered.store(true, Ordering::Release);
            edge.changed.notify_all();
            unsafe { koro_cont_state_set(raw, 1, 0) };
            std::ptr::null_mut()
        }
        1 => {
            unsafe { koro_cont_finish(raw) };
            1_usize as *mut c_void
        }
        state => panic!("unexpected resume point {state}"),
    }
}

unsafe extern "C" fn completed(context: *mut c_void, identity: *const Ticket) {
    let edge = unsafe { &*context.cast::<Edge>() };
    assert!(!identity.is_null());
    edge.completed.store(true, Ordering::Release);
    edge.changed.notify_all();
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_runtime_worker_cpu_ns_for_test(runtime: *mut c_void, out: *mut u64) -> i32;
    fn koro_cont_create_on(
        runtime: *mut c_void,
        config: *const ContConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn koro_cont_start(cont: *mut c_void) -> i32;
    fn koro_cont_resume(cont: *mut c_void, identity: *const Ticket) -> i32;
    fn koro_cont_identity(cont: *const c_void) -> Ticket;
    fn koro_cont_destroy(cont: *mut c_void) -> i32;
    fn koro_cont_argument(cont: *mut c_void) -> *mut c_void;
    fn koro_cont_state_get(cont: *const c_void) -> u32;
    fn koro_cont_state_set(cont: *mut c_void, state: u32, suspend_kind: u32);
    fn koro_cont_finish(cont: *mut c_void);
}

fn cpu(runtime: *mut c_void) -> u64 {
    let mut value = 0;
    assert_eq!(unsafe { kc_runtime_worker_cpu_ns_for_test(runtime, &mut value) }, 0);
    value
}

fn idle(runtime: *mut c_void, window: Duration) -> f64 {
    let before = cpu(runtime);
    let wall = Instant::now();
    std::thread::sleep(window);
    let elapsed = wall.elapsed().as_secs_f64();
    let after = cpu(runtime);
    100.0 * after.saturating_sub(before) as f64 / (elapsed * 1_000_000_000.0)
}

#[test]
fn bounded_workers_are_silent_before_and_after_a_callback_resume() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 4,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);

    std::thread::sleep(Duration::from_millis(300));
    let cold = idle(runtime, Duration::from_secs(1));
    eprintln!("cold kcoro pool idle: {cold:.3}% native-worker CPU");
    assert!(cold < IDLE_MAX_PCT, "kcoro pool spins cold at {cold:.3}% CPU");

    let edge = Edge {
        entered: AtomicBool::new(false),
        completed: AtomicBool::new(false),
        lock: Mutex::new(()),
        changed: Condvar::new(),
    };
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(step),
        argument: (&edge as *const Edge).cast_mut().cast(),
        frame_size: 32,
        worker_mask: 0,
        completion: Some(completed),
        completion_context: (&edge as *const Edge).cast_mut().cast(),
    };
    let mut continuation = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut continuation) },
        0
    );
    let identity = unsafe { koro_cont_identity(continuation) };
    assert_eq!(unsafe { koro_cont_start(continuation) }, 0);
    edge.wait(
        || edge.entered.load(Ordering::Acquire),
        "continuation never suspended",
    );
    assert_eq!(unsafe { koro_cont_resume(continuation, &identity) }, 0);
    edge.wait(
        || edge.completed.load(Ordering::Acquire),
        "correlated callback never completed the continuation",
    );
    assert_eq!(unsafe { koro_cont_destroy(continuation) }, 0);

    std::thread::sleep(Duration::from_millis(300));
    let reparked = idle(runtime, Duration::from_secs(1));
    eprintln!("post-resume kcoro pool idle: {reparked:.3}% native-worker CPU");
    assert!(
        reparked < IDLE_MAX_PCT,
        "kcoro pool did not repark after callback: {reparked:.3}% CPU"
    );

    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}
