use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

const ABI: u32 = 1;

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const ESTALE: i32 = 70;
#[cfg(any(target_os = "linux", target_os = "android"))]
const ESTALE: i32 = 116;

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

#[repr(C)]
#[derive(Default)]
struct Frame {
    marker: u64,
    first_worker: u32,
    second_worker: u32,
    resumes: u32,
}

struct Signal {
    stage: Mutex<u32>,
    changed: Condvar,
    completed: AtomicBool,
}

impl Signal {
    fn new() -> Self {
        Self {
            stage: Mutex::new(0),
            changed: Condvar::new(),
            completed: AtomicBool::new(false),
        }
    }

    fn publish(&self, stage: u32) {
        *self.stage.lock().unwrap() = stage;
        self.changed.notify_all();
    }

    fn wait(&self, expected: u32) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stage = self.stage.lock().unwrap();
        while *stage < expected {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "continuation callback timed out");
            stage = self.changed.wait_timeout(stage, remaining).unwrap().0;
        }
    }

    fn wait_completed(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stage = self.stage.lock().unwrap();
        while !self.completed.load(Ordering::Acquire) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "continuation retirement timed out");
            stage = self.changed.wait_timeout(stage, remaining).unwrap().0;
        }
    }
}

unsafe extern "C" fn frame_step(raw: *mut c_void) -> *mut c_void {
    let signal = unsafe { &*koro_cont_argument(raw).cast::<Signal>() };
    let frame = unsafe { &mut *koro_cont_frame(raw).cast::<Frame>() };
    match unsafe { koro_cont_state_get(raw) } {
        0 => {
            frame.marker = 0x51a7_1e55_c0de_f00d;
            frame.first_worker = unsafe { koro_cont_current_worker(raw) };
            frame.resumes = 1;
            signal.publish(1);
            unsafe { koro_cont_state_set(raw, 1, 0) };
            std::ptr::null_mut()
        }
        1 => {
            assert_eq!(frame.marker, 0x51a7_1e55_c0de_f00d);
            frame.second_worker = unsafe { koro_cont_current_worker(raw) };
            frame.resumes += 1;
            signal.publish(2);
            unsafe { koro_cont_state_set(raw, 2, 0) };
            std::ptr::null_mut()
        }
        2 => {
            assert_eq!(frame.marker, 0x51a7_1e55_c0de_f00d);
            assert_eq!(frame.resumes, 2);
            frame.resumes += 1;
            signal.publish(3);
            unsafe { koro_cont_finish(raw) };
            1usize as *mut c_void
        }
        state => panic!("unexpected continuation PC {state}"),
    }
}

unsafe extern "C" fn completed(context: *mut c_void, identity: *const Ticket) {
    let signal = unsafe { &*context.cast::<Signal>() };
    assert!(!identity.is_null());
    signal.completed.store(true, Ordering::Release);
    signal.changed.notify_all();
}

unsafe extern "C" fn exact_step(raw: *mut c_void) -> *mut c_void {
    let signal = unsafe { &*koro_cont_argument(raw).cast::<Signal>() };
    match unsafe { koro_cont_state_get(raw) } {
        0 => {
            signal.publish(1);
            unsafe { koro_cont_state_set(raw, 1, 0) };
            std::ptr::null_mut()
        }
        1 => {
            signal.publish(2);
            unsafe { koro_cont_finish(raw) };
            1_usize as *mut c_void
        }
        state => panic!("unexpected exact-correlation resume point {state}"),
    }
}

struct Blocker {
    entered: AtomicBool,
    completed: AtomicBool,
    release: Mutex<bool>,
    changed: Condvar,
}

unsafe extern "C" fn blocker_completed(context: *mut c_void, identity: *const Ticket) {
    let blocker = unsafe { &*context.cast::<Blocker>() };
    assert!(!identity.is_null());
    blocker.completed.store(true, Ordering::Release);
    blocker.changed.notify_all();
}

unsafe extern "C" fn blocker_step(raw: *mut c_void) -> *mut c_void {
    let blocker = unsafe { &*koro_cont_argument(raw).cast::<Blocker>() };
    blocker.entered.store(true, Ordering::Release);
    blocker.changed.notify_all();
    let mut release = blocker.release.lock().unwrap();
    while !*release {
        release = blocker.changed.wait(release).unwrap();
    }
    unsafe { koro_cont_finish(raw) };
    1usize as *mut c_void
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn koro_cont_create_on(
        runtime: *mut c_void,
        config: *const ContConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn koro_cont_start(cont: *mut c_void) -> i32;
    fn koro_cont_resume(cont: *mut c_void, identity: *const Ticket) -> i32;
    fn koro_cont_identity(cont: *const c_void) -> Ticket;
    fn koro_cont_destroy(cont: *mut c_void) -> i32;
    fn koro_cont_frame(cont: *mut c_void) -> *mut c_void;
    fn koro_cont_argument(cont: *mut c_void) -> *mut c_void;
    fn koro_cont_current_worker(cont: *const c_void) -> u32;
    fn koro_cont_state_get(cont: *const c_void) -> u32;
    fn koro_cont_state_set(cont: *mut c_void, state: u32, suspend_kind: u32);
    fn koro_cont_finish(cont: *mut c_void);
}

fn runtime(workers: u32) -> *mut c_void {
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: workers,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    runtime
}

fn destroy_runtime(runtime: *mut c_void) {
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn callback_restores_the_exact_frame_on_another_free_worker() {
    kcoro_sys::link_anchor();
    let runtime = runtime(2);
    let signal = Signal::new();
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(frame_step),
        argument: (&signal as *const Signal).cast_mut().cast(),
        frame_size: size_of::<Frame>(),
        worker_mask: 0,
        completion: Some(completed),
        completion_context: (&signal as *const Signal).cast_mut().cast(),
    };
    let mut target = std::ptr::null_mut();
    assert_eq!(unsafe { koro_cont_create_on(runtime, &config, &mut target) }, 0);
    let identity = unsafe { koro_cont_identity(target) };
    assert_ne!(identity.runtime_epoch, 0);
    assert_eq!(unsafe { koro_cont_start(target) }, 0);
    signal.wait(1);
    let frame = unsafe { &*koro_cont_frame(target).cast::<Frame>() };
    let first = frame.first_worker;

    let blocker = Blocker {
        entered: AtomicBool::new(false),
        completed: AtomicBool::new(false),
        release: Mutex::new(false),
        changed: Condvar::new(),
    };
    let blocker_config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(blocker_step),
        argument: (&blocker as *const Blocker).cast_mut().cast(),
        frame_size: 0,
        worker_mask: 1_u64 << first,
        completion: Some(blocker_completed),
        completion_context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut blocker_cont = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &blocker_config, &mut blocker_cont) },
        0
    );
    assert_eq!(unsafe { koro_cont_start(blocker_cont) }, 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = blocker.release.lock().unwrap();
    while !blocker.entered.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "affine blocker did not start");
        release = blocker.changed.wait_timeout(release, remaining).unwrap().0;
    }
    drop(release);

    let mut stale = identity;
    stale.generation = stale.generation.wrapping_add(1);
    assert_eq!(unsafe { koro_cont_resume(target, &stale) }, -ESTALE);
    assert_eq!(unsafe { koro_cont_resume(target, &identity) }, 0);
    signal.wait(2);
    let frame = unsafe { &*koro_cont_frame(target).cast::<Frame>() };
    assert_ne!(frame.first_worker, frame.second_worker);
    assert_eq!(frame.marker, 0x51a7_1e55_c0de_f00d);

    *blocker.release.lock().unwrap() = true;
    blocker.changed.notify_all();
    assert_eq!(unsafe { koro_cont_resume(target, &identity) }, 0);
    signal.wait(3);
    signal.wait_completed();

    assert_eq!(unsafe { koro_cont_destroy(target) }, 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = blocker.release.lock().unwrap();
    while !blocker.completed.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "blocker did not retire");
        release = blocker.changed.wait_timeout(release, remaining).unwrap().0;
    }
    drop(release);
    assert_eq!(unsafe { koro_cont_destroy(blocker_cont) }, 0);
    destroy_runtime(runtime);
}

struct LostWake {
    entered: AtomicBool,
    completed: AtomicBool,
    release: Mutex<bool>,
    changed: Condvar,
    stages: AtomicU32,
}


unsafe extern "C" fn lost_wake_completed(context: *mut c_void, identity: *const Ticket) {
    let state = unsafe { &*context.cast::<LostWake>() };
    assert!(!identity.is_null());
    state.completed.store(true, Ordering::Release);
    state.changed.notify_all();
}

unsafe extern "C" fn lost_wake_step(raw: *mut c_void) -> *mut c_void {
    let state = unsafe { &*koro_cont_argument(raw).cast::<LostWake>() };
    match unsafe { koro_cont_state_get(raw) } {
        0 => {
            state.entered.store(true, Ordering::Release);
            state.changed.notify_all();
            let mut release = state.release.lock().unwrap();
            while !*release {
                release = state.changed.wait(release).unwrap();
            }
            state.stages.store(1, Ordering::Release);
            unsafe { koro_cont_state_set(raw, 1, 0) };
            std::ptr::null_mut()
        }
        1 => {
            state.stages.store(2, Ordering::Release);
            state.changed.notify_all();
            unsafe { koro_cont_finish(raw) };
            1usize as *mut c_void
        }
        _ => std::ptr::null_mut(),
    }
}

#[test]
fn callback_during_running_becomes_the_next_resume_without_a_lost_edge() {
    let runtime = runtime(2);
    let state = LostWake {
        entered: AtomicBool::new(false),
        completed: AtomicBool::new(false),
        release: Mutex::new(false),
        changed: Condvar::new(),
        stages: AtomicU32::new(0),
    };
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(lost_wake_step),
        argument: (&state as *const LostWake).cast_mut().cast(),
        frame_size: 0,
        worker_mask: 0,
        completion: Some(lost_wake_completed),
        completion_context: (&state as *const LostWake).cast_mut().cast(),
    };
    let mut cont = std::ptr::null_mut();
    assert_eq!(unsafe { koro_cont_create_on(runtime, &config, &mut cont) }, 0);
    let identity = unsafe { koro_cont_identity(cont) };
    assert_eq!(unsafe { koro_cont_start(cont) }, 0);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = state.release.lock().unwrap();
    while !state.entered.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "first continuation step did not start");
        release = state.changed.wait_timeout(release, remaining).unwrap().0;
    }
    assert_eq!(unsafe { koro_cont_resume(cont, &identity) }, 0);
    *release = true;
    state.changed.notify_all();
    drop(release);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut guard = state.release.lock().unwrap();
    while state.stages.load(Ordering::Acquire) != 2 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "wake_pending edge was lost");
        guard = state.changed.wait_timeout(guard, remaining).unwrap().0;
    }
    while !state.completed.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "lost-wake continuation did not retire");
        guard = state.changed.wait_timeout(guard, remaining).unwrap().0;
    }
    drop(guard);
    assert_eq!(unsafe { koro_cont_destroy(cont) }, 0);
    destroy_runtime(runtime);
}

#[test]
fn callback_ticket_names_one_coroutine_not_a_fifo_position() {
    let runtime = runtime(2);
    let first = Signal::new();
    let second = Signal::new();
    let make = |signal: &Signal| ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(exact_step),
        argument: (signal as *const Signal).cast_mut().cast(),
        frame_size: 16,
        worker_mask: 0,
        completion: Some(completed),
        completion_context: (signal as *const Signal).cast_mut().cast(),
    };
    let mut first_cont = std::ptr::null_mut();
    let mut second_cont = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &make(&first), &mut first_cont) },
        0
    );
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &make(&second), &mut second_cont) },
        0
    );
    let first_id = unsafe { koro_cont_identity(first_cont) };
    let second_id = unsafe { koro_cont_identity(second_cont) };
    assert_ne!(first_id.sequence, second_id.sequence);
    assert_eq!(unsafe { koro_cont_start(first_cont) }, 0);
    assert_eq!(unsafe { koro_cont_start(second_cont) }, 0);
    first.wait(1);
    second.wait(1);

    assert_eq!(unsafe { koro_cont_resume(second_cont, &second_id) }, 0);
    second.wait(2);
    second.wait_completed();
    assert_eq!(
        *first.stage.lock().unwrap(),
        1,
        "resuming B must leave the neighboring A frame suspended"
    );
    assert_eq!(unsafe { koro_cont_resume(first_cont, &second_id) }, -ESTALE);
    assert_eq!(unsafe { koro_cont_resume(first_cont, &first_id) }, 0);
    first.wait(2);
    first.wait_completed();

    assert_eq!(unsafe { koro_cont_destroy(first_cont) }, 0);
    assert_eq!(unsafe { koro_cont_destroy(second_cont) }, 0);
    destroy_runtime(runtime);
}
