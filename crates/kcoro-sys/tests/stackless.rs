use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

const ABI: u32 = 1;
const EBUSY: i32 = 16;
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const ECANCELED: i32 = 89;
#[cfg(any(target_os = "linux", target_os = "android"))]
const ECANCELED: i32 = 125;

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
            if unsafe { koro_cont_finish(raw) } != 0 {
                1usize as *mut c_void
            } else {
                std::ptr::null_mut()
            }
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
            if unsafe { koro_cont_finish(raw) } != 0 {
                1_usize as *mut c_void
            } else {
                std::ptr::null_mut()
            }
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

struct TerminalRace {
    entered: AtomicBool,
    release: Mutex<bool>,
    changed: Condvar,
    runs: AtomicU32,
    completed: AtomicBool,
}

unsafe extern "C" fn terminal_race_step(raw: *mut c_void) -> *mut c_void {
    let race = unsafe { &*koro_cont_argument(raw).cast::<TerminalRace>() };
    let run = race.runs.fetch_add(1, Ordering::AcqRel) + 1;
    if run == 1 {
        race.entered.store(true, Ordering::Release);
        race.changed.notify_all();
        let mut release = race.release.lock().unwrap();
        while !*release {
            release = race.changed.wait(release).unwrap();
        }
    }
    if unsafe { koro_cont_finish(raw) } != 0 {
        1usize as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn terminal_race_completed(context: *mut c_void, identity: *const Ticket) {
    let race = unsafe { &*context.cast::<TerminalRace>() };
    assert!(!identity.is_null());
    race.completed.store(true, Ordering::Release);
    race.changed.notify_all();
}

struct CompletionGate {
    entered: AtomicBool,
    release: Mutex<bool>,
    changed: Condvar,
    completed: AtomicBool,
}

struct ClaimGate {
    entered: Barrier,
    release: Barrier,
}

unsafe extern "C" fn claim_pause(context: *mut c_void, _worker: u32, _slot: u32) {
    let gate = unsafe { &*context.cast::<ClaimGate>() };
    gate.entered.wait();
    gate.release.wait();
}

struct RegisterGate {
    runtime: usize,
    slot: AtomicU32,
}

unsafe extern "C" fn register_pause(context: *mut c_void, runtime: *mut c_void, slot: u32) {
    let gate = unsafe { &*context.cast::<RegisterGate>() };
    assert_eq!(gate.runtime, runtime as usize);
    assert_eq!(
        unsafe { kc_runtime_hold_closed_slot_reader_for_test(runtime, slot) },
        0
    );
    gate.slot.store(slot, Ordering::Release);
}

unsafe extern "C" fn completion_gate_step(raw: *mut c_void) -> *mut c_void {
    if unsafe { koro_cont_finish(raw) } != 0 {
        1usize as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn completion_gate_callback(context: *mut c_void, identity: *const Ticket) {
    let gate = unsafe { &*context.cast::<CompletionGate>() };
    assert!(!identity.is_null());
    gate.entered.store(true, Ordering::Release);
    gate.changed.notify_all();
    let mut release = gate.release.lock().unwrap();
    while !*release {
        release = gate.changed.wait(release).unwrap();
    }
    gate.completed.store(true, Ordering::Release);
    gate.changed.notify_all();
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
    if unsafe { koro_cont_finish(raw) } != 0 {
        1usize as *mut c_void
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_join_all(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_runtime_inject_claim_pause_for_test(
        runtime: *mut c_void,
        pause: Option<unsafe extern "C" fn(*mut c_void, u32, u32)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_runtime_inject_register_pause_for_test(
        runtime: *mut c_void,
        pause: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, u32)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_runtime_hold_closed_slot_reader_for_test(runtime: *mut c_void, slot: u32) -> i32;
    fn kc_runtime_release_closed_slot_reader_for_test(runtime: *mut c_void, slot: u32) -> i32;
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
    fn koro_cont_finish(cont: *mut c_void) -> i32;
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
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut target) },
        0
    );
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
            if unsafe { koro_cont_finish(raw) } != 0 {
                1usize as *mut c_void
            } else {
                std::ptr::null_mut()
            }
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
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut cont) },
        0
    );
    let identity = unsafe { koro_cont_identity(cont) };
    assert_eq!(unsafe { koro_cont_start(cont) }, 0);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = state.release.lock().unwrap();
    while !state.entered.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "first continuation step did not start"
        );
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
        assert!(!remaining.is_zero(), "callback resume edge was lost");
        guard = state.changed.wait_timeout(guard, remaining).unwrap().0;
    }
    while !state.completed.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "lost-wake continuation did not retire"
        );
        guard = state.changed.wait_timeout(guard, remaining).unwrap().0;
    }
    drop(guard);
    assert_eq!(unsafe { koro_cont_destroy(cont) }, 0);
    destroy_runtime(runtime);
}

#[test]
fn callback_wins_against_terminal_claim_and_receives_one_successor_invocation() {
    let runtime = runtime(2);
    let race = TerminalRace {
        entered: AtomicBool::new(false),
        release: Mutex::new(false),
        changed: Condvar::new(),
        runs: AtomicU32::new(0),
        completed: AtomicBool::new(false),
    };
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(terminal_race_step),
        argument: (&race as *const TerminalRace).cast_mut().cast(),
        frame_size: 0,
        worker_mask: 0,
        completion: Some(terminal_race_completed),
        completion_context: (&race as *const TerminalRace).cast_mut().cast(),
    };
    let mut cont = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut cont) },
        0
    );
    let identity = unsafe { koro_cont_identity(cont) };
    assert_eq!(unsafe { koro_cont_start(cont) }, 0);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = race.release.lock().unwrap();
    while !race.entered.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "terminal-race step did not start");
        release = race.changed.wait_timeout(release, remaining).unwrap().0;
    }
    assert_eq!(unsafe { koro_cont_resume(cont, &identity) }, 0);
    *release = true;
    race.changed.notify_all();
    while !race.completed.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "terminal successor did not retire");
        release = race.changed.wait_timeout(release, remaining).unwrap().0;
    }
    drop(release);
    assert_eq!(race.runs.load(Ordering::Acquire), 2);
    assert_eq!(unsafe { koro_cont_resume(cont, &identity) }, -ECANCELED);
    assert_eq!(unsafe { koro_cont_destroy(cont) }, 0);
    destroy_runtime(runtime);
}

#[test]
fn done_is_published_only_after_the_completion_callback_releases_its_context() {
    let runtime = runtime(1);
    let gate = CompletionGate {
        entered: AtomicBool::new(false),
        release: Mutex::new(false),
        changed: Condvar::new(),
        completed: AtomicBool::new(false),
    };
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(completion_gate_step),
        argument: std::ptr::null_mut(),
        frame_size: 0,
        worker_mask: 0,
        completion: Some(completion_gate_callback),
        completion_context: (&gate as *const CompletionGate).cast_mut().cast(),
    };
    let mut cont = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut cont) },
        0
    );
    assert_eq!(unsafe { koro_cont_start(cont) }, 0);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut release = gate.release.lock().unwrap();
    while !gate.entered.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "completion callback did not start");
        release = gate.changed.wait_timeout(release, remaining).unwrap().0;
    }
    assert_eq!(unsafe { koro_cont_destroy(cont) }, -EBUSY);
    *release = true;
    gate.changed.notify_all();
    while !gate.completed.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "completion callback did not finish");
        release = gate.changed.wait_timeout(release, remaining).unwrap().0;
    }
    drop(release);

    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
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

#[test]
fn slot_reuse_cannot_overtake_a_worker_holding_the_prior_publication() {
    kcoro_sys::link_anchor();
    let runtime = runtime(2);
    let gate = ClaimGate {
        entered: Barrier::new(2),
        release: Barrier::new(2),
    };
    let first = Signal::new();
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(completion_gate_step),
        argument: std::ptr::null_mut(),
        frame_size: 0,
        worker_mask: 0b11,
        completion: Some(completed),
        completion_context: (&first as *const Signal).cast_mut().cast(),
    };
    let mut old = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut old) },
        0
    );
    assert_eq!(
        unsafe {
            kc_runtime_inject_claim_pause_for_test(
                runtime,
                Some(claim_pause),
                (&gate as *const ClaimGate).cast_mut().cast(),
            )
        },
        0
    );
    assert_eq!(unsafe { koro_cont_start(old) }, 0);
    gate.entered.wait();
    first.wait_completed();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { koro_cont_destroy(old) }, 0);

    let second = Signal::new();
    let next = ContConfig {
        completion_context: (&second as *const Signal).cast_mut().cast(),
        ..config
    };
    let mut fresh = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &next, &mut fresh) },
        0
    );
    assert_eq!(unsafe { koro_cont_start(fresh) }, 0);
    second.wait_completed();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);

    gate.release.wait();
    assert_eq!(unsafe { koro_cont_destroy(fresh) }, 0);
    destroy_runtime(runtime);
}

#[test]
fn registration_cannot_erase_a_late_reader_of_the_closed_generation() {
    kcoro_sys::link_anchor();
    let runtime = runtime(2);
    let first = Signal::new();
    let config = ContConfig {
        size: size_of::<ContConfig>() as u32,
        abi_version: ABI,
        step: Some(completion_gate_step),
        argument: std::ptr::null_mut(),
        frame_size: 0,
        worker_mask: 0b11,
        completion: Some(completed),
        completion_context: (&first as *const Signal).cast_mut().cast(),
    };
    let mut old = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &config, &mut old) },
        0
    );
    assert_eq!(unsafe { koro_cont_start(old) }, 0);
    first.wait_completed();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { koro_cont_destroy(old) }, 0);

    let gate = RegisterGate {
        runtime: runtime as usize,
        slot: AtomicU32::new(u32::MAX),
    };
    assert_eq!(
        unsafe {
            kc_runtime_inject_register_pause_for_test(
                runtime,
                Some(register_pause),
                (&gate as *const RegisterGate).cast_mut().cast(),
            )
        },
        0
    );
    let second = Signal::new();
    let next = ContConfig {
        completion_context: (&second as *const Signal).cast_mut().cast(),
        ..config
    };
    let mut fresh = std::ptr::null_mut();
    assert_eq!(
        unsafe { koro_cont_create_on(runtime, &next, &mut fresh) },
        0
    );
    const NONE: u32 = u32::MAX;
    let held = gate.slot.load(Ordering::Acquire);
    assert_ne!(held, NONE);
    assert_eq!(
        unsafe { kc_runtime_release_closed_slot_reader_for_test(runtime, held) },
        0
    );
    assert_eq!(unsafe { koro_cont_start(fresh) }, 0);
    second.wait_completed();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { koro_cont_destroy(fresh) }, 0);
    destroy_runtime(runtime);
}
