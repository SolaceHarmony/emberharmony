use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

const ABI: u32 = 1;
const EBUSY: i32 = 16;
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const EDEADLK: i32 = 11;
#[cfg(any(target_os = "linux", target_os = "android"))]
const EDEADLK: i32 = 35;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "android"
)))]
const EDEADLK: i32 = 35;
const WAITING: u32 = 3;
const DONE: u32 = 4;

#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    arena_segment_size: usize,
    ticket_capacity: u32,
    reserved: u32,
}

#[repr(C)]
struct RuntimeSnapshot {
    size: u32,
    abi_version: u32,
    epoch: u64,
    next_sequence: u64,
    active: usize,
    queued: usize,
    running: usize,
    waiting: usize,
    live_operations: usize,
    live_timers: usize,
    live_channels: usize,
    live_scopes: usize,
    live_tickets: usize,
    completion_queued: usize,
    completion_running: usize,
    live_descriptors: usize,
    live_regions: usize,
    live_segments: usize,
    reserved_bytes: usize,
    wake_requests: u64,
    resumes: u64,
    workers: u32,
    accepting: u32,
    started: u32,
    stop_requested: u32,
    ticket_capacity: u32,
}

#[repr(C)]
struct MemorySnapshot {
    size: u32,
    abi_version: u32,
    live_descriptors: usize,
    live_regions: usize,
    live_segments: usize,
    logical_bytes: usize,
    reserved_bytes: usize,
    cumulative_bytes: u64,
    reclaimed_bytes: u64,
}

#[repr(C)]
struct AdminSnapshot {
    size: u32,
    abi_version: u32,
    capabilities: u64,
    runtime: RuntimeSnapshot,
    memory: MemorySnapshot,
    terminal_causes: [u64; 7],
    dropped_telemetry: u64,
}

type Callback = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct ServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<Callback>,
    context: *mut c_void,
    reserved: u64,
}

#[repr(C)]
#[derive(Default)]
struct ServiceSnapshot {
    size: u32,
    abi_version: u32,
    notifications: u64,
    handled_notifications: u64,
    callbacks: u64,
    run_state: u32,
    started: u32,
    stop_requested: u32,
    joined: u32,
}

#[derive(Default)]
struct Gate {
    calls: usize,
    inside: usize,
    maximum: usize,
    block_call: usize,
    released: bool,
}

struct Seen {
    gate: Mutex<Gate>,
    changed: Condvar,
}

struct JoinProbe {
    runtime: AtomicPtr<c_void>,
    service: AtomicPtr<c_void>,
    service_join: AtomicI32,
    idle: AtomicI32,
    join_all: AtomicI32,
    runtime_join: AtomicI32,
    done: Mutex<bool>,
    changed: Condvar,
}

struct Count {
    callbacks: AtomicU64,
}

struct Quota {
    service: AtomicPtr<c_void>,
    remaining: AtomicU64,
    callbacks: AtomicU64,
    status: AtomicI32,
}

unsafe extern "C" fn service_callback(context: *mut c_void) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    let mut gate = seen.gate.lock().unwrap();
    gate.calls += 1;
    gate.inside += 1;
    gate.maximum = gate.maximum.max(gate.inside);
    let call = gate.calls;
    seen.changed.notify_all();
    while call == gate.block_call && !gate.released {
        gate = seen.changed.wait(gate).unwrap();
    }
    gate.inside -= 1;
    seen.changed.notify_all();
}

unsafe extern "C" fn self_join_callback(context: *mut c_void) {
    let probe = unsafe { &*(context.cast::<JoinProbe>()) };
    let service = probe.service.load(Ordering::Acquire);
    let runtime = probe.runtime.load(Ordering::Acquire);
    unsafe { kc_service_request_stop(service) };
    probe
        .service_join
        .store(unsafe { kc_service_join(service) }, Ordering::Release);
    probe.idle.store(
        unsafe { kc_runtime_run_until_idle(runtime) },
        Ordering::Release,
    );
    probe
        .join_all
        .store(unsafe { kc_runtime_join_all(runtime) }, Ordering::Release);
    probe
        .runtime_join
        .store(unsafe { kc_runtime_join(runtime) }, Ordering::Release);
    *probe.done.lock().unwrap() = true;
    probe.changed.notify_all();
}

unsafe extern "C" fn count_callback(context: *mut c_void) {
    let count = unsafe { &*(context.cast::<Count>()) };
    count.callbacks.fetch_add(1, Ordering::Release);
}

unsafe extern "C" fn quota_callback(context: *mut c_void) {
    let quota = unsafe { &*(context.cast::<Quota>()) };
    let remaining = quota.remaining.fetch_sub(1, Ordering::AcqRel);
    quota.callbacks.fetch_add(1, Ordering::Release);
    if remaining == 0 {
        quota.status.store(-1, Ordering::Release);
        return;
    }
    if remaining == 1 {
        return;
    }
    let service = quota.service.load(Ordering::Acquire);
    quota.status.store(
        unsafe { kc_service_ready_again(service) },
        Ordering::Release,
    );
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_run_until_idle(runtime: *mut c_void) -> i32;
    fn kc_runtime_join_all(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_runtime_snapshot_get(runtime: *mut c_void, out: *mut RuntimeSnapshot) -> i32;
    fn kc_admin_snapshot_get(runtime: *mut c_void, out: *mut AdminSnapshot) -> i32;
    fn kc_service_create(
        runtime: *mut c_void,
        config: *const ServiceConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn kc_service_start(service: *mut c_void) -> i32;
    fn kc_service_notify(service: *mut c_void) -> i32;
    fn kc_service_notifier_create(service: *mut c_void, out: *mut *mut c_void) -> i32;
    fn kc_service_notifier_notify(notifier: *mut c_void) -> i32;
    fn kc_service_notifier_destroy(notifier: *mut c_void) -> i32;
    fn kc_service_ready_again(service: *mut c_void) -> i32;
    fn kc_service_request_stop(service: *mut c_void);
    fn kc_service_join(service: *mut c_void) -> i32;
    fn kc_service_snapshot_get(service: *mut c_void, out: *mut ServiceSnapshot) -> i32;
    fn kc_service_destroy(service: *mut c_void) -> i32;
}

#[test]
fn bounded_callback_can_reschedule_its_own_ready_predicate_without_an_external_edge() {
    kcoro_sys::link_anchor();
    const QUOTAS: u64 = 17;
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let quota = Quota {
        service: AtomicPtr::new(std::ptr::null_mut()),
        remaining: AtomicU64::new(QUOTAS),
        callbacks: AtomicU64::new(0),
        status: AtomicI32::new(0),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(quota_callback),
        context: (&quota as *const Quota).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    quota.service.store(service, Ordering::Release);
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    assert_eq!(unsafe { kc_service_notify(service) }, 0);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);

    let mut snapshot = ServiceSnapshot {
        size: size_of::<ServiceSnapshot>() as u32,
        abi_version: ABI,
        ..ServiceSnapshot::default()
    };
    assert_eq!(
        unsafe { kc_service_snapshot_get(service, &mut snapshot) },
        0
    );
    assert_eq!(quota.status.load(Ordering::Acquire), 0);
    assert_eq!(quota.remaining.load(Ordering::Acquire), 0);
    assert_eq!(quota.callbacks.load(Ordering::Acquire), QUOTAS);
    assert_eq!(snapshot.notifications, QUOTAS);
    assert_eq!(snapshot.handled_notifications, QUOTAS);
    assert_eq!(snapshot.callbacks, QUOTAS);
    assert_eq!(
        unsafe { kc_service_ready_again(service) },
        -1,
        "only the service's active continuation may request a local reschedule"
    );

    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn callback_side_service_and_runtime_joins_fail_instead_of_deadlocking() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let probe = JoinProbe {
        runtime: AtomicPtr::new(runtime),
        service: AtomicPtr::new(std::ptr::null_mut()),
        service_join: AtomicI32::new(i32::MIN),
        idle: AtomicI32::new(i32::MIN),
        join_all: AtomicI32::new(i32::MIN),
        runtime_join: AtomicI32::new(i32::MIN),
        done: Mutex::new(false),
        changed: Condvar::new(),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(self_join_callback),
        context: (&probe as *const JoinProbe).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    probe.service.store(service, Ordering::Release);
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    assert_eq!(unsafe { kc_service_notify(service) }, 0);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut done = probe.done.lock().unwrap();
    while !*done {
        let wait = deadline.saturating_duration_since(Instant::now());
        assert!(!wait.is_zero(), "self-join callback did not finish");
        done = probe.changed.wait_timeout(done, wait).unwrap().0;
    }
    drop(done);
    assert_eq!(probe.service_join.load(Ordering::Acquire), -EDEADLK);
    assert_eq!(probe.idle.load(Ordering::Acquire), -EDEADLK);
    assert_eq!(probe.join_all.load(Ordering::Acquire), -EDEADLK);
    assert_eq!(probe.runtime_join.load(Ordering::Acquire), -EDEADLK);

    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn run_until_idle_drains_every_realtime_edge_accepted_before_return() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let count = Count {
        callbacks: AtomicU64::new(0),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(count_callback),
        context: (&count as *const Count).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    let mut notifier = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_notifier_create(service, &mut notifier) },
        0
    );
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);

    for generation in 1..=512_u64 {
        assert_eq!(unsafe { kc_service_notifier_notify(notifier) }, 0);
        assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);
        let snapshot = service_snapshot(service);
        assert_eq!(snapshot.notifications, generation);
        assert_eq!(snapshot.handled_notifications, generation);
        assert_eq!(count.callbacks.load(Ordering::Acquire), generation);
    }

    assert_eq!(unsafe { kc_service_notifier_destroy(notifier) }, 0);
    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

fn wait_for_calls(seen: &Seen, calls: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gate = seen.gate.lock().unwrap();
    while gate.calls < calls {
        let wait = deadline.saturating_duration_since(Instant::now());
        assert!(!wait.is_zero(), "service callback timed out");
        gate = seen.changed.wait_timeout(gate, wait).unwrap().0;
    }
}

fn runtime_snapshot(runtime: *mut c_void) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot {
        size: size_of::<RuntimeSnapshot>() as u32,
        abi_version: ABI,
        epoch: 0,
        next_sequence: 0,
        active: 0,
        queued: 0,
        running: 0,
        waiting: 0,
        live_operations: 0,
        live_timers: 0,
        live_channels: 0,
        live_scopes: 0,
        live_tickets: 0,
        completion_queued: 0,
        completion_running: 0,
        live_descriptors: 0,
        live_regions: 0,
        live_segments: 0,
        reserved_bytes: 0,
        wake_requests: 0,
        resumes: 0,
        workers: 0,
        accepting: 0,
        started: 0,
        stop_requested: 0,
        ticket_capacity: 0,
    };
    assert_eq!(
        unsafe { kc_runtime_snapshot_get(runtime, &mut snapshot) },
        0
    );
    snapshot
}

fn admin_snapshot(runtime: *mut c_void) -> AdminSnapshot {
    let runtime_snapshot = runtime_snapshot(runtime);
    let mut snapshot = AdminSnapshot {
        size: size_of::<AdminSnapshot>() as u32,
        abi_version: ABI,
        capabilities: 0,
        runtime: runtime_snapshot,
        memory: MemorySnapshot {
            size: size_of::<MemorySnapshot>() as u32,
            abi_version: ABI,
            live_descriptors: 0,
            live_regions: 0,
            live_segments: 0,
            logical_bytes: 0,
            reserved_bytes: 0,
            cumulative_bytes: 0,
            reclaimed_bytes: 0,
        },
        terminal_causes: [0; 7],
        dropped_telemetry: 0,
    };
    assert_eq!(unsafe { kc_admin_snapshot_get(runtime, &mut snapshot) }, 0);
    snapshot
}

#[test]
fn service_stop_and_join_do_not_stop_the_explicit_runtime() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    let seen = Seen {
        gate: Mutex::new(Gate::default()),
        changed: Condvar::new(),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(service_callback),
        context: (&seen as *const Seen).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);
    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    let snapshot = runtime_snapshot(runtime);
    assert_eq!(snapshot.stop_requested, 0);
    assert_eq!(snapshot.accepting, 1);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

fn service_snapshot(service: *mut c_void) -> ServiceSnapshot {
    let mut snapshot = ServiceSnapshot {
        size: size_of::<ServiceSnapshot>() as u32,
        abi_version: ABI,
        ..ServiceSnapshot::default()
    };
    assert_eq!(
        unsafe { kc_service_snapshot_get(service, &mut snapshot) },
        0
    );
    snapshot
}

#[test]
fn retained_service_coalesces_edges_without_lost_wakes_or_concurrent_callbacks() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
        arena_segment_size: 0,
        ticket_capacity: 4,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);

    let seen = Seen {
        gate: Mutex::new(Gate::default()),
        changed: Condvar::new(),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(service_callback),
        context: (&seen as *const Seen).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(admin_snapshot(runtime).runtime.active, 0);

    assert!(unsafe { kc_service_notify(service) } < 0);
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, -EBUSY);
    for _ in 0..16 {
        assert_eq!(unsafe { kc_service_notify(service) }, 0);
    }
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    wait_for_calls(&seen, 1);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);

    let first = service_snapshot(service);
    assert_eq!(first.notifications, 16);
    assert_eq!(first.handled_notifications, 16);
    assert_eq!(first.callbacks, 1);
    assert_eq!(first.run_state, WAITING);
    let idle = runtime_snapshot(runtime);
    assert_eq!(
        (idle.active, idle.queued, idle.running, idle.waiting),
        (1, 0, 0, 1)
    );
    std::thread::sleep(Duration::from_millis(30));
    let after = runtime_snapshot(runtime);
    assert_eq!(after.wake_requests, idle.wake_requests);
    assert_eq!(after.resumes, idle.resumes);
    assert_eq!(service_snapshot(service).callbacks, 1);

    {
        let mut gate = seen.gate.lock().unwrap();
        gate.block_call = 2;
        gate.released = false;
    }
    assert_eq!(unsafe { kc_service_notify(service) }, 0);
    wait_for_calls(&seen, 2);
    for _ in 0..32 {
        assert_eq!(unsafe { kc_service_notify(service) }, 0);
    }
    assert_eq!(unsafe { kc_service_join(service) }, -EBUSY);
    unsafe { kc_runtime_request_stop(runtime) };
    assert!(unsafe { kc_service_notify(service) } < 0);
    {
        let mut gate = seen.gate.lock().unwrap();
        gate.released = true;
        seen.changed.notify_all();
    }
    wait_for_calls(&seen, 3);
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    let final_snapshot = service_snapshot(service);
    assert_eq!(final_snapshot.notifications, 49);
    assert_eq!(final_snapshot.handled_notifications, 49);
    assert_eq!(final_snapshot.callbacks, 3);
    let gate = seen.gate.lock().unwrap();
    assert_eq!(gate.maximum, 1);
    drop(gate);

    let stopped = service_snapshot(service);
    assert_eq!(stopped.run_state, DONE);
    assert_eq!((stopped.stop_requested, stopped.joined), (1, 1));
    assert_eq!(runtime_snapshot(runtime).active, 0);
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, -EBUSY);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn joining_an_unstarted_service_is_terminal_and_cannot_enable_uaf() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let seen = Seen {
        gate: Mutex::new(Gate::default()),
        changed: Condvar::new(),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(service_callback),
        context: (&seen as *const Seen).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert!(unsafe { kc_service_start(service) } < 0);
    assert!(unsafe { kc_service_notify(service) } < 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn realtime_notifier_drains_every_accepted_edge_before_stop_retires() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
        arena_segment_size: 0,
        ticket_capacity: 4,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let seen = Seen {
        gate: Mutex::new(Gate {
            block_call: 1,
            ..Gate::default()
        }),
        changed: Condvar::new(),
    };
    let config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: ABI,
        callback: Some(service_callback),
        context: (&seen as *const Seen).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    let mut notifier = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_notifier_create(service, &mut notifier) },
        0
    );
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);

    assert_eq!(unsafe { kc_service_notifier_notify(notifier) }, 0);
    wait_for_calls(&seen, 1);

    let start = Arc::new(Barrier::new(5));
    let raw = notifier as usize;
    let producers = (0..4)
        .map(|_| {
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                (0..4096)
                    .filter(|_| unsafe { kc_service_notifier_notify(raw as *mut c_void) } == 0)
                    .count()
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    unsafe { kc_service_request_stop(service) };
    let accepted = producers
        .into_iter()
        .map(|producer| producer.join().unwrap())
        .sum::<usize>();
    assert!(unsafe { kc_service_notifier_notify(notifier) } < 0);

    {
        let mut gate = seen.gate.lock().unwrap();
        gate.released = true;
        seen.changed.notify_all();
    }
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    let snapshot = service_snapshot(service);
    assert_eq!(snapshot.notifications, (accepted + 1) as u64);
    assert_eq!(snapshot.handled_notifications, snapshot.notifications);
    assert_eq!(seen.gate.lock().unwrap().maximum, 1);

    /* The setup-time edge owns the callback lifetime. Even a joined service
     * cannot be freed until the device producer has quiesced and releases it. */
    assert_eq!(unsafe { kc_service_destroy(service) }, -EBUSY);
    assert_eq!(unsafe { kc_service_notifier_destroy(notifier) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn realtime_notify_callgraph_and_publication_order_are_source_gated() {
    let service = include_str!("../vendor/kcoro_arena/core/src/kc_service.c");
    let begin = service.find("static int notify_realtime").unwrap();
    let end = service[begin..]
        .find("int kc_service_notifier_create")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("calloc"));
    let notified = body.find("&service->notifications").unwrap();
    let released = body.find("&service->realtime_gate, 1").unwrap();
    let ring = body.find("kc_runtime_ring_work_internal").unwrap();
    assert!(notified < released && released < ring);

    let begin = service.find("int kc_service_ready_again").unwrap();
    let end = service[begin..]
        .find("void kc_service_request_stop")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("kc_runtime_ring_work_internal"));
    assert!(!body.contains("kc_port_wait"));
    let notified = body.find("&service->notifications").unwrap();
    let ready = body.find("->wake_pending").unwrap();
    let released = body.find("&service->realtime_gate, 1").unwrap();
    assert!(notified < ready && ready < released);

    let runtime = include_str!("../vendor/kcoro_arena/core/src/kc_runtime.c");
    assert!(!runtime.contains("work_cv"));
    let worker = runtime.find("static void *worker_main").unwrap();
    let start = runtime[worker..]
        .find("int kc_runtime_start")
        .map(|offset| worker + offset)
        .unwrap();
    let body = &runtime[worker..start];
    let observe = body.find("kc_doorbell_observe").unwrap();
    let drain = body
        .find("kc_service_runtime_drain_realtime_locked")
        .unwrap();
    let park = body.find("kc_doorbell_wait").unwrap();
    assert!(observe < drain && drain < park);
}
