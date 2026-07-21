use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
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
const DORMANT: u32 = 3;
const DONE: u32 = 4;

#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    reserved: u32,
}

#[repr(C)]
struct RuntimeSnapshot {
    size: u32,
    abi_version: u32,
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

type Callback = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct ServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<Callback>,
    context: *mut c_void,
    reserved: u64,
    owner_init: Option<Callback>,
    owner_fini: Option<Callback>,
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

struct StartGate {
    target: u32,
    stage: AtomicU32,
    released: Mutex<bool>,
    changed: Condvar,
}

impl StartGate {
    fn new(target: u32) -> Self {
        Self {
            target,
            stage: AtomicU32::new(0),
            released: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut released = self.released.lock().unwrap();
        while self.stage.load(Ordering::Acquire) != self.target {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "service start hook did not arrive");
            released = self.changed.wait_timeout(released, remaining).unwrap().0;
        }
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
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

unsafe extern "C" fn start_hook(context: *mut c_void, stage: u32) {
    let gate = unsafe { &*context.cast::<StartGate>() };
    if stage != gate.target {
        return;
    }
    gate.stage.store(stage, Ordering::Release);
    gate.changed.notify_all();
    let mut released = gate.released.lock().unwrap();
    while !*released {
        released = gate.changed.wait(released).unwrap();
    }
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_join_all(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_runtime_snapshot_get(runtime: *mut c_void, out: *mut RuntimeSnapshot) -> i32;
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
    fn kc_service_inject_start_hook_for_test(
        service: *mut c_void,
        hook: Option<unsafe extern "C" fn(*mut c_void, u32)>,
        context: *mut c_void,
    ) -> i32;
}

#[test]
fn bounded_callback_can_reschedule_its_own_ready_predicate_without_an_external_edge() {
    kcoro_sys::link_anchor();
    const QUOTAS: u64 = 17;
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
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
        owner_init: None,
        owner_fini: None,
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
    wait_for_handled(service, QUOTAS);

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
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    let probe = JoinProbe {
        runtime: AtomicPtr::new(runtime),
        service: AtomicPtr::new(std::ptr::null_mut()),
        service_join: AtomicI32::new(i32::MIN),
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
        owner_init: None,
        owner_fini: None,
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
    assert_eq!(probe.join_all.load(Ordering::Acquire), -EDEADLK);
    assert_eq!(probe.runtime_join.load(Ordering::Acquire), -EDEADLK);

    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn serial_continuation_acknowledges_each_realtime_edge() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
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
        owner_init: None,
        owner_fini: None,
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

    for generation in 1..=512_u64 {
        assert_eq!(unsafe { kc_service_notifier_notify(notifier) }, 0);
        wait_for_handled(service, generation);
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
    assert_eq!(
        unsafe { kc_runtime_snapshot_get(runtime, &mut snapshot) },
        0
    );
    snapshot
}

#[test]
fn service_stop_and_join_do_not_stop_the_explicit_runtime() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 1,
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
        owner_init: None,
        owner_fini: None,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_start(service) }, 0);
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

fn wait_for_handled(service: *mut c_void, handled: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while service_snapshot(service).handled_notifications < handled {
        assert!(
            Instant::now() < deadline,
            "service did not acknowledge notification {handled}"
        );
        std::thread::yield_now();
    }
}

#[test]
fn retained_service_coalesces_edges_without_lost_wakes_or_concurrent_callbacks() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
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
        owner_init: None,
        owner_fini: None,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &config, &mut service) },
        0
    );
    assert_eq!(runtime_snapshot(runtime).active, 0);

    assert!(unsafe { kc_service_notify(service) } < 0);
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, -EBUSY);
    for _ in 0..16 {
        assert_eq!(unsafe { kc_service_notify(service) }, 0);
    }
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    wait_for_calls(&seen, 1);
    wait_for_handled(service, 16);

    let first = service_snapshot(service);
    assert_eq!(first.notifications, 16);
    assert_eq!(first.handled_notifications, 16);
    assert_eq!(first.callbacks, 1);
    assert_eq!(first.run_state, DORMANT);
    let idle = runtime_snapshot(runtime);
    assert_eq!(
        (idle.active, idle.queued, idle.running, idle.dormant),
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
        owner_init: None,
        owner_fini: None,
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
fn concurrent_start_and_runtime_stop_always_reaches_a_destroyable_terminal_state() {
    kcoro_sys::link_anchor();
    for _ in 0..256 {
        let config = RuntimeConfig {
            size: size_of::<RuntimeConfig>() as u32,
            abi_version: ABI,
            worker_count: 2,
            reserved: 0,
        };
        let mut runtime = std::ptr::null_mut();
        assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
        assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
        let count = Count {
            callbacks: AtomicU64::new(0),
        };
        let config = ServiceConfig {
            size: size_of::<ServiceConfig>() as u32,
            abi_version: ABI,
            callback: Some(count_callback),
            context: (&count as *const Count).cast_mut().cast(),
            reserved: 0,
            owner_init: None,
            owner_fini: None,
        };
        let mut service = std::ptr::null_mut();
        assert_eq!(
            unsafe { kc_service_create(runtime, &config, &mut service) },
            0
        );

        let gate = Arc::new(Barrier::new(3));
        let start_gate = Arc::clone(&gate);
        let start_service = service as usize;
        let starter = std::thread::spawn(move || {
            start_gate.wait();
            unsafe { kc_service_start(start_service as *mut c_void) }
        });
        let stop_gate = Arc::clone(&gate);
        let stop_runtime = runtime as usize;
        let stopper = std::thread::spawn(move || {
            stop_gate.wait();
            unsafe { kc_runtime_request_stop(stop_runtime as *mut c_void) };
        });
        gate.wait();
        let started = starter.join().unwrap();
        stopper.join().unwrap();
        assert!(started == 0 || started < 0);

        unsafe { kc_service_request_stop(service) };
        assert_eq!(unsafe { kc_service_join(service) }, 0);
        assert_eq!(unsafe { kc_service_destroy(service) }, 0);
        assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
        assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
    }
}

#[test]
fn stop_and_join_cannot_overtake_either_start_publication_boundary() {
    kcoro_sys::link_anchor();
    for target in [1, 2] {
        let config = RuntimeConfig {
            size: size_of::<RuntimeConfig>() as u32,
            abi_version: ABI,
            worker_count: 2,
            reserved: 0,
        };
        let mut runtime = std::ptr::null_mut();
        assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
        assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
        let count = Count {
            callbacks: AtomicU64::new(0),
        };
        let config = ServiceConfig {
            size: size_of::<ServiceConfig>() as u32,
            abi_version: ABI,
            callback: Some(count_callback),
            context: (&count as *const Count).cast_mut().cast(),
            reserved: 0,
            owner_init: None,
            owner_fini: None,
        };
        let mut service = std::ptr::null_mut();
        assert_eq!(
            unsafe { kc_service_create(runtime, &config, &mut service) },
            0
        );
        let gate = StartGate::new(target);
        assert_eq!(
            unsafe {
                kc_service_inject_start_hook_for_test(
                    service,
                    Some(start_hook),
                    (&gate as *const StartGate).cast_mut().cast(),
                )
            },
            0
        );
        let raw = service as usize;
        let starter = std::thread::spawn(move || unsafe { kc_service_start(raw as *mut c_void) });
        gate.wait();
        unsafe { kc_service_request_stop(service) };
        assert_eq!(unsafe { kc_service_join(service) }, -EBUSY);
        gate.release();
        assert!(starter.join().unwrap() < 0);
        assert_eq!(unsafe { kc_service_join(service) }, 0);
        assert_eq!(unsafe { kc_service_destroy(service) }, 0);
        unsafe { kc_runtime_request_stop(runtime) };
        assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
        assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
    }
}

#[test]
fn realtime_notifier_drains_every_accepted_edge_before_stop_retires() {
    kcoro_sys::link_anchor();
    let config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        worker_count: 2,
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
        owner_init: None,
        owner_fini: None,
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
    assert_eq!(
        snapshot.callbacks,
        1 + u64::from(accepted != 0),
        "accepted burst edges must coalesce into at most one migrated successor"
    );
    assert_eq!(seen.gate.lock().unwrap().maximum, 1);
    assert_eq!(runtime_snapshot(runtime).queued, 0);

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
    let begin = service.find("static void realtime_leave").unwrap();
    let end = service[begin..]
        .find("static int realtime_enter")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    for forbidden in ["KC_MUTEX", "for (;;)", "while (", "kc_port_wait"] {
        assert!(
            !body.contains(forbidden),
            "final realtime publication edge contains unbounded work: {forbidden}"
        );
    }
    let release = body.find("atomic_fetch_sub_explicit").unwrap();
    let final_publisher = body.find("prior == 1").unwrap();
    let closed = body.find("realtime_closed").unwrap();
    let successor = body.find("kc_runtime_publish_service_internal").unwrap();
    assert!(release < final_publisher && final_publisher < closed && closed < successor);

    let begin = service.find("static int realtime_enter").unwrap();
    let end = service[begin..]
        .find("static int notify_realtime")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    for forbidden in [
        "KC_MUTEX",
        "compare_exchange",
        "for (;;)",
        "while (",
        "kc_port_wait",
    ] {
        assert!(
            !body.contains(forbidden),
            "realtime admission contains unbounded work: {forbidden}"
        );
    }
    let first = body.find("&service->realtime_closed").unwrap();
    let lease = body.find("&service->realtime_publishers").unwrap();
    let second = body[first + 1..]
        .find("&service->realtime_closed")
        .map(|offset| first + 1 + offset)
        .unwrap();
    assert!(first < lease && lease < second);
    assert!(
        body.matches("memory_order_seq_cst").count() >= 3,
        "close and publisher observations must share one total order"
    );

    let begin = service
        .find("static int notify_realtime(kc_service_t *service)\n{")
        .unwrap();
    let end = service[begin..]
        .find("int kc_service_notifier_create")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("calloc"));
    assert!(!body.contains("compare_exchange"));
    assert!(!body.contains("for (;;)") && !body.contains("while ("));
    let entered = body.find("realtime_enter").unwrap();
    let notified = body.find("&service->notifications").unwrap();
    let released = body.find("realtime_leave(service, 1)").unwrap();
    let ready = body.find("kc_runtime_publish_service_internal").unwrap();
    assert!(entered < notified && notified < ready && ready < released);

    let begin = service.find("int kc_service_ready_again").unwrap();
    let end = service[begin..]
        .find("int kc_service_complete_current")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("kc_runtime_ring_workers_internal"));
    assert!(!body.contains("kc_port_wait"));
    assert!(!body.contains("compare_exchange"));
    assert!(!body.contains("for (;;)") && !body.contains("while ("));
    let entered = body.find("realtime_enter").unwrap();
    let notified = body.find("&service->notifications").unwrap();
    let released = body.find("realtime_leave(service, 1)").unwrap();
    let ready = body.find("kc_runtime_publish_service_internal").unwrap();
    assert!(entered < notified && notified < ready && ready < released);

    let begin = service.find("int kc_service_complete_current").unwrap();
    let end = service[begin..]
        .find("void kc_service_request_stop")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("kc_runtime_ring_workers_internal"));
    let current = body.find("kc_runtime_is_current_cont_internal").unwrap();
    let closed = body.find("realtime_closed").unwrap();
    let retiring = body.find("KC_SERVICE_RETIRING").unwrap();
    assert!(current < closed && closed < retiring);
    assert!(body.contains("memory_order_seq_cst"));

    let begin = service.find("void kc_service_request_stop").unwrap();
    let end = service[begin..]
        .find("int kc_service_join")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(body.contains("stop_service(service)"));
    let begin = service.find("static void stop_service").unwrap();
    let end = service[begin..]
        .find("int kc_service_create")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &service[begin..end];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("while (") && !body.contains("for (;;)"));
    assert!(body.contains("compare_exchange_strong"));
    assert!(body.find("realtime_closed").unwrap() < body.find("KC_SERVICE_RETIRING").unwrap());

    let runtime = include_str!("../vendor/kcoro_arena/core/src/kc_runtime.c");
    assert!(!runtime.contains("work_cv"));
    let resume = runtime
        .find("int kc_runtime_resume_continuation_internal")
        .unwrap();
    let publish = runtime[resume..]
        .find("void kc_runtime_publish_service_internal")
        .map(|offset| resume + offset)
        .unwrap();
    let body = &runtime[resume..publish];
    for forbidden in ["for (;;)", "while (", "compare_exchange"] {
        assert!(
            !body.contains(forbidden),
            "realtime callback resume contains retry/arbitration work: {forbidden}"
        );
    }
    assert!(body.contains("atomic_fetch_or_explicit"));
    assert!(body.contains("KORO_WAKE_BIT"));
    let worker = runtime.find("static void *worker_main").unwrap();
    let start = runtime[worker..]
        .find("int kc_runtime_start")
        .map(|offset| worker + offset)
        .unwrap();
    let body = &runtime[worker..start];
    assert!(!body.contains("KC_MUTEX"));
    assert!(!body.contains("runtime->head"));
    assert!(!body.contains("runtime->tail"));
    let observe = body.find("kc_doorbell_observe").unwrap();
    let drain = body.find("claim_ready(worker)").unwrap();
    let park = body.find("kc_doorbell_park").unwrap();
    assert!(observe < drain && drain < park);

    let execute = runtime.find("static void execute_continuation").unwrap();
    let body = &runtime[execute..worker];
    let done = body.find("KORO_DONE").unwrap();
    let signal = body.find("kc_runtime_signal_lifecycle_internal").unwrap();
    assert!(done < signal);

    let snapshot = service.find("int kc_service_snapshot_get").unwrap();
    let destroy = service[snapshot..]
        .find("int kc_service_destroy")
        .map(|offset| snapshot + offset)
        .unwrap();
    assert!(service[snapshot..destroy].contains("koro_run_public"));
}

#[test]
fn legacy_scheduler_and_timed_progress_surfaces_are_absent() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for source in [
        include_str!("../src/lib.rs"),
        include_str!("../vendor/kcoro_arena/include/kc_runtime.h"),
        include_str!("../vendor/kcoro_arena/core/src/kc_runtime.c"),
    ] {
        assert!(
            !source.contains("kc_runtime_run_until_idle"),
            "operation-level runtime waiter survived"
        );
    }
    for path in [
        "vendor/kcoro_arena/core/src/kc_actor.c",
        "vendor/kcoro_arena/core/src/kc_admin.c",
        "vendor/kcoro_arena/core/src/kc_cancel.c",
        "vendor/kcoro_arena/core/src/kc_chan_stackless.c",
        "vendor/kcoro_arena/core/src/kc_checkpoint.c",
        "vendor/kcoro_arena/core/src/kc_collective.c",
        "vendor/kcoro_arena/core/src/kc_durable.c",
        "vendor/kcoro_arena/core/src/kc_op.c",
        "vendor/kcoro_arena/core/src/kc_scope.c",
        "vendor/kcoro_arena/core/src/kc_shared.c",
        "vendor/kcoro_arena/core/src/kc_timer.c",
        "vendor/kcoro_arena/core/src/kc_transport.c",
        "vendor/kcoro_arena/core/src/kc_wal.c",
        "vendor/kcoro_arena/core/src/kc_workflow.c",
        "vendor/kcoro_arena/core/src/koro_sched_stackless.c",
        "vendor/kcoro_arena/include/kc_actor.h",
        "vendor/kcoro_arena/include/kc_admin.h",
        "vendor/kcoro_arena/include/kc_cancel.h",
        "vendor/kcoro_arena/include/kc_channel.h",
        "vendor/kcoro_arena/include/kc_collective.h",
        "vendor/kcoro_arena/include/kc_durable.h",
        "vendor/kcoro_arena/include/kc_op.h",
        "vendor/kcoro_arena/include/kc_scope.h",
        "vendor/kcoro_arena/include/kc_shared.h",
        "vendor/kcoro_arena/include/kc_store.h",
        "vendor/kcoro_arena/include/kc_timer.h",
        "vendor/kcoro_arena/include/kc_transport.h",
        "vendor/kcoro_arena/include/kc_wal.h",
        "vendor/kcoro_arena/include/kc_workflow.h",
        "vendor/kcoro_arena/include/kcoro_desc.h",
        "vendor/kcoro_arena/include/koro_sched_stackless.h",
        "vendor/kcoro_arena/include/kc_ticket.h",
        "vendor/kcoro_arena/include/kc_descriptor.h",
        "vendor/kcoro_arena/include/kc_payload.h",
        "vendor/kcoro_arena/core/src/kc_ticket.c",
        "vendor/kcoro_arena/core/src/kc_ticket_internal.h",
        "vendor/kcoro_arena/core/src/kc_desc.c",
        "vendor/kcoro_arena/core/src/kc_descriptor_internal.h",
    ] {
        assert!(!root.join(path).exists(), "legacy surface survived: {path}");
    }
    assert!(
        !root.join("../kcoro/Cargo.toml").exists(),
        "mutex/condvar Rust scheduler crate survived"
    );
    let umbrella = include_str!("../vendor/kcoro_arena/include/kcoro_arena.h");
    assert!(umbrella.contains("kcoro_stackless.h"));
    assert!(!umbrella.contains("kc_doorbell.h"));
    assert!(!umbrella.contains("kc_port.h"));
    assert!(!umbrella.contains("kc_collective.h"));
    assert!(!include_str!("../build.rs").contains("kc_collective.c"));

    let files = [
        include_str!("../vendor/kcoro_arena/core/src/kc_runtime.c"),
        include_str!("../vendor/kcoro_arena/core/src/kc_runtime_internal.h"),
        include_str!("../vendor/kcoro_arena/core/src/kcoro_stackless.c"),
        include_str!("../vendor/kcoro_arena/core/src/koro_internal.h"),
        include_str!("../vendor/kcoro_arena/include/kcoro_arena.h"),
        include_str!("../vendor/kcoro_arena/include/kc_doorbell.h"),
        include_str!("../vendor/kcoro_arena/core/src/kc_doorbell.c"),
        include_str!("../vendor/kcoro_arena/include/kc_port.h"),
        include_str!("../vendor/kcoro_arena/include/kcoro_port.h"),
        include_str!("../vendor/kcoro_arena/port/posix.c"),
        include_str!("../build.rs"),
    ];
    for source in files {
        for forbidden in [
            "KC_OP_TIMER",
            "KC_OP_TIMED_OUT",
            "KC_CAUSE_TIMEOUT",
            "kc_runtime_timer",
            "kc_timer_",
            "koro_sleep",
            "KORO_SLEEP",
            "timer_main",
            "live_timers",
            "KC_COND_TIMEDWAIT_NS",
            "kc_port_cond_timedwait",
            "kc_port_thread_yield",
            "kc_doorbell_wait",
            "deadline_ns",
            "deadline_mode",
            "KC_TICKET_CAUSE_TIMED_OUT",
            "KC_TICKET_DEADLINE_",
            "KORO_WAITING",
            "KORO_WAIT_UNTIL",
            "kc_runtime_default",
            "kc_runtime_spawn",
            "kc_runtime_legacy_break",
            "kc_runtime_register_op",
            "kc_runtime_register_channel",
            "kc_runtime_register_scope",
            "koro_sched",
            "KORO_SEND",
            "KORO_RECV",
            "KORO_SELECT",
        ] {
            assert!(
                !source.contains(forbidden),
                "non-causal progress surface survived: {forbidden}"
            );
        }
    }
}
