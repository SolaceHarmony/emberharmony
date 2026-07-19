use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

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
struct ServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    context: *mut c_void,
    reserved: u64,
}

#[repr(C)]
struct TeamConfig {
    size: u32,
    abi_version: u32,
    member_count: u32,
    reserved: u32,
    member: Option<unsafe extern "C" fn(*mut c_void, u32, u32, u64)>,
    context: *mut c_void,
}

#[repr(C)]
struct TeamSnapshot {
    size: u32,
    abi_version: u32,
    member_count: u32,
    started_members: u32,
    dispatched_generation: u64,
    completed_generation: u64,
    completed_members: u32,
    started: u32,
    stop_requested: u32,
    joined: u32,
}

struct Seen {
    team: AtomicPtr<c_void>,
    collective: AtomicPtr<c_void>,
    first: AtomicU32,
    second: AtomicU32,
    third: AtomicU32,
    calls: AtomicU32,
    bad: AtomicU32,
    latest: AtomicU64,
    finals: AtomicU32,
    notifications: AtomicU32,
    notified_generation: AtomicU64,
    callback_wait_status: AtomicI32,
}

struct Handoff {
    team: AtomicPtr<c_void>,
    service: AtomicPtr<c_void>,
    notifier: AtomicPtr<c_void>,
    first: AtomicU32,
    second: AtomicU32,
    edge_status: AtomicI32,
    dispatch_status: AtomicI32,
    gate: Mutex<()>,
    changed: Condvar,
}

unsafe extern "C" fn finalizer(context: *mut c_void) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    seen.finals.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn completed(context: *mut c_void, generation: u64) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    let mut snapshot = TeamSnapshot {
        size: size_of::<TeamSnapshot>() as u32,
        abi_version: 1,
        member_count: 0,
        started_members: 0,
        dispatched_generation: 0,
        completed_generation: 0,
        completed_members: 0,
        started: 0,
        stop_requested: 0,
        joined: 0,
    };
    let status = unsafe { kc_team_snapshot_get(seen.team.load(Ordering::Acquire), &mut snapshot) };
    if status != 0 || generation != 2 || snapshot.completed_generation != generation {
        seen.bad.fetch_add(1, Ordering::Relaxed);
    }
    seen.callback_wait_status.store(
        unsafe { kc_team_wait(seen.team.load(Ordering::Acquire), generation, 0) },
        Ordering::Release,
    );
    seen.notified_generation
        .store(generation, Ordering::Release);
    seen.notifications.fetch_add(1, Ordering::Release);
}

unsafe extern "C" fn member(context: *mut c_void, index: u32, members: u32, generation: u64) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    if members != 4 || index >= members {
        seen.bad.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mask = if generation == 1 {
        &seen.first
    } else if generation == 2 {
        &seen.second
    } else if generation == 3 {
        &seen.third
    } else {
        seen.bad.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if mask.fetch_or(1 << index, Ordering::AcqRel) & (1 << index) != 0 {
        seen.bad.fetch_add(1, Ordering::Relaxed);
    }
    seen.calls.fetch_add(1, Ordering::Relaxed);
    seen.latest.fetch_max(generation, Ordering::Release);
    let status = unsafe {
        kc_collective_arrive(
            seen.collective.load(Ordering::Acquire),
            index,
            Some(finalizer),
            context,
        )
    };
    if status < 0 {
        seen.bad.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe extern "C" fn handoff_member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    if members != 2 || index >= members {
        handoff.edge_status.store(-1, Ordering::Release);
        return;
    }
    let mask = if generation == 1 {
        &handoff.first
    } else if generation == 2 {
        &handoff.second
    } else {
        handoff.edge_status.store(-2, Ordering::Release);
        return;
    };
    mask.fetch_or(1 << index, Ordering::AcqRel);
}

unsafe extern "C" fn handoff_edge(context: *mut c_void, generation: u64) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    if generation != 1 {
        let _gate = handoff.gate.lock().unwrap();
        handoff.edge_status.store(-3, Ordering::Release);
        handoff.changed.notify_all();
        return;
    }
    let status = unsafe {
        kc_service_notifier_notify(handoff.notifier.load(Ordering::Acquire))
    };
    if status != 0 {
        let _gate = handoff.gate.lock().unwrap();
        handoff.edge_status.store(status, Ordering::Release);
        handoff.changed.notify_all();
        return;
    }

    /*
     * Hold the final team member inside the completion callback until the
     * resumed service attempts the next dispatch. This is the exact ordering
     * that used to return -EBUSY and then lose its only edge.
     */
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gate = handoff.gate.lock().unwrap();
    while handoff.dispatch_status.load(Ordering::Acquire) == i32::MIN {
        let wait = deadline.saturating_duration_since(Instant::now());
        if wait.is_zero() {
            handoff.edge_status.store(-4, Ordering::Release);
            handoff.changed.notify_all();
            return;
        }
        gate = handoff.changed.wait_timeout(gate, wait).unwrap().0;
    }
    handoff.edge_status.store(
        handoff.dispatch_status.load(Ordering::Acquire),
        Ordering::Release,
    );
    handoff.changed.notify_all();
}

unsafe extern "C" fn handoff_service(context: *mut c_void) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    let status = unsafe { kc_team_dispatch(handoff.team.load(Ordering::Acquire), 2) };
    let _gate = handoff.gate.lock().unwrap();
    handoff.dispatch_status.store(status, Ordering::Release);
    handoff.changed.notify_all();
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_run_until_idle(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_service_create(
        runtime: *mut c_void,
        config: *const ServiceConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn kc_service_start(service: *mut c_void) -> i32;
    fn kc_service_notifier_create(service: *mut c_void, out: *mut *mut c_void) -> i32;
    fn kc_service_notifier_notify(notifier: *mut c_void) -> i32;
    fn kc_service_notifier_destroy(notifier: *mut c_void) -> i32;
    fn kc_service_request_stop(service: *mut c_void);
    fn kc_service_join(service: *mut c_void) -> i32;
    fn kc_service_destroy(service: *mut c_void) -> i32;
    fn kc_team_create(config: *const TeamConfig, out: *mut *mut c_void) -> i32;
    fn kc_team_start(team: *mut c_void) -> i32;
    fn kc_team_dispatch(team: *mut c_void, generation: u64) -> i32;
    fn kc_team_dispatch_notify(
        team: *mut c_void,
        generation: u64,
        completion: Option<unsafe extern "C" fn(*mut c_void, u64)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_team_wait(team: *mut c_void, generation: u64, deadline_ns: u64) -> i32;
    fn kc_team_request_stop(team: *mut c_void);
    fn kc_team_join(team: *mut c_void) -> i32;
    fn kc_team_destroy(team: *mut c_void) -> i32;
    fn kc_team_snapshot_get(team: *mut c_void, out: *mut TeamSnapshot) -> i32;
    fn kc_collective_create(members: u32, out: *mut *mut c_void) -> i32;
    fn kc_collective_arrive(
        collective: *mut c_void,
        member: u32,
        finalizer: Option<unsafe extern "C" fn(*mut c_void)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_collective_generation(collective: *const c_void) -> u64;
    fn kc_collective_destroy(collective: *mut c_void);
}

#[test]
fn fixed_members_run_each_generation_once_and_join_under_team_ownership() {
    kcoro_sys::link_anchor();
    let seen = Seen {
        team: AtomicPtr::new(std::ptr::null_mut()),
        collective: AtomicPtr::new(std::ptr::null_mut()),
        first: AtomicU32::new(0),
        second: AtomicU32::new(0),
        third: AtomicU32::new(0),
        calls: AtomicU32::new(0),
        bad: AtomicU32::new(0),
        latest: AtomicU64::new(0),
        finals: AtomicU32::new(0),
        notifications: AtomicU32::new(0),
        notified_generation: AtomicU64::new(0),
        callback_wait_status: AtomicI32::new(i32::MIN),
    };
    let mut collective = std::ptr::null_mut();
    assert_eq!(unsafe { kc_collective_create(4, &mut collective) }, 0);
    seen.collective.store(collective, Ordering::Release);
    let config = TeamConfig {
        size: size_of::<TeamConfig>() as u32,
        abi_version: 1,
        member_count: 4,
        reserved: 0,
        member: Some(member),
        context: (&seen as *const Seen).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    seen.team.store(team, Ordering::Release);
    assert_eq!(unsafe { kc_team_start(team) }, 0);

    assert_eq!(unsafe { kc_team_dispatch(team, 1) }, 0);
    assert_eq!(unsafe { kc_team_wait(team, 1, 0) }, 0);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                2,
                Some(completed),
                (&seen as *const Seen).cast_mut().cast(),
            )
        },
        0
    );
    assert_eq!(unsafe { kc_team_wait(team, 2, 0) }, 0);
    assert_eq!(unsafe { kc_team_dispatch(team, 3) }, 0);
    assert_eq!(unsafe { kc_team_wait(team, 3, 0) }, 0);

    let mut snapshot = TeamSnapshot {
        size: size_of::<TeamSnapshot>() as u32,
        abi_version: 1,
        member_count: 0,
        started_members: 0,
        dispatched_generation: 0,
        completed_generation: 0,
        completed_members: 0,
        started: 0,
        stop_requested: 0,
        joined: 0,
    };
    assert_eq!(unsafe { kc_team_snapshot_get(team, &mut snapshot) }, 0);
    assert_eq!(snapshot.member_count, 4);
    assert_eq!(snapshot.started_members, 4);
    assert_eq!(snapshot.dispatched_generation, 3);
    assert_eq!(snapshot.completed_generation, 3);
    assert_eq!(snapshot.completed_members, 4);
    assert_eq!(seen.first.load(Ordering::Acquire), 0b1111);
    assert_eq!(seen.second.load(Ordering::Acquire), 0b1111);
    assert_eq!(seen.third.load(Ordering::Acquire), 0b1111);
    assert_eq!(seen.calls.load(Ordering::Acquire), 12);
    assert_eq!(seen.bad.load(Ordering::Acquire), 0);
    assert_eq!(seen.latest.load(Ordering::Acquire), 3);
    assert_eq!(seen.finals.load(Ordering::Acquire), 3);
    assert_eq!(seen.notifications.load(Ordering::Acquire), 1);
    assert_eq!(seen.notified_generation.load(Ordering::Acquire), 2);
    assert_eq!(seen.callback_wait_status.load(Ordering::Acquire), -EDEADLK);
    assert_eq!(unsafe { kc_collective_generation(collective) }, 3);

    unsafe { kc_team_request_stop(team) };
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    unsafe { kc_collective_destroy(collective) };
}

#[test]
fn completion_edge_resumes_service_after_next_dispatch_is_admissible() {
    kcoro_sys::link_anchor();
    let runtime_config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: 1,
        worker_count: 1,
        arena_segment_size: 0,
        ticket_capacity: 1,
        reserved: 0,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_runtime_create(&runtime_config, &mut runtime) },
        0
    );
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);

    let handoff = Handoff {
        team: AtomicPtr::new(std::ptr::null_mut()),
        service: AtomicPtr::new(std::ptr::null_mut()),
        notifier: AtomicPtr::new(std::ptr::null_mut()),
        first: AtomicU32::new(0),
        second: AtomicU32::new(0),
        edge_status: AtomicI32::new(i32::MIN),
        dispatch_status: AtomicI32::new(i32::MIN),
        gate: Mutex::new(()),
        changed: Condvar::new(),
    };
    let service_config = ServiceConfig {
        size: size_of::<ServiceConfig>() as u32,
        abi_version: 1,
        callback: Some(handoff_service),
        context: (&handoff as *const Handoff).cast_mut().cast(),
        reserved: 0,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &service_config, &mut service) },
        0
    );
    handoff.service.store(service, Ordering::Release);
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    let mut notifier = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_notifier_create(service, &mut notifier) },
        0
    );
    handoff.notifier.store(notifier, Ordering::Release);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);

    let team_config = TeamConfig {
        size: size_of::<TeamConfig>() as u32,
        abi_version: 1,
        member_count: 2,
        reserved: 0,
        member: Some(handoff_member),
        context: (&handoff as *const Handoff).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&team_config, &mut team) }, 0);
    handoff.team.store(team, Ordering::Release);
    assert_eq!(unsafe { kc_team_start(team) }, 0);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                1,
                Some(handoff_edge),
                (&handoff as *const Handoff).cast_mut().cast(),
            )
        },
        0
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut gate = handoff.gate.lock().unwrap();
    while handoff.edge_status.load(Ordering::Acquire) == i32::MIN {
        let wait = deadline.saturating_duration_since(Instant::now());
        assert!(
            !wait.is_zero(),
            "service did not receive team completion edge"
        );
        gate = handoff.changed.wait_timeout(gate, wait).unwrap().0;
    }
    drop(gate);
    assert_eq!(handoff.dispatch_status.load(Ordering::Acquire), 0);
    assert_eq!(unsafe { kc_team_wait(team, 2, 0) }, 0);
    assert_eq!(handoff.edge_status.load(Ordering::Acquire), 0);
    assert_eq!(handoff.first.load(Ordering::Acquire), 0b11);
    assert_eq!(handoff.second.load(Ordering::Acquire), 0b11);

    /* Quiesce the callback-side producer before releasing its notifier lease.
     * This is the production teardown order: team -> notifier -> service ->
     * runtime. */
    unsafe { kc_team_request_stop(team) };
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_notifier_destroy(notifier) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}
