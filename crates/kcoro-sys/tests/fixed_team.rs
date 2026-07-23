use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const EDEADLK: i32 = 11;
#[cfg(any(target_os = "linux", target_os = "android"))]
const EDEADLK: i32 = 35;
const EBUSY: i32 = 16;
const EINVAL: i32 = 22;
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const ECANCELED: i32 = 89;
#[cfg(any(target_os = "linux", target_os = "android"))]
const ECANCELED: i32 = 125;
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const EAGAIN: i32 = 35;
#[cfg(any(target_os = "linux", target_os = "android"))]
const EAGAIN: i32 = 11;
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const ESTALE: i32 = 70;
#[cfg(any(target_os = "linux", target_os = "android"))]
const ESTALE: i32 = 116;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "android"
)))]
const EDEADLK: i32 = 35;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "android"
)))]
const ESTALE: i32 = 116;

const MEMBER_COUNT: u32 = 4;
const MEMBER_MASK: u32 = (1 << MEMBER_COUNT) - 1;
const GENERATIONS: u64 = 3;
const CHAIN_DONE: u32 = GENERATIONS as u32 + 1;
const NEVER_ENTERED: u32 = 1;
const HEALTHY_GENERATIONS: u64 = 1_000_000;

#[repr(C)]
struct RuntimeConfig {
    worker_count: u32,
}

#[repr(C)]
struct ServiceConfig {
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    context: *mut c_void,
    owner_init: Option<unsafe extern "C" fn(*mut c_void)>,
    owner_fini: Option<unsafe extern "C" fn(*mut c_void)>,
}

#[repr(C)]
struct TeamConfig {
    member_count: u32,
    member: Option<unsafe extern "C" fn(*mut c_void, u32, u32, u64)>,
    context: *mut c_void,
    runtime: *mut c_void,
    retired: Option<unsafe extern "C" fn(*mut c_void, u64)>,
    retired_context: *mut c_void,
}

#[repr(C)]
struct TeamSnapshot {
    member_count: u32,
    started_members: u32,
    dispatched_generation: u64,
    completed_generation: u64,
    completed_members: u32,
    started: u32,
    stop_requested: u32,
    joined: u32,
}

#[repr(C)]
struct TeamQuorumSnapshot {
    generation: u64,
    expected_mask: u64,
    entered_mask: u64,
    returned_mask: u64,
}

struct Chain {
    team: AtomicPtr<c_void>,
    masks: [AtomicU32; GENERATIONS as usize],
    latest: [AtomicU64; MEMBER_COUNT as usize],
    callbacks: [AtomicU32; GENERATIONS as usize],
    dispatches: [AtomicI32; GENERATIONS as usize - 1],
    active: AtomicU32,
    calls: AtomicU32,
    bad: AtomicU32,
    phase: AtomicU32,
    callback_join_status: AtomicI32,
}

struct Handoff {
    team: AtomicPtr<c_void>,
    notifier: AtomicPtr<c_void>,
    masks: [AtomicU32; 2],
    latest: [AtomicU64; 2],
    callbacks: [AtomicU32; 2],
    active: AtomicU32,
    bad: AtomicU32,
    phase: AtomicU32,
    notifications: AtomicI32,
    dispatch: AtomicI32,
    service_callbacks: AtomicU32,
    callback_gate: Barrier,
}

struct PublisherRace {
    gate: Barrier,
    members: AtomicU32,
    bad: AtomicU32,
}

struct PublisherEdge {
    team: AtomicPtr<c_void>,
    callbacks: AtomicU32,
}

struct Quorum {
    team: AtomicPtr<c_void>,
    entered: Barrier,
    release: Barrier,
    completed: Barrier,
    retire: Barrier,
    calls: [AtomicU32; MEMBER_COUNT as usize],
    callbacks: AtomicU32,
    bad: AtomicU32,
}

struct Injected {
    team: AtomicPtr<c_void>,
    calls: AtomicU32,
    callbacks: AtomicU32,
    status: AtomicI32,
    entered: AtomicU64,
    returned: AtomicU64,
}

struct Healthy {
    team: AtomicPtr<c_void>,
    samples: Box<[AtomicU64]>,
    started: AtomicU64,
    callbacks: AtomicU64,
    bad: AtomicU32,
}

struct Retired {
    done: Mutex<bool>,
    changed: Condvar,
}

struct RetiredGate {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}

struct StartGate {
    entered: Barrier,
    release: Barrier,
}

unsafe extern "C" fn start_admitted(context: *mut c_void, member: u32) {
    let gate = unsafe { &*context.cast::<StartGate>() };
    assert_eq!(member, 1);
    gate.entered.wait();
    gate.release.wait();
}

impl RetiredGate {
    fn new() -> Self {
        Self {
            state: Mutex::new((false, false)),
            changed: Condvar::new(),
        }
    }

    fn wait_entered(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = self.state.lock().unwrap();
        while !state.0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "team retirement callback did not enter"
            );
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.1 = true;
        self.changed.notify_all();
    }
}

impl Retired {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut done = self.done.lock().unwrap();
        while !*done {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "team retirement callback timed out");
            done = self.changed.wait_timeout(done, remaining).unwrap().0;
        }
    }
}

unsafe extern "C" fn retired_edge(context: *mut c_void, _generation: u64) {
    let retired = unsafe { &*context.cast::<Retired>() };
    *retired.done.lock().unwrap() = true;
    retired.changed.notify_all();
}

unsafe extern "C" fn blocked_retired_edge(context: *mut c_void, _generation: u64) {
    let gate = unsafe { &*context.cast::<RetiredGate>() };
    let mut state = gate.state.lock().unwrap();
    state.0 = true;
    gate.changed.notify_all();
    while !state.1 {
        state = gate.changed.wait(state).unwrap();
    }
}

unsafe extern "C" fn noop_member(
    _context: *mut c_void,
    _index: u32,
    _members: u32,
    _generation: u64,
) {
}

unsafe extern "C" fn chain_member(context: *mut c_void, index: u32, members: u32, generation: u64) {
    let chain = unsafe { &*(context.cast::<Chain>()) };
    if members != MEMBER_COUNT || index >= members || !(1..=GENERATIONS).contains(&generation) {
        chain.bad.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let bit = 1 << index;
    if chain.active.fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    if chain.phase.load(Ordering::Acquire) != generation as u32 {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    if chain.latest[index as usize].swap(generation, Ordering::AcqRel) != generation - 1 {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    if chain.masks[generation as usize - 1].fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    chain.calls.fetch_add(1, Ordering::Relaxed);
    chain.active.fetch_and(!bit, Ordering::Release);
}

unsafe extern "C" fn chain_edge(context: *mut c_void, generation: u64) {
    let chain = unsafe { &*(context.cast::<Chain>()) };
    let team = chain.team.load(Ordering::Acquire);
    if !(1..=GENERATIONS).contains(&generation) {
        chain.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
        return;
    }

    let slot = generation as usize - 1;
    if chain.callbacks[slot].fetch_add(1, Ordering::AcqRel) != 0
        || chain.masks[slot].load(Ordering::Acquire) != MEMBER_MASK
        || chain.active.load(Ordering::Acquire) != 0
    {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    let mut snapshot = TeamSnapshot {
        member_count: 0,
        started_members: 0,
        dispatched_generation: 0,
        completed_generation: 0,
        completed_members: 0,
        started: 0,
        stop_requested: 0,
        joined: 0,
    };
    if unsafe { kc_team_snapshot_get(team, &mut snapshot) } != 0
        || snapshot.dispatched_generation != generation
        || snapshot.completed_generation != generation
        || snapshot.completed_members != MEMBER_COUNT
    {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    let mut quorum = quorum_snapshot();
    if unsafe { kc_team_quorum_snapshot_get(team, generation, &mut quorum) } != 0
        || quorum.generation != generation
        || quorum.expected_mask != MEMBER_MASK as u64
        || quorum.entered_mask != MEMBER_MASK as u64
        || quorum.returned_mask != MEMBER_MASK as u64
    {
        chain.bad.fetch_add(1, Ordering::Relaxed);
    }
    if generation == 2 {
        chain
            .callback_join_status
            .store(unsafe { kc_team_join(team) }, Ordering::Release);
    }
    if chain
        .phase
        .compare_exchange(
            generation as u32,
            generation as u32 + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        chain.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
        return;
    }
    if generation == GENERATIONS {
        unsafe { kc_team_request_stop(team) };
        return;
    }

    let status =
        unsafe { kc_team_dispatch_notify(team, generation + 1, Some(chain_edge), context) };
    chain.dispatches[slot].store(status, Ordering::Release);
    if status != 0 {
        chain.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
    }
}

unsafe extern "C" fn handoff_member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    if members != 2 || index >= members || !(1..=2).contains(&generation) {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let bit = 1 << index;
    if handoff.active.fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    let phase = if generation == 1 { 1 } else { 3 };
    if handoff.phase.load(Ordering::Acquire) != phase {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    if handoff.latest[index as usize].swap(generation, Ordering::AcqRel) != generation - 1 {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    if handoff.masks[generation as usize - 1].fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    handoff.active.fetch_and(!bit, Ordering::Release);
}

unsafe extern "C" fn handoff_edge(context: *mut c_void, generation: u64) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    let team = handoff.team.load(Ordering::Acquire);
    if !(1..=2).contains(&generation) {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
        return;
    }

    let slot = generation as usize - 1;
    if handoff.callbacks[slot].fetch_add(1, Ordering::AcqRel) != 0
        || handoff.masks[slot].load(Ordering::Acquire) != 0b11
        || handoff.active.load(Ordering::Acquire) != 0
    {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    let mut snapshot = TeamSnapshot {
        member_count: 0,
        started_members: 0,
        dispatched_generation: 0,
        completed_generation: 0,
        completed_members: 0,
        started: 0,
        stop_requested: 0,
        joined: 0,
    };
    if unsafe { kc_team_snapshot_get(team, &mut snapshot) } != 0
        || snapshot.dispatched_generation != generation
        || snapshot.completed_generation != generation
        || snapshot.completed_members != 2
    {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }

    if generation == 1 {
        if handoff
            .phase
            .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            handoff.bad.fetch_add(1, Ordering::Relaxed);
            unsafe { kc_team_request_stop(team) };
            return;
        }
        let status =
            unsafe { kc_service_notifier_notify(handoff.notifier.load(Ordering::Acquire)) };
        handoff.notifications.store(status, Ordering::Release);
        if status != 0 {
            handoff.bad.fetch_add(1, Ordering::Relaxed);
            unsafe { kc_team_request_stop(team) };
            return;
        }
        /* Test-only rendezvous: keep this callback on the stack until the
         * resumed service has attempted generation two. Product execution has
         * no corresponding waiter. */
        handoff.callback_gate.wait();
        return;
    }

    if handoff
        .phase
        .compare_exchange(3, 4, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
    }
    unsafe { kc_team_request_stop(team) };
}

unsafe extern "C" fn handoff_service(context: *mut c_void) {
    let handoff = unsafe { &*(context.cast::<Handoff>()) };
    handoff.service_callbacks.fetch_add(1, Ordering::AcqRel);
    let team = handoff.team.load(Ordering::Acquire);
    if handoff
        .phase
        .compare_exchange(2, 3, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
        return;
    }
    let status = unsafe { kc_team_dispatch_notify(team, 2, Some(handoff_edge), context) };
    handoff.dispatch.store(status, Ordering::Release);
    handoff.callback_gate.wait();
    if status != 0 {
        handoff.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
    }
}

unsafe extern "C" fn publisher_member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let race = unsafe { &*(context.cast::<PublisherRace>()) };
    if members != MEMBER_COUNT || index >= members || generation != 1 {
        race.bad.fetch_add(1, Ordering::Relaxed);
    }
    race.members.fetch_add(1, Ordering::AcqRel);
    race.gate.wait();
}

unsafe extern "C" fn publisher_edge(context: *mut c_void, generation: u64) {
    let edge = unsafe { &*(context.cast::<PublisherEdge>()) };
    if generation != 1 {
        edge.callbacks.fetch_add(2, Ordering::Relaxed);
    } else {
        edge.callbacks.fetch_add(1, Ordering::Release);
    }
    unsafe { kc_team_request_stop(edge.team.load(Ordering::Acquire)) };
}

unsafe extern "C" fn quorum_member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let quorum = unsafe { &*(context.cast::<Quorum>()) };
    if members != MEMBER_COUNT || index >= members || generation != 7 {
        quorum.bad.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if quorum.calls[index as usize].fetch_add(1, Ordering::AcqRel) != 0 {
        quorum.bad.fetch_add(1, Ordering::Relaxed);
    }
    quorum.entered.wait();
    quorum.release.wait();
}

unsafe extern "C" fn quorum_edge(context: *mut c_void, generation: u64) {
    let quorum = unsafe { &*(context.cast::<Quorum>()) };
    if generation != 7 || quorum.callbacks.fetch_add(1, Ordering::AcqRel) != 0 {
        quorum.bad.fetch_add(1, Ordering::Relaxed);
    }
    quorum.completed.wait();
    quorum.retire.wait();
    unsafe { kc_team_request_stop(quorum.team.load(Ordering::Acquire)) };
}

unsafe extern "C" fn injected_member(
    context: *mut c_void,
    _index: u32,
    _members: u32,
    _generation: u64,
) {
    let injected = unsafe { &*(context.cast::<Injected>()) };
    injected.calls.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn injected_completion(context: *mut c_void, _generation: u64) {
    let injected = unsafe { &*(context.cast::<Injected>()) };
    injected.callbacks.fetch_add(1, Ordering::Release);
}

unsafe extern "C" fn injected_ready(context: *mut c_void, generation: u64) {
    let injected = unsafe { &*(context.cast::<Injected>()) };
    let team = injected.team.load(Ordering::Acquire);
    let mut snapshot = quorum_snapshot();
    let status = unsafe { kc_team_quorum_snapshot_get(team, generation, &mut snapshot) };
    injected.status.store(status, Ordering::Release);
    if status == 0 {
        injected
            .entered
            .store(snapshot.entered_mask, Ordering::Release);
        injected
            .returned
            .store(snapshot.returned_mask, Ordering::Release);
    }
    unsafe { kc_team_request_stop(team) };
}

unsafe extern "C" fn healthy_member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let healthy = unsafe { &*(context.cast::<Healthy>()) };
    if members != MEMBER_COUNT
        || index >= members
        || !(1..=HEALTHY_GENERATIONS).contains(&generation)
    {
        healthy.bad.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe extern "C" fn healthy_edge(context: *mut c_void, generation: u64) {
    let healthy = unsafe { &*(context.cast::<Healthy>()) };
    let now = unsafe { kc_port_monotonic_ns() };
    let started = healthy.started.load(Ordering::Acquire);
    if started == 0 || now < started || generation == 0 || generation > HEALTHY_GENERATIONS {
        healthy.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(healthy.team.load(Ordering::Acquire)) };
        return;
    }
    healthy.samples[generation as usize - 1].store(now - started, Ordering::Release);
    healthy.callbacks.fetch_add(1, Ordering::Relaxed);
    let team = healthy.team.load(Ordering::Acquire);
    if generation == HEALTHY_GENERATIONS {
        unsafe { kc_team_request_stop(team) };
        return;
    }
    healthy
        .started
        .store(unsafe { kc_port_monotonic_ns() }, Ordering::Release);
    if unsafe { kc_team_dispatch_notify(team, generation + 1, Some(healthy_edge), context) } != 0 {
        healthy.bad.fetch_add(1, Ordering::Relaxed);
        unsafe { kc_team_request_stop(team) };
    }
}

unsafe extern "C" {
    fn kc_port_monotonic_ns() -> u64;
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_join_all(runtime: *mut c_void) -> i32;
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
    fn kc_team_dispatch_notify(
        team: *mut c_void,
        generation: u64,
        completion: Option<unsafe extern "C" fn(*mut c_void, u64)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_team_request_stop(team: *mut c_void);
    fn kc_team_join(team: *mut c_void) -> i32;
    fn kc_team_destroy(team: *mut c_void) -> i32;
    fn kc_team_snapshot_get(team: *mut c_void, out: *mut TeamSnapshot) -> i32;
    fn kc_team_quorum_snapshot_get(
        team: *mut c_void,
        generation: u64,
        out: *mut TeamQuorumSnapshot,
    ) -> i32;
    fn kc_team_inject_member_exit_for_test(
        team: *mut c_void,
        generation: u64,
        member: u32,
        point: u32,
        ready: Option<unsafe extern "C" fn(*mut c_void, u64)>,
        context: *mut c_void,
    ) -> i32;
    fn kc_team_inject_start_failure_for_test(team: *mut c_void, after_started: u32) -> i32;
    fn kc_team_inject_start_pause_for_test(
        team: *mut c_void,
        member: u32,
        pause: Option<unsafe extern "C" fn(*mut c_void, u32)>,
        context: *mut c_void,
    ) -> i32;
}

fn quorum_snapshot() -> TeamQuorumSnapshot {
    TeamQuorumSnapshot {
        generation: u64::MAX,
        expected_mask: u64::MAX,
        entered_mask: u64::MAX,
        returned_mask: u64::MAX,
    }
}

fn runtime(workers: u32) -> *mut c_void {
    let config = RuntimeConfig {
        worker_count: workers,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);
    runtime
}

fn retire_runtime(runtime: *mut c_void) {
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn completion_edges_drive_every_generation_and_terminal_teardown() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let chain = Chain {
        team: AtomicPtr::new(std::ptr::null_mut()),
        masks: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
        latest: [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ],
        callbacks: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
        dispatches: [AtomicI32::new(i32::MIN), AtomicI32::new(i32::MIN)],
        active: AtomicU32::new(0),
        calls: AtomicU32::new(0),
        bad: AtomicU32::new(0),
        phase: AtomicU32::new(1),
        callback_join_status: AtomicI32::new(i32::MIN),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(chain_member),
        context: (&chain as *const Chain).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    chain.team.store(team, Ordering::Release);
    assert_eq!(unsafe { kc_team_start(team) }, 0);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                1,
                Some(chain_edge),
                (&chain as *const Chain).cast_mut().cast(),
            )
        },
        0
    );

    /* The terminal callback publishes stop. This join only tears down the
     * stopped team; it is not a per-generation observation primitive. */
    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);

    let mut snapshot = TeamSnapshot {
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
    assert_eq!(snapshot.member_count, MEMBER_COUNT);
    assert_eq!(snapshot.started_members, MEMBER_COUNT);
    assert_eq!(snapshot.dispatched_generation, GENERATIONS);
    assert_eq!(snapshot.completed_generation, GENERATIONS);
    assert_eq!(snapshot.completed_members, MEMBER_COUNT);
    assert_eq!(snapshot.stop_requested, 1);
    assert_eq!(snapshot.joined, 1);
    assert_eq!(chain.phase.load(Ordering::Acquire), CHAIN_DONE);
    assert_eq!(
        chain.calls.load(Ordering::Acquire),
        MEMBER_COUNT * GENERATIONS as u32
    );
    assert_eq!(chain.active.load(Ordering::Acquire), 0);
    assert_eq!(chain.bad.load(Ordering::Acquire), 0);
    assert_eq!(chain.callback_join_status.load(Ordering::Acquire), -EDEADLK);
    for mask in &chain.masks {
        assert_eq!(mask.load(Ordering::Acquire), MEMBER_MASK);
    }
    for latest in &chain.latest {
        assert_eq!(latest.load(Ordering::Acquire), GENERATIONS);
    }
    for callbacks in &chain.callbacks {
        assert_eq!(callbacks.load(Ordering::Acquire), 1);
    }
    for dispatch in &chain.dispatches {
        assert_eq!(dispatch.load(Ordering::Acquire), 0);
    }
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn completion_edge_resumes_state_before_the_next_dispatch() {
    kcoro_sys::link_anchor();
    let runtime_config = RuntimeConfig {
        worker_count: 3,
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_runtime_create(&runtime_config, &mut runtime) },
        0
    );
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);

    let handoff = Handoff {
        team: AtomicPtr::new(std::ptr::null_mut()),
        notifier: AtomicPtr::new(std::ptr::null_mut()),
        masks: [AtomicU32::new(0), AtomicU32::new(0)],
        latest: [AtomicU64::new(0), AtomicU64::new(0)],
        callbacks: [AtomicU32::new(0), AtomicU32::new(0)],
        active: AtomicU32::new(0),
        bad: AtomicU32::new(0),
        phase: AtomicU32::new(1),
        notifications: AtomicI32::new(i32::MIN),
        dispatch: AtomicI32::new(i32::MIN),
        service_callbacks: AtomicU32::new(0),
        callback_gate: Barrier::new(2),
    };
    let service_config = ServiceConfig {
        callback: Some(handoff_service),
        context: (&handoff as *const Handoff).cast_mut().cast(),
        owner_init: None,
        owner_fini: None,
    };
    let mut service = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_create(runtime, &service_config, &mut service) },
        0
    );
    assert_eq!(unsafe { kc_service_start(service) }, 0);
    let mut notifier = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_service_notifier_create(service, &mut notifier) },
        0
    );
    handoff.notifier.store(notifier, Ordering::Release);
    let retired = Retired::new();

    let team_config = TeamConfig {
        member_count: 2,
        member: Some(handoff_member),
        context: (&handoff as *const Handoff).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
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

    /* The team edge wakes the suspended service state, the service publishes
     * generation two, and that generation's edge publishes terminal stop. */
    retired.wait();
    assert_eq!(handoff.phase.load(Ordering::Acquire), 4);
    assert_eq!(handoff.bad.load(Ordering::Acquire), 0);
    assert_eq!(handoff.active.load(Ordering::Acquire), 0);
    assert_eq!(handoff.notifications.load(Ordering::Acquire), 0);
    assert_eq!(handoff.dispatch.load(Ordering::Acquire), 0);
    assert_eq!(handoff.service_callbacks.load(Ordering::Acquire), 1);
    for mask in &handoff.masks {
        assert_eq!(mask.load(Ordering::Acquire), 0b11);
    }
    for latest in &handoff.latest {
        assert_eq!(latest.load(Ordering::Acquire), 2);
    }
    for callbacks in &handoff.callbacks {
        assert_eq!(callbacks.load(Ordering::Acquire), 1);
    }

    /* Retire the still-dormant handoff service, then use the runtime's
     * administrative terminal acknowledgement to prove both callbacks have
     * returned before either callback context is released. */
    assert_eq!(unsafe { kc_service_notifier_destroy(notifier) }, 0);
    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn concurrent_publishers_cannot_overwrite_the_accepted_ticket_edge() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let race = PublisherRace {
        gate: Barrier::new(MEMBER_COUNT as usize + 1),
        members: AtomicU32::new(0),
        bad: AtomicU32::new(0),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(publisher_member),
        context: (&race as *const PublisherRace).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    assert_eq!(unsafe { kc_team_start(team) }, 0);

    let first = PublisherEdge {
        team: AtomicPtr::new(team),
        callbacks: AtomicU32::new(0),
    };
    let second = PublisherEdge {
        team: AtomicPtr::new(team),
        callbacks: AtomicU32::new(0),
    };
    let start = Arc::new(Barrier::new(3));
    let spawn = |edge: &PublisherEdge| {
        let start = Arc::clone(&start);
        let team = team as usize;
        let edge = edge as *const PublisherEdge as usize;
        std::thread::spawn(move || {
            start.wait();
            unsafe {
                kc_team_dispatch_notify(
                    team as *mut c_void,
                    1,
                    Some(publisher_edge),
                    edge as *mut c_void,
                )
            }
        })
    };
    let one = spawn(&first);
    let two = spawn(&second);
    start.wait();
    let mut statuses = [one.join().unwrap(), two.join().unwrap()];
    statuses.sort_unstable();
    assert_eq!(statuses, [-EBUSY, 0]);

    /* The admitted generation is held inside its real member callbacks until
     * both publisher results are known, so the losing result cannot be an
     * accidental post-completion rejection. */
    race.gate.wait();
    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(race.members.load(Ordering::Acquire), MEMBER_COUNT);
    assert_eq!(race.bad.load(Ordering::Acquire), 0);
    assert_eq!(
        first.callbacks.load(Ordering::Acquire) + second.callbacks.load(Ordering::Acquire),
        1
    );
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn quorum_snapshot_tracks_one_exact_generation_without_duplicate_execution() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let quorum = Quorum {
        team: AtomicPtr::new(std::ptr::null_mut()),
        entered: Barrier::new(MEMBER_COUNT as usize + 1),
        release: Barrier::new(MEMBER_COUNT as usize + 1),
        completed: Barrier::new(2),
        retire: Barrier::new(2),
        calls: [
            AtomicU32::new(0),
            AtomicU32::new(0),
            AtomicU32::new(0),
            AtomicU32::new(0),
        ],
        callbacks: AtomicU32::new(0),
        bad: AtomicU32::new(0),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(quorum_member),
        context: (&quorum as *const Quorum).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    quorum.team.store(team, Ordering::Release);
    assert_eq!(unsafe { kc_team_start(team) }, 0);

    let mut before = quorum_snapshot();
    assert_eq!(
        unsafe { kc_team_quorum_snapshot_get(team, 7, &mut before) },
        -ESTALE
    );
    assert_eq!(before.generation, u64::MAX);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                7,
                Some(quorum_edge),
                (&quorum as *const Quorum).cast_mut().cast(),
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                7,
                Some(quorum_edge),
                (&quorum as *const Quorum).cast_mut().cast(),
            )
        },
        -EBUSY
    );

    quorum.entered.wait();
    let mut active = quorum_snapshot();
    assert_eq!(
        unsafe { kc_team_quorum_snapshot_get(team, 7, &mut active) },
        0
    );
    assert_eq!(active.generation, 7);
    assert_eq!(active.expected_mask, MEMBER_MASK as u64);
    assert_eq!(active.entered_mask, MEMBER_MASK as u64);
    assert_eq!(active.returned_mask, 0);
    let mut successor = quorum_snapshot();
    assert_eq!(
        unsafe { kc_team_quorum_snapshot_get(team, 8, &mut successor) },
        -ESTALE
    );
    assert_eq!(successor.generation, u64::MAX);

    quorum.release.wait();
    quorum.completed.wait();
    let mut complete = quorum_snapshot();
    assert_eq!(
        unsafe { kc_team_quorum_snapshot_get(team, 7, &mut complete) },
        0
    );
    assert_eq!(complete.entered_mask, MEMBER_MASK as u64);
    assert_eq!(complete.returned_mask, MEMBER_MASK as u64);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                7,
                Some(quorum_edge),
                (&quorum as *const Quorum).cast_mut().cast(),
            )
        },
        -EINVAL
    );
    quorum.retire.wait();

    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(quorum.bad.load(Ordering::Acquire), 0);
    assert_eq!(quorum.callbacks.load(Ordering::Acquire), 1);
    for calls in &quorum.calls {
        assert_eq!(calls.load(Ordering::Acquire), 1);
    }
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn never_entered_injection_preserves_exact_partial_quorum_without_completion() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let injected = Injected {
        team: AtomicPtr::new(std::ptr::null_mut()),
        calls: AtomicU32::new(0),
        callbacks: AtomicU32::new(0),
        status: AtomicI32::new(i32::MIN),
        entered: AtomicU64::new(0),
        returned: AtomicU64::new(0),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(injected_member),
        context: (&injected as *const Injected).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    injected.team.store(team, Ordering::Release);
    assert_eq!(
        unsafe {
            kc_team_inject_member_exit_for_test(
                team,
                1,
                1,
                NEVER_ENTERED,
                Some(injected_ready),
                (&injected as *const Injected).cast_mut().cast(),
            )
        },
        0
    );
    assert_eq!(unsafe { kc_team_start(team) }, 0);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                1,
                Some(injected_completion),
                (&injected as *const Injected).cast_mut().cast(),
            )
        },
        0
    );
    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);

    assert_eq!(injected.status.load(Ordering::Acquire), 0);
    assert_eq!(injected.entered.load(Ordering::Acquire), 0b1101);
    assert_eq!(injected.returned.load(Ordering::Acquire), 0b1101);
    assert_eq!(injected.calls.load(Ordering::Acquire), MEMBER_COUNT - 1);
    assert_eq!(injected.callbacks.load(Ordering::Acquire), 0);
    let mut snapshot = TeamSnapshot {
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
    assert_eq!(snapshot.dispatched_generation, 1);
    assert_eq!(snapshot.completed_generation, 0);
    assert_eq!(snapshot.completed_members, MEMBER_COUNT - 1);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
#[ignore = "one-million-generation target-hardware calibration"]
fn one_million_healthy_generations_report_the_fixed_team_budget_floor() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let healthy = Healthy {
        team: AtomicPtr::new(std::ptr::null_mut()),
        samples: (0..HEALTHY_GENERATIONS)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        started: AtomicU64::new(0),
        callbacks: AtomicU64::new(0),
        bad: AtomicU32::new(0),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(healthy_member),
        context: (&healthy as *const Healthy).cast_mut().cast(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    healthy.team.store(team, Ordering::Release);
    assert_eq!(unsafe { kc_team_start(team) }, 0);
    healthy
        .started
        .store(unsafe { kc_port_monotonic_ns() }, Ordering::Release);
    assert_eq!(
        unsafe {
            kc_team_dispatch_notify(
                team,
                1,
                Some(healthy_edge),
                (&healthy as *const Healthy).cast_mut().cast(),
            )
        },
        0
    );
    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(healthy.bad.load(Ordering::Acquire), 0);
    assert_eq!(
        healthy.callbacks.load(Ordering::Acquire),
        HEALTHY_GENERATIONS
    );

    let mut samples = healthy
        .samples
        .iter()
        .map(|sample| sample.load(Ordering::Acquire))
        .collect::<Vec<_>>();
    assert!(samples.iter().all(|sample| *sample != 0));
    samples.sort_unstable();
    let p99 = samples[(samples.len() * 99).div_ceil(100) - 1];
    let max = *samples.last().expect("healthy generation samples");
    let soft = 10_000_000_u64
        .max(p99.saturating_mul(8))
        .max(max.saturating_mul(4));
    let hard = 1_000_000_000_u64.max(soft.saturating_mul(4));
    eprintln!(
        "fixed_team_budget_floor generations={} members={} p99_ns={} max_ns={} soft_ns={} hard_ns={}",
        HEALTHY_GENERATIONS, MEMBER_COUNT, p99, max, soft, hard
    );
    assert!(soft >= 10_000_000);
    assert!(hard >= 1_000_000_000);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn fixed_team_rejects_members_that_cannot_fit_the_quorum_mask() {
    kcoro_sys::link_anchor();
    let runtime = runtime(1);
    let race = PublisherRace {
        gate: Barrier::new(1),
        members: AtomicU32::new(0),
        bad: AtomicU32::new(0),
    };
    let config = TeamConfig {
        member_count: 65,
        member: Some(publisher_member),
        context: (&race as *const PublisherRace).cast_mut().cast(),
        runtime,
        retired: None,
        retired_context: std::ptr::null_mut(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, -EINVAL);
    assert!(team.is_null());
    retire_runtime(runtime);
}

#[test]
fn partial_start_retires_only_started_frames_and_remains_destroyable() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(noop_member),
        context: std::ptr::null_mut(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    assert_eq!(unsafe { kc_team_inject_start_failure_for_test(team, 2) }, 0);
    assert_eq!(unsafe { kc_team_start(team) }, -EAGAIN);
    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert!(unsafe { kc_team_start(team) } < 0);
    assert!(unsafe { kc_team_dispatch_notify(team, 1, None, std::ptr::null_mut()) } < 0);

    let mut snapshot = TeamSnapshot {
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
    assert_eq!(snapshot.started_members, 2);
    assert_eq!(snapshot.stop_requested, 1);
    assert_eq!(snapshot.joined, 1);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn stopping_an_unstarted_team_has_no_retirement_callback_to_outlive_it() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(noop_member),
        context: std::ptr::null_mut(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    unsafe { kc_team_request_stop(team) };
    assert!(!*retired.done.lock().unwrap());
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn concurrent_stop_cannot_overtake_start_admission_and_lose_retirement() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let retired = Retired::new();
    let gate = StartGate {
        entered: Barrier::new(2),
        release: Barrier::new(2),
    };
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(noop_member),
        context: std::ptr::null_mut(),
        runtime,
        retired: Some(retired_edge),
        retired_context: (&retired as *const Retired).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    assert_eq!(
        unsafe {
            kc_team_inject_start_pause_for_test(
                team,
                1,
                Some(start_admitted),
                (&gate as *const StartGate).cast_mut().cast(),
            )
        },
        0
    );

    let raw = team as usize;
    let starter = std::thread::spawn(move || unsafe { kc_team_start(raw as *mut c_void) });
    gate.entered.wait();
    unsafe { kc_team_request_stop(team) };
    gate.release.wait();
    assert_eq!(starter.join().unwrap(), -ECANCELED);

    retired.wait();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    let mut snapshot = TeamSnapshot {
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
    assert_eq!(snapshot.started_members, 2);
    assert_eq!(snapshot.stop_requested, 1);
    assert_eq!(snapshot.joined, 1);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn team_join_cannot_overtake_the_last_retirement_callback() {
    kcoro_sys::link_anchor();
    let runtime = runtime(MEMBER_COUNT);
    let gate = RetiredGate::new();
    let config = TeamConfig {
        member_count: MEMBER_COUNT,
        member: Some(noop_member),
        context: std::ptr::null_mut(),
        runtime,
        retired: Some(blocked_retired_edge),
        retired_context: (&gate as *const RetiredGate).cast_mut().cast(),
    };
    let mut team = std::ptr::null_mut();
    assert_eq!(unsafe { kc_team_create(&config, &mut team) }, 0);
    assert_eq!(unsafe { kc_team_start(team) }, 0);
    unsafe { kc_team_request_stop(team) };
    gate.wait_entered();
    assert_eq!(unsafe { kc_team_join(team) }, -EBUSY);
    gate.release();
    assert_eq!(unsafe { kc_runtime_join_all(runtime) }, 0);
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    retire_runtime(runtime);
}

#[test]
fn fixed_team_execution_has_no_blocking_generation_observer() {
    const HEADER: &str = include_str!("../vendor/kcoro_arena/include/kc_team.h");
    const CORE: &str = include_str!("../vendor/kcoro_arena/core/src/kc_team.c");
    const TEST: &str = include_str!("fixed_team.rs");
    let legacy = ["kc_team_", "wait"].concat();

    for (path, source) in [
        ("kc_team.h", HEADER),
        ("kc_team.c", CORE),
        ("fixed_team.rs", TEST),
    ] {
        assert!(!source.contains(&legacy), "{path} exposes {legacy}");
    }
    assert!(CORE.contains("uint64_t seen_generation"));
    assert!(!CORE.contains("uint64_t seen ="));
    assert!(!CORE.contains("kc_port_thread_create"));
    assert!(!CORE.contains("pthread_create"));
    assert!(CORE.contains("koro_cont_create_on"));
}
