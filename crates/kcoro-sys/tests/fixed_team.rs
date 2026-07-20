use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const EDEADLK: i32 = 11;
#[cfg(any(target_os = "linux", target_os = "android"))]
const EDEADLK: i32 = 35;
const EBUSY: i32 = 16;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "android"
)))]
const EDEADLK: i32 = 35;

const MEMBER_COUNT: u32 = 4;
const MEMBER_MASK: u32 = (1 << MEMBER_COUNT) - 1;
const GENERATIONS: u64 = 3;
const CHAIN_DONE: u32 = GENERATIONS as u32 + 1;

#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    reserved: u32,
}

#[repr(C)]
struct ServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    context: *mut c_void,
    reserved: u64,
    owner_init: Option<unsafe extern "C" fn(*mut c_void)>,
    owner_fini: Option<unsafe extern "C" fn(*mut c_void)>,
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
    if unsafe { kc_team_snapshot_get(team, &mut snapshot) } != 0
        || snapshot.dispatched_generation != generation
        || snapshot.completed_generation != generation
        || snapshot.completed_members != MEMBER_COUNT
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

unsafe extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
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
}

#[test]
fn completion_edges_drive_every_generation_and_terminal_teardown() {
    kcoro_sys::link_anchor();
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
        size: size_of::<TeamConfig>() as u32,
        abi_version: 1,
        member_count: MEMBER_COUNT,
        reserved: 0,
        member: Some(chain_member),
        context: (&chain as *const Chain).cast_mut().cast(),
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
    assert_eq!(unsafe { kc_team_join(team) }, 0);

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
}

#[test]
fn completion_edge_resumes_state_before_the_next_dispatch() {
    kcoro_sys::link_anchor();
    let runtime_config = RuntimeConfig {
        size: size_of::<RuntimeConfig>() as u32,
        abi_version: 1,
        worker_count: 1,
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
        size: size_of::<ServiceConfig>() as u32,
        abi_version: 1,
        callback: Some(handoff_service),
        context: (&handoff as *const Handoff).cast_mut().cast(),
        reserved: 0,
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

    /* The team edge wakes the suspended service state, the service publishes
     * generation two, and that generation's edge publishes terminal stop. */
    assert_eq!(unsafe { kc_team_join(team) }, 0);
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

    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    assert_eq!(unsafe { kc_service_notifier_destroy(notifier) }, 0);
    unsafe { kc_service_request_stop(service) };
    assert_eq!(unsafe { kc_service_join(service) }, 0);
    assert_eq!(unsafe { kc_service_destroy(service) }, 0);
    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}

#[test]
fn concurrent_publishers_cannot_overwrite_the_accepted_ticket_edge() {
    kcoro_sys::link_anchor();
    let race = PublisherRace {
        gate: Barrier::new(MEMBER_COUNT as usize + 1),
        members: AtomicU32::new(0),
        bad: AtomicU32::new(0),
    };
    let config = TeamConfig {
        size: size_of::<TeamConfig>() as u32,
        abi_version: 1,
        member_count: MEMBER_COUNT,
        reserved: 0,
        member: Some(publisher_member),
        context: (&race as *const PublisherRace).cast_mut().cast(),
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
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(race.members.load(Ordering::Acquire), MEMBER_COUNT);
    assert_eq!(race.bad.load(Ordering::Acquire), 0);
    assert_eq!(
        first.callbacks.load(Ordering::Acquire) + second.callbacks.load(Ordering::Acquire),
        1
    );
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
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
}
