use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

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
    collective: AtomicPtr<c_void>,
    first: AtomicU32,
    second: AtomicU32,
    calls: AtomicU32,
    bad: AtomicU32,
    latest: AtomicU64,
    finals: AtomicU32,
}

unsafe extern "C" fn finalizer(context: *mut c_void) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    seen.finals.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn member(
    context: *mut c_void,
    index: u32,
    members: u32,
    generation: u64,
) {
    let seen = unsafe { &*(context.cast::<Seen>()) };
    if members != 4 || index >= members {
        seen.bad.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mask = if generation == 1 {
        &seen.first
    } else if generation == 2 {
        &seen.second
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

unsafe extern "C" {
    fn kc_team_create(config: *const TeamConfig, out: *mut *mut c_void) -> i32;
    fn kc_team_start(team: *mut c_void) -> i32;
    fn kc_team_dispatch(team: *mut c_void, generation: u64) -> i32;
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
        collective: AtomicPtr::new(std::ptr::null_mut()),
        first: AtomicU32::new(0),
        second: AtomicU32::new(0),
        calls: AtomicU32::new(0),
        bad: AtomicU32::new(0),
        latest: AtomicU64::new(0),
        finals: AtomicU32::new(0),
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
    assert_eq!(unsafe { kc_team_start(team) }, 0);

    assert_eq!(unsafe { kc_team_dispatch(team, 1) }, 0);
    assert_eq!(unsafe { kc_team_wait(team, 1, 0) }, 0);
    assert_eq!(unsafe { kc_team_dispatch(team, 2) }, 0);
    assert_eq!(unsafe { kc_team_wait(team, 2, 0) }, 0);

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
    assert_eq!(snapshot.dispatched_generation, 2);
    assert_eq!(snapshot.completed_generation, 2);
    assert_eq!(snapshot.completed_members, 4);
    assert_eq!(seen.first.load(Ordering::Acquire), 0b1111);
    assert_eq!(seen.second.load(Ordering::Acquire), 0b1111);
    assert_eq!(seen.calls.load(Ordering::Acquire), 8);
    assert_eq!(seen.bad.load(Ordering::Acquire), 0);
    assert_eq!(seen.latest.load(Ordering::Acquire), 2);
    assert_eq!(seen.finals.load(Ordering::Acquire), 2);
    assert_eq!(unsafe { kc_collective_generation(collective) }, 2);

    unsafe { kc_team_request_stop(team) };
    assert_eq!(unsafe { kc_team_join(team) }, 0);
    assert_eq!(unsafe { kc_team_destroy(team) }, 0);
    unsafe { kc_collective_destroy(collective) };
}
