use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Barrier;
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

const ABI: u32 = 1;
const TICKET_PASS: u32 = 4;
const TICKET_WORKFLOW: u32 = 7;
const TICKET_CONTROL: u32 = 8;
const TICKET_DEADLINE: u32 = 9;

const CHILD_FUNCTIONAL: u32 = 1;
const CHILD_TELEMETRY: u32 = 2;
const CAUSE_COMPLETE: u32 = 1;
const CAUSE_CANCELLED: u32 = 2;
const CAUSE_FAULT: u32 = 3;
const CAUSE_DEADLINE: u32 = 4;
const SCOPE_TERMINAL: u32 = 8;

const EVENT_EXPIRED: u32 = 1;
const EVENT_STALE: u32 = 2;
const RETIRE_RETIRED: i32 = 0;
const RETIRE_EXPIRY_WON: i32 = 1;
const SOURCE_STOPPING: u32 = 2;
const SOURCE_STOPPED: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

impl Ticket {
    fn new(sequence: u64, generation: u32, kind: u32) -> Self {
        Self {
            runtime_epoch: 17,
            sequence,
            generation,
            kind,
        }
    }
}

type Notify = unsafe extern "C" fn(*mut c_void);
type GuardHook = unsafe extern "C" fn(*mut c_void, u32);
type ScopeReady = unsafe extern "C" fn(*mut c_void, u64, u32);

#[repr(C)]
struct ScopeConfig {
    size: u32,
    abi_version: u32,
    child_capacity: u32,
    reserved: u32,
    ready: Option<ScopeReady>,
    context: *mut c_void,
}

#[repr(C)]
struct ChildConfig {
    size: u32,
    abi_version: u32,
    child_class: u32,
    reserved: u32,
    cancel: Option<ScopeCancel>,
    context: *mut c_void,
}

#[repr(C)]
struct CycleConfig {
    size: u32,
    abi_version: u32,
    child_count: u32,
    reserved: u32,
    generation: u64,
    parent: Ticket,
    child_tickets: *const Ticket,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Lease {
    size: u32,
    abi_version: u32,
    slot: u32,
    child_class: u32,
    scope_generation: u64,
    child_generation: u32,
    reserved: u32,
    parent: Ticket,
    child: Ticket,
}

impl Default for Lease {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI,
            slot: 0,
            child_class: 0,
            scope_generation: 0,
            child_generation: 0,
            reserved: 0,
            parent: Ticket::new(1, 1, TICKET_WORKFLOW),
            child: Ticket::new(1, 1, TICKET_PASS),
        }
    }
}

#[repr(C)]
struct ScopeSnapshot {
    size: u32,
    abi_version: u32,
    capacity: u32,
    children: u32,
    terminal_children: u32,
    functional_children: u32,
    telemetry_children: u32,
    phase: u32,
    cause: u32,
    cause_slot: u32,
    ready_edges: u32,
    cancelling_children: u32,
    reserved: u32,
    generation: u64,
    parent: Ticket,
}

impl Default for ScopeSnapshot {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI,
            capacity: 0,
            children: 0,
            terminal_children: 0,
            functional_children: 0,
            telemetry_children: 0,
            phase: 0,
            cause: 0,
            cause_slot: 0,
            ready_edges: 0,
            cancelling_children: 0,
            reserved: 0,
            generation: 0,
            parent: Ticket::new(1, 1, TICKET_WORKFLOW),
        }
    }
}

#[repr(C)]
struct DeadlineConfig {
    size: u32,
    abi_version: u32,
    capacity: u32,
    reserved: u32,
    notify: Option<Notify>,
    context: *mut c_void,
}

#[repr(C)]
struct ArmConfig {
    size: u32,
    abi_version: u32,
    slot: u32,
    reserved: u32,
    delay_ns: u64,
    child: Ticket,
    parent: Ticket,
    scope_generation: u64,
    epoch: u64,
    domain: u64,
    team_generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Arm {
    size: u32,
    abi_version: u32,
    slot: u32,
    reserved: u32,
    arm_generation: u64,
    child: Ticket,
    parent: Ticket,
    scope_generation: u64,
    epoch: u64,
    domain: u64,
    team_generation: u64,
}

impl Default for Arm {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI,
            slot: 0,
            reserved: 0,
            arm_generation: 0,
            child: Ticket::new(1, 1, TICKET_DEADLINE),
            parent: Ticket::new(2, 1, TICKET_WORKFLOW),
            scope_generation: 0,
            epoch: 0,
            domain: 0,
            team_generation: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeadlineEvent {
    size: u32,
    abi_version: u32,
    slot: u32,
    kind: u32,
    sequence: u64,
    scheduled_arm_generation: u64,
    current_arm_generation: u64,
    child: Ticket,
    parent: Ticket,
    scope_generation: u64,
    epoch: u64,
    domain: u64,
    team_generation: u64,
}

impl Default for DeadlineEvent {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI,
            slot: 0,
            kind: 0,
            sequence: 0,
            scheduled_arm_generation: 0,
            current_arm_generation: 0,
            child: Ticket::new(1, 1, TICKET_DEADLINE),
            parent: Ticket::new(2, 1, TICKET_WORKFLOW),
            scope_generation: 0,
            epoch: 0,
            domain: 0,
            team_generation: 0,
        }
    }
}

#[repr(C)]
struct DeadlineSnapshot {
    size: u32,
    abi_version: u32,
    capacity: u32,
    phase: u32,
    idle: u32,
    armed: u32,
    pending_events: u32,
    reserved: u32,
    published_events: u64,
    stale_events: u64,
    notifications: u64,
    cancellation_acks: u32,
    active_handlers: u32,
}

impl Default for DeadlineSnapshot {
    fn default() -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI,
            capacity: 0,
            phase: 0,
            idle: 0,
            armed: 0,
            pending_events: 0,
            reserved: 0,
            published_events: 0,
            stale_events: 0,
            notifications: 0,
            cancellation_acks: 0,
            active_handlers: 0,
        }
    }
}

struct Edge {
    count: AtomicU64,
    cancels: AtomicU64,
    owner: Thread,
}

struct Blocker {
    calls: AtomicU64,
    target: u64,
    entered: Barrier,
    release: Barrier,
}

struct GuardProbe {
    admitted: AtomicU32,
    closed: Barrier,
}

impl Blocker {
    fn new(target: u64) -> Self {
        Self {
            calls: AtomicU64::new(0),
            target,
            entered: Barrier::new(2),
            release: Barrier::new(2),
        }
    }

    fn hit(&self) {
        let call = self.calls.fetch_add(1, Ordering::AcqRel) + 1;
        if call != self.target {
            return;
        }
        self.entered.wait();
        self.release.wait();
    }
}

type ScopeCancel = unsafe extern "C" fn(*mut c_void, *const Lease, u32);

unsafe extern "C" fn notify(context: *mut c_void) {
    let edge = unsafe { &*(context.cast::<Edge>()) };
    edge.count.fetch_add(1, Ordering::Release);
    // Test observation only. Product callbacks use the retained kc_service
    // notifier edge and never run the consumer inline.
    edge.owner.unpark();
}

unsafe extern "C" fn ready(context: *mut c_void, _generation: u64, _cause: u32) {
    unsafe { notify(context) };
}

unsafe extern "C" fn count_ready(context: *mut c_void, _generation: u64, _cause: u32) {
    let edge = unsafe { &*(context.cast::<Edge>()) };
    edge.count.fetch_add(1, Ordering::Release);
}

unsafe extern "C" fn cancel(context: *mut c_void, _lease: *const Lease, _cause: u32) {
    let edge = unsafe { &*(context.cast::<Edge>()) };
    edge.cancels.fetch_add(1, Ordering::Release);
}

unsafe extern "C" fn blocking_notify(context: *mut c_void) {
    let blocker = unsafe { &*(context.cast::<Blocker>()) };
    blocker.hit();
}

unsafe extern "C" fn guard_probe(context: *mut c_void, admitted: u32) {
    let probe = unsafe { &*(context.cast::<GuardProbe>()) };
    probe.admitted.store(admitted, Ordering::Release);
    probe.closed.wait();
}

unsafe extern "C" fn blocking_ready(context: *mut c_void, _generation: u64, _cause: u32) {
    unsafe { blocking_notify(context) };
}

unsafe extern "C" {
    fn kc_fixed_scope_create(config: *const ScopeConfig, out: *mut *mut c_void) -> i32;
    fn kc_fixed_scope_add_role(
        scope: *mut c_void,
        config: *const ChildConfig,
        out: *mut u32,
    ) -> i32;
    fn kc_fixed_scope_seal(scope: *mut c_void) -> i32;
    fn kc_fixed_scope_cycle_begin(
        scope: *mut c_void,
        config: *const CycleConfig,
        leases: *mut Lease,
        capacity: usize,
    ) -> i32;
    fn kc_fixed_scope_child_terminal(scope: *mut c_void, lease: *const Lease, cause: u32) -> i32;
    fn kc_fixed_scope_cancel(
        scope: *mut c_void,
        generation: u64,
        parent: *const Ticket,
        cause: u32,
    ) -> i32;
    fn kc_fixed_scope_snapshot_get(scope: *const c_void, out: *mut ScopeSnapshot) -> i32;
    fn kc_fixed_scope_destroy(scope: *mut c_void) -> i32;

    fn kc_deadline_source_create(config: *const DeadlineConfig, out: *mut *mut c_void) -> i32;
    fn kc_deadline_source_create_manual_test(
        config: *const DeadlineConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn kc_deadline_source_arm(source: *mut c_void, config: *const ArmConfig, out: *mut Arm) -> i32;
    fn kc_deadline_source_retire(source: *mut c_void, slot: u32, generation: u64) -> i32;
    fn kc_deadline_source_disarm(source: *mut c_void, slot: u32, generation: u64) -> i32;
    fn kc_deadline_source_event_get(
        source: *const c_void,
        slot: u32,
        out: *mut DeadlineEvent,
    ) -> i32;
    fn kc_deadline_source_event_ack(source: *mut c_void, event: *const DeadlineEvent) -> i32;
    fn kc_deadline_source_advance_manual_test(source: *mut c_void, elapsed_ns: u64) -> i32;
    fn kc_deadline_source_fire_manual_test(source: *mut c_void, slot: u32) -> i32;
    fn kc_deadline_source_set_terminal_leave_hook_manual_test(
        source: *mut c_void,
        hook: Option<Notify>,
        context: *mut c_void,
    ) -> i32;
    fn kc_deadline_source_set_join_close_hook_manual_test(
        source: *mut c_void,
        hook: Option<GuardHook>,
        context: *mut c_void,
    ) -> i32;
    fn kc_deadline_source_request_stop(source: *mut c_void);
    fn kc_deadline_source_join(source: *mut c_void) -> i32;
    fn kc_deadline_source_snapshot_get(source: *const c_void, out: *mut DeadlineSnapshot) -> i32;
    fn kc_deadline_source_destroy(source: *mut c_void) -> i32;
}

fn wait_edge(edge: &Edge, target: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while edge.count.load(Ordering::Acquire) < target {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "supervision edge did not arrive");
        thread::park_timeout(remaining);
    }
}

fn scope(capacity: u32, edge: &Edge) -> *mut c_void {
    let config = ScopeConfig {
        size: size_of::<ScopeConfig>() as u32,
        abi_version: ABI,
        child_capacity: capacity,
        reserved: 0,
        ready: Some(ready),
        context: (edge as *const Edge).cast_mut().cast(),
    };
    let mut scope = std::ptr::null_mut();
    assert_eq!(unsafe { kc_fixed_scope_create(&config, &mut scope) }, 0);
    scope
}

fn add(scope: *mut c_void, edge: &Edge, slot: u32, class: u32) {
    let config = ChildConfig {
        size: size_of::<ChildConfig>() as u32,
        abi_version: ABI,
        child_class: class,
        reserved: 0,
        cancel: Some(cancel),
        context: (edge as *const Edge).cast_mut().cast(),
    };
    let mut added = u32::MAX;
    assert_eq!(
        unsafe { kc_fixed_scope_add_role(scope, &config, &mut added) },
        0
    );
    assert_eq!(added, slot);
}

fn begin(scope: *mut c_void, generation: u64, children: u32) -> Vec<Lease> {
    let tickets = (0..children)
        .map(|slot| {
            Ticket::new(
                generation * 100 + u64::from(slot) + 1,
                generation as u32 + slot + 1,
                TICKET_PASS,
            )
        })
        .collect::<Vec<_>>();
    let config = CycleConfig {
        size: size_of::<CycleConfig>() as u32,
        abi_version: ABI,
        child_count: children,
        reserved: 0,
        generation,
        parent: Ticket::new(generation * 1000 + 1, generation as u32, TICKET_WORKFLOW),
        child_tickets: tickets.as_ptr(),
    };
    let mut leases = vec![Lease::default(); children as usize];
    assert_eq!(
        unsafe { kc_fixed_scope_cycle_begin(scope, &config, leases.as_mut_ptr(), leases.len()) },
        0
    );
    leases
}

fn deadline(edge: &Edge, manual: bool, capacity: u32) -> *mut c_void {
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity,
        reserved: 0,
        notify: Some(notify),
        context: (edge as *const Edge).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    let status = if manual {
        unsafe { kc_deadline_source_create_manual_test(&config, &mut source) }
    } else {
        unsafe { kc_deadline_source_create(&config, &mut source) }
    };
    assert_eq!(status, 0);
    source
}

fn arm(source: *mut c_void, slot: u32, delay: Duration) -> Arm {
    let config = ArmConfig {
        size: size_of::<ArmConfig>() as u32,
        abi_version: ABI,
        slot,
        reserved: 0,
        delay_ns: delay.as_nanos() as u64,
        child: Ticket::new(300 + u64::from(slot), 30 + slot, TICKET_DEADLINE),
        parent: Ticket::new(400, 40, TICKET_WORKFLOW),
        scope_generation: 50,
        epoch: 60,
        domain: 70,
        team_generation: 80,
    };
    let mut armed = Arm::default();
    assert_eq!(
        unsafe { kc_deadline_source_arm(source, &config, &mut armed) },
        0
    );
    armed
}

fn event(source: *mut c_void, slot: u32) -> DeadlineEvent {
    let mut event = DeadlineEvent::default();
    assert_eq!(
        unsafe { kc_deadline_source_event_get(source, slot, &mut event) },
        0
    );
    event
}

fn deadline_snapshot(source: *mut c_void) -> DeadlineSnapshot {
    let mut snapshot = DeadlineSnapshot::default();
    assert_eq!(
        unsafe { kc_deadline_source_snapshot_get(source, &mut snapshot) },
        0
    );
    snapshot
}

unsafe fn deadline_join_destroy(source: *mut c_void) -> i32 {
    let joined = unsafe { kc_deadline_source_join(source) };
    if joined != 0 {
        return joined;
    }
    unsafe { kc_deadline_source_destroy(source) }
}

#[test]
fn canonical_deadline_ticket_kind_is_part_of_the_shared_identity() {
    let header = include_str!("../vendor/kcoro_arena/include/kc_identity.h");
    assert!(header.contains("#define KC_TICKET_KIND_DEADLINE 9u"));
    assert_eq!(TICKET_DEADLINE, 9);
}

#[test]
fn fixed_scope_fails_fast_functional_children_but_not_lossy_telemetry() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let scope = scope(4, &edge);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    add(scope, &edge, 1, CHILD_FUNCTIONAL);
    add(scope, &edge, 2, CHILD_FUNCTIONAL);
    add(scope, &edge, 3, CHILD_TELEMETRY);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);
    let leases = begin(scope, 44, 4);
    let first = leases[0];
    let failed = leases[1];
    let canceled = leases[2];
    let telemetry = leases[3];

    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &telemetry, CAUSE_FAULT) },
        0
    );
    assert_eq!(edge.count.load(Ordering::Acquire), 0);
    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &first, CAUSE_COMPLETE) },
        0
    );
    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &failed, CAUSE_FAULT) },
        0
    );
    assert_eq!(edge.count.load(Ordering::Acquire), 0);
    assert_eq!(edge.cancels.load(Ordering::Acquire), 1);
    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &canceled, CAUSE_CANCELLED) },
        0
    );
    wait_edge(&edge, 1);

    let mut snapshot = ScopeSnapshot::default();
    assert_eq!(
        unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
        0
    );
    assert_eq!(snapshot.phase, SCOPE_TERMINAL);
    assert_eq!(snapshot.children, 4);
    assert_eq!(snapshot.terminal_children, 4);
    assert_eq!(snapshot.functional_children, 3);
    assert_eq!(snapshot.telemetry_children, 1);
    assert_eq!(snapshot.cause, CAUSE_FAULT);
    assert_eq!(snapshot.cause_slot, failed.slot);
    assert_eq!(snapshot.ready_edges, 1);
    assert_ne!(
        unsafe { kc_fixed_scope_child_terminal(scope, &failed, CAUSE_FAULT) },
        0
    );
    let mut stale = canceled;
    stale.child_generation += 1;
    assert_ne!(
        unsafe { kc_fixed_scope_child_terminal(scope, &stale, CAUSE_COMPLETE) },
        0
    );
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn parent_cancel_structurally_retires_every_child_without_a_joiner() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let scope = scope(3, &edge);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    add(scope, &edge, 1, CHILD_FUNCTIONAL);
    add(scope, &edge, 2, CHILD_TELEMETRY);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);
    let leases = begin(scope, 44, 3);
    let first = leases[0];
    let second = leases[1];
    let telemetry = leases[2];
    assert_eq!(
        unsafe { kc_fixed_scope_cancel(scope, 44, &first.parent, CAUSE_DEADLINE) },
        0
    );
    assert_eq!(edge.count.load(Ordering::Acquire), 0);
    assert_eq!(edge.cancels.load(Ordering::Acquire), 3);
    for lease in [first, second, telemetry] {
        assert_eq!(
            unsafe { kc_fixed_scope_child_terminal(scope, &lease, CAUSE_CANCELLED) },
            0
        );
    }
    wait_edge(&edge, 1);
    let mut snapshot = ScopeSnapshot::default();
    assert_eq!(
        unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
        0
    );
    assert_eq!(snapshot.phase, SCOPE_TERMINAL);
    assert_eq!(snapshot.cause, CAUSE_DEADLINE);
    assert_eq!(snapshot.terminal_children, 3);
    assert_eq!(snapshot.ready_edges, 1);
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn concurrent_final_children_publish_one_parent_edge() {
    kcoro_sys::link_anchor();
    const CHILDREN: u32 = 32;
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let scope = scope(CHILDREN, &edge);
    for slot in 0..CHILDREN {
        add(scope, &edge, slot, CHILD_FUNCTIONAL);
    }
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);
    let leases = begin(scope, 44, CHILDREN);
    thread::scope(|threads| {
        for lease in leases {
            let address = scope as usize;
            threads.spawn(move || {
                assert_eq!(
                    unsafe {
                        kc_fixed_scope_child_terminal(
                            address as *mut c_void,
                            &lease,
                            CAUSE_COMPLETE,
                        )
                    },
                    0
                );
            });
        }
    });
    wait_edge(&edge, 1);
    let mut snapshot = ScopeSnapshot::default();
    assert_eq!(
        unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
        0
    );
    assert_eq!(snapshot.cause, CAUSE_COMPLETE);
    assert_eq!(snapshot.terminal_children, CHILDREN);
    assert_eq!(snapshot.ready_edges, 1);
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn sealed_scope_runs_one_million_allocation_free_cycles() {
    kcoro_sys::link_anchor();
    const CYCLES: u64 = 1_000_000;
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let config = ScopeConfig {
        size: size_of::<ScopeConfig>() as u32,
        abi_version: ABI,
        child_capacity: 1,
        reserved: 0,
        ready: Some(count_ready),
        context: (&edge as *const Edge).cast_mut().cast(),
    };
    let mut scope = std::ptr::null_mut();
    assert_eq!(unsafe { kc_fixed_scope_create(&config, &mut scope) }, 0);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);

    let mut ticket = [Ticket::new(1, 1, TICKET_PASS)];
    let mut leases = [Lease::default()];
    for generation in 1..=CYCLES {
        ticket[0] = Ticket::new(generation * 2, generation as u32, TICKET_PASS);
        let cycle = CycleConfig {
            size: size_of::<CycleConfig>() as u32,
            abi_version: ABI,
            child_count: 1,
            reserved: 0,
            generation,
            parent: Ticket::new(generation * 2 + 1, generation as u32, TICKET_WORKFLOW),
            child_tickets: ticket.as_ptr(),
        };
        assert_eq!(
            unsafe { kc_fixed_scope_cycle_begin(scope, &cycle, leases.as_mut_ptr(), leases.len()) },
            0
        );
        assert_eq!(
            unsafe { kc_fixed_scope_child_terminal(scope, &leases[0], CAUSE_COMPLETE) },
            0
        );
    }
    assert_eq!(edge.count.load(Ordering::Acquire), CYCLES);
    let mut snapshot = ScopeSnapshot::default();
    assert_eq!(
        unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
        0
    );
    assert_eq!(snapshot.phase, SCOPE_TERMINAL);
    assert_eq!(snapshot.generation, CYCLES);
    assert_eq!(snapshot.ready_edges, CYCLES as u32);

    let implementation = include_str!("../vendor/kcoro_arena/core/src/kc_fixed_scope.c");
    let begin = implementation
        .find("int kc_fixed_scope_cycle_begin(")
        .unwrap();
    let end = implementation[begin..]
        .find("int kc_fixed_scope_child_terminal(")
        .map(|offset| begin + offset)
        .unwrap();
    for forbidden in ["calloc", "malloc", "realloc", "free("] {
        assert!(
            !implementation[begin..end].contains(forbidden),
            "cycle begin contains {forbidden}"
        );
    }
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn prior_cycle_lease_is_estale_and_cannot_retire_its_successor() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let scope = scope(1, &edge);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);
    let first = begin(scope, 1, 1)[0];
    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &first, CAUSE_COMPLETE) },
        0
    );
    let second = begin(scope, 2, 1)[0];
    let stale_cancel = unsafe { kc_fixed_scope_cancel(scope, 1, &first.parent, CAUSE_CANCELLED) };
    #[cfg(target_os = "macos")]
    assert_eq!(stale_cancel, -70);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(stale_cancel, -116);
    let stale = unsafe { kc_fixed_scope_child_terminal(scope, &first, CAUSE_COMPLETE) };
    #[cfg(target_os = "macos")]
    assert_eq!(stale, -70);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(stale, -116);
    let mut snapshot = ScopeSnapshot::default();
    assert_eq!(
        unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
        0
    );
    assert_eq!(snapshot.terminal_children, 0);
    assert_eq!(
        unsafe { kc_fixed_scope_child_terminal(scope, &second, CAUSE_COMPLETE) },
        0
    );
    assert_eq!(edge.count.load(Ordering::Acquire), 2);
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn terminal_and_ready_callback_lifetime_are_published_in_order() {
    let implementation = include_str!("../vendor/kcoro_arena/core/src/kc_fixed_scope.c");
    let begin = implementation
        .find("static void publish_ready(kc_fixed_scope_t *scope)\n{")
        .expect("ready publication implementation missing");
    let end = implementation[begin..]
        .find("static void claim_terminal(")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &implementation[begin..end];
    assert!(
        body.contains("KC_SCOPE_PUBLISHERS_CLOSED"),
        "ready publication was not guarded by a drained publisher gate"
    );
    let terminal = body
        .find("KC_FIXED_SCOPE_TERMINAL")
        .expect("terminal publication missing");
    let ready = body.find("ready(context").expect("ready edge missing");
    assert!(terminal < ready, "ready ran before TERMINAL publication");
    let retained = body[..terminal]
        .rfind("active_ready")
        .expect("ready callback lifetime was not retained before TERMINAL");
    let released = body[ready..]
        .find("active_ready")
        .map(|offset| ready + offset)
        .expect("ready callback lifetime was not released after callback return");
    assert!(retained < terminal && ready < released);
}

#[test]
fn publisher_admission_is_one_bounded_rmw_without_a_retry_loop() {
    let implementation = include_str!("../vendor/kcoro_arena/core/src/kc_fixed_scope.c");
    let begin = implementation.find("static int publisher_enter(").unwrap();
    let end = implementation[begin..]
        .find("static void publisher_leave(")
        .map(|offset| begin + offset)
        .unwrap();
    let body = &implementation[begin..end];
    assert!(body.contains("atomic_fetch_add_explicit"));
    assert!(!body.contains("for (;;"));
    assert!(!body.contains("compare_exchange"));

    let leave_begin = end;
    let leave_end = implementation[leave_begin..]
        .find("static int publisher_return(")
        .map(|offset| leave_begin + offset)
        .unwrap();
    let leave = &implementation[leave_begin..leave_end];
    let retain = leave.find("active_ready, 1").unwrap();
    let count = leave.find("KC_SCOPE_PUBLISHER").unwrap();
    let ready = leave.find("publish_ready(scope)").unwrap();
    let release = leave.rfind("active_ready, 1").unwrap();
    assert!(retain < count && count < ready && ready < release);
}

#[test]
fn blocked_ready_callback_retains_scope_until_it_returns() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let blocker = Blocker::new(1);
    let config = ScopeConfig {
        size: size_of::<ScopeConfig>() as u32,
        abi_version: ABI,
        child_capacity: 1,
        reserved: 0,
        ready: Some(blocking_ready),
        context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut scope = std::ptr::null_mut();
    assert_eq!(unsafe { kc_fixed_scope_create(&config, &mut scope) }, 0);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);
    let lease = begin(scope, 1, 1)[0];
    let address = scope as usize;
    let terminal = thread::spawn(move || unsafe {
        kc_fixed_scope_child_terminal(address as *mut c_void, &lease, CAUSE_COMPLETE)
    });

    blocker.entered.wait();
    let during = unsafe { kc_fixed_scope_destroy(scope) };
    blocker.release.wait();
    assert_eq!(terminal.join().unwrap(), 0);
    assert_eq!(during, -16);
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[test]
fn final_child_and_parent_cancel_drain_before_one_ready_edge() {
    kcoro_sys::link_anchor();
    const CYCLES: u64 = 10_000;
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let scope = scope(1, &edge);
    add(scope, &edge, 0, CHILD_FUNCTIONAL);
    assert_eq!(unsafe { kc_fixed_scope_seal(scope) }, 0);

    for generation in 1..=CYCLES {
        let lease = begin(scope, generation, 1)[0];
        let parent = lease.parent;
        let gate = Barrier::new(3);
        let address = scope as usize;
        let (child, parent_status) = thread::scope(|threads| {
            let child_gate = &gate;
            let child = threads.spawn(move || {
                child_gate.wait();
                unsafe {
                    kc_fixed_scope_child_terminal(address as *mut c_void, &lease, CAUSE_COMPLETE)
                }
            });
            let parent_gate = &gate;
            let parent_task = threads.spawn(move || {
                parent_gate.wait();
                unsafe {
                    kc_fixed_scope_cancel(
                        address as *mut c_void,
                        generation,
                        &parent,
                        CAUSE_DEADLINE,
                    )
                }
            });
            gate.wait();
            (child.join().unwrap(), parent_task.join().unwrap())
        });
        assert_eq!(child, 0);
        assert!(
            parent_status == 0
                || parent_status == race_already()
                || parent_status == race_canceled(),
            "unexpected parent cancellation race status {parent_status}"
        );
        let count = edge.count.load(Ordering::Acquire);
        let mut snapshot = ScopeSnapshot::default();
        assert_eq!(
            unsafe { kc_fixed_scope_snapshot_get(scope, &mut snapshot) },
            0
        );
        assert_eq!(
            count,
            generation,
            "cycle {generation} missed/duplicated ready: parent={parent_status}, phase={}, cause={}, terminal={}, cancelling={}, ready_edges={}",
            snapshot.phase,
            snapshot.cause,
            snapshot.terminal_children,
            snapshot.cancelling_children,
            snapshot.ready_edges,
        );
        assert_eq!(snapshot.phase, SCOPE_TERMINAL);
        assert_eq!(snapshot.terminal_children, 1);
        assert_eq!(snapshot.ready_edges, generation as u32);
        assert!(snapshot.cause == CAUSE_COMPLETE || snapshot.cause == CAUSE_DEADLINE);
    }
    assert_eq!(unsafe { kc_fixed_scope_destroy(scope) }, 0);
}

#[cfg(target_os = "macos")]
const fn race_already() -> i32 {
    -37
}

#[cfg(not(target_os = "macos"))]
const fn race_already() -> i32 {
    -114
}

#[cfg(target_os = "macos")]
const fn race_canceled() -> i32 {
    -89
}

#[cfg(not(target_os = "macos"))]
const fn race_canceled() -> i32 {
    -125
}

#[cfg(target_os = "macos")]
const fn stale_errno() -> i32 {
    -70
}

#[cfg(not(target_os = "macos"))]
const fn stale_errno() -> i32 {
    -116
}

#[test]
fn deadline_stop_before_first_arm_joins_without_publishing_work() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 2);

    unsafe { kc_deadline_source_request_stop(source) };
    let stopped = deadline_snapshot(source);
    assert_eq!(stopped.phase, SOURCE_STOPPED);
    assert_eq!(stopped.cancellation_acks, 2);
    assert_eq!(stopped.published_events, 0);
    assert_eq!(stopped.pending_events, 0);
    let config = ArmConfig {
        size: size_of::<ArmConfig>() as u32,
        abi_version: ABI,
        slot: 0,
        reserved: 0,
        delay_ns: 1,
        child: Ticket::new(300, 30, TICKET_DEADLINE),
        parent: Ticket::new(400, 40, TICKET_WORKFLOW),
        scope_generation: 50,
        epoch: 60,
        domain: 70,
        team_generation: 80,
    };
    let mut rejected = Arm::default();
    assert_eq!(
        unsafe { kc_deadline_source_arm(source, &config, &mut rejected) },
        race_canceled()
    );
    assert_eq!(unsafe { kc_deadline_source_join(source) }, 0);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, 0);
}

#[test]
fn deadline_repeated_stop_and_join_are_idempotent() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);

    assert_eq!(unsafe { kc_deadline_source_join(source) }, -16);
    unsafe { kc_deadline_source_request_stop(source) };
    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(unsafe { kc_deadline_source_join(source) }, 0);
    assert_eq!(unsafe { kc_deadline_source_join(source) }, 0);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, 0);
}

#[test]
fn deadline_destroy_rejects_a_stopped_source_until_join() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);

    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, -16);
    assert_eq!(unsafe { kc_deadline_source_join(source) }, 0);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, 0);
}

#[test]
fn deadline_terminal_signal_guard_survives_concurrent_join_and_destroy() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let leave = Blocker::new(1);
    let close = GuardProbe {
        admitted: AtomicU32::new(0),
        closed: Barrier::new(2),
    };
    let source = deadline(&edge, true, 1);
    assert_eq!(
        unsafe {
            kc_deadline_source_set_terminal_leave_hook_manual_test(
                source,
                Some(blocking_notify),
                (&leave as *const Blocker).cast_mut().cast(),
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            kc_deadline_source_set_join_close_hook_manual_test(
                source,
                Some(guard_probe),
                (&close as *const GuardProbe).cast_mut().cast(),
            )
        },
        0
    );

    let stop_address = source as usize;
    let stop = thread::spawn(move || unsafe {
        kc_deadline_source_request_stop(stop_address as *mut c_void)
    });
    leave.entered.wait();
    let terminal = deadline_snapshot(source);
    assert_eq!(terminal.phase, SOURCE_STOPPED);
    assert_eq!(terminal.active_handlers, 0);
    assert_eq!(unsafe { kc_deadline_source_join(source) }, 0);

    let destroy_address = source as usize;
    let destroy = thread::spawn(move || unsafe {
        kc_deadline_source_destroy(destroy_address as *mut c_void)
    });
    close.closed.wait();
    let admitted = close.admitted.load(Ordering::Acquire);
    leave.release.wait();
    stop.join().unwrap();
    assert_eq!(destroy.join().unwrap(), 0);
    assert_eq!(admitted, 1, "destroy did not drain the terminal signal guard");
}

#[test]
fn exact_generation_retire_is_silent_and_rearm_safe() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    let first = arm(source, 0, Duration::from_millis(200));
    assert_eq!(first.arm_generation, 1);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, first.arm_generation) },
        RETIRE_RETIRED
    );

    let quiet = deadline_snapshot(source);
    assert_eq!(quiet.idle, 1);
    assert_eq!(quiet.armed, 0);
    assert_eq!(quiet.pending_events, 0);
    assert_eq!(quiet.published_events, 0);
    assert_eq!(quiet.stale_events, 0);
    assert_eq!(quiet.notifications, 0);
    assert_eq!(edge.count.load(Ordering::Acquire), 0);
    assert_ne!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);

    let second = arm(source, 0, Duration::from_millis(200));
    assert_eq!(second.arm_generation, first.arm_generation + 2);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, first.arm_generation) },
        stale_errno(),
        "an old completion retired its successor"
    );
    assert_eq!(deadline_snapshot(source).armed, 1);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, second.arm_generation) },
        RETIRE_RETIRED
    );
    let retired = deadline_snapshot(source);
    assert_eq!(retired.idle, 1);
    assert_eq!(retired.pending_events, 0);
    assert_eq!(retired.published_events, 0);
    assert_eq!(retired.notifications, 0);
    assert_eq!(edge.count.load(Ordering::Acquire), 0);

    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn exact_generation_retire_reports_when_expiry_already_won() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    let first = arm(source, 0, Duration::from_millis(1));
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, first.arm_generation) },
        RETIRE_EXPIRY_WON
    );
    let expired = event(source, 0);
    assert_eq!(expired.kind, EVENT_EXPIRED);
    assert_eq!(expired.scheduled_arm_generation, first.arm_generation);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, first.arm_generation) },
        RETIRE_EXPIRY_WON,
        "an acknowledged expiry lost its exact-generation outcome"
    );

    let second = arm(source, 0, Duration::from_millis(200));
    assert_eq!(second.arm_generation, first.arm_generation + 1);
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, second.arm_generation) },
        RETIRE_RETIRED
    );
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.pending_events, 0);
    assert_eq!(snapshot.published_events, 1);
    assert_eq!(snapshot.notifications, 1);
    assert_eq!(edge.count.load(Ordering::Acquire), 1);

    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn exact_generation_retire_and_expiry_share_one_terminal_cas() {
    kcoro_sys::link_anchor();
    const CYCLES: u64 = 256;
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    let address = source as usize;
    let mut expiries = 0_u64;
    let mut prior_generation = 0_u64;
    for _ in 0..CYCLES {
        let armed = arm(source, 0, Duration::from_millis(1));
        assert!(armed.arm_generation > prior_generation);
        prior_generation = armed.arm_generation;
        assert_eq!(
            unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
            0
        );
        let generation = armed.arm_generation;
        let gate = Barrier::new(3);
        let (retired, fired) = thread::scope(|threads| {
            let retire_gate = &gate;
            let retire = threads.spawn(move || {
                retire_gate.wait();
                unsafe { kc_deadline_source_retire(address as *mut c_void, 0, generation) }
            });
            let fire_gate = &gate;
            let fire = threads.spawn(move || {
                fire_gate.wait();
                unsafe { kc_deadline_source_fire_manual_test(address as *mut c_void, 0) }
            });
            gate.wait();
            (retire.join().unwrap(), fire.join().unwrap())
        });
        if retired == RETIRE_RETIRED {
            assert_ne!(fired, 0);
            let mut missing = DeadlineEvent::default();
            assert_ne!(
                unsafe { kc_deadline_source_event_get(source, 0, &mut missing) },
                0
            );
            continue;
        }
        assert_eq!(retired, RETIRE_EXPIRY_WON);
        assert_eq!(fired, 0);
        let expired = event(source, 0);
        assert_eq!(expired.kind, EVENT_EXPIRED);
        assert_eq!(expired.scheduled_arm_generation, generation);
        assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);
        expiries += 1;
    }
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.idle, 1);
    assert_eq!(snapshot.armed, 0);
    assert_eq!(snapshot.pending_events, 0);
    assert_eq!(snapshot.published_events, expiries);
    assert_eq!(snapshot.stale_events, 0);
    assert_eq!(snapshot.notifications, expiries);
    assert_eq!(edge.count.load(Ordering::Acquire), expiries);

    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn disarm_promptly_publishes_stale_identity_then_allows_reuse() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    let first = arm(source, 0, Duration::from_millis(200));
    assert_eq!(first.arm_generation, 1);
    assert_eq!(
        unsafe { kc_deadline_source_disarm(source, 0, first.arm_generation) },
        0
    );
    wait_edge(&edge, 1);
    let stale = event(source, 0);
    assert_eq!(stale.kind, EVENT_STALE);
    assert_eq!(stale.scheduled_arm_generation, 1);
    assert_eq!(stale.current_arm_generation, 2);
    assert_eq!(stale.child, first.child);
    assert_eq!(stale.parent, first.parent);
    assert_eq!(stale.scope_generation, 50);
    assert_eq!(stale.epoch, 60);
    assert_eq!(stale.domain, 70);
    assert_eq!(stale.team_generation, 80);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &stale) }, 0);

    let second = arm(source, 0, Duration::from_millis(1));
    assert_eq!(second.arm_generation, 3);
    assert_ne!(
        unsafe { kc_deadline_source_fire_manual_test(source, 0) },
        0,
        "a queued callback from the first arm expired its successor early"
    );
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    wait_edge(&edge, 2);
    let expired = event(source, 0);
    assert_eq!(expired.kind, EVENT_EXPIRED);
    assert_eq!(expired.scheduled_arm_generation, 3);
    assert_eq!(expired.current_arm_generation, 3);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);

    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 3);
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.phase, SOURCE_STOPPED);
    assert_eq!(snapshot.cancellation_acks, 1);
    assert_eq!(snapshot.published_events, 2);
    assert_eq!(snapshot.stale_events, 1);
    assert_eq!(snapshot.active_handlers, 0);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn stale_ack_cannot_clear_a_reused_slots_successor_event() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);

    let first = arm(source, 0, Duration::from_millis(1));
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    let stale = event(source, 0);
    assert_eq!(stale.current_arm_generation, first.arm_generation);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &stale) }, 0);

    let second = arm(source, 0, Duration::from_millis(1));
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    let successor = event(source, 0);
    assert_eq!(successor.current_arm_generation, second.arm_generation);
    assert_ne!(successor.sequence, stale.sequence);

    let stale_status = unsafe { kc_deadline_source_event_ack(source, &stale) };
    #[cfg(target_os = "macos")]
    assert_eq!(stale_status, -70);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(stale_status, -116);
    let retained = event(source, 0);
    assert_eq!(retained.sequence, successor.sequence);
    assert_eq!(retained.current_arm_generation, second.arm_generation);
    assert_eq!(retained.child, second.child);
    assert_eq!(
        unsafe { kc_deadline_source_event_ack(source, &retained) },
        0
    );

    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn copied_event_cannot_ack_another_slot_with_the_same_sequence_and_generation() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 2);
    let first = arm(source, 0, Duration::from_millis(1));
    let second = arm(source, 1, Duration::from_millis(1));
    assert_eq!(first.arm_generation, second.arm_generation);
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 1) }, 0);
    let first_event = event(source, 0);
    let second_event = event(source, 1);
    assert_eq!(first_event.sequence, second_event.sequence);
    assert_eq!(
        first_event.current_arm_generation,
        second_event.current_arm_generation
    );

    let mut corrupted = first_event;
    corrupted.slot = 1;
    let status = unsafe { kc_deadline_source_event_ack(source, &corrupted) };
    #[cfg(target_os = "macos")]
    assert_eq!(status, -70);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(status, -116);
    let retained = event(source, 1);
    assert_eq!(retained.child, second.child);
    assert_eq!(retained.parent, second.parent);

    assert_eq!(
        unsafe { kc_deadline_source_event_ack(source, &first_event) },
        0
    );
    assert_eq!(
        unsafe { kc_deadline_source_event_ack(source, &second_event) },
        0
    );
    unsafe { kc_deadline_source_request_stop(source) };
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn deadline_join_parks_until_final_cancellation_notify_returns() {
    kcoro_sys::link_anchor();
    let blocker = Blocker::new(2);
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity: 1,
        reserved: 0,
        notify: Some(blocking_notify),
        context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_deadline_source_create_manual_test(&config, &mut source) },
        0
    );
    let address = source as usize;
    let stop =
        thread::spawn(move || unsafe { kc_deadline_source_request_stop(address as *mut c_void) });

    blocker.entered.wait();
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.phase, SOURCE_STOPPED);
    assert_eq!(snapshot.active_handlers, 1);
    let join_address = source as usize;
    let join =
        thread::spawn(move || unsafe { kc_deadline_source_join(join_address as *mut c_void) });
    let during = unsafe { kc_deadline_source_destroy(source) };
    blocker.release.wait();
    stop.join().unwrap();
    assert_eq!(join.join().unwrap(), 0);

    assert_eq!(during, -16);
    assert_eq!(blocker.calls.load(Ordering::Acquire), 2);
    assert_eq!(deadline_snapshot(source).active_handlers, 0);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, 0);
}

#[test]
fn stopped_is_withheld_until_the_cancellation_walk_has_finished() {
    kcoro_sys::link_anchor();
    let blocker = Blocker::new(1);
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity: 1,
        reserved: 0,
        notify: Some(blocking_notify),
        context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_deadline_source_create_manual_test(&config, &mut source) },
        0
    );
    let address = source as usize;
    let stop =
        thread::spawn(move || unsafe { kc_deadline_source_request_stop(address as *mut c_void) });

    blocker.entered.wait();
    let during = deadline_snapshot(source);
    assert_eq!(during.phase, SOURCE_STOPPING);
    assert_eq!(during.cancellation_acks, 1);
    assert_eq!(during.active_handlers, 2);
    blocker.release.wait();
    stop.join().unwrap();

    let settled = deadline_snapshot(source);
    assert_eq!(settled.phase, SOURCE_STOPPED);
    assert_eq!(settled.active_handlers, 0);
    assert_eq!(blocker.calls.load(Ordering::Acquire), 2);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn disarm_notify_retains_arm_admission_until_callback_returns() {
    kcoro_sys::link_anchor();
    let blocker = Blocker::new(1);
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity: 1,
        reserved: 0,
        notify: Some(blocking_notify),
        context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_deadline_source_create_manual_test(&config, &mut source) },
        0
    );
    let armed = arm(source, 0, Duration::from_secs(1));
    let address = source as usize;
    let disarm = thread::spawn(move || unsafe {
        kc_deadline_source_disarm(address as *mut c_void, 0, armed.arm_generation)
    });

    blocker.entered.wait();
    unsafe { kc_deadline_source_request_stop(source) };
    let stale = event(source, 0);
    assert_eq!(stale.kind, EVENT_STALE);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &stale) }, 0);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPING);
    let during = unsafe { kc_deadline_source_destroy(source) };
    blocker.release.wait();
    assert_eq!(disarm.join().unwrap(), 0);

    assert_eq!(during, -16);
    assert_eq!(blocker.calls.load(Ordering::Acquire), 3);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn deadline_join_parks_across_a_queued_expiry_handler() {
    kcoro_sys::link_anchor();
    let blocker = Blocker::new(1);
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity: 1,
        reserved: 0,
        notify: Some(blocking_notify),
        context: (&blocker as *const Blocker).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_deadline_source_create_manual_test(&config, &mut source) },
        0
    );
    let _ = arm(source, 0, Duration::from_millis(1));
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    let address = source as usize;
    let expire = thread::spawn(move || unsafe {
        kc_deadline_source_fire_manual_test(address as *mut c_void, 0)
    });

    blocker.entered.wait();
    unsafe { kc_deadline_source_request_stop(source) };
    let expired = event(source, 0);
    assert_eq!(expired.kind, EVENT_EXPIRED);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.phase, SOURCE_STOPPED);
    assert_eq!(snapshot.active_handlers, 1);
    let join_address = source as usize;
    let join =
        thread::spawn(move || unsafe { kc_deadline_source_join(join_address as *mut c_void) });
    let during = unsafe { kc_deadline_source_destroy(source) };
    blocker.release.wait();
    assert_eq!(expire.join().unwrap(), 0);
    assert_eq!(join.join().unwrap(), 0);

    assert_eq!(during, -16);
    assert_eq!(blocker.calls.load(Ordering::Acquire), 3);
    assert_eq!(deadline_snapshot(source).active_handlers, 0);
    assert_eq!(unsafe { kc_deadline_source_destroy(source) }, 0);
}

#[test]
fn three_permanent_slots_cover_repeated_prepare_commit_and_forced_endpoint_arms() {
    kcoro_sys::link_anchor();
    const CYCLES: u64 = 10_000;
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 3);
    for cycle in 1..=CYCLES {
        for (slot, delay_ns) in [
            (0_u32, 200_000_000_u64),
            (1, 500_000_000),
            (2, 30_000_000_000),
        ] {
            let config = ArmConfig {
                size: size_of::<ArmConfig>() as u32,
                abi_version: ABI,
                slot,
                reserved: 0,
                delay_ns,
                child: Ticket::new(cycle * 10 + u64::from(slot), cycle as u32, TICKET_DEADLINE),
                parent: Ticket::new(cycle * 10 + 9, cycle as u32, TICKET_WORKFLOW),
                scope_generation: cycle,
                epoch: cycle,
                domain: u64::from(slot) + 1,
                team_generation: cycle,
            };
            let mut armed = Arm::default();
            assert_eq!(
                unsafe { kc_deadline_source_arm(source, &config, &mut armed) },
                0
            );
            assert_eq!(
                unsafe { kc_deadline_source_disarm(source, slot, armed.arm_generation) },
                0
            );
            let stale = event(source, slot);
            assert_eq!(stale.kind, EVENT_STALE);
            assert_eq!(stale.child, config.child);
            assert_eq!(stale.parent, config.parent);
            assert_eq!(stale.scope_generation, cycle);
            assert_eq!(stale.epoch, cycle);
            assert_eq!(stale.domain, u64::from(slot) + 1);
            assert_eq!(stale.team_generation, cycle);
            assert_eq!(unsafe { kc_deadline_source_event_ack(source, &stale) }, 0);
        }
    }
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.capacity, 3);
    assert_eq!(snapshot.idle, 3);
    assert_eq!(snapshot.armed, 0);
    assert_eq!(snapshot.pending_events, 0);
    assert_eq!(snapshot.published_events, CYCLES * 3);
    assert_eq!(snapshot.stale_events, CYCLES * 3);
    assert_eq!(edge.count.load(Ordering::Acquire), CYCLES * 3);
    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, CYCLES * 3 + 3);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn completion_disarm_and_stop_races_never_publish_twice_or_reuse_live_state() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    for iteration in 0..256_u64 {
        let armed = arm(source, 0, Duration::from_millis(1));
        assert_eq!(
            unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
            0
        );
        let address = source as usize;
        let generation = armed.arm_generation;
        let (disarm, expire) = thread::scope(|threads| {
            let disarm = threads.spawn(move || unsafe {
                kc_deadline_source_disarm(address as *mut c_void, 0, generation)
            });
            let expire = threads.spawn(move || unsafe {
                kc_deadline_source_fire_manual_test(address as *mut c_void, 0)
            });
            (disarm.join().unwrap(), expire.join().unwrap())
        });
        assert!(disarm == 0 || expire == 0);
        wait_edge(&edge, iteration + 1);
        let settled = event(source, 0);
        assert!(settled.kind == EVENT_EXPIRED || settled.kind == EVENT_STALE);
        assert_eq!(settled.scheduled_arm_generation, generation);
        assert_eq!(unsafe { kc_deadline_source_event_ack(source, &settled) }, 0);
    }
    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 257);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[test]
fn source_stop_preserves_an_unacknowledged_expiry_record() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, true, 1);
    let armed = arm(source, 0, Duration::from_millis(1));
    assert_eq!(
        unsafe { kc_deadline_source_advance_manual_test(source, 1_000_000) },
        0
    );
    assert_eq!(unsafe { kc_deadline_source_fire_manual_test(source, 0) }, 0);
    wait_edge(&edge, 1);
    let published = event(source, 0);
    assert_eq!(published.kind, EVENT_EXPIRED);

    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 2);
    let stopped = deadline_snapshot(source);
    assert_eq!(stopped.phase, SOURCE_STOPPED);
    assert_eq!(stopped.pending_events, 1);
    assert_ne!(unsafe { deadline_join_destroy(source) }, 0);

    let retained = event(source, 0);
    assert_eq!(retained.sequence, published.sequence);
    assert_eq!(retained.child, armed.child);
    assert_eq!(
        unsafe { kc_deadline_source_event_ack(source, &retained) },
        0
    );
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn gcd_source_is_monotonic_one_shot_and_cancellation_ack_prevents_late_uaf() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, false, 1);
    let armed = arm(source, 0, Duration::from_millis(5));
    wait_edge(&edge, 1);
    let expired = event(source, 0);
    assert_eq!(expired.kind, EVENT_EXPIRED);
    assert_eq!(expired.scheduled_arm_generation, armed.arm_generation);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);
    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 2);
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.phase, SOURCE_STOPPED);
    assert_eq!(snapshot.cancellation_acks, 1);
    assert_eq!(snapshot.active_handlers, 0);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn gcd_disarm_promptly_publishes_stale_identity_and_quarantines_no_slot() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, false, 1);
    let armed = arm(source, 0, Duration::from_millis(10));
    assert_eq!(
        unsafe { kc_deadline_source_disarm(source, 0, armed.arm_generation) },
        0
    );
    wait_edge(&edge, 1);
    let stale = event(source, 0);
    assert_eq!(stale.kind, EVENT_STALE);
    assert_eq!(stale.scheduled_arm_generation, armed.arm_generation);
    assert_eq!(stale.current_arm_generation, armed.arm_generation + 1);
    assert_eq!(stale.child, armed.child);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &stale) }, 0);
    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 2);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn gcd_retire_is_silent_and_disables_the_old_timer_before_rearm() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, false, 1);
    let first = arm(source, 0, Duration::from_secs(1));
    assert_eq!(
        unsafe { kc_deadline_source_retire(source, 0, first.arm_generation) },
        RETIRE_RETIRED
    );
    let quiet = deadline_snapshot(source);
    assert_eq!(quiet.idle, 1);
    assert_eq!(quiet.pending_events, 0);
    assert_eq!(quiet.published_events, 0);
    assert_eq!(quiet.notifications, 0);
    assert_eq!(edge.count.load(Ordering::Acquire), 0);

    let second = arm(source, 0, Duration::from_millis(5));
    assert_eq!(second.arm_generation, first.arm_generation + 2);
    wait_edge(&edge, 1);
    let expired = event(source, 0);
    assert_eq!(expired.kind, EVENT_EXPIRED);
    assert_eq!(expired.scheduled_arm_generation, second.arm_generation);
    assert_eq!(expired.child, second.child);
    assert_eq!(unsafe { kc_deadline_source_event_ack(source, &expired) }, 0);
    let settled = deadline_snapshot(source);
    assert_eq!(settled.published_events, 1);
    assert_eq!(settled.notifications, 1);

    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 2);
    assert_eq!(deadline_snapshot(source).phase, SOURCE_STOPPED);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn gcd_cancel_ack_quiesces_a_future_handler_before_storage_is_freed() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let source = deadline(&edge, false, 1);
    let _ = arm(source, 0, Duration::from_millis(50));
    unsafe { kc_deadline_source_request_stop(source) };
    wait_edge(&edge, 1);
    let snapshot = deadline_snapshot(source);
    assert_eq!(snapshot.phase, SOURCE_STOPPED);
    assert_eq!(snapshot.cancellation_acks, 1);
    assert_eq!(snapshot.published_events, 0);
    assert_eq!(snapshot.active_handlers, 0);
    assert_eq!(unsafe { deadline_join_destroy(source) }, 0);

    thread::sleep(Duration::from_millis(75));
    assert_eq!(
        edge.count.load(Ordering::Acquire),
        1,
        "canceled handler touched released deadline storage"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_apple_production_arm_is_explicitly_unsupported() {
    kcoro_sys::link_anchor();
    let edge = Edge {
        count: AtomicU64::new(0),
        cancels: AtomicU64::new(0),
        owner: thread::current(),
    };
    let config = DeadlineConfig {
        size: size_of::<DeadlineConfig>() as u32,
        abi_version: ABI,
        capacity: 1,
        reserved: 0,
        notify: Some(notify),
        context: (&edge as *const Edge).cast_mut().cast(),
    };
    let mut source = std::ptr::null_mut();
    assert_ne!(
        unsafe { kc_deadline_source_create(&config, &mut source) },
        0
    );
    assert!(
        source.is_null(),
        "unsupported construction published a source"
    );
}

#[test]
fn supervision_surface_has_only_administrative_deadline_join() {
    let scope = include_str!("../vendor/kcoro_arena/include/kc_fixed_scope.h");
    let deadline = include_str!("../vendor/kcoro_arena/include/kc_deadline.h");
    for forbidden in ["_wait", "channel", "dispatch_walltime"] {
        assert!(
            !scope.contains(forbidden),
            "fixed scope contains {forbidden}"
        );
        assert!(
            !deadline.contains(forbidden),
            "deadline surface contains {forbidden}"
        );
    }
    assert!(!scope.contains("_join"));
    assert!(deadline.contains("int kc_deadline_source_join("));
    assert!(deadline.contains("Terminal administrative teardown only"));
    assert!(deadline.contains("forbidden from numerical, coordinator, service"));
    assert!(deadline.contains("audio callback paths"));
    assert!(deadline.contains("never a product route,"));
    assert!(deadline.contains("conversation, or numerical scratch owner"));
    let implementation = include_str!("../vendor/kcoro_arena/core/src/kc_deadline.c");
    assert!(implementation.contains("dispatch_time(DISPATCH_TIME_NOW"));
    assert!(!implementation.contains("dispatch_walltime"));
    let begin = implementation.find("static int deliver_hint(").unwrap();
    let end = implementation[begin..]
        .find("static void cancel_ack(")
        .map(|offset| begin + offset)
        .unwrap();
    let handler = &implementation[begin..end];
    for forbidden in ["calloc", "malloc", "mutex", "_wait", "sleep"] {
        assert!(
            !handler.contains(forbidden),
            "deadline handler contains {forbidden}"
        );
    }
    let notify = handler.find("notify(context)").unwrap();
    let release = handler.rfind("handler_leave(source)").unwrap();
    assert!(
        notify < release,
        "expiry handler released lifetime before notify"
    );

    let leave_begin = implementation.find("static void handler_leave(").unwrap();
    let leave_end = implementation[leave_begin..]
        .find("static int publish_stopped(")
        .map(|offset| leave_begin + offset)
        .unwrap();
    let leave = &implementation[leave_begin..leave_end];
    let acquire = leave.find("kc_port_wait_u32_signal_acquire").unwrap();
    let terminal = leave.find("atomic_fetch_sub_explicit").unwrap();
    let publish = leave.find("kc_atomic_u32_fetch_add_release").unwrap();
    let wake = leave.find("kc_port_wait_u32_signal_all").unwrap();
    let unpin = leave.find("kc_port_wait_u32_signal_release").unwrap();
    assert!(acquire < terminal && terminal < publish && publish < wake && wake < unpin);

    let cancel_begin = implementation.rfind("static void cancel_ack(").unwrap();
    let cancel_end = implementation[cancel_begin..]
        .find("#if defined(__APPLE__)")
        .map(|offset| cancel_begin + offset)
        .unwrap();
    let cancel = &implementation[cancel_begin..cancel_end];
    assert!(
        cancel.find("notify(context)").unwrap() < cancel.rfind("handler_leave(source)").unwrap(),
        "cancel handler released lifetime before notify"
    );

    let stop_begin = implementation
        .find("static void start_cancellation(")
        .unwrap();
    let stop_end = implementation[stop_begin..]
        .find("static int arm_enter(")
        .map(|offset| stop_begin + offset)
        .unwrap();
    let stop = &implementation[stop_begin..stop_end];
    assert!(
        stop.find("notify(context)").unwrap() < stop.rfind("handler_leave(source)").unwrap(),
        "cancellation walk released lifetime before its final notify"
    );

    let disarm_begin = implementation
        .find("int kc_deadline_source_disarm(")
        .unwrap();
    let disarm_end = implementation[disarm_begin..]
        .find("int kc_deadline_source_event_get(")
        .map(|offset| disarm_begin + offset)
        .unwrap();
    let disarm = &implementation[disarm_begin..disarm_end];
    assert!(
        disarm.find("notify(context)").unwrap() < disarm.rfind("arm_leave(source)").unwrap(),
        "disarm released publisher lifetime before notify"
    );

    let retire_begin = implementation
        .find("int kc_deadline_source_retire(")
        .expect("silent deadline retirement is missing");
    let retire_end = implementation[retire_begin..]
        .find("int kc_deadline_source_disarm(")
        .map(|offset| retire_begin + offset)
        .unwrap();
    let retire = &implementation[retire_begin..retire_end];
    for forbidden in [
        "publish_event_record",
        "notify(context)",
        "calloc",
        "malloc",
        "_wait",
        "sleep",
    ] {
        assert!(
            !retire.contains(forbidden),
            "silent deadline retirement contains {forbidden}"
        );
    }

    let join_begin = implementation
        .find("int kc_deadline_source_join(")
        .expect("administrative deadline join is missing");
    let join_end = implementation[join_begin..]
        .find("int kc_deadline_source_snapshot_get(")
        .map(|offset| join_begin + offset)
        .unwrap();
    let join = &implementation[join_begin..join_end];
    assert!(join.contains("kc_port_wait_u32("));
    for forbidden in ["sched_yield", "nanosleep", "usleep"] {
        assert!(
            !join.contains(forbidden),
            "administrative deadline join contains {forbidden}"
        );
    }
    assert_eq!(CAUSE_CANCELLED, 2);
    assert_eq!(TICKET_CONTROL, 8);
}
