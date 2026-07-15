use std::ffi::c_void;
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

const ABI: u32 = 1;
const EXECUTION_COMPLETED: i32 = 1;
const STATE_COMMITTED: i32 = 1;
const PUBLICATION_COMMITTED: i32 = 1;
const CAUSE_SUCCESS: i32 = 0;

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
#[derive(Clone, Copy, Default)]
struct Id {
    epoch: u64,
    sequence: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TicketId {
    runtime_epoch: u64,
    sequence: u64,
    slot: u32,
    generation: u32,
    kind: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DescriptorId {
    runtime_epoch: u64,
    slot: u32,
    generation: u32,
}

#[repr(C)]
struct TicketEvent {
    size: u32,
    abi_version: u32,
    flags: u32,
    reserved0: u32,
    ticket: TicketId,
    parent: TicketId,
    correlation: Id,
    trace: Id,
    context_id: u64,
    epoch: u64,
    execution_status: i32,
    state_status: i32,
    publication_status: i32,
    terminal_cause: i32,
    status_code: i32,
    reserved1: u32,
    result: DescriptorId,
    accepted_ns: u64,
    dispatched_ns: u64,
    completed_ns: u64,
    published_ns: u64,
}

type Callback = unsafe extern "C" fn(*mut c_void, *const TicketEvent);
type ContextFn = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct TicketConfig {
    size: u32,
    abi_version: u32,
    flags: u32,
    kind: u32,
    parent: TicketId,
    correlation: Id,
    trace: Id,
    context_id: u64,
    epoch: u64,
    deadline_ns: u64,
    deadline_mode: i32,
    reserved: u32,
    callback: Option<Callback>,
    callback_context: *mut c_void,
    context_retain: Option<ContextFn>,
    context_release: Option<ContextFn>,
}

#[repr(C)]
struct TicketCompletion {
    size: u32,
    abi_version: u32,
    execution_status: i32,
    state_status: i32,
    publication_status: i32,
    terminal_cause: i32,
    status_code: i32,
    reserved: u32,
    result: *mut c_void,
}

struct State {
    result: Mutex<Option<(i32, i32, ThreadId)>>,
    ready: Condvar,
}

unsafe extern "C" fn retain(_: *mut c_void) {}
unsafe extern "C" fn release(_: *mut c_void) {}

unsafe extern "C" fn complete(context: *mut c_void, event: *const TicketEvent) {
    // SAFETY: the test keeps State and the ticket alive until the callback is drained.
    let state = unsafe { &*(context.cast::<State>()) };
    // SAFETY: kcoro_arena guarantees a non-null event for the callback duration.
    let event = unsafe { &*event };
    let mut result = state.result.lock().unwrap();
    *result = Some((
        event.terminal_cause,
        event.status_code,
        std::thread::current().id(),
    ));
    state.ready.notify_one();
}

extern "C" {
    fn kc_runtime_create(config: *const RuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_run_until_idle(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;
    fn kc_ticket_create(
        runtime: *mut c_void,
        config: *const TicketConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn kc_ticket_accept(ticket: *mut c_void) -> i32;
    fn kc_ticket_dispatch(ticket: *mut c_void) -> i32;
    fn kc_ticket_complete(ticket: *mut c_void, completion: *const TicketCompletion) -> i32;
    fn kc_ticket_complete_id(
        runtime: *mut c_void,
        ticket: TicketId,
        completion: *const TicketCompletion,
    ) -> i32;
    fn kc_ticket_cancel_id(runtime: *mut c_void, ticket: TicketId) -> i32;
    fn kc_ticket_id_get(ticket: *const c_void) -> TicketId;
    fn kc_ticket_release(ticket: *mut c_void);
}

#[test]
fn cargo_links_exact_ticket_completion() {
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
    // SAFETY: all pointers target live, correctly laid-out C ABI values.
    assert_eq!(unsafe { kc_runtime_create(&config, &mut runtime) }, 0);
    assert!(!runtime.is_null());
    assert_eq!(unsafe { kc_runtime_start(runtime) }, 0);

    let state = State {
        result: Mutex::new(None),
        ready: Condvar::new(),
    };
    let ticket_config = TicketConfig {
        size: size_of::<TicketConfig>() as u32,
        abi_version: ABI,
        flags: 0,
        kind: 5,
        parent: TicketId::default(),
        correlation: Id::default(),
        trace: Id::default(),
        context_id: 9,
        epoch: 11,
        deadline_ns: 0,
        deadline_mode: 0,
        reserved: 0,
        callback: Some(complete),
        callback_context: (&state as *const State).cast_mut().cast(),
        context_retain: Some(retain),
        context_release: Some(release),
    };
    let mut ticket = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_ticket_create(runtime, &ticket_config, &mut ticket) },
        0
    );
    assert_eq!(unsafe { kc_ticket_accept(ticket) }, 0);
    assert_eq!(unsafe { kc_ticket_dispatch(ticket) }, 0);
    let id = unsafe { kc_ticket_id_get(ticket) };
    assert_eq!(id.kind, 5);
    let completion = TicketCompletion {
        size: size_of::<TicketCompletion>() as u32,
        abi_version: ABI,
        execution_status: EXECUTION_COMPLETED,
        state_status: STATE_COMMITTED,
        publication_status: PUBLICATION_COMMITTED,
        terminal_cause: CAUSE_SUCCESS,
        status_code: 0,
        reserved: 0,
        result: std::ptr::null_mut(),
    };
    assert_eq!(
        unsafe { kc_ticket_complete_id(runtime, id, &completion) },
        1
    );
    assert_eq!(unsafe { kc_ticket_complete(ticket, &completion) }, 0);

    let producer = std::thread::current().id();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut result = state.result.lock().unwrap();
    while result.is_none() {
        let wait = deadline.saturating_duration_since(Instant::now());
        assert!(!wait.is_zero(), "ticket callback timed out");
        result = state.ready.wait_timeout(result, wait).unwrap().0;
    }
    let (cause, status, callback) = result.take().unwrap();
    drop(result);
    assert_eq!((cause, status), (CAUSE_SUCCESS, 0));
    assert_ne!(callback, producer, "callback ran on the submitting thread");

    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);
    unsafe { kc_ticket_release(ticket) };

    let mut replacement = std::ptr::null_mut();
    assert_eq!(
        unsafe { kc_ticket_create(runtime, &ticket_config, &mut replacement) },
        0
    );
    let replacement_id = unsafe { kc_ticket_id_get(replacement) };
    assert_eq!(replacement_id.slot, id.slot);
    assert_ne!(replacement_id.generation, id.generation);
    assert!(unsafe { kc_ticket_complete_id(runtime, id, &completion) } < 0);
    assert_eq!(unsafe { kc_ticket_cancel_id(runtime, replacement_id) }, 1);
    assert_eq!(unsafe { kc_runtime_run_until_idle(runtime) }, 0);
    unsafe { kc_ticket_release(replacement) };

    unsafe { kc_runtime_request_stop(runtime) };
    assert_eq!(unsafe { kc_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { kc_runtime_destroy(runtime) }, 0);
}
