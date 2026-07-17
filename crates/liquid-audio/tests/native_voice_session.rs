use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use liquid_audio as _;

const ABI: u32 = 1;
const INVALID: i32 = -22;
const BUSY: i32 = -16;
const WOULD_BLOCK: i32 = -11;
const STALE: i32 = -116;
const CANCELLED: i32 = -125;
const HOST_SINK: i32 = -1002;
const CAPTURE: u32 = 1;
const PLAYBACK: u32 = 2;
const DOCK_ONLY: u64 = 1 << 63;
const EVENT_STATE: u32 = 1;
const EVENT_TEXT: u32 = 2;
const EVENT_TURN: u32 = 3;
const EVENT_STOPPED: u32 = 5;

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<Option<u64>> = const { Cell::new(None) };
}

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: forwards the allocator contract unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        // SAFETY: forwards the allocator contract unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout came from this allocator's system allocation path.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        // SAFETY: forwards the allocator contract unchanged to the system allocator.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[repr(C)]
struct Runtime {
    _private: [u8; 0],
}

#[repr(C)]
struct Model {
    _private: [u8; 0],
}

#[repr(C)]
struct Conversation {
    _private: [u8; 0],
}

#[repr(C)]
struct Session {
    _private: [u8; 0],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    coordination_workers: u32,
    kernel_lanes: u32,
    event_capacity: u32,
    session_capacity: u32,
    reserved0: u32,
    reserved1: u32,
    flags: u64,
    reserved: [u64; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SessionConfig {
    size: u32,
    abi_version: u32,
    session_id: u64,
    capture_slots: u32,
    playback_slots: u32,
    capture_frames_per_slot: u32,
    playback_frames_per_slot: u32,
    pcm_channels: u32,
    pcm_sample_rate: u32,
    command_capacity: u32,
    max_new_tokens: u32,
    reserved0: u32,
    flags: u64,
    reserved: [u64; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Event {
    size: u32,
    abi_version: u32,
    kind: u32,
    flags: u32,
    session_id: u64,
    epoch: u64,
    ticket: Ticket,
    payload: *const c_void,
    payload_bytes: u32,
    status: i32,
}

type EventFn = unsafe extern "C" fn(*mut c_void, *const Event) -> c_int;

#[derive(Clone, Copy)]
#[repr(C)]
struct Callbacks {
    size: u32,
    abi_version: u32,
    context: *mut c_void,
    on_event: Option<EventFn>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Lease {
    size: u32,
    abi_version: u32,
    lease_id: u64,
    stream_epoch: u64,
    buffer_generation: u64,
    ticket: Ticket,
    frames: u32,
    channels: u32,
    sample_rate: u32,
    format: u32,
    offset_bytes: u32,
    length_bytes: u32,
    flags: u32,
    reserved: u32,
}

impl Default for Lease {
    fn default() -> Self {
        // SAFETY: this C value is plain integer fields and a nested integer ticket.
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SessionSnapshot {
    size: u32,
    abi_version: u32,
    session_id: u64,
    epoch: u64,
    state: u32,
    terminal_status: i32,
    coordinator_parks: u64,
    coordinator_wakes: u64,
    notification_parks: u64,
    callbacks_entered: u64,
    capture_consumed: u64,
    capture_stale: u64,
    playback_published: u64,
    playback_consumed: u64,
    text_commands_accepted: u64,
    text_commands_consumed: u64,
    text_commands_stale: u64,
    live_capture_leases: u32,
    live_playback_leases: u32,
    reliable_event_depth: u32,
    reliable_event_capacity: u32,
    reserved: [u64; 4],
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: ABI,
            // SAFETY: remaining C snapshot fields are integer output cells.
            ..unsafe { std::mem::zeroed() }
        }
    }
}

unsafe extern "C" {
    fn lfm_runtime_create(config: *const RuntimeConfig, out: *mut *mut Runtime) -> c_int;
    fn lfm_runtime_start(runtime: *mut Runtime) -> c_int;
    fn lfm_runtime_request_stop(runtime: *mut Runtime);
    fn lfm_runtime_join(runtime: *mut Runtime) -> c_int;
    fn lfm_runtime_destroy(runtime: *mut Runtime) -> c_int;

    fn lfm_session_create(
        runtime: *mut Runtime,
        model: *mut Model,
        conversation: *mut Conversation,
        config: *const SessionConfig,
        callbacks: *const Callbacks,
        out: *mut *mut Session,
    ) -> c_int;
    fn lfm_session_start(session: *mut Session) -> c_int;
    fn lfm_session_submit_text(
        session: *mut Session,
        text: *const c_char,
        bytes: usize,
        out: *mut Ticket,
    ) -> c_int;
    fn lfm_session_wait_submit_text(
        session: *mut Session,
        text: *const c_char,
        bytes: usize,
        out: *mut Ticket,
    ) -> c_int;
    fn lfm_session_submit_mixed(
        session: *mut Session,
        text: *const c_char,
        bytes: usize,
        capture: *const Lease,
        out: *mut Ticket,
    ) -> c_int;
    fn lfm_session_wait_submit_mixed(
        session: *mut Session,
        text: *const c_char,
        bytes: usize,
        capture: *const Lease,
        out: *mut Ticket,
    ) -> c_int;
    fn lfm_session_interrupt(session: *mut Session, out_epoch: *mut u64) -> c_int;
    fn lfm_session_request_stop(session: *mut Session);
    fn lfm_session_join(session: *mut Session) -> c_int;
    fn lfm_session_snapshot(session: *const Session, out: *mut SessionSnapshot) -> c_int;
    fn lfm_session_destroy(session: *mut Session) -> c_int;

    fn lfm_audio_dock_reserve(
        session: *mut Session,
        direction: u32,
        frames: u32,
        sample_rate: u32,
        out: *mut Lease,
    ) -> c_int;
    fn lfm_audio_dock_wait_reserve(
        session: *mut Session,
        direction: u32,
        frames: u32,
        sample_rate: u32,
        out: *mut Lease,
    ) -> c_int;
    fn lfm_audio_dock_resolve_mut(
        session: *mut Session,
        lease: *const Lease,
        out: *mut *mut f32,
        capacity: *mut usize,
    ) -> c_int;
    fn lfm_audio_dock_resolve(
        session: *const Session,
        lease: *const Lease,
        out: *mut *const f32,
        count: *mut usize,
    ) -> c_int;
    fn lfm_audio_dock_publish(session: *mut Session, lease: *const Lease) -> c_int;
    fn lfm_audio_dock_wait_playback(session: *mut Session, out: *mut Lease) -> c_int;
    fn lfm_audio_dock_release(session: *mut Session, lease: *const Lease) -> c_int;
}

#[derive(Clone, Debug)]
struct Seen {
    kind: u32,
    epoch: u64,
    ticket: Ticket,
    status: i32,
    payload: Vec<u8>,
}

struct Sink {
    events: Mutex<Vec<Seen>>,
    edge: Condvar,
    fail: bool,
}

struct GateSink {
    events: Mutex<Vec<Seen>>,
    edge: Condvar,
    blocked: Mutex<bool>,
    release: Condvar,
}

unsafe extern "C" fn collect(context: *mut c_void, event: *const Event) -> c_int {
    // SAFETY: session creation borrows this Sink until join; native passes a callback-local event.
    let sink = unsafe { &*(context.cast::<Sink>()) };
    // SAFETY: native guarantees a non-null event for the callback duration.
    let event = unsafe { &*event };
    let payload = if event.payload_bytes == 0 {
        Vec::new()
    } else {
        // SAFETY: payload is borrowed and bounded by payload_bytes for this callback.
        unsafe {
            std::slice::from_raw_parts(event.payload.cast::<u8>(), event.payload_bytes as usize)
                .to_vec()
        }
    };
    sink.events.lock().unwrap().push(Seen {
        kind: event.kind,
        epoch: event.epoch,
        ticket: event.ticket,
        status: event.status,
        payload,
    });
    sink.edge.notify_all();
    i32::from(sink.fail && event.kind != EVENT_STOPPED)
}

unsafe extern "C" fn gated(context: *mut c_void, event: *const Event) -> c_int {
    // SAFETY: the test pins GateSink until native join proves callback completion.
    let sink = unsafe { &*(context.cast::<GateSink>()) };
    // SAFETY: native guarantees the event and payload for this callback duration.
    let event = unsafe { &*event };
    let payload = if event.payload_bytes == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(event.payload.cast::<u8>(), event.payload_bytes as usize)
                .to_vec()
        }
    };
    sink.events.lock().unwrap().push(Seen {
        kind: event.kind,
        epoch: event.epoch,
        ticket: event.ticket,
        status: event.status,
        payload,
    });
    sink.edge.notify_all();
    if event.kind == EVENT_STATE {
        let mut blocked = sink.blocked.lock().unwrap();
        while *blocked {
            blocked = sink.release.wait(blocked).unwrap();
        }
    }
    0
}

fn runtime_with(event_capacity: u32) -> *mut Runtime {
    let config = RuntimeConfig {
        size: std::mem::size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        coordination_workers: 1,
        kernel_lanes: 1,
        event_capacity,
        session_capacity: 2,
        reserved0: 0,
        reserved1: 0,
        flags: 0,
        reserved: [0; 4],
    };
    let mut runtime = std::ptr::null_mut();
    // SAFETY: config and output storage are valid for the synchronous calls.
    assert_eq!(unsafe { lfm_runtime_create(&config, &mut runtime) }, 0);
    assert!(!runtime.is_null());
    // SAFETY: runtime is a live unique native handle.
    assert_eq!(unsafe { lfm_runtime_start(runtime) }, 0);
    runtime
}

fn runtime() -> *mut Runtime {
    runtime_with(16)
}

fn dock_config() -> SessionConfig {
    SessionConfig {
        size: std::mem::size_of::<SessionConfig>() as u32,
        abi_version: ABI,
        session_id: 0,
        capture_slots: 1,
        playback_slots: 1,
        capture_frames_per_slot: 32,
        playback_frames_per_slot: 32,
        pcm_channels: 1,
        pcm_sample_rate: 48_000,
        command_capacity: 4,
        max_new_tokens: 8,
        reserved0: 0,
        flags: DOCK_ONLY,
        reserved: [0; 4],
    }
}

fn session(runtime: *mut Runtime, sink: &Sink) -> *mut Session {
    let config = dock_config();
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(sink).cast_mut().cast(),
        on_event: Some(collect),
    };
    let mut session = std::ptr::null_mut();
    // SAFETY: dock-only mode intentionally accepts null numerical owners.
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                &callbacks,
                &mut session,
            )
        },
        0
    );
    assert!(!session.is_null());
    session
}

fn gated_session(runtime: *mut Runtime, sink: &GateSink) -> *mut Session {
    let mut config = dock_config();
    config.command_capacity = 1;
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(sink).cast_mut().cast(),
        on_event: Some(gated),
    };
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                &callbacks,
                &mut session,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut events = sink.events.lock().unwrap();
    while !events.iter().any(|event| event.kind == EVENT_STATE) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero());
        let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(!timeout.timed_out());
    }
    drop(events);
    session
}

fn saturate_reliable_ring(session: *mut Session) -> [Ticket; 3] {
    let mut tickets = [Ticket::default(); 3];
    for (index, ticket) in tickets.iter_mut().enumerate() {
        let text = [b'a' + index as u8];
        assert_eq!(
            unsafe {
                lfm_session_wait_submit_text(session, text.as_ptr().cast(), text.len(), ticket)
            },
            0
        );
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        if snapshot.text_commands_consumed == 3 && snapshot.reliable_event_depth == 2 {
            return tickets;
        }
        assert!(
            Instant::now() < deadline,
            "coordinator did not reach the reliable-event space wait"
        );
        std::thread::yield_now();
    }
}

fn wait_event(sink: &Sink, predicate: impl Fn(&Seen) -> bool) -> Seen {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut events = sink.events.lock().unwrap();
    loop {
        if let Some(event) = events.iter().find(|event| predicate(event)) {
            return event.clone();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "event deadline expired: {events:#?}");
        let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(!timeout.timed_out(), "event deadline expired: {events:#?}");
    }
}

fn wait_gate_event(sink: &GateSink, predicate: impl Fn(&Seen) -> bool) -> Seen {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut events = sink.events.lock().unwrap();
    loop {
        if let Some(event) = events.iter().find(|event| predicate(event)) {
            return event.clone();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "event deadline expired: {events:#?}");
        let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(!timeout.timed_out(), "event deadline expired: {events:#?}");
    }
}

unsafe fn stop_all(runtime: *mut Runtime, session: *mut Session, expected: i32) {
    // SAFETY: caller owns both live handles and no further dock operation follows.
    unsafe { lfm_session_request_stop(session) };
    // SAFETY: stop closes admission and joins both native threads before returning.
    assert_eq!(unsafe { lfm_session_join(session) }, expected);
    // SAFETY: joined session has no live leases in these tests.
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    // SAFETY: session is gone, so runtime now has no child.
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

fn lease_soak(iterations: u64) {
    let runtime = runtime();
    let config = dock_config();
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                std::ptr::null(),
                &mut session,
            )
        },
        0
    );

    let start = Instant::now();
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let mut completed = 0u64;
    let mut expected_sequence = 1u64;
    let mut expected_generation = 1u64;
    let mut failure = 0i32;
    while completed < iterations {
        let mut lease = Lease::default();
        let reserve =
            unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 1, 48_000, &mut lease) };
        if reserve != 0 {
            failure = reserve;
            break;
        }
        if lease.ticket.sequence != expected_sequence
            || lease.buffer_generation != expected_generation
        {
            failure = STALE;
            break;
        }
        let release = unsafe { lfm_audio_dock_release(session, &lease) };
        if release != 0 {
            failure = release;
            break;
        }
        completed += 1;
        expected_sequence += 1;
        expected_generation += 1;
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    let elapsed = start.elapsed();

    // SAFETY: the tight loop leaves no live lease and the session never started threads.
    unsafe { stop_all(runtime, session, 0) };
    assert_eq!(failure, 0, "lease soak failed after {completed} iterations");
    assert_eq!(completed, iterations);
    assert_eq!(
        allocations, 0,
        "lease hot path allocated {allocations} times"
    );
    eprintln!(
        "native ticket/lease soak: {iterations} cycles in {:.3}s ({:.0} cycles/s)",
        elapsed.as_secs_f64(),
        iterations as f64 / elapsed.as_secs_f64()
    );
}

#[test]
fn joining_a_never_started_session_closes_every_admission_path() {
    let runtime = runtime();
    let config = dock_config();
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                std::ptr::null(),
                &mut session,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_session_join(session) }, 0);

    let mut lease = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 1, 48_000, &mut lease) },
        CANCELLED
    );
    let mut ticket = Ticket::default();
    assert_eq!(
        unsafe { lfm_session_submit_text(session, c"x".as_ptr(), 1, &mut ticket) },
        CANCELLED
    );
    let mut epoch = 0;
    assert_eq!(
        unsafe { lfm_session_interrupt(session, &mut epoch) },
        CANCELLED
    );

    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn pcm_leases_are_bound_to_one_session_even_when_slots_align() {
    let runtime = runtime();
    let config = dock_config();
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();
    for out in [&mut first, &mut second] {
        assert_eq!(
            unsafe {
                lfm_session_create(
                    runtime,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &config,
                    std::ptr::null(),
                    out,
                )
            },
            0
        );
    }

    for direction in [CAPTURE, PLAYBACK] {
        let mut owned = Lease::default();
        let mut foreign = Lease::default();
        assert_eq!(
            unsafe { lfm_audio_dock_reserve(first, direction, 8, 48_000, &mut owned) },
            0
        );
        assert_eq!(
            unsafe { lfm_audio_dock_reserve(second, direction, 8, 48_000, &mut foreign) },
            0
        );
        assert_ne!(owned.lease_id, foreign.lease_id);

        let mut pcm = std::ptr::null_mut();
        let mut capacity = 0;
        assert_eq!(
            unsafe { lfm_audio_dock_resolve_mut(second, &owned, &mut pcm, &mut capacity) },
            STALE
        );
        assert_eq!(unsafe { lfm_audio_dock_publish(second, &owned) }, STALE);
        assert_eq!(unsafe { lfm_audio_dock_release(second, &owned) }, STALE);
        assert_eq!(unsafe { lfm_audio_dock_release(first, &owned) }, 0);
        assert_eq!(unsafe { lfm_audio_dock_release(second, &foreign) }, 0);
    }

    for session in [first, second] {
        unsafe { lfm_session_request_stop(session) };
        assert_eq!(unsafe { lfm_session_join(session) }, 0);
        assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    }
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn session_geometry_rejects_lease_byte_length_overflow_before_allocating() {
    let runtime = runtime();
    let mut config = dock_config();
    config.capture_frames_per_slot = u32::MAX;
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                std::ptr::null(),
                &mut session,
            )
        },
        INVALID
    );
    assert!(session.is_null());
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn native_session_parks_and_generation_checks_every_pcm_lease() {
    assert_eq!(std::mem::size_of::<Ticket>(), 24);
    assert_eq!(std::mem::size_of::<RuntimeConfig>(), 72);
    assert_eq!(std::mem::size_of::<SessionConfig>(), 96);
    assert_eq!(std::mem::size_of::<Event>(), 72);
    assert_eq!(std::mem::size_of::<Lease>(), 88);
    assert_eq!(std::mem::size_of::<SessionSnapshot>(), 168);

    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);

    // Parent/child and joined-destroy rules are explicit, never implicit joins.
    // SAFETY: both handles are live.
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, BUSY);
    assert_eq!(unsafe { lfm_session_destroy(session) }, BUSY);
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let running = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.payload == b"running"
    });
    assert_eq!(running.epoch, 1);

    let mut stale = Lease::default();
    // SAFETY: capture reservation and pointer resolution are private dock operations.
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 16_000, &mut stale) },
        -22
    );
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut stale) },
        0
    );
    assert_ne!(stale.ticket.sequence, 0);
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &stale, &mut pcm, &mut capacity) },
        0
    );
    assert!(capacity >= 8);
    // SAFETY: the generation-checked reservation owns at least eight f32 cells.
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(0.25) };

    let mut epoch = 0;
    // SAFETY: interrupt is callback-safe and only advances the publication epoch.
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(epoch, 2);
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &stale) }, 0);
    let stale_turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket.sequence == stale.ticket.sequence
    });
    assert_eq!(stale_turn.status, STALE);
    assert_eq!(stale_turn.epoch, 1);
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &stale, &mut pcm, &mut capacity) },
        STALE
    );

    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
        0
    );
    assert_ne!(capture.lease_id, stale.lease_id);
    assert_ne!(capture.buffer_generation, stale.buffer_generation);
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(-0.5) };
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &capture) }, 0);
    let capture_turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket.sequence == capture.ticket.sequence
    });
    assert_eq!(capture_turn.status, 0);
    assert_eq!(capture_turn.epoch, 2);

    let mut text_ticket = Ticket::default();
    let text = b"simultaneous typed control";
    assert_eq!(
        unsafe {
            lfm_session_submit_text(
                session,
                text.as_ptr().cast::<c_char>(),
                text.len(),
                &mut text_ticket,
            )
        },
        0
    );
    let text_turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket.sequence == text_ticket.sequence
    });
    assert_eq!(text_turn.status, 0);

    let mut playback = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, PLAYBACK, 8, 24_000, &mut playback) },
        0
    );
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &playback, &mut pcm, &mut capacity) },
        0
    );
    unsafe {
        std::slice::from_raw_parts_mut(pcm, 8)
            .copy_from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    };
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &playback) }, 0);
    let mut taken = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_playback(session, &mut taken) },
        0
    );
    assert_eq!(taken.lease_id, playback.lease_id);
    let mut read = std::ptr::null();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve(session, &taken, &mut read, &mut count) },
        0
    );
    assert_eq!(count, 8);
    // SAFETY: resolve retains the consuming lease until release below.
    assert_eq!(unsafe { std::slice::from_raw_parts(read, count) }[7], 0.7);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &taken) }, 0);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &taken) }, STALE);

    let park_deadline = Instant::now() + Duration::from_secs(2);
    let snapshot = loop {
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        if snapshot.coordinator_parks > 0 && snapshot.notification_parks > 0 {
            break snapshot;
        }
        assert!(
            Instant::now() < park_deadline,
            "native continuations never parked"
        );
        std::thread::yield_now();
    };
    assert_eq!(snapshot.epoch, 2);
    assert_eq!(snapshot.capture_stale, 1);
    assert_eq!(snapshot.capture_consumed, 1);
    assert_eq!(snapshot.text_commands_accepted, 1);
    assert_eq!(snapshot.text_commands_consumed, 1);
    assert_eq!(snapshot.live_capture_leases, 0);
    assert_eq!(snapshot.live_playback_leases, 0);

    // SAFETY: callback context remains pinned until join completes.
    unsafe { stop_all(runtime, session, 0) };
    let stopped = wait_event(&sink, |event| event.kind == EVENT_STOPPED);
    assert_eq!(stopped.status, 0);
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.kind != EVENT_TEXT));
}

#[test]
fn mixed_command_retains_pcm_and_correlates_one_terminal_ticket() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    wait_event(&sink, |event| event.kind == EVENT_STATE);

    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
        0
    );
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(0.125) };

    let text = b"typed while speaking";
    let mut ticket = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_session_wait_submit_mixed(
                session,
                text.as_ptr().cast(),
                text.len(),
                &capture,
                &mut ticket,
            )
        },
        0
    );
    assert_eq!(ticket, capture.ticket);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &capture) }, BUSY);

    let terminal = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == ticket
    });
    assert_eq!(terminal.epoch, capture.stream_epoch);
    assert_eq!(terminal.status, 0);
    let events = sink.events.lock().unwrap();
    let correlated: Vec<_> = events
        .iter()
        .filter(|event| event.ticket == ticket)
        .collect();
    assert_eq!(correlated.len(), 1);
    assert_eq!(correlated[0].kind, EVENT_TURN);
    drop(events);

    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.text_commands_accepted, 1);
    assert_eq!(snapshot.text_commands_consumed, 1);
    assert_eq!(snapshot.capture_consumed, 1);
    assert_eq!(snapshot.live_capture_leases, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn interrupt_stales_an_admitted_mixed_command_without_stale_output() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        blocked: Mutex::new(true),
        release: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let session = gated_session(runtime, &sink);
    let _fillers = saturate_reliable_ring(session);

    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
        0
    );
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(-0.25) };
    let text = b"cancel this mixed action";
    let mut ticket = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_session_submit_mixed(
                session,
                text.as_ptr().cast(),
                text.len(),
                &capture,
                &mut ticket,
            )
        },
        0
    );
    assert_eq!(ticket, capture.ticket);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &capture) }, BUSY);

    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(epoch, capture.stream_epoch + 1);
    *sink.blocked.lock().unwrap() = false;
    sink.release.notify_all();

    let terminal = wait_gate_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == ticket
    });
    assert_eq!(terminal.epoch, capture.stream_epoch);
    assert_eq!(terminal.status, STALE);
    let events = sink.events.lock().unwrap();
    let correlated: Vec<_> = events
        .iter()
        .filter(|event| event.ticket == ticket)
        .collect();
    assert_eq!(correlated.len(), 1);
    assert_eq!(correlated[0].kind, EVENT_TURN);
    drop(events);
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        STALE
    );

    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.capture_consumed, 0);
    assert_eq!(snapshot.capture_stale, 1);
    assert_eq!(snapshot.text_commands_stale, 1);
    assert_eq!(snapshot.live_capture_leases, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn failed_mixed_admission_never_transfers_or_partially_consumes_pcm() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        blocked: Mutex::new(true),
        release: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let session = gated_session(runtime, &sink);
    let _fillers = saturate_reliable_ring(session);

    let fourth = b"occupy command ring";
    let mut fourth_ticket = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_session_wait_submit_text(
                session,
                fourth.as_ptr().cast(),
                fourth.len(),
                &mut fourth_ticket,
            )
        },
        0
    );

    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
        0
    );
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(0.75) };
    let sentinel = Ticket {
        runtime_epoch: u64::MAX,
        sequence: u64::MAX,
        generation: u32::MAX,
        kind: u32::MAX,
    };
    let mut ticket = sentinel;
    let text = b"must remain caller owned";
    assert_eq!(
        unsafe {
            lfm_session_submit_mixed(
                session,
                text.as_ptr().cast(),
                text.len(),
                &capture,
                &mut ticket,
            )
        },
        WOULD_BLOCK
    );
    assert_eq!(ticket, sentinel);
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(unsafe { std::slice::from_raw_parts(pcm, 8) }[7], 0.75);

    let address = session as usize;
    let retained = capture;
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let waiter = std::thread::spawn(move || {
        let text = b"park without polling";
        let mut ticket = Ticket::default();
        let status = unsafe {
            lfm_session_wait_submit_mixed(
                address as *mut Session,
                text.as_ptr().cast(),
                text.len(),
                &retained,
                &mut ticket,
            )
        };
        tx.send((status, ticket)).unwrap();
    });
    assert!(matches!(
        rx.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let (status, waited_ticket) = rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(status, STALE);
    assert_eq!(waited_ticket, Ticket::default());
    waiter.join().unwrap();

    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(unsafe { lfm_audio_dock_release(session, &capture) }, 0);
    *sink.blocked.lock().unwrap() = false;
    sink.release.notify_all();
    let _ = wait_gate_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == fourth_ticket
    });
    let events = sink.events.lock().unwrap();
    assert!(!events.iter().any(|event| event.ticket == capture.ticket));
    drop(events);

    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.text_commands_accepted, 4);
    assert_eq!(snapshot.capture_consumed, 0);
    assert_eq!(snapshot.capture_stale, 0);
    assert_eq!(snapshot.live_capture_leases, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn published_lease_cannot_be_reclaimed_by_its_producer() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);
    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 4, 48_000, &mut capture) },
        0
    );
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &capture) }, 0);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &capture) }, BUSY);

    let mut playback = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, PLAYBACK, 4, 24_000, &mut playback) },
        0
    );
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &playback) }, 0);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &playback) }, BUSY);
    let mut taken = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_playback(session, &mut taken) },
        0
    );
    assert_eq!(unsafe { lfm_audio_dock_release(session, &taken) }, 0);

    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket.sequence == capture.ticket.sequence
    });
    assert_eq!(turn.status, 0);
    // SAFETY: queued capture and playback leases both reached their sole consumers.
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn concurrent_capture_publishers_preserve_every_ring_cell() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let mut config = dock_config();
    config.capture_slots = 8;
    config.capture_frames_per_slot = 8;
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(&sink).cast_mut().cast(),
        on_event: Some(collect),
    };
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                &callbacks,
                &mut session,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    wait_event(&sink, |event| event.kind == EVENT_STATE);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let address = session as usize;
    let threads: Vec<_> = (0..8)
        .map(|index| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let session = address as *mut Session;
                let mut lease = Lease::default();
                assert_eq!(
                    unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut lease) },
                    0
                );
                let mut samples = std::ptr::null_mut();
                let mut capacity = 0;
                assert_eq!(
                    unsafe {
                        lfm_audio_dock_resolve_mut(session, &lease, &mut samples, &mut capacity)
                    },
                    0
                );
                assert!(capacity >= 8);
                unsafe {
                    std::slice::from_raw_parts_mut(samples, 8).fill(index as f32);
                }
                barrier.wait();
                assert_eq!(unsafe { lfm_audio_dock_publish(session, &lease) }, 0);
                lease.ticket
            })
        })
        .collect();
    let tickets: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    for ticket in &tickets {
        let terminal = wait_event(&sink, |event| {
            event.kind == EVENT_TURN && event.ticket.sequence == ticket.sequence
        });
        assert_eq!(terminal.status, 0);
    }
    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.capture_consumed, 8);
    assert_eq!(snapshot.live_capture_leases, 0);
    // SAFETY: every publisher has returned and every accepted ticket is terminal.
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn full_capture_dock_parks_until_release_and_wakes_on_interrupt_or_stop() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);
    assert_eq!(unsafe { lfm_session_start(session) }, 0);

    let mut held = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut held) },
        0
    );
    let mut full = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut full) },
        WOULD_BLOCK
    );
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let address = session as usize;
    let release_waiter = std::thread::spawn(move || {
        let mut lease = Lease::default();
        let status = unsafe {
            lfm_audio_dock_wait_reserve(address as *mut Session, CAPTURE, 8, 48_000, &mut lease)
        };
        tx.send((status, lease)).unwrap();
    });
    assert!(matches!(
        rx.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(unsafe { lfm_audio_dock_release(session, &held) }, 0);
    let (status, released) = rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(status, 0);
    assert_ne!(released.lease_id, held.lease_id);
    assert_ne!(released.buffer_generation, held.buffer_generation);
    release_waiter.join().unwrap();
    assert_eq!(unsafe { lfm_audio_dock_release(session, &released) }, 0);

    let mut interrupted = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 8, 48_000, &mut interrupted,) },
        0
    );
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let address = session as usize;
    let interrupt_waiter = std::thread::spawn(move || {
        let mut lease = Lease::default();
        let status = unsafe {
            lfm_audio_dock_wait_reserve(address as *mut Session, CAPTURE, 8, 48_000, &mut lease)
        };
        tx.send(status).unwrap();
    });
    assert!(matches!(
        rx.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap(), STALE);
    interrupt_waiter.join().unwrap();
    assert_eq!(unsafe { lfm_audio_dock_release(session, &interrupted) }, 0);

    let mut stopped = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_wait_reserve(session, CAPTURE, 8, 48_000, &mut stopped) },
        0
    );
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let address = session as usize;
    let stop_waiter = std::thread::spawn(move || {
        let mut lease = Lease::default();
        let status = unsafe {
            lfm_audio_dock_wait_reserve(address as *mut Session, CAPTURE, 8, 48_000, &mut lease)
        };
        tx.send(status).unwrap();
    });
    assert!(matches!(
        rx.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap(), CANCELLED);
    stop_waiter.join().unwrap();
    assert_eq!(unsafe { lfm_audio_dock_release(session, &stopped) }, 0);

    // SAFETY: stop was already requested and every admission waiter and lease is gone.
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn full_text_ring_parks_concurrent_submitter_until_consumer_pop() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        blocked: Mutex::new(true),
        release: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let mut config = dock_config();
    config.command_capacity = 1;
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(&sink).cast_mut().cast(),
        on_event: Some(gated),
    };
    let mut session = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                &callbacks,
                &mut session,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut events = sink.events.lock().unwrap();
        while !events.iter().any(|event| event.kind == EVENT_STATE) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero());
            let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
            events = next;
            assert!(!timeout.timed_out());
        }
    }

    let commands: [&[u8]; 3] = [b"first", b"second", b"third"];
    let mut tickets = [Ticket::default(); 5];
    for (command, ticket) in commands.into_iter().zip(tickets.iter_mut()) {
        assert_eq!(
            unsafe {
                lfm_session_wait_submit_text(
                    session,
                    command.as_ptr().cast(),
                    command.len(),
                    ticket,
                )
            },
            0
        );
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        if snapshot.text_commands_consumed == 3 && snapshot.reliable_event_depth == 2 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "coordinator did not reach the full-ring gate"
        );
        std::thread::yield_now();
    }

    let fourth = b"fourth";
    assert_eq!(
        unsafe {
            lfm_session_wait_submit_text(
                session,
                fourth.as_ptr().cast(),
                fourth.len(),
                &mut tickets[3],
            )
        },
        0
    );
    let mut probe = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_session_submit_text(
                session,
                b"probe".as_ptr().cast(),
                b"probe".len(),
                &mut probe,
            )
        },
        WOULD_BLOCK
    );

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let address = session as usize;
    let waiter = std::thread::spawn(move || {
        let command = b"fifth";
        let mut ticket = Ticket::default();
        let status = unsafe {
            lfm_session_wait_submit_text(
                address as *mut Session,
                command.as_ptr().cast(),
                command.len(),
                &mut ticket,
            )
        };
        tx.send((status, ticket)).unwrap();
    });
    assert!(matches!(
        rx.recv_timeout(Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));

    *sink.blocked.lock().unwrap() = false;
    sink.release.notify_all();
    let (status, ticket) = rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(status, 0);
    tickets[4] = ticket;
    waiter.join().unwrap();

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut events = sink.events.lock().unwrap();
    loop {
        let delivered = tickets
            .iter()
            .filter(|ticket| {
                events.iter().any(|event| {
                    event.kind == EVENT_TURN && event.ticket.sequence == ticket.sequence
                })
            })
            .count();
        if delivered == tickets.len() {
            break;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "terminal callbacks missing: {events:#?}"
        );
        let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(
            !timeout.timed_out(),
            "terminal callbacks missing: {events:#?}"
        );
    }
    drop(events);
    // SAFETY: all five accepted commands reached their terminal callback.
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn reliable_event_saturation_parks_for_space_without_failing_the_session() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        blocked: Mutex::new(true),
        release: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let config = dock_config();
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(&sink).cast_mut().cast(),
        on_event: Some(gated),
    };
    let mut session = std::ptr::null_mut();
    // SAFETY: dock-only creation accepts null model owners and borrows the pinned callback.
    assert_eq!(
        unsafe {
            lfm_session_create(
                runtime,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &config,
                &callbacks,
                &mut session,
            )
        },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut events = sink.events.lock().unwrap();
        while !events.iter().any(|event| event.kind == EVENT_STATE) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero());
            let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
            events = next;
            assert!(!timeout.timed_out());
        }
    }

    let mut tickets = [Ticket::default(); 4];
    for (index, ticket) in tickets.iter_mut().enumerate() {
        let text = format!("queued-{index}");
        assert_eq!(
            unsafe {
                lfm_session_submit_text(session, text.as_ptr().cast::<c_char>(), text.len(), ticket)
            },
            0
        );
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        assert_eq!(
            snapshot.terminal_status, 0,
            "full reliable ring became a fault"
        );
        if snapshot.reliable_event_depth == 2 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "reliable ring never reached capacity"
        );
        std::thread::yield_now();
    }
    *sink.blocked.lock().unwrap() = false;
    sink.release.notify_all();

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut events = sink.events.lock().unwrap();
    loop {
        let delivered = tickets
            .iter()
            .filter(|ticket| {
                events.iter().any(|event| {
                    event.kind == EVENT_TURN && event.ticket.sequence == ticket.sequence
                })
            })
            .count();
        if delivered == tickets.len() {
            break;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "terminal callbacks missing: {events:#?}"
        );
        let (next, timeout) = sink.edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(
            !timeout.timed_out(),
            "terminal callbacks missing: {events:#?}"
        );
    }
    drop(events);
    // SAFETY: all accepted commands reached one reliable terminal callback.
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn reliable_callback_failure_stops_and_joins_exactly_once() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: true,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);
    // SAFETY: handles and callback context remain live through join.
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let stopped = wait_event(&sink, |event| event.kind == EVENT_STOPPED);
    assert_eq!(stopped.status, HOST_SINK);
    assert_eq!(unsafe { lfm_session_join(session) }, HOST_SINK);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
    let events = sink.events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == EVENT_STOPPED)
            .count(),
        1
    );
}

#[test]
fn ticket_lease_hot_path_is_allocation_free_for_100k_cycles() {
    lease_soak(100_000);
}

#[test]
#[ignore = "explicit million-cycle ticket/lease soak gate"]
fn ticket_lease_hot_path_million_cycle_soak() {
    lease_soak(1_000_000);
}
