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

#[repr(C)]
struct CaptureProducer {
    _private: [u8; 0],
}

#[repr(C)]
struct PlaybackConsumer {
    _private: [u8; 0],
}

#[repr(C)]
struct SessionControl {
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
    reserved_coordinator: [u64; 2],
    reserved_delivery: u64,
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
    fn lfm_session_submit_mixed(
        session: *mut Session,
        text: *const c_char,
        bytes: usize,
        capture: *const Lease,
        out: *mut Ticket,
    ) -> c_int;
    fn lfm_session_interrupt(session: *mut Session, out_epoch: *mut u64) -> c_int;
    fn lfm_session_host_capacity(session: *mut Session) -> c_int;
    fn lfm_session_request_stop(session: *mut Session);
    fn lfm_session_join(session: *mut Session) -> c_int;
    fn lfm_session_snapshot(session: *const Session, out: *mut SessionSnapshot) -> c_int;
    fn lfm_session_destroy(session: *mut Session) -> c_int;

    fn lfm_capture_producer_create(session: *mut Session, out: *mut *mut CaptureProducer) -> c_int;
    fn lfm_capture_producer_reserve(
        producer: *mut CaptureProducer,
        frames: u32,
        sample_rate: u32,
        out: *mut Lease,
    ) -> c_int;
    fn lfm_capture_producer_resolve_mut(
        producer: *mut CaptureProducer,
        lease: *const Lease,
        out: *mut *mut f32,
        capacity: *mut usize,
    ) -> c_int;
    fn lfm_capture_producer_finalize(
        producer: *mut CaptureProducer,
        lease: *mut Lease,
        offset_frames: u32,
        used_frames: u32,
    ) -> c_int;
    fn lfm_capture_producer_publish(producer: *mut CaptureProducer, lease: *const Lease) -> c_int;
    fn lfm_capture_producer_release(producer: *mut CaptureProducer, lease: *const Lease) -> c_int;
    fn lfm_capture_producer_destroy(producer: *mut CaptureProducer) -> c_int;
    fn lfm_playback_consumer_create(
        session: *mut Session,
        out: *mut *mut PlaybackConsumer,
    ) -> c_int;
    fn lfm_playback_consumer_claim(
        consumer: *mut PlaybackConsumer,
        ticket: *const Ticket,
        stream_epoch: u64,
        lease_id: u64,
        buffer_generation: u64,
        out: *mut Lease,
    ) -> c_int;
    fn lfm_playback_consumer_resolve(
        consumer: *const PlaybackConsumer,
        lease: *const Lease,
        out: *mut *const f32,
        count: *mut usize,
    ) -> c_int;
    fn lfm_playback_consumer_release(consumer: *mut PlaybackConsumer, lease: *const Lease)
    -> c_int;
    fn lfm_playback_consumer_destroy(consumer: *mut PlaybackConsumer) -> c_int;
    fn lfm_session_control_create(session: *mut Session, out: *mut *mut SessionControl) -> c_int;
    fn lfm_session_control_interrupt(control: *mut SessionControl, out_epoch: *mut u64) -> c_int;
    fn lfm_session_control_destroy(control: *mut SessionControl) -> c_int;

    fn lfm_audio_dock_reserve(
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
    fn lfm_audio_dock_finalize_capture(
        session: *mut Session,
        lease: *mut Lease,
        offset_frames: u32,
        used_frames: u32,
    ) -> c_int;
    fn lfm_audio_dock_resolve(
        session: *const Session,
        lease: *const Lease,
        out: *mut *const f32,
        count: *mut usize,
    ) -> c_int;
    fn lfm_audio_dock_publish(session: *mut Session, lease: *const Lease) -> c_int;
    fn lfm_audio_dock_try_playback(session: *mut Session, out: *mut Lease) -> c_int;
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
    event_edge: Condvar,
    blocked: Mutex<bool>,
    attempts: Mutex<u64>,
    attempt_edge: Condvar,
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
    if *sink.blocked.lock().unwrap() {
        *sink.attempts.lock().unwrap() += 1;
        sink.attempt_edge.notify_all();
        return WOULD_BLOCK;
    }
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
    sink.event_edge.notify_all();
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
    let mut attempts = sink.attempts.lock().unwrap();
    while *attempts == 0 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero());
        let (next, timeout) = sink.attempt_edge.wait_timeout(attempts, remaining).unwrap();
        attempts = next;
        assert!(!timeout.timed_out());
    }
    drop(attempts);
    session
}

fn submit_text_eventually(session: *mut Session, text: &[u8], ticket: &mut Ticket) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let status =
            unsafe { lfm_session_submit_text(session, text.as_ptr().cast(), text.len(), ticket) };
        if status == 0 {
            return;
        }
        assert_eq!(status, WOULD_BLOCK);
        assert!(Instant::now() < deadline, "command capacity never reopened");
        std::thread::yield_now();
    }
}

fn saturate_reliable_ring(session: *mut Session) -> [Ticket; 3] {
    let mut tickets = [Ticket::default(); 3];
    for (index, ticket) in tickets.iter_mut().enumerate() {
        let text = [b'a' + index as u8];
        submit_text_eventually(session, &text, ticket);
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
            "coordinator did not retain the full-ring result continuation"
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
        let (next, timeout) = sink.event_edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(!timeout.timed_out(), "event deadline expired: {events:#?}");
    }
}

fn wait_gate_attempt(sink: &GateSink) {
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut attempts = sink.attempts.lock().unwrap();
    while *attempts == 0 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "callback attempt deadline expired");
        let (next, timeout) = sink.attempt_edge.wait_timeout(attempts, remaining).unwrap();
        attempts = next;
        assert!(!timeout.timed_out(), "callback attempt deadline expired");
    }
}

fn open_gate(session: *mut Session, sink: &GateSink) {
    *sink.blocked.lock().unwrap() = false;
    assert_eq!(unsafe { lfm_session_host_capacity(session) }, 0);
}

unsafe fn stop_all(runtime: *mut Runtime, session: *mut Session, expected: i32) {
    // SAFETY: caller owns both live handles and no further dock operation follows.
    unsafe { lfm_session_request_stop(session) };
    // SAFETY: stop closes admission and joins both retained services before returning.
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
        let reserve = unsafe { lfm_audio_dock_reserve(session, CAPTURE, 1, 48_000, &mut lease) };
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
fn runtime_rejects_lane_counts_the_engine_cannot_construct() {
    let config = RuntimeConfig {
        size: std::mem::size_of::<RuntimeConfig>() as u32,
        abi_version: ABI,
        coordination_workers: 1,
        kernel_lanes: 17,
        event_capacity: 2,
        session_capacity: 1,
        reserved0: 0,
        reserved1: 0,
        flags: 0,
        reserved: [0; 4],
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_runtime_create(&config, &mut runtime) },
        INVALID
    );
    assert!(runtime.is_null());
}

#[test]
fn dock_only_session_requires_explicit_playback_geometry() {
    let runtime = runtime();
    let mut config = dock_config();
    config.playback_frames_per_slot = 0;
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
    assert_eq!(unsafe { lfm_session_start(session) }, CANCELLED);

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
fn stopping_a_created_session_permanently_prevents_start() {
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
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_session_start(session) }, CANCELLED);
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn concurrent_session_start_and_join_have_one_linearization_order() {
    for _ in 0..64 {
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

        let edge = std::sync::Arc::new(std::sync::Barrier::new(3));
        let start_edge = edge.clone();
        let start_address = session as usize;
        let starter = std::thread::spawn(move || {
            start_edge.wait();
            unsafe { lfm_session_start(start_address as *mut Session) }
        });
        let join_edge = edge.clone();
        let join_address = session as usize;
        let joiner = std::thread::spawn(move || {
            join_edge.wait();
            unsafe { lfm_session_join(join_address as *mut Session) }
        });
        edge.wait();
        let started = starter.join().unwrap();
        let joined = joiner.join().unwrap();

        if started == 0 {
            assert_eq!(joined, BUSY);
            unsafe { lfm_session_request_stop(session) };
            assert_eq!(unsafe { lfm_session_join(session) }, 0);
        } else {
            assert_eq!(started, CANCELLED);
            assert_eq!(joined, 0);
            assert_eq!(unsafe { lfm_session_join(session) }, 0);
        }
        assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
        unsafe { lfm_runtime_request_stop(runtime) };
        assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
        assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
    }
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
fn runtime_stop_and_session_start_have_one_linearization_order() {
    for _ in 0..32 {
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

        let edge = std::sync::Arc::new(std::sync::Barrier::new(3));
        let start_edge = edge.clone();
        let start_session = session as usize;
        let starter = std::thread::spawn(move || {
            start_edge.wait();
            unsafe { lfm_session_start(start_session as *mut Session) }
        });
        let stop_edge = edge.clone();
        let stop_runtime = runtime as usize;
        let stopper = std::thread::spawn(move || {
            stop_edge.wait();
            unsafe { lfm_runtime_request_stop(stop_runtime as *mut Runtime) };
        });
        edge.wait();
        let started = starter.join().unwrap();
        stopper.join().unwrap();
        assert!(
            started == 0 || started == BUSY,
            "unexpected start status {started}"
        );

        assert_eq!(unsafe { lfm_session_join(session) }, 0);
        assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
        assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
        assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
    }
}

#[test]
fn native_session_generation_checks_every_pcm_lease() {
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
    assert_eq!(capacity, 8);
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(-0.5) };
    let base = pcm;
    assert_eq!(
        unsafe { lfm_audio_dock_finalize_capture(session, &mut capture, 2, 3) },
        0
    );
    assert_eq!(capture.frames, 3);
    assert_eq!(capture.offset_bytes, 2 * std::mem::size_of::<f32>() as u32);
    assert_eq!(capture.length_bytes, 3 * std::mem::size_of::<f32>() as u32);
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(capacity, 3);
    assert_eq!(pcm, unsafe { base.add(2) });
    assert_eq!(
        unsafe { lfm_audio_dock_finalize_capture(session, &mut capture, 0, 4) },
        -22
    );
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
    let mut empty = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_try_playback(session, &mut empty) },
        WOULD_BLOCK
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
        unsafe { lfm_audio_dock_try_playback(session, &mut taken) },
        0
    );
    assert_eq!(taken.lease_id, playback.lease_id);
    let mut wrong_ticket = taken;
    wrong_ticket.ticket.sequence = wrong_ticket.ticket.sequence.wrapping_add(1);
    let mut rejected = std::ptr::null();
    let mut rejected_count = 0;
    assert_eq!(
        unsafe {
            lfm_audio_dock_resolve(session, &wrong_ticket, &mut rejected, &mut rejected_count)
        },
        STALE
    );
    let mut wrong_epoch = taken;
    wrong_epoch.stream_epoch = wrong_epoch.stream_epoch.wrapping_add(1);
    assert_eq!(
        unsafe {
            lfm_audio_dock_resolve(session, &wrong_epoch, &mut rejected, &mut rejected_count)
        },
        STALE
    );
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

    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.reserved_delivery, 0);
    assert_eq!(snapshot.epoch, 2);
    assert_eq!(snapshot.capture_stale, 1);
    assert_eq!(snapshot.capture_consumed, 1);
    assert_eq!(snapshot.text_commands_accepted, 1);
    assert_eq!(snapshot.text_commands_consumed, 1);
    assert_eq!(snapshot.live_capture_leases, 0);
    assert_eq!(snapshot.live_playback_leases, 0);
    assert_eq!(snapshot.reserved_coordinator, [0; 2]);

    // SAFETY: callback context remains pinned until join completes.
    unsafe { stop_all(runtime, session, 0) };
    let stopped = wait_event(&sink, |event| event.kind == EVENT_STOPPED);
    assert_eq!(stopped.status, 0);
    assert!(
        sink.events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.kind != EVENT_TEXT)
    );
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
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
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
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let session = gated_session(runtime, &sink);
    let _fillers = saturate_reliable_ring(session);

    let mut capture = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
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
    open_gate(session, &sink);

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
fn interrupt_epoch_is_applied_before_a_fresh_command_in_the_same_drain() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let session = gated_session(runtime, &sink);
    let _fillers = saturate_reliable_ring(session);

    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(epoch, 2);
    let text = b"fresh epoch command";
    let mut ticket = Ticket::default();
    assert_eq!(
        unsafe { lfm_session_submit_text(session, text.as_ptr().cast(), text.len(), &mut ticket,) },
        0
    );

    open_gate(session, &sink);
    let interrupted = wait_gate_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    let terminal = wait_gate_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == ticket
    });
    assert_eq!(interrupted.epoch, epoch);
    assert_eq!(terminal.epoch, epoch);
    assert_eq!(terminal.status, 0);

    let events = sink.events.lock().unwrap();
    let interrupted_index = events
        .iter()
        .position(|event| {
            event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
        })
        .unwrap();
    let terminal_index = events
        .iter()
        .position(|event| event.kind == EVENT_TURN && event.ticket == ticket)
        .unwrap();
    assert!(
        interrupted_index < terminal_index,
        "fresh command reached the coordinator before its epoch was applied: {events:#?}"
    );
    drop(events);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn failed_mixed_admission_never_transfers_or_partially_consumes_pcm() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let session = gated_session(runtime, &sink);
    let _fillers = saturate_reliable_ring(session);

    let fourth = b"occupy command ring";
    let mut fourth_ticket = Ticket::default();
    assert_eq!(
        unsafe {
            lfm_session_submit_text(
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
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut capture) },
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

    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &capture, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(unsafe { lfm_audio_dock_release(session, &capture) }, 0);
    open_gate(session, &sink);
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
        unsafe { lfm_audio_dock_try_playback(session, &mut taken) },
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
fn stop_retires_ingress_only_after_every_admitted_publication_edge() {
    for _ in 0..16 {
        let runtime = runtime_with(32);
        let mut config = dock_config();
        config.capture_slots = 8;
        config.capture_frames_per_slot = 8;
        config.command_capacity = 8;
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
        assert_eq!(unsafe { lfm_session_start(session) }, 0);

        let mut leases = Vec::with_capacity(8);
        for _ in 0..8 {
            let mut lease = Lease::default();
            assert_eq!(
                unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut lease) },
                0
            );
            leases.push(lease);
        }

        let edge = std::sync::Arc::new(std::sync::Barrier::new(18));
        let publishers: Vec<_> = leases
            .into_iter()
            .map(|lease| {
                let edge = edge.clone();
                let address = session as usize;
                std::thread::spawn(move || {
                    edge.wait();
                    let status = unsafe { lfm_audio_dock_publish(address as *mut Session, &lease) };
                    (lease, status)
                })
            })
            .collect();
        let commands: Vec<_> = (0..8)
            .map(|_| {
                let edge = edge.clone();
                let address = session as usize;
                std::thread::spawn(move || {
                    edge.wait();
                    let mut ticket = Ticket::default();
                    let status = unsafe {
                        lfm_session_submit_text(
                            address as *mut Session,
                            c"edge".as_ptr(),
                            4,
                            &mut ticket,
                        )
                    };
                    (ticket, status)
                })
            })
            .collect();
        let stop_edge = edge.clone();
        let stop_address = session as usize;
        let stopper = std::thread::spawn(move || {
            stop_edge.wait();
            unsafe { lfm_session_request_stop(stop_address as *mut Session) };
        });
        edge.wait();
        stopper.join().unwrap();

        for publisher in publishers {
            let (lease, status) = publisher.join().unwrap();
            assert!(
                status == 0 || status == CANCELLED,
                "unexpected publish status {status}"
            );
            if status == CANCELLED {
                assert_eq!(unsafe { lfm_audio_dock_release(session, &lease) }, 0);
            }
        }
        for command in commands {
            let (_, status) = command.join().unwrap();
            assert!(
                status == 0 || status == CANCELLED || status == WOULD_BLOCK,
                "unexpected command status {status}"
            );
        }

        assert_eq!(unsafe { lfm_session_join(session) }, 0);
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        assert_eq!(snapshot.live_capture_leases, 0);
        assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
        unsafe { lfm_runtime_request_stop(runtime) };
        assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
        assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
    }
}

#[test]
fn full_capture_dock_never_creates_a_capacity_waiter() {
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

    assert_eq!(unsafe { lfm_audio_dock_release(session, &held) }, 0);
    let mut next = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, CAPTURE, 8, 48_000, &mut next) },
        0
    );
    assert_ne!(next.lease_id, held.lease_id);
    assert_ne!(next.buffer_generation, held.buffer_generation);
    assert_eq!(unsafe { lfm_audio_dock_release(session, &next) }, 0);

    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn playback_consumer_preserves_fifo_on_mismatch_and_blocks_early_join() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let session = session(runtime, &sink);

    let mut consumer = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_playback_consumer_create(session, &mut consumer) },
        0
    );
    assert!(!consumer.is_null());
    let mut duplicate = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_playback_consumer_create(session, &mut duplicate) },
        BUSY
    );
    assert!(duplicate.is_null());

    let mut published = Lease::default();
    assert_eq!(
        unsafe { lfm_audio_dock_reserve(session, PLAYBACK, 8, 24_000, &mut published) },
        0
    );
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_audio_dock_resolve_mut(session, &published, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(capacity, 8);
    unsafe { std::slice::from_raw_parts_mut(pcm, 8).fill(0.25) };
    assert_eq!(unsafe { lfm_audio_dock_publish(session, &published) }, 0);

    let mut wrong = published.ticket;
    wrong.sequence = wrong.sequence.wrapping_add(1);
    let mut claimed = Lease::default();
    assert_eq!(
        unsafe {
            lfm_playback_consumer_claim(
                consumer,
                &wrong,
                published.stream_epoch,
                published.lease_id,
                published.buffer_generation,
                &mut claimed,
            )
        },
        STALE
    );
    /* The mismatched record was only inspected. The exact ticket still owns
     * the same FIFO head and can claim it without shifted audio. */
    assert_eq!(
        unsafe {
            lfm_playback_consumer_claim(
                consumer,
                &published.ticket,
                published.stream_epoch,
                published.lease_id,
                published.buffer_generation,
                &mut claimed,
            )
        },
        0
    );
    assert_eq!(claimed.lease_id, published.lease_id);
    let mut read = std::ptr::null();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_playback_consumer_resolve(consumer, &claimed, &mut read, &mut count) },
        0
    );
    assert_eq!(count, 8);
    assert_eq!(unsafe { std::slice::from_raw_parts(read, count) }[0], 0.25);

    unsafe { lfm_session_request_stop(session) };
    /* Join must reject before retiring the coordinator notifier that the
     * callback-side release below still needs. */
    assert_eq!(unsafe { lfm_session_join(session) }, BUSY);
    assert_eq!(
        unsafe { lfm_playback_consumer_release(consumer, &claimed) },
        0
    );
    assert_eq!(unsafe { lfm_playback_consumer_destroy(consumer) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn retained_capture_and_control_handles_outlive_callbacks_not_the_session() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let mut config = dock_config();
    config.capture_slots = 2;
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
    let mut producer = std::ptr::null_mut();
    let mut control = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_capture_producer_create(session, &mut producer) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_control_create(session, &mut control) },
        0
    );

    let mut lease = Lease::default();
    assert_eq!(
        unsafe { lfm_capture_producer_reserve(producer, 8, 48_000, &mut lease) },
        0
    );
    let mut pcm = std::ptr::null_mut();
    let mut capacity = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_mut(producer, &lease, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(capacity, 8);
    unsafe { std::slice::from_raw_parts_mut(pcm, capacity).fill(0.125) };
    assert_eq!(
        unsafe { lfm_capture_producer_finalize(producer, &mut lease, 2, 3) },
        0
    );
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_mut(producer, &lease, &mut pcm, &mut capacity) },
        0
    );
    assert_eq!(capacity, 3);
    /* One structural producer owns the bounded ping-pong set. It can switch
     * the hardware callback to a second WRITING generation before VAD hands
     * the completed first generation to the session. */
    let mut next = Lease::default();
    assert_eq!(
        unsafe { lfm_capture_producer_reserve(producer, 8, 48_000, &mut next) },
        0
    );
    assert_ne!(next.lease_id, lease.lease_id);
    assert_eq!(unsafe { lfm_capture_producer_publish(producer, &lease) }, 0);
    assert_eq!(unsafe { lfm_capture_producer_release(producer, &next) }, 0);

    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);
    let terminal = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == lease.ticket
    });
    assert_eq!(terminal.status, 0);

    let mut epoch = 0;
    assert_eq!(
        unsafe { lfm_session_control_interrupt(control, &mut epoch) },
        0
    );
    assert_eq!(epoch, 2);
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_session_join(session) }, BUSY);
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    assert_eq!(unsafe { lfm_session_control_destroy(control) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn full_text_ring_never_creates_an_admission_waiter() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
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
    wait_gate_attempt(&sink);

    let first = saturate_reliable_ring(session);
    let mut tickets = [first[0], first[1], first[2], Ticket::default()];
    submit_text_eventually(session, b"fourth", &mut tickets[3]);

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
    assert_eq!(probe, Ticket::default());

    open_gate(session, &sink);
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
        let (next, timeout) = sink.event_edge.wait_timeout(events, remaining).unwrap();
        events = next;
        assert!(
            !timeout.timed_out(),
            "terminal callbacks missing: {events:#?}"
        );
    }
    let order: Vec<_> = events
        .iter()
        .filter(|event| {
            event.kind == EVENT_TURN && tickets.iter().any(|ticket| *ticket == event.ticket)
        })
        .map(|event| event.ticket)
        .collect();
    assert_eq!(order, tickets);
    drop(events);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn reliable_event_saturation_yields_a_fixed_result_continuation() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
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
    wait_gate_attempt(&sink);

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
        assert_eq!(snapshot.reserved_coordinator, [0; 2]);
        if snapshot.reliable_event_depth == 2 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "reliable ring never reached capacity"
        );
        std::thread::yield_now();
    }
    open_gate(session, &sink);

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
        let (next, timeout) = sink.event_edge.wait_timeout(events, remaining).unwrap();
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
fn session_runtime_has_no_operation_wait_path() {
    let source = include_str!("../native/src/runtime/voice_session.cpp");
    for forbidden in [
        "event_space_doorbell",
        "work_doorbell",
        "capture_space_doorbell",
        "playback_space_doorbell",
        "kc_port_thread_create(&session->coordinator",
        "kc_port_thread_create",
        "kc_port_wait_u32(",
        "notification_main",
        "lfm_audio_dock_wait_playback",
        "lfm_conversation_begin_pcm_native",
        "lfm_conversation_begin_text_native",
        "lfm_conversation_begin_mixed_native",
        "lfm_conversation_interrupt_native",
        "commands.mutex",
        "publication_mutex",
        "compare_exchange_weak",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden session operation waiter returned: {forbidden}"
        );
    }
    assert!(source.contains("ResultRecord result"));
    assert!(source.contains("EventRecord delivery_record"));
    assert!(source.contains("kc_service_t *delivery"));
    assert!(source.contains("kc_service_notifier_notify"));
    assert!(source.contains("stage_results(session, records, 2"));
    assert!(source.contains("stage_playback_ready(session, published)"));
    assert!(source.contains("LfmConversationAdmissionHandle admission"));
    assert!(source.contains("ACTION_PHASE_ADMISSION_PENDING"));
    assert!(source.contains("lfm_conversation_begin_pcm_submit_native"));
    assert!(source.contains("lfm_conversation_begin_collect_native"));
    assert!(source.contains("lfm_conversation_interrupt_submit_native"));
    assert!(source.contains("lfm_conversation_interrupt_collect_native"));
    assert!(source.contains("TextRecordCell"));
    assert!(source.contains("lfm_audio_dock_finalize_capture"));
    assert!(source.contains("struct LfmCaptureProducer"));
    assert!(source.contains("lfm_capture_producer_reserve"));
    assert!(source.contains("struct LfmSessionControl"));
    assert!(source.contains("lfm_session_control_interrupt"));
    assert!(source.contains("Cursor<uint64_t> publication_gate"));
    assert!(source.contains("publication_gate.value.fetch_or(PUBLICATION_CLOSED"));
    let enter_begin = source.find("bool enter_publication").unwrap();
    let enter_end = source[enter_begin..]
        .find("void leave_publication")
        .map(|offset| enter_begin + offset)
        .unwrap();
    let enter = &source[enter_begin..enter_end];
    assert!(enter.contains("publication_gate.value.fetch_add"));
    assert!(enter.contains("PUBLICATION_CLOSED"));
    assert!(enter.contains("publication_gate.value.fetch_sub"));
    assert!(enter.contains("notify_session(session)"));
    assert!(!enter.contains("for (") && !enter.contains("while ("));
    let leave_begin = enter_end;
    let leave_end = source[leave_begin..]
        .find("EventRecord make_event")
        .map(|offset| leave_begin + offset)
        .unwrap();
    let leave = &source[leave_begin..leave_end];
    assert!(leave.contains("publication_gate.value.fetch_sub"));
    assert!(leave.contains("count == 1"));
    assert!(leave.contains("notify_session(session)"));
    assert!(!leave.contains("for (") && !leave.contains("while ("));
    let retire = source
        .find("if (session->publication_gate.value.load(")
        .unwrap();
    assert!(retire < source.find("if (session->command_pending)").unwrap());
    assert!(retire < source.find("flush_published(&session->capture)").unwrap());
    for (begin, end) in [
        ("int submit_text", "int submit_mixed"),
        ("int submit_mixed", "} // namespace"),
        (
            "int lfm_audio_dock_publish",
            "int lfm_audio_dock_try_playback",
        ),
    ] {
        let begin = source.find(begin).unwrap();
        let end = source[begin..]
            .find(end)
            .map(|offset| begin + offset)
            .unwrap();
        let publication = &source[begin..end];
        assert!(publication.contains("enter_publication(session)"));
        assert!(publication.contains("leave_publication(session)"));
        assert!(!publication.contains("while ("));
    }
    let push_begin = source.find("void pool_push").unwrap();
    let push_end = source[push_begin..]
        .find("bool pool_pop")
        .map(|offset| push_begin + offset)
        .unwrap();
    let push = &source[push_begin..push_end];
    assert!(push.contains("tail.value.fetch_add"));
    assert!(push.contains("cell->sequence.store"));
    assert!(!push.contains("compare_exchange"));
    assert!(!push.contains("for (") && !push.contains("while ("));
    let reserve_begin = source.find("int lfm_capture_producer_reserve").unwrap();
    let reserve_end = source[reserve_begin..]
        .find("int lfm_capture_producer_resolve_mut")
        .map(|offset| reserve_begin + offset)
        .unwrap();
    let reserve = &source[reserve_begin..reserve_end];
    assert!(reserve.contains("reserve_one"));
    assert!(!reserve.contains("lfm_audio_dock_reserve"));
    assert!(!reserve.contains("for (") && !reserve.contains("while ("));
    let publish_begin = source.find("int lfm_capture_producer_publish").unwrap();
    let publish_end = source[publish_begin..]
        .find("int lfm_capture_producer_release")
        .map(|offset| publish_begin + offset)
        .unwrap();
    let publish = &source[publish_begin..publish_end];
    assert!(publish.contains("producer->active_leases.fetch_sub"));
    assert!(!publish.contains("producer->lease"));
    let dock = include_str!("../native/include/lfm_audio_dock.h");
    assert!(!dock.contains("lfm_audio_dock_wait_playback"));
    assert!(!dock.contains("lfm_audio_dock_try_playback"));
    assert!(dock.contains("lfm_playback_consumer_claim"));
    let rust = include_str!("../src/native_voice.rs");
    for forbidden in [
        "spawn_playback",
        "lfm-native-playback-dock",
        "send_reply(",
        "recv_reply(",
        "await_ticket(",
        "crossbeam_channel",
        ".recv()",
    ] {
        assert!(
            !rust.contains(forbidden),
            "forbidden Rust playback waiter returned: {forbidden}"
        );
    }
    assert!(rust.contains("sink.replies.try_push(reply)"));
    assert!(rust.contains("resume.notify()"));
    assert!(rust.contains("fn drain_events("));
    assert!(rust.contains("lfm_session_host_capacity"));

    let engine = include_str!("../native/src/engine/flashkern_engine.cpp");
    for forbidden in [
        "kc_collective",
        "kc_port_wait_u32",
        "kc_team_wait",
        "compare_exchange_weak",
        "std::atomic<uint32_t> pass_admission",
        "std::atomic<uint32_t> route_admission",
    ] {
        assert!(
            !engine.contains(forbidden),
            "forbidden numerical waiter returned: {forbidden}"
        );
    }
    assert!(engine.contains("std::atomic<bool> pass_closed"));
    assert!(engine.contains("std::atomic<uint32_t> pass_publishers"));
    assert!(engine.contains("std::atomic<uint32_t> route_publishers"));
    for (begin, end) in [
        (
            "static bool enter_pass_admission",
            "static void leave_pass_admission",
        ),
        (
            "static bool enter_route_admission",
            "static void leave_route_admission",
        ),
    ] {
        let begin = engine.find(begin).unwrap();
        let end = engine[begin..]
            .find(end)
            .map(|offset| begin + offset)
            .unwrap();
        let admission = &engine[begin..end];
        assert!(!admission.contains("for (;;)") && !admission.contains("while ("));
        assert!(admission.contains("compare_exchange_strong"));
        assert!(admission.matches("memory_order_seq_cst").count() >= 3);
    }
}

#[test]
fn kernel_bridge_is_a_bounded_ticket_edge_not_a_descriptor_registry() {
    let bridge = include_str!("../native/src/runtime/kernel_bridge.cpp");
    let header = include_str!("../native/include/lfm_kernel_bridge.h");
    for forbidden in [
        "std::mutex",
        "compare_exchange_weak",
        "for (;;)",
        "while (",
        "DescriptorSlot",
        "descriptor_mutex",
        "descriptor_create",
        "descriptor_retain",
        "descriptor_release",
        "descriptor_get",
        "submit_borrowed",
        "producer_acquire",
        "producer_release",
        "BORROWED_DESCRIPTOR",
    ] {
        assert!(
            !bridge.contains(forbidden) && !header.contains(forbidden),
            "generic or retrying bridge machinery returned: {forbidden}"
        );
    }
    assert!(bridge.contains("ADMISSION_PUBLISHER"));
    assert!(bridge.contains("compare_exchange_strong"));
    assert!(bridge.contains("fetch_and"));
    assert!(bridge.matches("memory_order_seq_cst").count() >= 6);

    let engine = include_str!("../native/src/engine/flashkern_engine.cpp");
    assert!(engine.contains(".slot = slot->index"));
    assert!(engine.contains(".generation = ticket_generation"));
    assert!(engine.contains("submission.descriptor.slot < e->slots.size()"));
    assert!(!engine.contains("LfmKernelDescriptor"));
    assert!(!engine.contains("KC_COORD_SUBMISSION_BORROWED_DESCRIPTOR"));
}

#[test]
fn conversation_owned_frontend_state_never_waits_on_a_numerical_mutex() {
    let frontend = include_str!("../native/src/frontend/lfm_frontend.cpp");
    for forbidden in [
        "std::mutex",
        "lock_guard",
        "unique_lock",
        "condition_variable",
        "kc_port_wait",
    ] {
        assert!(
            !frontend.contains(forbidden),
            "frontend/resampler numerical ownership regressed: {forbidden}"
        );
    }
    assert!(frontend.contains("A workspace is mounted on one conversation"));
    assert!(frontend.contains("Stream state is conversation-owned"));
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
