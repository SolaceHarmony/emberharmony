use std::ffi::{c_char, c_int, c_void};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use liquid_audio as _;

const ABI: u32 = 4;
const INVALID: i32 = -22;
const BUSY: i32 = -16;
const WOULD_BLOCK: i32 = -11;
const STALE: i32 = -116;
const CANCELLED: i32 = -125;
const HOST_SINK: i32 = -1002;
#[cfg(target_os = "macos")]
const TIMED_OUT: i32 = -60;
#[cfg(not(target_os = "macos"))]
const TIMED_OUT: i32 = -110;
const CHUNK_GAP: u32 = 1;
const CHUNK_XRUN: u32 = 2;
const CHUNK_MUTED: u32 = 4;
const CAPTURE_F32: u32 = 1;
const CAPTURE_I16: u32 = 2;
const CAPTURE_U16: u32 = 3;
const WRITE_GAP_PUBLISHED: u32 = 1;
const DOCK_ONLY: u64 = 1 << 63;
const MANUAL_DEADLINES: u64 = 1 << 62;
const DEADLINE_PREPARE: u32 = 0;
const DEADLINE_COMMIT: u32 = 1;
const DEADLINE_FORCED: u32 = 2;
const EVENT_STATE: u32 = 1;
const EVENT_TURN: u32 = 3;
const EVENT_ERROR: u32 = 4;
const EVENT_STOPPED: u32 = 5;

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
    playback_slots: u32,
    capture_max_callback_frames: u32,
    playback_frames_per_slot: u32,
    pcm_channels: u32,
    capture_sample_rate: u32,
    playback_sample_rate: u32,
    command_capacity: u32,
    max_new_tokens: u32,
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

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct CaptureChunk {
    size: u32,
    abi_version: u32,
    stream: u64,
    lane: u32,
    flags: u32,
    chunk_sequence: u64,
    first_sample_cursor: u64,
    stream_epoch: u64,
    turn_ticket: Ticket,
    lease_id: u64,
    buffer_generation: u64,
    offset_frames: u32,
    frames: u32,
    channels: u32,
    sample_rate: u32,
    reserved: [u64; 2],
}

impl Default for CaptureChunk {
    fn default() -> Self {
        // SAFETY: this private C descriptor contains integer identity and bounds only.
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct CaptureWrite {
    size: u32,
    abi_version: u32,
    admitted_frames: u32,
    dropped_frames: u32,
    flags: u32,
    status: i32,
    reserved: [u64; 2],
}

impl Default for CaptureWrite {
    fn default() -> Self {
        // SAFETY: this private C outcome contains integer accounting only.
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct MutableSpan {
    data: *mut f32,
    count: usize,
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
    live_playback_leases: u32,
    reliable_event_depth: u32,
    reliable_event_capacity: u32,
    reserved: [u64; 4],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct CapturePolicySnapshot {
    size: u32,
    abi_version: u32,
    sample_rate: u32,
    state: u32,
    last_voice: u32,
    detector_backlog: u32,
    evidence_updates: u64,
    last_evidence_cursor: u64,
    turn_start_cursor: u64,
    last_voiced_cursor: u64,
    voiced_frames: u64,
    silence_frames: u64,
    pause_generation: u64,
    prepare_sample_generation: u64,
    commit_sample_generation: u64,
    forced_sample_generation: u64,
    last_score: f64,
    adaptive_min: u32,
    adaptive_max: u32,
    discarded_silence_frames: u64,
    reserved: [u64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct CaptureDeadlineSlotSnapshot {
    slot: u32,
    armed: u32,
    terminal: u32,
    cancel_cause: u32,
    arm_generation: u64,
    expiry_generation: u64,
    scope_generation: u64,
    epoch: u64,
    domain: u64,
    pause_generation: u64,
    child: Ticket,
    parent: Ticket,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct CaptureSupervisionSnapshot {
    size: u32,
    abi_version: u32,
    cycle_active: u32,
    scope_phase: u32,
    source_phase: u32,
    source_pending_events: u32,
    policy_state: u32,
    reserved0: u32,
    scope_generation: u64,
    epoch: u64,
    domain: u64,
    pause_generation: u64,
    prepare_ready_generation: u64,
    commit_ready_generation: u64,
    forced_ready_generation: u64,
    prepare_sample_generation: u64,
    commit_sample_generation: u64,
    forced_sample_generation: u64,
    turn_start_cursor: u64,
    last_evidence_cursor: u64,
    silence_frames: u64,
    parent: Ticket,
    slots: [CaptureDeadlineSlotSnapshot; 3],
}

impl Default for CaptureSupervisionSnapshot {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: ABI,
            // SAFETY: remaining private diagnostic fields are scalar identities.
            ..unsafe { std::mem::zeroed() }
        }
    }
}

impl Default for CapturePolicySnapshot {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: 1,
            // SAFETY: remaining private diagnostic fields are scalar output cells.
            ..unsafe { std::mem::zeroed() }
        }
    }
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
    fn lfm_session_interrupt(session: *mut Session, out_epoch: *mut u64) -> c_int;
    fn lfm_session_host_capacity(session: *mut Session) -> c_int;
    fn lfm_session_request_stop(session: *mut Session);
    fn lfm_session_join(session: *mut Session) -> c_int;
    fn lfm_session_snapshot(session: *const Session, out: *mut SessionSnapshot) -> c_int;
    fn lfm_session_capture_policy_snapshot(
        session: *const Session,
        out: *mut CapturePolicySnapshot,
    ) -> c_int;
    fn lfm_session_capture_supervision_snapshot(
        session: *const Session,
        out: *mut CaptureSupervisionSnapshot,
    ) -> c_int;
    fn lfm_session_capture_deadline_advance_manual_test(
        session: *mut Session,
        elapsed_ns: u64,
    ) -> c_int;
    fn lfm_session_capture_deadline_fire_manual_test(session: *mut Session, slot: u32) -> c_int;
    fn lfm_session_capture_deadline_identity_test(
        session: *const Session,
        slot: u32,
        identity: *const CaptureDeadlineSlotSnapshot,
    ) -> c_int;
    fn lfm_session_destroy(session: *mut Session) -> c_int;

    fn lfm_capture_chunk_producer_create(
        session: *mut Session,
        stream: u64,
        lane: u32,
        out: *mut *mut CaptureProducer,
    ) -> c_int;
    fn lfm_capture_producer_claim_chunk(
        producer: *mut CaptureProducer,
        frames: u32,
        sample_rate: u32,
        source_channels: u32,
        flags: u32,
        out: *mut CaptureChunk,
    ) -> c_int;
    #[link_name = "lfm_capture_producer_resolve_chunk"]
    fn lfm_capture_producer_resolve_chunk_spans(
        producer: *mut CaptureProducer,
        chunk: *const CaptureChunk,
        spans: *mut MutableSpan,
        count: *mut u32,
    ) -> c_int;
    fn lfm_capture_producer_commit_chunk(
        producer: *mut CaptureProducer,
        chunk: *const CaptureChunk,
    ) -> c_int;
    fn lfm_capture_producer_write_interleaved(
        producer: *mut CaptureProducer,
        samples: *const c_void,
        sample_count: usize,
        channels: u32,
        sample_rate: u32,
        format: u32,
        flags: u32,
        out: *mut CaptureWrite,
    ) -> c_int;
    fn lfm_capture_producer_abort_chunk(
        producer: *mut CaptureProducer,
        chunk: *const CaptureChunk,
    ) -> c_int;
    fn lfm_capture_producer_publish_gap(
        producer: *mut CaptureProducer,
        frames: u32,
        source_channels: u32,
        flags: u32,
        out: *mut CaptureChunk,
    ) -> c_int;
    fn lfm_capture_producer_destroy(producer: *mut CaptureProducer) -> c_int;
    fn lfm_playback_consumer_create(
        session: *mut Session,
        out: *mut *mut PlaybackConsumer,
    ) -> c_int;
    fn lfm_playback_consumer_destroy(consumer: *mut PlaybackConsumer) -> c_int;

}

unsafe fn lfm_capture_producer_resolve_chunk(
    producer: *mut CaptureProducer,
    chunk: *const CaptureChunk,
    out: *mut *mut f32,
    count: *mut usize,
) -> c_int {
    let mut spans = [MutableSpan::default(); 2];
    let mut span_count = 0;
    // SAFETY: this test adapter immediately consumes the private native views.
    let status = unsafe {
        lfm_capture_producer_resolve_chunk_spans(
            producer,
            chunk,
            spans.as_mut_ptr(),
            &mut span_count,
        )
    };
    if status != 0 {
        return status;
    }
    if span_count != 1 {
        return INVALID;
    }
    // SAFETY: callers supply valid scalar out parameters.
    unsafe {
        *out = spans[0].data;
        *count = spans[0].count;
    }
    0
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
        playback_slots: 1,
        capture_max_callback_frames: 1_920,
        playback_frames_per_slot: 32,
        pcm_channels: 1,
        capture_sample_rate: 48_000,
        playback_sample_rate: 24_000,
        command_capacity: 4,
        max_new_tokens: 8,
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

fn chunk_session(runtime: *mut Runtime, sink: &Sink) -> (*mut Session, *mut CaptureProducer) {
    let config = dock_config();
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(sink).cast_mut().cast(),
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 7, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    (session, producer)
}

fn manual_chunk_session(
    runtime: *mut Runtime,
    sink: &Sink,
) -> (*mut Session, *mut CaptureProducer) {
    manual_chunk_session_rate(runtime, sink, 48_000)
}

fn manual_chunk_session_rate(
    runtime: *mut Runtime,
    sink: &Sink,
    rate: u32,
) -> (*mut Session, *mut CaptureProducer) {
    let mut config = dock_config();
    config.capture_sample_rate = rate;
    config.capture_max_callback_frames = rate.div_ceil(50) * 2;
    config.flags |= MANUAL_DEADLINES;
    let callbacks = Callbacks {
        size: std::mem::size_of::<Callbacks>() as u32,
        abi_version: ABI,
        context: std::ptr::from_ref(sink).cast_mut().cast(),
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 71, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(sink, |event| event.kind == EVENT_STATE);
    (session, producer)
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

fn write_chunk(
    producer: *mut CaptureProducer,
    frames: u32,
    flags: u32,
    value: f32,
) -> CaptureChunk {
    let mut chunk = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, frames, 48_000, 1, flags, &mut chunk) },
        0
    );
    let mut samples = std::ptr::null_mut();
    let mut count = 0;
    let mut stale = chunk;
    stale.buffer_generation = stale.buffer_generation.wrapping_add(1);
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &stale, &mut samples, &mut count) },
        STALE
    );
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &chunk, &mut samples, &mut count) },
        0
    );
    assert_eq!(count, frames as usize);
    // SAFETY: resolve returned the exact generation-checked mono callback subspan.
    unsafe { std::slice::from_raw_parts_mut(samples, count) }.fill(value);
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &chunk) },
        0
    );
    chunk
}

fn write_signal(producer: *mut CaptureProducer, frames: u32, voiced: bool) -> CaptureChunk {
    write_signal_rate(producer, frames, 48_000, voiced)
}

fn write_signal_rate(
    producer: *mut CaptureProducer,
    frames: u32,
    rate: u32,
    voiced: bool,
) -> CaptureChunk {
    assert_ne!(frames, 0);
    let limit = rate.div_ceil(50) * 2;
    let half_period = (rate / 2_000).max(1) as usize;
    let mut remaining = frames;
    let mut offset = 0usize;
    let mut last = CaptureChunk::default();
    while remaining != 0 {
        let count = remaining.min(limit);
        let mut chunk = CaptureChunk::default();
        assert_eq!(
            unsafe { lfm_capture_producer_claim_chunk(producer, count, rate, 1, 0, &mut chunk) },
            0,
            "device-sized claim failed: total={frames} remaining={remaining} offset={offset} count={count} rate={rate}"
        );
        let mut samples = std::ptr::null_mut();
        let mut capacity = 0;
        assert_eq!(
            unsafe {
                lfm_capture_producer_resolve_chunk(producer, &chunk, &mut samples, &mut capacity)
            },
            0
        );
        let slice = unsafe { std::slice::from_raw_parts_mut(samples, capacity) };
        for (index, sample) in slice.iter_mut().enumerate() {
            *sample = if voiced && ((offset + index) / half_period) % 2 == 0 {
                0.25
            } else if voiced {
                -0.25
            } else {
                0.0
            };
        }
        assert_eq!(
            unsafe { lfm_capture_producer_commit_chunk(producer, &chunk) },
            0
        );
        last = chunk;
        remaining -= count;
        offset += count as usize;
    }
    last
}

fn sync_session(session: *mut Session, sink: &Sink, label: &[u8]) {
    let mut ticket = Ticket::default();
    submit_text_eventually(session, label, &mut ticket);
    let _ = wait_event(sink, |event| {
        event.kind == EVENT_TURN && event.ticket == ticket
    });
}

fn write_signal_batched(
    session: *mut Session,
    producer: *mut CaptureProducer,
    sink: &Sink,
    frames: u64,
    rate: u32,
    voiced: bool,
    label: u8,
) -> CaptureChunk {
    let limit = rate.div_ceil(50) * 2;
    /* One device-sized publication followed by a reliable continuation edge.
     * Background-silence reclamation may advance after any complete callback;
     * batching several callbacks without observing that edge would turn a
     * correct asynchronous handoff into a test-side busy retry. */
    let batch = u64::from(limit);
    let mut remaining = frames;
    let mut last = CaptureChunk::default();
    let mut sequence = 0u8;
    while remaining != 0 {
        let count = remaining.min(batch) as u32;
        last = write_signal_rate(producer, count, rate, voiced);
        sync_session(session, sink, &[label, sequence]);
        sequence = sequence.wrapping_add(1);
        remaining -= u64::from(count);
    }
    last
}

fn capture_supervision(session: *mut Session) -> CaptureSupervisionSnapshot {
    let mut snapshot = CaptureSupervisionSnapshot::default();
    assert_eq!(
        unsafe { lfm_session_capture_supervision_snapshot(session, &mut snapshot) },
        0
    );
    snapshot
}

fn drive_capture_to_candidate(
    session: *mut Session,
    producer: *mut CaptureProducer,
    sink: &Sink,
) -> (CaptureChunk, CaptureSupervisionSnapshot) {
    for index in 0..15u8 {
        let chunk = write_signal(producer, 960, true);
        sync_session(session, sink, &[b'c', index]);
        let snapshot = capture_supervision(session);
        if snapshot.policy_state == 1 {
            assert_eq!(snapshot.parent, chunk.turn_ticket);
            return (chunk, snapshot);
        }
        assert_eq!(
            snapshot.policy_state, 0,
            "candidate fixture advanced past CANDIDATE without exposing it"
        );
    }
    panic!("Sesame detector did not enter CANDIDATE after bounded voiced evidence");
}

fn drive_capture_to_speaking(
    session: *mut Session,
    producer: *mut CaptureProducer,
    sink: &Sink,
) -> (CaptureChunk, CaptureSupervisionSnapshot) {
    for index in 0..30u8 {
        let chunk = write_signal(producer, 960, true);
        sync_session(session, sink, &[b'v', index]);
        let snapshot = capture_supervision(session);
        if snapshot.policy_state == 2 {
            assert_eq!(snapshot.parent, chunk.turn_ticket);
            return (chunk, snapshot);
        }
        assert!(
            snapshot.policy_state <= 1,
            "voiced fixture reached an unexpected turn-policy state"
        );
    }
    panic!("Sesame detector did not enter SPEAKING after bounded voiced evidence");
}

fn drive_capture_to_pause(
    session: *mut Session,
    producer: *mut CaptureProducer,
    sink: &Sink,
) -> (Ticket, CaptureSupervisionSnapshot) {
    let (voice, _) = drive_capture_to_speaking(session, producer, sink);
    for index in 0..20u8 {
        let _ = write_signal(producer, 960, false);
        sync_session(session, sink, &[b'p', index]);
        let snapshot = capture_supervision(session);
        if snapshot.policy_state == 3 {
            assert_eq!(snapshot.cycle_active, 1);
            assert_ne!(snapshot.scope_generation, 0);
            assert_eq!(snapshot.parent, voice.turn_ticket);
            assert_eq!(snapshot.slots[DEADLINE_PREPARE as usize].armed, 1);
            assert_eq!(snapshot.slots[DEADLINE_COMMIT as usize].armed, 1);
            return (voice.turn_ticket, snapshot);
        }
    }
    panic!("Sesame detector did not enter PAUSE after bounded silent evidence");
}

#[test]
fn capture_chunks_append_arbitrary_blocks_without_a_manual_boundary() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    let first = write_chunk(producer, 3, 0, 0.1);
    let second = write_chunk(producer, 5, 0, 0.2);
    let mut aborted = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 2, 48_000, 1, 0, &mut aborted) },
        0
    );
    assert_eq!(
        unsafe { lfm_capture_producer_abort_chunk(producer, &aborted) },
        0
    );
    let last = write_chunk(producer, 7, 0, 0.3);

    assert_eq!((first.offset_frames, first.frames), (0, 3));
    assert_eq!((second.offset_frames, second.frames), (3, 5));
    assert_eq!((last.offset_frames, last.frames), (8, 7));
    assert_eq!(
        [
            first.chunk_sequence,
            second.chunk_sequence,
            last.chunk_sequence,
        ],
        [1, 2, 3]
    );
    assert_eq!(
        [
            first.first_sample_cursor,
            second.first_sample_cursor,
            last.first_sample_cursor,
        ],
        [0, 3, 8]
    );
    assert_ne!(first.lease_id, second.lease_id);
    assert_ne!(second.lease_id, last.lease_id);
    assert_eq!(first.turn_ticket, last.turn_ticket);

    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.payload, b"interrupted");
    let mut stale = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &first, &mut stale, &mut count) },
        STALE
    );
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &first) },
        STALE
    );
    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.capture_consumed, 0);
    assert_eq!(snapshot.capture_stale, 0);
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.ticket != first.turn_ticket));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_gap_is_explicit_and_never_splices_the_following_range() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    let first = write_chunk(producer, 4, 0, 0.25);
    let mut gap = CaptureChunk::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_publish_gap(producer, 6, 1, CHUNK_GAP | CHUNK_XRUN, &mut gap)
        },
        0
    );
    assert_eq!(gap.flags, CHUNK_GAP | CHUNK_XRUN);
    assert_eq!(gap.chunk_sequence, 2);
    assert_eq!(gap.first_sample_cursor, 4);
    assert_eq!(gap.offset_frames, 4);
    assert_eq!(gap.frames, 6);
    let mut barrier = Ticket::default();
    submit_text_eventually(session, b"gap-boundary", &mut barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == barrier
    });
    assert!(
        sink.events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.ticket != first.turn_ticket),
        "background/candidate PCM must not become a model turn"
    );

    let next = write_chunk(producer, 3, 0, 0.5);
    assert_ne!(next.lease_id, first.lease_id);
    assert_ne!(next.turn_ticket, first.turn_ticket);
    assert_eq!(next.chunk_sequence, 3);
    assert_eq!(next.first_sample_cursor, 10);
    assert_eq!(next.offset_frames, 10);
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.payload, b"interrupted");
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.ticket != next.turn_ticket));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_arena_wrap_publishes_two_borrowed_spans_without_relocation() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);
    const RATE: u32 = 48_000;
    const CALLBACK: u32 = 1_920;
    const CADENCE: u32 = RATE.div_ceil(50);
    const CAPACITY: u32 = 2 * RATE * 30 + 2 * CADENCE + 2 * CALLBACK;

    let mut gap = CaptureChunk::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_publish_gap(
                producer,
                CAPACITY - 1,
                1,
                CHUNK_GAP | CHUNK_MUTED,
                &mut gap,
            )
        },
        0
    );
    sync_session(session, &sink, b"arena-wrap-gap");

    let mut chunk = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 4, RATE, 1, 0, &mut chunk) },
        0
    );
    assert_eq!(chunk.first_sample_cursor, u64::from(CAPACITY - 1));
    assert_eq!(chunk.offset_frames, CAPACITY - 1);
    let mut spans = [MutableSpan::default(); 2];
    let mut count = 0;
    assert_eq!(
        unsafe {
            lfm_capture_producer_resolve_chunk_spans(
                producer,
                &chunk,
                spans.as_mut_ptr(),
                &mut count,
            )
        },
        0
    );
    assert_eq!(count, 2);
    assert_eq!((spans[0].count, spans[1].count), (1, 3));
    unsafe { spans[0].data.write(0.25) };
    unsafe { std::slice::from_raw_parts_mut(spans[1].data, 3) }
        .copy_from_slice(&[-0.25, 0.5, -0.5]);
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &chunk) },
        0
    );
    sync_session(session, &sink, b"arena-wrap-pcm");

    let mut next = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 1, RATE, 1, 0, &mut next) },
        0
    );
    assert_eq!(next.first_sample_cursor, u64::from(CAPACITY) + 3);
    assert_eq!(next.offset_frames, 3);
    assert_eq!(next.buffer_generation, 2);
    assert_eq!(
        unsafe { lfm_capture_producer_abort_chunk(producer, &next) },
        0
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_freeze_dormancy_is_resumed_by_the_active_writer_commit() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    let mut chunk = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 8, 48_000, 1, 0, &mut chunk) },
        0
    );
    let mut samples = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &chunk, &mut samples, &mut count) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(samples, count) }.fill(0.75);
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(epoch, chunk.stream_epoch + 1);
    let mut duplicate = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 1, 48_000, 1, 0, &mut duplicate) },
        WOULD_BLOCK
    );
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &chunk) },
        0
    );
    let successor = write_chunk(producer, 1, 0, 0.25);
    assert_eq!(successor.stream_epoch, epoch);
    sync_session(session, &sink, b"writer-retired");
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.ticket != chunk.turn_ticket));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn stopped_active_writer_publishes_its_idle_successor() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);
    let mut chunk = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 8, 48_000, 1, 0, &mut chunk) },
        0
    );

    unsafe { lfm_session_request_stop(session) };
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &chunk) },
        CANCELLED
    );
    assert_eq!(
        unsafe { lfm_capture_producer_abort_chunk(producer, &chunk) },
        STALE
    );
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn stale_active_callback_cannot_seed_the_new_epoch_detector() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let _ = drive_capture_to_speaking(session, producer, &sink);

    let mut stale = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 960, 48_000, 1, 0, &mut stale) },
        0
    );
    let mut samples = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &stale, &mut samples, &mut count) },
        0
    );
    let half_period = 24usize;
    for (index, sample) in unsafe { std::slice::from_raw_parts_mut(samples, count) }
        .iter_mut()
        .enumerate()
    {
        *sample = if (index / half_period) % 2 == 0 {
            0.25
        } else {
            -0.25
        };
    }

    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &stale) },
        0
    );
    let current = write_signal(producer, 960, false);
    assert_eq!(current.stream_epoch, epoch);
    sync_session(session, &sink, b"new-epoch-silence");

    let policy = capture_supervision(session);
    assert_eq!(policy.policy_state, 0);
    assert_eq!(policy.cycle_active, 0);
    assert_eq!(policy.turn_start_cursor, 0);
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.ticket != stale.turn_ticket));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_interrupt_rotates_correlation_without_relocating_the_arena() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    let stale_chunk = write_chunk(producer, 4, 0, -0.25);
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    assert_eq!(epoch, stale_chunk.stream_epoch + 1);

    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.payload, b"interrupted");

    let current = write_chunk(producer, 3, 0, 0.5);
    assert_eq!(current.stream_epoch, epoch);
    assert_ne!(current.lease_id, stale_chunk.lease_id);
    assert_ne!(current.turn_ticket, stale_chunk.turn_ticket);
    assert_eq!(current.chunk_sequence, 2);
    assert_eq!(current.first_sample_cursor, 4);
    assert_eq!(current.offset_frames, 4);
    let mut next_epoch = 0;
    assert_eq!(
        unsafe { lfm_session_interrupt(session, &mut next_epoch) },
        0
    );
    assert_eq!(next_epoch, epoch + 1);
    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == next_epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.payload, b"interrupted");
    assert!(sink.events.lock().unwrap().iter().all(|event| {
        event.ticket != stale_chunk.turn_ticket && event.ticket != current.turn_ticket
    }));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn native_interleaved_write_owns_conversion_commit_and_explicit_drop() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    let stereo = [0.75f32, -0.25, 0.5, 0.25];
    let mut first = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                stereo.as_ptr().cast(),
                stereo.len(),
                2,
                48_000,
                CAPTURE_F32,
                0,
                &mut first,
            )
        },
        0
    );
    assert_eq!((first.admitted_frames, first.dropped_frames), (2, 0));
    assert_eq!((first.flags, first.status), (0, 0));

    let partial = [1i16, -1, 2];
    let mut dropped = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                partial.as_ptr().cast(),
                partial.len(),
                2,
                48_000,
                CAPTURE_I16,
                0,
                &mut dropped,
            )
        },
        0
    );
    assert_eq!((dropped.admitted_frames, dropped.dropped_frames), (0, 2));
    assert_eq!(dropped.flags, WRITE_GAP_PUBLISHED);
    assert_eq!(dropped.status, INVALID);
    let mut barrier = Ticket::default();
    submit_text_eventually(session, b"discarded-gap", &mut barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == barrier
    });

    let mono = [i16::MIN, 0];
    let mut second = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                mono.as_ptr().cast(),
                mono.len(),
                1,
                48_000,
                CAPTURE_I16,
                0,
                &mut second,
            )
        },
        0
    );
    assert_eq!((second.admitted_frames, second.status), (2, 0));

    let unsigned = [0u16, u16::MAX, 32768, 32768];
    let mut last = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                unsigned.as_ptr().cast(),
                unsigned.len(),
                2,
                48_000,
                CAPTURE_U16,
                0,
                &mut last,
            )
        },
        0
    );
    assert_eq!((last.admitted_frames, last.status), (2, 0));
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_chunk_preserves_ingress_channels_but_resolves_one_mono_plane() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);

    for channels in [2, 6] {
        let mut chunk = CaptureChunk::default();
        assert_eq!(
            unsafe {
                lfm_capture_producer_claim_chunk(producer, 3, 48_000, channels, 0, &mut chunk)
            },
            0
        );
        assert_eq!(chunk.channels, channels);
        let mut mono = std::ptr::null_mut();
        let mut count = 0;
        assert_eq!(
            unsafe { lfm_capture_producer_resolve_chunk(producer, &chunk, &mut mono, &mut count) },
            0
        );
        assert_eq!(
            count, 3,
            "native capture storage is one mono destination plane"
        );
        unsafe { std::slice::from_raw_parts_mut(mono, count) }.fill(0.25);
        assert_eq!(
            unsafe { lfm_capture_producer_abort_chunk(producer, &chunk) },
            0
        );
    }

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn full_ring_xrun_debt_publishes_before_any_successor_pcm() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let config = dock_config();
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 7, 0, &mut producer) },
        0
    );

    let sample = [0.0f32];
    for _ in 0..512 {
        let mut wrote = CaptureWrite::default();
        assert_eq!(
            unsafe {
                lfm_capture_producer_write_interleaved(
                    producer,
                    sample.as_ptr().cast(),
                    sample.len(),
                    1,
                    48_000,
                    CAPTURE_F32,
                    0,
                    &mut wrote,
                )
            },
            0
        );
        assert_eq!((wrote.admitted_frames, wrote.dropped_frames), (1, 0));
    }

    let mut dropped = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                sample.as_ptr().cast(),
                sample.len(),
                1,
                48_000,
                CAPTURE_F32,
                0,
                &mut dropped,
            )
        },
        0
    );
    assert_eq!((dropped.admitted_frames, dropped.dropped_frames), (0, 1));
    assert_eq!((dropped.flags, dropped.status), (0, WOULD_BLOCK));
    let mut blocked = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 1, 48_000, 1, 0, &mut blocked) },
        WOULD_BLOCK,
        "unpublished gap debt must close direct PCM admission"
    );

    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let mut first_barrier = Ticket::default();
    submit_text_eventually(session, b"drain", &mut first_barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == first_barrier
    });

    let mut paid = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                sample.as_ptr().cast(),
                sample.len(),
                1,
                48_000,
                CAPTURE_F32,
                0,
                &mut paid,
            )
        },
        0
    );
    assert_eq!((paid.admitted_frames, paid.dropped_frames), (0, 1));
    assert_eq!(paid.flags, WRITE_GAP_PUBLISHED);
    assert_eq!(paid.status, WOULD_BLOCK);
    let early =
        unsafe { lfm_capture_producer_claim_chunk(producer, 1, 48_000, 1, 0, &mut blocked) };
    assert!(
        early == WOULD_BLOCK || early == 0,
        "successor admission may occur only after the sequenced gap rotates"
    );
    if early == 0 {
        assert_eq!(blocked.chunk_sequence, 514);
        assert_eq!(blocked.first_sample_cursor, 514);
        assert_eq!(blocked.offset_frames, 514);
    }

    let mut second_barrier = Ticket::default();
    submit_text_eventually(session, b"gap", &mut second_barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == second_barrier
    });
    let mut successor = blocked;
    if early == WOULD_BLOCK {
        assert_eq!(
            unsafe { lfm_capture_producer_claim_chunk(producer, 1, 48_000, 1, 0, &mut successor,) },
            0
        );
    }
    assert_eq!(successor.chunk_sequence, 514);
    assert_eq!(successor.first_sample_cursor, 514);
    let mut mono = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &successor, &mut mono, &mut count) },
        0
    );
    unsafe { std::slice::from_raw_parts_mut(mono, count) }.fill(0.5);
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &successor) },
        0
    );
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_disconnect_transfers_inflight_retirement_to_the_coordinator() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let config = dock_config();
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 7, 0, &mut producer) },
        0
    );
    let _ = write_chunk(producer, 8, 0, 0.25);
    let mut muted = CaptureChunk::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_publish_gap(producer, 16, 2, CHUNK_GAP | CHUNK_MUTED, &mut muted)
        },
        0
    );
    assert_eq!(muted.flags, CHUNK_GAP | CHUNK_MUTED);
    assert_eq!(muted.channels, 2);
    assert_eq!(
        unsafe { lfm_capture_producer_destroy(producer) },
        0,
        "disconnect must not return BUSY for accepted metadata/freeze work"
    );

    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let mut barrier = Ticket::default();
    submit_text_eventually(session, b"retired", &mut barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == barrier
    });

    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn started_capture_disconnect_retires_before_stopped_and_join() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = chunk_session(runtime, &sink);
    let _ = write_chunk(producer, 8, 0, 0.25);
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);

    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    let events = sink.events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == EVENT_STOPPED)
            .count(),
        1,
        "STOPPED is published exactly once after native capture retirement"
    );
    drop(events);
    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);

    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn native_sesame_policy_consumes_exact_sample_clock_windows() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let config = dock_config();
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 9, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);

    /* The 300 ms policy applies to detector-classified retained speech, not
     * raw device admission. This gate is about exact cadence accounting, so
     * keep the synthetic carrier comfortably beyond the adaptive detector's
     * bounded warm-up instead of coupling it to the decision threshold. */
    let _ = write_signal_batched(session, producer, &sink, 28_800, 48_000, true, b'v');
    let _ = write_signal_batched(session, producer, &sink, 72_000, 48_000, false, b's');
    let pause = capture_supervision(session);
    assert_eq!(
        pause.policy_state, 3,
        "speech followed by detector silence is PAUSE"
    );
    assert_ne!(pause.prepare_sample_generation, 0);
    assert_eq!(
        pause.commit_sample_generation, pause.prepare_sample_generation,
        "sample-clock prepare and commit readiness belong to one pause generation"
    );
    assert_eq!(pause.forced_sample_generation, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_session_join(session) }, 0);

    let mut policy = CapturePolicySnapshot::default();
    assert_eq!(
        unsafe { lfm_session_capture_policy_snapshot(session, &mut policy) },
        0
    );
    assert_eq!(policy.sample_rate, 48_000);
    assert_eq!(policy.evidence_updates, 105);
    assert_eq!(policy.last_evidence_cursor, 100_800);
    assert_eq!(policy.detector_backlog, 0);
    assert_eq!(policy.last_voice, 0);

    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn capture_commit_expiry_before_samples_remains_usable() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (parent, pause) = drive_capture_to_pause(session, producer, &sink);
    assert_eq!(pause.commit_sample_generation, 0);

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 500_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
        0
    );
    sync_session(session, &sink, b"expiry-first");
    let expired = capture_supervision(session);
    assert_eq!(
        expired.slots[DEADLINE_COMMIT as usize].expiry_generation,
        pause.pause_generation
    );
    assert_eq!(expired.slots[DEADLINE_COMMIT as usize].terminal, 0);
    assert_eq!(expired.commit_ready_generation, 0);
    assert!(
        sink.events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.ticket != parent),
        "wall-clock evidence alone committed an audio turn"
    );

    assert!(expired.silence_frames < 24_000);
    let _ = write_signal(producer, (24_000 - expired.silence_frames) as u32, false);
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == parent
    });
    assert_eq!(turn.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_commit_samples_before_expiry_remain_uncommitted() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (parent, pause) = drive_capture_to_pause(session, producer, &sink);
    assert!(pause.silence_frames < 24_000);
    let _ = write_signal(producer, (24_000 - pause.silence_frames) as u32, false);
    sync_session(session, &sink, b"samples-first");
    let sampled = capture_supervision(session);
    assert_eq!(sampled.commit_sample_generation, pause.pause_generation);
    assert_eq!(sampled.slots[DEADLINE_COMMIT as usize].expiry_generation, 0);
    assert_eq!(sampled.commit_ready_generation, 0);
    assert!(
        sink.events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.ticket != parent),
        "sample-clock evidence alone committed an audio turn"
    );

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 500_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
        0
    );
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == parent
    });
    assert_eq!(turn.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn consecutive_capture_turns_rearm_one_fresh_scope_and_ticket() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let mut prior = Ticket::default();

    for turn in 0..2u8 {
        let (parent, pause) = drive_capture_to_pause(session, producer, &sink);
        assert!(parent.sequence > prior.sequence);
        assert!(pause.silence_frames < 24_000);
        let _ = write_signal(producer, (24_000 - pause.silence_frames) as u32, false);
        assert_eq!(
            unsafe { lfm_session_capture_deadline_advance_manual_test(session, 500_000_000) },
            0
        );
        assert_eq!(
            unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
            0
        );
        let event = wait_event(&sink, |event| {
            event.kind == EVENT_TURN && event.ticket == parent
        });
        assert_eq!(event.status, 0);
        prior = parent;
        sync_session(session, &sink, &[b'x', turn]);
    }

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn candidate_accumulates_voice_across_brief_sesame_negative_valleys() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (first, _) = drive_capture_to_candidate(session, producer, &sink);

    for index in 0..10u8 {
        let _ = write_signal(producer, 960, false);
        sync_session(session, &sink, &[b'n', index]);
    }
    let valley = capture_supervision(session);
    assert_ne!(
        valley.policy_state, 0,
        "a brief spectral valley discarded a real candidate utterance"
    );
    assert_eq!(valley.parent, first.turn_ticket);

    let mut speaking = valley;
    for index in 0..20u8 {
        let _ = write_signal(producer, 960, true);
        sync_session(session, &sink, &[b'v', index]);
        speaking = capture_supervision(session);
        if speaking.policy_state == 2 {
            break;
        }
    }
    assert_eq!(speaking.policy_state, 2);
    assert_eq!(speaking.parent, first.turn_ticket);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn minimum_utterance_uses_retained_span_not_positive_window_sum() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (first, candidate) = drive_capture_to_candidate(session, producer, &sink);

    for index in 0..12u8 {
        let _ = write_signal(producer, 960, false);
        sync_session(session, &sink, &[b's', index]);
    }
    let valley = capture_supervision(session);
    assert_eq!(valley.policy_state, 1);
    assert_eq!(valley.parent, first.turn_ticket);

    let mut speaking = valley;
    for index in 0..5u8 {
        let _ = write_signal(producer, 960, true);
        sync_session(session, &sink, &[b'e', index]);
        speaking = capture_supervision(session);
        if speaking.policy_state == 2 {
            break;
        }
    }
    let minimum = 48_000 * 300 / 1_000;
    assert_eq!(speaking.policy_state, 2);
    assert_eq!(speaking.parent, first.turn_ticket);
    assert!(
        speaking.last_evidence_cursor - speaking.turn_start_cursor >= minimum,
        "retained utterance span did not reach the minimum: candidate={candidate:#?}, speaking={speaking:#?}"
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn candidate_false_start_retires_after_endpoint_length_silence() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (first, _) = drive_capture_to_candidate(session, producer, &sink);

    for index in 0..50u8 {
        let _ = write_signal(producer, 960, false);
        sync_session(session, &sink, &[b'f', index]);
        let snapshot = capture_supervision(session);
        if snapshot.policy_state == 0 && snapshot.cycle_active == 0 {
            break;
        }
    }
    let retired = capture_supervision(session);
    assert_eq!(retired.policy_state, 0);
    assert_eq!(retired.cycle_active, 0);
    assert!(sink
        .events
        .lock()
        .unwrap()
        .iter()
        .all(|event| event.ticket != first.turn_ticket));

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn resumed_voice_disarms_and_rearms_one_exact_pause_generation() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (parent, first) = drive_capture_to_pause(session, producer, &sink);
    const SPEAKING: u32 = 2;
    let mut resumed = first;
    for index in 0..8u8 {
        let _ = write_signal(producer, 960, true);
        sync_session(session, &sink, &[b'r', index]);
        resumed = capture_supervision(session);
        if resumed.policy_state == SPEAKING && resumed.scope_generation > first.scope_generation {
            break;
        }
    }
    assert_eq!(resumed.policy_state, SPEAKING);
    assert!(resumed.scope_generation > first.scope_generation);
    assert_eq!(resumed.parent, parent);
    assert_eq!(resumed.epoch, first.epoch);
    assert_eq!(resumed.domain, first.domain);
    assert_eq!(
        resumed.slots[DEADLINE_FORCED as usize].pause_generation, resumed.scope_generation,
        "forced liveness is correlated to the durable turn scope, not a transient pause"
    );

    let mut second = resumed;
    for index in 0..20u8 {
        let _ = write_signal(producer, 960, false);
        sync_session(session, &sink, &[b's', index]);
        second = capture_supervision(session);
        if second.policy_state == 3 {
            break;
        }
    }
    assert_eq!(second.policy_state, 3);
    assert!(second.scope_generation > first.scope_generation);
    assert_eq!(second.parent, parent);
    assert_eq!(second.epoch, first.epoch);
    assert_eq!(second.domain, first.domain);
    assert_eq!(
        second.slots[DEADLINE_FORCED as usize].pause_generation,
        second.scope_generation
    );
    assert!(
        second.slots[DEADLINE_COMMIT as usize].arm_generation
            > first.slots[DEADLINE_COMMIT as usize].arm_generation
    );
    assert_ne!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
        0,
        "a queued callback from the canceled pause expired its successor"
    );

    assert!(second.silence_frames < 24_000);
    let _ = write_signal(producer, (24_000 - second.silence_frames) as u32, false);
    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 500_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
        0
    );
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == parent
    });

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn prestart_voiced_capture_enters_a_presealed_supervision_scope() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let mut config = dock_config();
    config.flags |= MANUAL_DEADLINES;
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 79, 0, &mut producer) },
        0
    );
    let voice = write_signal(producer, 14_400, true);

    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);
    sync_session(session, &sink, b"prestart-voice");
    let supervised = capture_supervision(session);
    assert_eq!(supervised.cycle_active, 1);
    assert_eq!(supervised.parent, voice.turn_ticket);
    assert_eq!(supervised.epoch, voice.stream_epoch);
    assert_ne!(supervised.scope_generation, 0);
    assert_eq!(
        supervised.slots[DEADLINE_FORCED as usize].pause_generation,
        supervised.scope_generation
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_turn_may_begin_at_absolute_cursor_zero() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session_rate(runtime, &sink, 12_800);

    let mut first = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 256, 12_800, 1, 0, &mut first) },
        0
    );
    let mut samples = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &first, &mut samples, &mut count) },
        0
    );
    assert_eq!(count, 256);
    // Excite nearly the whole selected 600-2400 Hz band on the first 20 ms
    // evidence window. A single square-wave tone is intentionally too sparse
    // for Sesame's adaptive band-mean classifier at this rate.
    for (index, sample) in unsafe { std::slice::from_raw_parts_mut(samples, count) }
        .iter_mut()
        .enumerate()
    {
        *sample = (13usize..48)
            .map(|bin| {
                let amplitude = if bin < 18 { 0.1 } else { 1.0 };
                let phase = bin as f32 * 2.399_963_1;
                amplitude
                    * (std::f32::consts::TAU * bin as f32 * index as f32 / 256.0 + phase).cos()
            })
            .sum::<f32>()
            / 35.0;
    }
    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &first) },
        0
    );
    assert_eq!(first.first_sample_cursor, 0);
    sync_session(session, &sink, b"cursor-zero");
    // A second reliable command edge proves the capture callback and the
    // scope-begin continuation both ran before observation; this is not a
    // timed poll or a synthetic speech boundary.
    sync_session(session, &sink, b"cursor-zero-settled");
    let active = capture_supervision(session);
    assert_eq!(active.policy_state, 1);
    assert_eq!(active.cycle_active, 1);
    assert_eq!(active.turn_start_cursor, 0);
    assert_eq!(active.parent, first.turn_ticket);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn first_capture_ticket_is_minted_after_an_earlier_typed_action() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let mut typed = Ticket::default();
    submit_text_eventually(session, b"typed-before-capture", &mut typed);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == typed
    });

    let capture = write_signal(producer, 960, true);
    assert!(
        capture.turn_ticket.sequence > typed.sequence,
        "capture ticket was pre-minted before the typed action"
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn forced_turn_in_pause_retains_a_nonzero_prefix_at_non_48k() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    const RATE: u32 = 44_100;
    const PREFIX: u32 = RATE * 10;
    const VOICE: u32 = RATE * 3 / 10;
    const FORCED: u64 = RATE as u64 * 30;
    let (session, producer) = manual_chunk_session_rate(runtime, &sink, RATE);

    let prefix = write_signal_batched(
        session,
        producer,
        &sink,
        u64::from(PREFIX),
        RATE,
        false,
        b'l',
    );
    let listening = capture_supervision(session);
    assert_eq!(listening.policy_state, 0);
    assert_eq!(listening.turn_start_cursor, 0);

    let mut voice = write_signal_rate(producer, VOICE, RATE, true);
    sync_session(session, &sink, b"non48-voice");
    let mut speaking = capture_supervision(session);
    for index in 0..20u8 {
        if speaking.policy_state == 2 {
            break;
        }
        voice = write_signal_rate(producer, RATE / 25, RATE, true);
        sync_session(session, &sink, &[b'v', index]);
        speaking = capture_supervision(session);
    }
    assert_eq!(
        speaking.policy_state, 2,
        "the non-48k fixture never reached detector-confirmed speech"
    );
    assert_eq!(speaking.parent, voice.turn_ticket);
    assert_ne!(speaking.turn_start_cursor, 0);
    assert!(
        speaking.turn_start_cursor + 256 >= prefix.first_sample_cursor + prefix.frames as u64,
        "the retained turn begins at the detector's exact pre-roll view"
    );
    assert_eq!(
        speaking.slots[DEADLINE_FORCED as usize].pause_generation,
        speaking.scope_generation
    );

    let cursor = voice.first_sample_cursor + voice.frames as u64;
    let target = speaking.turn_start_cursor + FORCED + u64::from(RATE.div_ceil(50));
    assert!(target > cursor);
    let _ = write_signal_batched(session, producer, &sink, target - cursor, RATE, false, b'f');
    let paused = capture_supervision(session);
    assert_eq!(paused.policy_state, 3);
    let forced_generation = paused.slots[DEADLINE_FORCED as usize].pause_generation;
    assert_eq!(forced_generation, paused.scope_generation);
    assert_eq!(paused.forced_sample_generation, forced_generation);

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 30_000_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_FORCED) },
        0
    );
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == voice.turn_ticket
    });
    assert_eq!(turn.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_deadline_rejects_every_mismatched_correlation_coordinate() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (_, pause) = drive_capture_to_pause(session, producer, &sink);
    let exact = pause.slots[DEADLINE_COMMIT as usize];
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &exact) },
        0
    );

    let mut wrong = exact;
    wrong.child.sequence += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.parent.sequence += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.scope_generation += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.epoch += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.domain += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.pause_generation += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );
    wrong = exact;
    wrong.arm_generation += 1;
    assert_eq!(
        unsafe { lfm_session_capture_deadline_identity_test(session, DEADLINE_COMMIT, &wrong) },
        STALE
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn stopped_turn_cancels_children_before_source_and_notifier_retirement() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let _ = drive_capture_to_pause(session, producer, &sink);
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn capture_device_loss_is_explicit_while_listening() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    assert_eq!(capture_supervision(session).policy_state, 0);
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    let lost = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.status == CANCELLED
    });
    assert_eq!(lost.epoch, capture_supervision(session).epoch.max(1));
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_device_loss_cancels_candidate_with_its_parent_ticket() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (first, candidate) = drive_capture_to_candidate(session, producer, &sink);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    let lost = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == first.turn_ticket && event.status == CANCELLED
    });
    assert_eq!(lost.epoch, candidate.epoch);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_device_loss_cancels_speaking_with_its_parent_ticket() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (voice, speaking) = drive_capture_to_speaking(session, producer, &sink);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    let lost = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == voice.turn_ticket && event.status == CANCELLED
    });
    assert_eq!(lost.epoch, speaking.epoch);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn capture_device_loss_cancels_pause_after_deadline_evidence() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (parent, pause) = drive_capture_to_pause(session, producer, &sink);
    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 500_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_COMMIT) },
        0
    );
    sync_session(session, &sink, b"pause-expiry-before-device-loss");
    assert_ne!(
        capture_supervision(session).slots[DEADLINE_COMMIT as usize].expiry_generation,
        0
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    let lost = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == parent && event.status == CANCELLED
    });
    assert_eq!(lost.epoch, pause.epoch);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn forced_deadline_has_one_grace_edge_then_faults_a_silent_device() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (first, started) = drive_capture_to_speaking(session, producer, &sink);
    assert_eq!(started.parent, first.turn_ticket);
    assert_eq!(started.slots[DEADLINE_FORCED as usize].armed, 1);
    const FORCED_FRAMES: u64 = 1_440_000;
    const SHORT_BY: u64 = 480;
    let cursor = first.first_sample_cursor + u64::from(first.frames);
    let target = started.turn_start_cursor + FORCED_FRAMES - SHORT_BY;
    assert!(target > cursor);
    let _ = write_signal_batched(
        session,
        producer,
        &sink,
        target - cursor,
        48_000,
        true,
        b'n',
    );

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 30_000_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_FORCED) },
        0
    );
    sync_session(session, &sink, b"forced-grace");
    let grace = capture_supervision(session);
    assert_eq!(grace.slots[DEADLINE_FORCED as usize].terminal, 0);
    assert_eq!(grace.slots[DEADLINE_FORCED as usize].armed, 1);
    assert_ne!(grace.slots[DEADLINE_FORCED as usize].expiry_generation, 0);

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 20_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_FORCED) },
        0
    );
    let fault = wait_event(&sink, |event| {
        event.kind == EVENT_ERROR && event.status == TIMED_OUT
    });
    assert_eq!(fault.epoch, started.epoch);
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, TIMED_OUT);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn forced_sample_readiness_is_restamped_after_resume_restart() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (voice, started) = drive_capture_to_speaking(session, producer, &sink);
    let cursor = voice.first_sample_cursor + voice.frames as u64;
    /* The 30 s policy is sample-clock exact but is observed on the next
     * 20 ms Sesame evidence edge. Supply that edge explicitly. */
    let target = started.turn_start_cursor + 1_440_000 + 960;
    assert!(target > cursor);
    let _ = write_signal_batched(
        session,
        producer,
        &sink,
        target - cursor,
        48_000,
        false,
        b'q',
    );
    let first = capture_supervision(session);
    assert_eq!(first.policy_state, 3);
    assert_eq!(
        first.forced_sample_generation, first.slots[DEADLINE_FORCED as usize].pause_generation,
        "forced threshold evidence did not stamp its child: {first:#?}"
    );

    let mut restarted = first;
    for index in 0..12u8 {
        let _ = write_signal(producer, 1_920, true);
        sync_session(session, &sink, &[b'u', index]);
        restarted = capture_supervision(session);
        if restarted.policy_state == 2 && restarted.scope_generation > first.scope_generation {
            break;
        }
    }
    assert_eq!(restarted.policy_state, 2);
    assert!(restarted.scope_generation > first.scope_generation);
    assert_eq!(restarted.parent, voice.turn_ticket);
    assert_eq!(
        restarted.forced_sample_generation, restarted.scope_generation,
        "sample-first readiness must be restamped into the replacement forced child"
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_FORCED) },
        0
    );
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == voice.turn_ticket
    });
    assert_eq!(turn.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn forced_expiry_grants_one_bounded_writer_completion_edge() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let (session, producer) = manual_chunk_session(runtime, &sink);
    let (voice, started) = drive_capture_to_speaking(session, producer, &sink);
    const HELD: u64 = 1_920;
    let cursor = voice.first_sample_cursor + voice.frames as u64;
    /* Completing the held callback must also deliver the next 20 ms evidence
     * edge on which the exact 30 s sample threshold is observed. */
    let target = started.turn_start_cursor + 1_440_000 + 960 - HELD;
    assert!(target > cursor);
    let _ = write_signal_batched(
        session,
        producer,
        &sink,
        target - cursor,
        48_000,
        true,
        b'w',
    );

    let mut held = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, HELD as u32, 48_000, 1, 0, &mut held) },
        0
    );
    let mut samples = std::ptr::null_mut();
    let mut count = 0;
    assert_eq!(
        unsafe { lfm_capture_producer_resolve_chunk(producer, &held, &mut samples, &mut count) },
        0
    );
    for (index, sample) in unsafe { std::slice::from_raw_parts_mut(samples, count) }
        .iter_mut()
        .enumerate()
    {
        *sample = if (index / 24) % 2 == 0 { 0.25 } else { -0.25 };
    }

    assert_eq!(
        unsafe { lfm_session_capture_deadline_advance_manual_test(session, 30_000_000_000) },
        0
    );
    assert_eq!(
        unsafe { lfm_session_capture_deadline_fire_manual_test(session, DEADLINE_FORCED) },
        0
    );
    sync_session(session, &sink, b"writer-grace-expired");
    let grace = capture_supervision(session);
    assert_eq!(grace.slots[DEADLINE_FORCED as usize].terminal, 0);
    assert_eq!(grace.slots[DEADLINE_FORCED as usize].armed, 1);

    assert_eq!(
        unsafe { lfm_capture_producer_commit_chunk(producer, &held) },
        0
    );
    let turn = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == voice.turn_ticket
    });
    assert_eq!(turn.status, 0);

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn background_silence_reclaims_native_storage_without_relocation() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let config = dock_config();
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 11, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);

    let _ = write_signal_rate(producer, 2_048, 48_000, false);
    let mut barrier = Ticket::default();
    submit_text_eventually(session, b"silence-recycle", &mut barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == barrier
    });

    let mut next = CaptureChunk::default();
    assert_eq!(
        unsafe { lfm_capture_producer_claim_chunk(producer, 1, 48_000, 1, 0, &mut next) },
        0
    );
    assert_eq!(next.first_sample_cursor, 2_048);
    assert_eq!(next.offset_frames, 2_048);
    assert_eq!(
        unsafe { lfm_capture_producer_abort_chunk(producer, &next) },
        0
    );
    assert!(
        sink.events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.kind != EVENT_TURN || event.ticket == barrier),
        "classified background silence must not publish a model turn"
    );

    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { stop_all(runtime, session, 0) };
}

#[test]
fn sesame_cadence_is_rational_on_nondivisible_capture_rates() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let mut config = dock_config();
    config.capture_sample_rate = 44_101;
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 12, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);

    let _ = write_signal_rate(producer, 4_411, 44_101, false);
    let mut barrier = Ticket::default();
    submit_text_eventually(session, b"rational-cadence", &mut barrier);
    let _ = wait_event(&sink, |event| {
        event.kind == EVENT_TURN && event.ticket == barrier
    });
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    unsafe { lfm_session_request_stop(session) };
    assert_eq!(unsafe { lfm_session_join(session) }, 0);
    let mut policy = CapturePolicySnapshot::default();
    assert_eq!(
        unsafe { lfm_session_capture_policy_snapshot(session, &mut policy) },
        0
    );
    assert_eq!(policy.evidence_updates, 5);
    assert_eq!(policy.last_evidence_cursor, 4_411);

    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn unstarted_capture_disconnect_retires_without_a_coordinator() {
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
    let mut producer = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 8, 0, &mut producer) },
        0
    );
    let _ = write_chunk(producer, 8, 0, 0.25);
    let mut muted = CaptureChunk::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_publish_gap(producer, 16, 2, CHUNK_GAP | CHUNK_MUTED, &mut muted)
        },
        0
    );
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);

    /* Setup failed before mount/start. Administrative join must complete the
     * bounded retirement because no coordinator continuation ever ran. */
    unsafe { stop_all(runtime, session, 0) };
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
fn capture_rate_is_sealed_at_session_readiness() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: false,
    };
    let runtime = runtime();
    let mut config = dock_config();
    config.capture_sample_rate = 16_000;
    config.playback_sample_rate = 24_000;
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
    let _ = wait_event(&sink, |event| event.kind == EVENT_STATE);

    let mut producer = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 9, 0, &mut producer) },
        0
    );
    let input = [0.25f32; 8];
    let mut default_capture = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                input.as_ptr().cast(),
                input.len(),
                1,
                0,
                CAPTURE_F32,
                0,
                &mut default_capture,
            )
        },
        0
    );
    assert_eq!(default_capture.admitted_frames, 8);
    let mut capture = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                input.as_ptr().cast(),
                input.len(),
                1,
                16_000,
                CAPTURE_F32,
                0,
                &mut capture,
            )
        },
        0
    );
    assert_eq!(capture.admitted_frames, 8);
    let mut epoch = 0;
    assert_eq!(unsafe { lfm_session_interrupt(session, &mut epoch) }, 0);
    let interrupted = wait_event(&sink, |event| {
        event.kind == EVENT_STATE && event.epoch == epoch && event.payload == b"interrupted"
    });
    assert_eq!(interrupted.payload, b"interrupted");
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);

    unsafe { stop_all(runtime, session, 0) };
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
fn reliable_output_saturation_does_not_starve_native_capture_evidence() {
    let sink = GateSink {
        events: Mutex::new(Vec::new()),
        event_edge: Condvar::new(),
        blocked: Mutex::new(true),
        attempts: Mutex::new(0),
        attempt_edge: Condvar::new(),
    };
    let runtime = runtime_with(2);
    let mut config = dock_config();
    config.command_capacity = 2;
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
    let mut producer = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 10, 0, &mut producer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);
    wait_gate_attempt(&sink);

    let mut tickets = [Ticket::default(); 2];
    for (index, ticket) in tickets.iter_mut().enumerate() {
        let text = [b'a' + index as u8];
        assert_eq!(
            unsafe { lfm_session_submit_text(session, text.as_ptr().cast(), text.len(), ticket,) },
            0
        );
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let mut snapshot = SessionSnapshot::default();
        assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
        if snapshot.reliable_event_depth == 2 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "reliable output never reached its fixed capacity"
        );
        std::thread::yield_now();
    }

    let voiced: Vec<f32> = (0..960)
        .map(|sample| if (sample / 24) % 2 == 0 { 0.25 } else { -0.25 })
        .collect();
    let mut write = CaptureWrite::default();
    assert_eq!(
        unsafe {
            lfm_capture_producer_write_interleaved(
                producer,
                voiced.as_ptr().cast(),
                voiced.len(),
                1,
                48_000,
                CAPTURE_F32,
                0,
                &mut write,
            )
        },
        0
    );
    assert_eq!(write.status, 0);

    loop {
        let mut policy = CapturePolicySnapshot::default();
        assert_eq!(
            unsafe { lfm_session_capture_policy_snapshot(session, &mut policy) },
            0
        );
        if policy.evidence_updates == 1 {
            assert_eq!(policy.last_evidence_cursor, 960);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "capture detector starved behind reliable output capacity"
        );
        std::thread::yield_now();
    }

    open_gate(session, &sink);
    let _ = wait_gate_event(&sink, |event| event.kind == EVENT_STATE);
    for ticket in tickets {
        let _ = wait_gate_event(&sink, |event| {
            event.kind == EVENT_TURN && event.ticket == ticket
        });
    }
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
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
fn native_failure_keeps_coordinator_alive_until_device_endpoints_retire() {
    let sink = Sink {
        events: Mutex::new(Vec::new()),
        edge: Condvar::new(),
        fail: true,
    };
    let runtime = runtime();
    let config = dock_config();
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
    assert_eq!(
        unsafe { lfm_capture_chunk_producer_create(session, 17, 0, &mut producer) },
        0
    );
    let mut consumer = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_playback_consumer_create(session, &mut consumer) },
        0
    );
    assert_eq!(unsafe { lfm_session_start(session) }, 0);

    let stopped = wait_event(&sink, |event| event.kind == EVENT_STOPPED);
    assert_eq!(stopped.status, HOST_SINK);
    assert_eq!(unsafe { lfm_session_join(session) }, BUSY);

    /* STOPPED asks the platform owner to tear down its device endpoints. Each
     * close publishes the next edge; the retained native coordinator then
     * retires every capture range before its own terminal transition. */
    assert_eq!(unsafe { lfm_capture_producer_destroy(producer) }, 0);
    assert_eq!(unsafe { lfm_playback_consumer_destroy(consumer) }, 0);
    assert_eq!(unsafe { lfm_session_join(session) }, HOST_SINK);
    let mut snapshot = SessionSnapshot::default();
    assert_eq!(unsafe { lfm_session_snapshot(session, &mut snapshot) }, 0);
    assert_eq!(snapshot.live_playback_leases, 0);
    assert_eq!(unsafe { lfm_session_destroy(session) }, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
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
    assert!(source.contains("lfm_conversation_begin_pcm_spans_submit_native"));
    assert!(source.contains("lfm_conversation_begin_collect_native"));
    assert!(source.contains("lfm_conversation_interrupt_submit_native"));
    assert!(source.contains("lfm_conversation_interrupt_collect_native"));
    assert!(source.contains("TextRecordCell"));
    assert!(!source.contains("lfm_audio_dock_finalize_capture"));
    assert!(!source.contains("lfm_session_submit_mixed"));
    assert!(source.contains("struct LfmCaptureProducer"));
    assert!(source.contains("lfm_capture_producer_claim_chunk"));
    assert!(source.contains("lfm_capture_producer_write_interleaved"));
    assert!(source.contains("ACTION_CAPTURE_DRAIN_BUDGET"));
    assert!(source.contains("LFM_EVENT_TURN_STARTED"));
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
    assert!(retire < source.find("flush_published(&session->playback)").unwrap());
    for (begin, end) in [("int submit_text", "} // namespace")] {
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
    let begin = source.rfind("int playback_publish").unwrap();
    let end = source[begin..]
        .find("int playback_release")
        .map(|offset| begin + offset)
        .unwrap();
    let publication = &source[begin..end];
    assert!(publication.contains("enter_publication(session)"));
    assert!(publication.contains("leave_publication(session)"));
    assert!(!publication.contains("while ("));
    for legacy in [
        "PcmPool",
        "COMMAND_MIXED",
        "pending_capture",
        "capture_pending",
        "process_capture(",
        "process_mixed(",
        "lfm_conversation_begin_pcm_submit_native",
        "lfm_conversation_begin_mixed_submit_native",
        "lfm_audio_dock_reserve",
        "lfm_audio_dock_resolve_mut",
        "lfm_audio_dock_publish",
        "lfm_audio_dock_release",
        "lfm_audio_dock_try_playback",
    ] {
        assert!(
            !source.contains(legacy),
            "legacy capture-pool seam returned: {legacy}"
        );
    }
    assert!(source.contains("struct PlaybackPool"));
    assert!(source.contains("decode_playback_lease_id"));
    let push_begin = source.find("void pool_push").unwrap();
    let push_end = source[push_begin..]
        .find("bool pool_peek")
        .map(|offset| push_begin + offset)
        .unwrap();
    let push = &source[push_begin..push_end];
    assert!(push.contains("tail.value.fetch_add"));
    assert!(push.contains("cell->sequence.store"));
    assert!(!push.contains("compare_exchange"));
    assert!(!push.contains("for (") && !push.contains("while ("));
    let claim_begin = source.find("int lfm_capture_producer_claim_chunk").unwrap();
    let claim_end = source[claim_begin..]
        .find("int lfm_capture_producer_resolve_chunk")
        .map(|offset| claim_begin + offset)
        .unwrap();
    let claim = &source[claim_begin..claim_end];
    for forbidden in ["new ", "lock_guard", "mutex", "for (", "while ("] {
        assert!(
            !claim.contains(forbidden),
            "realtime chunk claim gained forbidden work: {forbidden}"
        );
    }
    let commit_begin = source
        .find("int lfm_capture_producer_commit_chunk")
        .unwrap();
    let commit_end = source[commit_begin..]
        .find("int lfm_capture_producer_abort_chunk")
        .map(|offset| commit_begin + offset)
        .unwrap();
    let commit = &source[commit_begin..commit_end];
    for forbidden in ["new ", "lock_guard", "mutex", "for (", "while ("] {
        assert!(
            !commit.contains(forbidden),
            "realtime chunk commit gained forbidden work: {forbidden}"
        );
    }
    let write_begin = source
        .find("int lfm_capture_producer_write_interleaved")
        .unwrap();
    let write_end = source[write_begin..]
        .find("int lfm_capture_producer_abort_chunk")
        .map(|offset| write_begin + offset)
        .unwrap();
    let write = &source[write_begin..write_end];
    for forbidden in ["new ", "lock_guard", "mutex", "for (", "while ("] {
        assert!(
            !write.contains(forbidden),
            "realtime interleaved write gained forbidden work: {forbidden}"
        );
    }
    assert!(write.contains("lfm_capture_downmix_f32"));
    assert!(write.contains("lfm_capture_downmix_i16"));
    assert!(write.contains("lfm_capture_downmix_u16"));
    let step_begin = source.find("SessionProgress session_step").unwrap();
    let step_end = source[step_begin..]
        .find("void coordinator_main")
        .map(|offset| step_begin + offset)
        .unwrap();
    let step = &source[step_begin..step_end];
    assert!(
        step.find("step_capture_policy").unwrap() < step.find("advance_action(session)").unwrap(),
        "capture policy must run before active recurrence"
    );
    assert!(source.contains("lfm_sesame_detector_process"));
    assert!(source.contains("capture_duration_frames(session->capture_rate, 200)"));
    assert!(source.contains("capture_duration_frames(session->capture_rate, 500)"));
    assert!(source.contains("recycle_background_silence"));
    let dock = include_str!("../native/include/lfm_audio_dock.h");
    let chunk_begin = dock.find("typedef struct LfmCaptureChunkV1").unwrap();
    let chunk_end = dock[chunk_begin..]
        .find("} LfmCaptureChunkV1;")
        .map(|offset| chunk_begin + offset)
        .unwrap();
    let chunk = &dock[chunk_begin..chunk_end];
    assert!(
        !chunk.contains('*'),
        "capture records must never carry pointers"
    );
    assert!(
        !dock.contains("TURN_END"),
        "capture transport must not regain a manual turn-boundary seam"
    );
    for legacy in [
        "lfm_capture_producer_create(",
        "lfm_capture_producer_reserve(",
        "lfm_capture_producer_resolve_mut(",
        "lfm_capture_producer_finalize(",
        "lfm_capture_producer_publish(",
        "lfm_capture_producer_release(",
        "lfm_capture_producer_request_turn_end(",
    ] {
        assert!(
            !dock.contains(legacy),
            "legacy capture ABI returned: {legacy}"
        );
        assert!(
            !source.contains(legacy),
            "legacy capture implementation returned: {legacy}"
        );
    }
    assert!(!dock.contains("lfm_audio_dock_wait_playback"));
    assert!(!dock.contains("lfm_audio_dock_try_playback"));
    assert!(!dock.contains("lfm_audio_dock_reserve"));
    assert!(!dock.contains("lfm_audio_dock_resolve_mut"));
    assert!(!dock.contains("lfm_audio_dock_publish"));
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
