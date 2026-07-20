//! Opaque Rust host seam for the native LFM2 voice runtime.
//!
//! This module deliberately exposes lifecycle, sampling policy, PCM leases and
//! semantic events only. Model bytes, weight-field names, token ids, mel rows, codec
//! codes and recurrence never cross this boundary.

use std::cell::UnsafeCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kcoro_sys::RealtimeNotifier;

use crate::ffi;
use crate::voice_api::{
    CaptureMute, CaptureSink, CaptureWrite, EngineProgress, PlaybackSource, PlaybackWrite,
    VoiceEngine, VoiceEvent,
};

const RUNTIME_ABI: u32 = 4;
const STATUS_WOULD_BLOCK: i32 = -11;
const STATUS_BUSY: i32 = -16;
const STATUS_STALE: i32 = -116;
const STATUS_CANCELLED: i32 = -125;
const STATUS_HOST_SINK: i32 = -1002;
const EVENT_STATE: u32 = 1;
const EVENT_TEXT: u32 = 2;
const EVENT_TURN: u32 = 3;
const EVENT_ERROR: u32 = 4;
const EVENT_STOPPED: u32 = 5;
const EVENT_PLAYBACK_READY: u32 = 6;
const EVENT_TURN_STARTED: u32 = 7;
const EVENT_HAS_AUDIO: u32 = 1;
const EVENT_TRUNCATED: u32 = 2;
const TICKET_SESSION: u32 = 1;
const TICKET_TURN: u32 = 2;
const TICKET_CONTROL: u32 = 8;
const REPLY_CAPACITY: usize = 128;
const TEXT_EVENT_MAX_BYTES: usize = 512;
const UTF8_CARRY_MAX_BYTES: usize = 3;
const EVENT_CAPACITY: u32 = 64;
const MAX_KERNEL_LANES: u32 = 16;
const PLAYBACK_SLOTS: u32 = 8;
const CAPTURE_INPUT_F32: u32 = 1;
const CAPTURE_INPUT_I16: u32 = 2;
const CAPTURE_INPUT_U16: u32 = 3;
const CAPTURE_WRITE_GAP_PUBLISHED: u32 = 1;
const CAPTURE_CHUNK_GAP: u32 = 1;
const CAPTURE_CHUNK_MUTED: u32 = 1 << 2;

#[repr(C)]
struct Runtime {
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
struct SessionControlHandle {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

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

#[repr(C)]
#[derive(Default)]
struct ModelMemory {
    size: u32,
    abi_version: u32,
    source_bytes: u64,
    resident_image_bytes: u64,
    directly_bound_bytes: u64,
    derived_immutable_bytes: u64,
    compatibility_copied_bytes: u64,
    load_ns: u64,
    load_workers: u32,
    load_tasks: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SamplingPolicy {
    size: u32,
    abi_version: u32,
    flags: u32,
    top_k: u32,
    temperature: f64,
    reserved: u64,
}

#[repr(C)]
struct ConversationOptions {
    size: u32,
    abi_version: u32,
    flags: u32,
    reserved0: u32,
    seed: u64,
    text: SamplingPolicy,
    audio: SamplingPolicy,
    reserved: [u64; 4],
}

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

#[repr(C)]
struct NativeEvent {
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

type OnEvent = unsafe extern "C" fn(*mut c_void, *const NativeEvent) -> i32;

#[repr(C)]
struct Callbacks {
    size: u32,
    abi_version: u32,
    context: *mut c_void,
    on_event: Option<OnEvent>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TurnEvent {
    size: u32,
    abi_version: u32,
    playback_leases: u32,
    emitted_items: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PlaybackReadyEvent {
    size: u32,
    abi_version: u32,
    lease_id: u64,
    buffer_generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PcmLease {
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

impl Default for PcmLease {
    fn default() -> Self {
        // Every field is an integer value and the C ABI defines zero as the
        // uninitialised state supplied to reserve/try-claim calls.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeCaptureWrite {
    size: u32,
    abi_version: u32,
    admitted_frames: u32,
    dropped_frames: u32,
    flags: u32,
    status: i32,
    reserved: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeCaptureChunk {
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PlaybackMeter {
    size: u32,
    abi_version: u32,
    rendered_frames: u64,
    sum_squares: f32,
    rms: f32,
    reserved: [u64; 3],
}

type PlaybackRender<T> = unsafe extern "C" fn(
    source: *const f32,
    destination: *mut T,
    frames: usize,
    channels: u32,
    destination_capacity: usize,
    meter: *mut PlaybackMeter,
) -> i32;

unsafe extern "C" {
    fn lfm_runtime_create(config: *const RuntimeConfig, out: *mut *mut Runtime) -> i32;
    fn lfm_runtime_start(runtime: *mut Runtime) -> i32;
    fn lfm_runtime_request_stop(runtime: *mut Runtime);
    fn lfm_runtime_join(runtime: *mut Runtime) -> i32;
    fn lfm_runtime_destroy(runtime: *mut Runtime) -> i32;
    fn lfm_runtime_model_open(
        runtime: *mut Runtime,
        path: *const c_char,
        out: *mut *mut ffi::Model,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_runtime_model_memory(
        runtime: *const Runtime,
        model: *const ffi::Model,
        out: *mut ModelMemory,
    ) -> i32;
    fn lfm_runtime_model_close(runtime: *mut Runtime, model: *mut ffi::Model) -> i32;
    fn lfm_runtime_conversation_create(
        runtime: *mut Runtime,
        model: *mut ffi::Model,
        options: *const ConversationOptions,
        out: *mut *mut ffi::Conversation,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_runtime_conversation_close(
        runtime: *mut Runtime,
        conversation: *mut ffi::Conversation,
    ) -> i32;
    fn lfm_session_create(
        runtime: *mut Runtime,
        model: *mut ffi::Model,
        conversation: *mut ffi::Conversation,
        config: *const SessionConfig,
        callbacks: *const Callbacks,
        out: *mut *mut Session,
    ) -> i32;
    fn lfm_session_start(session: *mut Session) -> i32;
    fn lfm_session_submit_text(
        session: *mut Session,
        utf8: *const c_char,
        utf8_bytes: usize,
        out_ticket: *mut Ticket,
    ) -> i32;
    fn lfm_session_host_capacity(session: *mut Session) -> i32;
    fn lfm_session_request_stop(session: *mut Session);
    fn lfm_session_join(session: *mut Session) -> i32;
    fn lfm_session_destroy(session: *mut Session) -> i32;
    fn lfm_playback_consumer_create(session: *mut Session, out: *mut *mut PlaybackConsumer) -> i32;
    fn lfm_playback_consumer_claim(
        consumer: *mut PlaybackConsumer,
        ticket: *const Ticket,
        stream_epoch: u64,
        lease_id: u64,
        buffer_generation: u64,
        out: *mut PcmLease,
    ) -> i32;
    fn lfm_playback_consumer_resolve(
        consumer: *const PlaybackConsumer,
        lease: *const PcmLease,
        out_samples: *mut *const f32,
        out_sample_count: *mut usize,
    ) -> i32;
    fn lfm_playback_consumer_release(
        consumer: *mut PlaybackConsumer,
        lease: *const PcmLease,
    ) -> i32;
    fn lfm_playback_consumer_destroy(consumer: *mut PlaybackConsumer) -> i32;
    fn lfm_playback_meter_reset(meter: *mut PlaybackMeter) -> i32;
    fn lfm_playback_render_f32(
        source: *const f32,
        destination: *mut f32,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
        meter: *mut PlaybackMeter,
    ) -> i32;
    fn lfm_playback_render_i16(
        source: *const f32,
        destination: *mut i16,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
        meter: *mut PlaybackMeter,
    ) -> i32;
    fn lfm_playback_render_u16(
        source: *const f32,
        destination: *mut u16,
        frames: usize,
        channels: u32,
        destination_capacity: usize,
        meter: *mut PlaybackMeter,
    ) -> i32;
    fn lfm_capture_chunk_producer_create(
        session: *mut Session,
        stream: u64,
        lane: u32,
        out: *mut *mut CaptureProducer,
    ) -> i32;
    fn lfm_capture_producer_write_interleaved(
        producer: *mut CaptureProducer,
        samples: *const c_void,
        sample_count: usize,
        channels: u32,
        sample_rate: u32,
        format: u32,
        flags: u32,
        out: *mut NativeCaptureWrite,
    ) -> i32;
    fn lfm_capture_producer_publish_gap(
        producer: *mut CaptureProducer,
        dropped_frames: u32,
        source_channels: u32,
        flags: u32,
        out: *mut NativeCaptureChunk,
    ) -> i32;
    fn lfm_capture_producer_destroy(producer: *mut CaptureProducer) -> i32;
    fn lfm_session_control_create(
        session: *mut Session,
        out: *mut *mut SessionControlHandle,
    ) -> i32;
    fn lfm_session_control_interrupt(
        control: *mut SessionControlHandle,
        out_epoch: *mut u64,
    ) -> i32;
    fn lfm_session_control_destroy(control: *mut SessionControlHandle) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeVoiceSampling {
    pub max_new_tokens: u32,
    pub text_temperature: Option<f64>,
    pub text_top_k: Option<u32>,
    pub audio_temperature: Option<f64>,
    pub audio_top_k: Option<u32>,
    pub seed: Option<u64>,
}

impl Default for NativeVoiceSampling {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            text_temperature: Some(1.0),
            text_top_k: None,
            audio_temperature: Some(1.0),
            audio_top_k: Some(4),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeVoiceRuntimeConfig {
    /// ABI v1 has one native coordinator; values other than one are rejected.
    pub coordination_workers: u32,
    /// Fixed native lane threads; the current engine accepts `1..=16`.
    pub kernel_lanes: u32,
    pub event_capacity: u32,
    pub session_capacity: u32,
}

impl Default for NativeVoiceRuntimeConfig {
    fn default() -> Self {
        Self {
            coordination_workers: 1,
            kernel_lanes: 8,
            event_capacity: EVENT_CAPACITY,
            session_capacity: 1,
        }
    }
}

/// Read-only accounting for the immutable model image owned by a native
/// lifecycle runtime. It contains no numerical model metadata or raw pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeVoiceModelMemory {
    pub source_bytes: u64,
    pub resident_image_bytes: u64,
    pub directly_bound_bytes: u64,
    pub derived_immutable_bytes: u64,
    pub compatibility_copied_bytes: u64,
    pub load_ns: u64,
    pub load_workers: u32,
    pub load_tasks: u32,
}

struct ModelOwner {
    runtime: NonNull<Runtime>,
    model: NonNull<ffi::Model>,
    config: NativeVoiceRuntimeConfig,
}

unsafe impl Send for ModelOwner {}
unsafe impl Sync for ModelOwner {}

impl Drop for ModelOwner {
    fn drop(&mut self) {
        let close = unsafe { lfm_runtime_model_close(self.runtime.as_ptr(), self.model.as_ptr()) };
        if close != 0 {
            eprintln!("[flashkern] native voice model close refused with status {close}");
            return;
        }
        unsafe { lfm_runtime_request_stop(self.runtime.as_ptr()) };
        let join = unsafe { lfm_runtime_join(self.runtime.as_ptr()) };
        if join != 0 {
            eprintln!("[flashkern] native voice runtime join refused with status {join}");
            return;
        }
        let destroy = unsafe { lfm_runtime_destroy(self.runtime.as_ptr()) };
        if destroy != 0 {
            eprintln!("[flashkern] native voice runtime destroy refused with status {destroy}");
        }
    }
}

/// One immutable native model image and its native executor.
#[derive(Clone)]
pub struct NativeVoiceModel(Arc<ModelOwner>);

impl NativeVoiceModel {
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_with_config(path, NativeVoiceRuntimeConfig::default())
    }

    pub fn open_with_config(
        path: &Path,
        runtime_config: NativeVoiceRuntimeConfig,
    ) -> Result<Self, String> {
        let _ = kcoro_sys::link_anchor as fn();
        if runtime_config.coordination_workers != 1
            || runtime_config.kernel_lanes == 0
            || runtime_config.kernel_lanes > MAX_KERNEL_LANES
            || runtime_config.event_capacity < 2
            || runtime_config.event_capacity > 64
            || runtime_config.session_capacity == 0
            || runtime_config.session_capacity > 64
        {
            return Err("native runtime configuration is outside its validated bounds".into());
        }
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| "model path contains a NUL byte".to_string())?;
        let config = RuntimeConfig {
            size: std::mem::size_of::<RuntimeConfig>() as u32,
            abi_version: RUNTIME_ABI,
            coordination_workers: runtime_config.coordination_workers,
            kernel_lanes: runtime_config.kernel_lanes,
            event_capacity: runtime_config.event_capacity,
            session_capacity: runtime_config.session_capacity,
            reserved0: 0,
            reserved1: 0,
            flags: 0,
            reserved: [0; 4],
        };
        let mut runtime = std::ptr::null_mut();
        status(
            unsafe { lfm_runtime_create(&config, &mut runtime) },
            "create native runtime",
        )?;
        let runtime = NonNull::new(runtime).ok_or("native runtime returned a null handle")?;
        if let Err(error) = status(
            unsafe { lfm_runtime_start(runtime.as_ptr()) },
            "start native runtime",
        ) {
            unsafe { lfm_runtime_request_stop(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_join(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_destroy(runtime.as_ptr()) };
            return Err(error);
        }

        let mut model = std::ptr::null_mut();
        let mut error = [0i8; 512];
        let open = unsafe {
            lfm_runtime_model_open(
                runtime.as_ptr(),
                path.as_ptr(),
                &mut model,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if open != 0 {
            unsafe { lfm_runtime_request_stop(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_join(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_destroy(runtime.as_ptr()) };
            return Err(native_error(open, &error));
        }
        let model = NonNull::new(model).ok_or_else(|| {
            unsafe { lfm_runtime_request_stop(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_join(runtime.as_ptr()) };
            let _ = unsafe { lfm_runtime_destroy(runtime.as_ptr()) };
            "native model open returned a null handle".to_string()
        })?;
        let owner = Arc::new(ModelOwner {
            runtime,
            model,
            config: runtime_config,
        });
        Ok(Self(owner))
    }

    pub fn runtime_config(&self) -> NativeVoiceRuntimeConfig {
        self.0.config
    }

    pub fn memory(&self) -> Result<NativeVoiceModelMemory, String> {
        let mut memory = ModelMemory {
            size: std::mem::size_of::<ModelMemory>() as u32,
            abi_version: ffi::ABI,
            ..Default::default()
        };
        status(
            unsafe {
                lfm_runtime_model_memory(
                    self.0.runtime.as_ptr(),
                    self.0.model.as_ptr(),
                    &mut memory,
                )
            },
            "query native voice model memory",
        )?;
        Ok(NativeVoiceModelMemory {
            source_bytes: memory.source_bytes,
            resident_image_bytes: memory.resident_image_bytes,
            directly_bound_bytes: memory.directly_bound_bytes,
            derived_immutable_bytes: memory.derived_immutable_bytes,
            compatibility_copied_bytes: memory.compatibility_copied_bytes,
            load_ns: memory.load_ns,
            load_workers: memory.load_workers,
            load_tasks: memory.load_tasks,
        })
    }

    pub fn engine(
        &self,
        sampling: NativeVoiceSampling,
        vault: Option<NativeConversationVault>,
        capture_rate: u32,
        playback_rate: u32,
        capture_max_callback_frames: u32,
    ) -> Result<NativeLfm2VoiceEngine, String> {
        NativeLfm2VoiceEngine::new(
            self.clone(),
            sampling,
            vault,
            capture_rate,
            playback_rate,
            capture_max_callback_frames,
        )
    }
}

struct ConversationOwner {
    pointer: NonNull<ffi::Conversation>,
    model: Arc<ModelOwner>,
    sampling: NativeVoiceSampling,
}

unsafe impl Send for ConversationOwner {}

impl Drop for ConversationOwner {
    fn drop(&mut self) {
        let close = unsafe {
            lfm_runtime_conversation_close(self.model.runtime.as_ptr(), self.pointer.as_ptr())
        };
        if close != 0 {
            eprintln!("[flashkern] native voice conversation close refused with status {close}");
        }
    }
}

#[derive(Default)]
struct VaultState {
    claimed: bool,
    conversation: Option<ConversationOwner>,
}

/// Lifecycle-proof home for one opaque native conversation.
#[derive(Clone, Default)]
pub struct NativeConversationVault(Arc<Mutex<VaultState>>);

struct ConversationClaim {
    vault: Option<NativeConversationVault>,
    conversation: Option<ConversationOwner>,
}

impl ConversationClaim {
    fn new(
        model: &NativeVoiceModel,
        sampling: NativeVoiceSampling,
        vault: Option<NativeConversationVault>,
    ) -> Result<Self, String> {
        let stored = if let Some(vault) = vault.as_ref() {
            let mut state = vault.0.lock().expect("conversation vault mutex poisoned");
            if state.claimed {
                return Err("native conversation is already attached to another session".into());
            }
            state.claimed = true;
            state.conversation.take()
        } else {
            None
        };
        let mut claim = Self {
            vault,
            conversation: stored,
        };
        if claim.conversation.as_ref().is_some_and(|conversation| {
            !Arc::ptr_eq(&conversation.model, &model.0) || conversation.sampling != sampling
        }) {
            claim.conversation.take();
        }
        if claim.conversation.is_none() {
            claim.conversation = Some(create_conversation(model, sampling)?);
        }
        Ok(claim)
    }

    fn into_conversation(mut self) -> ConversationOwner {
        // The engine now owns the claim; leave the vault's `claimed` latch set
        // until the engine has joined and destroyed its native session.
        self.vault = None;
        self.conversation
            .take()
            .expect("conversation claim is populated")
    }
}

impl Drop for ConversationClaim {
    fn drop(&mut self) {
        let Some(vault) = self.vault.as_ref() else {
            return;
        };
        let mut state = vault.0.lock().expect("conversation vault mutex poisoned");
        state.conversation = self.conversation.take();
        state.claimed = false;
    }
}

fn sampler(temperature: Option<f64>, top_k: Option<u32>) -> SamplingPolicy {
    SamplingPolicy {
        size: std::mem::size_of::<SamplingPolicy>() as u32,
        abi_version: RUNTIME_ABI,
        flags: temperature.is_none().then_some(1).unwrap_or(0),
        top_k: top_k.unwrap_or(0),
        temperature: temperature.unwrap_or(1.0),
        reserved: 0,
    }
}

fn create_conversation(
    model: &NativeVoiceModel,
    sampling: NativeVoiceSampling,
) -> Result<ConversationOwner, String> {
    if sampling.max_new_tokens == 0 {
        return Err("native max_new_tokens must be non-zero".into());
    }
    let options = ConversationOptions {
        size: std::mem::size_of::<ConversationOptions>() as u32,
        abi_version: RUNTIME_ABI,
        flags: sampling.seed.is_none().then_some(1).unwrap_or(0),
        reserved0: 0,
        seed: sampling.seed.unwrap_or(0),
        text: sampler(sampling.text_temperature, sampling.text_top_k),
        audio: sampler(sampling.audio_temperature, sampling.audio_top_k),
        reserved: [0; 4],
    };
    let mut pointer = std::ptr::null_mut();
    let mut error = [0i8; 512];
    let create = unsafe {
        lfm_runtime_conversation_create(
            model.0.runtime.as_ptr(),
            model.0.model.as_ptr(),
            &options,
            &mut pointer,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if create != 0 {
        return Err(native_error(create, &error));
    }
    Ok(ConversationOwner {
        pointer: NonNull::new(pointer).ok_or("native conversation returned a null handle")?,
        model: model.0.clone(),
        sampling,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flow {
    // The fixed record crossing each dock. PCM remains in its native lease;
    // this identity is what lets every continuation reject a stale handoff.
    session: u64,
    epoch: u64,
    ticket: Ticket,
}

impl Flow {
    fn new(event: &NativeEvent) -> Result<Self, i32> {
        let flow = Self {
            session: event.session_id,
            epoch: event.epoch,
            ticket: event.ticket,
        };
        if flow.session == 0
            || flow.epoch == 0
            || flow.ticket.runtime_epoch == 0
            || flow.ticket.sequence == 0
            || flow.ticket.generation == 0
            || flow.ticket.kind == 0
        {
            return Err(STATUS_HOST_SINK);
        }
        Ok(flow)
    }
}

enum Reply {
    TurnStarted {
        flow: Flow,
    },
    Text {
        flow: Flow,
        payload: TextPayload,
    },
    PlaybackReady {
        flow: Flow,
        lease_id: u64,
        buffer_generation: u64,
    },
    Turn {
        flow: Flow,
        status: i32,
        has_audio: bool,
        truncated: bool,
        playback_leases: u32,
        emitted_items: u32,
    },
    Error {
        flow: Flow,
        status: i32,
        payload: TextPayload,
    },
    Stopped {
        flow: Flow,
        status: i32,
    },
}

struct ReplyCell(UnsafeCell<MaybeUninit<Reply>>);

// One native delivery continuation is the sole producer and the Rust voice
// continuation is the sole consumer. The atomic cursors publish ownership of
// each cell; the payload itself is never concurrently accessed.
unsafe impl Sync for ReplyCell {}

struct ReplyRing {
    cells: Box<[ReplyCell]>,
    read: AtomicUsize,
    write: AtomicUsize,
}

unsafe impl Send for ReplyRing {}
unsafe impl Sync for ReplyRing {}

impl ReplyRing {
    fn new() -> Arc<Self> {
        let cells = (0..REPLY_CAPACITY)
            .map(|_| ReplyCell(UnsafeCell::new(MaybeUninit::uninit())))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Arc::new(Self {
            cells,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
        })
    }

    fn try_push(&self, reply: Reply) -> Result<(), Reply> {
        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.cells.len() {
            return Err(reply);
        }
        unsafe {
            (*self.cells[write % self.cells.len()].0.get()).write(reply);
        }
        self.write.store(write.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    fn try_pop(&self) -> Option<Reply> {
        let read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        if read == write {
            return None;
        }
        let reply = unsafe { (*self.cells[read % self.cells.len()].0.get()).assume_init_read() };
        self.read.store(read.wrapping_add(1), Ordering::Release);
        Some(reply)
    }

    fn is_empty(&self) -> bool {
        self.read.load(Ordering::Acquire) == self.write.load(Ordering::Acquire)
    }
}

impl Drop for ReplyRing {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

struct CopyCell<T: Copy>(UnsafeCell<MaybeUninit<T>>);

unsafe impl<T: Copy + Send> Sync for CopyCell<T> {}

struct CopyRing<T: Copy> {
    cells: Box<[CopyCell<T>]>,
    read: AtomicUsize,
    write: AtomicUsize,
}

unsafe impl<T: Copy + Send> Send for CopyRing<T> {}
unsafe impl<T: Copy + Send> Sync for CopyRing<T> {}

impl<T: Copy> CopyRing<T> {
    fn new(capacity: usize) -> Self {
        let cells = (0..capacity.max(1))
            .map(|_| CopyCell(UnsafeCell::new(MaybeUninit::uninit())))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            cells,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
        }
    }

    fn try_push(&self, value: T) -> bool {
        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.cells.len() {
            return false;
        }
        unsafe { (*self.cells[write % self.cells.len()].0.get()).write(value) };
        self.write.store(write.wrapping_add(1), Ordering::Release);
        true
    }

    fn try_pop(&self) -> Option<T> {
        let read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        if read == write {
            return None;
        }
        let value = unsafe { (*self.cells[read % self.cells.len()].0.get()).assume_init_read() };
        self.read.store(read.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    fn is_empty(&self) -> bool {
        self.read.load(Ordering::Acquire) == self.write.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy)]
struct PlaybackNotice {
    flow: Flow,
    lease_id: u64,
    buffer_generation: u64,
}

#[derive(Clone, Copy)]
struct PlaybackResult {
    flow: Flow,
    status: i32,
}

struct PlaybackState {
    ready: CopyRing<PlaybackNotice>,
    done: CopyRing<PlaybackResult>,
}

impl PlaybackState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            ready: CopyRing::new(REPLY_CAPACITY),
            done: CopyRing::new(REPLY_CAPACITY),
        })
    }

    fn active(&self, local: bool) -> bool {
        local || !self.ready.is_empty() || !self.done.is_empty()
    }

    fn audio_active(&self, local: bool) -> bool {
        local || !self.ready.is_empty()
    }
}

struct TextPayload {
    len: u16,
    bytes: [u8; TEXT_EVENT_MAX_BYTES],
}

impl TextPayload {
    fn new(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > TEXT_EVENT_MAX_BYTES {
            return None;
        }
        let mut payload = Self {
            len: bytes.len() as u16,
            bytes: [0; TEXT_EVENT_MAX_BYTES],
        };
        payload.bytes[..bytes.len()].copy_from_slice(bytes);
        Some(payload)
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

struct EventSink {
    replies: Arc<ReplyRing>,
    resume: Option<RealtimeNotifier>,
}

#[derive(Default)]
struct Utf8Stream {
    carry: [u8; UTF8_CARRY_MAX_BYTES],
    len: usize,
}

impl Utf8Stream {
    fn push<F>(&mut self, bytes: &[u8], emit: &mut F) -> Result<(), String>
    where
        F: FnMut(String) + ?Sized,
    {
        if bytes.len() > TEXT_EVENT_MAX_BYTES {
            self.reset();
            return Err("native text event exceeds its fixed payload bound".into());
        }
        if bytes.is_empty() {
            return Ok(());
        }

        let mut joined = [0u8; TEXT_EVENT_MAX_BYTES + UTF8_CARRY_MAX_BYTES];
        joined[..self.len].copy_from_slice(&self.carry[..self.len]);
        joined[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        let total = self.len + bytes.len();
        self.len = 0;

        let mut offset = 0;
        while offset < total {
            match std::str::from_utf8(&joined[offset..total]) {
                Ok(text) => {
                    if !text.is_empty() {
                        emit(text.to_owned());
                    }
                    return Ok(());
                }
                Err(error) => {
                    const REPLACEMENT: &str = "\u{fffd}";
                    let valid = error.valid_up_to();
                    if valid != 0 {
                        let text = std::str::from_utf8(&joined[offset..offset + valid])
                            .expect("UTF-8 validator returned an invalid prefix");
                        emit(text.to_owned());
                        offset += valid;
                    }
                    let Some(invalid) = error.error_len() else {
                        let tail = total - offset;
                        debug_assert!(tail <= UTF8_CARRY_MAX_BYTES);
                        if tail > UTF8_CARRY_MAX_BYTES {
                            self.reset();
                            return Err("native text event left an oversized UTF-8 carry".into());
                        }
                        self.carry[..tail].copy_from_slice(&joined[offset..total]);
                        self.len = tail;
                        return Ok(());
                    };
                    emit(REPLACEMENT.to_owned());
                    offset += invalid;
                }
            }
        }
        Ok(())
    }

    fn finish<F>(&mut self, emit: &mut F)
    where
        F: FnMut(String) + ?Sized,
    {
        if self.len != 0 {
            emit("\u{fffd}".to_owned());
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.len = 0;
    }
}

struct NativeAction {
    ticket: Ticket,
    flow: Option<Flow>,
    text: Utf8Stream,
    playback: u32,
    terminal: Option<(bool, u32)>,
    terminal_records: u32,
    cancelled: bool,
    #[cfg(test)]
    text_emissions: u32,
    #[cfg(test)]
    emitted_items: u32,
}

impl NativeAction {
    fn new(ticket: Ticket) -> Self {
        Self {
            ticket,
            flow: None,
            text: Utf8Stream::default(),
            playback: 0,
            terminal: None,
            terminal_records: 0,
            cancelled: false,
            #[cfg(test)]
            text_emissions: 0,
            #[cfg(test)]
            emitted_items: 0,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct TerminalProbe {
    flow: Flow,
    terminal_records: u32,
    text_emissions: u32,
    emitted_items: u32,
    playback_leases: u32,
    playback_retired: u32,
}

unsafe extern "C" fn on_event(context: *mut c_void, event: *const NativeEvent) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if context.is_null() || event.is_null() {
            return Err(STATUS_HOST_SINK);
        }
        // The native delivery kc_service invokes this callback serially. Its
        // retained context therefore owns the single producer endpoint and its
        // single-producer realtime notifier for the entire session lifetime.
        let sink = unsafe { &mut *(context.cast::<EventSink>()) };
        let event = unsafe { &*event };
        if event.size != std::mem::size_of::<NativeEvent>() as u32
            || event.abi_version != RUNTIME_ABI
            || event.payload_bytes as usize > TEXT_EVENT_MAX_BYTES
        {
            return Err(STATUS_HOST_SINK);
        }
        let bytes = if event.payload_bytes == 0 {
            &[][..]
        } else {
            if event.payload.is_null() {
                return Err(STATUS_HOST_SINK);
            }
            unsafe {
                std::slice::from_raw_parts(event.payload.cast::<u8>(), event.payload_bytes as usize)
            }
        };
        let reply = match event.kind {
            // Epoch/state records describe the session, not any action. Only
            // the terminal Turn carrying the submitted ticket can settle it.
            EVENT_STATE => None,
            EVENT_TEXT => Some(Reply::Text {
                flow: Flow::new(event)?,
                payload: TextPayload::new(bytes).ok_or(STATUS_HOST_SINK)?,
            }),
            EVENT_TURN => {
                if bytes.len() != std::mem::size_of::<TurnEvent>() {
                    return Err(STATUS_HOST_SINK);
                }
                let turn = unsafe { bytes.as_ptr().cast::<TurnEvent>().read_unaligned() };
                if turn.size != std::mem::size_of::<TurnEvent>() as u32
                    || turn.abi_version != RUNTIME_ABI
                {
                    return Err(STATUS_HOST_SINK);
                }
                Some(Reply::Turn {
                    flow: Flow::new(event)?,
                    status: event.status,
                    has_audio: event.flags & EVENT_HAS_AUDIO != 0,
                    truncated: event.flags & EVENT_TRUNCATED != 0,
                    playback_leases: turn.playback_leases,
                    emitted_items: turn.emitted_items,
                })
            }
            EVENT_PLAYBACK_READY => {
                if bytes.len() != std::mem::size_of::<PlaybackReadyEvent>() {
                    return Err(STATUS_HOST_SINK);
                }
                let ready = unsafe { bytes.as_ptr().cast::<PlaybackReadyEvent>().read_unaligned() };
                if ready.size != std::mem::size_of::<PlaybackReadyEvent>() as u32
                    || ready.abi_version != RUNTIME_ABI
                {
                    return Err(STATUS_HOST_SINK);
                }
                Some(Reply::PlaybackReady {
                    flow: Flow::new(event)?,
                    lease_id: ready.lease_id,
                    buffer_generation: ready.buffer_generation,
                })
            }
            EVENT_TURN_STARTED if bytes.is_empty() && event.status == 0 => {
                Some(Reply::TurnStarted {
                    flow: Flow::new(event)?,
                })
            }
            EVENT_ERROR => Some(Reply::Error {
                flow: Flow::new(event)?,
                status: event.status,
                payload: TextPayload::new(bytes).ok_or(STATUS_HOST_SINK)?,
            }),
            EVENT_STOPPED => Some(Reply::Stopped {
                flow: Flow::new(event)?,
                status: event.status,
            }),
            _ => return Err(STATUS_HOST_SINK),
        };
        let Some(reply) = reply else {
            return Ok(0);
        };
        if sink.replies.try_push(reply).is_err() {
            return Ok(STATUS_WOULD_BLOCK);
        }
        let Some(resume) = sink.resume.as_mut() else {
            // Setup may receive a native state record before the shared Rust
            // voice service installs its producer lease. The fixed record is
            // retained here and install_resume publishes the missing edge.
            return Ok(0);
        };
        resume.notify().map_err(|_| STATUS_HOST_SINK)?;
        Ok(0)
    });
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => STATUS_HOST_SINK,
    }
}

struct SessionControl(NonNull<SessionControlHandle>);

unsafe impl Send for SessionControl {}
unsafe impl Sync for SessionControl {}

impl SessionControl {
    fn interrupt(&self) -> Result<u64, String> {
        let mut epoch = 0;
        let code = unsafe { lfm_session_control_interrupt(self.0.as_ptr(), &mut epoch) };
        if code != 0 {
            return Err(format!(
                "native session interrupt failed with status {code}"
            ));
        }
        if epoch == 0 {
            return Err("native session interrupt returned an empty epoch".into());
        }
        Ok(epoch)
    }
}

impl Drop for SessionControl {
    fn drop(&mut self) {
        let code = unsafe { lfm_session_control_destroy(self.0.as_ptr()) };
        if code != 0 {
            eprintln!("[flashkern] native session control destroy failed with status {code}");
        }
    }
}

struct NativeCaptureSink {
    producer: NonNull<CaptureProducer>,
    rate: u32,
    max_callback_frames: u32,
}

unsafe impl Send for NativeCaptureSink {}

impl NativeCaptureSink {
    fn write<T>(&mut self, input: &[T], channels: usize, format: u32) -> CaptureWrite {
        if input.is_empty() {
            return CaptureWrite::default();
        }
        if channels == 0 {
            return CaptureWrite {
                dropped_frames: input.len(),
                ..CaptureWrite::default()
            };
        }
        let Ok(channels) = u32::try_from(channels) else {
            return CaptureWrite {
                dropped_frames: input.len(),
                ..CaptureWrite::default()
            };
        };
        let mut write = NativeCaptureWrite {
            size: std::mem::size_of::<NativeCaptureWrite>() as u32,
            abi_version: RUNTIME_ABI,
            ..NativeCaptureWrite::default()
        };
        let _ = unsafe {
            lfm_capture_producer_write_interleaved(
                self.producer.as_ptr(),
                input.as_ptr().cast(),
                input.len(),
                channels,
                self.rate,
                format,
                0,
                &mut write,
            )
        };
        CaptureWrite {
            admitted_frames: write.admitted_frames as usize,
            dropped_frames: write.dropped_frames as usize,
            gap_published: write.flags & CAPTURE_WRITE_GAP_PUBLISHED != 0,
        }
    }
}

impl CaptureSink for NativeCaptureSink {
    fn rate(&self) -> u32 {
        self.rate
    }

    fn max_callback_frames(&self) -> u32 {
        self.max_callback_frames
    }

    fn write_f32(&mut self, input: &[f32], channels: usize) -> CaptureWrite {
        self.write(input, channels, CAPTURE_INPUT_F32)
    }

    fn write_i16(&mut self, input: &[i16], channels: usize) -> CaptureWrite {
        self.write(input, channels, CAPTURE_INPUT_I16)
    }

    fn write_u16(&mut self, input: &[u16], channels: usize) -> CaptureWrite {
        self.write(input, channels, CAPTURE_INPUT_U16)
    }

    fn mute(&mut self, frames: usize, channels: usize) -> CaptureMute {
        if frames == 0 {
            return CaptureMute::default();
        }
        let (Ok(frames), Ok(channels)) = (u32::try_from(frames), u32::try_from(channels)) else {
            return CaptureMute {
                frames,
                published: false,
            };
        };
        let mut chunk = NativeCaptureChunk {
            size: std::mem::size_of::<NativeCaptureChunk>() as u32,
            abi_version: RUNTIME_ABI,
            ..NativeCaptureChunk::default()
        };
        let status = unsafe {
            lfm_capture_producer_publish_gap(
                self.producer.as_ptr(),
                frames,
                channels,
                CAPTURE_CHUNK_GAP | CAPTURE_CHUNK_MUTED,
                &mut chunk,
            )
        };
        CaptureMute {
            frames: frames as usize,
            published: status == 0,
        }
    }
}

impl Drop for NativeCaptureSink {
    fn drop(&mut self) {
        let status = unsafe { lfm_capture_producer_destroy(self.producer.as_ptr()) };
        if status != 0 {
            eprintln!("[flashkern] native capture sink retired late with status {status}");
        }
    }
}

struct ClaimedPlayback {
    session: u64,
    lease: PcmLease,
    count: usize,
    cursor: usize,
}

struct NativePlaybackSource {
    consumer: NonNull<PlaybackConsumer>,
    state: Arc<PlaybackState>,
    notify: RealtimeNotifier,
    notice: Option<PlaybackNotice>,
    current: Option<ClaimedPlayback>,
    result: Option<PlaybackResult>,
    rate: u32,
}

unsafe impl Send for NativePlaybackSource {}

impl NativePlaybackSource {
    fn active(&self) -> bool {
        self.state
            .active(self.result.is_some() || self.notice.is_some() || self.current.is_some())
    }

    fn audio_active(&self) -> bool {
        self.state
            .audio_active(self.notice.is_some() || self.current.is_some())
    }

    fn publish_result(&mut self, result: PlaybackResult) -> bool {
        if !self.state.done.try_push(result) {
            self.result = Some(result);
            return false;
        }
        // The record is already queued; the notify is its causal successor.
        // Closed admission means the owner is already retiring this device
        // endpoint. A hardware callback must never panic while that teardown
        // edge races it. Until the owner drains the record, `active()` also
        // exposes it as an independent continuation successor, so a rejected
        // notify cannot make callback-owned orchestration go dormant.
        self.notify.notify().is_ok()
    }

    fn flush_result(&mut self) -> bool {
        let Some(result) = self.result else {
            return true;
        };
        if !self.state.done.try_push(result) {
            return false;
        }
        self.result = None;
        self.notify.notify().is_ok()
    }

    fn finish_current(&mut self, status_code: i32) -> usize {
        let Some(current) = self.current.take() else {
            return 0;
        };
        let dropped = current.count.saturating_sub(current.cursor);
        let release =
            unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &current.lease) };
        let status = if status_code != 0 {
            status_code
        } else {
            release
        };
        let _ = self.publish_result(PlaybackResult {
            flow: Flow {
                session: current.session,
                epoch: current.lease.stream_epoch,
                ticket: current.lease.ticket,
            },
            status,
        });
        dropped
    }

    fn claim(&mut self, write: &mut PlaybackWrite) -> bool {
        if self.current.is_some() || self.result.is_some() {
            return self.current.is_some();
        }
        let Some(notice) = self.notice.take().or_else(|| self.state.ready.try_pop()) else {
            return false;
        };
        let mut lease = PcmLease::default();
        let claim = unsafe {
            lfm_playback_consumer_claim(
                self.consumer.as_ptr(),
                &notice.flow.ticket,
                notice.flow.epoch,
                notice.lease_id,
                notice.buffer_generation,
                &mut lease,
            )
        };
        if claim == STATUS_WOULD_BLOCK {
            self.notice = Some(notice);
            return false;
        }
        if claim == STATUS_STALE || claim == STATUS_CANCELLED {
            let _ = self.publish_result(PlaybackResult {
                flow: notice.flow,
                status: 0,
            });
            return false;
        }
        if claim != 0 {
            let _ = self.publish_result(PlaybackResult {
                flow: notice.flow,
                status: claim,
            });
            return false;
        }
        if lease.ticket != notice.flow.ticket
            || lease.stream_epoch != notice.flow.epoch
            || lease.lease_id != notice.lease_id
            || lease.buffer_generation != notice.buffer_generation
        {
            let _ = unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &lease) };
            let _ = self.publish_result(PlaybackResult {
                flow: notice.flow,
                status: STATUS_HOST_SINK,
            });
            return false;
        }
        let mut samples = std::ptr::null();
        let mut count = 0usize;
        let resolve = unsafe {
            lfm_playback_consumer_resolve(self.consumer.as_ptr(), &lease, &mut samples, &mut count)
        };
        if resolve != 0 || samples.is_null() {
            let _ = unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &lease) };
            let status = if resolve == STATUS_STALE || resolve == STATUS_CANCELLED {
                0
            } else {
                resolve
            };
            let _ = self.publish_result(PlaybackResult {
                flow: notice.flow,
                status,
            });
            return false;
        }
        write.claimed_samples = write.claimed_samples.saturating_add(count);
        self.current = Some(ClaimedPlayback {
            session: notice.flow.session,
            lease,
            count,
            cursor: 0,
        });
        true
    }

    fn resolve(&mut self) -> Option<*const f32> {
        let Some(current) = self.current.as_ref() else {
            return None;
        };
        let lease = current.lease;
        let expected = current.count;
        let mut samples = std::ptr::null();
        let mut count = 0usize;
        let status = unsafe {
            lfm_playback_consumer_resolve(self.consumer.as_ptr(), &lease, &mut samples, &mut count)
        };
        if status == 0 && !samples.is_null() && count == expected {
            return Some(samples);
        }
        let status = if status == STATUS_STALE || status == STATUS_CANCELLED {
            0
        } else if status != 0 {
            status
        } else {
            STATUS_HOST_SINK
        };
        self.finish_current(status);
        None
    }

    fn discard(&mut self, write: &mut PlaybackWrite) {
        if self.current.is_some() {
            write.dropped_samples = write.dropped_samples.saturating_add(self.finish_current(0));
        }
        while self.result.is_none() {
            if !self.claim(write) {
                break;
            }
            write.dropped_samples = write.dropped_samples.saturating_add(self.finish_current(0));
        }
    }

    fn write<T: Copy>(
        &mut self,
        output: &mut [T],
        channels: usize,
        flush: bool,
        silence: T,
        render: PlaybackRender<T>,
    ) -> PlaybackWrite {
        let mut write = PlaybackWrite::default();
        if channels == 0 || output.len() % channels != 0 {
            output.fill(silence);
            write.active = self.active();
            return write;
        }
        let Ok(native_channels) = u32::try_from(channels) else {
            output.fill(silence);
            write.active = self.active();
            return write;
        };
        let mut meter = PlaybackMeter::default();
        if unsafe { lfm_playback_meter_reset(&mut meter) } != 0 {
            std::process::abort();
        }
        if !self.flush_result() {
            output.fill(silence);
            write.active = true;
            if self.audio_active() {
                write.underrun_frames = output.len() / channels;
            }
            return write;
        }
        if flush {
            self.discard(&mut write);
            output.fill(silence);
            write.active = self.active();
            return write;
        }

        let frames = output.len() / channels;
        let mut frame = 0usize;
        while frame < frames {
            if self.current.is_none() && !self.claim(&mut write) {
                break;
            }
            let Some(samples) = self.resolve() else {
                if self.result.is_some() {
                    break;
                }
                continue;
            };
            let (source, count) = {
                let current = self
                    .current
                    .as_ref()
                    .expect("revalidated playback disappeared");
                (
                    unsafe { samples.add(current.cursor) },
                    (frames - frame).min(current.count - current.cursor),
                )
            };
            let offset = frame * channels;
            let status = unsafe {
                render(
                    source,
                    output.as_mut_ptr().add(offset),
                    count,
                    native_channels,
                    output.len() - offset,
                    &mut meter,
                )
            };
            if status != 0 {
                write.dropped_samples = write
                    .dropped_samples
                    .saturating_add(self.finish_current(STATUS_HOST_SINK));
                break;
            }
            let current = self
                .current
                .as_mut()
                .expect("rendered playback disappeared");
            current.cursor += count;
            frame += count;
            if current.cursor == current.count {
                self.finish_current(0);
                if self.result.is_some() {
                    break;
                }
            }
        }
        output[frame * channels..].fill(silence);
        write.played_frames = frame;
        write.rms = meter.rms;
        let audio_active = self.audio_active();
        write.active = self.active();
        if audio_active && frame < frames {
            write.underrun_frames = frames - frame;
        }
        write
    }
}

impl PlaybackSource for NativePlaybackSource {
    fn rate(&self) -> u32 {
        self.rate
    }

    fn write_f32(&mut self, output: &mut [f32], channels: usize, flush: bool) -> PlaybackWrite {
        self.write(output, channels, flush, 0.0, lfm_playback_render_f32)
    }

    fn write_i16(&mut self, output: &mut [i16], channels: usize, flush: bool) -> PlaybackWrite {
        self.write(output, channels, flush, 0, lfm_playback_render_i16)
    }

    fn write_u16(&mut self, output: &mut [u16], channels: usize, flush: bool) -> PlaybackWrite {
        self.write(
            output,
            channels,
            flush,
            u16::MAX / 2,
            lfm_playback_render_u16,
        )
    }
}

impl Drop for NativePlaybackSource {
    fn drop(&mut self) {
        if let Some(current) = self.current.take() {
            let _ =
                unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &current.lease) };
        }
        let mut notice = self.notice.take().or_else(|| self.state.ready.try_pop());
        while let Some(ready) = notice {
            let mut lease = PcmLease::default();
            let claim = unsafe {
                lfm_playback_consumer_claim(
                    self.consumer.as_ptr(),
                    &ready.flow.ticket,
                    ready.flow.epoch,
                    ready.lease_id,
                    ready.buffer_generation,
                    &mut lease,
                )
            };
            if claim == 0 {
                let _ = unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &lease) };
            }
            notice = self.state.ready.try_pop();
        }
        let destroy = unsafe { lfm_playback_consumer_destroy(self.consumer.as_ptr()) };
        if destroy != 0 {
            eprintln!("[flashkern] native playback consumer retired late with status {destroy}");
        }
    }
}

/// Turn-based `VoiceEngine` backed entirely by the native LFM2 session.
pub struct NativeLfm2VoiceEngine {
    _model: NativeVoiceModel,
    conversation: Option<ConversationOwner>,
    vault: Option<NativeConversationVault>,
    healthy: bool,
    session: NonNull<Session>,
    control: Option<SessionControl>,
    capture: Option<NativeCaptureSink>,
    capture_taken: bool,
    sink: Option<Box<EventSink>>,
    replies: Arc<ReplyRing>,
    active: Option<NativeAction>,
    playback: Arc<PlaybackState>,
    pending_playback: Option<PlaybackNotice>,
    playback_taken: bool,
    playback_rate: u32,
    session_id: Option<u64>,
    control_epoch: u64,
    started: bool,
    stopped: bool,
    stopped_flow: Option<Flow>,
    last_terminal: Option<Flow>,
    joined: bool,
    #[cfg(test)]
    terminal_probe: Option<TerminalProbe>,
}

unsafe impl Send for NativeLfm2VoiceEngine {}

impl NativeLfm2VoiceEngine {
    fn new(
        model: NativeVoiceModel,
        sampling: NativeVoiceSampling,
        vault: Option<NativeConversationVault>,
        capture_rate: u32,
        playback_rate: u32,
        capture_max_callback_frames: u32,
    ) -> Result<Self, String> {
        if capture_rate == 0 {
            return Err("native capture sample rate is zero".into());
        }
        if playback_rate == 0 {
            return Err("native playback sample rate is zero".into());
        }
        if capture_max_callback_frames == 0 {
            return Err("native capture callback bound is unknown".into());
        }
        let claim = ConversationClaim::new(&model, sampling, vault.clone())?;
        let replies = ReplyRing::new();
        let mut sink = Box::new(EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        });
        let callbacks = Callbacks {
            size: std::mem::size_of::<Callbacks>() as u32,
            abi_version: RUNTIME_ABI,
            context: (&mut *sink as *mut EventSink).cast(),
            on_event: Some(on_event),
        };
        let config = SessionConfig {
            size: std::mem::size_of::<SessionConfig>() as u32,
            abi_version: RUNTIME_ABI,
            session_id: 0,
            playback_slots: PLAYBACK_SLOTS,
            capture_max_callback_frames,
            // Zero delegates model/codec/rate geometry to native readiness.
            // Rust must not encode Mimi's frame capacity as model knowledge.
            playback_frames_per_slot: 0,
            pcm_channels: 1,
            capture_sample_rate: capture_rate,
            playback_sample_rate: playback_rate,
            command_capacity: 8,
            max_new_tokens: sampling.max_new_tokens,
            flags: 0,
            reserved: [0; 4],
        };
        let mut session = std::ptr::null_mut();
        status(
            unsafe {
                lfm_session_create(
                    model.0.runtime.as_ptr(),
                    model.0.model.as_ptr(),
                    claim.conversation.as_ref().expect("claim").pointer.as_ptr(),
                    &config,
                    &callbacks,
                    &mut session,
                )
            },
            "create native voice session",
        )?;
        let session = NonNull::new(session).ok_or("native session returned a null handle")?;
        let mut control = std::ptr::null_mut();
        if let Err(error) = status(
            unsafe { lfm_session_control_create(session.as_ptr(), &mut control) },
            "create native session control",
        ) {
            retire_unstarted_session(session);
            return Err(error);
        }
        let Some(control) = NonNull::new(control) else {
            retire_unstarted_session(session);
            return Err("native session control returned a null handle".into());
        };
        let control = SessionControl(control);
        let mut producer = std::ptr::null_mut();
        if let Err(error) = status(
            unsafe { lfm_capture_chunk_producer_create(session.as_ptr(), 1, 0, &mut producer) },
            "create native capture sink",
        ) {
            drop(control);
            retire_unstarted_session(session);
            return Err(error);
        }
        let Some(producer) = NonNull::new(producer) else {
            drop(control);
            retire_unstarted_session(session);
            return Err("native capture sink returned a null handle".into());
        };
        let capture = NativeCaptureSink {
            producer,
            rate: capture_rate,
            max_callback_frames: capture_max_callback_frames,
        };
        let playback = PlaybackState::new();
        Ok(Self {
            _model: model,
            conversation: Some(claim.into_conversation()),
            vault,
            healthy: true,
            session,
            control: Some(control),
            capture: Some(capture),
            capture_taken: false,
            sink: Some(sink),
            replies,
            active: None,
            playback,
            pending_playback: None,
            playback_taken: false,
            playback_rate,
            session_id: None,
            control_epoch: 0,
            started: false,
            stopped: false,
            stopped_flow: None,
            last_terminal: None,
            joined: false,
            #[cfg(test)]
            terminal_probe: None,
        })
    }

    #[cfg(test)]
    fn record_terminal(&mut self, action: &NativeAction) {
        let flow = action
            .flow
            .expect("completed native action has no correlated flow");
        self.terminal_probe = Some(TerminalProbe {
            flow,
            terminal_records: action.terminal_records,
            text_emissions: action.text_emissions,
            emitted_items: action.emitted_items,
            playback_leases: action.terminal.map(|terminal| terminal.1).unwrap_or(0),
            playback_retired: action.playback,
        });
    }

    fn accept_flow(&mut self, flow: Flow) -> Result<(), String> {
        if let Some(session) = self.session_id {
            if session != flow.session {
                return Err(format!(
                    "native event crossed session identity (expected={session}, received={})",
                    flow.session
                ));
            }
            return Ok(());
        }
        self.session_id = Some(flow.session);
        Ok(())
    }

    fn bind_action(&mut self, flow: Flow) -> Result<&mut NativeAction, String> {
        self.accept_flow(flow)?;
        if flow.ticket.kind != TICKET_TURN {
            return Err(format!(
                "native action record carried ticket kind {} instead of {TICKET_TURN}",
                flow.ticket.kind
            ));
        }
        let action = self
            .active
            .as_mut()
            .ok_or("native action record arrived without an active action")?;
        if action.ticket != flow.ticket {
            return Err(format!(
                "native action record changed ticket (expected={:?}, received={:?})",
                action.ticket, flow.ticket
            ));
        }
        if let Some(bound) = action.flow {
            if bound != flow {
                return Err(format!(
                    "native action record changed flow (expected={bound:?}, received={flow:?})"
                ));
            }
        } else {
            action.flow = Some(flow);
        }
        Ok(action)
    }

    fn install_resume(&mut self, resume: RealtimeNotifier) -> Result<(), String> {
        if self.started {
            return Err("native voice event continuation is already mounted".into());
        }
        let sink = self
            .sink
            .as_mut()
            .ok_or("native voice event sink is unavailable")?;
        if sink.resume.replace(resume).is_some() {
            return Err("native voice event producer already owns a resume lease".into());
        }
        status(
            unsafe { lfm_session_start(self.session.as_ptr()) },
            "start native voice session",
        )?;
        self.started = true;
        if !self.replies.is_empty() {
            sink.resume
                .as_mut()
                .expect("installed native event resume lease disappeared")
                .notify()
                .map_err(|code| format!("resume native event continuation failed: {code}"))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn retire_resume(&mut self) {
        if let Some(sink) = self.sink.as_mut() {
            sink.resume.take();
        }
    }

    fn begin_ticket(&mut self, ticket: Ticket) -> Result<bool, String> {
        if !self.started {
            return Err("native voice event continuation is not mounted".into());
        }
        if self.active.is_some() {
            return Ok(false);
        }
        self.active = Some(NativeAction::new(ticket));
        Ok(true)
    }

    fn queue_playback(&mut self, notice: PlaybackNotice) -> bool {
        if self.playback.ready.try_push(notice) {
            return true;
        }
        self.pending_playback = Some(notice);
        false
    }

    fn drain_playback_results(&mut self) -> Result<usize, String> {
        let mut drained = 0usize;
        while drained < REPLY_CAPACITY {
            let Some(done) = self.playback.done.try_pop() else {
                break;
            };
            drained += 1;
            if done.status != 0 && done.status != STATUS_STALE && done.status != STATUS_CANCELLED {
                return Err(format!(
                    "native playback callback failed with status {}",
                    done.status
                ));
            }
            let Some(action) = self.active.as_mut() else {
                return Err("native playback retired without an active action".into());
            };
            if action.flow != Some(done.flow) {
                return Err(format!(
                    "native playback retirement changed flow (expected={:?}, received={:?})",
                    action.flow, done.flow
                ));
            }
            action.playback = action.playback.saturating_add(1);
        }
        Ok(drained)
    }

    /// Submit one UTF-8 user turn and return after admission. Native completion
    /// resumes the retained Rust voice continuation through the installed edge.
    pub fn begin_text(&mut self, text: &str) -> Result<bool, String> {
        if text.is_empty() || text.len() > 2_048 {
            return Err("native typed input must contain 1..=2048 UTF-8 bytes".into());
        }
        if self.active.is_some() {
            return Ok(false);
        }
        let mut ticket = Ticket::default();
        let submit = unsafe {
            lfm_session_submit_text(
                self.session.as_ptr(),
                text.as_ptr().cast(),
                text.len(),
                &mut ticket,
            )
        };
        if submit == STATUS_WOULD_BLOCK || submit == STATUS_STALE || submit == STATUS_CANCELLED {
            return Ok(false);
        }
        status(submit, "submit native typed input")?;
        self.begin_ticket(ticket)
    }

    fn drain_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        if let Some(action) = self.active.as_mut() {
            if cancel.load(Ordering::Acquire) {
                action.cancelled = true;
                action.text.reset();
            }
        }

        self.drain_playback_results()?;
        let complete = self.active.as_ref().is_some_and(|action| {
            action
                .terminal
                .is_some_and(|(has_audio, playback)| !has_audio || action.playback >= playback)
        });
        if complete {
            let action = self
                .active
                .take()
                .expect("completed native action disappeared");
            self.last_terminal = action.flow;
            #[cfg(test)]
            self.record_terminal(&action);
            if !action.cancelled {
                emit(VoiceEvent::TurnComplete);
            } else {
                emit(VoiceEvent::Interrupted);
            }
            return Ok(EngineProgress::Complete);
        }
        if let Some(notice) = self.pending_playback {
            if !self.playback.ready.try_push(notice) {
                return Ok(EngineProgress::Dormant);
            }
            self.pending_playback = None;
        }

        let mut drained = 0usize;
        let mut result = Ok(EngineProgress::Dormant);
        while drained < REPLY_CAPACITY {
            let Some(reply) = self.replies.try_pop() else {
                break;
            };
            drained += 1;
            match reply {
                Reply::TurnStarted { flow } => {
                    if self.active.is_none() {
                        self.active = Some(NativeAction::new(flow.ticket));
                        emit(VoiceEvent::TurnStarted);
                    }
                    if let Err(error) = self.bind_action(flow) {
                        result = Err(error);
                        break;
                    }
                }
                Reply::Text { flow, payload } => match self.bind_action(flow) {
                    Ok(action) if !action.cancelled => {
                        #[cfg(test)]
                        {
                            action.text_emissions = action.text_emissions.saturating_add(1);
                        }
                        if let Err(error) = action.text.push(payload.as_bytes(), &mut |piece| {
                            emit(VoiceEvent::Text(piece))
                        }) {
                            result = Err(error);
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        result = Err(error);
                        break;
                    }
                },
                Reply::PlaybackReady {
                    flow,
                    lease_id,
                    buffer_generation,
                } => {
                    if let Err(error) = self.bind_action(flow) {
                        result = Err(error);
                        break;
                    }
                    if !self.queue_playback(PlaybackNotice {
                        flow,
                        lease_id,
                        buffer_generation,
                    }) {
                        break;
                    }
                }
                Reply::Turn {
                    flow,
                    status,
                    has_audio,
                    truncated,
                    playback_leases,
                    emitted_items,
                } => {
                    #[cfg(not(test))]
                    let _ = emitted_items;
                    let action = match self.bind_action(flow) {
                        Ok(action) => action,
                        Err(error) => {
                            if self.last_terminal == Some(flow) {
                                result =
                                    Err("native action published a duplicate terminal record"
                                        .into());
                            } else {
                                result = Err(error);
                            }
                            break;
                        }
                    };
                    action.terminal_records = action.terminal_records.saturating_add(1);
                    if action.terminal_records != 1 {
                        result =
                            Err("native action published more than one terminal record".into());
                        break;
                    }
                    if status == STATUS_STALE || status == STATUS_CANCELLED {
                        action.cancelled = true;
                        action.text.reset();
                        action.terminal = Some((false, 0));
                    } else if status != 0 {
                        result = Err(format!("native turn failed with status {status}"));
                        break;
                    } else {
                        #[cfg(test)]
                        {
                            action.emitted_items = emitted_items;
                        }
                        if !action.cancelled {
                            action
                                .text
                                .finish(&mut |piece| emit(VoiceEvent::Text(piece)));
                        }
                        if truncated {
                            crate::vtrace!("native turn reached max_new_tokens");
                        }
                        action.terminal = Some((has_audio, playback_leases));
                    }
                }
                Reply::Error {
                    flow,
                    status,
                    payload,
                } => {
                    let accepted = if flow.ticket.kind == TICKET_CONTROL {
                        self.accept_flow(flow)
                    } else {
                        self.bind_action(flow).map(|_| ())
                    };
                    if let Err(error) = accepted {
                        result = Err(error);
                        break;
                    }
                    result = Err(format!(
                        "{} (native status {status})",
                        String::from_utf8_lossy(payload.as_bytes())
                    ));
                    break;
                }
                Reply::Stopped { flow, status } => {
                    if let Err(error) = self.accept_flow(flow) {
                        result = Err(error);
                        break;
                    }
                    if flow.ticket.kind != TICKET_SESSION {
                        result = Err(format!(
                            "native STOPPED record carried ticket kind {} instead of {TICKET_SESSION}",
                            flow.ticket.kind
                        ));
                        break;
                    }
                    self.stopped = true;
                    self.stopped_flow = Some(flow);
                    if status != 0 {
                        result = Err(format!("native voice session stopped with status {status}"));
                        break;
                    }
                    if let Some(mut action) = self.active.take() {
                        action.cancelled = true;
                        action.text.reset();
                        emit(VoiceEvent::Interrupted);
                    }
                    result = Ok(EngineProgress::Stopped);
                    break;
                }
            }

            let complete = self.active.as_ref().is_some_and(|action| {
                action
                    .terminal
                    .is_some_and(|(has_audio, playback)| !has_audio || action.playback >= playback)
            });
            if complete {
                let action = self
                    .active
                    .take()
                    .expect("completed native action disappeared");
                self.last_terminal = action.flow;
                #[cfg(test)]
                self.record_terminal(&action);
                if !action.cancelled {
                    emit(VoiceEvent::TurnComplete);
                } else {
                    emit(VoiceEvent::Interrupted);
                }
                result = Ok(EngineProgress::Complete);
                break;
            }
        }

        if drained != 0 {
            let capacity = unsafe { lfm_session_host_capacity(self.session.as_ptr()) };
            if capacity != 0 && capacity != STATUS_CANCELLED && result.is_ok() {
                result = Err(format!(
                    "resume native host-capacity edge failed with status {capacity}"
                ));
            }
        }
        if result.is_err() {
            self.healthy = false;
            if let Some(action) = self.active.as_mut() {
                action.text.reset();
            }
            return result;
        }
        if matches!(
            result,
            Ok(EngineProgress::Complete | EngineProgress::Stopped)
        ) {
            return result;
        }
        if self.pending_playback.is_none()
            && (!self.replies.is_empty() || !self.playback.done.is_empty())
        {
            return Ok(EngineProgress::Continue);
        }
        Ok(EngineProgress::Dormant)
    }
}

impl VoiceEngine for NativeLfm2VoiceEngine {
    fn take_capture_sink(&mut self) -> Result<Option<Box<dyn CaptureSink>>, String> {
        if self.capture_taken {
            return Err("native capture sink was already transferred".into());
        }
        let Some(capture) = self.capture.take() else {
            return Ok(None);
        };
        self.capture_taken = true;
        Ok(Some(Box::new(capture)))
    }

    fn take_playback_source(
        &mut self,
        notify: RealtimeNotifier,
    ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
        if self.playback_taken {
            return Err("native playback consumer was already transferred".into());
        }
        let mut consumer = std::ptr::null_mut();
        status(
            unsafe { lfm_playback_consumer_create(self.session.as_ptr(), &mut consumer) },
            "create native playback consumer",
        )?;
        let consumer =
            NonNull::new(consumer).ok_or("native playback consumer returned a null handle")?;
        self.playback_taken = true;
        Ok(Some(Box::new(NativePlaybackSource {
            consumer,
            state: self.playback.clone(),
            notify,
            notice: None,
            current: None,
            result: None,
            rate: self.playback_rate,
        })))
    }

    fn mount_events(&mut self, notify: RealtimeNotifier) -> Result<(), String> {
        self.install_resume(notify)
    }

    fn advance_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        self.drain_events(cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<u64, String> {
        let control = self
            .control
            .as_ref()
            .ok_or("native session control edge is already retired")?;
        let epoch = control.interrupt()?;
        if epoch <= self.control_epoch {
            self.healthy = false;
            return Err(format!(
                "native session interrupt epoch did not advance (previous={}, received={epoch})",
                self.control_epoch
            ));
        }
        self.control_epoch = epoch;
        Ok(epoch)
    }

    fn request_stop(&mut self) {
        unsafe { lfm_session_request_stop(self.session.as_ptr()) };
    }

    fn stop_session(&mut self) -> Result<(), String> {
        if self.joined {
            return Ok(());
        }
        // Retire all Rust-owned native endpoints first. Session stop/join is a
        // terminal state-machine edge, not a way to cancel live callback leases.
        self.control.take();
        self.capture.take();
        self.request_stop();
        let join = unsafe { lfm_session_join(self.session.as_ptr()) };
        if join == 0 {
            self.joined = true;
            return Ok(());
        }
        self.healthy = false;
        Err(format!(
            "join native voice session failed with status {join}"
        ))
    }
}

impl Drop for NativeLfm2VoiceEngine {
    fn drop(&mut self) {
        if let Err(error) = self.stop_session() {
            eprintln!("[flashkern] {error}");
        }
        let destroy = unsafe { lfm_session_destroy(self.session.as_ptr()) };
        if destroy != 0 {
            eprintln!("[flashkern] native voice session destroy refused with status {destroy}");
            /* A refused destroy means native may still legally invoke the
             * callback and dereference its model/conversation. Keep the whole
             * callback lineage alive rather than freeing one component under
             * a leaked native session. This is a terminal containment path;
             * healthy teardown must never reach it. */
            if let Some(sink) = self.sink.take() {
                std::mem::forget(sink);
            }
            if let Some(conversation) = self.conversation.take() {
                std::mem::forget(conversation);
            }
            std::mem::forget(self._model.clone());
            return;
        }
        self.sink.take();
        let Some(vault) = self.vault.as_ref() else {
            return;
        };
        if !self.healthy {
            /* Begin/prefill and recurrence are not rollback-transactional. A
             * terminal numerical/session error may therefore leave native
             * cache planes partially advanced. Close that conversation after
             * session teardown instead of putting it back in the vault. */
            self.conversation.take();
        }
        let mut state = vault.0.lock().expect("conversation vault mutex poisoned");
        state.conversation = self.healthy.then(|| {
            self.conversation
                .take()
                .expect("healthy native engine lost its conversation")
        });
        state.claimed = false;
    }
}

fn status(code: i32, operation: &str) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    if code == STATUS_BUSY {
        return Err(format!("{operation}: native owner is busy"));
    }
    Err(format!("{operation} failed with native status {code}"))
}

#[cfg(all(test, target_os = "macos"))]
mod real_checkpoint_gate {
    use super::*;
    use kcoro_sys::{Runtime as CoroutineRuntime, ServiceOutcome};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    const OUTPUT_RATE: u32 = 24_000;
    const DEVICE_FRAMES: usize = 512;
    const CAPTURE_CHUNK_MS: usize = 20;
    const CAPTURE_TAIL_MS: usize = 1_000;
    const JOINED: u32 = 4;

    #[repr(C)]
    #[derive(Clone, Copy)]
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

    impl Default for SessionSnapshot {
        fn default() -> Self {
            Self {
                size: std::mem::size_of::<Self>() as u32,
                abi_version: RUNTIME_ABI,
                // The native snapshot is an integer-only output record.
                ..unsafe { std::mem::zeroed() }
            }
        }
    }

    unsafe extern "C" {
        fn lfm_session_snapshot(session: *const Session, out: *mut SessionSnapshot) -> i32;
    }

    struct Watchdog(*mut c_void);

    impl Watchdog {
        fn arm(timeout: Duration) -> Result<Self, String> {
            unsafe extern "C" {
                static _dispatch_source_type_timer: u8;
                fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
                fn dispatch_source_create(
                    kind: *const u8,
                    handle: usize,
                    mask: usize,
                    queue: *mut c_void,
                ) -> *mut c_void;
                fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
                fn dispatch_source_set_event_handler_f(
                    source: *mut c_void,
                    handler: unsafe extern "C" fn(*mut c_void),
                );
                fn dispatch_source_set_timer(
                    source: *mut c_void,
                    start: u64,
                    interval: u64,
                    leeway: u64,
                );
                fn dispatch_time(when: u64, delta: i64) -> u64;
                fn dispatch_resume(object: *mut c_void);
            }

            unsafe extern "C" fn expire(_: *mut c_void) {
                eprintln!("native real-checkpoint truth gate exceeded its OS watchdog");
                std::process::abort();
            }

            let nanos = i64::try_from(timeout.as_nanos())
                .map_err(|_| "native gate watchdog duration is too large".to_string())?;
            let queue = unsafe { dispatch_get_global_queue(0, 0) };
            if queue.is_null() {
                return Err("GCD did not expose a watchdog queue".into());
            }
            let source = unsafe {
                dispatch_source_create(std::ptr::addr_of!(_dispatch_source_type_timer), 0, 0, queue)
            };
            if source.is_null() {
                return Err("GCD did not create the native gate watchdog".into());
            }
            unsafe {
                dispatch_set_context(source, std::ptr::null_mut());
                dispatch_source_set_event_handler_f(source, expire);
                dispatch_source_set_timer(source, dispatch_time(0, nanos), u64::MAX, 0);
                dispatch_resume(source);
            }
            Ok(Self(source))
        }
    }

    impl Drop for Watchdog {
        fn drop(&mut self) {
            unsafe extern "C" {
                fn dispatch_source_cancel(source: *mut c_void);
            }
            // The source is intentionally process-retained after asynchronous
            // cancellation. Its handler has no borrowed context to outlive.
            unsafe { dispatch_source_cancel(self.0) };
        }
    }

    struct Wave {
        samples: Vec<i16>,
        silence: Vec<i16>,
        channels: usize,
        rate: u32,
    }

    fn read_wave(path: &Path) -> Result<Wave, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read question fixture {}: {error}", path.display()))?;
        if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(format!("{} is not a RIFF/WAVE file", path.display()));
        }
        let mut cursor = 12usize;
        let mut format = None;
        let mut data = None;
        while cursor + 8 <= bytes.len() {
            let size = u32::from_le_bytes(
                bytes[cursor + 4..cursor + 8]
                    .try_into()
                    .expect("four-byte WAV chunk length"),
            ) as usize;
            let body = cursor + 8;
            let end = body
                .checked_add(size)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| format!("{} has a truncated WAV chunk", path.display()))?;
            if &bytes[cursor..cursor + 4] == b"fmt " && size >= 16 {
                format = Some((
                    u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap()),
                    u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()),
                    u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap()),
                ));
            }
            if &bytes[cursor..cursor + 4] == b"data" {
                data = Some(&bytes[body..end]);
            }
            cursor = end + (size & 1);
        }
        let (encoding, channels, rate, bits) =
            format.ok_or_else(|| format!("{} has no WAV fmt chunk", path.display()))?;
        if encoding != 1 || bits != 16 || channels == 0 || rate == 0 {
            return Err(format!(
                "{} must be nonempty PCM16 WAV (format={encoding}, channels={channels}, rate={rate}, bits={bits})",
                path.display()
            ));
        }
        let data = data.ok_or_else(|| format!("{} has no WAV data chunk", path.display()))?;
        if data.len() % (channels as usize * 2) != 0 {
            return Err(format!("{} has a partial PCM frame", path.display()));
        }
        let channels = channels as usize;
        let samples = data
            .chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
            .collect::<Vec<_>>();
        if samples.is_empty() {
            return Err(format!("{} has no PCM", path.display()));
        }
        Ok(Wave {
            samples,
            silence: vec![0; DEVICE_FRAMES * channels],
            channels,
            rate,
        })
    }

    fn write_wave(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
        let bytes = u32::try_from(samples.len().saturating_mul(2))
            .map_err(|_| "native gate WAV exceeds the RIFF size bound".to_string())?;
        let mut wave = Vec::with_capacity(44 + bytes as usize);
        wave.extend_from_slice(b"RIFF");
        wave.extend_from_slice(&(36 + bytes).to_le_bytes());
        wave.extend_from_slice(b"WAVEfmt ");
        wave.extend_from_slice(&16u32.to_le_bytes());
        wave.extend_from_slice(&1u16.to_le_bytes());
        wave.extend_from_slice(&1u16.to_le_bytes());
        wave.extend_from_slice(&rate.to_le_bytes());
        wave.extend_from_slice(&(rate * 2).to_le_bytes());
        wave.extend_from_slice(&2u16.to_le_bytes());
        wave.extend_from_slice(&16u16.to_le_bytes());
        wave.extend_from_slice(b"data");
        wave.extend_from_slice(&bytes.to_le_bytes());
        for sample in samples {
            wave.extend_from_slice(
                &((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes(),
            );
        }
        std::fs::write(path, wave)
            .map_err(|error| format!("write native gate WAV {}: {error}", path.display()))
    }

    #[derive(Debug)]
    struct Turn {
        transcript: String,
        pcm: Vec<f32>,
        rate: u32,
        flow: Flow,
        claimed: usize,
        played: usize,
        blocks: usize,
        terminals: usize,
    }

    impl Turn {
        fn rms(&self) -> f64 {
            (self
                .pcm
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>()
                / self.pcm.len().max(1) as f64)
                .sqrt()
        }

        fn transcript_digest(&self) -> String {
            let mut hash = Sha256::new();
            hash.update(self.transcript.as_bytes());
            format!("{:x}", hash.finalize())
        }

        fn pcm_digest(&self) -> String {
            let mut hash = Sha256::new();
            for sample in &self.pcm {
                hash.update(sample.to_bits().to_le_bytes());
            }
            format!("{:x}", hash.finalize())
        }
    }

    struct Report {
        turns: Vec<Turn>,
        snapshot: SessionSnapshot,
        stopped: Flow,
    }

    #[derive(Debug)]
    enum Phase {
        Init,
        Generate(usize),
        Capture {
            turn: usize,
            cursor: usize,
            silence: usize,
        },
        AwaitAudio(usize),
        Stopping,
        Done,
    }

    struct GateExit {
        engine: NativeLfm2VoiceEngine,
        turns: Vec<Turn>,
        error: Option<String>,
    }

    type ResultCell = Arc<Mutex<Option<GateExit>>>;

    struct Gate {
        source: Option<Box<dyn PlaybackSource>>,
        events: Option<RealtimeNotifier>,
        capture: Option<Box<dyn CaptureSink>>,
        engine: Option<NativeLfm2VoiceEngine>,
        fixture: Wave,
        phase: Phase,
        turns: Vec<Turn>,
        transcript: String,
        pcm: Vec<f32>,
        ticket: Ticket,
        terminals: usize,
        claimed: usize,
        played: usize,
        blocks: usize,
        error: Option<String>,
        result: ResultCell,
        trace: bool,
    }

    impl Gate {
        fn step(&mut self) -> ServiceOutcome {
            match self.advance() {
                Ok(outcome) => outcome,
                Err(error) => self.begin_stop(Some(error)),
            }
        }

        fn engine(&mut self) -> Result<&mut NativeLfm2VoiceEngine, String> {
            self.engine
                .as_mut()
                .ok_or_else(|| "native truth-gate engine already retired".to_string())
        }

        fn advance(&mut self) -> Result<ServiceOutcome, String> {
            if matches!(self.phase, Phase::Init) {
                if self.trace {
                    eprintln!("[native-e2e] gate init: submit typed turn");
                }
                let events = self
                    .events
                    .take()
                    .ok_or("native truth gate lost its event producer edge")?;
                self.engine()?.mount_events(events)?;
                if !self
                    .engine()?
                    .begin_text("Answer briefly, then speak the same answer aloud.")?
                {
                    return Err("native typed turn was not admitted".into());
                }
                self.ticket = self
                    .engine()?
                    .active
                    .as_ref()
                    .ok_or("native typed turn has no correlated action")?
                    .ticket;
                self.check_ticket(self.ticket)?;
                self.phase = Phase::Generate(0);
                return Ok(ServiceOutcome::Dormant);
            }

            if matches!(self.phase, Phase::Stopping) {
                return self.settle();
            }

            let turn = match self.phase {
                Phase::Generate(turn) => turn,
                Phase::AwaitAudio(turn) => turn,
                Phase::Capture { turn, .. } => turn,
                Phase::Done => return Ok(ServiceOutcome::Complete),
                Phase::Init | Phase::Stopping => unreachable!(),
            };
            let mut events = Vec::new();
            let cancel = AtomicBool::new(false);
            let progress = self
                .engine()?
                .advance_events(&cancel, &mut |event| events.push(event))?;
            if self.trace && (!events.is_empty() || progress != EngineProgress::Dormant) {
                eprintln!(
                    "[native-e2e] gate phase={:?} progress={progress:?} events={}",
                    self.phase,
                    events.len()
                );
            }
            let mut terminals = 0usize;
            for event in events {
                match event {
                    VoiceEvent::TurnStarted => {
                        let audio_turn = match self.phase {
                            Phase::AwaitAudio(audio_turn)
                            | Phase::Capture {
                                turn: audio_turn, ..
                            } => audio_turn,
                            _ => {
                                return Err(
                                    "native truth gate received an unexpected turn-start".into()
                                )
                            }
                        };
                        let ticket = self
                            .engine()?
                            .active
                            .as_ref()
                            .ok_or("native audio turn-start has no correlated action")?
                            .ticket;
                        self.check_ticket(ticket)?;
                        self.ticket = ticket;
                        self.phase = Phase::Generate(audio_turn);
                    }
                    VoiceEvent::Text(text) => self.transcript.push_str(&text),
                    VoiceEvent::TurnComplete => terminals += 1,
                    VoiceEvent::Interrupted => {
                        return Err(format!(
                            "native truth gate turn {} was interrupted",
                            turn + 1
                        ))
                    }
                    VoiceEvent::Error(error) => return Err(error),
                }
            }
            self.terminals = self.terminals.saturating_add(terminals);
            if matches!(self.phase, Phase::Capture { .. }) {
                if self.terminals != 0 || progress == EngineProgress::Complete {
                    return Err("native truth gate completed an action before turn-start".into());
                }
                return self.capture();
            }
            if self.terminals > 1 {
                return Err(format!(
                    "native truth gate turn {} published {} terminal events",
                    turn + 1,
                    self.terminals
                ));
            }
            if self.terminals == 1 {
                if progress != EngineProgress::Complete {
                    return Err(
                        "native terminal event did not complete its correlated action".into(),
                    );
                }
                self.finish_turn(turn)?;
                if self.trace {
                    eprintln!("[native-e2e] gate completed turn {}", turn + 1);
                }
                if turn == 2 {
                    return Ok(self.begin_stop(None));
                }
                // Both audio submissions preserve the configured capture
                // clock. The second submission reuses the real fixture on the
                // same conversation to exercise retained suffix recurrence.
                if self.fixture.samples.is_empty() {
                    return Err("native retained-context input fixture is empty".into());
                }
                self.phase = Phase::Capture {
                    turn: turn + 1,
                    cursor: 0,
                    silence: 0,
                };
                if self.trace {
                    eprintln!("[native-e2e] gate begin capture turn {}", turn + 2);
                }
                return Ok(ServiceOutcome::Continue);
            }

            let mut block = [0.0f32; DEVICE_FRAMES];
            let write = self
                .source
                .as_mut()
                .ok_or("native playback source retired during generation")?
                .write_f32(&mut block, 1, false);
            if write.dropped_samples != 0 || write.underrun_frames != 0 {
                return Err(format!(
                    "native playback was discontinuous (dropped={}, underrun={})",
                    write.dropped_samples, write.underrun_frames
                ));
            }
            if write.played_frames > block.len() {
                return Err("native playback overran its device block".into());
            }
            if block[..write.played_frames]
                .iter()
                .any(|sample| !sample.is_finite())
            {
                return Err("native playback published non-finite PCM".into());
            }
            self.claimed = self.claimed.saturating_add(write.claimed_samples);
            self.played = self.played.saturating_add(write.played_frames);
            if write.played_frames != 0 {
                self.blocks += 1;
                self.pcm.extend_from_slice(&block[..write.played_frames]);
            }
            if write.active || progress == EngineProgress::Continue {
                return Ok(ServiceOutcome::Continue);
            }
            Ok(ServiceOutcome::Dormant)
        }

        fn capture(&mut self) -> Result<ServiceOutcome, String> {
            if let Phase::Capture {
                turn,
                cursor,
                silence,
            } = &mut self.phase
            {
                let chunk = (self.fixture.rate as usize * CAPTURE_CHUNK_MS / 1_000).max(1);
                let frames = self.fixture.samples.len() / self.fixture.channels;
                if *cursor != frames {
                    let end = (*cursor + chunk).min(frames);
                    let begin = *cursor * self.fixture.channels;
                    let limit = end * self.fixture.channels;
                    let write = self
                        .capture
                        .as_mut()
                        .ok_or("native capture sink retired before the audio turns")?
                        .write_i16(&self.fixture.samples[begin..limit], self.fixture.channels);
                    if write.admitted_frames != end - *cursor
                        || write.dropped_frames != 0
                        || write.gap_published
                    {
                        return Err("native capture sink dropped a complete callback block".into());
                    }
                    *cursor = end;
                    return Ok(ServiceOutcome::Continue);
                }

                /* The test supplies acoustic silence exactly as a hardware
                 * callback would. It does not declare a turn boundary: the
                 * native detector and correlated pause deadline alone own
                 * that decision. */
                let target = (self.fixture.rate as usize * CAPTURE_TAIL_MS / 1_000).max(1);
                if *silence != target {
                    let frames = (target - *silence).min(DEVICE_FRAMES);
                    let write = self
                        .capture
                        .as_mut()
                        .ok_or("native capture sink retired during acoustic silence")?
                        .write_i16(
                            &self.fixture.silence[..frames * self.fixture.channels],
                            self.fixture.channels,
                        );
                    if write.admitted_frames != frames
                        || write.dropped_frames != 0
                        || write.gap_published
                    {
                        return Err("native capture sink dropped a silent callback block".into());
                    }
                    *silence += frames;
                    return Ok(ServiceOutcome::Continue);
                }
                let turn = *turn;
                let silence = *silence;
                self.phase = Phase::AwaitAudio(turn);
                if self.trace {
                    eprintln!(
                        "[native-e2e] gate capture turn {} dormant after {} fixture frames and {} silence frames",
                        turn + 1,
                        frames,
                        silence
                    );
                }
                // No test-owned wake is published here. The native detector's
                // correlated deadline child must publish TurnStarted and make
                // this suspended continuation runnable.
                return Ok(ServiceOutcome::Dormant);
            }
            Err("native capture phase lost its callback cursor".into())
        }

        fn check_ticket(&self, ticket: Ticket) -> Result<(), String> {
            if ticket.runtime_epoch == 0
                || ticket.sequence == 0
                || ticket.generation == 0
                || ticket.kind == 0
            {
                return Err(format!(
                    "native action returned an empty ticket: {ticket:?}"
                ));
            }
            if self
                .turns
                .last()
                .is_some_and(|turn| turn.flow.ticket.sequence >= ticket.sequence)
            {
                return Err("native action ticket sequence did not advance".into());
            }
            Ok(())
        }

        fn finish_turn(&mut self, turn: usize) -> Result<(), String> {
            let probe = self
                .engine()?
                .terminal_probe
                .take()
                .ok_or("native terminal has no correlated decoder evidence")?;
            let audio = probe
                .emitted_items
                .checked_sub(probe.text_emissions)
                .ok_or("native terminal text accounting exceeds emitted items")?;
            if probe.flow.ticket != self.ticket
                || probe.terminal_records != 1
                || probe.playback_retired != probe.playback_leases
                || probe.playback_leases != audio
            {
                return Err(format!(
                    "native turn {} did not publish and retire every audio emission (ticket={:?}/{:?}, emitted={}, text={}, playback={}/{}, expected={})",
                    turn + 1,
                    probe.flow.ticket,
                    self.ticket,
                    probe.emitted_items,
                    probe.text_emissions,
                    probe.playback_retired,
                    probe.playback_leases,
                    audio,
                ));
            }
            if let Some(previous) = self.turns.last() {
                if probe.flow.session != previous.flow.session
                    || probe.flow.epoch != previous.flow.epoch
                    || probe.flow.ticket.runtime_epoch != previous.flow.ticket.runtime_epoch
                    || probe.flow.ticket.sequence <= previous.flow.ticket.sequence
                {
                    return Err(format!(
                        "native turn {} did not continue the same correlated session/conversation lineage",
                        turn + 1
                    ));
                }
            }
            if self.transcript.trim().is_empty() {
                return Err(format!(
                    "native truth gate turn {} has no transcript",
                    turn + 1
                ));
            }
            if self.pcm.is_empty() || self.pcm.iter().any(|sample| !sample.is_finite()) {
                return Err(format!(
                    "native truth gate turn {} has no finite PCM",
                    turn + 1
                ));
            }
            if self.claimed != self.played || self.played != self.pcm.len() {
                return Err(format!(
                    "native truth gate turn {} did not retire every promised sample (claimed={}, played={}, retained={})",
                    turn + 1,
                    self.claimed,
                    self.played,
                    self.pcm.len()
                ));
            }
            let rms = (self
                .pcm
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>()
                / self.pcm.len() as f64)
                .sqrt();
            if rms <= 1e-6 || self.blocks == 0 {
                return Err(format!("native truth gate turn {} is silent", turn + 1));
            }
            self.turns.push(Turn {
                transcript: std::mem::take(&mut self.transcript),
                pcm: std::mem::take(&mut self.pcm),
                rate: self
                    .source
                    .as_ref()
                    .ok_or("native playback source retired before turn evidence")?
                    .rate(),
                flow: probe.flow,
                claimed: std::mem::take(&mut self.claimed),
                played: std::mem::take(&mut self.played),
                blocks: std::mem::take(&mut self.blocks),
                terminals: std::mem::take(&mut self.terminals),
            });
            Ok(())
        }

        fn begin_stop(&mut self, error: Option<String>) -> ServiceOutcome {
            if let Some(error) = error {
                if let Some(previous) = self.error.as_mut() {
                    previous.push_str("; ");
                    previous.push_str(&error);
                } else {
                    self.error = Some(error);
                }
            }
            if !matches!(self.phase, Phase::Stopping | Phase::Done) {
                self.source.take();
                self.capture.take();
                let started = self.engine.as_ref().is_some_and(|engine| engine.started);
                if let Some(engine) = self.engine.as_mut() {
                    engine.request_stop();
                }
                if !started {
                    let mut engine = self
                        .engine
                        .take()
                        .expect("unstarted truth-gate engine disappeared");
                    engine.retire_resume();
                    let exit = GateExit {
                        engine,
                        turns: std::mem::take(&mut self.turns),
                        error: self.error.take(),
                    };
                    *self
                        .result
                        .lock()
                        .expect("native gate result mutex poisoned") = Some(exit);
                    self.phase = Phase::Done;
                    return ServiceOutcome::Complete;
                }
                self.phase = Phase::Stopping;
            }
            ServiceOutcome::Continue
        }

        fn settle(&mut self) -> Result<ServiceOutcome, String> {
            let cancel = AtomicBool::new(false);
            let mut events = Vec::new();
            let progress = match self
                .engine()?
                .advance_events(&cancel, &mut |event| events.push(event))
            {
                Ok(progress) => progress,
                Err(error) => {
                    self.begin_stop(Some(error));
                    if !self.engine.as_ref().is_some_and(|engine| engine.stopped) {
                        return Ok(ServiceOutcome::Dormant);
                    }
                    EngineProgress::Stopped
                }
            };
            for event in events {
                if !matches!(event, VoiceEvent::Interrupted) {
                    return Err(
                        "native truth gate published a non-stop event while retiring".into(),
                    );
                }
            }
            if progress != EngineProgress::Stopped {
                return Ok(if progress == EngineProgress::Continue {
                    ServiceOutcome::Continue
                } else {
                    ServiceOutcome::Dormant
                });
            }
            let mut engine = self
                .engine
                .take()
                .ok_or("native truth-gate stop edge lost its engine")?;
            let flow = engine
                .stopped_flow
                .ok_or("native truth-gate STOPPED edge had no correlation")?;
            if engine.session_id != Some(flow.session) {
                return Err("native truth-gate STOPPED edge changed session identity".into());
            }
            engine.retire_resume();
            let exit = GateExit {
                engine,
                turns: std::mem::take(&mut self.turns),
                error: self.error.take(),
            };
            *self
                .result
                .lock()
                .expect("native gate result mutex poisoned") = Some(exit);
            self.phase = Phase::Done;
            Ok(ServiceOutcome::Complete)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Side {
        A,
        B,
    }

    impl Side {
        fn other(self) -> Self {
            match self {
                Self::A => Self::B,
                Self::B => Self::A,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::A => "A",
                Self::B => "B",
            }
        }
    }

    struct Exchange {
        side: Side,
        turn: Turn,
    }

    struct DuoReport {
        exchanges: Vec<Exchange>,
        a: SessionSnapshot,
        b: SessionSnapshot,
        stopped_a: Flow,
        stopped_b: Flow,
    }

    const DUO_ROUTE: [(Side, bool); 3] = [(Side::A, true), (Side::B, true), (Side::A, false)];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum DuoLeg {
        Idle,
        Running(usize),
    }

    struct DuoLane {
        leg: DuoLeg,
        transcript: String,
        pcm: Vec<f32>,
        ticket: Ticket,
        claimed: usize,
        played: usize,
        blocks: usize,
        terminals: usize,
        forwarded: usize,
    }

    impl DuoLane {
        fn new() -> Self {
            Self {
                leg: DuoLeg::Idle,
                transcript: String::new(),
                pcm: Vec::new(),
                ticket: Ticket::default(),
                claimed: 0,
                played: 0,
                blocks: 0,
                terminals: 0,
                forwarded: 0,
            }
        }
    }

    #[derive(Debug)]
    enum DuoPhase {
        Init,
        Running,
        Stopping,
        Done,
    }

    struct DuoBootstrap {
        cursor: usize,
        silence: usize,
    }

    struct DuoTail {
        exchange: usize,
        silence: usize,
    }

    struct DuoExit {
        engine_a: NativeLfm2VoiceEngine,
        engine_b: NativeLfm2VoiceEngine,
        exchanges: Vec<Exchange>,
        error: Option<String>,
    }

    type DuoResultCell = Arc<Mutex<Option<DuoExit>>>;

    struct Duo {
        source_a: Option<Box<dyn PlaybackSource>>,
        source_b: Option<Box<dyn PlaybackSource>>,
        events_a: Option<RealtimeNotifier>,
        events_b: Option<RealtimeNotifier>,
        capture_a: Option<Box<dyn CaptureSink>>,
        capture_b: Option<Box<dyn CaptureSink>>,
        engine_a: Option<NativeLfm2VoiceEngine>,
        engine_b: Option<NativeLfm2VoiceEngine>,
        fixture: Wave,
        phase: DuoPhase,
        bootstrap: Option<DuoBootstrap>,
        tail_a: Option<DuoTail>,
        tail_b: Option<DuoTail>,
        lane_a: DuoLane,
        lane_b: DuoLane,
        exchanges: Vec<Option<Exchange>>,
        next: usize,
        pending_a: Vec<VoiceEvent>,
        pending_b: Vec<VoiceEvent>,
        error: Option<String>,
        result: DuoResultCell,
        trace: bool,
    }

    impl Duo {
        fn step(&mut self) -> ServiceOutcome {
            match self.advance() {
                Ok(outcome) => outcome,
                Err(error) => self.begin_stop(Some(error)),
            }
        }

        fn drain_all(&mut self) -> Result<(EngineProgress, EngineProgress), String> {
            let cancel = AtomicBool::new(false);
            let a = self
                .engine_a
                .as_mut()
                .ok_or("self-chat A engine already retired")?
                .advance_events(&cancel, &mut |event| self.pending_a.push(event));
            let b = self
                .engine_b
                .as_mut()
                .ok_or("self-chat B engine already retired")?
                .advance_events(&cancel, &mut |event| self.pending_b.push(event));
            match (a, b) {
                (Ok(a), Ok(b)) => Ok((a, b)),
                (Err(a), Err(b)) => Err(format!(
                    "self-chat engines failed together (A: {a}; B: {b})"
                )),
                (Err(error), Ok(_)) => Err(format!("self-chat A failed: {error}")),
                (Ok(_), Err(error)) => Err(format!("self-chat B failed: {error}")),
            }
        }

        fn advance(&mut self) -> Result<ServiceOutcome, String> {
            if matches!(self.phase, DuoPhase::Init) {
                if self.trace {
                    eprintln!("[native-e2e] self-chat init: mount both native event docks");
                }
                let events_a = self
                    .events_a
                    .take()
                    .ok_or("self-chat A lost its event producer edge")?;
                let events_b = self
                    .events_b
                    .take()
                    .ok_or("self-chat B lost its event producer edge")?;
                self.engine_a
                    .as_mut()
                    .ok_or("self-chat A engine disappeared before mount")?
                    .mount_events(events_a)?;
                self.engine_b
                    .as_mut()
                    .ok_or("self-chat B engine disappeared before mount")?
                    .mount_events(events_b)?;
                self.bootstrap = Some(DuoBootstrap {
                    cursor: 0,
                    silence: 0,
                });
                self.phase = DuoPhase::Running;
                return Ok(ServiceOutcome::Continue);
            }

            if matches!(self.phase, DuoPhase::Stopping) {
                return self.settle();
            }
            if matches!(self.phase, DuoPhase::Done) {
                return Ok(ServiceOutcome::Complete);
            }

            let (progress_a, progress_b) = self.drain_all()?;
            if self.trace
                && (!self.pending_a.is_empty()
                    || !self.pending_b.is_empty()
                    || progress_a != EngineProgress::Dormant
                    || progress_b != EngineProgress::Dormant)
            {
                eprintln!(
                    "[native-e2e] self-chat phase={:?} progress={progress_a:?}/{progress_b:?} events={}/{}",
                    self.phase,
                    self.pending_a.len(),
                    self.pending_b.len()
                );
            }
            let events_a = std::mem::take(&mut self.pending_a);
            let events_b = std::mem::take(&mut self.pending_b);
            if DUO_ROUTE
                .get(self.next)
                .is_some_and(|route| route.0 == Side::B)
            {
                self.process(Side::B, events_b, progress_b)?;
                self.process(Side::A, events_a, progress_a)?;
            } else {
                self.process(Side::A, events_a, progress_a)?;
                self.process(Side::B, events_b, progress_b)?;
            }

            let mut ready =
                progress_a == EngineProgress::Continue || progress_b == EngineProgress::Continue;
            if matches!(self.lane_a.leg, DuoLeg::Running(_)) {
                ready |= self.pump(Side::A)?;
            }
            if matches!(self.lane_b.leg, DuoLeg::Running(_)) {
                ready |= self.pump(Side::B)?;
            }
            ready |= self.feed_bootstrap()?;
            ready |= self.feed_tail(Side::A)?;
            ready |= self.feed_tail(Side::B)?;

            if self.exchanges.iter().all(Option::is_some)
                && self.lane_a.leg == DuoLeg::Idle
                && self.lane_b.leg == DuoLeg::Idle
                && self.bootstrap.is_none()
                && self.tail_a.is_none()
                && self.tail_b.is_none()
            {
                return Ok(self.begin_stop(None));
            }
            Ok(if ready {
                ServiceOutcome::Continue
            } else {
                ServiceOutcome::Dormant
            })
        }

        fn lane(&self, side: Side) -> &DuoLane {
            match side {
                Side::A => &self.lane_a,
                Side::B => &self.lane_b,
            }
        }

        fn lane_mut(&mut self, side: Side) -> &mut DuoLane {
            match side {
                Side::A => &mut self.lane_a,
                Side::B => &mut self.lane_b,
            }
        }

        fn process(
            &mut self,
            side: Side,
            events: Vec<VoiceEvent>,
            progress: EngineProgress,
        ) -> Result<(), String> {
            for event in events {
                match event {
                    VoiceEvent::TurnStarted => self.admit(side)?,
                    VoiceEvent::Text(text) => {
                        let lane = self.lane_mut(side);
                        if !matches!(lane.leg, DuoLeg::Running(_)) {
                            return Err(format!(
                                "self-chat {} action data arrived before its correlated turn-start",
                                side.name()
                            ));
                        }
                        lane.transcript.push_str(&text);
                    }
                    VoiceEvent::TurnComplete => {
                        let lane = self.lane_mut(side);
                        if !matches!(lane.leg, DuoLeg::Running(_)) {
                            return Err(format!(
                                "self-chat {} received an uncorrelated terminal",
                                side.name()
                            ));
                        }
                        lane.terminals = lane.terminals.saturating_add(1);
                    }
                    VoiceEvent::Interrupted => {
                        return Err(format!("self-chat {} was interrupted", side.name()))
                    }
                    VoiceEvent::Error(error) => return Err(error),
                }
            }

            if progress == EngineProgress::Stopped {
                return Err(format!(
                    "self-chat {} stopped during an action",
                    side.name()
                ));
            }
            let (leg, terminals) = {
                let lane = self.lane(side);
                (lane.leg, lane.terminals)
            };
            if terminals > 1 {
                return Err(format!(
                    "self-chat {} published {} terminal events",
                    side.name(),
                    terminals
                ));
            }
            if progress == EngineProgress::Complete {
                let DuoLeg::Running(exchange) = leg else {
                    return Err(format!(
                        "self-chat {} completed without a correlated action",
                        side.name()
                    ));
                };
                if terminals != 1 {
                    return Err(format!(
                        "self-chat {} completed without exactly one terminal event",
                        side.name()
                    ));
                }
                self.finish_exchange(side, exchange)?;
                return Ok(());
            }
            if terminals != 0 {
                return Err(format!(
                    "self-chat {} terminal did not complete its action",
                    side.name()
                ));
            }
            Ok(())
        }

        fn pump(&mut self, side: Side) -> Result<bool, String> {
            let mut block = [0.0f32; DEVICE_FRAMES];
            let write = match side {
                Side::A => self
                    .source_a
                    .as_mut()
                    .ok_or("self-chat A playback source retired")?
                    .write_f32(&mut block, 1, false),
                Side::B => self
                    .source_b
                    .as_mut()
                    .ok_or("self-chat B playback source retired")?
                    .write_f32(&mut block, 1, false),
            };
            if write.dropped_samples != 0 || write.underrun_frames != 0 {
                return Err(format!(
                    "self-chat {} playback was discontinuous (dropped={}, underrun={})",
                    side.name(),
                    write.dropped_samples,
                    write.underrun_frames
                ));
            }
            if write.played_frames > block.len()
                || block[..write.played_frames]
                    .iter()
                    .any(|sample| !sample.is_finite())
            {
                return Err(format!("self-chat {} published invalid PCM", side.name()));
            }
            let leg = self.lane(side).leg;
            if (write.claimed_samples != 0 || write.played_frames != 0)
                && !matches!(leg, DuoLeg::Running(_))
            {
                return Err(format!(
                    "self-chat {} playback escaped its correlated action",
                    side.name()
                ));
            }
            if write.played_frames != 0 {
                let DuoLeg::Running(exchange) = leg else {
                    unreachable!()
                };
                if DUO_ROUTE[exchange].1 {
                    let target = match side.other() {
                        Side::A => self.capture_a.as_mut(),
                        Side::B => self.capture_b.as_mut(),
                    }
                    .ok_or("self-chat playback has no direct peer capture sink")?;
                    let capture = target.write_f32(&block[..write.played_frames], 1);
                    if capture.admitted_frames != write.played_frames
                        || capture.dropped_frames != 0
                        || capture.gap_published
                    {
                        return Err(format!(
                            "self-chat {} dropped a direct peer capture block",
                            side.name()
                        ));
                    }
                    let forwarded = self
                        .lane(side)
                        .forwarded
                        .saturating_add(write.played_frames);
                    self.lane_mut(side).forwarded = forwarded;
                }
                // This is evidence only. The production handoff above writes
                // each device-sized block into the peer's native reservation
                // while the source lease remains claimed.
                let lane = self.lane_mut(side);
                lane.pcm.extend_from_slice(&block[..write.played_frames]);
                lane.blocks = lane.blocks.saturating_add(1);
            }
            let lane = self.lane_mut(side);
            lane.claimed = lane.claimed.saturating_add(write.claimed_samples);
            lane.played = lane.played.saturating_add(write.played_frames);
            Ok(write.active || write.played_frames != 0)
        }

        fn feed_bootstrap(&mut self) -> Result<bool, String> {
            let Some(feed) = self.bootstrap.as_mut() else {
                return Ok(false);
            };
            let chunk = (self.fixture.rate as usize * CAPTURE_CHUNK_MS / 1_000).max(1);
            let frames = self.fixture.samples.len() / self.fixture.channels;
            if feed.cursor != frames {
                let end = (feed.cursor + chunk).min(frames);
                let begin = feed.cursor * self.fixture.channels;
                let limit = end * self.fixture.channels;
                let write = self
                    .capture_a
                    .as_mut()
                    .ok_or("self-chat A capture sink retired")?
                    .write_i16(&self.fixture.samples[begin..limit], self.fixture.channels);
                if write.admitted_frames != end - feed.cursor
                    || write.dropped_frames != 0
                    || write.gap_published
                {
                    return Err("self-chat bootstrap dropped a capture block".into());
                }
                feed.cursor = end;
                return Ok(true);
            }

            let target = (self.fixture.rate as usize * CAPTURE_TAIL_MS / 1_000).max(1);
            if feed.silence != target {
                let frames = (target - feed.silence).min(DEVICE_FRAMES);
                let write = self
                    .capture_a
                    .as_mut()
                    .ok_or("self-chat A capture sink retired during acoustic silence")?
                    .write_i16(
                        &self.fixture.silence[..frames * self.fixture.channels],
                        self.fixture.channels,
                    );
                if write.admitted_frames != frames
                    || write.dropped_frames != 0
                    || write.gap_published
                {
                    return Err("self-chat bootstrap dropped a silent callback block".into());
                }
                feed.silence += frames;
                return Ok(true);
            }
            if self.trace {
                eprintln!(
                    "[native-e2e] self-chat bootstrap dormant after {} fixture frames and {} silence frames",
                    frames,
                    feed.silence
                );
            }
            self.bootstrap = None;
            Ok(false)
        }

        fn feed_tail(&mut self, side: Side) -> Result<bool, String> {
            let trace = self.trace;
            let (tail, capture) = match side {
                Side::A => (&mut self.tail_a, self.capture_a.as_mut()),
                Side::B => (&mut self.tail_b, self.capture_b.as_mut()),
            };
            let Some(feed) = tail.as_mut() else {
                return Ok(false);
            };
            let capture = capture.ok_or("self-chat peer capture sink retired during silence")?;
            let target = (capture.rate() as usize * CAPTURE_TAIL_MS / 1_000).max(1);
            if feed.silence != target {
                let frames = (target - feed.silence).min(DEVICE_FRAMES);
                let block = [0.0f32; DEVICE_FRAMES];
                let write = capture.write_f32(&block[..frames], 1);
                if write.admitted_frames != frames
                    || write.dropped_frames != 0
                    || write.gap_published
                {
                    return Err("self-chat peer dropped a silent callback block".into());
                }
                feed.silence += frames;
                return Ok(true);
            }
            if trace {
                eprintln!(
                    "[native-e2e] self-chat {} tail dormant after {} silence frames (exchange={})",
                    side.name(),
                    feed.silence,
                    feed.exchange + 1
                );
            }
            *tail = None;
            Ok(false)
        }

        fn admit(&mut self, side: Side) -> Result<(), String> {
            if self.lane(side).leg != DuoLeg::Idle {
                return Err(format!(
                    "self-chat {} received an overlapping turn-start",
                    side.name()
                ));
            }
            let Some(&(expected, _)) = DUO_ROUTE.get(self.next) else {
                return Err(format!(
                    "self-chat {} started after the fixed route completed",
                    side.name()
                ));
            };
            if side != expected {
                return Err(format!(
                    "self-chat {} changed the fixed A/B/A route at exchange {}",
                    side.name(),
                    self.next + 1
                ));
            }
            let ticket = match side {
                Side::A => self
                    .engine_a
                    .as_ref()
                    .and_then(|engine| engine.active.as_ref()),
                Side::B => self
                    .engine_b
                    .as_ref()
                    .and_then(|engine| engine.active.as_ref()),
            }
            .ok_or("self-chat turn-start has no active action")?
            .ticket;
            if ticket.runtime_epoch == 0
                || ticket.sequence == 0
                || ticket.generation == 0
                || ticket.kind == 0
            {
                return Err(format!(
                    "self-chat {} received an empty ticket",
                    side.name()
                ));
            }
            if self
                .exchanges
                .iter()
                .filter_map(Option::as_ref)
                .rev()
                .find(|exchange| exchange.side == side)
                .is_some_and(|exchange| exchange.turn.flow.ticket.sequence >= ticket.sequence)
            {
                return Err(format!(
                    "self-chat {} ticket sequence did not advance",
                    side.name()
                ));
            }
            if self
                .exchanges
                .iter()
                .filter_map(Option::as_ref)
                .any(|exchange| exchange.turn.flow.ticket == ticket)
                || self.lane(side.other()).ticket == ticket
            {
                return Err(format!(
                    "self-chat {} reused another session's canonical ticket",
                    side.name()
                ));
            }
            let active = match side {
                Side::A => self
                    .engine_a
                    .as_ref()
                    .and_then(|engine| engine.active.as_ref()),
                Side::B => self
                    .engine_b
                    .as_ref()
                    .and_then(|engine| engine.active.as_ref()),
            }
            .ok_or("self-chat admitted audio has no active action")?
            .ticket;
            if active != ticket {
                return Err(format!(
                    "self-chat {} changed its capture ticket",
                    side.name()
                ));
            }
            let exchange = self.next;
            self.next += 1;
            let lane = self.lane_mut(side);
            *lane = DuoLane::new();
            lane.leg = DuoLeg::Running(exchange);
            lane.ticket = ticket;
            if self.trace {
                eprintln!(
                    "[native-e2e] self-chat admitted exchange {} on {} ticket sequence={} (forward={})",
                    exchange + 1,
                    side.name(),
                    ticket.sequence,
                    DUO_ROUTE[exchange].1
                );
            }
            Ok(())
        }

        fn finish_exchange(&mut self, side: Side, exchange: usize) -> Result<(), String> {
            if exchange >= self.exchanges.len() || self.exchanges[exchange].is_some() {
                return Err(format!(
                    "self-chat {} attempted to settle exchange {} twice",
                    side.name(),
                    exchange + 1
                ));
            }
            let probe = match side {
                Side::A => self
                    .engine_a
                    .as_mut()
                    .and_then(|engine| engine.terminal_probe.take()),
                Side::B => self
                    .engine_b
                    .as_mut()
                    .and_then(|engine| engine.terminal_probe.take()),
            }
            .ok_or("self-chat terminal has no correlated decoder evidence")?;
            let session = match side {
                Side::A => self.engine_a.as_ref().and_then(|engine| engine.session_id),
                Side::B => self.engine_b.as_ref().and_then(|engine| engine.session_id),
            };
            let lane = std::mem::replace(self.lane_mut(side), DuoLane::new());
            if lane.leg != DuoLeg::Running(exchange) || DUO_ROUTE[exchange].0 != side {
                return Err("self-chat terminal changed its fixed route".into());
            }
            let audio = probe
                .emitted_items
                .checked_sub(probe.text_emissions)
                .ok_or("self-chat text accounting exceeds emitted items")?;
            if probe.flow.ticket != lane.ticket
                || session != Some(probe.flow.session)
                || probe.terminal_records != 1
                || probe.playback_leases != audio
                || probe.playback_retired != probe.playback_leases
            {
                return Err(format!(
                    "self-chat {} did not publish and retire every correlated audio emission",
                    side.name()
                ));
            }
            if let Some(previous) = self
                .exchanges
                .iter()
                .filter_map(Option::as_ref)
                .rev()
                .find(|prior| prior.side == side)
            {
                if probe.flow.session != previous.turn.flow.session
                    || probe.flow.epoch != previous.turn.flow.epoch
                    || probe.flow.ticket.runtime_epoch != previous.turn.flow.ticket.runtime_epoch
                    || probe.flow.ticket.sequence <= previous.turn.flow.ticket.sequence
                {
                    return Err(format!(
                        "self-chat {} changed its session/ticket lineage",
                        side.name()
                    ));
                }
            }
            if lane.transcript.trim().is_empty()
                || lane.pcm.is_empty()
                || lane.pcm.iter().any(|sample| !sample.is_finite())
                || lane.claimed != lane.played
                || lane.played != lane.pcm.len()
                || lane.blocks == 0
            {
                return Err(format!(
                    "self-chat {} has incomplete transcript/PCM evidence",
                    side.name()
                ));
            }
            if DUO_ROUTE[exchange].1 && lane.forwarded != lane.pcm.len() {
                return Err(format!(
                    "self-chat {} direct capture did not contain every playback sample",
                    side.name()
                ));
            }
            let rms = (lane
                .pcm
                .iter()
                .map(|sample| f64::from(*sample) * f64::from(*sample))
                .sum::<f64>()
                / lane.pcm.len() as f64)
                .sqrt();
            if rms <= 1e-6 {
                return Err(format!("self-chat {} playback is silent", side.name()));
            }
            let rate = match side {
                Side::A => self.source_a.as_ref().unwrap().rate(),
                Side::B => self.source_b.as_ref().unwrap().rate(),
            };
            self.exchanges[exchange] = Some(Exchange {
                side,
                turn: Turn {
                    transcript: lane.transcript,
                    pcm: lane.pcm,
                    rate,
                    flow: probe.flow,
                    claimed: lane.claimed,
                    played: lane.played,
                    blocks: lane.blocks,
                    terminals: lane.terminals,
                },
            });
            if DUO_ROUTE[exchange].1 {
                let target = side.other();
                let tail = match target {
                    Side::A => &mut self.tail_a,
                    Side::B => &mut self.tail_b,
                };
                if tail.is_some() {
                    return Err(format!(
                        "self-chat {} acoustic tail overlapped its predecessor",
                        target.name()
                    ));
                }
                *tail = Some(DuoTail {
                    exchange,
                    silence: 0,
                });
            }
            if self.trace {
                eprintln!(
                    "[native-e2e] self-chat completed exchange {} on {}",
                    exchange + 1,
                    side.name()
                );
            }
            Ok(())
        }

        fn begin_stop(&mut self, error: Option<String>) -> ServiceOutcome {
            if let Some(error) = error {
                if let Some(previous) = self.error.as_mut() {
                    previous.push_str("; ");
                    previous.push_str(&error);
                } else {
                    self.error = Some(error);
                }
            }
            if !matches!(self.phase, DuoPhase::Stopping | DuoPhase::Done) {
                self.source_a.take();
                self.source_b.take();
                self.capture_a.take();
                self.capture_b.take();
                if let Some(engine) = self.engine_a.as_mut() {
                    engine.request_stop();
                }
                if let Some(engine) = self.engine_b.as_mut() {
                    engine.request_stop();
                }
                self.phase = DuoPhase::Stopping;
            }
            ServiceOutcome::Continue
        }

        fn settle(&mut self) -> Result<ServiceOutcome, String> {
            let cancel = AtomicBool::new(false);
            let a = self
                .engine_a
                .as_mut()
                .ok_or("self-chat A stop edge lost its engine")?
                .advance_events(&cancel, &mut |event| self.pending_a.push(event));
            let b = self
                .engine_b
                .as_mut()
                .ok_or("self-chat B stop edge lost its engine")?
                .advance_events(&cancel, &mut |event| self.pending_b.push(event));
            if let Err(error) = &a {
                self.begin_stop(Some(format!("self-chat A stop: {error}")));
            }
            if let Err(error) = &b {
                self.begin_stop(Some(format!("self-chat B stop: {error}")));
            }
            if self
                .pending_a
                .iter()
                .chain(&self.pending_b)
                .any(|event| !matches!(event, VoiceEvent::Interrupted))
            {
                self.begin_stop(Some(
                    "self-chat published a non-stop action event while retiring".into(),
                ));
            }
            self.pending_a.clear();
            self.pending_b.clear();
            let stopped_a = self
                .engine_a
                .as_ref()
                .is_some_and(|engine| !engine.started || engine.stopped);
            let stopped_b = self
                .engine_b
                .as_ref()
                .is_some_and(|engine| !engine.started || engine.stopped);
            if !stopped_a || !stopped_b {
                return Ok(
                    if matches!(a, Ok(EngineProgress::Continue))
                        || matches!(b, Ok(EngineProgress::Continue))
                    {
                        ServiceOutcome::Continue
                    } else {
                        ServiceOutcome::Dormant
                    },
                );
            }
            for (side, engine) in [
                (Side::A, self.engine_a.as_ref().unwrap()),
                (Side::B, self.engine_b.as_ref().unwrap()),
            ] {
                if !engine.started {
                    continue;
                }
                let flow = engine.stopped_flow.ok_or_else(|| {
                    format!("self-chat {} STOPPED edge had no correlation", side.name())
                })?;
                if engine.session_id != Some(flow.session) {
                    return Err(format!(
                        "self-chat {} STOPPED edge changed session identity",
                        side.name()
                    ));
                }
            }
            self.engine_a
                .as_mut()
                .expect("validated self-chat A engine disappeared")
                .retire_resume();
            self.engine_b
                .as_mut()
                .expect("validated self-chat B engine disappeared")
                .retire_resume();
            let exchanges = self.exchanges.iter_mut().filter_map(Option::take).collect();
            let exit = DuoExit {
                engine_a: self.engine_a.take().unwrap(),
                engine_b: self.engine_b.take().unwrap(),
                exchanges,
                error: self.error.take(),
            };
            *self.result.lock().expect("self-chat result mutex poisoned") = Some(exit);
            self.phase = DuoPhase::Done;
            Ok(ServiceOutcome::Complete)
        }
    }

    fn run_duo(model: &NativeVoiceModel, fixture: &Path, tokens: u32) -> Result<DuoReport, String> {
        if model.runtime_config().session_capacity < 2 {
            return Err("self-chat requires a native model runtime with two session slots".into());
        }
        let sampling = NativeVoiceSampling {
            max_new_tokens: tokens,
            text_temperature: None,
            text_top_k: None,
            audio_temperature: Some(1.0),
            audio_top_k: Some(4),
            seed: Some(0),
        };
        let vault_a = NativeConversationVault::default();
        let vault_b = NativeConversationVault::default();
        if Arc::ptr_eq(&vault_a.0, &vault_b.0) {
            return Err("self-chat conversation vaults are not independent".into());
        }
        let fixture = read_wave(fixture)?;
        let callback_frames = fixture.rate.div_ceil(50).max(DEVICE_FRAMES as u32);
        let mut engine_a = model.engine(
            sampling,
            Some(vault_a),
            fixture.rate,
            fixture.rate,
            callback_frames,
        )?;
        let mut engine_b = model.engine(
            sampling,
            Some(vault_b),
            fixture.rate,
            fixture.rate,
            callback_frames,
        )?;
        let capture_a = engine_a
            .take_capture_sink()?
            .ok_or("self-chat A did not expose its capture sink")?;
        let capture_b = engine_b
            .take_capture_sink()?
            .ok_or("self-chat B did not expose its capture sink")?;
        let result: DuoResultCell = Arc::new(Mutex::new(None));
        let answer = result.clone();
        let runtime = CoroutineRuntime::new()
            .map_err(|code| format!("create self-chat kcoro runtime: {code}"))?;
        let (service, ()) = runtime
            .owner_state_service_factory(|setup| {
                let events_a = setup.realtime_notifier()?;
                let events_b = setup.realtime_notifier()?;
                let playback_a = setup.realtime_notifier()?;
                let playback_b = setup.realtime_notifier()?;
                let source_a = engine_a
                    .take_playback_source(playback_a)
                    .map_err(|_| -1)?
                    .ok_or(-1)?;
                let source_b = engine_b
                    .take_playback_source(playback_b)
                    .map_err(|_| -1)?
                    .ok_or(-1)?;
                let init = move || {
                    let mut duo = Duo {
                        source_a: Some(source_a),
                        source_b: Some(source_b),
                        events_a: Some(events_a),
                        events_b: Some(events_b),
                        capture_a: Some(capture_a),
                        capture_b: Some(capture_b),
                        engine_a: Some(engine_a),
                        engine_b: Some(engine_b),
                        fixture,
                        phase: DuoPhase::Init,
                        bootstrap: None,
                        tail_a: None,
                        tail_b: None,
                        lane_a: DuoLane::new(),
                        lane_b: DuoLane::new(),
                        exchanges: (0..DUO_ROUTE.len()).map(|_| None).collect(),
                        next: 0,
                        pending_a: Vec::new(),
                        pending_b: Vec::new(),
                        error: None,
                        result,
                        trace: std::env::var_os("LFM_E2E_TRACE").is_some(),
                    };
                    move || duo.step()
                };
                Ok::<_, i32>((init, ()))
            })
            .map_err(|code| format!("mount self-chat kcoro service: {code}"))?;
        runtime
            .start()
            .map_err(|code| format!("start self-chat kcoro runtime: {code}"))?;
        service
            .start()
            .map_err(|code| format!("start self-chat kcoro service: {code}"))?;
        service
            .notify()
            .map_err(|code| format!("admit self-chat state machine: {code}"))?;
        runtime
            .join_all()
            .map_err(|code| format!("observe self-chat terminal edge: {code}"))?;
        service
            .join()
            .map_err(|code| format!("join completed self-chat service: {code}"))?;
        if service.callback_panicked() {
            return Err("self-chat kcoro callback panicked".into());
        }
        if let Some(code) = service.reschedule_error() {
            return Err(format!("self-chat kcoro reschedule failed: {code}"));
        }
        service
            .destroy()
            .map_err(|code| format!("destroy self-chat kcoro service: {code}"))?;
        runtime
            .destroy()
            .map_err(|code| format!("destroy self-chat kcoro runtime: {code}"))?;
        let exit = answer
            .lock()
            .expect("self-chat result mutex poisoned")
            .take()
            .ok_or_else(|| "self-chat service retired without evidence".to_string())?;
        // STOPPED retired both callback continuations. Administrative pthread
        // joins happen only now, outside the retained kcoro callback.
        settle_duo(exit)
    }

    fn settle_duo(mut exit: DuoExit) -> Result<DuoReport, String> {
        for (side, engine) in [(Side::A, &mut exit.engine_a), (Side::B, &mut exit.engine_b)] {
            if let Err(settle) = engine.stop_session() {
                let message = format!("self-chat {} settlement failed: {settle}", side.name());
                if let Some(error) = exit.error.as_mut() {
                    error.push_str("; ");
                    error.push_str(&message);
                } else {
                    exit.error = Some(message);
                }
            }
        }
        let mut a = SessionSnapshot::default();
        let mut b = SessionSnapshot::default();
        status(
            unsafe { lfm_session_snapshot(exit.engine_a.session.as_ptr(), &mut a) },
            "snapshot settled self-chat A",
        )?;
        status(
            unsafe { lfm_session_snapshot(exit.engine_b.session.as_ptr(), &mut b) },
            "snapshot settled self-chat B",
        )?;
        for (side, snapshot, capture) in [(Side::A, a, 2), (Side::B, b, 1)] {
            if snapshot.state != JOINED
                || snapshot.terminal_status != 0
                || snapshot.capture_consumed != capture
                || snapshot.capture_stale != 0
                || snapshot.text_commands_accepted != 0
                || snapshot.text_commands_consumed != 0
                || snapshot.text_commands_stale != 0
                || snapshot.playback_published == 0
                || snapshot.playback_published != snapshot.playback_consumed
                || snapshot.live_playback_leases != 0
                || snapshot.reliable_event_depth != 0
            {
                let message = format!("self-chat {} did not retire cleanly", side.name());
                if let Some(error) = exit.error.as_mut() {
                    error.push_str("; ");
                    error.push_str(&message);
                } else {
                    exit.error = Some(message);
                }
            }
        }
        if a.session_id == b.session_id {
            let message = "self-chat engines shared a native session identity".to_string();
            if let Some(error) = exit.error.as_mut() {
                error.push_str("; ");
                error.push_str(&message);
            } else {
                exit.error = Some(message);
            }
        }
        if let Some(error) = exit.error {
            return Err(error);
        }
        let stopped_a = exit
            .engine_a
            .stopped_flow
            .ok_or("self-chat A retired without an exact STOPPED flow")?;
        let stopped_b = exit
            .engine_b
            .stopped_flow
            .ok_or("self-chat B retired without an exact STOPPED flow")?;
        if a.session_id != stopped_a.session
            || a.epoch != stopped_a.epoch
            || b.session_id != stopped_b.session
            || b.epoch != stopped_b.epoch
        {
            return Err("self-chat STOPPED flow did not match its settled snapshot".into());
        }
        Ok(DuoReport {
            exchanges: exit.exchanges,
            a,
            b,
            stopped_a,
            stopped_b,
        })
    }

    fn run(model: &NativeVoiceModel, fixture: &Path, tokens: u32) -> Result<Report, String> {
        let sampling = NativeVoiceSampling {
            max_new_tokens: tokens,
            text_temperature: None,
            text_top_k: None,
            audio_temperature: Some(1.0),
            audio_top_k: Some(4),
            seed: Some(0),
        };
        let wave = read_wave(fixture)?;
        let mut engine = model.engine(
            sampling,
            None,
            wave.rate,
            OUTPUT_RATE,
            wave.rate.div_ceil(50).max(DEVICE_FRAMES as u32),
        )?;
        let capture = engine
            .take_capture_sink()?
            .ok_or("native engine did not expose its capture sink")?;
        let result: ResultCell = Arc::new(Mutex::new(None));
        let answer = result.clone();
        let runtime = CoroutineRuntime::new()
            .map_err(|code| format!("create truth-gate kcoro runtime: {code}"))?;
        let (service, ()) = runtime
            .owner_state_service_factory(|setup| {
                let events = setup.realtime_notifier()?;
                let playback = setup.realtime_notifier()?;
                let source = engine.take_playback_source(playback).map_err(|_| -1)?;
                let source = source.ok_or(-1)?;
                let init = move || {
                    let mut gate = Gate {
                        source: Some(source),
                        events: Some(events),
                        capture: Some(capture),
                        engine: Some(engine),
                        fixture: wave,
                        phase: Phase::Init,
                        turns: Vec::with_capacity(3),
                        transcript: String::new(),
                        pcm: Vec::new(),
                        ticket: Ticket::default(),
                        claimed: 0,
                        played: 0,
                        blocks: 0,
                        terminals: 0,
                        error: None,
                        result,
                        trace: std::env::var_os("LFM_E2E_TRACE").is_some(),
                    };
                    move || gate.step()
                };
                Ok::<_, i32>((init, ()))
            })
            .map_err(|code| format!("mount truth-gate kcoro service: {code}"))?;
        runtime
            .start()
            .map_err(|code| format!("start truth-gate kcoro runtime: {code}"))?;
        service
            .start()
            .map_err(|code| format!("start truth-gate kcoro service: {code}"))?;
        // This is the only test-owned advancement edge. Native event records,
        // playback retirements, and bounded Continue outcomes drive everything
        // after it; the test never polls or sleeps on inference progress.
        service
            .notify()
            .map_err(|code| format!("admit truth-gate state machine: {code}"))?;
        runtime
            .join_all()
            .map_err(|code| format!("observe truth-gate terminal edge: {code}"))?;
        service
            .join()
            .map_err(|code| format!("join completed truth-gate service: {code}"))?;
        if service.callback_panicked() {
            return Err("truth-gate kcoro callback panicked".into());
        }
        if let Some(code) = service.reschedule_error() {
            return Err(format!("truth-gate kcoro reschedule failed: {code}"));
        }
        service
            .destroy()
            .map_err(|code| format!("destroy truth-gate kcoro service: {code}"))?;
        runtime
            .destroy()
            .map_err(|code| format!("destroy truth-gate kcoro runtime: {code}"))?;
        let exit = answer
            .lock()
            .expect("native gate result mutex poisoned")
            .take()
            .ok_or_else(|| "truth-gate service retired without evidence".to_string())?;
        // STOPPED retired the callback continuation. Administrative pthread
        // join and the settled snapshot happen outside the kcoro callback.
        settle_gate(exit)
    }

    fn settle_gate(mut exit: GateExit) -> Result<Report, String> {
        if let Err(settle) = exit.engine.stop_session() {
            if let Some(error) = exit.error.as_mut() {
                error.push_str(&format!("; session settlement also failed: {settle}"));
            } else {
                exit.error = Some(settle);
            }
        }
        let mut snapshot = SessionSnapshot::default();
        status(
            unsafe { lfm_session_snapshot(exit.engine.session.as_ptr(), &mut snapshot) },
            "snapshot settled native truth-gate session",
        )?;
        if snapshot.state != JOINED
            || snapshot.terminal_status != 0
            || snapshot.live_playback_leases != 0
            || snapshot.reliable_event_depth != 0
            || snapshot.capture_consumed != 2
            || snapshot.capture_stale != 0
            || snapshot.text_commands_accepted != 1
            || snapshot.text_commands_consumed != 1
            || snapshot.text_commands_stale != 0
            || snapshot.playback_published == 0
            || snapshot.playback_published != snapshot.playback_consumed
        {
            let settlement = format!(
                "native truth-gate session did not retire cleanly: state={}, status={}, capture={}/{}, playback={}/{}, live={}, events={}, text={}/{}/{}",
                snapshot.state,
                snapshot.terminal_status,
                snapshot.capture_consumed,
                snapshot.capture_stale,
                snapshot.playback_published,
                snapshot.playback_consumed,
                snapshot.live_playback_leases,
                snapshot.reliable_event_depth,
                snapshot.text_commands_accepted,
                snapshot.text_commands_consumed,
                snapshot.text_commands_stale,
            );
            if let Some(error) = exit.error.as_mut() {
                error.push_str("; ");
                error.push_str(&settlement);
            } else {
                exit.error = Some(settlement);
            }
        }
        if let Some(error) = exit.error {
            return Err(error);
        }
        let stopped = exit
            .engine
            .stopped_flow
            .ok_or("native truth-gate session retired without an exact STOPPED flow")?;
        if snapshot.session_id != stopped.session || snapshot.epoch != stopped.epoch {
            return Err("native truth-gate STOPPED flow did not match its settled snapshot".into());
        }
        Ok(Report {
            turns: exit.turns,
            snapshot,
            stopped,
        })
    }

    fn output(
        report: &Report,
        duo: &DuoReport,
        model: NativeVoiceModelMemory,
        tokens: u32,
        self_chat_tokens: u32,
        dir: &Path,
    ) -> Result<(), String> {
        // Evidence serialization happens only after both state machines have
        // retired. WAV and JSON files are never an inference or self-chat hop;
        // exchanged audio travels directly from PlaybackSource into the peer's
        // callback-owned native CaptureSink while its playback lease is live.
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("create native gate output {}: {error}", dir.display()))?;
        let dir = dir
            .canonicalize()
            .map_err(|error| format!("resolve native gate output {}: {error}", dir.display()))?;
        let roles = ["typed", "question-audio", "question-audio-retained-context"];
        for (index, turn) in report.turns.iter().enumerate() {
            write_wave(
                &dir.join(format!(
                    "native-truth-gate-{}-{}.wav",
                    index + 1,
                    roles[index]
                )),
                &turn.pcm,
                turn.rate,
            )?;
        }
        for (index, exchange) in duo.exchanges.iter().enumerate() {
            write_wave(
                &dir.join(format!(
                    "native-self-chat-{}-{}.wav",
                    index + 1,
                    exchange.side.name().to_ascii_lowercase()
                )),
                &exchange.turn.pcm,
                exchange.turn.rate,
            )?;
        }
        let roles = ["typed", "question_audio", "question_audio_retained_context"];
        let turns = report
            .turns
            .iter()
            .enumerate()
            .map(|(index, turn)| {
                json!({
                    "turn": index + 1,
                    "role": roles[index],
                    "transcript": turn.transcript,
                    "pcm_samples": turn.pcm.len(),
                    "sample_rate": turn.rate,
                    "pcm_rms": turn.rms(),
                    "transcript_sha256": turn.transcript_digest(),
                    "pcm_sha256": turn.pcm_digest(),
                    "device_blocks": turn.blocks,
                    "claimed_samples": turn.claimed,
                    "played_samples": turn.played,
                    "terminal_events": turn.terminals,
                    "ticket": {
                        "session": turn.flow.session,
                        "stream_epoch": turn.flow.epoch,
                        "runtime_epoch": turn.flow.ticket.runtime_epoch,
                        "sequence": turn.flow.ticket.sequence,
                        "generation": turn.flow.ticket.generation,
                        "kind": turn.flow.ticket.kind,
                    },
                })
            })
            .collect::<Vec<_>>();
        let self_chat = duo
            .exchanges
            .iter()
            .enumerate()
            .map(|(index, exchange)| {
                json!({
                    "turn": index + 1,
                    "speaker": exchange.side.name(),
                    "transcript_evidence_only": exchange.turn.transcript,
                    "transport": if index < 2 { "direct_playback_pcm_to_peer_native_capture" } else { "terminal_observer_output" },
                    "pcm_samples": exchange.turn.pcm.len(),
                    "sample_rate": exchange.turn.rate,
                    "pcm_rms": exchange.turn.rms(),
                    "transcript_sha256": exchange.turn.transcript_digest(),
                    "pcm_sha256": exchange.turn.pcm_digest(),
                    "device_blocks": exchange.turn.blocks,
                    "terminal_events": exchange.turn.terminals,
                    "ticket": {
                        "session": exchange.turn.flow.session,
                        "stream_epoch": exchange.turn.flow.epoch,
                        "runtime_epoch": exchange.turn.flow.ticket.runtime_epoch,
                        "sequence": exchange.turn.flow.ticket.sequence,
                        "generation": exchange.turn.flow.ticket.generation,
                        "kind": exchange.turn.flow.ticket.kind,
                    },
                })
            })
            .collect::<Vec<_>>();
        let manifest = json!({
            "schema": 2,
            "seed": 0,
            "max_new_tokens": tokens,
            "self_chat_max_new_tokens": self_chat_tokens,
            "text_sampling": "greedy",
            "audio_temperature": 1.0,
            "audio_top_k": 4,
            "typed_output_sample_rate": OUTPUT_RATE,
            "pcm_digest_encoding": "sha256 over little-endian f32 bit patterns",
            "turns": turns,
            "retained_context_evidence": {
                "same_native_session": report.turns.iter().all(|turn| turn.flow.session == report.turns[0].flow.session),
                "same_stream_epoch": report.turns.iter().all(|turn| turn.flow.epoch == report.turns[0].flow.epoch),
                "strictly_advancing_ticket_sequence": report.turns.windows(2).all(|pair| pair[0].flow.ticket.sequence < pair[1].flow.ticket.sequence),
                "capture_turns_consumed": report.snapshot.capture_consumed,
            },
            "two_engine_self_chat": {
                "bootstrap": "question.wav audio only",
                "scripted_dialogue": false,
                "text_relay": false,
                "exchanges": self_chat,
                "sessions": {
                    "a": {
                        "id": duo.a.session_id,
                        "stopped_epoch": duo.stopped_a.epoch,
                        "stopped_ticket_sequence": duo.stopped_a.ticket.sequence,
                        "capture_consumed": duo.a.capture_consumed,
                        "playback_published": duo.a.playback_published,
                        "playback_consumed": duo.a.playback_consumed,
                    },
                    "b": {
                        "id": duo.b.session_id,
                        "stopped_epoch": duo.stopped_b.epoch,
                        "stopped_ticket_sequence": duo.stopped_b.ticket.sequence,
                        "capture_consumed": duo.b.capture_consumed,
                        "playback_published": duo.b.playback_published,
                        "playback_consumed": duo.b.playback_consumed,
                    },
                },
            },
            "model_memory": {
                "source_bytes": model.source_bytes,
                "resident_image_bytes": model.resident_image_bytes,
                "directly_bound_bytes": model.directly_bound_bytes,
                "derived_immutable_bytes": model.derived_immutable_bytes,
                "compatibility_copied_bytes": model.compatibility_copied_bytes,
                "load_ns": model.load_ns,
                "load_workers": model.load_workers,
                "load_tasks": model.load_tasks,
            },
            "session": {
                "id": report.snapshot.session_id,
                "epoch": report.snapshot.epoch,
                "stopped_epoch": report.stopped.epoch,
                "stopped_ticket_sequence": report.stopped.ticket.sequence,
                "callbacks_entered": report.snapshot.callbacks_entered,
                "capture_consumed": report.snapshot.capture_consumed,
                "playback_published": report.snapshot.playback_published,
                "playback_consumed": report.snapshot.playback_consumed,
            },
        });
        std::fs::write(
            dir.join("native-truth-gate.json"),
            serde_json::to_vec_pretty(&manifest)
                .map_err(|error| format!("encode native gate manifest: {error}"))?,
        )
        .map_err(|error| format!("write native gate manifest: {error}"))?;
        eprintln!("native truth-gate evidence: {}", dir.display());
        Ok(())
    }

    #[test]
    #[ignore = "requires explicit LFM_MODEL_DIR and the real question.wav fixture"]
    fn native_real_checkpoint_e2e_truth_gate() {
        let model_dir = PathBuf::from(
            std::env::var_os("LFM_MODEL_DIR")
                .expect("LFM_MODEL_DIR must explicitly name the complete local checkpoint"),
        );
        assert!(
            model_dir.is_dir(),
            "LFM_MODEL_DIR is not a directory: {}",
            model_dir.display()
        );
        let fixture = std::env::var_os("LFM_E2E_QUESTION_WAV")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav"));
        assert!(
            fixture.is_file(),
            "real question.wav fixture is missing: {}",
            fixture.display()
        );
        let timeout = std::env::var("LFM_E2E_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1_800);
        let tokens = std::env::var("LFM_E2E_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(192);
        assert!(tokens > 0, "LFM_E2E_MAX_TOKENS must be nonzero");
        let self_chat_tokens = std::env::var("LFM_E2E_SELF_CHAT_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(tokens);
        assert!(
            self_chat_tokens > 0,
            "LFM_E2E_SELF_CHAT_MAX_TOKENS must be nonzero"
        );
        let watchdog = Watchdog::arm(Duration::from_secs(timeout))
            .expect("arm monotonic native truth-gate watchdog");
        let model = NativeVoiceModel::open_with_config(
            &model_dir,
            NativeVoiceRuntimeConfig {
                session_capacity: 2,
                ..NativeVoiceRuntimeConfig::default()
            },
        )
        .expect("open complete native checkpoint with two session slots");
        let before = model.memory().expect("read native model accounting");
        assert!(before.source_bytes > 0 && before.directly_bound_bytes > 0);
        assert_eq!(before.compatibility_copied_bytes, 0);

        let first = run(&model, &fixture, tokens).expect("first native truth-gate run");
        let middle = model
            .memory()
            .expect("read accounting after first generation");
        assert_eq!(middle, before, "model accounting changed after generation");
        let second = run(&model, &fixture, tokens).expect("repeated native truth-gate run");
        let after = model
            .memory()
            .expect("read accounting after repeated generation");
        assert_eq!(
            after, before,
            "model accounting changed after repeated generation"
        );
        assert_eq!(first.turns.len(), 3);
        assert_eq!(second.turns.len(), 3);
        for (index, (left, right)) in first.turns.iter().zip(&second.turns).enumerate() {
            assert_eq!(
                left.transcript,
                right.transcript,
                "turn {} text drift",
                index + 1
            );
            assert_eq!(
                left.pcm.len(),
                right.pcm.len(),
                "turn {} PCM length drift",
                index + 1
            );
            assert_eq!(
                left.pcm_digest(),
                right.pcm_digest(),
                "turn {} PCM/content drift",
                index + 1
            );
            assert_eq!(left.terminals, 1);
            assert_eq!(right.terminals, 1);
        }
        let duo =
            run_duo(&model, &fixture, self_chat_tokens).expect("two-engine native audio self-chat");
        assert_eq!(duo.exchanges.len(), 3);
        assert_eq!(duo.exchanges[0].side, Side::A);
        assert_eq!(duo.exchanges[1].side, Side::B);
        assert_eq!(duo.exchanges[2].side, Side::A);
        assert_eq!(
            model
                .memory()
                .expect("read accounting after two-engine self-chat"),
            before,
            "model accounting changed after two-engine self-chat"
        );
        let dir = std::env::var_os("LFM_E2E_OUTPUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "emberharmony-native-truth-gate-{}",
                    std::process::id()
                ))
            });
        output(&first, &duo, before, tokens, self_chat_tokens, &dir)
            .expect("write native truth-gate evidence");
        drop(watchdog);
    }
}

fn retire_unstarted_session(session: NonNull<Session>) {
    unsafe { lfm_session_request_stop(session.as_ptr()) };
    let _ = unsafe { lfm_session_join(session.as_ptr()) };
    let _ = unsafe { lfm_session_destroy(session.as_ptr()) };
}

fn native_error(status: i32, error: &[i8]) -> String {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    if message.is_empty() {
        return format!("native model open failed with status {status}");
    }
    format!("{message} (native status {status})")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(stream: &mut Utf8Stream, chunks: &[&[u8]], finish: bool) -> String {
        let mut pieces = Vec::new();
        for chunk in chunks {
            stream.push(chunk, &mut |piece| pieces.push(piece)).unwrap();
        }
        if finish {
            stream.finish(&mut |piece| pieces.push(piece));
        }
        pieces.concat()
    }

    #[test]
    fn text_stream_preserves_codepoints_split_across_events() {
        let mut stream = Utf8Stream::default();
        let text = collect(
            &mut stream,
            &[b"hello \xf0", b"\x9f", b"\x8c", b"\x8d!"],
            true,
        );
        assert_eq!(text, "hello \u{1f30d}!");
    }

    #[test]
    fn text_stream_matches_whole_buffer_lossy_decode() {
        let chunks: [&[u8]; 4] = [b"\xe2", b"(\xa1ok\xf0", b"\x9f", b"tail"];
        let mut bytes = Vec::new();
        for chunk in chunks {
            bytes.extend_from_slice(chunk);
        }
        let mut stream = Utf8Stream::default();
        let text = collect(&mut stream, &chunks, true);
        assert_eq!(text, String::from_utf8_lossy(&bytes));
    }

    #[test]
    fn text_stream_reset_drops_cancelled_turn_carry() {
        let mut stream = Utf8Stream::default();
        assert_eq!(collect(&mut stream, &[b"\xf0\x9f"], false), "");
        stream.reset();
        assert_eq!(collect(&mut stream, &[b"next turn"], true), "next turn");
    }

    #[test]
    fn text_stream_finish_flushes_incomplete_sequence() {
        let mut stream = Utf8Stream::default();
        assert_eq!(
            collect(&mut stream, &[b"tail \xe2\x82"], true),
            "tail \u{fffd}"
        );
    }

    #[test]
    fn text_stream_rejects_oversized_native_record() {
        let mut stream = Utf8Stream::default();
        let bytes = [b'x'; TEXT_EVENT_MAX_BYTES + 1];
        let error = stream.push(&bytes, &mut |_| {}).unwrap_err();
        assert!(error.contains("fixed payload bound"));
    }

    #[test]
    fn later_interrupt_state_is_not_a_fresh_action_outcome() {
        let replies = ReplyRing::new();
        let mut sink = EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        };
        let payload = b"interrupted";
        let event = NativeEvent {
            size: std::mem::size_of::<NativeEvent>() as u32,
            abi_version: RUNTIME_ABI,
            kind: EVENT_STATE,
            flags: 0,
            session_id: 1,
            epoch: 2,
            ticket: Ticket {
                runtime_epoch: 1,
                sequence: 99,
                generation: 1,
                kind: TICKET_CONTROL,
            },
            payload: payload.as_ptr().cast(),
            payload_bytes: payload.len() as u32,
            status: 0,
        };
        assert_eq!(
            unsafe {
                on_event(
                    std::ptr::from_mut(&mut sink).cast(),
                    std::ptr::from_ref(&event),
                )
            },
            0
        );
        assert!(replies.try_pop().is_none());
    }

    #[test]
    fn callback_backpressure_retains_the_exact_fixed_record() {
        let replies = ReplyRing::new();
        let mut sink = EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        };
        let ticket = Ticket {
            runtime_epoch: 7,
            sequence: 11,
            generation: 3,
            kind: 2,
        };
        let first = NativeEvent {
            size: std::mem::size_of::<NativeEvent>() as u32,
            abi_version: RUNTIME_ABI,
            kind: EVENT_TEXT,
            flags: 0,
            session_id: 1,
            epoch: 4,
            ticket,
            payload: b"first".as_ptr().cast(),
            payload_bytes: 5,
            status: 0,
        };
        let context = std::ptr::from_mut(&mut sink).cast();
        assert_eq!(unsafe { on_event(context, &first) }, 0);
        for _ in 1..REPLY_CAPACITY {
            assert_eq!(unsafe { on_event(context, &first) }, 0);
        }
        let second = NativeEvent {
            payload: b"second".as_ptr().cast(),
            payload_bytes: 6,
            ..first
        };
        assert_eq!(unsafe { on_event(context, &second) }, STATUS_WOULD_BLOCK);
        assert!(
            matches!(replies.try_pop().unwrap(), Reply::Text { payload, .. } if payload.as_bytes() == b"first")
        );
        assert_eq!(unsafe { on_event(context, &second) }, 0);
        let mut last = None;
        while let Some(reply) = replies.try_pop() {
            last = Some(reply);
        }
        assert!(
            matches!(last, Some(Reply::Text { payload, .. }) if payload.as_bytes() == b"second")
        );
    }

    #[test]
    fn action_callback_rejects_an_uncorrelated_flow() {
        let replies = ReplyRing::new();
        let mut sink = EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        };
        let event = NativeEvent {
            size: std::mem::size_of::<NativeEvent>() as u32,
            abi_version: RUNTIME_ABI,
            kind: EVENT_TEXT,
            flags: 0,
            session_id: 0,
            epoch: 1,
            ticket: Ticket {
                runtime_epoch: 1,
                sequence: 1,
                generation: 1,
                kind: TICKET_TURN,
            },
            payload: b"orphan".as_ptr().cast(),
            payload_bytes: 6,
            status: 0,
        };
        assert_eq!(
            unsafe {
                on_event(
                    std::ptr::from_mut(&mut sink).cast(),
                    std::ptr::from_ref(&event),
                )
            },
            STATUS_HOST_SINK
        );
        assert!(replies.try_pop().is_none());
    }

    #[test]
    fn queued_playback_result_is_an_active_successor() {
        let state = PlaybackState::new();
        assert!(!state.active(false));
        let result = PlaybackResult {
            flow: Flow {
                session: 1,
                epoch: 2,
                ticket: Ticket {
                    runtime_epoch: 3,
                    sequence: 4,
                    generation: 5,
                    kind: TICKET_TURN,
                },
            },
            status: 0,
        };
        assert!(state.done.try_push(result));
        assert!(state.active(false));
        assert!(!state.audio_active(false));
        assert_eq!(state.done.try_pop().unwrap().flow, result.flow);
        assert!(!state.active(false));
    }

    #[test]
    fn playback_ready_callback_preserves_ticket_epoch_and_lease_identity() {
        let replies = ReplyRing::new();
        let mut sink = EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        };
        let ready = PlaybackReadyEvent {
            size: std::mem::size_of::<PlaybackReadyEvent>() as u32,
            abi_version: RUNTIME_ABI,
            lease_id: 0x1234,
            buffer_generation: 9,
        };
        let ticket = Ticket {
            runtime_epoch: 3,
            sequence: 5,
            generation: 7,
            kind: 2,
        };
        let event = NativeEvent {
            size: std::mem::size_of::<NativeEvent>() as u32,
            abi_version: RUNTIME_ABI,
            kind: EVENT_PLAYBACK_READY,
            flags: 0,
            session_id: 1,
            epoch: 13,
            ticket,
            payload: std::ptr::from_ref(&ready).cast(),
            payload_bytes: std::mem::size_of::<PlaybackReadyEvent>() as u32,
            status: 0,
        };
        assert_eq!(
            unsafe {
                on_event(
                    std::ptr::from_mut(&mut sink).cast(),
                    std::ptr::from_ref(&event),
                )
            },
            0
        );
        assert!(matches!(
            replies.try_pop().unwrap(),
            Reply::PlaybackReady {
                flow: Flow {
                    session: 1,
                    epoch: 13,
                    ticket: seen,
                },
                lease_id: 0x1234,
                buffer_generation: 9,
            } if seen == ticket
        ));
    }

    #[test]
    fn turn_started_callback_preserves_the_native_action_ticket() {
        let replies = ReplyRing::new();
        let mut sink = EventSink {
            replies: Arc::clone(&replies),
            resume: None,
        };
        let ticket = Ticket {
            runtime_epoch: 17,
            sequence: 23,
            generation: 4,
            kind: 2,
        };
        let event = NativeEvent {
            size: std::mem::size_of::<NativeEvent>() as u32,
            abi_version: RUNTIME_ABI,
            kind: EVENT_TURN_STARTED,
            flags: 0,
            session_id: 9,
            epoch: 17,
            ticket,
            payload: std::ptr::null(),
            payload_bytes: 0,
            status: 0,
        };
        assert_eq!(
            unsafe {
                on_event(
                    std::ptr::from_mut(&mut sink).cast(),
                    std::ptr::from_ref(&event),
                )
            },
            0
        );
        assert!(matches!(
            replies.try_pop(),
            Some(Reply::TurnStarted {
                flow: Flow {
                    session: 9,
                    epoch: 17,
                    ticket: seen,
                },
            }) if seen == ticket
        ));
    }
}
