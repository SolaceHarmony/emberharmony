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
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kcoro_sys::RealtimeNotifier;

use crate::ffi;
use crate::voice_api::{
    CaptureDock, CaptureReservation, CaptureStorage, CaptureTicket, EngineProgress, PlaybackSource,
    PlaybackWrite, VoiceEngine, VoiceEvent,
};

const RUNTIME_ABI: u32 = 1;
const STATUS_WOULD_BLOCK: i32 = -11;
const STATUS_BUSY: i32 = -16;
const STATUS_STALE: i32 = -116;
const STATUS_CANCELLED: i32 = -125;
const STATUS_HOST_SINK: i32 = -1002;
const PCM_F32: u32 = 1;
const EVENT_STATE: u32 = 1;
const EVENT_TEXT: u32 = 2;
const EVENT_TURN: u32 = 3;
const EVENT_ERROR: u32 = 4;
const EVENT_STOPPED: u32 = 5;
const EVENT_PLAYBACK_READY: u32 = 6;
const EVENT_HAS_AUDIO: u32 = 1;
const EVENT_TRUNCATED: u32 = 2;
const TICKET_CONTROL: u32 = 8;
const REPLY_CAPACITY: usize = 128;
const TEXT_EVENT_MAX_BYTES: usize = 512;
const UTF8_CARRY_MAX_BYTES: usize = 3;
const EVENT_CAPACITY: u32 = 64;
const MAX_KERNEL_LANES: u32 = 16;
const CAPTURE_SLOTS: u32 = 2;
const PLAYBACK_SLOTS: u32 = 8;
const MAX_CAPTURE_FRAMES: u32 = 48_000 * 30;

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

type Ticket = CaptureTicket;

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
    fn lfm_capture_producer_create(session: *mut Session, out: *mut *mut CaptureProducer) -> i32;
    fn lfm_capture_producer_reserve(
        producer: *mut CaptureProducer,
        frames: u32,
        sample_rate: u32,
        out: *mut PcmLease,
    ) -> i32;
    fn lfm_capture_producer_resolve_mut(
        producer: *mut CaptureProducer,
        lease: *const PcmLease,
        out_samples: *mut *mut f32,
        out_sample_capacity: *mut usize,
    ) -> i32;
    fn lfm_capture_producer_finalize(
        producer: *mut CaptureProducer,
        lease: *mut PcmLease,
        offset_frames: u32,
        used_frames: u32,
    ) -> i32;
    fn lfm_capture_producer_publish(producer: *mut CaptureProducer, lease: *const PcmLease) -> i32;
    fn lfm_capture_producer_release(producer: *mut CaptureProducer, lease: *const PcmLease) -> i32;
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
        playback_rate: u32,
    ) -> Result<NativeLfm2VoiceEngine, String> {
        NativeLfm2VoiceEngine::new(self.clone(), sampling, vault, playback_rate)
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
            let mut state = vault
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        let mut state = vault
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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

enum Reply {
    Text {
        ticket: Ticket,
        payload: TextPayload,
    },
    PlaybackReady {
        ticket: Ticket,
        epoch: u64,
        lease_id: u64,
        buffer_generation: u64,
    },
    Turn {
        ticket: Ticket,
        status: i32,
        has_audio: bool,
        truncated: bool,
        playback_leases: u32,
    },
    Error {
        ticket: Ticket,
        status: i32,
        payload: TextPayload,
    },
    Stopped(i32),
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
    ticket: Ticket,
    epoch: u64,
    lease_id: u64,
    buffer_generation: u64,
}

#[derive(Clone, Copy)]
struct PlaybackResult {
    ticket: Ticket,
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
    text: Utf8Stream,
    playback: u32,
    terminal: Option<(bool, u32)>,
    cancelled: bool,
}

impl NativeAction {
    fn new(ticket: Ticket) -> Self {
        Self {
            ticket,
            text: Utf8Stream::default(),
            playback: 0,
            terminal: None,
            cancelled: false,
        }
    }
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
                ticket: event.ticket,
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
                    ticket: event.ticket,
                    status: event.status,
                    has_audio: event.flags & EVENT_HAS_AUDIO != 0,
                    truncated: event.flags & EVENT_TRUNCATED != 0,
                    playback_leases: turn.playback_leases,
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
                    ticket: event.ticket,
                    epoch: event.epoch,
                    lease_id: ready.lease_id,
                    buffer_generation: ready.buffer_generation,
                })
            }
            EVENT_ERROR => Some(Reply::Error {
                ticket: event.ticket,
                status: event.status,
                payload: TextPayload::new(bytes).ok_or(STATUS_HOST_SINK)?,
            }),
            EVENT_STOPPED => Some(Reply::Stopped(event.status)),
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
    fn interrupt(&self) {
        let mut epoch = 0;
        let code = unsafe { lfm_session_control_interrupt(self.0.as_ptr(), &mut epoch) };
        if code != 0 && code != STATUS_CANCELLED {
            eprintln!("[flashkern] native session interrupt failed with status {code}");
        }
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

struct NativeCaptureProducer(NonNull<CaptureProducer>);

unsafe impl Send for NativeCaptureProducer {}
unsafe impl Sync for NativeCaptureProducer {}

impl Drop for NativeCaptureProducer {
    fn drop(&mut self) {
        let status = unsafe { lfm_capture_producer_destroy(self.0.as_ptr()) };
        if status != 0 {
            eprintln!("[flashkern] native capture producer retired late with status {status}");
        }
    }
}

struct NativeCaptureStorage {
    producer: Arc<NativeCaptureProducer>,
    lease: UnsafeCell<PcmLease>,
    samples: AtomicPtr<f32>,
    capacity: AtomicUsize,
    frames: u32,
    rate: u32,
    active: AtomicBool,
    published: AtomicBool,
}

unsafe impl Send for NativeCaptureStorage {}
unsafe impl Sync for NativeCaptureStorage {}

impl NativeCaptureStorage {
    fn reserve(
        producer: Arc<NativeCaptureProducer>,
        frames: usize,
        rate: u32,
    ) -> Result<Option<Arc<CaptureReservation>>, String> {
        if frames == 0 || rate == 0 {
            return Err("native capture reservation geometry is empty".into());
        }
        let frames = u32::try_from(frames)
            .map_err(|_| "native capture reservation exceeds u32 frames".to_string())?;
        if frames > MAX_CAPTURE_FRAMES {
            return Err(format!(
                "native capture reservation exceeds the {MAX_CAPTURE_FRAMES}-frame bound"
            ));
        }
        let mut storage = Self {
            producer,
            lease: UnsafeCell::new(PcmLease::default()),
            samples: AtomicPtr::new(std::ptr::null_mut()),
            capacity: AtomicUsize::new(0),
            frames,
            rate,
            active: AtomicBool::new(false),
            published: AtomicBool::new(false),
        };
        match storage.arm() {
            Ok(true) => Ok(Some(CaptureReservation::new(Arc::new(storage)))),
            Ok(false) => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn arm(&mut self) -> Result<bool, String> {
        let mut lease = PcmLease::default();
        let reserve = unsafe {
            lfm_capture_producer_reserve(
                self.producer.0.as_ptr(),
                self.frames,
                self.rate,
                &mut lease,
            )
        };
        if reserve == STATUS_WOULD_BLOCK || reserve == STATUS_STALE || reserve == STATUS_CANCELLED {
            return Ok(false);
        }
        status(reserve, "reserve native capture producer")?;
        if lease.format != PCM_F32 || lease.channels != 1 {
            let _ = unsafe { lfm_capture_producer_release(self.producer.0.as_ptr(), &lease) };
            return Err("native capture producer returned an unsupported PCM view".into());
        }
        let mut samples = std::ptr::null_mut();
        let mut capacity = 0usize;
        let resolve = unsafe {
            lfm_capture_producer_resolve_mut(
                self.producer.0.as_ptr(),
                &lease,
                &mut samples,
                &mut capacity,
            )
        };
        if resolve != 0 || samples.is_null() || capacity < self.frames as usize {
            let _ = unsafe { lfm_capture_producer_release(self.producer.0.as_ptr(), &lease) };
            return Err(format!(
                "resolve native capture producer failed with status {resolve}"
            ));
        }
        unsafe { *self.lease.get() = lease };
        self.samples.store(samples, Ordering::Release);
        self.capacity.store(capacity, Ordering::Release);
        self.published.store(false, Ordering::Release);
        self.active.store(true, Ordering::Release);
        Ok(true)
    }
}

impl CaptureStorage for NativeCaptureStorage {
    fn samples(&self) -> *mut f32 {
        self.samples.load(Ordering::Acquire)
    }

    fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    fn publish(&self, offset: usize, frames: usize) -> Result<Option<CaptureTicket>, String> {
        let offset = u32::try_from(offset)
            .map_err(|_| "native capture offset exceeds u32 frames".to_string())?;
        let frames = u32::try_from(frames)
            .map_err(|_| "native capture length exceeds u32 frames".to_string())?;
        let lease = unsafe { &mut *self.lease.get() };
        let finalize = unsafe {
            lfm_capture_producer_finalize(self.producer.0.as_ptr(), lease, offset, frames)
        };
        if finalize == STATUS_STALE || finalize == STATUS_CANCELLED {
            return Ok(None);
        }
        status(finalize, "finalize native capture view")?;
        let publish = unsafe { lfm_capture_producer_publish(self.producer.0.as_ptr(), lease) };
        if publish == STATUS_STALE || publish == STATUS_CANCELLED {
            return Ok(None);
        }
        status(publish, "publish native capture view")?;
        self.published.store(true, Ordering::Release);
        Ok(Some(lease.ticket))
    }

    fn try_rearm(&self) -> Result<bool, String> {
        let mut lease = PcmLease::default();
        let reserve = unsafe {
            lfm_capture_producer_reserve(
                self.producer.0.as_ptr(),
                self.frames,
                self.rate,
                &mut lease,
            )
        };
        if reserve == STATUS_WOULD_BLOCK || reserve == STATUS_STALE || reserve == STATUS_CANCELLED {
            return Ok(false);
        }
        status(reserve, "rearm native capture producer")?;
        let mut samples = std::ptr::null_mut();
        let mut capacity = 0usize;
        let resolve = unsafe {
            lfm_capture_producer_resolve_mut(
                self.producer.0.as_ptr(),
                &lease,
                &mut samples,
                &mut capacity,
            )
        };
        if resolve != 0 || samples.is_null() || capacity < self.frames as usize {
            let _ = unsafe { lfm_capture_producer_release(self.producer.0.as_ptr(), &lease) };
            return Err(format!(
                "resolve rearmed native capture producer failed with status {resolve}"
            ));
        }
        unsafe { *self.lease.get() = lease };
        self.samples.store(samples, Ordering::Release);
        self.capacity.store(capacity, Ordering::Release);
        self.published.store(false, Ordering::Release);
        self.active.store(true, Ordering::Release);
        Ok(true)
    }

    fn release(&self) {
        if !self.active.load(Ordering::Acquire) || self.published.load(Ordering::Acquire) {
            return;
        }
        let lease = unsafe { &*self.lease.get() };
        let release = unsafe { lfm_capture_producer_release(self.producer.0.as_ptr(), lease) };
        if release == 0 || release == STATUS_STALE || release == STATUS_CANCELLED {
            self.active.store(false, Ordering::Release);
        }
    }
}

impl Drop for NativeCaptureStorage {
    fn drop(&mut self) {
        self.release();
    }
}

struct NativeCaptureDock {
    producer: Arc<NativeCaptureProducer>,
}

unsafe impl Send for NativeCaptureDock {}

impl CaptureDock for NativeCaptureDock {
    fn reserve(
        &mut self,
        frames: usize,
        rate: u32,
    ) -> Result<Option<Arc<CaptureReservation>>, String> {
        NativeCaptureStorage::reserve(self.producer.clone(), frames, rate)
    }
}

struct ClaimedPlayback {
    lease: PcmLease,
    samples: *const f32,
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
    fn publish_result(&mut self, result: PlaybackResult) -> bool {
        if !self.state.done.try_push(result) {
            self.result = Some(result);
            return false;
        }
        let _ = self.notify.notify();
        true
    }

    fn flush_result(&mut self) -> bool {
        let Some(result) = self.result else {
            return true;
        };
        if !self.state.done.try_push(result) {
            return false;
        }
        self.result = None;
        let _ = self.notify.notify();
        true
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
            ticket: current.lease.ticket,
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
                &notice.ticket,
                notice.epoch,
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
                ticket: notice.ticket,
                status: 0,
            });
            return false;
        }
        if claim != 0 {
            let _ = self.publish_result(PlaybackResult {
                ticket: notice.ticket,
                status: claim,
            });
            return false;
        }
        if lease.ticket != notice.ticket
            || lease.stream_epoch != notice.epoch
            || lease.lease_id != notice.lease_id
            || lease.buffer_generation != notice.buffer_generation
        {
            let _ = unsafe { lfm_playback_consumer_release(self.consumer.as_ptr(), &lease) };
            let _ = self.publish_result(PlaybackResult {
                ticket: notice.ticket,
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
                ticket: notice.ticket,
                status,
            });
            return false;
        }
        write.claimed_samples = write.claimed_samples.saturating_add(count);
        self.current = Some(ClaimedPlayback {
            lease,
            samples,
            count,
            cursor: 0,
        });
        true
    }

    fn revalidate(&mut self) -> bool {
        let Some(current) = self.current.as_mut() else {
            return false;
        };
        let mut samples = std::ptr::null();
        let mut count = 0usize;
        let status = unsafe {
            lfm_playback_consumer_resolve(
                self.consumer.as_ptr(),
                &current.lease,
                &mut samples,
                &mut count,
            )
        };
        if status == 0 && !samples.is_null() && count == current.count {
            current.samples = samples;
            return true;
        }
        let status = if status == STATUS_STALE || status == STATUS_CANCELLED {
            0
        } else if status != 0 {
            status
        } else {
            STATUS_HOST_SINK
        };
        self.finish_current(status);
        false
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
        convert: impl Fn(f32) -> T,
    ) -> PlaybackWrite {
        output.fill(silence);
        let mut write = PlaybackWrite::default();
        if channels == 0 || output.len() % channels != 0 {
            return write;
        }
        if !self.flush_result() {
            write.active = true;
            write.underrun_frames = output.len() / channels;
            return write;
        }
        if flush {
            self.discard(&mut write);
            write.active = self.result.is_some()
                || self.notice.is_some()
                || self.current.is_some()
                || !self.state.ready.is_empty();
            return write;
        }

        let frames = output.len() / channels;
        let mut frame = 0usize;
        let mut sum = 0.0f32;
        while frame < frames {
            if self.current.is_none() && !self.claim(&mut write) {
                break;
            }
            if !self.revalidate() {
                if self.result.is_some() {
                    break;
                }
                continue;
            }
            let current = self
                .current
                .as_mut()
                .expect("revalidated playback disappeared");
            let count = (frames - frame).min(current.count - current.cursor);
            for offset in 0..count {
                let sample = unsafe { current.samples.add(current.cursor + offset).read() };
                sum += sample * sample;
                let sample = convert(sample);
                output[(frame + offset) * channels..(frame + offset + 1) * channels].fill(sample);
            }
            current.cursor += count;
            frame += count;
            if current.cursor == current.count {
                self.finish_current(0);
                if self.result.is_some() {
                    break;
                }
            }
        }
        write.played_frames = frame;
        if frame != 0 {
            write.rms = (sum / frame as f32).sqrt();
        }
        write.active = self.result.is_some()
            || self.notice.is_some()
            || self.current.is_some()
            || !self.state.ready.is_empty();
        if write.active && frame < frames {
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
        self.write(output, channels, flush, 0.0, |sample| sample)
    }

    fn write_i16(&mut self, output: &mut [i16], channels: usize, flush: bool) -> PlaybackWrite {
        self.write(output, channels, flush, 0, |sample| {
            (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
        })
    }

    fn write_u16(&mut self, output: &mut [u16], channels: usize, flush: bool) -> PlaybackWrite {
        self.write(output, channels, flush, u16::MAX / 2, |sample| {
            ((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i32 + 32768) as u16
        })
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
                    &ready.ticket,
                    ready.epoch,
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
    capture: Option<Arc<NativeCaptureProducer>>,
    capture_taken: bool,
    sink: Option<Box<EventSink>>,
    replies: Arc<ReplyRing>,
    active: Option<NativeAction>,
    playback: Arc<PlaybackState>,
    pending_playback: Option<PlaybackNotice>,
    playback_taken: bool,
    playback_rate: u32,
    started: bool,
    joined: bool,
}

unsafe impl Send for NativeLfm2VoiceEngine {}

impl NativeLfm2VoiceEngine {
    fn new(
        model: NativeVoiceModel,
        sampling: NativeVoiceSampling,
        vault: Option<NativeConversationVault>,
        playback_rate: u32,
    ) -> Result<Self, String> {
        if playback_rate == 0 {
            return Err("native playback sample rate is zero".into());
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
            capture_slots: CAPTURE_SLOTS,
            playback_slots: PLAYBACK_SLOTS,
            capture_frames_per_slot: MAX_CAPTURE_FRAMES,
            // Zero delegates model/codec/rate geometry to native readiness.
            // Rust must not encode Mimi's frame capacity as model knowledge.
            playback_frames_per_slot: 0,
            pcm_channels: 1,
            pcm_sample_rate: playback_rate,
            command_capacity: 8,
            max_new_tokens: sampling.max_new_tokens,
            reserved0: 0,
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
            unsafe { lfm_capture_producer_create(session.as_ptr(), &mut producer) },
            "create native capture producer",
        ) {
            drop(control);
            retire_unstarted_session(session);
            return Err(error);
        }
        let Some(producer) = NonNull::new(producer) else {
            drop(control);
            retire_unstarted_session(session);
            return Err("native capture producer returned a null handle".into());
        };
        let capture = Arc::new(NativeCaptureProducer(producer));
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
            started: false,
            joined: false,
        })
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
                continue;
            };
            if action.ticket == done.ticket {
                action.playback = action.playback.saturating_add(1);
            }
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
                Reply::Text {
                    ticket: reply_ticket,
                    payload,
                } => {
                    let Some(action) = self.active.as_mut() else {
                        continue;
                    };
                    if reply_ticket == action.ticket && !action.cancelled {
                        if let Err(error) = action.text.push(payload.as_bytes(), &mut |piece| {
                            emit(VoiceEvent::Text(piece))
                        }) {
                            result = Err(error);
                            break;
                        }
                    }
                }
                Reply::PlaybackReady {
                    ticket: reply_ticket,
                    epoch,
                    lease_id,
                    buffer_generation,
                } => {
                    if !self.queue_playback(PlaybackNotice {
                        ticket: reply_ticket,
                        epoch,
                        lease_id,
                        buffer_generation,
                    }) {
                        break;
                    }
                }
                Reply::Turn {
                    ticket: reply_ticket,
                    status,
                    has_audio,
                    truncated,
                    playback_leases,
                } => {
                    let Some(action) = self.active.as_mut() else {
                        continue;
                    };
                    if reply_ticket != action.ticket {
                        continue;
                    }
                    if status == STATUS_STALE || status == STATUS_CANCELLED {
                        action.cancelled = true;
                        action.text.reset();
                        action.terminal = Some((false, 0));
                    } else if status != 0 {
                        result = Err(format!("native turn failed with status {status}"));
                        break;
                    } else {
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
                    ticket: reply_ticket,
                    status,
                    payload,
                } => {
                    let applies = reply_ticket.kind == TICKET_CONTROL
                        || self
                            .active
                            .as_ref()
                            .is_some_and(|action| action.ticket == reply_ticket);
                    if applies {
                        result = Err(format!(
                            "{} (native status {status})",
                            String::from_utf8_lossy(payload.as_bytes())
                        ));
                        break;
                    }
                }
                Reply::Stopped(status) => {
                    result = Err(format!("native voice session stopped with status {status}"));
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
            unsafe { lfm_session_request_stop(self.session.as_ptr()) };
            return result;
        }
        if result == Ok(EngineProgress::Complete) {
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
    fn take_capture_dock(&mut self) -> Result<Option<Box<dyn CaptureDock>>, String> {
        if self.capture_taken {
            return Err("native capture producer was already transferred".into());
        }
        let Some(producer) = self.capture.as_ref() else {
            return Ok(None);
        };
        self.capture_taken = true;
        Ok(Some(Box::new(NativeCaptureDock {
            producer: producer.clone(),
        })))
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

    fn mount_events(&mut self, notify: RealtimeNotifier) -> Result<bool, String> {
        self.install_resume(notify)?;
        Ok(true)
    }

    fn begin_capture(&mut self, ticket: CaptureTicket) -> Result<bool, String> {
        self.begin_ticket(ticket)
    }

    fn advance_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        self.drain_events(cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        if let Some(control) = self.control.as_ref() {
            control.interrupt();
        }
        Ok(())
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
        self.sink.take();
        if destroy != 0 {
            eprintln!("[flashkern] native voice session destroy refused with status {destroy}");
            if let Some(conversation) = self.conversation.take() {
                std::mem::forget(conversation);
            }
            return;
        }
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
        let mut state = vault
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
                ticket: seen,
                epoch: 13,
                lease_id: 0x1234,
                buffer_generation: 9,
            } if seen == ticket
        ));
    }
}
