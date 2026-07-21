//! Opaque Rust host seam for the native LFM2 voice runtime.
//!
//! This module deliberately exposes lifecycle, sampling policy, platform-audio
//! controls and semantic events only. Model bytes, weight-field names, token ids,
//! PCM leases, mel rows, codec codes and recurrence never cross this boundary.

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
    EngineProgress, PlatformAudioConfig, PlatformAudioSnapshot, VoiceEngine, VoiceEvent,
};

const RUNTIME_ABI: u32 = 4;
const STATUS_WOULD_BLOCK: i32 = -11;
const STATUS_BUSY: i32 = -16;
const STATUS_STALE: i32 = -116;
const STATUS_CANCELLED: i32 = -125;
const STATUS_HOST_SINK: i32 = -1002;
const STATUS_UNSUPPORTED: i32 = -1004;
const MODEL_ACCOUNTING_PAYLOAD_READS_COMPLETE: u32 = 1;
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

#[repr(C)]
struct Runtime {
    _private: [u8; 0],
}

#[repr(C)]
struct Session {
    _private: [u8; 0],
}

#[repr(C)]
struct SessionControlHandle {
    _private: [u8; 0],
}

#[repr(C)]
struct PlatformAudioHandle {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativePlatformAudioConfig {
    size: u32,
    abi_version: u32,
    capture_device: u32,
    playback_device: u32,
    capture_sample_rate: u32,
    playback_sample_rate: u32,
    capture_callback_frames: u32,
    playback_callback_frames: u32,
    flags: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativePlatformAudioSnapshot {
    size: u32,
    abi_version: u32,
    started: u32,
    capture_enabled: u32,
    terminal_status: i32,
    reserved0: u32,
    captured_frames: u64,
    dropped_capture_frames: u64,
    played_frames: u64,
    silent_playback_frames: u64,
    playback_leases: u64,
    playback_releases: u64,
    claimed_playback_frames: u64,
    dropped_playback_frames: u64,
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
    materialized_weight_bytes: u64,
    compatibility_copied_bytes: u64,
    payload_read_calls: u64,
    payload_read_bytes: u64,
    post_publication_read_calls: u64,
    post_publication_read_bytes: u64,
    post_publication_materialization_attempts: u64,
    post_publication_materialization_bytes: u64,
    publication_generation: u64,
    load_ns: u64,
    load_workers: u32,
    load_tasks: u32,
    payload_read_coverage: u32,
    accounting_flags: u32,
    post_readiness_allocation_attempts: u64,
    post_readiness_allocation_bytes: u64,
    reserved: [u64; 2],
}

const _: [(); 168] = [(); std::mem::size_of::<ModelMemory>()];
const _: [(); 136] = [(); std::mem::offset_of!(ModelMemory, post_readiness_allocation_attempts)];
const _: [(); 144] = [(); std::mem::offset_of!(ModelMemory, post_readiness_allocation_bytes)];

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
#[derive(Clone, Copy)]
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
    fn lfm_platform_audio_default_config(out: *mut NativePlatformAudioConfig) -> i32;
    fn lfm_platform_audio_create(
        session: *mut Session,
        config: *const NativePlatformAudioConfig,
        out: *mut *mut PlatformAudioHandle,
    ) -> i32;
    fn lfm_platform_audio_start(audio: *mut PlatformAudioHandle) -> i32;
    fn lfm_platform_audio_set_capture_enabled(audio: *mut PlatformAudioHandle, enabled: u32)
        -> i32;
    fn lfm_platform_audio_retire(audio: *mut PlatformAudioHandle) -> i32;
    fn lfm_platform_audio_snapshot(
        audio: *const PlatformAudioHandle,
        out: *mut NativePlatformAudioSnapshot,
    ) -> i32;
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
    pub materialized_weight_bytes: u64,
    pub compatibility_copied_bytes: u64,
    pub payload_read_calls: u64,
    pub payload_read_bytes: u64,
    pub post_publication_read_calls: u64,
    pub post_publication_read_bytes: u64,
    pub post_publication_materialization_attempts: u64,
    pub post_publication_materialization_bytes: u64,
    /// Rejected requests to change a conversation's numerical allocation
    /// geometry after its first complete capture/playback preparation.
    pub post_readiness_allocation_attempts: u64,
    /// Logical numerical capacity requested by those rejected calls. This is
    /// deliberately independent of allocator metadata and object overhead.
    pub post_readiness_allocation_bytes: u64,
    pub publication_generation: u64,
    /// Bitmask of model payload sources included in the read totals.
    pub payload_read_coverage: u32,
    /// True only when every possible model payload source is routed through
    /// the rejecting owner-scoped recorder.
    pub payload_read_accounting_complete: bool,
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
        if runtime_config.coordination_workers == 0
            || runtime_config.coordination_workers > 64
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
            materialized_weight_bytes: memory.materialized_weight_bytes,
            compatibility_copied_bytes: memory.compatibility_copied_bytes,
            payload_read_calls: memory.payload_read_calls,
            payload_read_bytes: memory.payload_read_bytes,
            post_publication_read_calls: memory.post_publication_read_calls,
            post_publication_read_bytes: memory.post_publication_read_bytes,
            post_publication_materialization_attempts: memory
                .post_publication_materialization_attempts,
            post_publication_materialization_bytes: memory.post_publication_materialization_bytes,
            post_readiness_allocation_attempts: memory.post_readiness_allocation_attempts,
            post_readiness_allocation_bytes: memory.post_readiness_allocation_bytes,
            publication_generation: memory.publication_generation,
            payload_read_coverage: memory.payload_read_coverage,
            payload_read_accounting_complete: memory.accounting_flags
                & MODEL_ACCOUNTING_PAYLOAD_READS_COMPLETE
                != 0,
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

const START_OPEN: usize = 0;
const START_WRITE: usize = 1;
const START_HELD: usize = 2;
const START_BLOCKED: usize = 3;

struct StartGate {
    flow: UnsafeCell<MaybeUninit<Flow>>,
    state: AtomicUsize,
}

// The native delivery continuation is the sole writer and the Rust event
// continuation is the sole reader. `state` transfers ownership of `flow`.
unsafe impl Sync for StartGate {}

impl StartGate {
    fn new() -> Self {
        Self {
            flow: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicUsize::new(START_OPEN),
        }
    }

    fn hold(&self, flow: Flow) -> bool {
        if self
            .state
            .compare_exchange(
                START_OPEN,
                START_WRITE,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }
        unsafe { (*self.flow.get()).write(flow) };
        self.state.store(START_HELD, Ordering::Release);
        true
    }

    fn block(&self) -> bool {
        match self.state.compare_exchange(
            START_HELD,
            START_BLOCKED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) | Err(START_BLOCKED | START_WRITE) => true,
            Err(START_OPEN) => false,
            Err(_) => unreachable!("invalid native turn-start gate state"),
        }
    }

    fn release(&self, flow: Flow) -> Result<bool, String> {
        const INVALID: &str = "native turn-start gate has no retained record";
        const CHANGED: &str = "native turn-start gate changed exact flow";
        const STATE: &str = "native turn-start gate entered an invalid state";
        let state = self.state.load(Ordering::Acquire);
        if state != START_HELD && state != START_BLOCKED {
            return Err(if state == START_OPEN { INVALID } else { STATE }.into());
        }
        let retained = unsafe { (*self.flow.get()).assume_init_read() };
        if retained != flow {
            return Err(format!(
                "{CHANGED} (expected={retained:?}, received={flow:?})"
            ));
        }
        match self
            .state
            .compare_exchange(state, START_OPEN, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(state == START_BLOCKED),
            Err(START_BLOCKED) if state == START_HELD => self
                .state
                .compare_exchange(
                    START_BLOCKED,
                    START_OPEN,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map(|_| true)
                .map_err(|state| if state == START_OPEN { INVALID } else { STATE }.into()),
            Err(state) => Err(if state == START_OPEN { INVALID } else { STATE }.into()),
        }
    }
}

struct ReplyRing {
    cells: Box<[ReplyCell]>,
    read: AtomicUsize,
    write: AtomicUsize,
    start: StartGate,
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
            start: StartGate::new(),
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

    fn settled(&self) -> bool {
        self.terminal.is_some()
    }
}

fn retain_successor(
    pending: &mut Option<Flow>,
    action: &NativeAction,
    flow: Flow,
) -> Result<(), String> {
    if action.ticket == flow.ticket {
        return Err("native action published a duplicate turn-start".into());
    }
    let prior = action
        .flow
        .ok_or("native successor turn-start followed an unbound prior action")?;
    if flow.session != prior.session
        || flow.epoch != prior.epoch
        || flow.ticket.runtime_epoch != prior.ticket.runtime_epoch
        || flow.ticket.generation != prior.ticket.generation
        || flow.ticket.kind != prior.ticket.kind
        || flow.ticket.sequence <= prior.ticket.sequence
    {
        return Err(format!(
            "native successor turn-start broke ticket lineage (prior={prior:?}, successor={flow:?})"
        ));
    }
    if let Some(successor) = pending {
        if *successor == flow {
            return Err("native action published a duplicate successor turn-start".into());
        }
        return Err("native action published more than one pending successor".into());
    }
    let Some((has_audio, playback)) = action.terminal else {
        return Err("native successor turn-start arrived before the prior terminal".into());
    };
    if !has_audio || action.playback >= playback {
        return Err("native successor turn-start arrived after the prior action settled".into());
    }
    *pending = Some(flow);
    Ok(())
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
        if sink.replies.start.block() {
            return Ok(STATUS_WOULD_BLOCK);
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
            EVENT_PLAYBACK_READY => return Err(STATUS_HOST_SINK),
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
        let start = match &reply {
            Reply::TurnStarted { flow } => Some(*flow),
            _ => None,
        };
        if start.is_some_and(|flow| !sink.replies.start.hold(flow)) {
            return Ok(STATUS_WOULD_BLOCK);
        }
        if sink.replies.try_push(reply).is_err() {
            if let Some(flow) = start {
                sink.replies
                    .start
                    .release(flow)
                    .map_err(|_| STATUS_HOST_SINK)?;
            }
            return Ok(STATUS_WOULD_BLOCK);
        }
        let Some(resume) = sink.resume.as_mut() else {
            // Setup may receive a native state record before the shared Rust
            // voice service installs its producer lease. The fixed record is
            // retained here and install_resume publishes the missing edge.
            return Ok(0);
        };
        if resume.notify().is_err() {
            if let Some(flow) = start {
                sink.replies
                    .start
                    .release(flow)
                    .map_err(|_| STATUS_HOST_SINK)?;
            }
            return Err(STATUS_HOST_SINK);
        }
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

struct NativePlatformAudio {
    handle: NonNull<PlatformAudioHandle>,
    retired: bool,
}

unsafe impl Send for NativePlatformAudio {}

impl NativePlatformAudio {
    fn start(&mut self) -> Result<(), String> {
        status(
            unsafe { lfm_platform_audio_start(self.handle.as_ptr()) },
            "start native platform audio",
        )
    }

    fn set_capture_enabled(&mut self, enabled: bool) -> Result<(), String> {
        status(
            unsafe {
                lfm_platform_audio_set_capture_enabled(self.handle.as_ptr(), u32::from(enabled))
            },
            "set native capture state",
        )
    }

    fn snapshot(&self) -> Result<PlatformAudioSnapshot, String> {
        let mut native = NativePlatformAudioSnapshot::default();
        status(
            unsafe { lfm_platform_audio_snapshot(self.handle.as_ptr(), &mut native) },
            "snapshot native platform audio",
        )?;
        if native.size as usize != std::mem::size_of::<NativePlatformAudioSnapshot>()
            || native.abi_version != RUNTIME_ABI
            || native.reserved0 != 0
        {
            return Err("native platform-audio snapshot broke its ABI".into());
        }
        Ok(PlatformAudioSnapshot {
            started: native.started != 0,
            capture_enabled: native.capture_enabled != 0,
            terminal_status: native.terminal_status,
            captured_frames: native.captured_frames,
            dropped_capture_frames: native.dropped_capture_frames,
            played_frames: native.played_frames,
            silent_playback_frames: native.silent_playback_frames,
            playback_leases: native.playback_leases,
            playback_releases: native.playback_releases,
            claimed_playback_frames: native.claimed_playback_frames,
            dropped_playback_frames: native.dropped_playback_frames,
        })
    }

    fn retire(&mut self) -> Result<(), String> {
        if self.retired {
            return Ok(());
        }
        status(
            unsafe { lfm_platform_audio_retire(self.handle.as_ptr()) },
            "retire native platform audio",
        )?;
        self.retired = true;
        Ok(())
    }
}

impl Drop for NativePlatformAudio {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            eprintln!("[flashkern] {error}");
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
    platform_audio: Option<NativePlatformAudio>,
    sink: Option<Box<EventSink>>,
    replies: Arc<ReplyRing>,
    active: Option<NativeAction>,
    pending_start: Option<Flow>,
    session_id: Option<u64>,
    control_epoch: u64,
    started: bool,
    stopped: bool,
    stopped_flow: Option<Flow>,
    last_terminal: Option<Flow>,
    joined: bool,
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
        Ok(Self {
            _model: model,
            conversation: Some(claim.into_conversation()),
            vault,
            healthy: true,
            session,
            control: Some(control),
            platform_audio: None,
            sink: Some(sink),
            replies,
            active: None,
            pending_start: None,
            session_id: None,
            control_epoch: 0,
            started: false,
            stopped: false,
            stopped_flow: None,
            last_terminal: None,
            joined: false,
        })
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

    fn release_start(&self, flow: Flow) -> Result<bool, String> {
        self.replies.start.release(flow)
    }

    fn resume_host(&self) -> Result<(), String> {
        const CANCELLED: [i32; 2] = [0, STATUS_CANCELLED];
        let capacity = unsafe { lfm_session_host_capacity(self.session.as_ptr()) };
        if CANCELLED.contains(&capacity) {
            return Ok(());
        }
        Err(format!(
            "resume native host-capacity edge failed with status {capacity}"
        ))
    }

    fn activate_pending_start(&mut self, emit: &mut dyn FnMut(VoiceEvent)) -> Result<bool, String> {
        if self.active.is_some() {
            return Ok(false);
        }
        let Some(flow) = self.pending_start.take() else {
            return Ok(false);
        };
        self.active = Some(NativeAction::new(flow.ticket));
        self.bind_action(flow)?;
        emit(VoiceEvent::TurnStarted);
        if self.release_start(flow)? {
            self.resume_host()?;
        }
        Ok(true)
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
        if self.active.is_some() || self.pending_start.is_some() {
            return Ok(false);
        }
        self.active = Some(NativeAction::new(ticket));
        Ok(true)
    }

    /// Submit one UTF-8 user turn and return after admission. Native completion
    /// resumes the retained Rust voice continuation through the installed edge.
    pub fn begin_text(&mut self, text: &str) -> Result<bool, String> {
        if text.is_empty() || text.len() > 2_048 {
            return Err("native typed input must contain 1..=2048 UTF-8 bytes".into());
        }
        if self.active.is_some() || self.pending_start.is_some() {
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
        self.activate_pending_start(emit)?;
        if let Some(action) = self.active.as_mut() {
            if cancel.load(Ordering::Acquire) {
                action.cancelled = true;
                action.text.reset();
            }
        }

        let complete = self.active.as_ref().is_some_and(NativeAction::settled);
        if complete {
            let action = self
                .active
                .take()
                .expect("completed native action disappeared");
            self.last_terminal = action.flow;
            if !action.cancelled {
                emit(VoiceEvent::TurnComplete);
            } else {
                emit(VoiceEvent::Interrupted);
            }
            return Ok(EngineProgress::Complete);
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
                        if let Err(error) = self.bind_action(flow) {
                            let _ = self.release_start(flow);
                            result = Err(error);
                            break;
                        }
                        emit(VoiceEvent::TurnStarted);
                        if let Err(error) = self.release_start(flow) {
                            result = Err(error);
                            break;
                        }
                        continue;
                    }
                    let same = self
                        .active
                        .as_ref()
                        .is_some_and(|action| action.ticket == flow.ticket);
                    if same {
                        if self
                            .active
                            .as_ref()
                            .is_some_and(|action| action.flow.is_some())
                        {
                            let _ = self.release_start(flow);
                            result = Err("native action published a duplicate turn-start".into());
                            break;
                        }
                        if let Err(error) = self.bind_action(flow) {
                            let _ = self.release_start(flow);
                            result = Err(error);
                            break;
                        }
                        if let Err(error) = self.release_start(flow) {
                            result = Err(error);
                            break;
                        }
                        continue;
                    }
                    if let Err(error) = self.accept_flow(flow) {
                        let _ = self.release_start(flow);
                        result = Err(error);
                        break;
                    }
                    if flow.ticket.kind != TICKET_TURN {
                        let _ = self.release_start(flow);
                        result = Err(format!(
                            "native successor turn-start carried ticket kind {} instead of {TICKET_TURN}",
                            flow.ticket.kind
                        ));
                        break;
                    }
                    let action = self.active.as_ref().expect("active checked above");
                    if let Err(error) = retain_successor(&mut self.pending_start, action, flow) {
                        let _ = self.release_start(flow);
                        result = Err(error);
                        break;
                    }
                    break;
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
                    #[cfg(test)]
                    {
                        action.emitted_items = emitted_items;
                    }
                    if status == STATUS_STALE || status == STATUS_CANCELLED {
                        action.cancelled = true;
                        action.text.reset();
                        // Native publishes this terminal only after every
                        // promised platform lease retires.
                        action.terminal = Some((has_audio, playback_leases));
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
                    let mut failure = self.pending_start.take().map(|successor| {
                        format!(
                            "native STOPPED skipped retained successor turn-start {successor:?}"
                        )
                    });
                    if status != 0 {
                        let status = format!("native voice session stopped with status {status}");
                        if let Some(previous) = failure.as_mut() {
                            previous.push_str("; ");
                            previous.push_str(&status);
                        } else {
                            failure = Some(status);
                        }
                    }
                    if let Some(failure) = failure {
                        self.healthy = false;
                        emit(VoiceEvent::Error(failure));
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

            let complete = self.active.as_ref().is_some_and(NativeAction::settled);
            if complete {
                let action = self
                    .active
                    .take()
                    .expect("completed native action disappeared");
                self.last_terminal = action.flow;
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
            if let Err(error) = self.resume_host() {
                if result.is_ok() {
                    result = Err(error);
                }
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
        if self.pending_start.is_some() || !self.replies.is_empty() {
            return Ok(EngineProgress::Continue);
        }
        Ok(EngineProgress::Dormant)
    }
}

impl VoiceEngine for NativeLfm2VoiceEngine {
    fn mount_platform_audio(&mut self, config: PlatformAudioConfig) -> Result<(), String> {
        if self.started {
            return Err("native platform audio must mount before session readiness".into());
        }
        if self.platform_audio.is_some() {
            return Err("native platform audio is already mounted".into());
        }
        let native = native_platform_audio_config(config);
        let mut handle = std::ptr::null_mut();
        status(
            unsafe { lfm_platform_audio_create(self.session.as_ptr(), &native, &mut handle) },
            "mount native platform audio",
        )?;
        let handle = NonNull::new(handle).ok_or("native platform audio returned a null handle")?;
        self.platform_audio = Some(NativePlatformAudio {
            handle,
            retired: false,
        });
        Ok(())
    }

    fn start_platform_audio(&mut self) -> Result<(), String> {
        self.platform_audio
            .as_mut()
            .ok_or("native platform audio is not mounted")?
            .start()
    }

    fn set_capture_enabled(&mut self, enabled: bool) -> Result<(), String> {
        self.platform_audio
            .as_mut()
            .ok_or("native platform audio is not mounted")?
            .set_capture_enabled(enabled)
    }

    fn platform_audio_snapshot(&self) -> Result<PlatformAudioSnapshot, String> {
        self.platform_audio
            .as_ref()
            .ok_or("native platform audio is not mounted")?
            .snapshot()
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
        /* Closing native numerical admission and retiring the hardware dock
         * are one ownership edge. Deferring CoreAudio retirement until after
         * the outer kcoro service joins can deadlock when playback is the only
         * remaining producer of lease-retirement callbacks. */
        if let Some(audio) = self.platform_audio.as_mut() {
            if let Err(error) = audio.retire() {
                self.healthy = false;
                eprintln!("[flashkern] {error}");
            }
        }
    }

    fn stop_session(&mut self) -> Result<(), String> {
        if self.joined {
            return Ok(());
        }
        // Close native admission and callback admission before the terminal
        // administrative latch. Destroying a live endpoint without this edge
        // remains a real device-loss fault.
        self.request_stop();
        self.control.take();
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
        self.platform_audio.take();
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
    if code == STATUS_UNSUPPORTED {
        return Err(format!(
            "{operation}: this platform has no production monotonic deadline backend"
        ));
    }
    Err(format!("{operation} failed with native status {code}"))
}

pub fn default_platform_audio_config() -> Result<PlatformAudioConfig, String> {
    let mut native = NativePlatformAudioConfig::default();
    status(
        unsafe { lfm_platform_audio_default_config(&mut native) },
        "query native platform audio",
    )?;
    if native.size as usize != std::mem::size_of::<NativePlatformAudioConfig>()
        || native.abi_version != RUNTIME_ABI
        || native.flags != 0
        || native.reserved0 != 0
        || native.reserved != [0; 4]
        || native.capture_device == 0
        || native.playback_device == 0
        || native.capture_sample_rate == 0
        || native.playback_sample_rate == 0
        || native.capture_callback_frames == 0
        || native.playback_callback_frames == 0
    {
        return Err("native platform-audio query returned an invalid contract".into());
    }
    Ok(PlatformAudioConfig {
        capture_device: native.capture_device,
        playback_device: native.playback_device,
        capture_rate: native.capture_sample_rate,
        playback_rate: native.playback_sample_rate,
        capture_frames: native.capture_callback_frames,
        playback_frames: native.playback_callback_frames,
    })
}

fn native_platform_audio_config(config: PlatformAudioConfig) -> NativePlatformAudioConfig {
    NativePlatformAudioConfig {
        size: std::mem::size_of::<NativePlatformAudioConfig>() as u32,
        abi_version: RUNTIME_ABI,
        capture_device: config.capture_device,
        playback_device: config.playback_device,
        capture_sample_rate: config.capture_rate,
        playback_sample_rate: config.playback_rate,
        capture_callback_frames: config.capture_frames,
        playback_callback_frames: config.playback_frames,
        flags: 0,
        reserved0: 0,
        reserved: [0; 4],
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
