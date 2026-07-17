//! Opaque Rust host seam for the native LFM2 voice runtime.
//!
//! This module deliberately exposes lifecycle, sampling policy, PCM leases and
//! semantic events only. Model bytes, tensor names, token ids, mel rows, codec
//! codes and recurrence never cross this boundary.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::ffi;
use crate::voice_api::{Utterance, VoiceEngine, VoiceEvent};

const RUNTIME_ABI: u32 = 1;
const STATUS_BUSY: i32 = -16;
const STATUS_STALE: i32 = -116;
const STATUS_CANCELLED: i32 = -125;
const PCM_CAPTURE: u32 = 1;
const PCM_F32: u32 = 1;
const EVENT_STATE: u32 = 1;
const EVENT_TEXT: u32 = 2;
const EVENT_TURN: u32 = 3;
const EVENT_ERROR: u32 = 4;
const EVENT_STOPPED: u32 = 5;
const EVENT_HAS_AUDIO: u32 = 1;
const EVENT_TRUNCATED: u32 = 2;
const TICKET_CONTROL: u32 = 4;
const REPLY_CAPACITY: usize = 128;
const EVENT_CAPACITY: u32 = 64;
const CAPTURE_SLOTS: u32 = 1;
const PLAYBACK_SLOTS: u32 = 8;
const MAX_CAPTURE_FRAMES: u32 = 48_000 * 30;
const PLAYBACK_FRAMES: u32 = 3_840;

#[repr(C)]
struct Runtime {
    _private: [u8; 0],
}

#[repr(C)]
struct Session {
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
        // uninitialised state supplied to reserve/wait calls.
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
    fn lfm_session_wait_submit_text(
        session: *mut Session,
        utf8: *const c_char,
        utf8_bytes: usize,
        out_ticket: *mut Ticket,
    ) -> i32;
    fn lfm_session_wait_submit_mixed(
        session: *mut Session,
        utf8: *const c_char,
        utf8_bytes: usize,
        capture: *const PcmLease,
        out_ticket: *mut Ticket,
    ) -> i32;
    fn lfm_session_interrupt(session: *mut Session, out_epoch: *mut u64) -> i32;
    fn lfm_session_request_stop(session: *mut Session);
    fn lfm_session_join(session: *mut Session) -> i32;
    fn lfm_session_destroy(session: *mut Session) -> i32;
    fn lfm_audio_dock_wait_reserve(
        session: *mut Session,
        direction: u32,
        frames: u32,
        sample_rate: u32,
        out: *mut PcmLease,
    ) -> i32;
    fn lfm_audio_dock_resolve_mut(
        session: *mut Session,
        lease: *const PcmLease,
        out_samples: *mut *mut f32,
        out_sample_capacity: *mut usize,
    ) -> i32;
    fn lfm_audio_dock_resolve(
        session: *const Session,
        lease: *const PcmLease,
        out_samples: *mut *const f32,
        out_sample_count: *mut usize,
    ) -> i32;
    fn lfm_audio_dock_publish(session: *mut Session, lease: *const PcmLease) -> i32;
    fn lfm_audio_dock_wait_playback(session: *mut Session, out: *mut PcmLease) -> i32;
    fn lfm_audio_dock_release(session: *mut Session, lease: *const PcmLease) -> i32;
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
            || runtime_config.kernel_lanes > 64
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
    ) -> Result<NativeLfm2VoiceEngine, String> {
        NativeLfm2VoiceEngine::new(self.clone(), sampling, vault)
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
        flags: temperature
            .is_none()
            .then_some(1)
            .unwrap_or(0),
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
        flags: sampling
            .seed
            .is_none()
            .then_some(1)
            .unwrap_or(0),
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
        text: String,
    },
    Audio {
        pcm: Vec<f32>,
        rate: u32,
        ticket: Ticket,
    },
    Turn {
        ticket: Ticket,
        status: i32,
        has_audio: bool,
        truncated: bool,
        playback_leases: u32,
    },
    Interrupted {
        ticket: Ticket,
    },
    Error {
        ticket: Option<Ticket>,
        error: String,
    },
    Stopped(i32),
}

struct EventSink {
    tx: Sender<Reply>,
    shutdown: Receiver<()>,
}

fn send_reply(tx: &Sender<Reply>, shutdown: &Receiver<()>, reply: Reply) -> Result<(), ()> {
    crossbeam_channel::select! {
        send(tx, reply) -> result => result.map_err(|_| ()),
        recv(shutdown) -> _ => Err(()),
    }
}

unsafe extern "C" fn on_event(context: *mut c_void, event: *const NativeEvent) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if context.is_null() || event.is_null() {
            return Err(());
        }
        let sink = unsafe { &*(context.cast::<EventSink>()) };
        let event = unsafe { &*event };
        if event.size != std::mem::size_of::<NativeEvent>() as u32
            || event.abi_version != RUNTIME_ABI
        {
            return Err(());
        }
        let bytes = if event.payload_bytes == 0 {
            &[][..]
        } else {
            if event.payload.is_null() {
                return Err(());
            }
            unsafe {
                std::slice::from_raw_parts(event.payload.cast::<u8>(), event.payload_bytes as usize)
            }
        };
        let reply = match event.kind {
            EVENT_STATE if bytes == b"interrupted" => Some(Reply::Interrupted {
                ticket: event.ticket,
            }),
            EVENT_STATE => None,
            EVENT_TEXT => Some(Reply::Text {
                ticket: event.ticket,
                text: String::from_utf8_lossy(bytes).into_owned(),
            }),
            EVENT_TURN => {
                if bytes.len() != std::mem::size_of::<TurnEvent>() {
                    return Err(());
                }
                let mut turn = TurnEvent {
                    size: 0,
                    abi_version: 0,
                    playback_leases: 0,
                    emitted_items: 0,
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        (&mut turn as *mut TurnEvent).cast::<u8>(),
                        bytes.len(),
                    );
                }
                if turn.size != std::mem::size_of::<TurnEvent>() as u32
                    || turn.abi_version != RUNTIME_ABI
                {
                    return Err(());
                }
                Some(Reply::Turn {
                    ticket: event.ticket,
                    status: event.status,
                    has_audio: event.flags & EVENT_HAS_AUDIO != 0,
                    truncated: event.flags & EVENT_TRUNCATED != 0,
                    playback_leases: turn.playback_leases,
                })
            }
            EVENT_ERROR => Some(Reply::Error {
                ticket: Some(event.ticket),
                error: format!(
                    "{} (native status {})",
                    String::from_utf8_lossy(bytes),
                    event.status
                ),
            }),
            EVENT_STOPPED => Some(Reply::Stopped(event.status)),
            _ => return Err(()),
        };
        match reply {
            Some(reply) => send_reply(&sink.tx, &sink.shutdown, reply),
            None => Ok(()),
        }
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(())) | Err(_) => 1,
    }
}

struct SessionControl(Mutex<Option<NonNull<Session>>>);

unsafe impl Send for SessionControl {}
unsafe impl Sync for SessionControl {}

impl SessionControl {
    fn interrupt(&self) {
        let session = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(session) = *session else {
            return;
        };
        let mut epoch = 0;
        let code = unsafe { lfm_session_interrupt(session.as_ptr(), &mut epoch) };
        if code != 0 && code != STATUS_CANCELLED {
            eprintln!("[flashkern] native session interrupt failed with status {code}");
        }
    }
}

/// Turn-based `VoiceEngine` backed entirely by the native LFM2 session.
pub struct NativeLfm2VoiceEngine {
    _model: NativeVoiceModel,
    conversation: Option<ConversationOwner>,
    vault: Option<NativeConversationVault>,
    session: NonNull<Session>,
    control: Arc<SessionControl>,
    sink: Option<Box<EventSink>>,
    replies: Receiver<Reply>,
    shutdown: Option<Sender<()>>,
    playback: Option<JoinHandle<()>>,
}

unsafe impl Send for NativeLfm2VoiceEngine {}

impl NativeLfm2VoiceEngine {
    fn new(
        model: NativeVoiceModel,
        sampling: NativeVoiceSampling,
        vault: Option<NativeConversationVault>,
    ) -> Result<Self, String> {
        let claim = ConversationClaim::new(&model, sampling, vault.clone())?;
        let (tx, replies) = bounded(REPLY_CAPACITY);
        let (shutdown, shutdown_rx) = bounded(0);
        let mut sink = Box::new(EventSink {
            tx: tx.clone(),
            shutdown: shutdown_rx.clone(),
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
            playback_frames_per_slot: PLAYBACK_FRAMES,
            pcm_channels: 1,
            pcm_sample_rate: 48_000,
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
        if let Err(error) = status(
            unsafe { lfm_session_start(session.as_ptr()) },
            "start native voice session",
        ) {
            let _ = unsafe { lfm_session_join(session.as_ptr()) };
            let _ = unsafe { lfm_session_destroy(session.as_ptr()) };
            return Err(error);
        }
        let control = Arc::new(SessionControl(Mutex::new(Some(session))));
        let playback = match spawn_playback(session, tx, shutdown_rx) {
            Ok(playback) => playback,
            Err(error) => {
                drop(shutdown);
                unsafe { lfm_session_request_stop(session.as_ptr()) };
                let _ = unsafe { lfm_session_join(session.as_ptr()) };
                let _ = unsafe { lfm_session_destroy(session.as_ptr()) };
                return Err(error);
            }
        };
        Ok(Self {
            _model: model,
            conversation: Some(claim.into_conversation()),
            vault,
            session,
            control,
            sink: Some(sink),
            replies,
            shutdown: Some(shutdown),
            playback: Some(playback),
        })
    }

    fn capture(&self, utterance: &Utterance) -> Result<Option<PcmLease>, String> {
        if utterance.samples.is_empty() || utterance.rate == 0 {
            return Err("native voice utterance must contain PCM at a nonzero rate".into());
        }
        if utterance.rate != 48_000 {
            return Err(format!(
                "native voice session was prepared for 48000 Hz capture, received {} Hz",
                utterance.rate
            ));
        }
        let frames = u32::try_from(utterance.samples.len())
            .map_err(|_| "native voice utterance is too large".to_string())?;
        if frames > MAX_CAPTURE_FRAMES {
            return Err(format!(
                "native voice utterance exceeds the {}-frame lease bound",
                MAX_CAPTURE_FRAMES
            ));
        }
        let mut lease = PcmLease::default();
        let reserve = unsafe {
            lfm_audio_dock_wait_reserve(
                self.session.as_ptr(),
                PCM_CAPTURE,
                frames,
                utterance.rate,
                &mut lease,
            )
        };
        if reserve == STATUS_STALE || reserve == STATUS_CANCELLED {
            return Ok(None);
        }
        status(reserve, "reserve native capture lease")?;
        if lease.format != PCM_F32 || lease.channels != 1 {
            let _ = unsafe { lfm_audio_dock_release(self.session.as_ptr(), &lease) };
            return Err("native capture lease returned an unsupported PCM format".into());
        }
        let mut samples = std::ptr::null_mut();
        let mut capacity = 0usize;
        let resolve = unsafe {
            lfm_audio_dock_resolve_mut(self.session.as_ptr(), &lease, &mut samples, &mut capacity)
        };
        if resolve != 0 || samples.is_null() || capacity < utterance.samples.len() {
            let _ = unsafe { lfm_audio_dock_release(self.session.as_ptr(), &lease) };
            return Err(format!(
                "resolve native capture lease failed with status {resolve}"
            ));
        }
        // This is the sole transitional capture copy: the physical audio owner
        // still supplies a Rust Vec. A native device dock will write this
        // reservation directly; every session/model stage after this point
        // retains the slot.
        unsafe {
            std::ptr::copy_nonoverlapping(
                utterance.samples.as_ptr(),
                samples,
                utterance.samples.len(),
            );
        }
        Ok(Some(lease))
    }

    /// Submit one UTF-8 user turn through the bounded native control ring and
    /// consume only records carrying the returned action ticket.
    pub fn respond_text(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        if text.is_empty() || text.len() > 2_048 {
            return Err("native typed input must contain 1..=2048 UTF-8 bytes".into());
        }
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        let mut ticket = Ticket::default();
        let submit = unsafe {
            lfm_session_wait_submit_text(
                self.session.as_ptr(),
                text.as_ptr().cast(),
                text.len(),
                &mut ticket,
            )
        };
        if submit == STATUS_STALE || submit == STATUS_CANCELLED {
            return Ok(false);
        }
        status(submit, "submit native typed input")?;
        self.await_ticket(ticket, cancel, emit)
    }

    /// Submit typed and spoken input as one native action. Successful admission
    /// transfers the filled capture reservation to the command; text, audio,
    /// generated records, and the terminal event all carry the same ticket.
    pub fn respond_mixed(
        &mut self,
        text: &str,
        utterance: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        if text.is_empty() || text.len() > 2_048 {
            return Err("native typed input must contain 1..=2048 UTF-8 bytes".into());
        }
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        let Some(lease) = self.capture(utterance)? else {
            return Ok(false);
        };
        let mut ticket = Ticket::default();
        let submit = unsafe {
            lfm_session_wait_submit_mixed(
                self.session.as_ptr(),
                text.as_ptr().cast(),
                text.len(),
                &lease,
                &mut ticket,
            )
        };
        if submit != 0 {
            let _ = unsafe { lfm_audio_dock_release(self.session.as_ptr(), &lease) };
            if submit == STATUS_STALE || submit == STATUS_CANCELLED {
                return Ok(false);
            }
            return Err(format!(
                "submit native mixed text/PCM input failed with status {submit}"
            ));
        }
        self.await_ticket(ticket, cancel, emit)
    }

    fn await_ticket(
        &mut self,
        ticket: Ticket,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        let mut turn: Option<(Ticket, bool, u32)> = None;
        let mut audio_ticket: Option<Ticket> = None;
        let mut audio_count = 0u32;
        loop {
            let reply = self
                .replies
                .recv()
                .map_err(|_| "native voice event channel disconnected".to_string())?;
            if cancel.load(Ordering::Acquire) {
                return Ok(false);
            }
            match reply {
                Reply::Text {
                    ticket: reply_ticket,
                    text,
                } if reply_ticket == ticket && !text.is_empty() => emit(VoiceEvent::Text(text)),
                Reply::Text { .. } => {}
                Reply::Audio {
                    pcm,
                    rate,
                    ticket: reply_ticket,
                } if reply_ticket == ticket => {
                    audio_count = audio_count.saturating_add(1);
                    audio_ticket = Some(reply_ticket);
                    emit(VoiceEvent::Audio { pcm, rate });
                }
                Reply::Audio { .. } => {}
                Reply::Turn {
                    ticket: reply_ticket,
                    status,
                    has_audio,
                    truncated,
                    playback_leases,
                } if reply_ticket == ticket => {
                    if status == STATUS_STALE || status == STATUS_CANCELLED {
                        return Ok(false);
                    }
                    if status != 0 {
                        return Err(format!("native turn failed with status {status}"));
                    }
                    if truncated {
                        crate::vtrace!("native turn reached max_new_tokens");
                    }
                    turn = Some((reply_ticket, has_audio, playback_leases));
                }
                Reply::Turn { .. } => {}
                Reply::Interrupted {
                    ticket: reply_ticket,
                    ..
                } if reply_ticket.sequence > ticket.sequence => return Ok(false),
                Reply::Interrupted { .. } => {}
                Reply::Error {
                    ticket: Some(reply_ticket),
                    error,
                } if reply_ticket == ticket || reply_ticket.kind == TICKET_CONTROL => {
                    return Err(error);
                }
                Reply::Error {
                    ticket: Some(_), ..
                } => {}
                Reply::Error {
                    ticket: None,
                    error,
                } => return Err(error),
                Reply::Stopped(status) => {
                    return Err(format!("native voice session stopped with status {status}"));
                }
            }
            if let Some((turn_ticket, has_audio, playback_leases)) = turn {
                if !has_audio {
                    return Ok(true);
                }
                if audio_ticket == Some(turn_ticket) && audio_count >= playback_leases {
                    return Ok(true);
                }
            }
        }
    }
}

fn spawn_playback(
    session: NonNull<Session>,
    tx: Sender<Reply>,
    shutdown: Receiver<()>,
) -> Result<JoinHandle<()>, String> {
    let address = session.as_ptr() as usize;
    std::thread::Builder::new()
        .name("lfm-native-playback-dock".into())
        .spawn(move || {
            let session = address as *mut Session;
            loop {
                let mut lease = PcmLease::default();
                let wait = unsafe { lfm_audio_dock_wait_playback(session, &mut lease) };
                if wait == STATUS_CANCELLED {
                    return;
                }
                if wait != 0 {
                    let _ = send_reply(
                        &tx,
                        &shutdown,
                        Reply::Error {
                            ticket: None,
                            error: format!("native playback wait failed with status {wait}"),
                        },
                    );
                    unsafe { lfm_session_request_stop(session) };
                    return;
                }
                let mut samples = std::ptr::null();
                let mut count = 0usize;
                let resolve =
                    unsafe { lfm_audio_dock_resolve(session, &lease, &mut samples, &mut count) };
                if resolve != 0 || samples.is_null() {
                    let _ = unsafe { lfm_audio_dock_release(session, &lease) };
                    let _ = send_reply(
                        &tx,
                        &shutdown,
                        Reply::Error {
                            ticket: None,
                            error: format!("native playback resolve failed with status {resolve}"),
                        },
                    );
                    unsafe { lfm_session_request_stop(session) };
                    return;
                }
                let pcm = unsafe { std::slice::from_raw_parts(samples, count) }.to_vec();
                let reply = Reply::Audio {
                    pcm,
                    rate: lease.sample_rate,
                    ticket: lease.ticket,
                };
                let release = unsafe { lfm_audio_dock_release(session, &lease) };
                if release != 0 {
                    let _ = send_reply(
                        &tx,
                        &shutdown,
                        Reply::Error {
                            ticket: None,
                            error: format!("native playback release failed with status {release}"),
                        },
                    );
                    unsafe { lfm_session_request_stop(session) };
                    return;
                }
                if send_reply(&tx, &shutdown, reply).is_err() {
                    return;
                }
            }
        })
        .map_err(|error| format!("spawn native playback dock failed: {error}"))
}

impl VoiceEngine for NativeLfm2VoiceEngine {
    fn respond(
        &mut self,
        utterance: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        let Some(lease) = self.capture(utterance)? else {
            return Ok(false);
        };
        let publish = unsafe { lfm_audio_dock_publish(self.session.as_ptr(), &lease) };
        if publish != 0 {
            let _ = unsafe { lfm_audio_dock_release(self.session.as_ptr(), &lease) };
            return Err(format!(
                "publish native capture lease failed with status {publish}"
            ));
        }

        self.await_ticket(lease.ticket, cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        self.control.interrupt();
        Ok(())
    }

    fn interrupt_signal(&self) -> Option<Arc<dyn Fn() + Send + Sync>> {
        let control = self.control.clone();
        Some(Arc::new(move || control.interrupt()))
    }
}

impl Drop for NativeLfm2VoiceEngine {
    fn drop(&mut self) {
        // A reliable callback or playback publication may be parked on bounded
        // Rust-side flow control. Close that wait edge before either join.
        drop(self.shutdown.take());
        unsafe { lfm_session_request_stop(self.session.as_ptr()) };
        if let Some(playback) = self.playback.take() {
            let _ = playback.join();
        }
        let join = unsafe { lfm_session_join(self.session.as_ptr()) };
        if join != 0 {
            eprintln!("[flashkern] native voice session joined with status {join}");
        }
        let mut control = self
            .control
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *control = None;
        let destroy = unsafe { lfm_session_destroy(self.session.as_ptr()) };
        drop(control);
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
        let mut state = vault
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.conversation = self.conversation.take();
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

fn native_error(status: i32, error: &[i8]) -> String {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    if message.is_empty() {
        return format!("native model open failed with status {status}");
    }
    format!("{message} (native status {status})")
}
