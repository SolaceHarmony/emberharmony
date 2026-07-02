//! Multi-threaded realtime speech-to-speech pipeline.
//!
//! Faithful port of the threading model in `liquid_audio/demo/chat.py`, restructured
//! toward `moshi/server.py`'s inference coroutine.
//!
//! **Python (`chat.py`)**: `chat_response` spawns a `chat_producer` `Thread` that owns the
//! model + Mimi codec, runs `generate_interleaved`, decodes each audio frame to PCM, and
//! `q.put()`s tokens + PCM onto a `queue.Queue`; the main thread drains the queue and
//! relays text to the UI and PCM to WebRTC playback. Generation overlaps playback because
//! they run on different threads.
//!
//! **Here**: a *persistent* inference worker thread owns the [`VoiceEngine`] and loops
//! `recv utterance → respond (emit text + decode audio → emit PCM) → TurnComplete`. The
//! consumer (UI / playback feeder) drains [`VoiceEvent`]s off a channel. Because the model
//! lives on its own thread, capture and playback are never blocked by generation
//! (full-duplex), and a newly-detected utterance can request **barge-in** — an
//! `AtomicBool` the generate loop polls (see
//! [`LFM2AudioModel::generate_interleaved_cancellable`]) to abort the in-flight reply
//! instead of running it to `max_new_tokens` and tying up the P-cores.
//!
//! The engine is a trait so the threading is unit-tested with a fake (no model needed);
//! [`Lfm2VoiceEngine`] is the real implementation that owns the model + processor.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::moshi::demo::chat::decode_audio_frame;
use crate::moshi::models::{
    load_realtime_moshi, MimiModel, RealtimeMoshi, RealtimeMoshiEvent, RealtimeMoshiFiles,
    RealtimeMoshiParams,
};

#[cfg(test)]
const MIMI_RATE: u32 = 24_000; // Mimi/LFM2 detokenizer output rate.

const UTTERANCE_QUEUE_CAP: usize = 1;
const EVENT_QUEUE_CAP: usize = 128;
const FRAME_QUEUE_CAP: usize = 8;
const STREAM_IDLE_FRAMES: usize = 5;

fn try_send_event(tx: &Sender<VoiceEvent>, ev: VoiceEvent, cancel: &AtomicBool) -> bool {
    match tx.try_send(ev) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
            cancel.store(true, Ordering::SeqCst);
            false
        }
    }
}

/// A captured user utterance handed to the worker: mono f32 samples + their sample rate.
pub struct Utterance {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// Fixed-rate frame contract for models that are truly realtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    pub sample_rate: u32,
    pub frame_size: usize,
}

/// One streamed reply item the worker emits — the Rust analog of `chat.py`'s `q.put`.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    /// A decoded text fragment (one or more detokenized text tokens).
    Text(String),
    /// A decoded PCM chunk (mono f32) at the pipeline's output rate.
    Audio(Vec<f32>),
    /// The reply for the current utterance finished normally (`chat.py`'s `q.put(None)`).
    TurnComplete,
    /// The reply was cut short by [`RealtimePipeline::interrupt`] (barge-in).
    Interrupted,
    /// The engine errored on this turn. The worker stays alive for the next utterance.
    Error(String),
}

/// The model side of the pipeline, abstracted so the worker-thread machinery can be
/// exercised with a fake. The real implementation ([`Lfm2VoiceEngine`]) owns the model +
/// processor + detokenizer; it must be `Send` to move onto the worker thread.
pub trait VoiceEngine: Send {
    /// Respond to one utterance, calling `emit` for each [`VoiceEvent`] in order. Poll
    /// `cancel` frequently and return `Ok(false)` promptly once it is set (barge-in);
    /// return `Ok(true)` when the reply ran to completion. `Err` is surfaced as
    /// [`VoiceEvent::Error`] and does not kill the worker.
    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String>;

    /// Realtime engines advertise this to bypass utterance/VAD batching and receive exact
    /// PCM frames continuously, matching upstream Moshi's sounddevice/WebRTC loop.
    fn frame_config(&self) -> Option<FrameConfig> {
        None
    }

    /// Process one fixed-size realtime PCM frame. Returning `Ok(false)` reports an
    /// interruption to the event stream; ordinary realtime silence should return `Ok(true)`.
    fn respond_frame(
        &mut self,
        _frame: &[f32],
        _cancel: &AtomicBool,
        _emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        Err("voice engine does not support realtime PCM frames".into())
    }

    /// Flush stream-local output state after a UI stop/interrupt. Model context reset is
    /// engine-specific; the default is only to acknowledge the interrupt.
    fn interrupt_stream(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl<T: VoiceEngine + ?Sized> VoiceEngine for Box<T> {
    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        (**self).respond(utt, cancel, emit)
    }

    fn frame_config(&self) -> Option<FrameConfig> {
        (**self).frame_config()
    }

    fn respond_frame(
        &mut self,
        frame: &[f32],
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        (**self).respond_frame(frame, cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        (**self).interrupt_stream()
    }
}

enum FrameCommand {
    Pcm { pcm: Vec<f32>, epoch: u64 },
    Interrupt { epoch: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSubmitError {
    WrongSize,
    Full,
    Disconnected,
}

struct QueuedUtterance {
    utt: Utterance,
    epoch: u64,
}

/// Handle to the inference worker thread: submit utterances, receive reply events, request
/// barge-in. Dropping it closes the channel and joins the worker.
pub struct RealtimePipeline {
    utt_tx: Option<Sender<QueuedUtterance>>,
    event_rx: Receiver<VoiceEvent>,
    cancel: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

/// Cloneable control handle for producers that feed an existing realtime worker.
#[derive(Clone)]
pub struct RealtimePipelineHandle {
    utt_tx: Sender<QueuedUtterance>,
    cancel: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
}

impl RealtimePipelineHandle {
    /// Hand the worker a new utterance. Returns `false` if the bounded queue is full or the
    /// worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        let epoch = self.epoch.load(Ordering::Acquire);
        match self.utt_tx.try_send(QueuedUtterance { utt, epoch }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Request barge-in: abort the in-flight reply.
    pub fn interrupt(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.store(true, Ordering::SeqCst);
    }
}

impl RealtimePipeline {
    /// Spawn the worker thread; it owns `engine` for its lifetime and serves utterances
    /// until this handle is dropped (which closes the utterance channel).
    pub fn spawn<E: VoiceEngine + 'static>(mut engine: E) -> Result<Self, String> {
        if engine.frame_config().is_some() {
            return Err(
                "frame-capable voice engines must use RealtimeFramePipeline, not RealtimePipeline"
                    .into(),
            );
        }

        // The realtime mic/VAD side must not accumulate stale speech behind a busy model.
        // One queued utterance is enough: current generation + next turn. Barge-in sets the
        // cancel flag first; if this slot is still full, the caller gets backpressure instead
        // of silently growing latency.
        let (utt_tx, utt_rx) = bounded::<QueuedUtterance>(UTTERANCE_QUEUE_CAP);
        let (event_tx, event_rx) = bounded::<VoiceEvent>(EVENT_QUEUE_CAP);
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let epoch = Arc::new(AtomicU64::new(0));
        let epoch_worker = epoch.clone();

        let worker = std::thread::Builder::new()
            .name("lfm2-inference".into())
            .spawn(move || {
                // The inference coroutine: serve utterances until the sender is dropped
                // (channel closed) ⇒ `iter()` ends ⇒ the thread returns and joins.
                let mut current_epoch = 0u64;
                for QueuedUtterance { utt, epoch } in utt_rx.iter() {
                    let latest_epoch = epoch_worker.load(Ordering::Acquire);
                    if epoch < latest_epoch {
                        if latest_epoch > current_epoch {
                            current_epoch = latest_epoch;
                            cancel_worker.store(true, Ordering::SeqCst);
                            let reset = engine.interrupt_stream();
                            cancel_worker.store(false, Ordering::SeqCst);
                            match reset {
                                Ok(()) => {
                                    if !try_send_event(
                                        &event_tx,
                                        VoiceEvent::Interrupted,
                                        &cancel_worker,
                                    ) {
                                        break;
                                    }
                                }
                                Err(error) => {
                                    if !try_send_event(
                                        &event_tx,
                                        VoiceEvent::Error(error),
                                        &cancel_worker,
                                    ) {
                                        break;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    if epoch > current_epoch {
                        current_epoch = epoch;
                        cancel_worker.store(true, Ordering::SeqCst);
                        let reset = engine.interrupt_stream();
                        cancel_worker.store(false, Ordering::SeqCst);
                        match reset {
                            Ok(()) => {
                                if !try_send_event(
                                    &event_tx,
                                    VoiceEvent::Interrupted,
                                    &cancel_worker,
                                ) {
                                    break;
                                }
                            }
                            Err(error) => {
                                if !try_send_event(
                                    &event_tx,
                                    VoiceEvent::Error(error),
                                    &cancel_worker,
                                ) {
                                    break;
                                }
                                continue;
                            }
                        }
                    }
                    // A fresh turn clears any barge-in left set by the previous reply, so
                    // it cannot carry over and abort the new one before it starts.
                    cancel_worker.store(false, Ordering::SeqCst);
                    let mut event_backpressure = false;
                    let responded = {
                        let mut emit = |ev: VoiceEvent| {
                            if event_backpressure {
                                return;
                            }
                            if !try_send_event(&event_tx, ev, &cancel_worker) {
                                event_backpressure = true;
                            }
                        };
                        engine.respond(&utt, &cancel_worker, &mut emit)
                    };
                    let terminal = if event_backpressure {
                        VoiceEvent::Error("voice event queue full or disconnected".into())
                    } else {
                        match responded {
                            Ok(true) => VoiceEvent::TurnComplete,
                            Ok(false) => {
                                let latest_epoch = epoch_worker.load(Ordering::Acquire);
                                if latest_epoch > current_epoch {
                                    current_epoch = latest_epoch;
                                    cancel_worker.store(true, Ordering::SeqCst);
                                    let reset = engine.interrupt_stream();
                                    cancel_worker.store(false, Ordering::SeqCst);
                                    match reset {
                                        Ok(()) => VoiceEvent::Interrupted,
                                        Err(error) => VoiceEvent::Error(error),
                                    }
                                } else {
                                    VoiceEvent::Interrupted
                                }
                            }
                            Err(e) => VoiceEvent::Error(e),
                        }
                    };
                    if !try_send_event(&event_tx, terminal, &cancel_worker) {
                        break; // consumer hung up — nothing left to serve.
                    }
                }
            })
            .map_err(|e| format!("spawn lfm2-inference worker failed: {e}"))?;

        Ok(Self {
            utt_tx: Some(utt_tx),
            event_rx,
            cancel,
            epoch,
            worker: Some(worker),
        })
    }

    /// Hand the worker a new utterance. Returns `false` if the bounded queue is full or the
    /// worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        let Some(tx) = self.utt_tx.as_ref() else {
            return false;
        };
        let epoch = self.epoch.load(Ordering::Acquire);
        match tx.try_send(QueuedUtterance { utt, epoch }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Request barge-in: abort the in-flight reply. The engine polls this and returns
    /// early, after which the worker emits [`VoiceEvent::Interrupted`]. Call this before
    /// submitting the interrupting utterance.
    pub fn interrupt(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// The stream of reply events; drain it in the consumer (UI / playback feeder).
    pub fn events(&self) -> &Receiver<VoiceEvent> {
        &self.event_rx
    }

    /// A cloneable producer/control handle for external audio transports.
    pub fn handle(&self) -> Option<RealtimePipelineHandle> {
        self.utt_tx.as_ref().map(|utt_tx| RealtimePipelineHandle {
            utt_tx: utt_tx.clone(),
            cancel: self.cancel.clone(),
            epoch: self.epoch.clone(),
        })
    }
}

impl Drop for RealtimePipeline {
    fn drop(&mut self) {
        // Abort any in-flight reply fast, then close the utterance channel so the worker's
        // `iter()` ends, then join — no detached thread, no leak.
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.store(true, Ordering::SeqCst);
        drop(self.utt_tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Frame-fed realtime worker for Moshi-style engines. Unlike [`RealtimePipeline`], this
/// keeps the model advancing on exact PCM frames instead of waiting for a VAD utterance.
pub struct RealtimeFramePipeline {
    frame_tx: Option<Sender<FrameCommand>>,
    event_rx: Receiver<VoiceEvent>,
    cancel: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    cfg: FrameConfig,
    worker: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct RealtimeFramePipelineHandle {
    frame_tx: Sender<FrameCommand>,
    cancel: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    cfg: FrameConfig,
}

impl RealtimeFramePipelineHandle {
    pub fn config(&self) -> FrameConfig {
        self.cfg
    }

    pub fn try_submit_frame(&self, pcm: Vec<f32>) -> Result<(), FrameSubmitError> {
        if pcm.len() != self.cfg.frame_size {
            return Err(FrameSubmitError::WrongSize);
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        match self.frame_tx.try_send(FrameCommand::Pcm { pcm, epoch }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(FrameSubmitError::Full),
            Err(TrySendError::Disconnected(_)) => Err(FrameSubmitError::Disconnected),
        }
    }

    pub fn submit_frame(&self, pcm: Vec<f32>) -> bool {
        self.try_submit_frame(pcm).is_ok()
    }

    pub fn interrupt(&self) {
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        self.cancel.store(true, Ordering::SeqCst);
        let _ = self.frame_tx.try_send(FrameCommand::Interrupt { epoch });
    }
}

impl RealtimeFramePipeline {
    pub fn spawn<E: VoiceEngine + 'static>(mut engine: E) -> Result<Self, String> {
        let cfg = engine
            .frame_config()
            .ok_or_else(|| "voice engine does not expose a realtime frame config".to_string())?;
        if cfg.sample_rate == 0 {
            return Err("realtime frame sample rate must be non-zero".into());
        }
        if cfg.frame_size == 0 {
            return Err("realtime frame size must be non-zero".into());
        }

        let (frame_tx, frame_rx) = bounded::<FrameCommand>(FRAME_QUEUE_CAP);
        let (event_tx, event_rx) = bounded::<VoiceEvent>(EVENT_QUEUE_CAP);
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let epoch = Arc::new(AtomicU64::new(0));
        let epoch_worker = epoch.clone();

        let worker = std::thread::Builder::new()
            .name("moshi-frame-inference".into())
            .spawn(move || {
                let mut active = false;
                let mut idle_frames = 0usize;
                let mut current_epoch = 0u64;
                for cmd in frame_rx.iter() {
                    match cmd {
                        FrameCommand::Interrupt { epoch } => {
                            if epoch <= current_epoch {
                                continue;
                            }
                            current_epoch = epoch;
                            cancel_worker.store(true, Ordering::SeqCst);
                            let interrupted = engine.interrupt_stream().is_ok();
                            cancel_worker.store(false, Ordering::SeqCst);
                            active = false;
                            idle_frames = 0;
                            if interrupted
                                && !try_send_event(
                                    &event_tx,
                                    VoiceEvent::Interrupted,
                                    &cancel_worker,
                                )
                            {
                                break;
                            }
                        }
                        FrameCommand::Pcm { pcm: frame, epoch } => {
                            let latest_epoch = epoch_worker.load(Ordering::Acquire);
                            if epoch < latest_epoch {
                                if latest_epoch > current_epoch {
                                    current_epoch = latest_epoch;
                                    cancel_worker.store(true, Ordering::SeqCst);
                                    let reset = engine.interrupt_stream();
                                    cancel_worker.store(false, Ordering::SeqCst);
                                    active = false;
                                    idle_frames = 0;
                                    match reset {
                                        Ok(()) => {
                                            if !try_send_event(
                                                &event_tx,
                                                VoiceEvent::Interrupted,
                                                &cancel_worker,
                                            ) {
                                                break;
                                            }
                                        }
                                        Err(error) => {
                                            if !try_send_event(
                                                &event_tx,
                                                VoiceEvent::Error(error),
                                                &cancel_worker,
                                            ) {
                                                break;
                                            }
                                        }
                                    }
                                }
                                continue;
                            }
                            if epoch > current_epoch {
                                current_epoch = epoch;
                                active = false;
                                idle_frames = 0;
                                if engine.interrupt_stream().is_err() {
                                    if !try_send_event(
                                        &event_tx,
                                        VoiceEvent::Error(
                                            "failed to interrupt voice stream".into(),
                                        ),
                                        &cancel_worker,
                                    ) {
                                        break;
                                    }
                                    continue;
                                }
                                let _ = try_send_event(
                                    &event_tx,
                                    VoiceEvent::Interrupted,
                                    &cancel_worker,
                                );
                            }
                            cancel_worker.store(false, Ordering::SeqCst);
                            let mut event_backpressure = false;
                            let mut emitted_output = false;
                            let responded = {
                                let mut emit = |ev: VoiceEvent| {
                                    if event_backpressure {
                                        return;
                                    }
                                    emitted_output |=
                                        matches!(ev, VoiceEvent::Text(_) | VoiceEvent::Audio(_));
                                    if !try_send_event(&event_tx, ev, &cancel_worker) {
                                        event_backpressure = true;
                                    }
                                };
                                engine.respond_frame(&frame, &cancel_worker, &mut emit)
                            };
                            if event_backpressure {
                                let _ = try_send_event(
                                    &event_tx,
                                    VoiceEvent::Error(
                                        "voice event queue full or disconnected".into(),
                                    ),
                                    &cancel_worker,
                                );
                                break;
                            }
                            match responded {
                                Ok(true) => {
                                    if emitted_output {
                                        active = true;
                                        idle_frames = 0;
                                    } else if active {
                                        idle_frames += 1;
                                        if idle_frames >= STREAM_IDLE_FRAMES {
                                            active = false;
                                            idle_frames = 0;
                                            if !try_send_event(
                                                &event_tx,
                                                VoiceEvent::TurnComplete,
                                                &cancel_worker,
                                            ) {
                                                break;
                                            }
                                        }
                                    }
                                }
                                Ok(false) => {
                                    active = false;
                                    idle_frames = 0;
                                    if !try_send_event(
                                        &event_tx,
                                        VoiceEvent::Interrupted,
                                        &cancel_worker,
                                    ) {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    active = false;
                                    idle_frames = 0;
                                    if !try_send_event(
                                        &event_tx,
                                        VoiceEvent::Error(e),
                                        &cancel_worker,
                                    ) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            })
            .map_err(|e| format!("spawn moshi-frame-inference worker failed: {e}"))?;

        Ok(Self {
            frame_tx: Some(frame_tx),
            event_rx,
            cancel,
            epoch,
            cfg,
            worker: Some(worker),
        })
    }

    pub fn config(&self) -> FrameConfig {
        self.cfg
    }

    pub fn try_submit_frame(&self, pcm: Vec<f32>) -> Result<(), FrameSubmitError> {
        if pcm.len() != self.cfg.frame_size {
            return Err(FrameSubmitError::WrongSize);
        }
        let Some(tx) = self.frame_tx.as_ref() else {
            return Err(FrameSubmitError::Disconnected);
        };
        let epoch = self.epoch.load(Ordering::Acquire);
        match tx.try_send(FrameCommand::Pcm { pcm, epoch }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(FrameSubmitError::Full),
            Err(TrySendError::Disconnected(_)) => Err(FrameSubmitError::Disconnected),
        }
    }

    pub fn submit_frame(&self, pcm: Vec<f32>) -> bool {
        self.try_submit_frame(pcm).is_ok()
    }

    pub fn interrupt(&self) {
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(tx) = self.frame_tx.as_ref() {
            let _ = tx.try_send(FrameCommand::Interrupt { epoch });
        }
    }

    pub fn events(&self) -> &Receiver<VoiceEvent> {
        &self.event_rx
    }

    pub fn handle(&self) -> Option<RealtimeFramePipelineHandle> {
        self.frame_tx
            .as_ref()
            .map(|frame_tx| RealtimeFramePipelineHandle {
                frame_tx: frame_tx.clone(),
                cancel: self.cancel.clone(),
                epoch: self.epoch.clone(),
                cfg: self.cfg,
            })
    }
}

impl Drop for RealtimeFramePipeline {
    fn drop(&mut self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.store(true, Ordering::SeqCst);
        drop(self.frame_tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// ---------------------------------------------------------------------------------------
// Real engine
// ---------------------------------------------------------------------------------------

use candle_core::{DType, Device, Tensor};

use crate::{ChatState, GenParams, GenToken, LFM2AudioModel, LFM2AudioProcessor, LFMModality};

/// The accumulated conversation tensors persisted across turns — the five model-input fields
/// of a [`ChatState`]. The engine holds these (not a `ChatState`, which borrows the processor
/// and so can't be stored beside it) and rebuilds a transient `ChatState` each turn via
/// [`ChatState::from_parts`]. This is the Rust analog of Python keeping ONE persistent
/// `ChatState` object across turns: the discrete `audio_out` generated each turn is fed back
/// here as context for the next prefill (`generate_interleaved`'s `audio_out` scatter).
#[derive(Clone)]
pub struct ConversationState {
    pub text: Tensor,
    pub audio_in: Tensor,
    pub audio_in_lens: Tensor,
    pub audio_out: Tensor,
    pub modality_flag: Tensor,
}

/// Build a `(rows, cols)` i64 tensor, using a zero-length view of a 1-col buffer when
/// `cols == 0` (candle can't allocate a zero-size buffer on Metal — the same guard
/// `ChatState::new` uses for its empty placeholders). Used to stack the per-turn generated
/// `text`/`audio_out`/`modality_flag` for [`ChatState::append`].
fn stack_i64(
    vals: Vec<i64>,
    rows: usize,
    cols: usize,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    if cols == 0 {
        Tensor::zeros((rows, 1), DType::I64, dev)?.narrow(1, 0, 0)
    } else {
        Tensor::from_vec(vals, (rows, cols), dev)
    }
}

/// The real [`VoiceEngine`]: owns the LFM2-Audio model + processor and answers each
/// utterance via [`LFM2AudioModel::generate_interleaved_cancellable`], decoding audio
/// frames to PCM through the processor's streaming detokenizer — the headless equivalent
/// of `chat.py`'s `chat_producer` body.
pub struct Lfm2VoiceEngine {
    model: LFM2AudioModel,
    proc: LFM2AudioProcessor,
    params: GenParams,
    codebooks: usize,
    device: Device,
    /// Resample target for emitted PCM. `0` ⇒ emit the codec's native rate;
    /// otherwise resample each chunk to this device rate.
    out_rate: u32,
    /// The assistant's system prompt, added once at the start of the conversation (Python adds
    /// the system turn once, before the first user turn). Defaults to the demo's interleaved
    /// prompt; the desktop layer's `TurnMode` sets the ASR/TTS variants via
    /// [`Self::with_system_prompt`]. (Per-mode `max_new_tokens` rides on `params`.)
    system_prompt: String,
    /// The persistent conversation: `None` until the first turn completes (cold start = fresh
    /// `ChatState::new` + the system turn), then `Some`, carrying the accumulated tensors so
    /// each later turn *continues* the conversation — feeding the prior turn's discrete
    /// `audio_out` back as context (the `audio_out` → prefill-scatter path). Cold-start-per-turn
    /// (the prior behavior) made every utterance context-free.
    conv: Option<ConversationState>,
}

impl Lfm2VoiceEngine {
    pub fn new(
        model: LFM2AudioModel,
        proc: LFM2AudioProcessor,
        params: GenParams,
        codebooks: usize,
        device: Device,
        out_rate: u32,
    ) -> Self {
        Self {
            model,
            proc,
            params,
            codebooks,
            device,
            out_rate,
            system_prompt: "Respond with interleaved text and audio.".to_string(),
            conv: None,
        }
    }

    /// Override the system prompt (the desktop `TurnMode` → ASR / TTS / Interleaved prompt,
    /// verbatim from the demo's `audio-model.js`). Builder form so the single `new` call site
    /// and the tests stay unchanged.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

impl VoiceEngine for Lfm2VoiceEngine {
    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        let s = |e: candle_core::Error| e.to_string();

        // STREAMING decode is the Mimi codec — required, never optional. Mimi is the only
        // backend with a true `decode_step` (it carries codec state ACROSS frames for gapless
        // output), exactly the Python demo, which streams every frame inside `mimi.streaming(1)`
        // via `mimi.decode(frame)` (chat.py:21,34). We always ship Mimi; if it is missing the
        // model load is broken — hard-error. No fallback to the LFM2 detokenizer's degenerate
        // per-frame one-shot: a fallback would only mask a broken build behind choppy audio.
        let mimi = MimiModel::new(
            self.proc
                .mimi()
                .ok_or("Mimi codec not loaded — required for streaming audio out")?,
        );
        let mut stream = mimi.streaming(1).map_err(s)?; // turn boundary (`mimi.streaming(1)`).
        let mimi_rate = mimi.sample_rate();
        if utt.rate == 0 {
            return Err("utterance sample rate must be non-zero".into());
        }
        if mimi_rate == 0 {
            return Err("Mimi codec sample rate must be non-zero".into());
        }
        if self.out_rate == 0 {
            return Err("output sample rate must be non-zero".into());
        }

        let wave = Tensor::from_vec(utt.samples.clone(), (1, utt.samples.len()), &self.device)
            .map_err(s)?;

        // Restore the persistent conversation (clone the prior tensors — cheap Arc bumps — so a
        // hard error that returns early leaves prior history intact; we never `take`). First turn
        // → fresh `ChatState` + the system turn (added once, like Python); later turns → seed
        // from the accumulated state so the prior discrete `audio_out` conditions this prefill.
        let prior = self.conv.clone();
        let mut chat = match &prior {
            None => {
                let mut c = ChatState::new(&self.proc, self.codebooks).map_err(s)?;
                c.new_turn("system").map_err(s)?;
                c.add_text(&self.system_prompt).map_err(s)?;
                c.end_turn().map_err(s)?;
                c
            }
            Some(conv) => ChatState::from_parts(
                &self.proc,
                self.codebooks,
                conv.text.clone(),
                conv.audio_in.clone(),
                conv.audio_in_lens.clone(),
                conv.audio_out.clone(),
                conv.modality_flag.clone(),
            )
            .map_err(s)?,
        };
        chat.new_turn("user").map_err(s)?;
        chat.add_audio(&wave, utt.rate).map_err(s)?; // CONTINUOUS audio-in (mel → Conformer)
        chat.end_turn().map_err(s)?;
        chat.new_turn("assistant").map_err(s)?;

        let text = self.proc.text();
        let device = &self.device;
        let codebooks = self.codebooks;
        let mut resampler = StreamingPcmResampler::new(mimi_rate, self.out_rate);
        let mut cb_err: Option<String> = None;

        // Collect the generated stream for `chat.append` (the discrete `audio_out` → context
        // path). `text_ids` / `audio_frames` are the SEPARATED in-order streams (Python's
        // `text_out` / `audio_out` lists); `modality_out` preserves the INTERLEAVED generation
        // order (Python's `modality_out`). `audio_frames` keeps ALL frames including the
        // all-2048 EOAudio terminator — `append` needs the full `audio_out`
        // (`torch.stack(audio_out, 1)`); only the Mimi DECODE drops the last (`audio_out[:-1]`).
        let mut text_ids: Vec<i64> = Vec::new();
        let mut audio_frames: Vec<Vec<u32>> = Vec::new();
        let mut modality_out: Vec<i64> = Vec::new();

        self.model
            .generate_interleaved_cancellable(&chat, &self.params, cancel, |tok| {
                if cb_err.is_some() {
                    return;
                }
                match tok {
                    GenToken::Text(id) => {
                        text_ids.push(id as i64);
                        modality_out.push(LFMModality::Text as i64);
                        match text.decode(&[id], true) {
                            Ok(text) => emit(VoiceEvent::Text(text)),
                            Err(e) => cb_err = Some(e.to_string()),
                        }
                    }
                    GenToken::Audio(frame) => {
                        // Collect EVERY frame (incl. EOAudio) for `append` BEFORE the streaming
                        // skip — append needs the full audio_out; only PCM playback drops it.
                        modality_out.push(LFMModality::AudioOut as i64);
                        audio_frames.push(frame.clone());
                        // Decode the 8-code frame to PCM via the streaming detokenizer.
                        let decoded = (|| -> candle_core::Result<Option<Vec<f32>>> {
                            match decode_audio_frame(&mut stream, &frame, codebooks, device)? {
                                Some(chunk) => {
                                    let mut pcm = chunk
                                        .flatten_all()?
                                        .to_dtype(DType::F32)?
                                        .to_vec1::<f32>()?;
                                    pcm = resampler.process(pcm);
                                    Ok(Some(pcm))
                                }
                                None => Ok(None),
                            }
                        })();
                        match decoded {
                            Ok(Some(pcm)) => emit(VoiceEvent::Audio(pcm)),
                            Ok(None) => {}
                            Err(e) => cb_err = Some(e.to_string()),
                        }
                    }
                }
            })
            .map_err(s)?;

        if let Some(e) = cb_err {
            return Err(e);
        }

        // Completed unless the loop broke because barge-in was requested.
        let completed = !cancel.load(Ordering::Acquire);

        // Persist the GENERATED response into the conversation — keyed on what the model
        // PRODUCED, never on whether it was played or whether a mic was open. The audio frames
        // were collected above at generation time, before the playback/mute branch, so a muted
        // speaker changes nothing; and we persist even on barge-in (`completed == false`),
        // because a thought the model started is still a prior thought. Dropping the model's own
        // (possibly partial) response would make the context depend on an I/O event — the very
        // failure mode to avoid: the model would lose its own response. `append` weaves the
        // generated text + discrete `audio_out` (interleaved per `modality_flag`) in; `end_turn`
        // closes the assistant turn (early, if barge-in cut it short). Only a hard error (returned
        // above) or a genuinely empty generation skips this.
        if !text_ids.is_empty() || !audio_frames.is_empty() {
            let n_text = text_ids.len();
            let n_audio = audio_frames.len();
            let n_flag = modality_out.len();
            let text_t = stack_i64(text_ids, 1, n_text, &self.device).map_err(s)?;
            // Stack frames into (codebooks, n_audio) column-major: flat[c*n_audio + f] =
            // frame[f][c] — the transpose of the (frame, codebook) collection, matching
            // `torch.stack(audio_out, 1)` (each frame is a column).
            let mut flat = Vec::with_capacity(codebooks * n_audio);
            for c in 0..codebooks {
                for f in &audio_frames {
                    flat.push(f[c] as i64);
                }
            }
            let audio_t = stack_i64(flat, codebooks, n_audio, &self.device).map_err(s)?;
            let mod_t = stack_i64(modality_out, 1, n_flag, &self.device).map_err(s)?;
            chat.append(&text_t, &audio_t, &mod_t).map_err(s)?;
            chat.end_turn().map_err(s)?;

            let saved = ConversationState {
                text: chat.text.clone(),
                audio_in: chat.audio_in.clone(),
                audio_in_lens: chat.audio_in_lens.clone(),
                audio_out: chat.audio_out.clone(),
                modality_flag: chat.modality_flag.clone(),
            };
            drop(chat); // end the `&self.proc` borrow before writing `self.conv`.
            self.conv = Some(saved);
        }

        Ok(completed)
    }
}

/// Native Moshi realtime engine: persistent Mimi + LMGen-style state fed with fixed PCM frames.
///
/// This is the desktop-side counterpart of upstream `moshi/server.py` without the websocket/Opus
/// wrapper. The surrounding [`RealtimeFramePipeline`] owns the model on a worker thread and feeds
/// continuous PCM frames; this engine keeps the canonical Moshi step order inside that worker:
/// PCM frame -> Mimi encode -> multistream LM step -> Mimi decode.
pub struct MoshiVoiceEngine {
    realtime: RealtimeMoshi,
    text: sentencepiece_rust::SentencePieceProcessor,
    out_rate: u32,
    out_resampler: StreamingPcmResampler,
}

impl MoshiVoiceEngine {
    pub fn new(
        realtime: RealtimeMoshi,
        text: sentencepiece_rust::SentencePieceProcessor,
        out_rate: u32,
    ) -> Self {
        let rate = realtime.sample_rate();
        Self {
            realtime,
            text,
            out_rate,
            out_resampler: StreamingPcmResampler::new(rate, out_rate),
        }
    }

    pub fn from_files(
        files: &RealtimeMoshiFiles,
        dtype: DType,
        device: &Device,
        params: RealtimeMoshiParams,
        out_rate: u32,
    ) -> Result<Self, String> {
        let moshi_weights = files
            .moshi_weights
            .to_str()
            .ok_or_else(|| "Moshi checkpoint path is not UTF-8".to_string())?;
        let mimi_weights = files
            .mimi_weights
            .to_str()
            .ok_or_else(|| "Mimi checkpoint path is not UTF-8".to_string())?;
        let realtime = load_realtime_moshi(moshi_weights, mimi_weights, dtype, device, params)
            .map_err(|e| e.to_string())?;
        let text = sentencepiece_rust::SentencePieceProcessor::open(&files.tokenizer)
            .map_err(|e| e.to_string())?;
        Ok(Self::new(realtime, text, out_rate))
    }

    fn decode_text_token(&self, token: u32) -> Option<String> {
        let piece = self.text.id_to_piece(token as i32)?;
        let text = piece.replace('▁', " ");
        if text.is_empty() {
            return None;
        }
        Some(text)
    }

    fn emit_events(
        &mut self,
        events: Vec<RealtimeMoshiEvent>,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> bool {
        for event in events {
            if cancel.load(Ordering::Acquire) {
                return false;
            }
            match event {
                RealtimeMoshiEvent::InputAudioTokenFrame(_) => {}
                RealtimeMoshiEvent::TextToken(token) => {
                    if let Some(text) = self.decode_text_token(token) {
                        emit(VoiceEvent::Text(text));
                    }
                }
                RealtimeMoshiEvent::AudioTokenFrame(_) => {}
                RealtimeMoshiEvent::Audio { pcm, rate } => {
                    let pcm = if rate == self.out_rate {
                        pcm
                    } else {
                        self.out_resampler.process(pcm)
                    };
                    emit(VoiceEvent::Audio(pcm));
                }
            }
        }
        true
    }
}

impl VoiceEngine for MoshiVoiceEngine {
    fn respond(
        &mut self,
        _utt: &Utterance,
        _cancel: &AtomicBool,
        _emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        Err(
            "Moshi realtime engine requires the frame pipeline; feed fixed PCM frames with respond_frame"
                .into(),
        )
    }

    fn frame_config(&self) -> Option<FrameConfig> {
        Some(FrameConfig {
            sample_rate: self.realtime.sample_rate(),
            frame_size: self.realtime.frame_size(),
        })
    }

    fn respond_frame(
        &mut self,
        frame: &[f32],
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        if cancel.load(Ordering::Acquire) {
            return Ok(false);
        }
        let events = self
            .realtime
            .step_pcm_frame(frame)
            .map_err(|e| e.to_string())?;
        Ok(self.emit_events(events, cancel, emit) && !cancel.load(Ordering::Acquire))
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        self.realtime.reset_stream();
        self.out_resampler.reset();
        Ok(())
    }
}

/// Stateful PCM conversion for the streaming Mimi decoder.
///
/// The reference model emits Mimi audio at 24 kHz, while the built-in macOS output path is
/// usually 48 kHz. The old path ran the offline sinc resampler independently on each tiny
/// decoded frame, which resets the filter at every chunk boundary and can produce audible
/// discontinuities. The hot desktop path needs continuity over the whole turn.
struct StreamingPcmResampler {
    from: u32,
    to: u32,
    prev: Option<f32>,
}

impl StreamingPcmResampler {
    fn new(from: u32, to: u32) -> Self {
        Self {
            from,
            to,
            prev: None,
        }
    }

    fn process(&mut self, pcm: Vec<f32>) -> Vec<f32> {
        if pcm.is_empty() || self.to == 0 || self.to == self.from {
            self.prev = pcm.last().copied().or(self.prev);
            return pcm;
        }
        if self.to > self.from && self.to % self.from == 0 {
            return self.upsample_integer(&pcm, (self.to / self.from) as usize);
        }
        self.prev = pcm.last().copied();
        crate::resample::resample_slice(&pcm, self.from, self.to)
    }

    fn reset(&mut self) {
        self.prev = None;
    }

    fn upsample_integer(&mut self, pcm: &[f32], ratio: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(pcm.len() * ratio);
        for &sample in pcm {
            let last = self.prev.unwrap_or(sample);
            for step in 1..=ratio {
                let t = step as f32 / ratio as f32;
                out.push(last + (sample - last) * t);
            }
            self.prev = Some(sample);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::AtomicUsize, mpsc};
    use std::time::Duration;

    fn utt(n: usize) -> Utterance {
        Utterance {
            samples: vec![0.0; n],
            rate: 16_000,
        }
    }

    /// Drain events until a terminal one (TurnComplete / Interrupted / Error), bounded by
    /// a timeout so a wiring bug fails the test instead of hanging it.
    fn collect_turn(rx: &Receiver<VoiceEvent>) -> Vec<VoiceEvent> {
        let mut out = Vec::new();
        loop {
            let ev = rx
                .recv_timeout(Duration::from_secs(5))
                .expect("expected an event before timeout");
            let terminal = matches!(
                ev,
                VoiceEvent::TurnComplete | VoiceEvent::Interrupted | VoiceEvent::Error(_)
            );
            out.push(ev);
            if terminal {
                return out;
            }
        }
    }

    #[test]
    fn streaming_resampler_keeps_integer_upsample_continuity() {
        let mut resampler = StreamingPcmResampler::new(24_000, 48_000);
        assert_eq!(resampler.process(vec![0.0, 1.0]), vec![0.0, 0.0, 0.5, 1.0]);
        assert_eq!(resampler.process(vec![0.0]), vec![0.5, 0.0]);
    }

    struct BlockingFrameEngine {
        calls: Arc<AtomicUsize>,
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl VoiceEngine for BlockingFrameEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            unreachable!("frame engine should not receive utterances")
        }

        fn frame_config(&self) -> Option<FrameConfig> {
            Some(FrameConfig {
                sample_rate: 24_000,
                frame_size: 2,
            })
        }

        fn respond_frame(
            &mut self,
            _frame: &[f32],
            cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                let _ = self.entered.send(());
                self.release
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|e| format!("release wait failed: {e}"))?;
            }
            if cancel.load(Ordering::SeqCst) {
                return Ok(false);
            }
            emit(VoiceEvent::Text(format!("frame {call}")));
            Ok(true)
        }
    }

    #[test]
    fn frame_pipeline_interrupt_drops_queued_stale_frames() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimeFramePipeline::spawn(BlockingFrameEngine {
            calls: calls.clone(),
            entered: entered_tx,
            release: release_rx,
        })
        .expect("spawn frame pipeline");

        assert!(pipe.submit_frame(vec![0.0, 0.0]));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame should enter engine");
        assert!(pipe.submit_frame(vec![0.1, 0.1]));
        assert!(pipe.submit_frame(vec![0.2, 0.2]));
        pipe.interrupt();
        assert!(pipe.submit_frame(vec![0.3, 0.3]));
        release_tx.send(()).unwrap();

        let mut saw_new_frame = false;
        for _ in 0..4 {
            match pipe
                .events()
                .recv_timeout(Duration::from_secs(5))
                .expect("expected frame pipeline event")
            {
                VoiceEvent::Text(text) if text == "frame 1" => {
                    saw_new_frame = true;
                    break;
                }
                VoiceEvent::Interrupted => {}
                other => panic!("unexpected event {other:?}"),
            }
        }

        assert!(saw_new_frame, "new-epoch frame should still run");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "stale queued frames must not reach the engine"
        );
    }

    struct SaturatedFrameEngine {
        calls: Arc<AtomicUsize>,
        resets: Arc<AtomicUsize>,
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl VoiceEngine for SaturatedFrameEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            unreachable!("frame engine should not receive utterances")
        }

        fn frame_config(&self) -> Option<FrameConfig> {
            Some(FrameConfig {
                sample_rate: 24_000,
                frame_size: 2,
            })
        }

        fn respond_frame(
            &mut self,
            _frame: &[f32],
            _cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                let _ = self.entered.send(());
                self.release
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|e| format!("release wait failed: {e}"))?;
            }
            emit(VoiceEvent::Text(format!("frame {call}")));
            Ok(true)
        }

        fn interrupt_stream(&mut self) -> Result<(), String> {
            self.resets.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn frame_pipeline_interrupt_resets_even_when_command_queue_is_full() {
        let calls = Arc::new(AtomicUsize::new(0));
        let resets = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimeFramePipeline::spawn(SaturatedFrameEngine {
            calls: calls.clone(),
            resets: resets.clone(),
            entered: entered_tx,
            release: release_rx,
        })
        .expect("spawn frame pipeline");

        assert!(pipe.submit_frame(vec![0.0, 0.0]));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame should enter engine");
        for _ in 0..FRAME_QUEUE_CAP {
            assert!(pipe.submit_frame(vec![0.1, 0.1]));
        }
        pipe.interrupt();
        release_tx.send(()).unwrap();

        let mut saw_interrupted = false;
        for _ in 0..4 {
            match pipe
                .events()
                .recv_timeout(Duration::from_secs(5))
                .expect("expected frame pipeline event")
            {
                VoiceEvent::Text(text) if text == "frame 0" => {}
                VoiceEvent::Interrupted => {
                    saw_interrupted = true;
                    break;
                }
                other => panic!("unexpected event {other:?}"),
            }
        }

        assert!(saw_interrupted, "full queues must not drop stream reset");
        assert_eq!(
            resets.load(Ordering::SeqCst),
            1,
            "the stale queued frame should trigger the missed interrupt reset"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "stale queued frames must not reach the engine after interrupt"
        );
    }

    #[test]
    fn frame_queue_pressure_does_not_reset_stream() {
        let calls = Arc::new(AtomicUsize::new(0));
        let resets = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimeFramePipeline::spawn(SaturatedFrameEngine {
            calls: calls.clone(),
            resets: resets.clone(),
            entered: entered_tx,
            release: release_rx,
        })
        .expect("spawn frame pipeline");

        assert!(pipe.submit_frame(vec![0.0, 0.0]));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame should enter engine");
        for _ in 0..FRAME_QUEUE_CAP {
            assert!(pipe.submit_frame(vec![0.1, 0.1]));
        }
        assert_eq!(
            pipe.try_submit_frame(vec![0.2, 0.2]),
            Err(FrameSubmitError::Full),
            "ordinary queue pressure should be reported to the producer"
        );
        release_tx.send(()).unwrap();

        let mut text = 0usize;
        while text <= FRAME_QUEUE_CAP {
            match pipe
                .events()
                .recv_timeout(Duration::from_secs(5))
                .expect("expected frame pipeline event")
            {
                VoiceEvent::Text(_) => text += 1,
                VoiceEvent::TurnComplete => {}
                other => panic!("unexpected event {other:?}"),
            }
        }

        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "queue pressure alone must not reset Mimi/LM stream state"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            FRAME_QUEUE_CAP + 1,
            "accepted frames should still reach the engine in order"
        );
    }

    struct RejectTurnFrameEngine {
        frames: Arc<AtomicUsize>,
    }

    impl VoiceEngine for RejectTurnFrameEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            Err("turn path must not be used for frame engines".into())
        }

        fn frame_config(&self) -> Option<FrameConfig> {
            Some(FrameConfig {
                sample_rate: 24_000,
                frame_size: 2,
            })
        }

        fn respond_frame(
            &mut self,
            _frame: &[f32],
            _cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            let n = self.frames.fetch_add(1, Ordering::SeqCst);
            emit(VoiceEvent::Text(format!("frame {n}")));
            Ok(true)
        }
    }

    #[test]
    fn frame_capable_engines_use_frame_pipeline_not_turn_respond() {
        let frames = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimeFramePipeline::spawn(RejectTurnFrameEngine {
            frames: frames.clone(),
        })
        .expect("spawn frame pipeline");

        assert!(pipe.submit_frame(vec![0.0, 0.0]));
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::Text("frame 0".into())
        );
        assert_eq!(frames.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn turn_pipeline_rejects_frame_capable_engines() {
        let frames = Arc::new(AtomicUsize::new(0));
        let err = match RealtimePipeline::spawn(RejectTurnFrameEngine {
            frames: frames.clone(),
        }) {
            Ok(_) => panic!("turn pipeline accepted a frame-capable engine"),
            Err(err) => err,
        };
        assert!(
            err.contains("frame-capable voice engines must use RealtimeFramePipeline"),
            "{err}"
        );
        assert_eq!(frames.load(Ordering::SeqCst), 0);
    }

    /// Emits a scripted (Text, Audio) pair then completes; counts the turns it served.
    struct ScriptEngine {
        turns: Arc<AtomicUsize>,
    }
    impl VoiceEngine for ScriptEngine {
        fn respond(
            &mut self,
            utt: &Utterance,
            _cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            self.turns.fetch_add(1, Ordering::SeqCst);
            emit(VoiceEvent::Text(format!("got {}", utt.samples.len())));
            emit(VoiceEvent::Audio(vec![0.1, 0.2, 0.3]));
            Ok(true)
        }
    }

    #[test]
    fn emits_events_in_order_then_turn_complete() {
        let turns = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimePipeline::spawn(ScriptEngine {
            turns: turns.clone(),
        })
        .expect("spawn realtime pipeline");
        assert!(pipe.submit(utt(5)));
        assert_eq!(
            collect_turn(pipe.events()),
            vec![
                VoiceEvent::Text("got 5".into()),
                VoiceEvent::Audio(vec![0.1, 0.2, 0.3]),
                VoiceEvent::TurnComplete,
            ]
        );
        assert_eq!(turns.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn worker_persists_across_turns() {
        let turns = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimePipeline::spawn(ScriptEngine {
            turns: turns.clone(),
        })
        .expect("spawn realtime pipeline");
        for n in [3usize, 7] {
            assert!(pipe.submit(utt(n)));
            let evs = collect_turn(pipe.events());
            assert_eq!(evs.last(), Some(&VoiceEvent::TurnComplete));
        }
        assert_eq!(
            turns.load(Ordering::SeqCst),
            2,
            "the worker should serve every utterance"
        );
    }

    struct BlockingEngine {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl VoiceEngine for BlockingEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            let _ = self.entered.send(());
            self.release
                .recv_timeout(Duration::from_secs(5))
                .map_err(|e| format!("release wait failed: {e}"))?;
            Ok(true)
        }
    }

    #[test]
    fn utterance_queue_is_bounded_to_one_pending_turn() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimePipeline::spawn(BlockingEngine {
            entered: entered_tx,
            release: release_rx,
        })
        .expect("spawn realtime pipeline");

        assert!(pipe.submit(utt(1)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker should start first utterance");
        assert!(pipe.submit(utt(2)), "one pending utterance is allowed");
        assert!(
            !pipe.submit(utt(3)),
            "third utterance must hit backpressure instead of growing an unbounded queue"
        );

        release_tx.send(()).unwrap();
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::TurnComplete
        );
        release_tx.send(()).unwrap();
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::TurnComplete
        );
    }

    #[test]
    fn event_backpressure_sets_cancel_without_blocking() {
        let (tx, rx) = bounded::<VoiceEvent>(1);
        let cancel = AtomicBool::new(false);

        assert!(try_send_event(&tx, VoiceEvent::Text("a".into()), &cancel));
        assert!(!cancel.load(Ordering::SeqCst));
        assert!(!try_send_event(&tx, VoiceEvent::Text("b".into()), &cancel));
        assert!(cancel.load(Ordering::SeqCst));

        drop(rx);
        cancel.store(false, Ordering::SeqCst);
        assert!(!try_send_event(&tx, VoiceEvent::Text("c".into()), &cancel));
        assert!(cancel.load(Ordering::SeqCst));
    }

    /// Emits Audio forever until `cancel` is set — stands in for a long generation.
    struct LoopEngine;
    impl VoiceEngine for LoopEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            for _ in 0..100_000 {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                emit(VoiceEvent::Audio(vec![0.0]));
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(true)
        }
    }

    #[test]
    fn interrupt_aborts_in_flight_turn() {
        let pipe = RealtimePipeline::spawn(LoopEngine).expect("spawn realtime pipeline");
        assert!(pipe.submit(utt(1)));
        // Wait until generation is actually under way.
        let first = pipe
            .events()
            .recv_timeout(Duration::from_secs(5))
            .expect("turn should start");
        assert!(matches!(first, VoiceEvent::Audio(_)));

        pipe.interrupt();

        // It must terminate with Interrupted (not TurnComplete), and promptly.
        let mut seen = 0;
        loop {
            let ev = pipe
                .events()
                .recv_timeout(Duration::from_secs(5))
                .expect("expected terminal event");
            match ev {
                VoiceEvent::Interrupted => break,
                VoiceEvent::TurnComplete => panic!("barge-in should interrupt, not complete"),
                _ => {
                    seen += 1;
                    assert!(seen < 50_000, "engine did not stop after interrupt()");
                }
            }
        }
    }

    struct ErrEngine;
    impl VoiceEngine for ErrEngine {
        fn respond(
            &mut self,
            _utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            Err("boom".into())
        }
    }

    #[test]
    fn engine_error_is_reported_and_worker_survives() {
        let pipe = RealtimePipeline::spawn(ErrEngine).expect("spawn realtime pipeline");
        assert!(pipe.submit(utt(0)));
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::Error("boom".into())
        );
        // The worker must still be alive to serve a second utterance.
        assert!(pipe.submit(utt(0)));
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::Error("boom".into())
        );
    }

    /// Real-model proof of the multi-turn persistence (Gap A): two `respond()` calls on the
    /// actual LFM2.5-Audio engine must GROW the persisted `conv` — turn 2 seeds from turn 1's
    /// state (`from_parts`) and appends again, so every field is strictly longer than after
    /// turn 1. This is the engine analog of `examples/chat_multiturn` and the only test that
    /// exercises `from_parts` + the collect→`append`→save path end-to-end.
    ///
    /// `#[ignore]` (needs the model + is slow); run with:
    ///   LFM_DEVICE=metal cargo test --features metal --lib -- --ignored engine_multiturn
    /// (or `LFM_MODEL_DIR=/abs/model cargo test --lib -- --ignored engine_multiturn` on CPU BF16).
    #[test]
    #[ignore = "needs the real LFM2.5-Audio model; slow"]
    fn engine_multiturn_grows_conv() {
        use crate::{from_pretrained, get_model_dir, GenParams};

        // Device: Metal BF16 when built with the feature + LFM_DEVICE=metal; else CPU BF16.
        let device = match std::env::var("LFM_DEVICE").ok().as_deref() {
            #[cfg(feature = "metal")]
            Some("metal") => Device::new_metal(0).expect("metal device"),
            _ => {
                assert!(
                    crate::bf16_gemm::bf16_gemm_available(),
                    "CPU BF16 needs the NEON BFMMLA kernel; use Metal on this Mac"
                );
                Device::Cpu
            }
        };

        let model_ref = std::env::var("LFM_MODEL")
            .or_else(|_| std::env::var("LFM_MODEL_DIR"))
            .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
        let dir = get_model_dir(&model_ref, None).expect("resolve model dir");
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let codebooks = cfg["codebooks"].as_u64().expect("codebooks") as usize;
        let (model, proc) = from_pretrained(&dir, &device).expect("load model");

        // Short budget keeps the test quick; demo audio sampling so frames are non-degenerate.
        let params = GenParams {
            max_new_tokens: 96,
            ..GenParams::demo_defaults()
        };
        let mut engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, MIMI_RATE);

        // Reuse the reference question clip for both turns (a smoke test of growth, not content).
        let wav_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../../experiments/lfm2-audio-voice/upstream-liquid-audio/assets/question.wav"
        );
        let bytes = std::fs::read(wav_path).expect("read question.wav");
        // PCM16 mono/stereo → mono f32; assume 16-bit (the reference asset is).
        let hdr = 44usize;
        let pcm: Vec<f32> = bytes[hdr..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let utt = || Utterance {
            samples: pcm.clone(),
            rate: 16_000,
        };

        let cancel = AtomicBool::new(false);
        let mut sink = |_ev: VoiceEvent| {};

        // Turn 1: cold start → fresh ChatState + system turn → generate → append → save.
        let done1 = engine.respond(&utt(), &cancel, &mut sink).expect("turn 1");
        assert!(done1, "turn 1 should complete");
        let c1 = engine.conv.clone().expect("conv persisted after turn 1");
        let (t1, a1, m1) = (
            c1.text.dim(1).unwrap(),
            c1.audio_out.dim(1).unwrap(),
            c1.modality_flag.dim(1).unwrap(),
        );
        assert!(
            t1 > 0 && a1 > 0,
            "turn 1 must produce text + discrete audio_out (got {t1}, {a1})"
        );

        // Turn 2: seeds from c1 via from_parts → generate (prefill scatters c1's audio_out as
        // context) → append → save. Every field must be strictly longer than after turn 1.
        let done2 = engine.respond(&utt(), &cancel, &mut sink).expect("turn 2");
        assert!(done2, "turn 2 should complete");
        let c2 = engine.conv.clone().expect("conv persisted after turn 2");
        let (t2, a2, m2) = (
            c2.text.dim(1).unwrap(),
            c2.audio_out.dim(1).unwrap(),
            c2.modality_flag.dim(1).unwrap(),
        );
        assert!(t2 > t1, "turn 2 text must grow: {t1} -> {t2}");
        assert!(a2 > a1, "turn 2 audio_out must grow: {a1} -> {a2}");
        assert!(m2 > m1, "turn 2 modality_flag must grow: {m1} -> {m2}");
    }

    #[test]
    fn drop_joins_worker_and_drops_engine() {
        // A guard inside the engine flips a flag when the engine is dropped — which only
        // happens when the worker thread ends. If Drop didn't close the channel + join,
        // the flag would still be false (or the test would hang on a detached thread).
        struct Guard(Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        struct GuardedEngine {
            _g: Guard,
        }
        impl VoiceEngine for GuardedEngine {
            fn respond(
                &mut self,
                _utt: &Utterance,
                _cancel: &AtomicBool,
                emit: &mut dyn FnMut(VoiceEvent),
            ) -> Result<bool, String> {
                emit(VoiceEvent::Text("x".into()));
                Ok(true)
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        {
            let pipe = RealtimePipeline::spawn(GuardedEngine {
                _g: Guard(dropped.clone()),
            })
            .expect("spawn realtime pipeline");
            assert!(pipe.submit(utt(1)));
            let _ = collect_turn(pipe.events());
            // `pipe` drops here → channel closes → worker loop ends → engine (+ Guard) drop.
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "worker must drop the engine on shutdown"
        );
    }
}
