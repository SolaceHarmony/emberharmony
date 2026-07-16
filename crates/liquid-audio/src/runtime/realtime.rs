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
//! **Here**: turn-based engines use a persistent inference worker thread that owns the
//! [`VoiceEngine`] and loops `recv utterance → respond (emit text + decode audio → emit PCM)
//! → TurnComplete`. Moshi-style engines bypass that utterance path through
//! [`RealtimeFramePipeline`], which owns the engine on a worker thread and advances on exact
//! PCM frames without VAD gating or playback-based resets.
//!
//! The engine is a trait so the threading is unit-tested with a fake (no model needed);
//! [`Lfm2VoiceEngine`] is the real implementation that owns the model + processor.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::moshi::demo::chat::decode_audio_frame;
use crate::moshi::models::compression::MimiModel;

#[cfg(test)]
const MIMI_RATE: u32 = 24_000; // Mimi/LFM2 detokenizer output rate.

// One queued utterance is enough: current generation + next turn. Speculative
// Prepare/Discard control messages travel a SEPARATE channel (below) so a
// prepare can never occupy the slot a committed utterance needs — a prepare
// causing a "pipeline busy" utterance drop is the failure mode this split
// exists to make impossible.
const UTTERANCE_QUEUE_CAP: usize = 1;
const CONTROL_QUEUE_CAP: usize = 2;
const EVENT_QUEUE_CAP: usize = 128;
const FRAME_QUEUE_CAP: usize = 8;

struct EventEmitter<'a> {
    tx: &'a Sender<VoiceEvent>,
    cancel: &'a AtomicBool,
    blocked: bool,
}

impl<'a> EventEmitter<'a> {
    fn new(tx: &'a Sender<VoiceEvent>, cancel: &'a AtomicBool) -> Self {
        Self {
            tx,
            cancel,
            blocked: false,
        }
    }

    fn emit(&mut self, ev: VoiceEvent) -> bool {
        if self.blocked {
            return false;
        }
        match self.tx.try_send(ev) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.cancel.store(true, Ordering::SeqCst);
                self.blocked = true;
                false
            }
        }
    }

    fn blocked(&self) -> bool {
        self.blocked
    }
}

fn interrupt_epoch(cancel: &AtomicBool, epoch: &AtomicU64) -> u64 {
    let next = epoch.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
    cancel.store(true, Ordering::SeqCst);
    next
}

struct WorkerSignals {
    cancel: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    /// Monotonic teardown latch. Set true (never cleared) as the FIRST act of
    /// shutdown, before the epoch bump and before the shutdown channel drops.
    /// The worker checks it at the decision point (loop top + before running a
    /// speculative prepare), so a queued prepare — pure optimization — can never
    /// win a race against teardown and delay the join with a non-cancellable
    /// prefill. Channel disconnect stays as the wake/backstop; the latch is the
    /// ordering guarantee disconnect alone couldn't give.
    shutdown: Arc<AtomicBool>,
    shutdown_tx: Option<Sender<()>>,
}

impl WorkerSignals {
    fn new(shutdown_tx: Sender<()>) -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            epoch: Arc::new(AtomicU64::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    fn cancel(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    fn epoch(&self) -> Arc<AtomicU64> {
        self.epoch.clone()
    }

    /// A clone of the teardown latch for the worker to read.
    fn shutdown_flag(&self) -> Arc<AtomicBool> {
        self.shutdown.clone()
    }

    fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    fn interrupt(&self) -> u64 {
        interrupt_epoch(&self.cancel, &self.epoch)
    }

    fn shutdown(&mut self) {
        // Latch FIRST: any worker that observes the release of an in-flight turn
        // after this point sees the latch set and returns without running queued
        // control. Then interrupt (abort the in-flight reply) and drop the
        // channel (wakes a blocked select).
        self.shutdown.store(true, Ordering::SeqCst);
        self.interrupt();
        drop(self.shutdown_tx.take());
    }
}

#[cfg(test)]
fn try_send_event(tx: &Sender<VoiceEvent>, ev: VoiceEvent, cancel: &AtomicBool) -> bool {
    let mut events = EventEmitter::new(tx, cancel);
    events.emit(ev)
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
    /// One decoded PCM chunk. `rate` is the ACTUAL sample rate of `pcm` —
    /// carried on every hand-off so no consumer re-asserts a constant it
    /// merely believes (the half-speed-rumble bug class: producer and
    /// consumer agreeing by coincidence until one side changes).
    Audio { pcm: Vec<f32>, rate: u32 },
    /// The reply for the current utterance finished normally (`chat.py`'s `q.put(None)`).
    /// Frame-fed Moshi pipelines do not synthesize this on silence; the stream is continuous.
    TurnComplete,
    /// The reply/output stream was cut short by an explicit interrupt.
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

    /// Handle a hard pipeline interrupt/reset. Turn pipelines call this when aborting a
    /// generated turn; frame pipelines deliberately skip it for soft output interrupts
    /// so Moshi keeps its Mimi/LM stream state alive. The default only acknowledges it.
    fn interrupt_stream(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Optional speculative prefill: the VAD calls this when the user PAUSES —
    /// before the pause has lasted long enough to commit as a turn end — so the
    /// engine can prefill the utterance while the silence window runs out. A later
    /// [`Self::respond`] with the identical utterance skips straight to generation.
    /// Best-effort: engines without a prefill notion ignore it.
    fn prepare(&mut self, _utt: &Utterance) -> Result<(), String> {
        Ok(())
    }

    /// Drop any speculative prefill state (the pause was not a turn end — speech
    /// resumed). Must restore the engine to exactly the state before `prepare`.
    fn discard_prepared(&mut self) {}
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

    fn prepare(&mut self, utt: &Utterance) -> Result<(), String> {
        (**self).prepare(utt)
    }

    fn discard_prepared(&mut self) {
        (**self).discard_prepared()
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

/// Speculative-prefill control messages — a separate channel from utterances so
/// they can never starve a committed turn (see [`VoiceEngine::prepare`]). Both
/// directions are best-effort: a lost Prepare loses only the head start, a lost
/// Discard is caught by the engine's own staleness check on the next respond.
enum Control {
    /// A PROBABLE utterance (pause detected, not yet committed): prefill it so
    /// the matching utterance skips straight to generation.
    Prepare { utt: Utterance, epoch: u64 },
    /// The pause was not a turn end — roll the speculative prefill back.
    DiscardPrepared,
}

/// Handle to the inference worker thread: submit utterances, receive reply events, request
/// barge-in. Dropping it closes the channel and joins the worker.
pub struct RealtimePipeline {
    utt_tx: Option<Sender<QueuedUtterance>>,
    ctl_tx: Option<Sender<Control>>,
    signals: WorkerSignals,
    event_rx: Receiver<VoiceEvent>,
    worker: Option<JoinHandle<()>>,
}

/// Cloneable control handle for producers that feed an existing realtime worker.
#[derive(Clone)]
pub struct RealtimePipelineHandle {
    utt_tx: Sender<QueuedUtterance>,
    ctl_tx: Sender<Control>,
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

    /// Best-effort speculative prefill of a PROBABLE utterance (see
    /// [`VoiceEngine::prepare`]). Returns `false` if the control queue is full —
    /// the caller loses the head start, never correctness.
    pub fn prepare(&self, utt: Utterance) -> bool {
        let epoch = self.epoch.load(Ordering::Acquire);
        self.ctl_tx
            .try_send(Control::Prepare { utt, epoch })
            .is_ok()
    }

    /// Roll back a speculative prefill (the pause was not a turn end). Best-effort:
    /// if the message is lost, the engine detects the stale prepared state itself
    /// on the next respond.
    pub fn discard_prepared(&self) {
        let _ = self.ctl_tx.try_send(Control::DiscardPrepared);
    }

    /// Request barge-in: abort the in-flight reply.
    pub fn interrupt(&self) {
        interrupt_epoch(&self.cancel, &self.epoch);
    }
}

/// Outcome of serving one queued utterance.
enum Served {
    Continue,
    /// Event consumer hung up — stop the worker.
    Stop,
}

/// Set the cancel flag, reset the engine's stream state, clear the flag.
fn interrupt_engine<E: VoiceEngine>(engine: &mut E, cancel: &AtomicBool) -> Result<(), String> {
    cancel.store(true, Ordering::SeqCst);
    let reset = engine.interrupt_stream();
    cancel.store(false, Ordering::SeqCst);
    reset
}

/// Handle one speculative-prefill control message (best-effort, no events).
fn handle_control<E: VoiceEngine>(engine: &mut E, ctl: Control, epoch_worker: &AtomicU64) {
    match ctl {
        Control::Prepare { utt, epoch } => {
            // A stale prepare (barge-in happened since) is simply dropped; the
            // engine's own staleness check is the real guard.
            if epoch >= epoch_worker.load(Ordering::Acquire) {
                if let Err(error) = engine.prepare(&utt) {
                    crate::vtrace!("worker: speculative prepare failed: {error}");
                }
            }
        }
        Control::DiscardPrepared => engine.discard_prepared(),
    }
}

/// Serve one committed utterance: the epoch/barge-in fencing plus the respond
/// call — the body of the inference coroutine, extracted so the worker can be
/// driven from a two-channel `select`.
fn serve_utterance<E: VoiceEngine>(
    engine: &mut E,
    utt: Utterance,
    epoch: u64,
    current_epoch: &mut u64,
    epoch_worker: &AtomicU64,
    cancel_worker: &AtomicBool,
    event_tx: &Sender<VoiceEvent>,
) -> Served {
    let mut events = EventEmitter::new(event_tx, cancel_worker);
    let latest_epoch = epoch_worker.load(Ordering::Acquire);
    if epoch < latest_epoch {
        // A stale utterance implies any speculative prefill is stale too —
        // release its parked cache now instead of at the next turn.
        engine.discard_prepared();
        if latest_epoch > *current_epoch {
            *current_epoch = latest_epoch;
            match interrupt_engine(engine, cancel_worker) {
                Ok(()) => {
                    if !events.emit(VoiceEvent::Interrupted) {
                        return Served::Stop;
                    }
                }
                Err(error) => {
                    if !events.emit(VoiceEvent::Error(error)) {
                        return Served::Stop;
                    }
                }
            }
        }
        return Served::Continue;
    }
    if epoch > *current_epoch {
        *current_epoch = epoch;
        match interrupt_engine(engine, cancel_worker) {
            Ok(()) => {
                if !events.emit(VoiceEvent::Interrupted) {
                    return Served::Stop;
                }
            }
            Err(error) => {
                if !events.emit(VoiceEvent::Error(error)) {
                    return Served::Stop;
                }
                return Served::Continue;
            }
        }
    }
    // A fresh turn clears any barge-in left set by the previous reply, so
    // it cannot carry over and abort the new one before it starts.
    cancel_worker.store(false, Ordering::SeqCst);
    // Close the clear-vs-interrupt race: if interrupt() bumped the epoch after
    // the fences above but before that store, the store just erased a live
    // barge-in and THIS utterance is stale — without this re-check, a full
    // stale reply would generate uncancellable, playing over the user.
    let latest_epoch = epoch_worker.load(Ordering::Acquire);
    if latest_epoch > *current_epoch {
        *current_epoch = latest_epoch;
        engine.discard_prepared();
        match interrupt_engine(engine, cancel_worker) {
            Ok(()) => {
                if !events.emit(VoiceEvent::Interrupted) {
                    return Served::Stop;
                }
            }
            Err(error) => {
                if !events.emit(VoiceEvent::Error(error)) {
                    return Served::Stop;
                }
            }
        }
        return Served::Continue;
    }
    let responded = {
        let mut emit = |ev: VoiceEvent| {
            events.emit(ev);
        };
        engine.respond(&utt, cancel_worker, &mut emit)
    };
    let terminal = if events.blocked() {
        VoiceEvent::Error("voice event queue full or disconnected".into())
    } else {
        match responded {
            Ok(true) => VoiceEvent::TurnComplete,
            Ok(false) => {
                let latest_epoch = epoch_worker.load(Ordering::Acquire);
                if latest_epoch > *current_epoch {
                    *current_epoch = latest_epoch;
                    match interrupt_engine(engine, cancel_worker) {
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
    if !events.emit(terminal) {
        return Served::Stop; // consumer hung up — nothing left to serve.
    }
    Served::Continue
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
        // of silently growing latency. Speculative Prepare/Discard ride their own channel so
        // they can never occupy the utterance slot.
        let (utt_tx, utt_rx) = bounded::<QueuedUtterance>(UTTERANCE_QUEUE_CAP);
        let (ctl_tx, ctl_rx) = bounded::<Control>(CONTROL_QUEUE_CAP);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(0);
        let (event_tx, event_rx) = bounded::<VoiceEvent>(EVENT_QUEUE_CAP);
        let signals = WorkerSignals::new(shutdown_tx);
        let cancel_worker = signals.cancel();
        let epoch_worker = signals.epoch();
        let shutdown_flag = signals.shutdown_flag();

        let worker = std::thread::Builder::new()
            .name("lfm2-inference".into())
            .spawn(move || {
                // The inference coroutine: serve until the pipeline drops. Stop is
                // the `shutdown` LATCH (set as the first act of teardown, before
                // any channel drop) — checked at every decision point so a queued
                // speculative Prepare can never win a race against teardown and
                // stall the join with a non-cancellable prefill. `shutdown_rx`
                // disconnect stays as the wake for a blocked select.
                let shutting_down = || shutdown_flag.load(Ordering::Acquire);
                // A speculative Prepare is pure optimization: skip it whenever an
                // utterance is already waiting (stale by construction) OR teardown
                // has latched. Both make running it wasted, non-cancellable work.
                let skip_prepare = |utt_rx: &Receiver<QueuedUtterance>| {
                    !utt_rx.is_empty() || shutting_down()
                };
                let mut current_epoch = 0u64;
                let mut ctl_open = true;
                loop {
                    if shutting_down() {
                        return;
                    }
                    // Drain queued control first (never blocks): a Prepare sent
                    // before its committing utterance is guaranteed to run first —
                    // unless it should be skipped (utterance waiting or teardown).
                    while ctl_open {
                        match ctl_rx.try_recv() {
                            Ok(Control::Prepare { .. }) if skip_prepare(&utt_rx) => {
                                crate::vtrace!(
                                    "worker: skipping speculative prepare (utterance waiting or teardown)"
                                );
                            }
                            Ok(ctl) => handle_control(&mut engine, ctl, &epoch_worker),
                            Err(crossbeam_channel::TryRecvError::Empty) => break,
                            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                ctl_open = false;
                            }
                        }
                    }
                    let served = if ctl_open {
                        crossbeam_channel::select! {
                            recv(shutdown_rx) -> _ => return,
                            recv(ctl_rx) -> msg => match msg {
                                Ok(Control::Prepare { .. }) if skip_prepare(&utt_rx) => {
                                    crate::vtrace!(
                                        "worker: skipping speculative prepare (utterance waiting or teardown)"
                                    );
                                    Served::Continue
                                }
                                Ok(ctl) => {
                                    // Same teardown guard as the loop top: when both
                                    // arms are ready, select may pick this one even
                                    // though the latch is already set.
                                    if shutting_down() {
                                        Served::Stop
                                    } else {
                                        handle_control(&mut engine, ctl, &epoch_worker);
                                        Served::Continue
                                    }
                                }
                                Err(_) => {
                                    ctl_open = false;
                                    Served::Continue
                                }
                            },
                            recv(utt_rx) -> msg => match msg {
                                Ok(QueuedUtterance { utt, epoch }) => serve_utterance(
                                    &mut engine,
                                    utt,
                                    epoch,
                                    &mut current_epoch,
                                    &epoch_worker,
                                    &cancel_worker,
                                    &event_tx,
                                ),
                                Err(_) => return,
                            },
                        }
                    } else {
                        crossbeam_channel::select! {
                            recv(shutdown_rx) -> _ => return,
                            recv(utt_rx) -> msg => match msg {
                                Ok(QueuedUtterance { utt, epoch }) => serve_utterance(
                                    &mut engine,
                                    utt,
                                    epoch,
                                    &mut current_epoch,
                                    &epoch_worker,
                                    &cancel_worker,
                                    &event_tx,
                                ),
                                Err(_) => return,
                            },
                        }
                    };
                    if matches!(served, Served::Stop) {
                        return;
                    }
                }
            })
            .map_err(|e| format!("spawn lfm2-inference worker failed: {e}"))?;

        Ok(Self {
            utt_tx: Some(utt_tx),
            ctl_tx: Some(ctl_tx),
            signals,
            event_rx,
            worker: Some(worker),
        })
    }

    /// Hand the worker a new utterance. Returns `false` if the bounded queue is full or the
    /// worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        let Some(tx) = self.utt_tx.as_ref() else {
            return false;
        };
        let epoch = self.signals.current_epoch();
        match tx.try_send(QueuedUtterance { utt, epoch }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Best-effort speculative prefill of a PROBABLE utterance (see
    /// [`VoiceEngine::prepare`]): the VAD calls this at pause onset so the prefill
    /// runs during the remaining silence window. Rides the control channel — it
    /// can never occupy an utterance slot. Returns `false` if the control queue
    /// is full: the caller loses the head start, never correctness.
    pub fn prepare(&self, utt: Utterance) -> bool {
        let Some(tx) = self.ctl_tx.as_ref() else {
            return false;
        };
        let epoch = self.signals.current_epoch();
        tx.try_send(Control::Prepare { utt, epoch }).is_ok()
    }

    /// Roll back a speculative prefill (the pause was not a turn end). Best-effort:
    /// if this message is lost, the engine detects the stale prepared state itself
    /// on the next respond.
    pub fn discard_prepared(&self) {
        let Some(tx) = self.ctl_tx.as_ref() else {
            return;
        };
        let _ = tx.try_send(Control::DiscardPrepared);
    }

    /// Request barge-in: abort the in-flight reply. The engine polls this and returns
    /// early, after which the worker emits [`VoiceEvent::Interrupted`]. Call this before
    /// submitting the interrupting utterance.
    pub fn interrupt(&self) {
        self.signals.interrupt();
    }

    /// The stream of reply events; drain it in the consumer (UI / playback feeder).
    pub fn events(&self) -> &Receiver<VoiceEvent> {
        &self.event_rx
    }

    /// A cloneable producer/control handle for external audio transports.
    pub fn handle(&self) -> Option<RealtimePipelineHandle> {
        match (self.utt_tx.as_ref(), self.ctl_tx.as_ref()) {
            (Some(utt_tx), Some(ctl_tx)) => Some(RealtimePipelineHandle {
                utt_tx: utt_tx.clone(),
                ctl_tx: ctl_tx.clone(),
                cancel: self.signals.cancel(),
                epoch: self.signals.epoch(),
            }),
            _ => None,
        }
    }
}

impl Drop for RealtimePipeline {
    fn drop(&mut self) {
        // Abort any in-flight reply fast, then drop the shutdown sender — the
        // worker's stop signal. Never wait on the DATA channels disconnecting:
        // handle clones keep those alive, and a join gated on them deadlocks
        // when a handle outlives the pipeline on the same stack (the
        // native-LiveKit session teardown did exactly that).
        self.signals.shutdown();
        drop(self.ctl_tx.take());
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
    signals: WorkerSignals,
    event_rx: Receiver<VoiceEvent>,
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

    /// Soft-interrupt frame output without resetting the engine stream. This advances
    /// the epoch so already-queued PCM frames are dropped, and it flips `cancel` so
    /// any in-flight `respond_frame` stops emitting promptly. Moshi's Mimi/LM state
    /// is preserved; a full session stop tears down the whole worker instead.
    pub fn interrupt(&self) {
        let epoch = interrupt_epoch(&self.cancel, &self.epoch);
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
        let (shutdown_tx, shutdown_rx) = bounded::<()>(0);
        let (event_tx, event_rx) = bounded::<VoiceEvent>(EVENT_QUEUE_CAP);
        let signals = WorkerSignals::new(shutdown_tx);
        let cancel_worker = signals.cancel();
        let epoch_worker = signals.epoch();

        let worker = std::thread::Builder::new()
            .name("moshi-frame-inference".into())
            .spawn(move || {
                let mut current_epoch = 0u64;
                // Stop on `shutdown_rx` disconnect (guaranteed at pipeline drop),
                // never on the frame channel alone — handle clones keep it alive.
                loop {
                    let cmd = crossbeam_channel::select! {
                        recv(shutdown_rx) -> _ => return,
                        recv(frame_rx) -> msg => match msg {
                            Ok(cmd) => cmd,
                            Err(_) => return,
                        },
                    };
                    match cmd {
                        FrameCommand::Interrupt { epoch } => {
                            if epoch <= current_epoch {
                                continue;
                            }
                            current_epoch = epoch;
                            cancel_worker.store(false, Ordering::SeqCst);
                            let mut events = EventEmitter::new(&event_tx, &cancel_worker);
                            if !events.emit(VoiceEvent::Interrupted) {
                                break;
                            }
                        }
                        FrameCommand::Pcm { pcm: frame, epoch } => {
                            let latest_epoch = epoch_worker.load(Ordering::Acquire);
                            if epoch < latest_epoch {
                                if latest_epoch > current_epoch {
                                    current_epoch = latest_epoch;
                                    cancel_worker.store(false, Ordering::SeqCst);
                                    let mut events = EventEmitter::new(&event_tx, &cancel_worker);
                                    if !events.emit(VoiceEvent::Interrupted) {
                                        break;
                                    }
                                }
                                continue;
                            }
                            if epoch > current_epoch {
                                current_epoch = epoch;
                                cancel_worker.store(false, Ordering::SeqCst);
                                let mut events = EventEmitter::new(&event_tx, &cancel_worker);
                                events.emit(VoiceEvent::Interrupted);
                            }
                            cancel_worker.store(false, Ordering::SeqCst);
                            let mut events = EventEmitter::new(&event_tx, &cancel_worker);
                            let responded = {
                                let mut emit = |ev: VoiceEvent| {
                                    events.emit(ev);
                                };
                                engine.respond_frame(&frame, &cancel_worker, &mut emit)
                            };
                            if events.blocked() {
                                events.emit(VoiceEvent::Error(
                                    "voice event queue full or disconnected".into(),
                                ));
                                break;
                            }
                            match responded {
                                Ok(true) => {}
                                Ok(false) => {
                                    let latest_epoch = epoch_worker.load(Ordering::Acquire);
                                    if latest_epoch > current_epoch {
                                        current_epoch = latest_epoch;
                                    }
                                    cancel_worker.store(false, Ordering::SeqCst);
                                    if !events.emit(VoiceEvent::Interrupted) {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if !events.emit(VoiceEvent::Error(e)) {
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
            signals,
            event_rx,
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
        let epoch = self.signals.current_epoch();
        match tx.try_send(FrameCommand::Pcm { pcm, epoch }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(FrameSubmitError::Full),
            Err(TrySendError::Disconnected(_)) => Err(FrameSubmitError::Disconnected),
        }
    }

    pub fn submit_frame(&self, pcm: Vec<f32>) -> bool {
        self.try_submit_frame(pcm).is_ok()
    }

    /// Soft-interrupt frame output without resetting the engine stream. See
    /// [`RealtimeFramePipelineHandle::interrupt`].
    pub fn interrupt(&self) {
        let epoch = self.signals.interrupt();
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
                cancel: self.signals.cancel(),
                epoch: self.signals.epoch(),
                cfg: self.cfg,
            })
    }
}

impl Drop for RealtimeFramePipeline {
    fn drop(&mut self) {
        // Stop signal first — see RealtimePipeline::drop: never gate the join
        // on data-channel disconnect while handle clones may be alive.
        self.signals.shutdown();
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

use crate::model::lfm2_hf::{Cache as LfmCache, CacheSnapshot};
use crate::{
    ChatState, GenParams, GenToken, LFM2AudioModel, LFM2AudioProcessor, LFMModality, PrefillCursor,
};

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

/// Dimensional fingerprint of a [`ConversationState`] — the staleness guard for
/// consuming a speculative prefill. Covers ALL FIVE persisted tensors: dropping
/// any one would let a prepared turn built on conversation A be consumed against
/// conversation B (a silent cache/context desync). `audio_in` (mel-frame width)
/// is included even though today's call graph never grows it without also moving
/// a segment/text dim — the fingerprint must not depend on that coincidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ConversationMark {
    text: usize,
    modality: usize,
    audio_out: usize,
    audio_in: usize,
    audio_segments: usize,
}

impl ConversationState {
    pub fn from_chat(chat: &ChatState<'_>) -> Self {
        Self {
            text: chat.text.clone(),
            audio_in: chat.audio_in.clone(),
            audio_in_lens: chat.audio_in_lens.clone(),
            audio_out: chat.audio_out.clone(),
            modality_flag: chat.modality_flag.clone(),
        }
    }

    pub fn to_chat<'p>(
        &self,
        proc: &'p LFM2AudioProcessor,
        codebooks: usize,
    ) -> candle_core::Result<ChatState<'p>> {
        ChatState::from_parts(
            proc,
            codebooks,
            self.text.clone(),
            self.audio_in.clone(),
            self.audio_in_lens.clone(),
            self.audio_out.clone(),
            self.modality_flag.clone(),
        )
    }

    fn mark(&self) -> ConversationMark {
        ConversationMark {
            text: self.text.dim(1).unwrap_or(usize::MAX),
            modality: self.modality_flag.dim(1).unwrap_or(usize::MAX),
            audio_out: self.audio_out.dim(1).unwrap_or(usize::MAX),
            audio_in: self.audio_in.dim(1).unwrap_or(usize::MAX),
            audio_segments: self.audio_in_lens.dim(0).unwrap_or(usize::MAX),
        }
    }

    /// Move all five tensors to `device` (no-op copies when already there).
    /// Used when restoring a vaulted conversation into an engine on a different
    /// compute device than the session that persisted it.
    pub fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            text: self.text.to_device(device)?,
            audio_in: self.audio_in.to_device(device)?,
            audio_in_lens: self.audio_in_lens.to_device(device)?,
            audio_out: self.audio_out.to_device(device)?,
            modality_flag: self.modality_flag.to_device(device)?,
        })
    }
}

impl ConversationMark {
    fn from_optional(conv: &Option<ConversationState>) -> Self {
        match conv {
            Some(conv) => conv.mark(),
            None => Self::default(),
        }
    }
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
/// Shared, lifecycle-proof home for a conversation (spec 09; upstream `chat.py` keeps
/// its `ChatState` in Gradio session state for the whole browser session — context must
/// outlive any one voice session). The desktop app owns one vault per chat session and
/// hands it to each engine it builds; the engine restores from it at attach and writes
/// through after every turn, so a UI-driven session rebuild no longer wipes the model's
/// conversation. The KV cache is deliberately NOT vaulted — it is engine-local and is
/// rebuilt by one full prefill on the first turn after a restore.
pub type ConversationVault = Arc<std::sync::Mutex<Option<ConversationState>>>;

pub struct Lfm2VoiceEngine {
    model: Arc<LFM2AudioModel>,
    proc: Arc<LFM2AudioProcessor>,
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
    /// The persistent backbone KV/conv cache + cursor (spec 09, W2a): what of the
    /// conversation context has already been forwarded. Each turn forwards only the
    /// suffix past the cursor, so per-turn cost stops growing with conversation length.
    /// Invariant: the cache prefix must equal the `conv`-derived context prefix — on any
    /// doubt (error mid-turn, cursor mismatch) this is dropped and the next turn falls
    /// back to a full re-prefill from `conv`, which rebuilds it. `conv` stays the source
    /// of truth; the cache is purely an accelerator.
    session_cache: Option<(LfmCache, PrefillCursor)>,
    /// Optional lifecycle-proof conversation home — restored from at attach, written
    /// through after every turn. See [`ConversationVault`].
    vault: Option<ConversationVault>,
    /// A speculative prefill of the (probable) next utterance, built by
    /// [`VoiceEngine::prepare`] during the VAD pause window and consumed by the
    /// matching `respond` — or rolled back by [`VoiceEngine::discard_prepared`].
    pending: Option<PreparedTurn>,
    /// Lifetime counts of speculative prefills consumed vs rolled back. The
    /// consumption count is what makes the accelerator FALSIFIABLE: without it,
    /// a prepare that never matches (broken fingerprint, drifted VAD trim) is
    /// indistinguishable from one that works — every equivalence test passes
    /// either way, and the feature silently degrades into pure waste. Shared
    /// atomics so a caller that moves the engine into a pipeline (the runtime,
    /// the e2e test) can still observe them from outside.
    spec_consumed: Arc<AtomicU64>,
    spec_discarded: Arc<AtomicU64>,
    // Conversation context length (cursor.positions) mirrored at each turn end —
    // the 32k-soak seam: tests watch the context grow toward
    // max_position_embeddings without reaching into the worker thread.
    ctx_positions: Arc<AtomicU64>,
}

/// See [`Lfm2VoiceEngine::pending`]: the turn context is fully assembled and every
/// suffix position EXCEPT the last is already forwarded through `cache`, so the
/// consuming `respond` starts generation after one single-position forward.
struct PreparedTurn {
    /// Identity of the utterance this was built for ([`utterance_fingerprint`]).
    fingerprint: u64,
    /// [`ConversationMark`] of the conversation this was built on — a turn landing in
    /// between (barge-in persistence) makes the prepared context stale.
    conv_mark: ConversationMark,
    /// The prepared chat's five tensors (user turn + open assistant fence included).
    chat: ConversationState,
    cache: LfmCache,
    /// Rollback point taken before the speculative forward.
    snapshot: CacheSnapshot,
    /// `Some(cursor)` → prepare continued the prior session cache, and a rollback
    /// restores `session_cache` with this cursor. `None` → prepare built a fresh
    /// cache; discarding simply drops it.
    restore_cursor: Option<PrefillCursor>,
    /// Turn-start cursor (generation bookkeeping on consume).
    cursor: PrefillCursor,
    /// Positions already forwarded: `cursor.positions + (suffix_len - 1)`.
    index_pos: usize,
    /// The final suffix embedding position, handed to `generate_with_cache`.
    tail_emb: Tensor,
}

struct TurnSetup<'p> {
    chat: ChatState<'p>,
    cache: LfmCache,
    cursor: PrefillCursor,
    input: Tensor,
    continued: bool,
}

/// FNV-1a over the utterance PCM bits + rate + length — the identity check that
/// lets a committed utterance consume the speculative prefill built for it.
fn utterance_fingerprint(utt: &Utterance) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    h = (h ^ utt.rate as u64).wrapping_mul(PRIME);
    h = (h ^ utt.samples.len() as u64).wrapping_mul(PRIME);
    for s in &utt.samples {
        h = (h ^ s.to_bits() as u64).wrapping_mul(PRIME);
    }
    h
}

/// The shared front half of [`Lfm2VoiceEngine::respond`] and `prepare`: build the
/// turn context (conversation + the new spoken user turn + the open assistant
/// fence) and select the backbone cache.
#[allow(clippy::too_many_arguments)]
fn setup_turn<'p>(
    proc: &'p LFM2AudioProcessor,
    model: &LFM2AudioModel,
    device: &Device,
    codebooks: usize,
    system_prompt: &str,
    prior: &Option<ConversationState>,
    session_cache: Option<(LfmCache, PrefillCursor)>,
    utt: &Utterance,
) -> Result<TurnSetup<'p>, String> {
    let s = |e: candle_core::Error| e.to_string();

    // First turn → fresh `ChatState` + the system turn (added once, like Python);
    // later turns → seed from the accumulated state so the prior discrete
    // `audio_out` conditions this prefill.
    let mut chat = match prior {
        None => {
            let mut c = ChatState::new(proc, codebooks).map_err(s)?;
            c.new_turn("system").map_err(s)?;
            c.add_text(system_prompt).map_err(s)?;
            c.end_turn().map_err(s)?;
            c
        }
        Some(conv) => conv.to_chat(proc, codebooks).map_err(s)?,
    };
    chat.new_turn("user").map_err(s)?;
    // Retain and borrow the utterance PCM in place. Constructing this on the
    // model device would upload it only for native mel to download it again.
    chat.add_audio_slice(&utt.samples, utt.rate).map_err(s)?;
    chat.end_turn().map_err(s)?;
    chat.new_turn("assistant").map_err(s)?;

    // Persistent cross-turn cache (spec 09, W2a): forward only the context suffix
    // the cache has not seen.
    let (prior_cache, mut cursor) = match session_cache {
        Some((c, p)) => (Some(c), p),
        None => (None, PrefillCursor::default()),
    };
    let in_emb = match model.prefill_suffix(&chat, &cursor) {
        Ok(e) => e,
        Err(err) if cursor != PrefillCursor::default() => {
            // Desynced cursor — rebuild from scratch off `conv` rather than fail the turn.
            eprintln!("[voice] persistent cache desync ({err}); falling back to full prefill");
            cursor = PrefillCursor::default();
            model.prefill_suffix(&chat, &cursor).map_err(s)?
        }
        Err(err) => return Err(s(err)),
    };
    let continued = cursor != PrefillCursor::default();
    let cache = match if continued { prior_cache } else { None } {
        Some(c) => c,
        None => model.make_cache(in_emb.dtype(), device).map_err(s)?,
    };
    Ok(TurnSetup {
        chat,
        cache,
        cursor,
        input: in_emb,
        continued,
    })
}

impl Lfm2VoiceEngine {
    /// Model and processor are shared handles (`Arc`), so the app can keep the loaded
    /// weights resident across voice sessions and hand each engine a cheap clone —
    /// building an engine must never mean loading the model again. Plain by-value
    /// call sites keep working via `Into<Arc<_>>`.
    pub fn new(
        model: impl Into<Arc<LFM2AudioModel>>,
        proc: impl Into<Arc<LFM2AudioProcessor>>,
        params: GenParams,
        codebooks: usize,
        device: Device,
        out_rate: u32,
    ) -> Self {
        Self {
            model: model.into(),
            proc: proc.into(),
            params,
            codebooks,
            device,
            out_rate,
            system_prompt: "Respond with interleaved text and audio.".to_string(),
            conv: None,
            session_cache: None,
            vault: None,
            pending: None,
            spec_consumed: Arc::new(AtomicU64::new(0)),
            spec_discarded: Arc::new(AtomicU64::new(0)),
            ctx_positions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// `(consumed, discarded)` lifetime counts for speculative prefills — the
    /// observable that proves prepare-consumption actually happens (see the
    /// field docs on why this must be assertable, not just traceable).
    pub fn speculative_stats(&self) -> (u64, u64) {
        (
            self.spec_consumed.load(Ordering::Relaxed),
            self.spec_discarded.load(Ordering::Relaxed),
        )
    }

    /// Handles to the speculative counters, cloneable BEFORE the engine moves
    /// into a pipeline — how the runtime e2e asserts that live consumption
    /// actually happened (not just that replies were equivalent).
    pub fn speculative_counters(&self) -> (Arc<AtomicU64>, Arc<AtomicU64>) {
        (self.spec_consumed.clone(), self.spec_discarded.clone())
    }

    /// Handle to the conversation context length (positions), updated at each
    /// turn end — cloneable BEFORE the engine moves into a pipeline. The
    /// 32k-context soak watches this climb toward `max_position_embeddings`.
    pub fn context_positions(&self) -> Arc<AtomicU64> {
        self.ctx_positions.clone()
    }

    /// Speculative prefill of a PROBABLE next utterance (the VAD saw a pause that
    /// has not yet lasted long enough to commit): assemble the turn context and
    /// forward every suffix position except the last, so a `respond` with the
    /// identical utterance goes straight to generation. On any error the session
    /// cache is dropped (next turn full-prefills from `conv`) — same fail-safe as
    /// `respond` itself.
    pub fn prepare_turn(&mut self, utt: &Utterance) -> Result<(), String> {
        let s = |e: candle_core::Error| e.to_string();
        self.discard_prepared_turn();
        if utt.rate == 0 {
            return Err("utterance sample rate must be non-zero".into());
        }
        let started = std::time::Instant::now();
        let prior = self.conv.clone();
        let taken = self.session_cache.take();
        let prior_cursor = taken.as_ref().map(|(_, c)| *c);
        let setup = setup_turn(
            &self.proc,
            &self.model,
            &self.device,
            self.codebooks,
            &self.system_prompt,
            &prior,
            taken,
            utt,
        )?;
        let TurnSetup {
            chat,
            mut cache,
            cursor,
            input,
            continued,
        } = setup;
        let snapshot = cache.snapshot().map_err(s)?;
        let n = input.dim(1).map_err(s)?;
        let mut index_pos = cursor.positions;
        let tail_emb = if n > 1 {
            let head = input.narrow(1, 0, n - 1).map_err(s)?;
            self.model
                .forward_embeds(&head, index_pos, &mut cache)
                .map_err(s)?;
            index_pos += n - 1;
            input.narrow(1, n - 1, 1).map_err(s)?
        } else {
            input
        };
        let chat_parts = ConversationState::from_chat(&chat);
        drop(chat);
        crate::vtrace!(
            "engine: speculative prefill ready in {:.0}ms ({} suffix positions pre-forwarded)",
            started.elapsed().as_secs_f64() * 1e3,
            n.saturating_sub(1)
        );
        self.pending = Some(PreparedTurn {
            fingerprint: utterance_fingerprint(utt),
            conv_mark: ConversationMark::from_optional(&prior),
            chat: chat_parts,
            cache,
            snapshot,
            restore_cursor: if continued { prior_cursor } else { None },
            cursor,
            index_pos,
            tail_emb,
        });
        Ok(())
    }

    /// Undo a speculative prefill: roll the cache back to its pre-`prepare_turn`
    /// state and restore `session_cache`. Called when the pause turned out not to
    /// be a turn end, or when a `respond` arrives with a different utterance.
    pub fn discard_prepared_turn(&mut self) {
        let pending = self.pending.take();
        if pending.is_some() {
            self.spec_discarded.fetch_add(1, Ordering::Relaxed);
        }
        Self::rollback_pending(pending, &mut self.session_cache);
    }

    /// Field-level body of [`Self::discard_prepared_turn`] — an associated fn over
    /// the two fields it touches, so `respond` (whose `mimi` holds a borrow of
    /// `self.proc` for the whole turn) can also call it.
    fn rollback_pending(
        pending: Option<PreparedTurn>,
        session_cache: &mut Option<(LfmCache, PrefillCursor)>,
    ) {
        if let Some(p) = pending {
            match p.restore_cursor {
                Some(cursor) => {
                    let mut cache = p.cache;
                    match cache.rollback(&p.snapshot) {
                        Ok(()) => {
                            crate::vtrace!("engine: speculative prefill rolled back");
                            *session_cache = Some((cache, cursor));
                        }
                        Err(e) => eprintln!(
                            "[voice] speculative-prefill rollback failed ({e}); \
                             next turn re-prefills from the conversation"
                        ),
                    }
                }
                // Prepare built a fresh cache (no prior session cache to protect):
                // discarding is just dropping it.
                None => crate::vtrace!("engine: speculative prefill dropped (fresh cache)"),
            }
        }
    }

    /// Override the system prompt (the desktop `TurnMode` → ASR / TTS / Interleaved prompt,
    /// verbatim from the demo's `audio-model.js`). Builder form so the single `new` call site
    /// and the tests stay unchanged.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Attach a [`ConversationVault`]: restore the conversation it holds (context
    /// survives session rebuilds) and write every completed turn back into it.
    ///
    /// The vaulted tensors are pinned to the device of the session that wrote
    /// them; this engine may live on a different one (the user flipped compute
    /// device in Settings). Migrate on restore — without it, the first
    /// `Tensor::cat(old-device, new-device)` errors and, since `conv` never
    /// changes on the error path, EVERY later turn re-errors: the conversation
    /// is bricked until app restart. If migration itself fails, drop the
    /// conversation loudly — a fresh context beats a dead one.
    pub fn with_conversation_vault(mut self, vault: ConversationVault) -> Self {
        if let Ok(saved) = vault.lock() {
            self.conv = match saved.clone() {
                Some(conv) => match conv.to_device(&self.device) {
                    Ok(conv) => Some(conv),
                    Err(e) => {
                        eprintln!(
                            "[voice] conversation vault restore failed to migrate to \
                             {:?} ({e}); starting the conversation fresh",
                            self.device.location()
                        );
                        None
                    }
                },
                None => None,
            };
        }
        self.vault = Some(vault);
        self
    }
}

impl VoiceEngine for Lfm2VoiceEngine {
    fn prepare(&mut self, utt: &Utterance) -> Result<(), String> {
        self.prepare_turn(utt)
    }

    fn discard_prepared(&mut self) {
        self.discard_prepared_turn()
    }

    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        let s = |e: candle_core::Error| e.to_string();
        // Stage clock for #141 (turn-1 latency): respond entry → prefill done →
        // first token → first audio frame, logged per turn when runtime tracing is enabled.
        // Attributes the turn-1 penalty to a stage (Metal first-generate warmup is
        // the prime suspect: first token late, not decode late).
        let respond_entry = std::time::Instant::now();

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

        // Restore the persistent conversation (clone the prior tensors — cheap Arc bumps — so a
        // hard error that returns early leaves prior history intact; we never `take`).
        let prior = self.conv.clone();

        // Consume a matching speculative prefill (built by `prepare` during the VAD
        // pause window): identical utterance AND unchanged conversation → skip
        // straight to the last suffix position (everything before it is already in
        // the cache); anything else → roll it back and take the normal path.
        let pending = self.pending.take();
        let use_pending = pending.as_ref().is_some_and(|p| {
            p.fingerprint == utterance_fingerprint(utt)
                && p.conv_mark == ConversationMark::from_optional(&prior)
        });
        let (mut chat, mut cache, mut cursor, in_emb, mut index_pos) = if use_pending {
            let p = pending.expect("matched above");
            self.spec_consumed.fetch_add(1, Ordering::Relaxed);
            crate::vtrace!(
                "engine: consuming speculative prefill ({} positions pre-forwarded)",
                p.index_pos - p.cursor.positions
            );
            let chat = p.chat.to_chat(&self.proc, self.codebooks).map_err(s)?;
            (chat, p.cache, p.cursor, p.tail_emb, p.index_pos)
        } else {
            if let Some(p) = pending {
                crate::vtrace!("engine: pending speculative prefill is stale -> rollback");
                self.spec_discarded.fetch_add(1, Ordering::Relaxed);
                Self::rollback_pending(Some(p), &mut self.session_cache);
            }
            // `take()` means any error path below leaves `session_cache = None` — the
            // next turn falls back to a full re-prefill from `conv` (source of truth).
            let taken = self.session_cache.take();
            let setup = setup_turn(
                &self.proc,
                &self.model,
                &self.device,
                self.codebooks,
                &self.system_prompt,
                &prior,
                taken,
                utt,
            )?;
            let TurnSetup {
                chat,
                cache,
                cursor,
                input: in_emb,
                continued: _continued,
            } = setup;
            let index_pos = cursor.positions;
            (chat, cache, cursor, in_emb, index_pos)
        };

        let text = self.proc.text();
        let device = &self.device;
        let codebooks = self.codebooks;
        let mut resampler = StreamingPcmResampler::new(mimi_rate, self.out_rate);
        let out_rate = self.out_rate; // captured for the emit closure (rate-honest events)
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

        // Context length at generation start: the suffix forward brings the cache
        // exactly here; every further forward is one generated token.
        let n_ctx = chat.modality_flag.dim(1).map_err(s)?;
        let text_total = chat.text.dim(1).map_err(s)?;
        let seg_total = chat.audio_in_lens.dim(0).map_err(s)?;
        let ao_total = chat.audio_out.dim(1).map_err(s)?;
        crate::vtrace!(
            "engine: turn-start — utterance {:.2}s in context as segment #{seg_total}; \
             ctx {n_ctx} positions (cache had {}, suffix {}), totals: text {text_total}, \
             audio-out {ao_total}",
            utt.samples.len() as f32 / utt.rate.max(1) as f32,
            cursor.positions,
            n_ctx - cursor.positions
        );
        if crate::voice_runtime::voice_trace_enabled() {
            // The definitive "is the turn grammar in context" answer: the exact
            // sequence the model attends over, fences visible, audio as ⟨runs⟩.
            if let Ok(t) = chat.transcript() {
                const TAIL: usize = 1600;
                let shown = if t.len() > TAIL {
                    let mut i = t.len() - TAIL;
                    while !t.is_char_boundary(i) {
                        i += 1;
                    }
                    format!("… (+{} chars)\n{}", i, &t[i..])
                } else {
                    t
                };
                crate::vtrace!("engine: context transcript:\n{shown}");
            }
        }
        let turn_started = std::time::Instant::now();
        crate::vtrace!(
            "engine: prefill done in {:.0}ms (respond entry → generation start)",
            respond_entry.elapsed().as_secs_f64() * 1e3
        );
        // First-token / first-audio-frame stage marks (set once inside the loop).
        let (mut first_token_ms, mut first_audio_ms): (Option<f64>, Option<f64>) = (None, None);

        self.model
            .generate_with_cache(
                &mut cache,
                &mut index_pos,
                in_emb,
                &self.params,
                cancel,
                |tok| {
                    if cb_err.is_some() {
                        return;
                    }
                    if first_token_ms.is_none() {
                        first_token_ms = Some(turn_started.elapsed().as_secs_f64() * 1e3);
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
                            if first_audio_ms.is_none() {
                                first_audio_ms = Some(turn_started.elapsed().as_secs_f64() * 1e3);
                            }
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
                                Ok(Some(pcm)) => emit(VoiceEvent::Audio {
                                    pcm,
                                    rate: out_rate,
                                }),
                                Ok(None) => {}
                                Err(e) => cb_err = Some(e.to_string()),
                            }
                        }
                    }
                },
            )
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
        // closes the assistant turn (early, if barge-in cut it short).
        //
        // The cache saw everything the loop forwarded: the whole pre-generation context
        // (suffix forward) plus every emitted token except the last (the loop forwards the
        // previous token before sampling the next). Advance the cursor to exactly that.
        let forwarded_generated = index_pos.saturating_sub(n_ctx);
        // `index_pos < n_ctx` means generation aborted before even the context
        // suffix was forwarded (instant barge-in). The old `saturating_sub`
        // masked that as forwarded_generated == 0 and saved a cursor whose
        // per-modality totals (full-context) contradicted its `positions` —
        // technically caught by next turn's desync guard, but only by luck.
        // Be honest: an under-forwarded cache is not resumable; drop it.
        let cache_ok = index_pos >= n_ctx && forwarded_generated <= modality_out.len();
        let (n_text_gen, n_audio_gen) = (text_ids.len(), audio_frames.len());
        if cache_ok {
            cursor.text = text_total;
            cursor.audio_segments = seg_total;
            cursor.audio_out = ao_total;
            for m in modality_out.iter().take(forwarded_generated) {
                if *m == LFMModality::Text as i64 {
                    cursor.text += 1;
                } else {
                    cursor.audio_out += 1;
                }
            }
            cursor.positions = index_pos;
            self.ctx_positions
                .store(index_pos as u64, Ordering::Relaxed);
        }

        // Even a genuinely empty generation commits the turn: the user's utterance is
        // context the moment it was spoken (and the cache has already forwarded it) —
        // the assistant turn just closes empty.
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
        }
        chat.end_turn().map_err(s)?;

        let saved = ConversationState::from_chat(&chat);
        drop(chat); // end the `&self.proc` borrow before writing `self.conv`.
        self.conv = Some(saved);
        crate::vtrace!(
            "engine: stage marks — prefill {:.0}ms, first-token {}ms, first-audio-frame {}ms \
             (first-token/first-audio from generation start; #141 turn-1 attribution)",
            turn_started.duration_since(respond_entry).as_secs_f64() * 1e3,
            first_token_ms.map_or("-".into(), |m| format!("{m:.0}")),
            first_audio_ms.map_or("-".into(), |m| format!("{m:.0}")),
        );
        crate::vtrace!(
            "engine: turn-end in {:.2}s — generated {} text + {} audio frames, \
             completed {completed}, cache {} (cursor -> {} positions), vault {}",
            turn_started.elapsed().as_secs_f32(),
            n_text_gen,
            n_audio_gen,
            if cache_ok { "kept" } else { "DROPPED" },
            cursor.positions,
            if self.vault.is_some() {
                "written"
            } else {
                "absent"
            }
        );
        if cache_ok {
            self.session_cache = Some((cache, cursor));
        }
        // Write-through to the lifecycle-proof vault: the conversation must survive a
        // UI-driven session rebuild (tensor clones are cheap Arc bumps).
        if let Some(vault) = &self.vault {
            if let Ok(mut slot) = vault.lock() {
                *slot = self.conv.clone();
            }
        }

        Ok(completed)
    }
}

mod moshi_model {
    //! In-process realtime Moshi frame loop.
    //!
    //! This is the Rust form of the core loop in upstream `liquid_audio/moshi/server.py`:
    //! fixed-size 24 kHz PCM frame -> Mimi streaming encode -> LMGen-style step ->
    //! Mimi streaming decode. No websocket, no Python process, no HTTP boundary.

    use std::path::{Path, PathBuf};

    use candle_core::{DType, Device, Error, Result, Tensor};
    use candle_transformers::generation::{LogitsProcessor, Sampling};

    const DEFAULT_MOSHI_NAME: &str = "model.safetensors";
    const DEFAULT_MIMI_NAME: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
    const DEFAULT_TEXT_TOKENIZER_NAME: &str = "tokenizer_spm_32k_3.model";
    pub const REALTIME_MOSHI_WARMUP_FRAMES: usize = 4;

    #[derive(Debug, Clone, PartialEq)]
    pub struct RealtimeMoshiFiles {
        pub moshi_weights: PathBuf,
        pub mimi_weights: PathBuf,
        pub tokenizer: PathBuf,
        pub model_type: String,
        pub params: RealtimeMoshiParams,
    }

    /// Sampling defaults from Python `moshi.models.lm.LMGen`.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct RealtimeMoshiParams {
        pub max_steps: usize,
        pub seed: u64,
        pub use_sampling: bool,
        pub audio_temperature: f64,
        pub audio_top_k: usize,
        pub text_temperature: f64,
        pub text_top_k: usize,
    }

    impl Default for RealtimeMoshiParams {
        fn default() -> Self {
            Self {
                max_steps: 4096,
                seed: 42424242,
                use_sampling: true,
                audio_temperature: 0.8,
                audio_top_k: 250,
                text_temperature: 0.7,
                text_top_k: 25,
            }
        }
    }

    impl RealtimeMoshiParams {
        pub fn with_seed(mut self, seed: u64) -> Self {
            self.seed = seed;
            self
        }

        fn from_lm_gen_config(config: Option<&serde_json::Value>) -> Result<Self> {
            let mut params = Self::default();
            let Some(config) = config.and_then(serde_json::Value::as_object) else {
                return Ok(params);
            };
            if let Some(value) = config.get("max_steps").and_then(serde_json::Value::as_u64) {
                params.max_steps = usize::try_from(value)
                    .map_err(|_| Error::Msg("lm_gen_config.max_steps does not fit usize".into()))?;
            }
            if let Some(value) = config
                .get("use_sampling")
                .and_then(serde_json::Value::as_bool)
            {
                params.use_sampling = value;
            }
            if let Some(value) = config.get("temp").and_then(serde_json::Value::as_f64) {
                params.audio_temperature = value;
            }
            if let Some(value) = config.get("top_k").and_then(serde_json::Value::as_u64) {
                params.audio_top_k = usize::try_from(value)
                    .map_err(|_| Error::Msg("lm_gen_config.top_k does not fit usize".into()))?;
            }
            if let Some(value) = config.get("temp_text").and_then(serde_json::Value::as_f64) {
                params.text_temperature = value;
            }
            if let Some(value) = config.get("top_k_text").and_then(serde_json::Value::as_u64) {
                params.text_top_k = usize::try_from(value).map_err(|_| {
                    Error::Msg("lm_gen_config.top_k_text does not fit usize".into())
                })?;
            }
            Ok(params)
        }

        fn sampling(&self, temperature: f64, top_k: usize) -> Sampling {
            if !self.use_sampling || temperature <= 1e-7 {
                return Sampling::ArgMax;
            }
            if top_k == 0 {
                return Sampling::All { temperature };
            }
            Sampling::TopK {
                k: top_k,
                temperature,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum RealtimeMoshiEvent {
        InputAudioTokenFrame(Vec<u32>),
        TextToken(u32),
        AudioTokenFrame(Vec<u32>),
        Audio { pcm: Vec<f32>, rate: u32 },
    }

    pub struct RealtimeMoshi {
        device: Device,
        mimi: ::moshi::mimi::Mimi,
        lm: ::moshi::lm::LmModel,
        state: ::moshi::lm_generate_multistream::State,
        config: ::moshi::lm_generate_multistream::Config,
        params: RealtimeMoshiParams,
        text_token: u32,
        text_pad_token: u32,
        text_eop_token: u32,
        generated_codebooks: usize,
        sample_rate: u32,
        frame_size: usize,
        skip_frames: usize,
    }

    impl RealtimeMoshi {
        pub fn new(
            mut mimi: ::moshi::mimi::Mimi,
            lm: ::moshi::lm::LmModel,
            device: Device,
            params: RealtimeMoshiParams,
        ) -> Self {
            let cfg = ::moshi::lm_generate_multistream::Config::v0_1();
            let text_token = cfg.text_start_token;
            let text_pad_token = cfg.text_pad_token;
            let text_eop_token = cfg.text_eop_token;
            let generated_codebooks = cfg.generated_audio_codebooks;
            let sample_rate = mimi.config().sample_rate as u32;
            let frame_size = (mimi.config().sample_rate / mimi.config().frame_rate) as usize;
            let state = Self::new_state(lm.clone(), params, cfg.clone());
            mimi.reset_state();
            Self {
                device,
                mimi,
                lm,
                state,
                config: cfg,
                params,
                text_token,
                text_pad_token,
                text_eop_token,
                generated_codebooks,
                sample_rate,
                frame_size,
                skip_frames: 1,
            }
        }

        fn new_state(
            lm: ::moshi::lm::LmModel,
            params: RealtimeMoshiParams,
            config: ::moshi::lm_generate_multistream::Config,
        ) -> ::moshi::lm_generate_multistream::State {
            let audio_lp = LogitsProcessor::from_sampling(
                params.seed,
                params.sampling(params.audio_temperature, params.audio_top_k),
            );
            let text_lp = LogitsProcessor::from_sampling(
                params.seed,
                params.sampling(params.text_temperature, params.text_top_k),
            );
            ::moshi::lm_generate_multistream::State::new(
                lm,
                params.max_steps,
                audio_lp,
                text_lp,
                None,
                None,
                None,
                config,
            )
        }

        pub fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        pub fn frame_size(&self) -> usize {
            self.frame_size
        }

        pub fn reset_stream(&mut self) {
            self.mimi.reset_state();
            self.state = Self::new_state(self.lm.clone(), self.params, self.config.clone());
            self.text_token = self.config.text_start_token;
            self.skip_frames = 1;
        }

        pub fn warmup(&mut self, frames: usize) -> Result<()> {
            for _ in 0..frames {
                let wav = Tensor::zeros((1, 1, self.frame_size), DType::F32, &self.device)?;
                let codes = self.mimi.encode_step(
                    &::moshi::StreamTensor::from_tensor(wav),
                    &::moshi::StreamMask::empty(),
                )?;
                if let Some(codes) = codes.as_option() {
                    let _ = self.step_codes(codes, false)?;
                }
            }
            self.reset_stream();
            Ok(())
        }

        pub fn step_pcm_frame(&mut self, pcm: &[f32]) -> Result<Vec<RealtimeMoshiEvent>> {
            if pcm.len() != self.frame_size {
                return Err(Error::Msg(format!(
                    "Moshi realtime frame must be exactly {} samples, got {}",
                    self.frame_size,
                    pcm.len()
                )));
            }
            let wav = Tensor::from_vec(pcm.to_vec(), (1, 1, pcm.len()), &self.device)?;
            let codes = self.mimi.encode_step(
                &::moshi::StreamTensor::from_tensor(wav),
                &::moshi::StreamMask::empty(),
            )?;
            let Some(codes) = codes.as_option() else {
                return Ok(Vec::new());
            };
            let reset_mimi_after_encode = self.skip_frames > 0;
            if reset_mimi_after_encode {
                // Python server.py encodes the first PCM frame, resets Mimi's streaming state,
                // then still feeds those codes into LMGen.step. The reset reapplies Mimi's
                // left-padding structure on the next encoder call without shifting LMGen.
                self.mimi.reset_state();
                self.skip_frames -= 1;
            }
            self.step_codes(codes, true)
        }

        fn step_codes(
            &mut self,
            codes: &Tensor,
            emit_events: bool,
        ) -> Result<Vec<RealtimeMoshiEvent>> {
            let codes = codes.to_dtype(DType::U32)?.to_vec3::<u32>()?;
            let mut events = Vec::new();
            for frame in 0..codes[0][0].len() {
                let input = codes[0]
                    .iter()
                    .map(|codebook| codebook[frame])
                    .collect::<Vec<_>>();
                if emit_events {
                    events.push(RealtimeMoshiEvent::InputAudioTokenFrame(input.clone()));
                }
                let text = self
                    .state
                    .step_without_ca_src(self.text_token, &input, None)?;
                self.text_token = text;
                if emit_events && text != self.text_pad_token && text != self.text_eop_token {
                    events.push(RealtimeMoshiEvent::TextToken(text));
                }
                let Some(audio) = self.state.last_audio_tokens() else {
                    continue;
                };
                let generated = audio[..self.generated_codebooks.min(audio.len())].to_vec();
                if emit_events {
                    events.push(RealtimeMoshiEvent::AudioTokenFrame(generated.clone()));
                }
                let frame =
                    Tensor::from_vec(generated.clone(), (1, generated.len(), 1), &self.device)?;
                let out = self.mimi.decode_step(
                    &::moshi::StreamTensor::from_tensor(frame),
                    &::moshi::StreamMask::empty(),
                )?;
                if let Some(out) = out.as_option() {
                    if emit_events {
                        let pcm = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                        events.push(RealtimeMoshiEvent::Audio {
                            pcm,
                            rate: self.sample_rate,
                        });
                    }
                }
            }
            Ok(events)
        }
    }

    pub fn load_realtime_moshi(
        moshi_weights: &str,
        mimi_weights: &str,
        dtype: DType,
        device: &Device,
        params: RealtimeMoshiParams,
    ) -> Result<RealtimeMoshi> {
        load_realtime_moshi_with_warmup(
            moshi_weights,
            mimi_weights,
            dtype,
            device,
            params,
            REALTIME_MOSHI_WARMUP_FRAMES,
        )
    }

    pub fn load_realtime_moshi_with_warmup(
        moshi_weights: &str,
        mimi_weights: &str,
        dtype: DType,
        device: &Device,
        params: RealtimeMoshiParams,
        warmup_frames: usize,
    ) -> Result<RealtimeMoshi> {
        let cfg = ::moshi::lm_generate_multistream::Config::v0_1();
        let mimi = ::moshi::mimi::load_b(
            None,
            mimi_weights,
            Some(cfg.generated_audio_codebooks),
            device,
        )?;
        let lm = ::moshi::lm::load_streaming(moshi_weights, dtype, device)?;
        let mut realtime = RealtimeMoshi::new(mimi, lm, device.clone(), params);
        realtime.warmup(warmup_frames)?;
        Ok(realtime)
    }

    fn is_floating_dtype(dtype: DType) -> bool {
        matches!(
            dtype,
            DType::BF16
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::F8E4M3
                | DType::F6E2M3
                | DType::F6E3M2
                | DType::F4
                | DType::F8E8M0
        )
    }

    pub fn safetensors_floating_dtype(path: &Path) -> Result<DType> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
        let mut found: Option<(DType, String)> = None;
        for (name, view) in tensors.tensors() {
            let dtype: DType = view.dtype().try_into()?;
            if !is_floating_dtype(dtype) {
                continue;
            }
            match &found {
                Some((prev, first)) if *prev != dtype => {
                    return Err(Error::Msg(format!(
                        "mixed floating safetensor dtypes: `{first}` is {prev:?}, `{name}` is {dtype:?}",
                    )));
                }
                None => found = Some((dtype, name)),
                _ => {}
            }
        }
        found
            .map(|(dtype, _)| dtype)
            .ok_or_else(|| Error::Msg("checkpoint has no floating safetensor tensors".into()))
    }

    fn validate_candle_moshi_checkpoint(path: &Path) -> Result<()> {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Err(Error::Msg(
                "native realtime Moshi requires safetensors weights; GGUF checkpoints are unsupported"
                    .into(),
            ));
        }

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
        let mut has_candle_layout = false;
        let mut has_python_layout = false;
        for (name, _) in tensors.tensors() {
            if name.starts_with("depformer.") && name.ends_with(".linear_in.weight") {
                has_candle_layout = true;
            }
            if name.starts_with("depformer_in.")
                || name.starts_with("linears.")
                || name.starts_with("depformer_emb.")
            {
                has_python_layout = true;
            }
        }
        if has_candle_layout {
            return Ok(());
        }
        if has_python_layout {
            return Err(Error::Msg(
                "Moshi checkpoint uses the PyTorch weight layout; the native desktop runtime uses the Candle Moshi layout. Download `kyutai/moshiko-candle-bf16` or choose a Candle Moshi snapshot."
                    .into(),
            ));
        }
        Err(Error::Msg(
            "Moshi checkpoint does not look like a Candle Moshi snapshot: missing `depformer.0.linear_in.weight`. Download `kyutai/moshiko-candle-bf16` or choose a Candle Moshi snapshot."
                .into(),
        ))
    }

    fn has_unimplemented_conditioning(config: &serde_json::Value) -> bool {
        const KEYS: &[&str] = &[
            "conditioners",
            "condition_provider",
            "condition_tensors",
            "fuser",
        ];
        KEYS.iter()
            .any(|key| config.get(key).is_some_and(|value| !value.is_null()))
    }

    fn has_unimplemented_cfg(config: Option<&serde_json::Value>) -> bool {
        let Some(config) = config else {
            return false;
        };
        if config
            .get("cfg_is_masked_until")
            .is_some_and(|value| !value.is_null())
            || config
                .get("cfg_is_no_text")
                .is_some_and(|value| value.as_bool().unwrap_or(true))
        {
            return true;
        }
        ["cfg_coef", "cfg_alpha"].iter().any(|key| {
            config.get(key).is_some_and(|value| {
                value
                    .as_f64()
                    .map(|n| (n - 1.0).abs() > f64::EPSILON)
                    .unwrap_or(!value.is_null())
            })
        })
    }

    fn has_unimplemented_lora(config: Option<&serde_json::Value>) -> bool {
        config.is_some_and(|config| {
            config
                .get("lora_name")
                .is_some_and(|value| !value.is_null())
        })
    }

    pub fn realtime_moshi_files(dir: &Path) -> Result<Option<RealtimeMoshiFiles>> {
        let config = dir.join("config.json");
        let (moshi_name, mimi_name, tokenizer_name, model_type, params) = if config.is_file() {
            let value: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&config).map_err(|e| Error::Msg(e.to_string()))?,
            )
            .map_err(|e| Error::Msg(e.to_string()))?;
            let lm = value
                .get("lm_config")
                .and_then(serde_json::Value::as_object);
            let name = |key: &str, default: &str| {
                value
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| {
                        lm.and_then(|lm| lm.get(key))
                            .and_then(serde_json::Value::as_str)
                    })
                    .unwrap_or(default)
                    .to_string()
            };
            let model_type = value
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    lm.and_then(|lm| lm.get("model_type"))
                        .and_then(serde_json::Value::as_str)
                })
                .unwrap_or("moshi")
                .to_string();
            if model_type != "moshi" {
                return Err(Error::Msg(format!(
                    "native realtime Moshi only supports plain `moshi` checkpoints; `{model_type}` needs the upstream Python conditioning/streaming path"
                )));
            }
            let lm_value = value.get("lm_config");
            if has_unimplemented_lora(Some(&value)) || has_unimplemented_lora(lm_value) {
                return Err(Error::Msg(
                    "native realtime Moshi does not yet implement Liquid's LoRA fuse path; use a base unconditioned Moshiko Candle snapshot"
                        .into(),
                ));
            }
            let lm_gen_config = value
                .get("lm_gen_config")
                .or_else(|| lm_value.and_then(|lm| lm.get("lm_gen_config")));
            let conditioned = has_unimplemented_conditioning(&value)
                || lm_value.is_some_and(has_unimplemented_conditioning);
            let cfg = has_unimplemented_cfg(Some(&value)) || has_unimplemented_cfg(lm_gen_config);
            if conditioned || cfg {
                return Err(Error::Msg(
                    "native realtime Moshi does not yet implement Liquid's condition_tensors/CFG fuser path; use an unconditioned Moshiko Candle snapshot"
                        .into(),
                ));
            }
            (
                name("moshi_name", DEFAULT_MOSHI_NAME),
                name("mimi_name", DEFAULT_MIMI_NAME),
                name("tokenizer_name", DEFAULT_TEXT_TOKENIZER_NAME),
                model_type,
                RealtimeMoshiParams::from_lm_gen_config(lm_gen_config)?,
            )
        } else {
            (
                DEFAULT_MOSHI_NAME.to_string(),
                DEFAULT_MIMI_NAME.to_string(),
                DEFAULT_TEXT_TOKENIZER_NAME.to_string(),
                "moshi".to_string(),
                RealtimeMoshiParams::default(),
            )
        };
        let files = RealtimeMoshiFiles {
            moshi_weights: dir.join(moshi_name),
            mimi_weights: dir.join(mimi_name),
            tokenizer: dir.join(tokenizer_name),
            model_type,
            params,
        };
        if files.moshi_weights.is_file()
            && files.mimi_weights.is_file()
            && files.tokenizer.is_file()
        {
            validate_candle_moshi_checkpoint(&files.moshi_weights)?;
            return Ok(Some(files));
        }
        Ok(None)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn temp_dir(name: &str) -> PathBuf {
            let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        fn touch(path: &Path) {
            std::fs::write(path, "").unwrap();
        }

        fn write_candle_moshi(path: &Path) {
            let mut tensors = std::collections::HashMap::new();
            let value = Tensor::zeros((1, 1), DType::BF16, &Device::Cpu).unwrap();
            tensors.insert("depformer.0.linear_in.weight", value);
            candle_core::safetensors::save(&tensors, path).unwrap();
        }

        fn write_python_moshi(path: &Path) {
            let mut tensors = std::collections::HashMap::new();
            let value = Tensor::zeros((1, 1), DType::BF16, &Device::Cpu).unwrap();
            tensors.insert("depformer_in.0.weight", value);
            candle_core::safetensors::save(&tensors, path).unwrap();
        }

        #[test]
        fn realtime_moshi_files_accepts_legacy_default_names() {
            let dir = temp_dir("emberharmony-moshi-default");
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let files = realtime_moshi_files(&dir).unwrap().unwrap();
            assert_eq!(files.moshi_weights, dir.join(DEFAULT_MOSHI_NAME));
            assert_eq!(files.mimi_weights, dir.join(DEFAULT_MIMI_NAME));
            assert_eq!(files.tokenizer, dir.join(DEFAULT_TEXT_TOKENIZER_NAME));
            assert_eq!(files.model_type, "moshi");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_uses_python_root_config_overrides() {
            let dir = temp_dir("emberharmony-moshi-config");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "moshi_name": "moshi-custom.safetensors",
                    "mimi_name": "mimi-custom.safetensors",
                    "tokenizer_name": "custom.model",
                    "model_type": "moshi",
                    "lm_gen_config": {
                        "temp": 0.6,
                        "top_k": 40,
                        "temp_text": 0.5,
                        "top_k_text": 7,
                        "use_sampling": false
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join("moshi-custom.safetensors"));
            touch(&dir.join("mimi-custom.safetensors"));
            touch(&dir.join("custom.model"));

            let files = realtime_moshi_files(&dir).unwrap().unwrap();
            assert_eq!(files.moshi_weights, dir.join("moshi-custom.safetensors"));
            assert_eq!(files.mimi_weights, dir.join("mimi-custom.safetensors"));
            assert_eq!(files.tokenizer, dir.join("custom.model"));
            assert_eq!(files.model_type, "moshi");
            assert_eq!(files.params.audio_temperature, 0.6);
            assert_eq!(files.params.audio_top_k, 40);
            assert_eq!(files.params.text_temperature, 0.5);
            assert_eq!(files.params.text_top_k, 7);
            assert!(!files.params.use_sampling);
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_accepts_legacy_lm_config_overrides() {
            let dir = temp_dir("emberharmony-moshi-legacy-config");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "lm_config": {
                        "moshi_name": "moshi-custom.safetensors",
                        "mimi_name": "mimi-custom.safetensors",
                        "tokenizer_name": "custom.model",
                        "lm_gen_config": {
                            "temp": 0.55,
                            "top_k": 33,
                            "temp_text": 0.45,
                            "top_k_text": 9,
                            "use_sampling": false
                        }
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join("moshi-custom.safetensors"));
            touch(&dir.join("mimi-custom.safetensors"));
            touch(&dir.join("custom.model"));

            let files = realtime_moshi_files(&dir).unwrap().unwrap();
            assert_eq!(files.moshi_weights, dir.join("moshi-custom.safetensors"));
            assert_eq!(files.mimi_weights, dir.join("mimi-custom.safetensors"));
            assert_eq!(files.tokenizer, dir.join("custom.model"));
            assert_eq!(files.params.audio_temperature, 0.55);
            assert_eq!(files.params.audio_top_k, 33);
            assert_eq!(files.params.text_temperature, 0.45);
            assert_eq!(files.params.text_top_k, 9);
            assert!(!files.params.use_sampling);
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_incomplete_snapshot() {
            let dir = temp_dir("emberharmony-moshi-incomplete");
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));

            assert!(realtime_moshi_files(&dir).unwrap().is_none());
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_pytorch_weight_layout() {
            let dir = temp_dir("emberharmony-moshi-pytorch");
            write_python_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("PyTorch weight layout"), "{err}");
            assert!(err.contains("kyutai/moshiko-candle-bf16"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_gguf_before_dtype_probe() {
            let dir = temp_dir("emberharmony-moshi-gguf");
            std::fs::write(dir.join("config.json"), r#"{ "moshi_name": "model.gguf" }"#).unwrap();
            touch(&dir.join("model.gguf"));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("requires safetensors"), "{err}");
            assert!(err.contains("GGUF"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_conditioned_model_types() {
            let dir = temp_dir("emberharmony-moshi-conditioned");
            std::fs::write(dir.join("config.json"), r#"{ "model_type": "hibiki" }"#).unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("plain `moshi`"), "{err}");
            assert!(err.contains("upstream Python conditioning"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_lora_config() {
            let dir = temp_dir("emberharmony-moshi-lora");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "model_type": "moshi",
                    "lora_name": "adapter.safetensors"
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("LoRA fuse"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_nested_lora_config() {
            let dir = temp_dir("emberharmony-moshi-nested-lora");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "model_type": "moshi",
                    "lm_config": {
                        "lora_name": "adapter.safetensors"
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("LoRA fuse"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_conditioner_fuser_config() {
            let dir = temp_dir("emberharmony-moshi-fuser");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "model_type": "moshi",
                    "lm_config": {
                        "conditioners": { "description": {} },
                        "fuser": { "sum": ["description"] }
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("condition_tensors/CFG"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_condition_provider_config() {
            let dir = temp_dir("emberharmony-moshi-condition-provider");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "model_type": "moshi",
                    "lm_config": {
                        "condition_provider": { "description": {} }
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("condition_tensors/CFG"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn realtime_moshi_files_rejects_cfg_generation_config() {
            let dir = temp_dir("emberharmony-moshi-cfg");
            std::fs::write(
                dir.join("config.json"),
                r#"{
                    "model_type": "moshi",
                    "lm_gen_config": {
                        "cfg_coef": 2.0,
                        "cfg_is_no_text": true
                    }
                }"#,
            )
            .unwrap();
            write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
            touch(&dir.join(DEFAULT_MIMI_NAME));
            touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

            let err = realtime_moshi_files(&dir).unwrap_err().to_string();
            assert!(err.contains("condition_tensors/CFG"), "{err}");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}

pub use moshi_model::{
    load_realtime_moshi, load_realtime_moshi_with_warmup, realtime_moshi_files,
    safetensors_floating_dtype, RealtimeMoshi, RealtimeMoshiEvent, RealtimeMoshiFiles,
    RealtimeMoshiParams, REALTIME_MOSHI_WARMUP_FRAMES,
};

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
                    emit(VoiceEvent::Audio {
                        pcm,
                        rate: self.out_rate,
                    });
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
    use crate::processor::{FilterbankFeatures, MelConfig};
    use std::collections::HashMap;
    use std::sync::{atomic::AtomicUsize, mpsc, Mutex};
    use std::time::Duration;
    use tokenizers::models::wordlevel::WordLevel;

    fn utt(n: usize) -> Utterance {
        Utterance {
            samples: vec![0.0; n],
            rate: 16_000,
        }
    }

    fn test_processor() -> LFM2AudioProcessor {
        let dev = Device::Cpu;
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<|startoftext|>".to_string(), 1);
        let tokenizer = tokenizers::Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );
        let audio = FilterbankFeatures::new(
            MelConfig {
                sample_rate: 16_000,
                n_window_size: 400,
                n_window_stride: 160,
                n_fft: 512,
                nfilt: 8,
                preemph: 0.97,
                log_zero_guard_value: 2f64.powi(-24),
                mag_power: 2.0,
                pad_to: 16,
                exact_pad: false,
            },
            &dev,
        )
        .unwrap();
        LFM2AudioProcessor::new(tokenizer, audio, None, None, dev)
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
    fn frame_pipeline_interrupt_drops_stale_frames_without_reset() {
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

        assert!(
            saw_interrupted,
            "full queues must not drop interrupt acknowledgement"
        );
        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "interrupting frame output must not reset Mimi/LM stream state"
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

    struct OneOutputFrameEngine {
        calls: Arc<AtomicUsize>,
    }

    impl VoiceEngine for OneOutputFrameEngine {
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
                emit(VoiceEvent::Audio {
                    pcm: vec![0.0, 0.0],
                    rate: 24_000,
                });
            }
            Ok(true)
        }
    }

    #[test]
    fn frame_pipeline_does_not_synthesize_turn_complete_on_idle_frames() {
        let calls = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimeFramePipeline::spawn(OneOutputFrameEngine {
            calls: calls.clone(),
        })
        .expect("spawn frame pipeline");

        for _ in 0..8 {
            assert!(pipe.submit_frame(vec![0.0, 0.0]));
        }
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
            VoiceEvent::Audio {
                pcm: vec![0.0, 0.0],
                rate: 24_000
            }
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while calls.load(Ordering::SeqCst) < 8 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        assert_eq!(
            pipe.events().recv_timeout(Duration::from_millis(50)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout),
            "frame-fed Moshi is continuous and must not invent turn boundaries"
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
            emit(VoiceEvent::Audio {
                pcm: vec![0.1, 0.2, 0.3],
                rate: 24_000,
            });
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
                VoiceEvent::Audio {
                    pcm: vec![0.1, 0.2, 0.3],
                    rate: 24_000
                },
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

    /// Records prepare/respond/discard call order (by utterance length) and can
    /// optionally block inside respond — the fake that pins the worker's
    /// control-channel semantics, which no test covered before.
    struct RecordingEngine {
        log: Arc<Mutex<Vec<String>>>,
        prepare_fails: bool,
        entered: Option<mpsc::Sender<()>>,
        release: Option<mpsc::Receiver<()>>,
    }

    impl RecordingEngine {
        fn free(log: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                log,
                prepare_fails: false,
                entered: None,
                release: None,
            }
        }

        fn blocking(
            log: Arc<Mutex<Vec<String>>>,
            entered: mpsc::Sender<()>,
            release: mpsc::Receiver<()>,
        ) -> Self {
            Self {
                log,
                prepare_fails: false,
                entered: Some(entered),
                release: Some(release),
            }
        }
    }

    impl VoiceEngine for RecordingEngine {
        fn respond(
            &mut self,
            utt: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("respond:{}", utt.samples.len()));
            if let (Some(entered), Some(release)) = (&self.entered, &self.release) {
                let _ = entered.send(());
                release
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|e| format!("release wait failed: {e}"))?;
            }
            Ok(true)
        }

        fn prepare(&mut self, utt: &Utterance) -> Result<(), String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("prepare:{}", utt.samples.len()));
            if self.prepare_fails {
                Err("prepare exploded".into())
            } else {
                Ok(())
            }
        }

        fn discard_prepared(&mut self) {
            self.log.lock().unwrap().push("discard".into());
        }
    }

    /// Poll the recording log until it holds `want` entries (bounded).
    fn wait_log(log: &Arc<Mutex<Vec<String>>>, want: usize) -> Vec<String> {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let snapshot = log.lock().unwrap().clone();
            if snapshot.len() >= want {
                return snapshot;
            }
            if std::time::Instant::now() >= deadline {
                panic!("log never reached {want} entries: {snapshot:?}");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Regression for the shutdown deadlock: dropping the pipeline while a
    /// handle clone is still alive must join promptly — the live-LiveKit
    /// teardown drops the pipeline with a handle on the same stack, which
    /// wedged forever when shutdown depended on data-channel disconnect.
    #[test]
    fn drop_with_live_handle_joins_promptly() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let pipe = RealtimePipeline::spawn(RecordingEngine::free(log)).expect("spawn");
        let handle = pipe.handle().expect("handle");

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(pipe);
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("pipeline drop deadlocked while a handle clone was alive");
        drop(handle);
    }

    #[test]
    fn shutdown_skips_queued_prepare() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimePipeline::spawn(RecordingEngine::blocking(
            log.clone(),
            entered_tx,
            release_rx,
        ))
        .expect("spawn");
        let handle = pipe.handle().expect("handle");

        assert!(pipe.submit(utt(1)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 1 started");
        assert!(handle.prepare(utt(7)));

        // Establish the teardown latch BEFORE releasing the in-flight turn —
        // faithfully modeling production, where `drop()` latches shutdown and
        // THEN the running turn completes and the worker loops seeing the latch.
        // The worker is parked inside respond:1 (blocked on release), so setting
        // the latch now guarantees it is observed the instant the worker loops —
        // no dependence on drop-thread scheduling, which is what made this racy.
        let shutdown_flag = pipe.signals.shutdown_flag();

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(pipe); // shutdown() re-latches idempotently, then joins
            let _ = done_tx.send(());
        });
        shutdown_flag.store(true, Ordering::SeqCst);
        release_tx.send(()).unwrap();
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("pipeline drop deadlocked with queued prepare");
        drop(handle);

        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            log.lock().unwrap().clone(),
            vec!["respond:1".to_string()],
            "shutdown must not run queued speculative prepare"
        );
    }

    /// Same contract for the frame pipeline (identical shutdown structure).
    #[test]
    fn frame_pipeline_drop_with_live_handle_joins_promptly() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, _entered_rx) = mpsc::channel();
        let (_release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimeFramePipeline::spawn(BlockingFrameEngine {
            calls,
            entered: entered_tx,
            release: release_rx,
        })
        .expect("spawn frame pipeline");
        let handle = pipe.handle().expect("handle");

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            drop(pipe);
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("frame pipeline drop deadlocked while a handle clone was alive");
        drop(handle);
    }

    /// A Prepare sent before its committing utterance runs first when the
    /// worker is otherwise idle — the ordering the consume path depends on.
    #[test]
    fn prepare_runs_before_its_committing_utterance() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimePipeline::spawn(RecordingEngine::blocking(
            log.clone(),
            entered_tx,
            release_rx,
        ))
        .expect("spawn");

        assert!(pipe.submit(utt(1)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 1 started");
        // Queued while the worker is busy; no other utterance waits.
        assert!(pipe.prepare(utt(7)));
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());

        // After turn 1, the drain runs the prepare; then commit the utterance.
        wait_log(&log, 2);
        assert!(pipe.submit(utt(7)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 7 started");
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());

        assert_eq!(
            wait_log(&log, 3),
            vec!["respond:1", "prepare:7", "respond:7"],
            "prepare must run after the busy turn but before its own utterance"
        );
    }

    /// A Prepare must never DELAY a committed utterance that is already
    /// waiting: it is stale by construction (built without that turn) and is
    /// skipped, not run-then-rolled-back — the time-axis version of the
    /// queue-slot starvation this channel split exists to prevent.
    #[test]
    fn prepare_skipped_when_utterance_is_waiting() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimePipeline::spawn(RecordingEngine::blocking(
            log.clone(),
            entered_tx,
            release_rx,
        ))
        .expect("spawn");

        assert!(pipe.submit(utt(1)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 1 started");
        assert!(pipe.prepare(utt(7)), "control queue accepts the prepare");
        assert!(pipe.submit(utt(2)), "utterance queue accepts the next turn");
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 2 started");
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());

        assert_eq!(
            wait_log(&log, 2),
            vec!["respond:1", "respond:2"],
            "the stale prepare must be skipped, not run ahead of the waiting utterance"
        );
    }

    /// A Prepare queued before a barge-in carries a stale epoch and must be
    /// dropped without reaching the engine.
    #[test]
    fn stale_epoch_prepare_is_dropped() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let pipe = RealtimePipeline::spawn(RecordingEngine::blocking(
            log.clone(),
            entered_tx,
            release_rx,
        ))
        .expect("spawn");

        assert!(pipe.submit(utt(1)));
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 1 started");
        assert!(pipe.prepare(utt(7))); // epoch 0
        pipe.interrupt(); // epoch -> 1: the prepare is now stale
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());

        // Prove the worker is alive and the stale prepare never ran.
        assert!(pipe.submit(utt(8)));
        // The epoch bump makes the worker emit Interrupted before serving.
        let _ = collect_turn(pipe.events());
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn 8 started");
        release_tx.send(()).unwrap();
        let _ = collect_turn(pipe.events());

        let entries = wait_log(&log, 2);
        assert!(
            !entries.iter().any(|e| e.starts_with("prepare")),
            "stale-epoch prepare reached the engine: {entries:?}"
        );
        assert_eq!(entries, vec!["respond:1", "respond:8"]);
    }

    /// DiscardPrepared is delivered to the engine.
    #[test]
    fn discard_prepared_reaches_engine() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let pipe = RealtimePipeline::spawn(RecordingEngine::free(log.clone())).expect("spawn");

        assert!(pipe.prepare(utt(7)));
        wait_log(&log, 1);
        pipe.discard_prepared();
        assert_eq!(wait_log(&log, 2), vec!["prepare:7", "discard"]);
    }

    /// A failing prepare is logged-and-swallowed; the worker keeps serving.
    #[test]
    fn prepare_error_does_not_kill_worker() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut engine = RecordingEngine::free(log.clone());
        engine.prepare_fails = true;
        let pipe = RealtimePipeline::spawn(engine).expect("spawn");

        assert!(pipe.prepare(utt(7)));
        wait_log(&log, 1);
        assert!(pipe.submit(utt(1)));
        assert_eq!(
            collect_turn(pipe.events()).last(),
            Some(&VoiceEvent::TurnComplete),
            "worker must survive a prepare error and serve the next turn"
        );
        assert_eq!(wait_log(&log, 2), vec!["prepare:7", "respond:1"]);
    }

    /// The five tensors of a vaulted conversation migrate across compute
    /// devices on restore — without this, a settings device flip mid-chat
    /// bricked every subsequent voice turn (device-mismatch on the first cat).
    #[cfg(feature = "metal")]
    #[test]
    fn conversation_state_migrates_devices() {
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0).expect("metal device");
        let conv = ConversationState {
            text: Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &cpu).unwrap(),
            audio_in: Tensor::zeros((4, 2), DType::F32, &cpu).unwrap(),
            audio_in_lens: Tensor::from_vec(vec![2i64], (1,), &cpu).unwrap(),
            audio_out: Tensor::zeros((8, 2), DType::I64, &cpu).unwrap(),
            modality_flag: Tensor::from_vec(vec![0i64, 0, 0], (1, 3), &cpu).unwrap(),
        };
        let moved = conv.to_device(&metal).expect("migrate to metal");
        assert_eq!(moved.text.device().location(), metal.location());
        assert_eq!(moved.audio_out.device().location(), metal.location());
        // Values survive the trip.
        let round = moved.to_device(&cpu).expect("migrate back");
        assert_eq!(
            round.text.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn conversation_state_round_trips_chat_state_dimensions_and_mark() {
        let proc = test_processor();
        let dev = proc.device();
        let mut chat = ChatState::new(&proc, 8).unwrap();
        let wave = Tensor::zeros((1, 320), DType::F32, dev).unwrap();
        chat.new_turn("user").unwrap();
        chat.add_audio(&wave, 16_000).unwrap();
        chat.end_turn().unwrap();
        chat.new_turn("assistant").unwrap();
        let text = Tensor::from_vec(vec![0i64], (1, 1), dev).unwrap();
        let audio = Tensor::zeros((8, 1), DType::I64, dev)
            .unwrap()
            .narrow(1, 0, 0)
            .unwrap();
        let modality = Tensor::from_vec(vec![LFMModality::Text as i64], (1, 1), dev).unwrap();
        chat.append(&text, &audio, &modality).unwrap();
        chat.end_turn().unwrap();

        let saved = ConversationState::from_chat(&chat);
        let restored = saved.to_chat(&proc, 8).unwrap();
        let round = ConversationState::from_chat(&restored);

        assert_eq!(saved.mark(), round.mark());
        assert_eq!(saved.text.dims(), round.text.dims());
        assert_eq!(saved.audio_in.dims(), round.audio_in.dims());
        assert_eq!(saved.audio_in_lens.dims(), round.audio_in_lens.dims());
        assert_eq!(saved.audio_out.dims(), round.audio_out.dims());
        assert_eq!(saved.modality_flag.dims(), round.modality_flag.dims());

        // The mark must fingerprint ALL FIVE tensors: a change to audio_in's
        // mel-frame width alone (with every other dim held constant) must make
        // the marks differ, or a stale speculative prefill could be consumed
        // against a mutated conversation. Guards against silently dropping a
        // tensor from the fingerprint (adversarial-review finding).
        let mut wider = saved.clone();
        wider.audio_in = Tensor::zeros(
            (
                saved.audio_in.dim(0).unwrap(),
                saved.audio_in.dim(1).unwrap() + 1,
            ),
            DType::F32,
            dev,
        )
        .unwrap();
        assert_ne!(
            saved.mark(),
            wider.mark(),
            "mark must distinguish an audio_in-only change"
        );
    }

    #[test]
    fn utterance_queue_is_bounded() {
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
        // Speculative control messages ride their own channel: a prepare can
        // NEVER occupy the utterance slot (the regression that caused live
        // "pipeline busy" utterance drops).
        assert!(
            pipe.prepare(utt(9)),
            "prepare must be accepted even with the utterance queue full"
        );

        for _ in 0..2 {
            release_tx.send(()).unwrap();
            assert_eq!(
                pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(),
                VoiceEvent::TurnComplete
            );
        }
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
                emit(VoiceEvent::Audio {
                    pcm: vec![0.0],
                    rate: 24_000,
                });
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
        assert!(matches!(first, VoiceEvent::Audio { .. }));

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
    /// The caller supplies the device, just like the production settings boundary.
    /// Backend features control availability only; they never select a device.
    fn engine_multiturn_grows_conv_on(device: Device) {
        use crate::{from_pretrained, GenParams};

        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../experiments/lfm2-audio-voice/model");
        assert!(
            dir.join("config.json").is_file(),
            "missing model fixture: {}",
            dir.display()
        );
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

        // Reuse the reference question clip for both turns (a smoke test of growth, not
        // content). The upstream clip is vendored in-crate.
        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/question.wav");
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

    /// `#[ignore]` because it needs the repository model fixture and is slow.
    /// Run with `cargo test --lib -- --ignored engine_multiturn_grows_conv_cpu`.
    #[test]
    #[ignore = "needs the real LFM2.5-Audio model; slow"]
    fn engine_multiturn_grows_conv_cpu() {
        assert!(
            crate::flashkern::native_engine::bf16_gemm_available(),
            "CPU BF16 needs the in-tree BFMMLA kernel"
        );
        engine_multiturn_grows_conv_on(Device::Cpu);
    }

    /// Run with
    /// `cargo test --features metal --lib -- --ignored engine_multiturn_grows_conv_metal`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "needs the real LFM2.5-Audio model; slow"]
    fn engine_multiturn_grows_conv_metal() {
        engine_multiturn_grows_conv_on(Device::new_metal(0).expect("metal device"));
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
