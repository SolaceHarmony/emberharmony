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

pub use crate::voice_api::{
    CaptureDock, CaptureTicket, FrameConfig, Utterance, VoiceEngine, VoiceEvent,
};

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

enum TurnInput {
    Owned(Utterance),
    Capture(CaptureTicket),
}

struct QueuedUtterance {
    input: TurnInput,
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
    capture: Option<Arc<dyn CaptureDock>>,
    admission: Arc<AtomicBool>,
    signals: WorkerSignals,
    interrupt: Option<Arc<dyn Fn() + Send + Sync>>,
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
    admission: Arc<AtomicBool>,
    interrupt: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl RealtimePipelineHandle {
    /// Hand the worker a new utterance. Returns `false` if the bounded queue is full or the
    /// worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        if self
            .admission
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        let sent = match self.utt_tx.try_send(QueuedUtterance {
            input: TurnInput::Owned(utt),
            epoch,
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        };
        self.admission.store(false, Ordering::Release);
        sent
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
        if let Some(interrupt) = self.interrupt.as_ref() {
            interrupt();
        }
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
    input: TurnInput,
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
        match input {
            TurnInput::Owned(utt) => engine.respond(&utt, cancel_worker, &mut emit),
            TurnInput::Capture(ticket) => {
                engine.await_capture(ticket, cancel_worker, &mut emit)
            }
        }
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
        let interrupt = engine.interrupt_signal();
        let capture = engine.capture_dock();
        let admission = Arc::new(AtomicBool::new(false));
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
                                Ok(QueuedUtterance { input, epoch }) => serve_utterance(
                                    &mut engine,
                                    input,
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
                                Ok(QueuedUtterance { input, epoch }) => serve_utterance(
                                    &mut engine,
                                    input,
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
            capture,
            admission,
            signals,
            interrupt,
            event_rx,
            worker: Some(worker),
        })
    }

    /// Latch teardown and wake the worker without waiting for it to join.
    /// Drop remains the ownership boundary that performs the join.
    pub fn request_stop(&mut self) {
        self.signals.shutdown();
    }

    /// Hand the worker a new utterance. Returns `false` if the bounded queue is full or the
    /// worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        let Some(tx) = self.utt_tx.as_ref() else {
            return false;
        };
        if self
            .admission
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let epoch = self.signals.current_epoch();
        let sent = match tx.try_send(QueuedUtterance {
            input: TurnInput::Owned(utt),
            epoch,
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        };
        self.admission.store(false, Ordering::Release);
        sent
    }

    /// Publish a borrowed utterance directly into a native capture lease when
    /// available. Compatibility engines retain the owned `Utterance` queue.
    pub fn submit_pcm(&self, pcm: &[f32], rate: u32) -> Result<bool, String> {
        let Some(capture) = self.capture.as_ref() else {
            return Ok(self.submit(Utterance {
                samples: pcm.to_vec(),
                rate,
            }));
        };
        let Some(tx) = self.utt_tx.as_ref() else {
            return Ok(false);
        };
        if self
            .admission
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }
        if tx.is_full() {
            self.admission.store(false, Ordering::Release);
            return Ok(false);
        }
        let ticket = match capture.submit(pcm, rate) {
            Ok(ticket) => ticket,
            Err(error) => {
                self.admission.store(false, Ordering::Release);
                return Err(error);
            }
        };
        let Some(ticket) = ticket else {
            self.admission.store(false, Ordering::Release);
            return Ok(false);
        };
        let epoch = self.signals.current_epoch();
        let sent = tx
            .try_send(QueuedUtterance {
                input: TurnInput::Capture(ticket),
                epoch,
            })
            .is_ok();
        self.admission.store(false, Ordering::Release);
        if !sent {
            self.interrupt();
        }
        Ok(sent)
    }

    pub fn has_direct_capture(&self) -> bool {
        self.capture.is_some()
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
        if let Some(interrupt) = self.interrupt.as_ref() {
            interrupt();
        }
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
                admission: self.admission.clone(),
                interrupt: self.interrupt.clone(),
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
        if let Some(interrupt) = self.interrupt.as_ref() {
            interrupt();
        }
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

#[cfg(test)]
mod native_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::time::Duration;

    struct NativeParkedEngine {
        entered: mpsc::Sender<()>,
        wake: mpsc::Receiver<()>,
        signal: mpsc::Sender<()>,
        calls: Arc<AtomicUsize>,
    }

    impl VoiceEngine for NativeParkedEngine {
        fn respond(
            &mut self,
            _utterance: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            self.entered.send(()).unwrap();
            self.wake.recv().unwrap();
            Ok(false)
        }

        fn interrupt_signal(&self) -> Option<Arc<dyn Fn() + Send + Sync>> {
            let signal = self.signal.clone();
            let calls = self.calls.clone();
            Some(Arc::new(move || {
                calls.fetch_add(1, Ordering::SeqCst);
                let _ = signal.send(());
            }))
        }
    }

    #[test]
    fn interrupt_edge_wakes_a_native_parked_responder_without_polling() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (wake_tx, wake_rx) = mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let engine = NativeParkedEngine {
            entered: entered_tx,
            wake: wake_rx,
            signal: wake_tx,
            calls: calls.clone(),
        };
        let pipeline = RealtimePipeline::spawn(engine).unwrap();
        let handle = pipeline.handle().unwrap();
        assert!(handle.submit(Utterance {
            samples: vec![0.0; 160],
            rate: 16_000,
        }));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        handle.interrupt();
        assert_eq!(
            pipeline
                .events()
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            VoiceEvent::Interrupted
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    struct DirectDock {
        input: Mutex<Vec<f32>>,
    }

    impl CaptureDock for DirectDock {
        fn submit(&self, pcm: &[f32], rate: u32) -> Result<Option<CaptureTicket>, String> {
            assert_eq!(rate, 48_000);
            self.input.lock().unwrap().extend_from_slice(pcm);
            Ok(Some(CaptureTicket {
                runtime_epoch: 7,
                sequence: pcm.len() as u64,
                generation: 3,
                kind: 1,
            }))
        }
    }

    struct DirectEngine {
        dock: Arc<DirectDock>,
        owned_calls: Arc<AtomicUsize>,
    }

    impl VoiceEngine for DirectEngine {
        fn capture_dock(&self) -> Option<Arc<dyn CaptureDock>> {
            Some(self.dock.clone())
        }

        fn await_capture(
            &mut self,
            ticket: CaptureTicket,
            _cancel: &AtomicBool,
            emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            assert_eq!(ticket.runtime_epoch, 7);
            assert_eq!(ticket.sequence, 4);
            emit(VoiceEvent::Text("direct".into()));
            Ok(true)
        }

        fn respond(
            &mut self,
            _utterance: &Utterance,
            _cancel: &AtomicBool,
            _emit: &mut dyn FnMut(VoiceEvent),
        ) -> Result<bool, String> {
            self.owned_calls.fetch_add(1, Ordering::SeqCst);
            Err("owned utterance path was used".into())
        }
    }

    #[test]
    fn direct_capture_queues_only_the_native_ticket() {
        let dock = Arc::new(DirectDock {
            input: Mutex::new(Vec::new()),
        });
        let owned_calls = Arc::new(AtomicUsize::new(0));
        let pipeline = RealtimePipeline::spawn(DirectEngine {
            dock: dock.clone(),
            owned_calls: owned_calls.clone(),
        })
        .unwrap();
        let pcm = [0.25, -0.5, 0.75, -1.0];
        assert_eq!(pipeline.submit_pcm(&pcm, 48_000), Ok(true));
        assert_eq!(
            pipeline
                .events()
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            VoiceEvent::Text("direct".into())
        );
        assert_eq!(
            pipeline
                .events()
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            VoiceEvent::TurnComplete
        );
        assert_eq!(*dock.input.lock().unwrap(), pcm);
        assert_eq!(owned_calls.load(Ordering::SeqCst), 0);
    }
}
