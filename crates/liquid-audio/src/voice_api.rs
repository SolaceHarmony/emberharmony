//! Tensor-free host protocol shared by the native session and orchestration rims.

use std::sync::atomic::AtomicBool;

use kcoro_sys::RealtimeNotifier;

/// Semantic output from an opaque voice session.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    /// Native accepted one detector-committed turn and published its exact
    /// ticket before any numerical output for that turn.
    TurnStarted,
    Text(String),
    TurnComplete,
    Interrupted,
    Error(String),
}

/// Result of draining the engine's already-published fixed event records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineProgress {
    /// The event predicate is empty; only a producer edge should resume it.
    Dormant,
    /// The bounded drain quota expired with records still ready.
    Continue,
    /// The current admitted turn and all of its promised playback leases have
    /// settled. The voice service itself remains alive for the next turn.
    Complete,
    /// The native session published its correlated terminal lifecycle edge.
    /// No further capture, playback, or event callback may be admitted after
    /// the host owner retires its platform streams for this transition.
    Stopped,
}

/// Fixed result of one hardware playback callback. The callback writes device
/// frames directly from an opaque native lease; this record contains telemetry,
/// never a PCM payload or a native address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaybackWrite {
    pub claimed_samples: usize,
    pub dropped_samples: usize,
    pub played_frames: usize,
    pub underrun_frames: usize,
    pub active: bool,
}

/// Opaque, non-cloneable playback endpoint transferred into exactly one
/// platform audio callback. Implementations retain native ticket/epoch/lease
/// identity until the final device callback retires the view. Every method is
/// bounded and realtime-safe: no allocation, mutex, blocking wait, or callback
/// into application code is permitted.
pub trait PlaybackSource: Send {
    fn rate(&self) -> u32;
    fn write_f32(&mut self, output: &mut [f32], channels: usize, flush: bool) -> PlaybackWrite;
    fn write_i16(&mut self, output: &mut [i16], channels: usize, flush: bool) -> PlaybackWrite;
    fn write_u16(&mut self, output: &mut [u16], channels: usize, flush: bool) -> PlaybackWrite;
}

/// Result of one hardware capture callback. PCM never appears in this record:
/// the endpoint writes the callback block directly into a generation-checked
/// native span and commits its descriptor exactly once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureWrite {
    pub admitted_frames: usize,
    pub dropped_frames: usize,
    pub gap_published: bool,
}

/// Result of an intentional hardware-capture discontinuity. Muted frames are
/// not acoustic silence and are not an XRUN: native advances the sample cursor
/// and rotates detector/turn state without admitting fabricated PCM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureMute {
    pub frames: usize,
    pub published: bool,
}

/// Opaque, non-cloneable capture endpoint transferred into exactly one
/// platform audio callback. Every write makes one bounded native chunk claim,
/// resolves only that claim's pointer view, fills it in place, and commits once.
/// Admission failure is nonblocking and attempts to publish an explicit XRUN
/// gap record; no PCM reservation, cursor, VAD policy, or turn identity lives
/// in Rust.
pub trait CaptureSink: Send {
    fn rate(&self) -> u32;
    /// Maximum whole-device callback admitted as one atomic native chunk.
    /// This is the setup-time admission ceiling, not CPAL's requested fixed
    /// callback cadence. The platform owner seals both before native readiness;
    /// an actual callback above this ceiling is one terminal device-fault edge.
    fn max_callback_frames(&self) -> u32;
    fn write_f32(&mut self, input: &[f32], channels: usize) -> CaptureWrite;
    fn write_i16(&mut self, input: &[i16], channels: usize) -> CaptureWrite;
    fn write_u16(&mut self, input: &[u16], channels: usize) -> CaptureWrite;
    fn mute(&mut self, frames: usize, channels: usize) -> CaptureMute;
}

/// Tensor-free model/session edge used by application orchestration.
pub trait VoiceEngine: Send {
    /// Transfer the sole callback-owned native capture endpoint.
    fn take_capture_sink(&mut self) -> Result<Option<Box<dyn CaptureSink>>, String>;

    /// Transfer the one native playback consumer into the platform device
    /// callback. Completion of a lease publishes through `notify`; no Rust or
    /// native thread waits on behalf of playback progress.
    fn take_playback_source(
        &mut self,
        notify: RealtimeNotifier,
    ) -> Result<Option<Box<dyn PlaybackSource>>, String>;

    /// Install the engine-event producer edge into the voice service before
    /// the first turn is admitted. Native engines publish fixed records, then
    /// notify this single-producer lease. Implementations that cannot mount
    /// this edge are not valid voice engines.
    fn mount_events(&mut self, notify: RealtimeNotifier) -> Result<(), String>;

    /// Drain a bounded set of fixed native event records. The caller is the
    /// retained voice service; this method never waits, allocates a waiter, or
    /// owns a worker thread.
    fn advance_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String>;

    /// Advance the native stream epoch and return the acknowledged epoch. A
    /// caller may flush playback or publish an interrupted UI state only after
    /// this edge succeeds; an unacknowledged interrupt is a terminal control
    /// fault, not a best-effort hint.
    fn interrupt_stream(&mut self) -> Result<u64, String>;

    /// Close native admission and wake owned continuations without joining.
    /// Capture/playback endpoints quiesce and release their retained leases
    /// before [`stop_session`](Self::stop_session) performs administrative
    /// settlement.
    fn request_stop(&mut self);

    /// Stop and join native continuations while retained capture producers
    /// have already been disconnected and released by the owner.
    fn stop_session(&mut self) -> Result<(), String>;
}

impl<T: VoiceEngine + ?Sized> VoiceEngine for Box<T> {
    fn take_capture_sink(&mut self) -> Result<Option<Box<dyn CaptureSink>>, String> {
        (**self).take_capture_sink()
    }

    fn take_playback_source(
        &mut self,
        notify: RealtimeNotifier,
    ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
        (**self).take_playback_source(notify)
    }

    fn mount_events(&mut self, notify: RealtimeNotifier) -> Result<(), String> {
        (**self).mount_events(notify)
    }

    fn advance_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        (**self).advance_events(cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<u64, String> {
        (**self).interrupt_stream()
    }

    fn request_stop(&mut self) {
        (**self).request_stop()
    }

    fn stop_session(&mut self) -> Result<(), String> {
        (**self).stop_session()
    }
}
