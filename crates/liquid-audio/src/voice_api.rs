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

/// Setup-time identity of the native platform devices. It carries no PCM or
/// process pointer. The exact value queried before model/session construction
/// is validated again when CoreAudio is mounted, so rate drift fails instead
/// of stretching or compressing generated speech.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlatformAudioConfig {
    pub capture_device: u32,
    pub playback_device: u32,
    pub capture_rate: u32,
    pub playback_rate: u32,
    pub capture_frames: u32,
    pub playback_frames: u32,
}

/// Metadata-only snapshot of the native hardware dock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlatformAudioSnapshot {
    pub started: bool,
    pub capture_enabled: bool,
    pub terminal_status: i32,
    pub captured_frames: u64,
    pub dropped_capture_frames: u64,
    pub played_frames: u64,
    pub silent_playback_frames: u64,
    pub playback_leases: u64,
    pub playback_releases: u64,
    pub claimed_playback_frames: u64,
    pub dropped_playback_frames: u64,
}

/// Tensor-free model/session edge used by application orchestration.
pub trait VoiceEngine: Send {
    /// Mount the OS callback owner directly onto the native capture/playback
    /// leases while the session is still CREATED.
    fn mount_platform_audio(&mut self, config: PlatformAudioConfig) -> Result<(), String>;

    /// Start CoreAudio after the native session/event continuation is ready.
    fn start_platform_audio(&mut self) -> Result<(), String>;

    fn set_capture_enabled(&mut self, enabled: bool) -> Result<(), String>;
    fn platform_audio_snapshot(&self) -> Result<PlatformAudioSnapshot, String>;

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

    /// Advance the native stream epoch and return the acknowledged epoch. The
    /// same native transition publishes the correlated playback-flush edge;
    /// Rust never sequences PCM retirement separately. An unacknowledged
    /// interrupt is a terminal control fault, not a best-effort hint.
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
    fn mount_platform_audio(&mut self, config: PlatformAudioConfig) -> Result<(), String> {
        (**self).mount_platform_audio(config)
    }

    fn start_platform_audio(&mut self) -> Result<(), String> {
        (**self).start_platform_audio()
    }

    fn set_capture_enabled(&mut self, enabled: bool) -> Result<(), String> {
        (**self).set_capture_enabled(enabled)
    }

    fn platform_audio_snapshot(&self) -> Result<PlatformAudioSnapshot, String> {
        (**self).platform_audio_snapshot()
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
