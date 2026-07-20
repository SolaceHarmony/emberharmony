//! Tensor-free host protocol shared by the native session and orchestration rims.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use kcoro_sys::RealtimeNotifier;

/// Semantic output from an opaque voice session.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
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
}

/// Fixed result of one hardware playback callback. The callback writes device
/// frames directly from an opaque native lease; this record contains telemetry,
/// never a PCM payload or a native address.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaybackWrite {
    pub claimed_samples: usize,
    pub dropped_samples: usize,
    pub played_frames: usize,
    pub underrun_frames: usize,
    pub rms: f32,
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

/// Opaque correlation identity returned when a borrowed capture span has been
/// copied once into its final retained native lease and published.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureTicket {
    pub(crate) runtime_epoch: u64,
    pub(crate) sequence: u64,
    pub(crate) generation: u32,
    pub(crate) kind: u32,
}

const CAPTURE_OPEN: u8 = 0;
const CAPTURE_WRITING: u8 = 1;
const CAPTURE_SEALED: u8 = 2;
const CAPTURE_PUBLISHED: u8 = 3;
const CAPTURE_REARMING: u8 = 4;

/// Native storage behind one retained capture reservation. Only
/// [`CaptureReservation`] touches this surface; its atomic state machine makes
/// the callback producer and continuation consumer disjoint.
pub(crate) trait CaptureStorage: Send + Sync {
    fn samples(&self) -> *mut f32;
    fn capacity(&self) -> usize;
    fn publish(&self, offset: usize, frames: usize) -> Result<Option<CaptureTicket>, String>;
    fn try_rearm(&self) -> Result<bool, String>;
    fn release(&self);
}

/// One native-resident PCM block shared by the hardware callback and retained
/// VAD continuation. Rust owns only this view and its cursors; sample storage
/// remains in the native dock for the reservation's entire generation.
pub struct CaptureReservation {
    storage: Arc<dyn CaptureStorage>,
    state: AtomicU8,
    tail: AtomicUsize,
}

impl CaptureReservation {
    pub(crate) fn new(storage: Arc<dyn CaptureStorage>) -> Arc<Self> {
        Arc::new(Self {
            storage,
            state: AtomicU8::new(CAPTURE_OPEN),
            tail: AtomicUsize::new(0),
        })
    }

    pub(crate) fn is_open(&self) -> bool {
        self.state.load(Ordering::Acquire) == CAPTURE_OPEN
    }

    pub(crate) fn len(&self) -> usize {
        self.tail.load(Ordering::Acquire)
    }

    /// Publish one mono callback block directly into the native reservation.
    /// The callback owns WRITING only for the bounded copy and never waits for
    /// a consumer. The returned count is either zero or the entire block.
    pub(crate) fn write_mono_f32(&self, samples: &[f32]) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if self
            .state
            .compare_exchange(
                CAPTURE_OPEN,
                CAPTURE_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return samples.len();
        }
        let start = self.tail.load(Ordering::Relaxed);
        if samples.len() > self.storage.capacity().saturating_sub(start) {
            self.state.store(CAPTURE_OPEN, Ordering::Release);
            return samples.len();
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                samples.as_ptr(),
                self.storage.samples().add(start),
                samples.len(),
            );
        }
        self.tail.store(start + samples.len(), Ordering::Release);
        self.state.store(CAPTURE_OPEN, Ordering::Release);
        0
    }

    /// Downmix one complete interleaved f32 callback block directly into this
    /// native reservation. The block is admitted atomically and no shuttle
    /// allocation is created.
    pub(crate) fn write_interleaved_f32(&self, samples: &[f32], channels: usize) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if channels == 1 {
            return self.write_mono_f32(samples);
        }
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        if self
            .state
            .compare_exchange(
                CAPTURE_OPEN,
                CAPTURE_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return frames;
        }
        let start = self.tail.load(Ordering::Relaxed);
        if frames > self.storage.capacity().saturating_sub(start) {
            self.state.store(CAPTURE_OPEN, Ordering::Release);
            return frames;
        }
        let target = self.storage.samples();
        for offset in 0..frames {
            let frame = &samples[offset * channels..(offset + 1) * channels];
            let sample = frame.iter().copied().sum::<f32>() / channels as f32;
            unsafe { target.add(start + offset).write(sample) };
        }
        self.tail.store(start + frames, Ordering::Release);
        self.state.store(CAPTURE_OPEN, Ordering::Release);
        0
    }

    /// Downmix one complete interleaved signed-16 callback block directly into
    /// the native reservation. No intermediate PCM allocation is created.
    pub(crate) fn write_interleaved_i16(&self, samples: &[i16], channels: usize) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        if self
            .state
            .compare_exchange(
                CAPTURE_OPEN,
                CAPTURE_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return frames;
        }
        let start = self.tail.load(Ordering::Relaxed);
        if frames > self.storage.capacity().saturating_sub(start) {
            self.state.store(CAPTURE_OPEN, Ordering::Release);
            return frames;
        }
        let target = self.storage.samples();
        for offset in 0..frames {
            let frame = &samples[offset * channels..(offset + 1) * channels];
            let sample = frame.iter().map(|sample| *sample as f32).sum::<f32>()
                / (channels as f32 * i16::MAX as f32);
            unsafe { target.add(start + offset).write(sample) };
        }
        self.tail.store(start + frames, Ordering::Release);
        self.state.store(CAPTURE_OPEN, Ordering::Release);
        0
    }

    /// Downmix one complete interleaved unsigned-16 callback block directly
    /// into this native reservation.
    pub(crate) fn write_interleaved_u16(&self, samples: &[u16], channels: usize) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        if self
            .state
            .compare_exchange(
                CAPTURE_OPEN,
                CAPTURE_WRITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return frames;
        }
        let start = self.tail.load(Ordering::Relaxed);
        if frames > self.storage.capacity().saturating_sub(start) {
            self.state.store(CAPTURE_OPEN, Ordering::Release);
            return frames;
        }
        let target = self.storage.samples();
        for offset in 0..frames {
            let frame = &samples[offset * channels..(offset + 1) * channels];
            let sample = frame
                .iter()
                .map(|sample| (*sample as f32 - 32768.0) / 32768.0)
                .sum::<f32>()
                / channels as f32;
            unsafe { target.add(start + offset).write(sample) };
        }
        self.tail.store(start + frames, Ordering::Release);
        self.state.store(CAPTURE_OPEN, Ordering::Release);
        0
    }

    pub(crate) fn with_span<R>(
        &self,
        start: usize,
        end: usize,
        consume: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        if start > end || end > self.tail.load(Ordering::Acquire) {
            return None;
        }
        let span = unsafe {
            std::slice::from_raw_parts(self.storage.samples().add(start).cast_const(), end - start)
        };
        Some(consume(span))
    }

    /// Seal at a callback boundary. If a callback is currently writing, the
    /// continuation yields; that callback publishes its tail and wakes it.
    pub(crate) fn try_seal(&self) -> bool {
        self.state
            .compare_exchange(
                CAPTURE_OPEN,
                CAPTURE_SEALED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn discard_sealed(&self) {
        debug_assert_eq!(self.state.load(Ordering::Acquire), CAPTURE_SEALED);
        self.tail.store(0, Ordering::Release);
        self.state.store(CAPTURE_OPEN, Ordering::Release);
    }

    pub(crate) fn publish(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Option<CaptureTicket>, String> {
        if self.state.load(Ordering::Acquire) != CAPTURE_SEALED
            || start >= end
            || end > self.tail.load(Ordering::Acquire)
        {
            return Err("native capture reservation has an invalid sealed view".into());
        }
        let ticket = self.storage.publish(start, end - start)?;
        self.state.store(CAPTURE_PUBLISHED, Ordering::Release);
        Ok(ticket)
    }

    pub(crate) fn try_rearm(&self) -> Result<bool, String> {
        if self
            .state
            .compare_exchange(
                CAPTURE_PUBLISHED,
                CAPTURE_REARMING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Ok(false);
        }
        match self.storage.try_rearm() {
            Ok(true) => {
                self.tail.store(0, Ordering::Release);
                self.state.store(CAPTURE_OPEN, Ordering::Release);
                Ok(true)
            }
            Ok(false) => {
                self.state.store(CAPTURE_PUBLISHED, Ordering::Release);
                Ok(false)
            }
            Err(error) => {
                self.state.store(CAPTURE_PUBLISHED, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for CaptureReservation {
    fn drop(&mut self) {
        self.storage.release();
    }
}

/// Direct production capture seam. Reservations are created during service
/// setup and write directly into their final native PCM blocks.
pub trait CaptureDock: Send {
    fn reserve(
        &mut self,
        frames: usize,
        rate: u32,
    ) -> Result<Option<Arc<CaptureReservation>>, String>;
}

/// Tensor-free model/session edge used by application orchestration.
pub trait VoiceEngine: Send {
    /// Transfer the sole capture-reservation owner. A mutable, non-cloneable
    /// endpoint structurally enforces the native producer's single reserver;
    /// the platform callback receives only the resulting block reservations.
    fn take_capture_dock(&mut self) -> Result<Option<Box<dyn CaptureDock>>, String> {
        Ok(None)
    }

    /// Transfer the one native playback consumer into the platform device
    /// callback. Completion of a lease publishes through `notify`; no Rust or
    /// native thread waits on behalf of playback progress.
    fn take_playback_source(
        &mut self,
        _notify: RealtimeNotifier,
    ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
        Ok(None)
    }

    /// Install the engine-event producer edge into the voice service before
    /// the first turn is admitted. Native engines publish fixed records, then
    /// notify this single-producer lease. `false` means this is an offline
    /// compatibility engine rather than the production continuation path.
    fn mount_events(&mut self, _notify: RealtimeNotifier) -> Result<bool, String> {
        Ok(false)
    }

    /// Correlate a capture ticket already published through [`CaptureDock`]
    /// with the native event continuation. This never waits for completion.
    fn begin_capture(&mut self, _ticket: CaptureTicket) -> Result<bool, String> {
        Err("voice engine does not support callback-driven capture".into())
    }

    /// Drain a bounded set of fixed native event records. The caller is the
    /// retained voice service; this method never waits, allocates a waiter, or
    /// owns a worker thread.
    fn advance_events(
        &mut self,
        _cancel: &AtomicBool,
        _emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        Err("voice engine does not expose callback-driven events".into())
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Close native admission and wake owned continuations without joining.
    /// Capture/playback endpoints quiesce and release their retained leases
    /// before [`stop_session`](Self::stop_session) performs administrative
    /// settlement.
    fn request_stop(&mut self) {}

    /// Stop and join native continuations while retained capture producers
    /// have already been disconnected and released by the owner.
    fn stop_session(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl<T: VoiceEngine + ?Sized> VoiceEngine for Box<T> {
    fn take_capture_dock(&mut self) -> Result<Option<Box<dyn CaptureDock>>, String> {
        (**self).take_capture_dock()
    }

    fn take_playback_source(
        &mut self,
        notify: RealtimeNotifier,
    ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
        (**self).take_playback_source(notify)
    }

    fn mount_events(&mut self, notify: RealtimeNotifier) -> Result<bool, String> {
        (**self).mount_events(notify)
    }

    fn begin_capture(&mut self, ticket: CaptureTicket) -> Result<bool, String> {
        (**self).begin_capture(ticket)
    }

    fn advance_events(
        &mut self,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<EngineProgress, String> {
        (**self).advance_events(cancel, emit)
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        (**self).interrupt_stream()
    }

    fn request_stop(&mut self) {
        (**self).request_stop()
    }

    fn stop_session(&mut self) -> Result<(), String> {
        (**self).stop_session()
    }
}
