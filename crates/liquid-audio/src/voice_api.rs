//! Tensor-free host protocol shared by the native session and orchestration rims.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// A captured user utterance handed to the native worker.
pub struct Utterance {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// Fixed-rate frame contract for continuous duplex engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    pub sample_rate: u32,
    pub frame_size: usize,
}

/// Semantic output from an opaque voice session.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    Text(String),
    Audio { pcm: Vec<f32>, rate: u32 },
    TurnComplete,
    Interrupted,
    Error(String),
}

/// Tensor-free model/session edge used by application orchestration.
pub trait VoiceEngine: Send {
    fn respond(
        &mut self,
        utterance: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String>;

    fn frame_config(&self) -> Option<FrameConfig> {
        None
    }

    fn respond_frame(
        &mut self,
        _frame: &[f32],
        _cancel: &AtomicBool,
        _emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        Err("voice engine does not support realtime PCM frames".into())
    }

    fn interrupt_stream(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn interrupt_signal(&self) -> Option<Arc<dyn Fn() + Send + Sync>> {
        None
    }

    fn prepare(&mut self, _utterance: &Utterance) -> Result<(), String> {
        Ok(())
    }

    fn discard_prepared(&mut self) {}
}

impl<T: VoiceEngine + ?Sized> VoiceEngine for Box<T> {
    fn respond(
        &mut self,
        utterance: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        (**self).respond(utterance, cancel, emit)
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

    fn interrupt_signal(&self) -> Option<Arc<dyn Fn() + Send + Sync>> {
        (**self).interrupt_signal()
    }

    fn prepare(&mut self, utterance: &Utterance) -> Result<(), String> {
        (**self).prepare(utterance)
    }

    fn discard_prepared(&mut self) {
        (**self).discard_prepared()
    }
}
