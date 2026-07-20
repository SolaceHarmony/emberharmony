//! Training-only audio-code encoder.
//!
//! Production Mimi encode/decode is native. The oracle retains the upstream
//! Candle Mimi encoder solely to prepare teacher-forced training examples. It
//! is constructed from a component-qualified view of the native resident image;
//! Rust never opens or parses the codec checkpoint.

use std::sync::Mutex;

use candle_core::{Result, Tensor};

pub trait AudioEncoder: Send + Sync {
    fn encode(&self, wav: &Tensor) -> Result<Tensor>;

    fn sample_rate(&self) -> u32 {
        24_000
    }
}

pub struct MimiEncoder {
    inner: Mutex<::moshi::mimi::Mimi>,
}

impl MimiEncoder {
    pub fn new(mimi: ::moshi::mimi::Mimi) -> Self {
        Self {
            inner: Mutex::new(mimi),
        }
    }
}

impl AudioEncoder for MimiEncoder {
    fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let mut mimi = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        mimi.reset_state();
        mimi.encode(wav)
    }
}
