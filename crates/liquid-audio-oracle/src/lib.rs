//! Workspace-only training and mathematical reference implementation.
//!
//! Production inference is owned by native Flashkern. This crate contains no
//! alternate runtime, generation loop, native parity transport, or checkpoint
//! parser. Checkpoints enter only through the opaque native resident image.

pub mod audio_out;
pub mod candle_ext;
pub mod chat_template;
pub mod data;
pub mod loader;
pub mod model;
pub mod processor;
pub mod resample;
#[path = "compute/threads.rs"]
pub mod threads;
pub mod trainer;
pub mod utils;
#[path = "compute/weights.rs"]
mod weights;

pub use loader::{from_pretrained_trainable, TrainableLoad};
pub use model::lfm2_audio::{LFM2AudioModel, LFM2AudioModelInput, LFM2AudioModelOutput};
pub use processor::LFM2AudioProcessor;
pub use trainer::{Trainer, TrainerConfig};
