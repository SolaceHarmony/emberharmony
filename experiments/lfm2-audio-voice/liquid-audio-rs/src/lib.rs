//! Faithful Rust port of Liquid AI's `liquid_audio` (LFM2.5-Audio).
//!
//! Mirrors the Python package `src/liquid_audio/` module-for-module. Re-exports
//! follow `liquid_audio/__init__.py`; entries are uncommented as each module is
//! ported (see PORT_STATUS.md).
//!
//! ```python
//! from liquid_audio.detokenizer import LFM2AudioDetokenizer
//! from liquid_audio.model.lfm2_audio import LFM2AudioModel
//! from liquid_audio.processor import ChatState, LFM2AudioProcessor
//! from liquid_audio.utils import LFMModality
//! ```

pub mod audio_out; // AudioDetokenizer trait + backends (LFM2 detok / Mimi)
pub mod bf16_gemm; // NEON BFMMLA bf16 CPU matmul (closes candle's bf16 gemm gap)
pub mod candle_ext; // vendored candle 0.10 backports + extensions (kept on the 0.9.2 pin)
pub mod data; // data/ (data-pipeline value types)
pub mod detokenizer; // detokenizer.py
pub mod loader; // config.json + safetensors → model + processor
pub mod model;
pub mod processor; // processor.py
pub mod realtime; // multi-threaded worker pipeline (chat.py producer/consumer threading)
pub mod resample; // torchaudio.functional.resample (windowed-sinc) port
pub mod threads; // intra-op thread-pool parity with torch (at::intraop_default_num_threads)
pub mod trainer; // trainer.py
pub mod utils;

pub use audio_out::{AudioDetokenizer, MimiDetokenizer};
pub use threads::{configure_intraop_threads, intraop_default_num_threads};
pub use detokenizer::LFM2AudioDetokenizer;
pub use loader::{from_pretrained, from_pretrained_hub};
pub use model::lfm2_audio::{GenParams, GenToken, LFM2AudioModel};
pub use processor::{ChatState, LFM2AudioProcessor};
pub use trainer::{Trainer, TrainerConfig};
pub use utils::{get_model_dir, LFMModality};
// pub use model::lfm2_audio::LFM2AudioModel;
// pub use processor::{ChatState, LFM2AudioProcessor};
