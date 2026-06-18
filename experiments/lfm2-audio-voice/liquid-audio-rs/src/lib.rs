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

pub mod model;
pub mod utils;
// pub mod detokenizer;   // detokenizer.py
// pub mod processor;     // processor.py

pub use utils::LFMModality;
// pub use detokenizer::LFM2AudioDetokenizer;
// pub use model::lfm2_audio::LFM2AudioModel;
// pub use processor::{ChatState, LFM2AudioProcessor};
