//! Workspace-only reference implementation for native LFM2 development.
//!
//! This package deliberately enables `liquid-audio/oracle`, which contains the
//! Candle model, Moshi compatibility engine, training pipeline, and fixture
//! capture code. Production applications depend on `liquid-audio` directly and
//! therefore cannot acquire these dependencies through a default feature.

pub use liquid_audio::*;
