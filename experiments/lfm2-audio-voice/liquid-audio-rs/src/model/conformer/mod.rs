//! Port of `liquid_audio/model/conformer/` — NVIDIA NeMo FastConformer encoder.
//!
//! **Scope:** the numerical *inference forward path* only. The training /
//! streaming / deployment scaffolding is intentionally not ported because it is
//! never executed for offline encode:
//! - `conformer/utils.py` (autocast helpers, `CacheAwareStreamingConfig`,
//!   stochastic-depth drop probs) — training/streaming only.
//! - cache-aware streaming, `conv_split_by_*` chunking, `forward_for_export`,
//!   `change_attention_model`, export input/output name hooks.
//! - the `use_pytorch_sdpa` branch (we port the equivalent manual attention).
//!
//! See PORT_STATUS.md for the per-file mapping.

pub mod mha;
pub mod modules; // ConformerLayer / ConformerConvolution / FeedForward / CausalConv1D
pub mod encoder; // ConformerEncoder
pub mod subsampling; // ConvSubsampling
// pub mod processor;    // AudioToMelSpectrogramPreprocessor (mel features)
