//! Port of `liquid_audio/model/conformer/` — NVIDIA NeMo FastConformer encoder.
//!
//! **Scope:** the full module set, ported 1:1 (class-for-class, fn-for-fn). The
//! numerical *inference forward path* is parity-verified against Python; the
//! training / streaming / deployment scaffolding (cache-aware streaming,
//! `conv_split_by_*` chunking, `forward_for_export`, autocast guards,
//! stochastic-depth, the `use_pytorch_sdpa` branch) is cold on the offline path
//! and ported for inventory completeness — marked `// PORT:` where a member has
//! no candle referent (torch autocast / pickle serialization / autograd).
//!
//! See PORT_STATUS.md for the per-file class/fn mapping.

pub mod encoder; // ConformerEncoder
pub mod mha;
pub mod modules; // ConformerLayer / ConformerConvolution / FeedForward / CausalConv1D
pub mod subsampling; // ConvSubsampling / MaskedConvSequential
pub mod utils; // CacheAwareStreamingConfig, stochastic-depth, autocast guard
