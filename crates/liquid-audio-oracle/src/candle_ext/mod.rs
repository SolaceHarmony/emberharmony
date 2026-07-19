//! Candle backports / extensions.
//!
//! This crate is pinned to candle **0.9.2** because the `moshi` crate (which
//! provides the Mimi codec we depend on) requires candle `^0.9.1`, and candle's
//! `Tensor`/`Device` types do not cross minor-version boundaries. Newer candle
//! (0.10.x) ships primitives that are *already written to extend candle* — exactly
//! what this port should reuse rather than re-implement. Rather than fork the whole
//! dependency tree off `moshi`, we **vendor the specific missing pieces** here,
//! adapted to the 0.9.2 API (in practice: the import path only).
//!
//! Provenance is recorded per item. Everything here is upstream candle code
//! (MIT/Apache-2.0) or a thin extension built from candle's public ops, kept in
//! one place so it is trivial to drop once `moshi` moves to candle 0.10+.
//!
//! - [`kv_cache::ConcatKvCache`] — verbatim backport from `candle-nn` 0.10.2
//!   (`src/kv_cache.rs`); the cat-based KV cache that the LFM2 depthformer's
//!   `LayerKVCache` is a structural 1:1 of.
//! - [`loss::cross_entropy_none`] — `nn.functional.cross_entropy(reduction="none")`,
//!   the per-row form candle's mean-reducing [`candle_nn::loss::cross_entropy`] does
//!   not provide. Written in candle's `loss.rs` style.
//! - [`transformers_utils::build_causal_mask`] / [`transformers_utils::repeat_kv`] —
//!   verbatim backport from `candle-transformers` `src/utils.rs`; the exact two
//!   `crate::utils::*` helpers upstream `models/lfm2.rs` imports, so the backbone port
//!   uses the reference helpers rather than a hand-rolled substitute.
//! - [`tensor_ext::TensorExt::to_vec4`] — the rank-4 rung candle's `to_vecN` ladder
//!   stops short of (absent in 0.9.2 *and* 0.10.2); built from candle's public ops.

pub mod kv_cache;
pub mod loss;
pub mod tensor_ext;
pub mod transformers_utils;
