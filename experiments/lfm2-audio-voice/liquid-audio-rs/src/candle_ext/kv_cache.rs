//! `ConcatKvCache` — verbatim backport from `candle-nn` 0.10.2 (`src/kv_cache.rs`).
//!
//! The only change from upstream is the import path (`candle` → `candle_core`),
//! per this crate's 0.9.2 pin. Upstream is MIT/Apache-2.0. Keep this in sync with
//! candle so it can be deleted once we move to candle 0.10+.
//!
//! A concatenation-based KV-cache: each `append` is a `Tensor::cat` along the
//! sequence dimension, growing the cache by exactly the new step (no pre-allocated
//! `max_seq_len` buffer like 0.9.2's [`candle_nn::kv_cache::KvCache`]). This is the
//! exact shape of the Python `liquid_audio.model.transformer.LayerKVCache.update`
//! (`torch.cat([key_cache, k], dim=1)`), which is why the depthformer's
//! [`LayerKvCache`](crate::model::transformer::LayerKvCache) wraps it.

use candle_core::{Result, Tensor};

/// Concatenation-based KV-cache that grows by `Tensor::cat` on each append.
///
/// **Recommended for:**
/// - GPU inference (CUDA, Metal)
/// - Autoregressive generation (token-by-token decoding)
///
/// Use 0.9.2's `KvCache` instead for fixed up-front allocation / CPU-only.
#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    /// Create a new empty concatenation-based KV-cache.
    ///
    /// `dim` is the dimension along which to concatenate:
    /// - attention shaped `[batch, heads, seq, head_dim]` → `dim = 2`
    /// - attention shaped `[batch, seq, heads, head_dim]` → `dim = 1`
    pub fn new(dim: usize) -> Self {
        Self { k: None, v: None, dim }
    }

    /// Current sequence length in the cache (0 if empty).
    pub fn current_seq_len(&self) -> usize {
        self.k.as_ref().and_then(|k| k.dims().get(self.dim).copied()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    /// The concatenation dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append key/value tensors, returning the full `(k, v)` including the new step.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Ensure inputs are contiguous for optimal concatenation performance.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => Tensor::cat(&[k_cache, &k], self.dim)?,
        });
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
        });
        Ok((self.k.as_ref().unwrap().clone(), self.v.as_ref().unwrap().clone()))
    }

    /// Reset the cache (clear all stored keys and values).
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Reference to the current K cache data (`None` if empty).
    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// Reference to the current V cache data (`None` if empty).
    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }

    /// Mutable reference to the current K cache data (`None` if empty).
    pub fn k_mut(&mut self) -> Option<&mut Tensor> {
        self.k.as_mut()
    }

    /// Mutable reference to the current V cache data (`None` if empty).
    pub fn v_mut(&mut self) -> Option<&mut Tensor> {
        self.v.as_mut()
    }

    /// Consume the cache, returning the owned `(k, v)` (`None` if empty).
    pub fn into_inner(self) -> Option<(Tensor, Tensor)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}
