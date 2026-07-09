//! Backports from `candle-transformers` `src/utils.rs` (candle 0.10.x / `main`).
//!
//! candle 0.9.2 ships no full-precision `lfm2` model, so our backbone is a port of
//! upstream `candle-transformers/src/models/lfm2.rs`. These are the exact two
//! `crate::utils::*` helpers that file imports (`repeat_kv`, `build_causal_mask`).
//! Vendored verbatim (MIT/Apache-2.0), adapted only for the 0.9.2 API surface
//! (`candle` → `candle_core`), so the port uses the **same** helpers as the reference
//! rather than a hand-rolled substitute — kept here so it is trivial to drop once
//! `moshi` (and thus our candle pin) moves to 0.10+.

use candle_core::{Device, Result, Tensor};

/// `candle_transformers::utils::build_causal_mask` — the boolean (`u8`) causal mask used
/// by `lfm2.rs`'s `Cache::mask`: `mask[i][j] = 1` (i.e. *mask out*) where the key column
/// `j` is in the future of query row `i`, `j > index_pos + i`. Shape `(seq_len, kv_len)`
/// with `kv_len = index_pos + seq_len`. Pair with [`Tensor::where_cond`] / `masked_fill`
/// to set those positions to `-inf` before the softmax.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

/// `candle_transformers::utils::repeat_kv` — GQA head expansion. Repeats each of the
/// `n_kv` key/value heads `n_rep` times into **consecutive** output heads, so query-head
/// group `g` shares KV head `g / n_rep`. Uses `cat` rather than `unsqueeze`+`expand`+
/// `reshape` to avoid a potentially strided copy (huggingface/candle#2043); the output is
/// element-for-element identical to the expand form.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy. https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_marks_future_positions() {
        let dev = Device::Cpu;
        // Prefill: seq_len=3, index_pos=0 ⇒ kv_len=3. Row i may attend to j ≤ i; the mask
        // (1 = blocked) is the strict upper triangle.
        let m = build_causal_mask(3, 0, &dev).unwrap();
        assert_eq!(m.dims(), &[3, 3]);
        assert_eq!(
            m.to_vec2::<u8>().unwrap(),
            vec![vec![0, 1, 1], vec![0, 0, 1], vec![0, 0, 0]]
        );

        // Decode step: seq_len=1 at index_pos=3 ⇒ kv_len=4. The single new query attends to
        // all four cached keys (nothing is in its future).
        let m2 = build_causal_mask(1, 3, &dev).unwrap();
        assert_eq!(m2.dims(), &[1, 4]);
        assert_eq!(m2.to_vec2::<u8>().unwrap(), vec![vec![0, 0, 0, 0]]);
    }

    #[test]
    fn repeat_kv_matches_expand_reference_and_identity() {
        let dev = Device::Cpu;
        let (b, n_kv, t, d) = (1usize, 2usize, 4usize, 3usize);
        let n = (b * n_kv * t * d) as f32;
        let xs = Tensor::arange(0f32, n, &dev)
            .unwrap()
            .reshape((b, n_kv, t, d))
            .unwrap();

        use crate::candle_ext::tensor_ext::TensorExt;

        // n_rep == 1 is the identity (same values) — compared as nested rank-4 vecs.
        let same = repeat_kv(xs.clone(), 1).unwrap();
        assert_eq!(same.to_vec4::<f32>().unwrap(), xs.to_vec4::<f32>().unwrap());

        // n_rep == 3 (cat form) must equal the unsqueeze+expand+reshape reference exactly.
        let got = repeat_kv(xs.clone(), 3).unwrap();
        let want = xs
            .unsqueeze(2)
            .unwrap()
            .expand((b, n_kv, 3, t, d))
            .unwrap()
            .reshape((b, n_kv * 3, t, d))
            .unwrap();
        assert_eq!(got.dims(), &[b, n_kv * 3, t, d]);
        assert_eq!(
            got.to_vec4::<f32>().unwrap(),
            want.to_vec4::<f32>().unwrap(),
            "cat-based repeat_kv diverged from the expand reference"
        );
    }
}
