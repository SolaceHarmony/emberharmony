//! Small `Tensor` extensions that extend candle's public API by one rank.
//!
//! `to_vec4` is the `numpy.ndarray.tolist()` / `torch.Tensor.tolist()` pattern at rank 4:
//! materialize an N-d tensor into nested `Vec`s by slicing the outermost dimension and
//! recursing. candle's `to_vec1`/`to_vec2`/`to_vec3` are themselves the Rust translation
//! of that pattern — just **capped at rank 3** (in 0.9.2 *and* 0.10.2), so there is no
//! upstream `to_vec4` to backport. The rest of the Rust tensor ecosystem has the same gap:
//! `ndarray` steers users to a flat `Vec` + `from_shape_vec`, and `tch` doesn't expose a
//! nested form either — so the crate fills it here.
//!
//! candle's `to_vecN` impls read private layout/storage internals (`self.layout`,
//! `strided_index`, `storage()`), so a verbatim extension is impossible from outside the
//! crate. Instead this builds the next rung from candle's **public** ops — slice dim 0,
//! drop the singleton, recurse into candle's own (already stride-correct)
//! [`Tensor::to_vec3`] — the canonical way to extend the ladder by one from the outside.
//! Same result a native `to_vec4` would give, including for non-contiguous layouts.

use candle_core::{Result, Tensor, WithDType};

pub trait TensorExt {
    /// Materialize a rank-4 tensor to nested `Vec`s — the `to_vec4` candle never shipped.
    /// Equivalent to slicing dim 0 and calling [`Tensor::to_vec3`] on each `(d1,d2,d3)`
    /// sub-tensor, so it honours non-contiguous layouts exactly as `to_vec3` does.
    fn to_vec4<S: WithDType>(&self) -> Result<Vec<Vec<Vec<Vec<S>>>>>;
}

impl TensorExt for Tensor {
    fn to_vec4<S: WithDType>(&self) -> Result<Vec<Vec<Vec<Vec<S>>>>> {
        let (d0, _, _, _) = self.dims4()?;
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            // narrow dim 0 to one slice, drop it → rank-3, then reuse candle's to_vec3.
            out.push(self.narrow(0, i, 1)?.squeeze(0)?.to_vec3::<S>()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn to_vec4_matches_index_arithmetic_contiguous_and_strided() {
        let dev = Device::Cpu;
        let (a, b, c, d) = (2usize, 3usize, 4usize, 5usize);
        let n = (a * b * c * d) as i64;
        let t = Tensor::arange(0i64, n, &dev)
            .unwrap()
            .reshape((a, b, c, d))
            .unwrap();

        let v = t.to_vec4::<i64>().unwrap();
        assert_eq!(v.len(), a);
        assert_eq!(v[0].len(), b);
        assert_eq!(v[0][0].len(), c);
        assert_eq!(v[0][0][0].len(), d);
        // arange ⇒ v[i][j][k][l] == ((i*b + j)*c + k)*d + l.
        assert_eq!(v[0][0][0][0], 0);
        assert_eq!(v[1][2][3][4], (((1 * b + 2) * c + 3) * d + 4) as i64);

        // Non-contiguous: transpose dims 1,2 → (a,c,b,d). to_vec4 must still read the
        // logical layout, so vv[i][k][j][l] == v[i][j][k][l].
        let tt = t.transpose(1, 2).unwrap();
        assert!(!tt.is_contiguous());
        let vv = tt.to_vec4::<i64>().unwrap();
        assert_eq!(vv.len(), a);
        assert_eq!(vv[0].len(), c);
        assert_eq!(vv[1][3][2][4], v[1][2][3][4]);
    }
}
