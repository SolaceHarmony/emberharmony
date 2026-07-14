//! Fused causal depthwise conv1d UPDATE — the decode-step kernel, one dispatch.
//!
//! What the LFM2 authors run on CUDA is Tri Dao's `causal_conv1d_update`
//! (Dao-AILab/causal-conv1d): one thread per channel, the K-tap window held in
//! registers, the conv state advanced in place, f32 accumulate with one low-precision
//! rounding at store. LFM2's ShortConv block then wraps that kernel in two SEPARATE
//! elementwise ops: `Bx = B ⊙ x` before and `y = C ⊙ conv` after (HF
//! `modeling_lfm2.py::Lfm2ShortConv.cuda_kernels_forward`).
//!
//! This op fuses all three: `(B ⊙ x) → causal K-tap conv over carried state → C ⊙`
//! in ONE dispatch, register window, no `[state|x]` concat staging, no gate
//! intermediates. Accumulation is ascending-tap f32 with the multiply-adds left
//! contractible (CUDA's `acc += w*x` compiles to FMA — LFM2 was trained in that
//! regime, so contractible IS the faithful choice here; the strict-order
//! [`crate::depthwise3_causal`] remains the bit-exactness instrument).
//!
//! Functional-style state (candle ops can't mutate inputs): the kernel writes
//! `[out | new_state]` into one buffer and the host splits it with zero-copy
//! narrows. `new_state` carries the last K-1 CONV INPUTS (`Bx`), matching both
//! the HF cache contract and [`crate::depthwise_conv1d_stream`]'s cache.
//!
//! f32 and bf16 (compute always f32; bf16 storage rounds once at store, the
//! trained-around regime).

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// Maximum K the register window supports (LFM2 uses K=3; Mamba-family ≤ 4).
const MAX_K: usize = 8;

#[cfg(feature = "metal")]
fn kernel_source(ty: &str, from_f32: &str, to_f32: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

struct UpdateParams {{
    uint B; uint D; uint T; uint K;
}};

kernel void causal_conv1d_update_fused_{ty}(
    constant UpdateParams& p [[buffer(0)]],
    device const {ty}*  bcx   [[buffer(1)]], // [B, 3D, T] — rows: B-gate | C-gate | x (HF chunk order)
    device const {ty}*  state [[buffer(2)]], // [B, D, K-1] carried conv inputs (Bx)
    device const {ty}*  w     [[buffer(3)]], // [D, K] depthwise taps
    device {ty}*        out   [[buffer(4)]], // [B, D, T + K-1] = [y | new_state]
    uint gid [[thread_position_in_grid]]
) {{
    uint B = p.B, D = p.D, T = p.T, K = p.K;
    if (gid >= B * D) return;
    uint b = gid / D, c = gid % D;

    device const {ty}* brow = bcx + ((b * 3u + 0u) * D + c) * T;
    device const {ty}* crow = bcx + ((b * 3u + 1u) * D + c) * T;
    device const {ty}* xrow = bcx + ((b * 3u + 2u) * D + c) * T;
    device const {ty}* srow = state + (b * D + c) * (K - 1u);
    device {ty}*       orow = out + (b * D + c) * (T + K - 1u);

    float wv[{max_k}];
    for (uint j = 0u; j < K; ++j) {{ wv[j] = {to_f32}(w[c * K + j]); }}

    float win[{max_k}];
    for (uint j = 0u; j + 1u < K; ++j) {{ win[j] = {to_f32}(srow[j]); }}

    for (uint t = 0u; t < T; ++t) {{
        // Round Bx through the storage dtype before the conv reads it — torch
        // materializes B*x as a {ty} tensor, so the trained regime includes this
        // rounding (no-op for float).
        float bx = {to_f32}(({ty})({to_f32}(brow[t]) * {to_f32}(xrow[t])));
        win[K - 1u] = bx;
        float acc = 0.0f;
        for (uint j = 0u; j < K; ++j) {{ acc += wv[j] * win[j]; }}
        // Round the conv output through the storage dtype before the C gate —
        // the CUDA update kernel stores conv_out and torch's C-multiply reads
        // the rounded tensor (no-op for float).
        float conv_val = {to_f32}(({ty})acc);
        orow[t] = {from_f32}({to_f32}(crow[t]) * conv_val);
        for (uint j = 0u; j + 1u < K; ++j) {{ win[j] = win[j + 1u]; }}
    }}
    for (uint j = 0u; j + 1u < K; ++j) {{ orow[T + j] = {from_f32}(win[j]); }}
}}
"#,
        ty = ty,
        from_f32 = from_f32,
        to_f32 = to_f32,
        max_k = MAX_K,
    )
}

struct CausalConv1dUpdateFused;

fn dims(ul: &Layout, sl: &Layout, wl: &Layout) -> Result<(usize, usize, usize, usize)> {
    let (b, d3, t) = ul.shape().dims3()?;
    let (sb, sd, skm1) = sl.shape().dims3()?;
    let (wd, k) = wl.shape().dims2()?;
    let d = d3 / 3;
    if d3 != 3 * d || sb != b || sd != d || wd != d || skm1 + 1 != k {
        candle_core::bail!(
            "conv1d_update: inconsistent dims bcx {:?} state {:?} w {:?}",
            ul.shape(),
            sl.shape(),
            wl.shape()
        );
    }
    if k > MAX_K {
        candle_core::bail!("conv1d_update: K={k} exceeds register window {MAX_K}");
    }
    Ok((b, d, t, k))
}

/// Shared scalar reference over f32 slices (both dtypes route through it).
#[allow(clippy::too_many_arguments)]
fn cpu_ref(
    bcx: &[f32],
    state: &[f32],
    w: &[f32],
    b: usize,
    d: usize,
    t_len: usize,
    k: usize,
    out: &mut [f32],
) {
    let km1 = k - 1;
    for bi in 0..b {
        for c in 0..d {
            let brow = &bcx[((bi * 3) * d + c) * t_len..][..t_len];
            let crow = &bcx[((bi * 3 + 1) * d + c) * t_len..][..t_len];
            let xrow = &bcx[((bi * 3 + 2) * d + c) * t_len..][..t_len];
            let srow = &state[(bi * d + c) * km1..][..km1];
            let orow = &mut out[(bi * d + c) * (t_len + km1)..][..t_len + km1];
            let mut win = [0f32; MAX_K];
            win[..km1].copy_from_slice(srow);
            for t in 0..t_len {
                win[k - 1] = brow[t] * xrow[t];
                let mut acc = 0f32;
                for j in 0..k {
                    acc += w[c * k + j] * win[j];
                }
                orow[t] = crow[t] * acc;
                for j in 0..km1 {
                    win[j] = win[j + 1];
                }
            }
            orow[t_len..].copy_from_slice(&win[..km1]);
        }
    }
}

/// bf16-regime reference: identical to [`cpu_ref`] but Bx rounds through bf16
/// before entering the window, matching the torch-materialized B*x tensor.
#[allow(clippy::too_many_arguments)]
fn cpu_ref_bf16_bx(
    bcx: &[f32],
    state: &[f32],
    w: &[f32],
    b: usize,
    d: usize,
    t_len: usize,
    k: usize,
    out: &mut [f32],
) {
    let km1 = k - 1;
    for bi in 0..b {
        for c in 0..d {
            let brow = &bcx[((bi * 3) * d + c) * t_len..][..t_len];
            let crow = &bcx[((bi * 3 + 1) * d + c) * t_len..][..t_len];
            let xrow = &bcx[((bi * 3 + 2) * d + c) * t_len..][..t_len];
            let srow = &state[(bi * d + c) * km1..][..km1];
            let orow = &mut out[(bi * d + c) * (t_len + km1)..][..t_len + km1];
            let mut win = [0f32; MAX_K];
            win[..km1].copy_from_slice(srow);
            for t in 0..t_len {
                win[k - 1] = half::bf16::from_f32(brow[t] * xrow[t]).to_f32();
                let mut acc = 0f32;
                for j in 0..k {
                    acc += w[c * k + j] * win[j];
                }
                orow[t] = crow[t] * half::bf16::from_f32(acc).to_f32();
                for j in 0..km1 {
                    win[j] = win[j + 1];
                }
            }
            orow[t_len..].copy_from_slice(&win[..km1]);
        }
    }
}

fn contig<'a, T: candle_core::WithDType>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [T]> {
    let data = s.as_slice::<T>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("conv1d_update expects contiguous inputs"),
    }
}

impl CustomOp3 for CausalConv1dUpdateFused {
    fn name(&self) -> &'static str {
        "causal_conv1d_update_fused"
    }

    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ss: &CpuStorage,
        sl: &Layout,
        ws: &CpuStorage,
        wl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, d, t, k) = dims(ul, sl, wl)?;
        let out_len = b * d * (t + k - 1);
        match us {
            CpuStorage::F32(_) => {
                let bcx = contig::<f32>(us, ul)?;
                let state = contig::<f32>(ss, sl)?;
                let w = contig::<f32>(ws, wl)?;
                let mut out = vec![0f32; out_len];
                cpu_ref(bcx, state, w, b, d, t, k, &mut out);
                Ok((CpuStorage::F32(out), Shape::from((b, d, t + k - 1))))
            }
            CpuStorage::BF16(_) => {
                // Compute f32, store bf16 — and round Bx through bf16 before the conv
                // reads it (torch materializes B*x as a bf16 tensor): trained regime.
                let bcx: Vec<f32> = contig::<half::bf16>(us, ul)?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let state: Vec<f32> = contig::<half::bf16>(ss, sl)?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let w: Vec<f32> = contig::<half::bf16>(ws, wl)?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let mut out = vec![0f32; out_len];
                cpu_ref_bf16_bx(&bcx, &state, &w, b, d, t, k, &mut out);
                let out: Vec<half::bf16> = out.iter().map(|&v| half::bf16::from_f32(v)).collect();
                Ok((CpuStorage::BF16(out), Shape::from((b, d, t + k - 1))))
            }
            _ => candle_core::bail!("conv1d_update: f32 or bf16 only"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ss: &candle_core::MetalStorage,
        sl: &Layout,
        ws: &candle_core::MetalStorage,
        wl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::MTLSize;

        let (b, d, t, k) = dims(ul, sl, wl)?;
        let dtype = us.dtype();
        let (name, ty) = match dtype {
            DType::F32 => ("causal_conv1d_update_fused_float", "float"),
            DType::BF16 => ("causal_conv1d_update_fused_bfloat", "bfloat"),
            other => candle_core::bail!("conv1d_update: unsupported dtype {other:?}"),
        };
        let src = match dtype {
            DType::F32 => kernel_source("float", "float", "float"),
            _ => kernel_source("bfloat", "bfloat", "float"),
        };
        let dev = us.device();
        let p = crate::metal_util::pipeline(dev, name, &src)?;
        let out_el = b * d * (t + k - 1);
        let out = dev.new_buffer(out_el, dtype, "conv1d_update")?;

        #[repr(C)]
        struct UpdateParams {
            b: u32,
            d: u32,
            t: u32,
            k: u32,
        }
        let params = UpdateParams {
            b: b as u32,
            d: d as u32,
            t: t as u32,
            k: k as u32,
        };
        let dts = dtype.size_in_bytes();
        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), ul.start_offset() * dts);
        enc.set_buffer(2, Some(ss.buffer()), sl.start_offset() * dts);
        enc.set_buffer(3, Some(ws.buffer()), wl.start_offset() * dts);
        enc.set_buffer(4, Some(&*out), 0);
        let total = b * d;
        let tg = 256usize.min(total.max(1));
        enc.dispatch_threads(
            MTLSize {
                width: total,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), out_el, dtype),
            Shape::from((b, d, t + k - 1)),
        ))
    }
}

/// Fused LFM2 ShortConv decode step: `y = C ⊙ conv1d_causal(B ⊙ x, w, state)` in one
/// dispatch, with the carried state advanced functionally.
///
/// - `bcx` `[B, 3D, T]` — the `in_proj` output in HF chunk order (B-gate | C-gate | x).
/// - `state` `[B, D, K-1]` — the last K-1 conv inputs (`Bx`) from prior steps
///   (zeros at sequence start).
/// - `w` `[D, K]` — depthwise taps.
///
/// Returns `(y [B,D,T], new_state [B,D,K-1])` — both zero-copy views of one kernel
/// output. Intended for small `T` (decode / short continuation); use
/// [`crate::depthwise_conv1d_stream`] + explicit gates for long prefill.
pub fn causal_conv1d_update_fused(
    bcx: &Tensor,
    state: &Tensor,
    w: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if state.dtype() != bcx.dtype() || w.dtype() != bcx.dtype() {
        candle_core::bail!(
            "conv1d_update: dtype mismatch bcx {:?} state {:?} w {:?}",
            bcx.dtype(),
            state.dtype(),
            w.dtype()
        );
    }
    let bcx = bcx.contiguous()?;
    let state = state.contiguous()?;
    let w = w.contiguous()?;
    let (_, _, t) = bcx.dims3()?;
    let t = t.max(1);
    let combined = bcx.apply_op3(&state, &w, CausalConv1dUpdateFused)?;
    let y = combined.narrow(2, 0, t)?;
    let new_state = combined.narrow(2, t, combined.dim(2)? - t)?;
    Ok((y, new_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depthwise_conv1d_stream;
    use candle_core::{DType, Device, Tensor};

    fn composed_reference(
        bcx: &Tensor,
        state: &Tensor,
        w: &Tensor,
        d: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // The existing verified path: explicit gates + depthwise_conv1d_stream.
        let bgate = bcx.narrow(1, 0, d).unwrap();
        let cgate = bcx.narrow(1, d, d).unwrap();
        let x = bcx.narrow(1, 2 * d, d).unwrap();
        let bx = bgate.mul(&x).unwrap().contiguous().unwrap();
        let (conv, new_state) = depthwise_conv1d_stream(&bx, w, Some(state)).unwrap();
        let y = cgate.mul(&conv).unwrap();
        (
            y.flatten_all().unwrap().to_vec1().unwrap(),
            new_state.flatten_all().unwrap().to_vec1().unwrap(),
        )
    }

    #[test]
    fn fused_update_matches_composed_path() {
        let dev = Device::Cpu;
        for (b, d, t, k) in [(1usize, 8usize, 1usize, 3usize), (2, 5, 4, 4), (1, 3, 2, 3)] {
            let bcx: Vec<f32> = (0..b * 3 * d * t)
                .map(|i| (i as f32 * 0.13).sin())
                .collect();
            let st: Vec<f32> = (0..b * d * (k - 1))
                .map(|i| (i as f32 * 0.07).cos())
                .collect();
            let wv: Vec<f32> = (0..d * k).map(|i| 0.1 + 0.02 * i as f32).collect();
            let bcxt = Tensor::from_vec(bcx, (b, 3 * d, t), &dev).unwrap();
            let stt = Tensor::from_vec(st, (b, d, k - 1), &dev).unwrap();
            let wt = Tensor::from_vec(wv, (d, k), &dev).unwrap();

            let (y, ns) = causal_conv1d_update_fused(&bcxt, &stt, &wt).unwrap();
            let (ry, rns) = composed_reference(&bcxt, &stt, &wt, d);
            let fy: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
            let fns: Vec<f32> = ns.flatten_all().unwrap().to_vec1().unwrap();

            let dy = fy
                .iter()
                .zip(&ry)
                .map(|(a, r)| (a - r).abs())
                .fold(0f32, f32::max);
            let ds = fns
                .iter()
                .zip(&rns)
                .map(|(a, r)| (a - r).abs())
                .fold(0f32, f32::max);
            assert!(dy < 1e-5, "(b{b} d{d} t{t} k{k}) y diff {dy}");
            assert!(ds < 1e-6, "(b{b} d{d} t{t} k{k}) state diff {ds}");
        }
    }

    #[test]
    fn fused_update_rejects_mixed_dtypes_before_dispatch() {
        let dev = Device::Cpu;
        let bcx = Tensor::zeros((1, 12, 1), DType::F32, &dev).unwrap();
        let state = Tensor::zeros((1, 4, 2), DType::BF16, &dev).unwrap();
        let w = Tensor::zeros((4, 3), DType::F32, &dev).unwrap();
        let err = causal_conv1d_update_fused(&bcx, &state, &w)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dtype mismatch"), "{err}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_update_metal_matches_cpu_f32_and_bf16() {
        let cpu = Device::Cpu;
        let met = Device::new_metal(0).unwrap();
        let (b, d, t, k) = (2usize, 64usize, 1usize, 3usize);
        let bcx: Vec<f32> = (0..b * 3 * d * t)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let st: Vec<f32> = (0..b * d * (k - 1))
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let wv: Vec<f32> = (0..d * k).map(|i| 0.05 + 0.01 * i as f32).collect();

        let run = |dev: &Device, dtype: candle_core::DType| -> (Vec<f32>, Vec<f32>) {
            let bcxt = Tensor::from_vec(bcx.clone(), (b, 3 * d, t), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let stt = Tensor::from_vec(st.clone(), (b, d, k - 1), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let wt = Tensor::from_vec(wv.clone(), (d, k), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let (y, ns) = causal_conv1d_update_fused(&bcxt, &stt, &wt).unwrap();
            (
                y.to_dtype(candle_core::DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap(),
                ns.to_dtype(candle_core::DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap(),
            )
        };

        for dtype in [candle_core::DType::F32, candle_core::DType::BF16] {
            let (cy, cs) = run(&cpu, dtype);
            let (my, ms) = run(&met, dtype);
            let tol = if dtype == candle_core::DType::F32 {
                1e-6
            } else {
                1e-2
            };
            let dy = cy
                .iter()
                .zip(&my)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let ds = cs
                .iter()
                .zip(&ms)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(dy <= tol, "{dtype:?} y metal vs cpu: {dy}");
            assert!(ds <= tol, "{dtype:?} state metal vs cpu: {ds}");
        }
    }
}
