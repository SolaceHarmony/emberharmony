//! candle `CustomOp` bridges for the flashkern kernels — the seam the model wires through.
//!
//! Mirrors the contract of `candle_flashfftconv::causal_conv1d_update_fused` exactly (same
//! shapes, same `[y | new_state]` combined output split with zero-copy narrows), so
//! `ShortConv::forward` can pick per device: this op on CPU when the SIMD kernel is built
//! and supported, the JIT Metal kernel otherwise. No silent fallback inside the op — if the
//! kernel is unavailable, `cpu_fwd` errors loudly; the call site gates on
//! [`conv1d_update_available`] first (same pattern as `bf16_gemm::bf16_gemm_available`).

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

/// `true` when the flashkern fused conv1d update kernel is built in and this CPU can run it:
/// baseline NEON on aarch64 (no FEAT gate — the bf16 rounding uses integer RNE), AVX2+FMA on
/// x86-64. The decode wiring in `lfm2_hf.rs` checks this before choosing the CPU op.
pub fn conv1d_update_available() -> bool {
    #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
    {
        true
    }
    #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
    {
        let f = super::x86::x86_features();
        f.avx2 && f.fma
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", has_flashkern_neon),
        all(target_arch = "x86_64", has_flashkern_x86)
    )))]
    {
        false
    }
}

/// Register-window bound shared with the GPU kernel (`conv1d_update.rs` MAX_K) and the
/// flashkern C kernels.
const MAX_K: usize = 8;

struct FlashkernConv1dUpdate;

fn dims(ul: &Layout, sl: &Layout, wl: &Layout) -> Result<(usize, usize, usize, usize)> {
    let (b, d3, t) = ul.shape().dims3()?;
    let (sb, sd, skm1) = sl.shape().dims3()?;
    let (wd, k) = wl.shape().dims2()?;
    let d = d3 / 3;
    if d3 != 3 * d || sb != b || sd != d || wd != d || skm1 + 1 != k {
        candle_core::bail!(
            "flashkern conv1d_update: inconsistent dims bcx {:?} state {:?} w {:?}",
            ul.shape(),
            sl.shape(),
            wl.shape()
        );
    }
    if k > MAX_K {
        candle_core::bail!("flashkern conv1d_update: K={k} exceeds register window {MAX_K}");
    }
    Ok((b, d, t, k))
}

fn contig<'a, T: candle_core::WithDType>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [T]> {
    let data = s.as_slice::<T>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("flashkern conv1d_update expects contiguous inputs"),
    }
}

impl CustomOp3 for FlashkernConv1dUpdate {
    fn name(&self) -> &'static str {
        "flashkern_conv1d_update"
    }

    #[allow(unused_variables)] // off-kernel builds only reach the bail! arms
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
        if !conv1d_update_available() {
            // Not a fallback: the ShortConv call site must gate on conv1d_update_available().
            candle_core::bail!("flashkern conv1d_update kernel not built/supported on this CPU");
        }
        let out_len = b * d * (t + k - 1);
        let shape = Shape::from((b, d, t + k - 1));
        match us {
            CpuStorage::F32(_) => {
                let bcx = contig::<f32>(us, ul)?;
                let state = contig::<f32>(ss, sl)?;
                let w = contig::<f32>(ws, wl)?;
                #[allow(unused_mut)]
                let mut out = vec![0f32; out_len];
                #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
                super::neon::conv1d_update_f32(bcx, state, w, &mut out, b, d, t, k);
                #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
                super::x86::conv1d_update_f32(bcx, state, w, &mut out, b, d, t, k);
                Ok((CpuStorage::F32(out), shape))
            }
            CpuStorage::BF16(_) => {
                // half::bf16 is repr(transparent) over u16, so the bit-slice views are sound
                // (same trick as Bf16Gemm::cpu_fwd).
                let bcx = contig::<half::bf16>(us, ul)?;
                let state = contig::<half::bf16>(ss, sl)?;
                let w = contig::<half::bf16>(ws, wl)?;
                let bcx_b =
                    unsafe { std::slice::from_raw_parts(bcx.as_ptr() as *const u16, bcx.len()) };
                let state_b = unsafe {
                    std::slice::from_raw_parts(state.as_ptr() as *const u16, state.len())
                };
                let w_b = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u16, w.len()) };
                #[allow(unused_mut)]
                let mut out = vec![0u16; out_len];
                #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
                super::neon::conv1d_update_bf16(bcx_b, state_b, w_b, &mut out, b, d, t, k);
                #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
                super::x86::conv1d_update_bf16(bcx_b, state_b, w_b, &mut out, b, d, t, k);
                let out: Vec<half::bf16> = out.iter().map(|&v| half::bf16::from_bits(v)).collect();
                Ok((CpuStorage::BF16(out), shape))
            }
            _ => candle_core::bail!("flashkern conv1d_update: f32 or bf16 only"),
        }
    }
}

/// Fused LFM2 ShortConv decode step on the flashkern CPU kernel: `y = C ⊙ conv1d_causal(B ⊙ x,
/// w, state)` with the carried state advanced functionally — the CPU-device twin of
/// `candle_flashfftconv::causal_conv1d_update_fused` (identical shapes and split).
///
/// - `bcx` `[B, 3D, T]` — `in_proj` output in HF chunk order (B-gate | C-gate | x).
/// - `state` `[B, D, K-1]` — the last K-1 conv inputs (`Bx`) from prior steps.
/// - `w` `[D, K]` — depthwise taps.
///
/// Returns `(y [B,D,T], new_state [B,D,K-1])`, zero-copy views of one kernel output.
/// **Precondition:** [`conv1d_update_available`] (the op errors rather than degrading).
pub fn causal_conv1d_update_fused(
    bcx: &Tensor,
    state: &Tensor,
    w: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let bcx = bcx.contiguous()?;
    let state = state.contiguous()?;
    let w = w.contiguous()?;
    let (_, _, t) = bcx.dims3()?;
    let t = t.max(1);
    let combined = bcx.apply_op3(&state, &w, FlashkernConv1dUpdate)?;
    let y = combined.narrow(2, 0, t)?;
    let new_state = combined.narrow(2, t, combined.dim(2)? - t)?;
    Ok((y, new_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    // Cross-op parity: the flashkern CPU op must agree with the candle-flashfftconv op it
    // replaces on the CPU device — THE wiring contract. f32 tight (FMA-vs-not only), bf16
    // through the same trained-regime rounding points.
    #[test]
    fn matches_flashfftconv_op_on_cpu() {
        if !conv1d_update_available() {
            eprintln!("flashkern conv1d_update unavailable on this CPU — skipping");
            return;
        }
        let dev = Device::Cpu;
        for (b, d, t, k) in [
            (1usize, 8usize, 1usize, 3usize),
            (2, 5, 4, 4),
            (1, 2048, 1, 3),
        ] {
            let bcx: Vec<f32> = (0..b * 3 * d * t)
                .map(|i| (i as f32 * 0.13).sin())
                .collect();
            let st: Vec<f32> = (0..b * d * (k - 1))
                .map(|i| (i as f32 * 0.07).cos())
                .collect();
            let wv: Vec<f32> = (0..d * k).map(|i| 0.1 + 0.002 * (i % 50) as f32).collect();
            for dtype in [DType::F32, DType::BF16] {
                let mk = |v: &[f32], shape: (usize, usize, usize)| {
                    Tensor::from_vec(v.to_vec(), shape, &dev)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap()
                };
                let bcxt = mk(&bcx, (b, 3 * d, t));
                let stt = mk(&st, (b, d, k - 1));
                let wt = Tensor::from_vec(wv.clone(), (d, k), &dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let (y0, s0) =
                    candle_flashfftconv::causal_conv1d_update_fused(&bcxt, &stt, &wt).unwrap();
                let (y1, s1) = causal_conv1d_update_fused(&bcxt, &stt, &wt).unwrap();
                let flat = |t: &Tensor| -> Vec<f32> {
                    t.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap()
                };
                let tol = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
                let dy = flat(&y0)
                    .iter()
                    .zip(flat(&y1))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let ds = flat(&s0)
                    .iter()
                    .zip(flat(&s1))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(dy <= tol, "{dtype:?} (b{b} d{d} t{t} k{k}) y diff {dy}");
                // The carried state is a plain gated product on both sides — exact.
                assert_eq!(ds, 0.0, "{dtype:?} (b{b} d{d} t{t} k{k}) state diff {ds}");
            }
        }
    }
}
