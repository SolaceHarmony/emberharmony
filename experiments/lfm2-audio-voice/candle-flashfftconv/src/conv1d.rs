//! Depthwise causal 1-D convolution — the FlashFFTConv short-filter path and the
//! LFM2 short-conv (`conv_L_cache`).
//!
//! `out[b, d, l] = bias[d] + Σ_k u[b, d, l − padding + k] · weight[d, k]`
//! (cross-correlation, one filter per channel; out-of-range taps contribute 0).
//! `L_out = L + 2·padding − K + 1`. For the causal short-conv, call with
//! `padding = K − 1` and narrow the result to the first `L` columns — identical to
//! `candle_nn::Conv1d` with `groups = channels`, which is what the parity suite
//! verifies against.
//!
//! The op carries a CPU reference (`cpu_fwd`) and a Metal kernel (`metal_fwd`);
//! `candle` dispatches to whichever matches the input tensor's device, so a single
//! [`depthwise_conv1d`] call runs the exact reference on CPU and the fused shader on
//! Metal.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};

/// Metal shader (f32). CUDA→Metal translation of `flashfftconv/conv1d/conv1d_bhl.cu`
/// (via the `mx.fast.metal_kernel` port in `csm-mlx/.../monarch_metal/conv1d_forward.py`):
/// the K=3 fast path and `fma` accumulation are preserved; the CUDA tiled grid is
/// generalized to one thread per output element so any `L_out` is covered. Buffers
/// are bound explicitly (`[[buffer(i)]]`) for candle's manual dispatch rather than
/// MLX's auto-generated signature.
#[cfg(feature = "metal")]
const SRC_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void depthwise_causal_conv1d_f32(
    device const float* u        [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint& B       [[buffer(4)]],
    constant uint& D       [[buffer(5)]],
    constant uint& L       [[buffer(6)]],
    constant uint& K       [[buffer(7)]],
    constant uint& padding [[buffer(8)]],
    constant uint& L_out   [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * D * L_out;
    if (gid >= total) { return; }
    uint l  = gid % L_out;
    uint bd = gid / L_out;
    uint d  = bd % D;
    uint b  = bd / D;
    uint u_base = b * D * L + d * L;
    uint w_base = d * K;
    float acc = bias[d];
    if (K == 3) {
        int idx = int(l) - int(padding);
        if (idx >= 0 && idx < int(L)) { acc = fma(u[u_base + uint(idx)], weights[w_base + 0], acc); }
        idx++;
        if (idx >= 0 && idx < int(L)) { acc = fma(u[u_base + uint(idx)], weights[w_base + 1], acc); }
        idx++;
        if (idx >= 0 && idx < int(L)) { acc = fma(u[u_base + uint(idx)], weights[w_base + 2], acc); }
    } else {
        for (uint k = 0; k < K; ++k) {
            int idx = int(l) - int(padding) + int(k);
            if (idx >= 0 && idx < int(L)) { acc = fma(u[u_base + uint(idx)], weights[w_base + k], acc); }
        }
    }
    out[b * D * L_out + d * L_out + l] = acc;
}
"#;

/// Metal shader (bf16): bf16 weights + input, **f32 accumulate, bf16 store** — the
/// deployed/trained dtype regime (the same f32-accumulate-bf16-store as FlashFFTConv's
/// `causal_conv1d` and this crate's faithful-bf16 path). Same indexing as the f32 kernel.
#[cfg(feature = "metal")]
const SRC_BF16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void depthwise_causal_conv1d_bf16(
    device const bfloat* u        [[buffer(0)]],
    device const bfloat* weights  [[buffer(1)]],
    device const bfloat* bias     [[buffer(2)]],
    device bfloat*       out      [[buffer(3)]],
    constant uint& B       [[buffer(4)]],
    constant uint& D       [[buffer(5)]],
    constant uint& L       [[buffer(6)]],
    constant uint& K       [[buffer(7)]],
    constant uint& padding [[buffer(8)]],
    constant uint& L_out   [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * D * L_out;
    if (gid >= total) { return; }
    uint l  = gid % L_out;
    uint bd = gid / L_out;
    uint d  = bd % D;
    uint b  = bd / D;
    uint u_base = b * D * L + d * L;
    uint w_base = d * K;
    float acc = float(bias[d]); // f32 accumulate
    if (K == 3) {
        int idx = int(l) - int(padding);
        if (idx >= 0 && idx < int(L)) { acc = fma(float(u[u_base + uint(idx)]), float(weights[w_base + 0]), acc); }
        idx++;
        if (idx >= 0 && idx < int(L)) { acc = fma(float(u[u_base + uint(idx)]), float(weights[w_base + 1]), acc); }
        idx++;
        if (idx >= 0 && idx < int(L)) { acc = fma(float(u[u_base + uint(idx)]), float(weights[w_base + 2]), acc); }
    } else {
        for (uint k = 0; k < K; ++k) {
            int idx = int(l) - int(padding) + int(k);
            if (idx >= 0 && idx < int(L)) { acc = fma(float(u[u_base + uint(idx)]), float(weights[w_base + k]), acc); }
        }
    }
    out[b * D * L_out + d * L_out + l] = bfloat(acc); // single bf16 store
}
"#;

/// Depthwise causal conv1d op. `padding` is applied symmetrically (like
/// `candle_nn::Conv1d`); the caller narrows for the causal slice.
pub struct DepthwiseCausalConv1d {
    pub padding: usize,
}

impl DepthwiseCausalConv1d {
    fn out_len(&self, l: usize, k: usize) -> Result<usize> {
        let lo = l as i64 + 2 * self.padding as i64 - k as i64 + 1;
        if lo <= 0 {
            candle_core::bail!("depthwise_conv1d: non-positive output length (L={l}, K={k}, padding={})", self.padding);
        }
        Ok(lo as usize)
    }
}

/// Borrow a contiguous f32 view of a CPU tensor, honoring its layout offset.
fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("depthwise_conv1d expects contiguous f32 inputs"),
    }
}

/// Read a contiguous bf16 CPU tensor up-converted to f32 (the accumulate dtype).
fn contig_bf16_to_f32(s: &CpuStorage, l: &Layout) -> Result<Vec<f32>> {
    let data = s.as_slice::<half::bf16>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(data[start..end].iter().map(|x| x.to_f32()).collect()),
        None => candle_core::bail!("depthwise_conv1d expects contiguous bf16 inputs"),
    }
}

/// The causal depthwise conv with **f32 accumulate** — shared by the f32 and bf16 CPU
/// paths (the bf16 path up-converts its inputs here and rounds the result back to bf16,
/// i.e. f32-accumulate, bf16-store).
#[allow(clippy::too_many_arguments)]
fn conv_cpu_f32(u: &[f32], w: &[f32], bias: &[f32], b: usize, d: usize, l: usize, l_out: usize, k: usize, pad: i64) -> Vec<f32> {
    let mut out = vec![0f32; b * d * l_out];
    for bi in 0..b {
        for di in 0..d {
            let u_base = bi * d * l + di * l;
            let w_base = di * k;
            let o_base = bi * d * l_out + di * l_out;
            for li in 0..l_out {
                let mut acc = bias[di];
                for ki in 0..k {
                    let idx = li as i64 - pad + ki as i64;
                    if idx >= 0 && (idx as usize) < l {
                        acc += u[u_base + idx as usize] * w[w_base + ki];
                    }
                }
                out[o_base + li] = acc;
            }
        }
    }
    out
}

impl CustomOp3 for DepthwiseCausalConv1d {
    fn name(&self) -> &'static str {
        "depthwise_causal_conv1d"
    }

    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ws: &CpuStorage,
        wl: &Layout,
        bs: &CpuStorage,
        bl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let (b, d, l) = ul.shape().dims3()?;
        let (dw, k) = wl.shape().dims2()?;
        if dw != d {
            candle_core::bail!("depthwise_conv1d: weight channels {dw} != input channels {d}");
        }
        let l_out = self.out_len(l, k)?;
        let pad = self.padding as i64;
        let shape = Shape::from((b, d, l_out));
        match us.dtype() {
            DType::F32 => {
                let out = conv_cpu_f32(contig_f32(us, ul)?, contig_f32(ws, wl)?, contig_f32(bs, bl)?, b, d, l, l_out, k, pad);
                Ok((CpuStorage::F32(out), shape))
            }
            DType::BF16 => {
                // f32 accumulate, bf16 store — the deployed/trained regime.
                let out = conv_cpu_f32(
                    &contig_bf16_to_f32(us, ul)?,
                    &contig_bf16_to_f32(ws, wl)?,
                    &contig_bf16_to_f32(bs, bl)?,
                    b, d, l, l_out, k, pad,
                );
                let out: Vec<half::bf16> = out.iter().map(|&x| half::bf16::from_f32(x)).collect();
                Ok((CpuStorage::BF16(out), shape))
            }
            other => candle_core::bail!("depthwise_conv1d cpu: f32/bf16 only (got {other:?})"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ws: &candle_core::MetalStorage,
        wl: &Layout,
        bs: &candle_core::MetalStorage,
        bl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::MTLSize;

        let (b, d, l) = ul.shape().dims3()?;
        let (dw, k) = wl.shape().dims2()?;
        if dw != d {
            candle_core::bail!("depthwise_conv1d: weight channels {dw} != input channels {d}");
        }
        let l_out = self.out_len(l, k)?;
        let total = b * d * l_out;
        let dev = us.device();
        let dt = us.dtype();
        let (pipeline, dts) = match dt {
            DType::F32 => (f32_pipeline(dev)?, DType::F32.size_in_bytes()),
            DType::BF16 => (bf16_pipeline(dev)?, DType::BF16.size_in_bytes()),
            other => candle_core::bail!("metal depthwise_conv1d: f32/bf16 only (got {other:?})"),
        };

        let out_buf = dev.new_buffer(total, dt, "depthwise_conv1d")?;
        let encoder = dev.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        // Buffers u/weights/bias (honoring layout offsets), then the output.
        encoder.set_buffer(0, Some(us.buffer()), ul.start_offset() * dts);
        encoder.set_buffer(1, Some(ws.buffer()), wl.start_offset() * dts);
        encoder.set_buffer(2, Some(bs.buffer()), bl.start_offset() * dts);
        encoder.set_buffer(3, Some(&*out_buf), 0);
        // Scalar params at [[buffer(4..10)]].
        encoder.set_bytes(4, &(b as u32));
        encoder.set_bytes(5, &(d as u32));
        encoder.set_bytes(6, &(l as u32));
        encoder.set_bytes(7, &(k as u32));
        encoder.set_bytes(8, &(self.padding as u32));
        encoder.set_bytes(9, &(l_out as u32));

        // One thread per output element; the kernel guards `gid >= total`.
        let max_tg = pipeline.max_total_threads_per_threadgroup().max(1);
        let tg = total.clamp(1, max_tg);
        let n_groups = total.div_ceil(tg);
        encoder.dispatch_thread_groups(
            MTLSize { width: n_groups, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        // candle owns the command-buffer lifecycle (commit/flush on sync); the
        // shared encoder is neither ended nor committed here.

        Ok((MetalStorage::new(out_buf, dev.clone(), total, dt), Shape::from((b, d, l_out))))
    }
}

/// Compile + cache the f32 pipeline. The cache is per-thread and assumes a single
/// Metal device (the common case); a second device would recompile per thread.
#[cfg(feature = "metal")]
fn f32_pipeline(dev: &candle_core::MetalDevice) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use std::cell::OnceCell;
    thread_local! {
        static P: OnceCell<candle_metal_kernels::metal::ComputePipeline> = const { OnceCell::new() };
    }
    P.with(|cell| {
        if let Some(p) = cell.get() {
            return Ok(p.clone());
        }
        let mtl = dev.metal_device();
        let lib = mtl
            .new_library_with_source(SRC_F32, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal compile: {e}")))?;
        let func = lib
            .get_function("depthwise_causal_conv1d_f32", None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function: {e}")))?;
        let pipeline = mtl
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline: {e}")))?;
        let _ = cell.set(pipeline.clone());
        Ok(pipeline)
    })
}

/// Compile + cache the bf16 pipeline (mirrors [`f32_pipeline`]).
#[cfg(feature = "metal")]
fn bf16_pipeline(dev: &candle_core::MetalDevice) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use std::cell::OnceCell;
    thread_local! {
        static P: OnceCell<candle_metal_kernels::metal::ComputePipeline> = const { OnceCell::new() };
    }
    P.with(|cell| {
        if let Some(p) = cell.get() {
            return Ok(p.clone());
        }
        let mtl = dev.metal_device();
        let lib = mtl
            .new_library_with_source(SRC_BF16, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal compile bf16: {e}")))?;
        let func = lib
            .get_function("depthwise_causal_conv1d_bf16", None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function bf16: {e}")))?;
        let pipeline = mtl
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline bf16: {e}")))?;
        let _ = cell.set(pipeline.clone());
        Ok(pipeline)
    })
}

/// Depthwise causal conv1d over `u` `[B, D, L]` with `weight` `[D, K]` and an optional
/// `bias` `[D]`. Returns `[B, D, L + 2·padding − K + 1]`. **f32 or bf16** — bf16 runs the
/// deployed/trained regime (f32 accumulate, bf16 store), so the short-conv keeps the
/// model's dtype with no upcast. The bias defaults to zeros in the input dtype.
pub fn depthwise_conv1d(u: &Tensor, weight: &Tensor, bias: Option<&Tensor>, padding: usize) -> Result<Tensor> {
    let (_b, d, _l) = u.dims3()?;
    let dt = u.dtype();
    if dt != DType::F32 && dt != DType::BF16 {
        candle_core::bail!("depthwise_conv1d supports f32 and bf16 (got {dt:?})");
    }
    let u = u.contiguous()?;
    let weight = weight.contiguous()?;
    let bias = match bias {
        Some(x) => x.contiguous()?,
        None => Tensor::zeros(d, dt, u.device())?,
    };
    u.apply_op3(&weight, &bias, DepthwiseCausalConv1d { padding })
}

/// Streaming causal depthwise conv1d with a cache of the prior `K-1` input samples — the
/// form LFM2's short-conv (`conv_l_cache`) decode needs. Processes `x` `[B,D,T]` (`T ≥ 1`)
/// preceded by `cache` `[B,D,K-1]` (the last K-1 inputs from the previous call; `None` =
/// zero-pad, i.e. a fresh prefill), and returns `(y [B,D,T], new_cache [B,D,K-1])` where
/// `new_cache` is the last K-1 samples of `[cache | x]`, ready for the next call.
///
/// The conv is just [`depthwise_conv1d`] run as a *valid* (no-pad) conv over the
/// `K-1`-prepended stream — the same verified Metal kernel, with the left boundary fed
/// from the cache instead of zeros (the cache cat/extract is host buffer management, the
/// convolution is the kernel). f32 or bf16 (the dtype is preserved). For `cache = None` this equals
/// `depthwise_conv1d(x, weight, None, K-1).narrow(2, 0, T)`, and a `T=1` step is exactly
/// LFM2's decode gather `Σ_k window[k]·w[k]`.
pub fn depthwise_conv1d_stream(x: &Tensor, weight: &Tensor, cache: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    let (b, d, t) = x.dims3()?;
    let (_d, k) = weight.dims2()?;
    let p = k - 1; // cache length = the K-1 prior input samples
    let cache = match cache {
        Some(c) => c.contiguous()?,
        None => Tensor::zeros((b, d, p), x.dtype(), x.device())?,
    };
    let stream = Tensor::cat(&[&cache, &x.contiguous()?], 2)?; // [B, D, K-1+T]
    let y = depthwise_conv1d(&stream, weight, None, 0)?; // valid conv → [B, D, T]
    let new_cache = stream.narrow(2, t, p)?.contiguous()?; // last K-1 of [cache | x]
    Ok((y, new_cache))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn naive(u: &[f32], w: &[f32], bias: &[f32], b: usize, d: usize, l: usize, k: usize, pad: usize) -> (Vec<f32>, usize) {
        let l_out = (l as i64 + 2 * pad as i64 - k as i64 + 1) as usize;
        let mut out = vec![0f32; b * d * l_out];
        for bi in 0..b {
            for di in 0..d {
                for li in 0..l_out {
                    let mut acc = bias[di];
                    for ki in 0..k {
                        let idx = li as i64 - pad as i64 + ki as i64;
                        if idx >= 0 && (idx as usize) < l {
                            acc += u[bi * d * l + di * l + idx as usize] * w[di * k + ki];
                        }
                    }
                    out[bi * d * l_out + di * l_out + li] = acc;
                }
            }
        }
        (out, l_out)
    }

    #[test]
    fn cpu_depthwise_matches_naive() {
        let dev = Device::Cpu;
        let (b, d, l, k, pad) = (2usize, 3, 7, 3, 2);
        let u: Vec<f32> = (0..b * d * l).map(|i| i as f32 * 0.1 - 1.0).collect();
        let w: Vec<f32> = (0..d * k).map(|i| i as f32 * 0.05).collect();
        let bias: Vec<f32> = (0..d).map(|i| i as f32 * 0.3).collect();
        let ut = Tensor::from_vec(u.clone(), (b, d, l), &dev).unwrap();
        let wt = Tensor::from_vec(w.clone(), (d, k), &dev).unwrap();
        let bt = Tensor::from_vec(bias.clone(), (d,), &dev).unwrap();
        let out = depthwise_conv1d(&ut, &wt, Some(&bt), pad).unwrap();
        let (exp, l_out) = naive(&u, &w, &bias, b, d, l, k, pad);
        assert_eq!(out.dims(), &[b, d, l_out]);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for (a, e) in got.iter().zip(exp.iter()) {
            assert!((a - e).abs() < 1e-5, "{a} vs {e}");
        }
    }

    // The streaming-cache correctness property: feeding a sequence through the stream op in
    // ragged chunks (incl. single-step `T=1` decode chunks), carrying the cache, must equal
    // the full-sequence zero-pad causal conv. If this holds, LFM2's per-token decode is exact.
    #[test]
    fn stream_chunks_match_full_sequence() {
        let dev = Device::Cpu;
        let (b, d, k) = (2usize, 4, 3);
        let total = 11usize;
        let x: Vec<f32> = (0..b * d * total).map(|i| ((i * 7 % 13) as f32 * 0.1) - 0.6).collect();
        let w: Vec<f32> = (0..d * k).map(|i| (i * 5 % 7) as f32 * 0.05).collect();
        let xt = Tensor::from_vec(x, (b, d, total), &dev).unwrap();
        let wt = Tensor::from_vec(w, (d, k), &dev).unwrap();
        // reference: one-shot full-sequence causal conv.
        let full: Vec<f32> = depthwise_conv1d(&xt, &wt, None, k - 1)
            .unwrap().narrow(2, 0, total).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // streamed: prefill chunk then single-step decode chunks then larger chunks.
        let mut cache: Option<Tensor> = None;
        let mut ys: Vec<Tensor> = Vec::new();
        let mut pos = 0usize;
        for &chunk in &[3usize, 1, 1, 4, 2] {
            let xc = xt.narrow(2, pos, chunk).unwrap();
            let (y, nc) = depthwise_conv1d_stream(&xc, &wt, cache.as_ref()).unwrap();
            ys.push(y);
            cache = Some(nc);
            pos += chunk;
        }
        let streamed: Vec<f32> = Tensor::cat(&ys.iter().collect::<Vec<_>>(), 2)
            .unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let maxd = full.iter().zip(&streamed).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-5, "streamed (chunked, incl. T=1) != full-sequence conv: max diff {maxd}");
        eprintln!("depthwise stream (incl. single-step decode) == full sequence, max diff {maxd:.2e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, d, l, k, pad) = (2usize, 4, 33, 3, 2);
        let u: Vec<f32> = (0..b * d * l).map(|i| (i * 7 % 13) as f32 * 0.1 - 0.6).collect();
        let w: Vec<f32> = (0..d * k).map(|i| (i * 5 % 7) as f32 * 0.05).collect();
        let bias: Vec<f32> = (0..d).map(|i| i as f32 * 0.2).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let ut = Tensor::from_vec(u.clone(), (b, d, l), dev).unwrap();
            let wt = Tensor::from_vec(w.clone(), (d, k), dev).unwrap();
            let bt = Tensor::from_vec(bias.clone(), (d,), dev).unwrap();
            depthwise_conv1d(&ut, &wt, Some(&bt), pad)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        assert_eq!(cpu.len(), met.len());
        let maxd = cpu.iter().zip(met.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-5, "metal vs cpu max diff {maxd}");
        eprintln!("metal == cpu depthwise conv1d, max diff {maxd:.2e}");
    }

    // bf16 path (the deployed dtype): f32 accumulate, bf16 store — metal == cpu.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu_bf16() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, d, l, k, pad) = (2usize, 4, 33, 3, 2);
        let u: Vec<f32> = (0..b * d * l).map(|i| (i * 7 % 13) as f32 * 0.1 - 0.6).collect();
        let w: Vec<f32> = (0..d * k).map(|i| (i * 5 % 7) as f32 * 0.05).collect();
        let bias: Vec<f32> = (0..d).map(|i| i as f32 * 0.2).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let cv = |v: &[f32], s: (usize, usize, usize)| Tensor::from_vec(v.to_vec(), s, dev).unwrap().to_dtype(DType::BF16).unwrap();
            let ut = cv(&u, (b, d, l));
            let wt = Tensor::from_vec(w.clone(), (d, k), dev).unwrap().to_dtype(DType::BF16).unwrap();
            let bt = Tensor::from_vec(bias.clone(), (d,), dev).unwrap().to_dtype(DType::BF16).unwrap();
            depthwise_conv1d(&ut, &wt, Some(&bt), pad)
                .unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        // both f32-accumulate then a single bf16 store; fma vs mul-add order can differ by ≤1 bf16 ULP.
        let maxd = cpu.iter().zip(&met).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-2, "bf16 metal vs cpu max diff {maxd}");
        eprintln!("depthwise bf16 (f32-accum, bf16-store): metal == cpu, max diff {maxd:.2e}");
    }
}
