//! Decode-step speed test: the fused causal_conv1d_update_fused vs the composed
//! path liquid-audio's ShortConv runs today (explicit B⊙x / C⊙ gate ops around
//! depthwise_conv1d_stream's [state|x] concat conv). LFM2 shape: D=2048, K=3, T=1.
//!
//! Run:  cargo run --release --features metal --example bench_conv_update

use candle_core::{DType, Device, Tensor};
use candle_flashfftconv::{causal_conv1d_update_fused, depthwise_conv1d_stream};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() -> Res<()> {
    let dev = Device::new_metal(0)?;
    let (b, d, t, k) = (1usize, 2048usize, 1usize, 3usize);
    let iters = 2000usize;
    let dtype = DType::BF16; // the deployed regime

    let bcx = Tensor::randn(0f32, 1f32, (b, 3 * d, t), &dev)?.to_dtype(dtype)?;
    let state0 = Tensor::zeros((b, d, k - 1), dtype, &dev)?;
    let w = Tensor::randn(0f32, 0.2f32, (d, k), &dev)?.to_dtype(dtype)?;

    // --- composed path (today's ShortConv): gates as separate ops + concat conv ---
    let composed = |state: &Tensor| -> Res<(Tensor, Tensor)> {
        let bgate = bcx.narrow(1, 0, d)?;
        let cgate = bcx.narrow(1, d, d)?;
        let x = bcx.narrow(1, 2 * d, d)?;
        let bxv = bgate.mul(&x)?.contiguous()?;
        let (conv, ns) = depthwise_conv1d_stream(&bxv, &w, Some(state))?;
        Ok((cgate.mul(&conv)?, ns))
    };

    // --- fused path: one dispatch ---
    let fused = |state: &Tensor| -> Res<(Tensor, Tensor)> {
        Ok(causal_conv1d_update_fused(&bcx, state, &w)?)
    };

    // Warmup both (pipeline compiles) + correctness spot check.
    let (yc, _) = composed(&state0)?;
    let (yf, _) = fused(&state0)?;
    let diff = (yc.to_dtype(DType::F32)? - yf.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("spot check |composed - fused| = {diff:.3e}");
    dev.synchronize()?;

    let bench = |name: &str, f: &dyn Fn(&Tensor) -> Res<(Tensor, Tensor)>| -> Res<f64> {
        let mut state = state0.clone();
        // realistic decode: state feeds forward step to step
        let start = Instant::now();
        for _ in 0..iters {
            let (_y, ns) = f(&state)?;
            state = ns;
        }
        dev.synchronize()?;
        let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{name:>10}: {us:8.1} µs/step  ({iters} steps, D={d}, K={k}, {dtype:?})");
        Ok(us)
    };

    let uc = bench("composed", &composed)?;
    let uf = bench("fused", &fused)?;
    println!(
        "speedup: {:.2}x per ShortConv step ({} conv layers/token)",
        uc / uf,
        10
    );
    println!(
        "per-token estimate: {:.1} µs -> {:.1} µs across 10 conv layers",
        uc * 10.0,
        uf * 10.0
    );
    Ok(())
}
