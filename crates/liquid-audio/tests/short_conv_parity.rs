//! Parity gate for the LFM2 short-conv prefill swap: the FlashFFTConv `depthwise_conv1d`
//! must reproduce candle's generic `Conv1d` (the op it replaces in `ShortConv::forward`)
//! to f32 precision. This is the exact prefill math — causal depthwise, `padding = K-1`,
//! `groups = H`, narrowed to the input length — so if it matches, the swap is faithful and
//! the backbone is unchanged (modulo float-accumulation order). Synthetic F32 tensors; no model weights.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};

fn run(h: usize, l: usize, k: usize, seed: u64) -> f32 {
    let dev = Device::Cpu;
    let b = 2usize;
    // deterministic pseudo-random inputs
    let bx: Vec<f32> = (0..b * h * l)
        .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..h * k)
        .map(|i| (((i as u64 * 40503 + seed) % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    let bx = Tensor::from_vec(bx, (b, h, l), &dev).unwrap();
    let weight_3d = Tensor::from_vec(w, (h, 1, k), &dev).unwrap(); // (H,1,K) — checkpoint layout

    // reference: candle Conv1d (what ShortConv used before the swap)
    let conv = Conv1d::new(
        weight_3d.clone(),
        None,
        Conv1dConfig {
            padding: k - 1,
            groups: h,
            ..Default::default()
        },
    );
    let reference = conv.forward(&bx).unwrap().narrow(2, 0, l).unwrap();

    // new path: the FlashFFTConv depthwise kernel (weight squeezed to (H,K))
    let w2d = weight_3d.squeeze(1).unwrap();
    let fused = candle_flashfftconv::depthwise_conv1d(&bx, &w2d, None, k - 1)
        .unwrap()
        .narrow(2, 0, l)
        .unwrap();

    let a: Vec<f32> = reference
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let c: Vec<f32> = fused
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    a.iter()
        .zip(&c)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn depthwise_conv1d_matches_candle_conv1d() {
    // K=3 (the LFM2 conv_l_cache default) plus a couple of other sizes / lengths.
    let cases = [
        (2048usize, 64usize, 3usize),
        (512, 7, 3),
        (256, 33, 4),
        (128, 1, 3),
    ];
    for (h, l, k) in cases {
        let d = run(h, l, k, 1234);
        assert!(
            d < 1e-5,
            "depthwise_conv1d vs candle Conv1d (H={h} L={l} K={k}): max diff {d}"
        );
        eprintln!("short-conv prefill parity H={h} L={l} K={k}: max diff {d:.2e}");
    }
}

// The DEPLOYED dtype: the bf16 depthwise must run the faithful regime — f32 accumulate,
// bf16 store — i.e. it equals the f32 conv of the SAME bf16 inputs, rounded to bf16.
// (candle Conv1d has no bf16 CPU path, so we can't compare against it directly here; with
// the f32 parity above showing depthwise-f32 == candle Conv1d, this transitively gives
// bf16 == bf16_round(candle Conv1d) — the deployed prefill numerics.)
fn run_bf16(h: usize, l: usize, k: usize, seed: u64) -> f32 {
    let dev = Device::Cpu;
    let b = 2usize;
    let bx: Vec<f32> = (0..b * h * l)
        .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..h * k)
        .map(|i| (((i as u64 * 40503 + seed) % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    // bf16-rounded inputs (what the model holds).
    let bxb = Tensor::from_vec(bx, (b, h, l), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let wb = Tensor::from_vec(w, (h, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    // reference: same bf16 inputs, conv in f32, then a single round to bf16.
    let reference = candle_flashfftconv::depthwise_conv1d(
        &bxb.to_dtype(DType::F32).unwrap(),
        &wb.to_dtype(DType::F32).unwrap(),
        None,
        k - 1,
    )
    .unwrap()
    .narrow(2, 0, l)
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    // our bf16 kernel: bf16 inputs straight through (f32 accumulate, bf16 store).
    let bf16 = candle_flashfftconv::depthwise_conv1d(&bxb, &wb, None, k - 1)
        .unwrap()
        .narrow(2, 0, l)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let a: Vec<f32> = reference.flatten_all().unwrap().to_vec1().unwrap();
    let c: Vec<f32> = bf16.flatten_all().unwrap().to_vec1().unwrap();
    a.iter()
        .zip(&c)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn depthwise_conv1d_bf16_is_f32_accum_bf16_store() {
    for (h, l, k) in [(2048usize, 64usize, 3usize), (512, 7, 3), (256, 33, 4)] {
        let d = run_bf16(h, l, k, 1234);
        assert!(
            d < 1e-6,
            "bf16 depthwise != f32-accumulate/bf16-store (H={h} L={l} K={k}): max diff {d}"
        );
        eprintln!("short-conv prefill bf16 regime (f32-accum, bf16-store) H={h} L={l} K={k}: max diff {d:.2e}");
    }
}
