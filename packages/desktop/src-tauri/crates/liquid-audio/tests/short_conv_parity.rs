//! Parity gate for the LFM2 short-conv prefill swap: the FlashFFTConv `depthwise_conv1d`
//! must reproduce candle's generic `Conv1d` (the op it replaces in `ShortConv::forward`)
//! to f32 precision. This is the exact prefill math — causal depthwise, `padding = K-1`,
//! `groups = H`, narrowed to the input length — so if it matches, the swap is faithful and
//! the backbone is unchanged (modulo float-accumulation order). CPU/f32; no model needed.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};

fn run(h: usize, l: usize, k: usize, seed: u64) -> f32 {
    let dev = Device::Cpu;
    let b = 2usize;
    // deterministic pseudo-random inputs
    let bx: Vec<f32> = (0..b * h * l).map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0) - 1.0).collect();
    let w: Vec<f32> = (0..h * k).map(|i| (((i as u64 * 40503 + seed) % 1000) as f32 / 1000.0) - 0.5).collect();
    let bx = Tensor::from_vec(bx, (b, h, l), &dev).unwrap();
    let weight_3d = Tensor::from_vec(w, (h, 1, k), &dev).unwrap(); // (H,1,K) — checkpoint layout

    // reference: candle Conv1d (what ShortConv used before the swap)
    let conv = Conv1d::new(
        weight_3d.clone(),
        None,
        Conv1dConfig { padding: k - 1, groups: h, ..Default::default() },
    );
    let reference = conv.forward(&bx).unwrap().narrow(2, 0, l).unwrap();

    // new path: the FlashFFTConv depthwise kernel (weight squeezed to (H,K))
    let w2d = weight_3d.squeeze(1).unwrap();
    let fused = candle_flashfftconv::depthwise_conv1d(&bx, &w2d, None, k - 1)
        .unwrap()
        .narrow(2, 0, l)
        .unwrap();

    let a: Vec<f32> = reference.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    let c: Vec<f32> = fused.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    a.iter().zip(&c).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn depthwise_conv1d_matches_candle_conv1d() {
    // K=3 (the LFM2 conv_l_cache default) plus a couple of other sizes / lengths.
    let cases = [(2048usize, 64usize, 3usize), (512, 7, 3), (256, 33, 4), (128, 1, 3)];
    for (h, l, k) in cases {
        let d = run(h, l, k, 1234);
        assert!(d < 1e-5, "depthwise_conv1d vs candle Conv1d (H={h} L={l} K={k}): max diff {d}");
        eprintln!("short-conv prefill parity H={h} L={l} K={k}: max diff {d:.2e}");
    }
}
