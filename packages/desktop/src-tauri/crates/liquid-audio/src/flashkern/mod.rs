//! flashkern — CPU replicas of the Metal JIT kernels.
//!
//! `experiments/lfm2-audio-voice/candle-flashfftconv/` embeds its Metal kernels as MSL source
//! strings inside `.rs` files and JIT-compiles them at runtime. This module is the CPU edition
//! of that kernel library: each Metal kernel is replicated with the closest native SIMD idiom —
//! NEON on aarch64 ([`neon`]), AVX2 / AVX-512 on x86-64 ([`x86`]) — and the multi-stage kernels
//! run under a faithful CPU port of the GPU dispatch model ([`fanout`]: threadgroup grid →
//! rayon, simdgroup lanes → a scoped thread team, `threadgroup_barrier` → `std::sync::Barrier`).
//!
//! Design docs + the Metal-idiom → opcode map: `csrc/FLASHKERN.md`.

pub mod candle_ops; // candle CustomOp bridges — the seam the model wires through on CPU
pub mod dd; // double-double toolkit (CPU port of double_double.metal) for the dd kernels
pub mod decode; // fused decode blocks: one threadgroup dispatch per block, spin barriers
pub mod engine; // the kcoro tile engine — the chassis (zero-spin dispatch, ENGINE_DESIGN.md §2)
pub mod native_engine; // the resident native engine's Rust rim (csrc/flashkern_engine.cpp)

pub(crate) use fanout::Shared;
pub mod fanout; // GPU dispatch model on threads: grid fan-out, lane teams, real barriers
pub mod neon; // aarch64 bridge to csrc/flashkern_neon.cpp (BFMMLA GEMM, FFT, DD, fast-math)
pub mod x86; // x86-64 bridge to csrc/flashkern_x86.cpp (VDPBF16PS / AVX2 sibling)

/// Arch-dispatched nt GEMV over full rows (test/support helper for the engine smoke).
#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
pub(crate) fn neon_or_x86_gemv(x: &[u16], w: &[u16], out: &mut [f32], n: usize, k: usize) {
    #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
    neon::bf16_gemm_nt_into(x, w, out, 1, n, k);
    #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
    x86::bf16_gemm_nt_into(x, w, out, 1, n, k);
}
