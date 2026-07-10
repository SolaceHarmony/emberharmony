//! flashkern — CPU replicas of the Metal JIT kernels.
//!
//! `crates/candle-flashfftconv/` embeds its Metal kernels as MSL source
//! strings inside `.rs` files and JIT-compiles them at runtime. This module is the CPU edition
//! of that kernel library: each Metal kernel is replicated with the closest native SIMD idiom —
//! NEON on aarch64 ([`neon`]), AVX2 / AVX-512 on x86-64 ([`x86`]). The older fused regions run
//! under a faithful CPU port of the GPU dispatch model ([`fanout`]: threadgroup grid → rayon,
//! simdgroup lanes → a scoped thread team, `threadgroup_barrier` → `std::sync::Barrier`). The
//! live FFN decode mount is moving beyond that rung into the resident native stage machine
//! ([`native_engine`]).
//!
//! Design docs + the Metal-idiom → opcode map: `docs/FLASHKERN.md`.

pub mod candle_ops; // candle CustomOp bridges — the seam the model wires through on CPU
pub mod dd; // double-double toolkit (CPU port of double_double.metal) for the dd kernels
pub mod decode; // fused decode blocks: threadgroup fallback + ShortConv/DepthDecode blocks
pub mod engine; // prototype/reference kcoro channel engine; not the live production mount
pub mod native_engine; // resident native stage-machine rim (native/src/engine/flashkern_engine.cpp)

pub(crate) use fanout::Shared;
pub mod fanout; // GPU dispatch model on threads: grid fan-out, lane teams, real barriers
pub mod neon; // aarch64 bridge to native/kernels/aarch64/flashkern_neon.cpp (BFMMLA GEMM, FFT, DD, fast-math)
pub mod x86; // x86-64 bridge to native/kernels/x86_64/flashkern_x86.cpp (VDPBF16PS / AVX2 sibling)

/// Arch-dispatched nt GEMV over full rows (test/support helper for the engine smoke).
#[cfg(test)]
pub(crate) fn neon_or_x86_gemv(x: &[u16], w: &[u16], out: &mut [f32], n: usize, k: usize) {
    #[cfg(target_arch = "aarch64")]
    neon::bf16_gemm_nt_into(x, w, out, 1, n, k);
    #[cfg(target_arch = "x86_64")]
    x86::bf16_gemm_nt_into(x, w, out, 1, n, k);
}
