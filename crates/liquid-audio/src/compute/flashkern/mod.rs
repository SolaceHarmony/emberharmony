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
pub mod native_engine; // resident native stage-machine rim (native/src/engine/flashkern_engine.cpp)
#[cfg(test)]
mod bridge; // private Rust-kcoro/native SQ/CQ conformance tests

pub(crate) use fanout::Shared;
pub mod fanout; // GPU dispatch model on threads: grid fan-out, lane teams, real barriers
pub mod neon; // aarch64 bridge to native/kernels/aarch64/flashkern_neon.cpp (BFMMLA GEMM, FFT, DD, fast-math)
pub mod x86; // x86-64 bridge to native/kernels/x86_64/flashkern_x86.cpp (VDPBF16PS / AVX2 sibling)
