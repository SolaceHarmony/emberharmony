//! flashkern — CPU replicas of the Metal JIT kernels.
//!
//! `crates/candle-flashfftconv/` embeds its Metal kernels as MSL source
//! strings inside `.rs` files and JIT-compiles them at runtime. The production CPU path is the
//! resident native stage machine ([`native_engine`]): C++ owns plans, tickets, and barriers;
//! architecture assembly owns numerical work. Rust modules here are temporary ABI rims and
//! test-only parity instruments pending the native audio-session cutover.
//!
//! Design docs + the Metal-idiom → opcode map: `docs/FLASHKERN.md`.

#[cfg(test)]
mod bridge;
#[cfg(feature = "oracle")]
pub mod candle_ops; // candle CustomOp bridges — oracle-only compatibility seam
#[cfg(test)]
pub mod dd; // ABI record used only by native DD conformance tests
pub mod decode; // fused decode blocks: threadgroup fallback + ShortConv/DepthDecode blocks
pub mod native_engine; // resident native stage-machine rim (native/src/engine/flashkern_engine.cpp) // private Rust-kcoro/native SQ/CQ conformance tests

pub mod neon; // aarch64 bridge to native/kernels/aarch64/flashkern_neon.cpp (BFMMLA GEMM, FFT, DD, fast-math)
pub mod x86; // x86-64 bridge to native/kernels/x86_64/flashkern_x86.cpp (VDPBF16PS / AVX2 sibling)
