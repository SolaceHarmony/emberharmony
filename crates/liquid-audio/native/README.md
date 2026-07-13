# liquid-audio Native Sources

- `include/` — shared native ABI headers.
- `src/io/` — native model-file readers and resident weight-image construction.
- `src/engine/` — resident native stage-machine implementation.
- `kernels/aarch64/` — NEON/AArch64 kernels.
- `kernels/x86_64/` — AVX/x86-64 kernels.
- `reference/` — reference or fallback kernels.

Cargo builds these sources through `../build.rs`; symbol names are kept stable for
the Rust FFI layer in `src/compute/flashkern`.

## Resident Weights

`include/lfm_safetensors.h` is the C ABI for native checkpoint ownership.
`src/io/safetensors.cpp` accepts a safetensors file, Hugging Face shard index, or
checkpoint directory; reads every selected shard directly into one 64-byte-aligned
allocation; validates the complete payload; and returns immutable pointer/offset
views. The native image itself never materializes payloads as Rust or Candle
tensors. `src/compute/weights.rs` contains the explicit, counted compatibility
copies still required by model components that have not moved to native kernels.

The JSON parser is nlohmann/json 3.11.3, vendored from the local `ember-ml` tree
under `vendor/nlohmann/` with its MIT license.
