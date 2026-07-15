# liquid-audio Native Sources

- `include/` — shared native ABI headers.
- `src/io/` — native model-file readers and resident weight-image construction.
- `src/engine/` — non-numerical resident control, stage, queue, and barrier implementation.
- `kernels/aarch64/` — hand-written AArch64/NEON assembly math.
- `kernels/x86_64/` — hand-written x86-64 assembly math.
- `reference/` — reference or fallback kernels.

Cargo builds these sources through `../build.rs`. Rust sees opaque engine/model
handles and PCM/control docking records, never numerical kernel symbols. The
remaining architecture `.cpp` numerical bodies are migration debt and must be
deleted as their paired `.S` families land.

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
