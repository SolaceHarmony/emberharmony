# Native ABI Headers

Shared C/C++ ABI headers live here when an engine interface is consumed by more
than one translation unit or mirrored by a host language.

- `lfm_safetensors.h` — opaque resident weight image and immutable tensor views.
- `lfm_kernel_bridge.h` — private fixed-record SQ/CQ and doorbell ABI between
  Rust kcoro policy and the native fixed executor.
