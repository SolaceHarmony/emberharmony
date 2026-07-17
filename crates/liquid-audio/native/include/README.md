# Native ABI Headers

Shared C/C++ ABI headers live here when an engine interface is consumed by more
than one translation unit or mirrored by a host language. The product control
ABI is exactly `lfm_types.h`, `lfm_runtime.h`, `lfm_model.h` (a tombstone), and
`lfm_session.h`. Other headers in this directory are native build-private or
temporary parity surfaces; their location does not make them installable APIs.

- `lfm_safetensors.h` — opaque resident weight image and immutable tensor views.
- `lfm_types.h` — product status, opaque owner, and ticket identities.
- `lfm_runtime.h` — native executor ownership, opaque model and conversation
  creation, sampling control policy, and strict runtime-scoped child lifecycle.
- `lfm_model.h` — compatibility tombstone that includes only the product model
  lifecycle header. Numerical cutover/oracle declarations live privately under
  `native/src/model`.
- `lfm_session.h` — self-recurring session, typed UTF-8 commands, reliable
  semantic callbacks, interruption, stop, join, and bounded snapshots.
- `lfm_audio_dock.h` — private generation-checked capture/playback lease cells
  and pointer resolution. This header is not part of the product control ABI.
- `lfm_kernel_bridge.h` — private fixed-record native model SQ/CQ and doorbell
  ABI. Production Rust does not broker this queue; its bindings are test-only.
