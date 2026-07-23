# Native Interface Headers

Shared C/C++ headers live here when an engine interface is consumed by more
than one native translation unit. The native lifecycle surface is
`lfm_types.h`, `lfm_runtime.h`, and `lfm_session.h`. Rust does not mirror or
invoke it. Other headers in this directory are native build-private; their
location does not make them installable interfaces.

- `lfm_safetensors.h` — opaque resident weight image and immutable tensor views.
- `lfm_types.h` — product status, opaque owner, and ticket identities.
- `lfm_runtime.h` — native executor ownership, opaque model and conversation
  creation, sampling control policy, and strict runtime-scoped child lifecycle.
- `lfm_session.h` — self-recurring session, typed UTF-8 commands, reliable
  semantic callbacks, interruption, stop, join, and bounded snapshots.
- `lfm_audio_dock.h` — private generation-checked capture/playback lease cells
  and pointer resolution. This header is not part of the product control ABI.
