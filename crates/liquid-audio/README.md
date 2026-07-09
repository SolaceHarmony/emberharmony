# liquid-audio

Rust-native LFM2.5-Audio engine used by the desktop voice stack.

- Detailed architecture docs: `docs/`.
- Native C/C++ sources: `native/`.
- Public Rust API: `src/lib.rs`.
- Tauri integration: `packages/desktop/src-tauri` depends on this crate by path.

Run from the repo root:

```sh
cargo test -p liquid-audio --lib -- --nocapture
cargo build -p liquid-audio --all-targets
```
