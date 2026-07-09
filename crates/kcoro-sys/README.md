# kcoro-sys

Build-only sys crate for the vendored kcoro coroutine runtime.

This crate compiles the C runtime and per-architecture context-switch assembly from
`vendor/kcoro`. It intentionally does not provide a safe Rust wrapper; `liquid-audio`
owns the private FFI declarations needed by its native engine.

Run from the repo root:

```sh
cargo build -p kcoro-sys
```
