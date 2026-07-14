# kcoro-sys

Build-only sys crate for the vendored `kcoro_arena` coordination runtime.

This crate compiles the portable stackless core and the host POSIX adapter from
`vendor/kcoro_arena` as separate native archives. The core archive therefore
contains no OS implementation. The crate contains no context-switch assembly or
stackful dispatcher. `liquid-audio` reaches tickets through its private native C++
boundary; Rust does not schedule numerical lanes.

Run from the repo root:

```sh
cargo build -p kcoro-sys
```
