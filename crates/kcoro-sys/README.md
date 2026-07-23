# kcoro-sys

Native bindings for the vendored `kcoro_arena` coordination runtime.

This crate compiles the saved-frame stackless runtime and logical numerical
teams alongside the host POSIX adapter. Exact tickets resume one continuation
on any free eligible worker in the bounded pool. Product payload tickets and
borrowed numerical views remain in Flashkern instead of being duplicated here.
The generic Rust executor/ring crate and C channel/work-stealing compatibility
surfaces stay deleted. Rust may publish producer edges and own platform
callbacks; it does not schedule numerical lanes or carry numerical payloads
through a framework channel.

Run from the repo root:

```sh
cargo build -p kcoro-sys
```
