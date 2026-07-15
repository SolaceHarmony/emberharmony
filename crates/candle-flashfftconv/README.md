# candle-flashfftconv

Candle CPU/Metal operators ported from FlashFFTConv and used by `liquid-audio`.

- Architecture notes: `docs/ARCHITECTURE.md`.
- Metal kernels: `src/metal/`.
- Rust `CustomOp` implementations: `src/`.

Run from the repo root:

```sh
cargo test -p candle-flashfftconv
```
