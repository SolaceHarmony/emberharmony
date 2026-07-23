# liquid-audio

Native LFM2.5-Audio engine used by the desktop voice stack. Native
C++/kcoro/Flashkern and architecture kernels own the model image, inference,
turn policy, CoreAudio callbacks, and PCM docks. The Rust crate contains desktop
control records and checkpoint download support only; it has no native linkage
or inference entry point.

- Detailed architecture docs: `docs/`.
- Native C/C++ sources: `native/`.
- Desktop Rust support: `src/lib.rs`.
- Tauri UI/control integration: `packages/desktop/src-tauri`.

Run from the repo root:

```sh
cmake -S crates/kcoro-sys/vendor/kcoro_arena -B build/kcoro
cmake --build build/kcoro
ctest --test-dir build/kcoro
make -C crates/liquid-audio/native/tools
```

The slow two-agent speech test is a native release-acceptance executable. It
receives the checkpoint and output mode as ordinary arguments; no environment
variable changes its implementation:

```sh
crates/liquid-audio/native/tools/build/lfm-native-speech-test \
  /absolute/LFM2.5-Audio-1.5B 8 silent
```

Use `buffered` or `stream` in place of `silent` only for deliberate audible
acceptance. All modes consume the same in-memory native PCM leases.

On Apple Silicon, cross-build the Darwin x86_64 suites and let macOS execute
the test binaries through Rosetta directly:

Configure the same CMake projects with
`-DCMAKE_OSX_ARCHITECTURES=x86_64` for the Rosetta build. Unsupported
instruction sets fail readiness; no scalar inference substitute is selected.
