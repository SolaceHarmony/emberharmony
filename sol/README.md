# Sol workspace

This directory owns Sol's experimental programs. It does not own `vera/` and
must not modify Vera's files.

The hard boundary here is native-only experimentation: C++23 orchestration,
kcoro continuations, Flashkern passes, immutable safetensors image views, and
PCM held in memory. No Rust launcher, Candle path, tensor object, WAV transfer,
or stdout PCM transport belongs in an experiment in this directory.

`native_spec_replay_probe.cpp` preserves the first prefix-replay experiment as
a standalone C++ executable. Its next correction is semantic, not a rewrite:
each microphone prefix must be independently encoded through the native
frontend and Conformer so future audio cannot leak, and the evidence must score
the immediate token prediction/horizon at each prefix rather than require the
entire eventual spoken reply to be byte-identical.

The Makefile links the already-built production native archives without adding
anything to Rust's build or test graph. Supply the exact current archive paths:

```sh
make -C sol \
  AUDIO_OUT=/absolute/path/to/target/debug/build/liquid-audio-HASH/out \
  KCORO_OUT=/absolute/path/to/target/debug/build/kcoro-sys-HASH/out

sol/build/native_spec_replay_probe /path/to/LFM2.5-Audio-1.5B 8
```

The program reports bounded textual evidence only. All source and generated
PCM remains in native memory.
