# LFM2.5-Audio local voice — a reset

An exploration of a **different** voice architecture from the LiveKit/brain-session
work on `fix/appimage-cleanup-dev-channel`. This branch (`explore/lfm2-audio-voice`,
off `dev`) deliberately does **not** use LiveKit and does **not** use the
SessionLLM brain bridge.

## The idea

Invert the brain approach. Instead of *dumb voice I/O + a big bridged brain*, run a
**small, local, CPU-friendly LFM2.5-Audio model as the conversational agent itself**,
and give it a single primitive tool: **delegate the hard work to a GLM subagent**.

```
mic ─► LFM2.5-Audio (local, llama.cpp, interleaved speech+text)
            │  converses in its own voice; emits text on a side channel
            ├─ no marker        ─► play LFM's own spoken reply         (small talk / quick answers)
            └─ "DELEGATE: task" ─► GLM-5.1 subagent (ollama-cloud API) does the work
                                     └─► LFM2.5-Audio TTS speaks the result
```

- **LFM2.5-Audio** = the front-line intelligence + the ears and voice. ~1.5B,
  runs on CPU/Metal via llama.cpp GGUF (Q4_0 ≈ <1 GB). English. Speech-to-speech,
  plus standalone ASR/TTS modes.
- **GLM-5.1** (`ollama-cloud/glm-5.1`, OpenAI-compatible API) = the heavy lifter,
  invoked only when LFM flags real work. Optional `bash` tool so it can actually
  *do* things (off by default).
- No LiveKit. No EmberHarmony session/brain bridge. "Just another local model"
  with a constrained tool set.

### Why a text-marker "primitive tool"?

LFM2.5-Audio has **no native function calling** (verified — Liquid ships tool
calling only in separate text models like `LFM2-1.2B-Tool`). So delegation rides a
simple convention the small model can follow: it emits one line `DELEGATE: <task>`
on the text channel, and the orchestrator routes that to GLM. This is the core
hypothesis under test — whether a 1.5B audio model reliably converses *and* knows
when to hand off.

## Decisions (chosen for this exploration)

| Choice | Decision |
| --- | --- |
| Branch base | off `dev` (clean of the LiveKit/brain work) |
| LFM runtime | llama.cpp GGUF (lightest, CPU/Metal) via PR #18641 binaries |
| Hard-work model | `ollama-cloud/glm-5.1` over the OpenAI-compatible API |
| Implementation | **Rust** — a native `cargo` binary, like the rest of the native voice code (no Python layer) |

## Files (Rust crate)

- `Cargo.toml` — binary `lfm-voice`; deps cpal, hound, reqwest (rustls/blocking), serde, anyhow.
- `src/glm.rs` — GLM-5.1 delegate over `https://ollama.com/v1` (key read from
  `auth.json`/`$OLLAMA_API_KEY`, never logged). Optional `bash` tool loop, gated by
  `LFM_ALLOW_EXEC=1`. **Verified working** (the API path was confirmed live).
- `src/lfm.rs` — wrapper over `llama-liquid-audio-cli` (ASR / TTS / interleaved).
  Exact CLI taken verbatim from the official GGUF model card. Errors clearly until
  the binary + GGUFs are installed.
- `src/audio.rs` — cpal mic capture (RMS-gated utterance recorder → 16 kHz mono WAV)
  and WAV playback, the same I/O approach as the native client.
- `src/main.rs` — mic → LFM conversation → delegate-marker routing → speech.
  Routes: `marker` (default), `chat` (LFM only), `delegate` (everything to GLM).
- `setup.sh` — builds the llama.cpp liquid-audio runners and downloads the GGUFs.

## Run

```bash
cd experiments/lfm2-audio-voice
./setup.sh                         # builds llama.cpp PR #18641 + downloads GGUFs (one-time)
export LFM_BIN="$PWD/llama.cpp/build/bin/llama-liquid-audio-cli"
cargo run --release --bin lfm-voice   # start talking
```

Useful env: `LFM_ROUTE=chat|delegate|marker`, `LFM_ALLOW_EXEC=1` (let GLM run
shell), `LFM_VOICE="Use the UK male voice."`, `LFM_RMS_THRESHOLD` (mic gate).

## Status

- ✅ Rust crate compiles (`cargo check`); GLM subagent path verified against
  `ollama-cloud/glm-5.1`.
- ⏳ LFM runtime is real code but unrun — needs `setup.sh` (build PR #18641 +
  download GGUFs) on this machine.
- 📌 Per-turn the CLI reloads the model; once the loop proves out, switch
  `src/lfm.rs` to talk to `llama-liquid-audio-server` so the model stays resident.
- 🔬 Open questions to settle by running it:
  1. Does a 1.5B audio model follow the `DELEGATE:` marker reliably, or do we need a
     tiny dedicated router (e.g. `LFM2-1.2B-Tool`) between ASR and delegation?
  2. Interleaved latency on CPU/Metal vs. the cascade.
  3. Whether to acknowledge in LFM's voice *before* the GLM round-trip (barge-in feel).
  4. Mic endpointing (current: simple RMS gate) — good enough, or add a VAD?
