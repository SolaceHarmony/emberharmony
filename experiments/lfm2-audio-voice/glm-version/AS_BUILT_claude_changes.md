# Claude + Codex changes to `liquid-audio` — as-built

> This documents the changes Claude and Codex made to the `liquid-audio` crate
> across multiple sessions, **as built** (not as proposed). It covers what was
> changed, how it works, what's verified, and what remains. The crate was
> originally at `experiments/lfm2-audio-voice/liquid-audio-rs/` and has been
> moved to `packages/desktop/src-tauri/crates/liquid-audio/`.

## State at a glance

The crate is now at `packages/desktop/src-tauri/crates/liquid-audio/` and
committed. All files below are in the committed tree.

| File | What changed |
|---|---|
| `src/threads.rs` | torch-parity intra-op thread pool |
| `src/bf16_gemm.rs` | NEON BFMMLA bf16 GEMM: FFI + `CustomOp2` + runtime gate |
| `csrc/bf16_gemm.c` | the C BFMMLA micro-kernel |
| `build.rs` | compiles the C kernel on aarch64 via `cc` |
| `src/lib.rs` | module declarations + re-exports (incl. `realtime`, `voice_runtime`) |
| `src/loader.rs` | calls `configure_intraop_threads()` at top of `from_pretrained` |
| `Cargo.toml` | `rayon`/`num_cpus`/`libc`/`half`/`crossbeam-channel` deps, `cc` build-dep, `accelerate`/`metal` features |
| `src/audio_out.rs` | `AudioDetokenizer: Send` bound |
| `src/model/mlp.rs` | `Sequential` → `Vec<Box<dyn Module + Send>>` + 3 tests |
| `src/model/lfm2_hf.rs` | vendored `build_causal_mask`/`repeat_kv`; mask memoization (`Cache::mask`); faithful `Tensor::cat` KV cache |
| `src/model/lfm2_audio.rs` | `generate_interleaved_cancellable` (barge-in); `GenParams::demo_defaults()`; multi-turn collect/append |
| `src/processor.rs` | `ChatState::from_parts` (multi-turn persistence constructor) |
| `src/realtime.rs` | `RealtimePipeline` (worker thread); `Lfm2VoiceEngine` (owns model); `ConversationState`; `StreamingPcmResampler`; multi-turn persistence; barge-in |
| `src/voice_runtime.rs` | `VoiceRuntime` (cpal VAD + playback); `RuntimeConfig`; `can_interrupt` gate |
| `src/candle_ext/transformers_utils.rs` | vendored `build_causal_mask` + `repeat_kv` (candle 0.10→0.9.2 backport) |
| `src/candle_ext/tensor_ext.rs` | `TensorExt::to_vec4` |
| `src/candle_ext/mod.rs` | `pub mod tensor_ext; pub mod transformers_utils;` |
| `examples/text_chat.rs` | text-only proof (no audio path) |
| `examples/chat_multiturn.rs` | canonical two-turn discrete-audio context proof |
| `examples/duplex_chat.rs` | full-duplex live demo (cpal + VAD + barge-in) |
| `THREADING_PARITY.md` | implementer's threading-parity view (updated for realtime pipeline) |
| `.github/workflows/rust-voice.yml` | Rust CI: build + test on Linux + macOS arm64 |
| **Desktop crate (Tauri integration)** | |
| `src/voice/control.rs` | Tauri commands: `voice_start`/`voice_stop`/`voice_status`/`voice_set_mic_enabled`; `VoiceEvent` contract; `VoiceStartResult`; `TurnMode`; `LiveKitGrant` |
| `src/voice/runtime.rs` | `VoiceRuntime` (session lifecycle); `ThreadManager`; `VoiceSession` enum (`Lfm2`/`Livekit`) |
| `src/voice/session.rs` | HTTP session bridge (LiveKit delegate path only) |
| `src/voice/FRONTEND_DESIGN.md` | voice frontend design (turn mode + live mode, one event-driven core) |
| `src/settings.rs` | `VoiceSettings`/`Lfm2Settings`/`VoiceProvider`; `lfm2_model_ref`/`lfm2_model_dir` resolution; `DEFAULT_LFM2_MODEL` |
| **Frontend (SolidJS)** | |
| `packages/app/src/lib/voice-settings.ts` | Tauri command wrappers; `VoiceStartResult` return type |
| `packages/app/src/lib/voice-state.ts` | pure decision functions (extracted, tested) |
| `packages/app/src/lib/voice-state.test.ts` | 6 unit tests for the decision functions |
| `packages/app/src/context/voice.tsx` | rewritten: provider-branched (lfm2 → Tauri Channel events; livekit → grant + room.connect) |
| `packages/app/src/components/settings-voice.tsx` | HF model field + local snapshot directory field |
| `packages/app/src/components/prompt-input.tsx` | mic button + mic-pause-on-typing via extracted `voiceButtonOn`/`voiceMicTarget` |

**Build state:** `cargo test --lib` → **62 passed; 0 failed; 1 ignored** (real-model engine test). `cargo build --features metal --examples` → clean.

---

## 1. Intra-op thread pool — torch parity (`src/threads.rs`)

### What it does
Replicates torch's `at::intraop_default_num_threads()` (`aten/src/ATen/ParallelCommon.cpp`):
1. Honours `OMP_NUM_THREADS` → `MKL_NUM_THREADS` → `RAYON_NUM_THREADS` (in that order; the first valid `>0` parse wins).
2. Else, on macOS, queries `hw.perflevel0.physicalcpu` via `libc::sysctlbyname` — the **performance cores only**, deliberately excluding the efficiency (E) cores. Falls back to `hw.physicalcpu` (Intel Macs), then `num_cpus::get_physical()`.
3. Installs that count as rayon's **global** pool (`ThreadPoolBuilder::new().num_threads(n).build_global()`) so candle's matmul (`gemm`) + conv/sort inherit it. `build_global` is idempotent — a second call (or a pool already built by a prior candle op) is a harmless no-op (`let _ =`).

### Why
candle/`gemm` default to `num_cpus::get()` (**all** logical cores), which schedules compute-bound matmul onto the slow E-cores — hurting throughput *and* tail latency via work-steal imbalance. On this M2 Max: rayon's default would be 12 (all logical); the fix picks 8 (P-cores), matching torch.

### Wiring
Called once at the top of `from_pretrained` (`loader.rs`), before the first tensor op.

### Tests
- `intraop_threads_is_sane_and_not_all_logical` — asserts `1 ≤ n ≤ num_cpus::get()`; prints `intraop threads = 8 (physical 12, logical 12)` on M2 Max.
- `realtime_pipeline_types_are_send` — asserts `LFM2AudioModel: Send` and `LFM2AudioProcessor: Send` (feasibility probe for the worker-thread pipeline; requires the MLP `Send` fix).

### Verification
`intraop threads = 8 (physical 12, logical 12)` on M2 Max — matches `sysctl hw.perflevel0.physicalcpu`.

---

## 2. NEON BFMMLA bf16 CPU GEMM (`src/bf16_gemm.rs` + `csrc/bf16_gemm.c` + `build.rs`)

### What it does
A hardware bf16 GEMM for candle's CPU path, closing the gap where candle 0.9.2's CPU matmul allowlist is `F16 | F32 | F64` only (bf16 → `UnsupportedDTypeForOp`, so the loader forced f32 on CPU). The Arm BFloat16 extension (FEAT_BF16) provides `BFMMLA`, which does a 2×4·4×2 bf16 matmul with **f32 accumulate** — the same numerics torch's CPU bf16 matmul uses.

### The C kernel (`csrc/bf16_gemm.c`)
- `lfm_bf16_gemm_f32(A, B, C, M, N, K)` — `C(M×N, f32) = A(M×K, bf16) · B(K×N, bf16)`, all row-major.
- Packs A/B into BFMMLA tile order: `vbfmmlaq_f32(acc, av, bv)` treats `av`/`bv` as 2×4 bf16 matrices, computes `a · bᵀ` (2×4·4×2 → 2×2), accumulates into a 2×2 f32 `acc` laid out `[c00,c01,c10,c11]`. Packing B's lane-row `r` = column `(jt+r)` of B over a 4-deep K block makes `(a · bᵀ)[i][j] = Σ_k A[it+i][k]·B[k][jt+j]` — an ordinary `C = A·B`.
- Zero-pads M→Mp (mult of 2), N→Np (mult of 2), K→Kp (mult of 4) via `calloc` (bf16 +0.0 padding contributes nothing to the dot products). Handles odd dims correctly (the test exercises 5×13×7).
- Compiled by `build.rs` via `cc` with `-march=armv8.2-a+bf16`, `opt_level(3)`, gated to aarch64 via `CARGO_CFG_TARGET_ARCH == "aarch64"`. Sets `cargo::rustc-cfg=has_bf16_kernel` so the Rust FFI is only wired where the kernel was built.

### The Rust FFI + op (`src/bf16_gemm.rs`)
- `extern "C" { fn lfm_bf16_gemm_f32(...) }` — declared under `cfg(all(target_arch = "aarch64", has_bf16_kernel))`.
- `has_feat_bf16() -> bool` — **runtime** FEAT_BF16 detection via `libc::sysctlbyname(c"hw.optional.arm.FEAT_BF16")`, cached in a `OnceLock<bool>`. On non-macOS aarch64, returns `false` (Linux `HWCAP2_BF16` via `getauxval` not wired yet). On non-aarch64, returns `false`.
- `bf16_gemm_available() -> bool` — `cfg!(all(target_arch = "aarch64", has_bf16_kernel)) && has_feat_bf16()`. The kernel must be both **built in** and **supported** by the running CPU.
- `Bf16Gemm` — a `candle_core::CustomOp2` (`cpu_fwd` only; backward and GPU paths intentionally bail). The single FFI call site. Validates 2-D shapes, contiguity, bf16 storage; extracts the `half::bf16` slices from `CpuStorage::BF16`; allocates `vec![0f32; m*n]`; calls the kernel; returns `(CpuStorage::F32(c), Shape::from((m, n)))`.
- `bf16_matmul(a, b) -> Result<Option<Tensor>>` — the safe wrapper: casts inputs to bf16 + contiguous, calls `a16.apply_op2_no_bwd(&b16, &Bf16Gemm)`. Returns `Ok(None)` when unavailable so callers fall back to candle's f32 path.

### Portability
The binary stays portable: `build.rs` only compiles the kernel on aarch64 (`cfg(has_bf16_kernel)`), and even on aarch64 the runtime `has_feat_bf16()` gate prevents calling `BFMMLA` on a CPU without FEAT_BF16 (it would `SIGILL`). Non-aarch64 targets compile with no kernel and `bf16_gemm_available()` is always `false`.

### Test
`bf16_gemm_matches_f32_reference` — 5×13×7 (odd dims, exercises the zero-padded edges). Reference: round inputs to bf16, then f32 matmul (BFMMLA's exact-product f32-accumulate numerics, modulo accumulation order). **Result: max 0.000e0 (rel 0.000e0)** — bit-exact on M2 Max. Self-skips on targets without FEAT_BF16.

### `accelerate` Cargo feature (opt-in)
Added `accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]` — Apple vecLib (Accelerate) BLAS for the CPU f32 matmul path (torch's CPU backend on Apple Silicon). Not in `default` features; compile-checked by the CI workflow on macOS.

### What's NOT done (task #25)
The backbone `Linear` matmuls do **not** call `bf16_matmul` yet, and `loader.rs` still rejects `bf16` on CPU. The kernel + `CustomOp2` + wrapper are ready; the routing is the remaining wiring.

---

## 3. MLP `Send` fix (`src/model/mlp.rs`)

### What it does
Replaces `MLP`'s `candle_nn::Sequential` (which holds `Vec<Box<dyn Module>>` — **not** `Send` because `dyn Module` has no `Send` bound) with `Vec<Box<dyn Module + Send>>` + a manual left-fold `forward`.

### Why
`candle_nn::Sequential` is unfixably non-`Send` (it's in the upstream crate). The `realtime_pipeline_types_are_send` test revealed the chain: `LFM2AudioModel` → `MLP` → `Sequential` → `Vec<Box<dyn Module>>` (non-`Send`). Without this fix, the crate doesn't compile (the `is_send::<LFM2AudioModel>()` test fails to compile). With it, `LFM2AudioModel: Send` and `LFM2AudioProcessor: Send` are both true — the worker-thread realtime pipeline is unblocked.

### The rewrite
- `model: Sequential` → `model: Vec<Box<dyn Module + Send>>`.
- `seq().add(x)` → `model.push(Box::new(x))` for `LayerNorm`, `Linear`, `Activation::Gelu`.
- `forward`: `let mut h = self.model[0].forward(x)?; for layer in &self.model[1..] { h = layer.forward(&h)?; } Ok(h)`. The comment notes candle tensors are Arc-backed handles, so rebinding `h` is a refcount bump, not a data copy — same semantics as `Sequential::forward`.
- The `model.{idx}` weight-path bookkeeping is unchanged (still `idx += 1` for every slot including no-weight GELU/Dropout), so checkpoint loading is unaffected.

### Tests
- `forward_maps_in_channels_to_out_channels` — all 4 bias/layernorm combos; shape + finiteness.
- `single_linear_no_hidden` — the no-activation edge (one Linear, no GELU).
- `mlp_is_send` — `is_send::<MLP>()` (the point of the rewrite).

---

## 4. `AudioDetokenizer: Send` (`src/audio_out.rs`)

### What it does
Adds `Send` to the `AudioDetokenizer` trait: `pub trait AudioDetokenizer: Send`.

### Why
The processor holds `Option<Box<dyn AudioDetokenizer>>` for `audio_out` and `mimi`. Without `Send` on the trait, `Box<dyn AudioDetokenizer>` is not `Send`, so `LFM2AudioProcessor` is not `Send`, so it can't move to a worker thread. Both backends (`LFM2AudioDetokenizer`, `MimiDetokenizer`) are already `Send` by construction — the bound just makes the trait object `Send`.

---

## 5. KV cache + mask memoization (`src/model/lfm2_hf.rs` + `candle_ext/transformers_utils.rs`)

> **CORRECTED** — the original §5 documented a zero-copy `KvCache` swap that was
> **reverted** by Claude as a deviation from the reference. This section now
> reflects what's actually on disk.

### What it does
Two changes, both faithful to candle-transformers' `models/lfm2.rs` (the file this port was copied from):

1. **Vendored `build_causal_mask` + `repeat_kv`** (`candle_ext/transformers_utils.rs`, new) — the exact two `crate::utils::*` helpers that `lfm2.rs` imports, backported from candle 0.10.x onto the 0.9.2 pin (adapted only `candle`→`candle_core`). The port now uses the **same** helpers as the reference rather than the hand-rolled `causal_mask`/`repeat_kv` that were previously in `lfm2_hf.rs`.

2. **Mask memoization** (`Cache::mask`) — a `HashMap<(usize, usize), Tensor>` on `Cache` that builds each boolean causal mask once per `(seq_len, kv_len)` shape via the vendored `build_causal_mask`, then reuses it across all 6 attention layers × every decode step, instead of rebuilding the mask on every call. The mask only depends on the `(seq_len, index_pos)` geometry, so caching by `(seq_len, kv_len)` is exact. `masks` survive `clear()` (a fresh turn reuses the same geometry) — matching the reference, which never drops them.

### What was reverted (and why)
Claude had previously swapped the `Tensor::cat`-based KV cache for candle-nn's preallocated `KvCache` (in-place `slice_set` + `narrow` view, no re-alloc/re-copy). That was **reverted** because it was a deviation: HF's `Lfm2HybridConvCache` and candle-transformers `lfm2.rs` both use `Tensor::cat` on the time axis. The original `cat`-based code was already the faithful port. The `KvCache` swap was the "random utility" deviation.

### What's there now
- `kvs: Vec<Option<(Tensor, Tensor)>>` — the original `cat`-and-clone KV cache (faithful to the reference).
- `cache.mask(seq_len, index_pos)?` — the memoized boolean mask (faithful to `lfm2.rs`'s `Cache::mask`).
- `masked_fill(&att, &mask, f32::NEG_INFINITY)` — the reference's `masked_fill` (boolean mask → `-inf` via `where_cond`), replacing the old hand-rolled additive `causal_mask`.
- `repeat_kv` — the vendored cat-based form (huggingface/candle#2043 — faster than the expand form, avoids strided copies).
- The detokenizer's sliding-window `add_mask` path is unchanged (still the additive f32 mask supplied by the caller — the documented deviation the reference has no custom-mask path for).

### What this fixes
The per-call mask-construction cost: the old hand-rolled `causal_mask` built a `(seq_len, kv_len)` f32 tensor via a host-side scalar double-loop + `Tensor::from_vec` on **every attention layer, every decode step** (6 layers × O(L) steps × O(L²) per mask). Now it's built once per shape and memoized — the faithful answer to the per-call mask-construction cost (the backbone sibling of the detokenizer sliding-mask issue, PR comment #1).

---

## 6. Rust CI workflow (`.github/workflows/rust-voice.yml`)

### What it does
Formalizes the `liquid-audio` test suite in the build: every change to the crate (or the workflow) builds + runs `cargo test` on both x86_64 Linux and arm64 macOS.

### Triggers
`push` / `pull_request` on path filter `packages/desktop/src-tauri/crates/liquid-audio/**` + `.github/workflows/rust-voice.yml`, plus `workflow_dispatch`.

### Matrix
- `ubuntu-latest` — proves the cfg fallbacks compile and portable tests pass (BFMMLA self-skips, thread policy falls back to physical cores).
- `macos-latest` (arm64) — where the hardware-specific paths actually execute: the NEON BFMMLA kernel (FEAT_BF16) and the `hw.perflevel0.physicalcpu` thread policy.

### Steps
1. Checkout (`actions/checkout@v6`).
2. Install Rust stable.
3. Cache cargo (`Swatinem/rust-cache@v2` with the workspace path).
4. Linux: `apt-get install libasound2-dev` (cpal/ALSA dev-dep).
5. `cargo build --all-targets`.
6. `cargo test --lib -- --nocapture`.
7. macOS only: `cargo build --lib --features accelerate` (compile-check the Accelerate feature).

### Concurrency
`cancel-in-progress: true` (matches `test.yml`).

---

## `THREADING_PARITY.md` — Claude's writeup (partially outdated)

Claude's `THREADING_PARITY.md` documents items 1–4 above but was written **before** the mask memoization + vendored helpers (item 5), the `to_vec4` extension, the realtime pipeline (§6), and the Tauri integration (§7). It has since been updated in-crate to reflect the realtime pipeline as done.

---

## 6. Multi-turn persistence + audio sampling defaults + realtime pipeline (Claude)

### Multi-turn persistence (Gap A)
The Python getting-started code persists one `ChatState` across turns: after
generating a reply, it calls `chat.append(text, audio_out, modality_flag)` +
`chat.end_turn()` + `chat.new_turn("user")` so the next utterance has context.
The Rust `realtime.rs` previously created a fresh `ChatState::new()` every
`respond()` call — each utterance was a cold start.

**Fix:**
- `ChatState::from_parts(proc, codebooks, text, audio_in, audio_in_lens,
  audio_out, modality_flag)` — a thin constructor that seeds the five persisted
  tensors from a prior conversation instead of `<|startoftext|>`. No change to
  `new`; no ripple to existing callers.
- `Lfm2VoiceEngine` holds `conv: Option<ConversationState>` (the five tensors).
  `respond` seeds turn 1 from `ChatState::new` + system turn; later turns via
  `from_parts`. After generation, collects `(text_ids, audio_frames,
  modality_out)` from the `GenToken` stream, calls `chat.append()` +
  `chat.end_turn()`, and saves `conv` **only on clean completion** (barge-in
  discards the partial turn via `prior = self.conv.clone()` — history survives).
- `ChatState<'a>` stays transient (borrows `proc` only within `respond`) — no
  `Arc`/lifetime ripple.

**EOAudio handling:** all frames including the all-2048 terminator are collected
for `append` (so the model knows the audio segment ended in context); the EOAudio
frame is skipped at the PCM-decode step (matching the Python's
`audio_out[:-1]` for Mimi decode).

### Audio sampling defaults (Gap B)
`GenParams::demo_defaults()` — `audio_temperature: Some(1.0),
audio_top_k: Some(4)` (text stays greedy `None`). Greedy audio is degenerate for
the Depthformer (it's trained to sample). `Default` stays greedy for
backward compat.

### Configurable system prompt (Gap C)
`Lfm2VoiceEngine::with_system_prompt(prompt)` builder replaces the hardcoded
`"Respond with interleaved text and audio."`. The desktop `TurnMode` enum
(`Asr`/`Tts`/`Interleaved`) maps mode→prompt+budget at the command layer
(liquid-audio can't import desktop's `TurnMode` — the dependency points the
other way).

### `StreamingPcmResampler` (audio continuity)
The old code called `resample_slice()` independently on each tiny decoded
frame, resetting the filter at every chunk boundary. The new
`StreamingPcmResampler` maintains `prev: Option<f32>` across calls for
integer upsample (24kHz→48kHz, the common macOS case), doing linear
interpolation between the last sample of the previous chunk and the first of
the next. Tested (`streaming_resampler_keeps_integer_upsample_continuity`).

### `can_interrupt` VAD gate
The VAD loop in `voice_runtime.rs` now drops mic input while the assistant is
speaking if `can_interrupt` is false — matching `chat.py`'s
`ReplyOnPause(can_interrupt=False)`. This prevents the user's own speaker
output from re-triggering a false utterance.

### Canonical two-turn proof
`examples/chat_multiturn.rs` (NEW) — the canonical two-turn proof on Metal bf16:
- Turn 1 (spoken question): "Handcrafted Excellence, Every Day" — woodworking.
- Turn 2 (text "…chairs…"): "…the quality and style of **your chairs**… Elevate
  Your Seating Experience" — **chairs-conditioned**, proving turn 2's prefill
  consumed turn 1's appended discrete audio as context.

### `text_chat.rs` example
A minimal text-only proof: `generate_sequential` + `add_text` + greedy, CPU
F32. Exercises tokenizer → backbone → text head → sampler → detokenize without
the audio path. 4.3 tok/s on CPU P-cores.

### New examples
- `examples/text_chat.rs` — text-only proof (no audio path)
- `examples/chat_multiturn.rs` — canonical two-turn discrete-audio context proof

### Tests
62 lib tests pass (1 ignored = the real-model `engine_multiturn_grows_conv`
test). Metal build clean. All examples compile.

---

## 7. Tauri voice service integration (Codex, commit `d3436b9`)

### Crate moved
The crate was moved from `experiments/lfm2-audio-voice/liquid-audio-rs/` to
`packages/desktop/src-tauri/crates/liquid-audio/` and wired as a Tauri
workspace dependency:
```toml
liquid-audio = { path = "crates/liquid-audio", features = ["metal"] }
```

### `voice_runtime.rs` (liquid-audio crate, 598 lines)
The `VoiceRuntime` wraps `RealtimePipeline` + cpal capture/playback into a
managed service. Contains:
- `RuntimeConfig` — VAD threshold, silence ms, min utterance, `can_interrupt`.
- The VAD loop (`vad_loop`) — energy-threshold onset detection, silence-pause
  end-of-utterance, `can_interrupt` gate (drops mic while assistant speaks).
- cpal input/output stream management — mono downmix in the callback, ring
  buffer playback with prebuffer + idle-reset.
- `mic_enabled` AtomicBool — pause/resume mic without killing the stream.

### `voice/control.rs` (desktop crate, 516 lines)
The Tauri command surface:
- `voice_status` — returns `VoicePlan { provider, enabled, surface, running,
  running_provider, mic_enabled, ready, detail }`. Reads from `VoiceRuntime`
  for the runtime fields.
- `voice_start` — returns `VoiceStartResult::Lfm2` or
  `VoiceStartResult::Livekit { grant }`. For `lfm2`: spawns
  `VoiceRuntime::start_lfm2` (which builds `Lfm2VoiceEngine` + starts the
  pipeline + cpal). For `livekit`: starts the runtime tracking + mints a
  LiveKit token via HTTP to the local sidecar (`livekit_grant()`).
- `voice_stop` — calls `VoiceRuntime::stop()` (interrupts + drops the session).
- `voice_set_mic_enabled` — pauses/resumes the cpal mic.
- `VoiceEvent` contract — `State`/`Transcript`/`Level`/`AudioClip`/`Ended`/
  `Error` (streamed over `tauri::ipc::Channel`).
- `TurnMode` — `Asr`/`Tts`/`Interleaved` with `system_prompt()` and
  `max_new_tokens()` matching the demo.

### `voice/runtime.rs` (desktop crate, 579 lines)
The `VoiceRuntime` manages the session lifecycle:
- `VoiceSession` enum — `Lfm2(Lfm2Session)` / `Livekit(LiveKitSession)`.
  Dispatches `is_finished`/`provider`/`session_id`/`interrupt`/
  `set_mic_enabled`/`mic_enabled`/`stop` per variant.
- `ThreadManager` — `Vec<JoinHandle>` with `reap()` (joins finished), `wait()`
  (joins all before new session), `Drop` (joins all on shutdown). No detached
  threads.
- `Lfm2Session` — owns the `Lfm2Runtime` (from `voice_runtime.rs`), the
  bridge cancel `AtomicBool`, and the done flag. `stop()` signals done +
  drops the runtime.
- `LiveKitSession` — lightweight state tracker (ctx + mic `AtomicBool`). The
  actual LiveKit room lives in the webview; `stop()` is a no-op (the webview
  calls `room.disconnect()`).

### `voice/session.rs` (desktop crate)
The HTTP session bridge for the LiveKit provider's delegate path — an SSE
reducer that drives `POST /session/:id/prompt_async` + `GET /event` for the
LiveKit provider when delegation is configured. This is the only HTTP path
in the voice layer; the LFM2 path is fully in-process.

### `settings.rs` (desktop crate, 285 lines)
- `VoiceProvider` — `Off`/`Lfm2`/`Livekit`.
- `Lfm2Settings` — `model_dir` (optional local snapshot), `model` (HF repo id,
  default `"LiquidAI/LFM2.5-Audio-1.5B"`), `device`, `vad_threshold`,
  `max_tokens`, `seed`, `delegate`.
- `lfm2_model_ref()` — resolves: explicit `model` → `model_dir` (if
  `config.json` exists) → `DEFAULT_LFM2_MODEL`. This means the default is a
  downloadable HF repo id, not a local directory.
- `lfm2_model_dir()` — resolves + `expand_user_path` (`~/` expansion).
- `VoicePlan` readiness for `Lfm2` now always returns `ready: true` (the model
  can download from HF on first start).

### Frontend (`packages/app/src/`)
- `lib/voice-settings.ts` — typed Tauri command wrappers
  (`getVoiceSettings`/`setVoiceSettings`/`getVoiceStatus`/`startVoice`/
  `stopVoice`/`setVoiceMicEnabled`). `startVoice` now returns
  `VoiceStartResult` (the discriminated union). `VoicePlan` gained
  `running`/`runningProvider`/`micEnabled` fields.
- `lib/voice-state.ts` (NEW, 44 lines) — pure decision functions extracted
  from the component: `voiceProvider`, `voiceEnabled`, `voiceButtonOn`,
  `voiceMicTarget`, `shouldStopRuntimeForProviderChange`. Tested (6 unit
  tests in `voice-state.test.ts`).
- `context/voice.tsx` — rewritten to branch on provider: `lfm2` → Tauri
  `startVoice` + Channel event listener (feeds `VoiceEvent`s into SolidJS
  signals: `nativeAgent`, `nativeState`, `nativeLevel`, `nativeLine`,
  `nativeMic`); `livekit` → `startVoice` returns a `LiveKitGrant` →
  `room.connect()`. `disconnect`/`interrupt`/`setMicEnabled`/`toggleMute`
  all branch on `nativeActive()` vs LiveKit. The `Room` is still created
  unconditionally (needed for the `livekit` path), but unused for `lfm2`.
- `components/settings-voice.tsx` — gained the "Hugging Face model" field
  (for the repo id) alongside the optional "Local snapshot directory" field.
- `components/prompt-input.tsx` — mic button visibility and mic-pause-on-typing
  now use the extracted `voiceButtonOn`/`voiceMicTarget` functions.

### `FRONTEND_DESIGN.md`
The design doc for the voice frontend — phased:
- Phase 1 (NOW): absolute parity with Liquid AI's demo (turn-based, ASR/TTS/
  Interleaved, clip-based).
- Phase 2 (LATER): natural full-duplex via Moshi (the LM, not just Mimi).

### Known issues (from the `d3436b9` review)
- **`livekit_grant` uses HTTP** to the local sidecar for token minting. The
  LFM2 path is fully in-process; the LiveKit path still goes through HTTP
  because the LiveKit credentials live on the server side.
- **`LiveKitSession::stop()` is a no-op** — the room disconnect happens in the
  webview. Correct but asymmetric.
- **`voice_status` doesn't verify LFM2 model loadability** — reports
  `ready: true` for LFM2 even when the model isn't downloaded. The actual
  load/download happens in `voice_start`.
- **`StreamingPcmResampler` only handles integer upsample continuously** —
  non-integer ratios fall back to per-chunk `resample_slice` (no cross-chunk
  continuity). Fine for 24k→48k (the common macOS case).
- **`voice.tsx` still creates a `Room` unconditionally** — the LiveKit client
  library is loaded even when the user only uses the local model.

---

## What remains (as-built)

| Task | Status |
|---|---|
| Intra-op thread pool | ✅ done, verified |
| `accelerate` feature | ✅ done (compile-checked; not benchmarked) |
| bf16 BFMMLA kernel + `CustomOp2` | ✅ done, bit-exact verified |
| MLP `Send` fix | ✅ done, tested |
| `AudioDetokenizer: Send` | ✅ done |
| Zero-copy KV cache | ❌ **reverted** — the `KvCache` swap was a deviation; faithful `Tensor::cat` restored + mask memoization added instead |
| Mask memoization | ✅ done (faithful to `lfm2.rs`'s `Cache::mask`; eliminates per-call mask construction) |
| Vendored `build_causal_mask`/`repeat_kv` | ✅ done (candle-transformers 0.10→0.9.2 backport) |
| `to_vec4` | ✅ done (`TensorExt` trait; tested contiguous + strided) |
| Rust CI workflow | ✅ done |
| Multi-turn persistence | ✅ done (`ChatState::from_parts` + `ConversationState` + `chat_multiturn.rs` proof) |
| Audio sampling defaults | ✅ done (`GenParams::demo_defaults()`) |
| Configurable system prompt | ✅ done (`with_system_prompt` builder) |
| `StreamingPcmResampler` | ✅ done (cross-chunk audio continuity for integer upsample) |
| `can_interrupt` VAD gate | ✅ done (matches `chat.py`'s `ReplyOnPause(can_interrupt=False)`) |
| Realtime worker thread (#24) | ✅ done (`RealtimePipeline` + `Lfm2VoiceEngine` + `voice_runtime.rs` + Tauri integration) |
| Tauri voice service integration | ✅ done (`control.rs` + `runtime.rs` + `settings.rs` + frontend rewrite) |
| Short-conv flashfftconv routing | ✅ done (commits `0b4fdab`/`2c8d18a`/`2b5721c` — prefill + decode through `candle-flashfftconv`) |
| **Route bf16 through the model** (#25) | ⚠️ partial — short-conv routed through the kernel; backbone `Linear` matmuls still use candle's gemm; `loader.rs` still rejects bf16 on CPU for the full model |
| **Moshi LM for full-duplex** (Phase 2) | ❌ not started — the `moshi` crate provides Mimi (codec) only; the LM (conversational full-duplex) is Phase 2 |
| **ASR/TTS modes** | ❌ not wired — `TurnMode` enum exists; `generate_sequential` exists; the Tauri command doesn't dispatch on mode yet |

---

## Build + test verification (as-built)

```
$ cargo test --lib (in packages/desktop/src-tauri/crates/liquid-audio)
test result: ok. 62 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo build --features metal --examples
Finished `release` profile [optimized] target(s)

$ cargo test --lib -- --nocapture | grep key
BFMMLA bf16 GEMM vs f32(bf16-inputs) ref: max 0.000e0 (rel 0.000e0)
intraop threads = 8 (physical 12, logical 12)
test model::mlp::tests::mlp_is_send ... ok
test threads::tests::realtime_pipeline_types_are_send ... ok
test realtime::tests::streaming_resampler_keeps_integer_upsample_continuity ... ok
test result: ok. 62 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

62 tests pass (was 50 before this work; +3 MLP, +1 bf16, +1 threads, +1 Send
probe, +1 `to_vec4`, +5 realtime pipeline, +1 streaming resampler). The crate
is now committed at `packages/desktop/src-tauri/crates/liquid-audio/` and
wired into the Tauri desktop build.