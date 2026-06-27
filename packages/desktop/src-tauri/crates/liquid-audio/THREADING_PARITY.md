# Threading & compute parity with torch

Goal: stop "making it similar" and actually match torch's execution model — intra-op
thread policy, the CPU matmul backend, and the realtime pipeline — reading torch's source
where needed (we don't link libtorch's C++ ABI, so we replicate it).

## 1. Intra-op thread pool — DONE, verified

**torch** (`aten/src/ATen/ParallelCommon.cpp` `intraop_default_num_threads()`): honours
`OMP_NUM_THREADS` then `MKL_NUM_THREADS`, else `TaskThreadPoolBase::defaultNumThreads()`,
which **on Apple Silicon queries `hw.perflevel0.physicalcpu` — the performance cores only**
(excludes the efficiency cores).

**candle** (`utils.rs::get_num_threads`): `RAYON_NUM_THREADS` else `num_cpus::get()` — **all
logical cores**, i.e. it schedules compute-bound matmul (`gemm`, rayon) onto the slow E-cores
that torch deliberately avoids → different pool, worse throughput/tail latency.

**Fix** (`src/threads.rs`): `intraop_default_num_threads()` replicates torch's policy exactly
(`OMP`/`MKL`/`RAYON` env → `hw.perflevel0.physicalcpu` → `hw.physicalcpu` → `num_cpus::get_physical()`),
and `configure_intraop_threads()` installs it as rayon's **global** pool (candle + `gemm`
inherit it). Called once at the top of `from_pretrained`.

Verified on **M2 Max** (8 P-cores + 4 E-cores = 12 logical): candle would use **12**; we now
use **8** — byte-matching `sysctl hw.perflevel0.physicalcpu`. (`threads::tests`.)

## 2. CPU matmul backend — DONE

torch's CPU backend on Apple Silicon is Accelerate/vecLib BLAS (multi-threaded, tuned).
Added an opt-in **`accelerate` feature** (`candle-core/accelerate`,`candle-nn/accelerate`)
so CPU-mode `sgemm`/`dgemm` use Apple BLAS instead of candle's pure-Rust `gemm`. Builds clean.

## 3. bf16 CPU matmul — DONE (kernel), verified bit-exact

candle 0.9.2's CPU matmul allowlist is **`F16 | F32 | F64`** (`cpu_backend/mod.rs:1368`;
Accelerate path F32/F64) — **bf16 → `UnsupportedDTypeForOp`**, so the loader forced f32 on
CPU. (candle *has* bf16 everywhere else — dtype, every elementwise op, conversions, Metal
matmul; the gap is only this CPU-gemm allowlist, and `gemm-f16` already handles the `half`
types, so it's a candle choice, not a `gemm` limit.)

The M2 Max has **FEAT_BF16** (`sysctl hw.optional.arm.FEAT_BF16 = 1`) → `BFMMLA`. So we wrote
a real kernel instead of falling back to f32:
- **`csrc/bf16_gemm.c`** — a NEON `vbfmmlaq_f32` micro-kernel: packs A/B into BFMMLA tile
  order (2×4 · 4×2 → 2×2), **bf16 inputs, f32 accumulate** (torch's bf16-matmul numerics),
  zero-padded edges for M%2/N%2/K%4. Compiled by **`build.rs`** (`cc`, `-march=armv8.2-a+bf16`),
  gated to aarch64 via `cfg(has_bf16_kernel)`.
- **`src/bf16_gemm.rs`** — FFI + **runtime FEAT_BF16 gate** (`has_feat_bf16()` via sysctl; a
  binary stays portable, BFMMLA `SIGILL`s without it) + `Bf16Gemm` **`CustomOp2`** (the single
  FFI site, composes as a candle tensor op) + `bf16_matmul(&a,&b) -> Option<Tensor>` wrapper.
- **Verified**: `bf16_gemm_matches_f32_reference` → **max 0.000e0 (rel 0.000e0)** vs the f32
  reference (bf16-rounded inputs, f32 matmul) on 5×13×7 (exercises the padded edges).

**Remaining (task #25):** route the backbone `Linear` matmuls through `Bf16Gemm`/`bf16_matmul`
when `device==CPU && dtype==bf16`, and relax `loader.rs`'s bf16-on-CPU rejection — then the
model runs **bf16 natively on CPU** (Metal already does). The kernel + op are ready; this is
the wiring.

**Caution — this is a decision, not just wiring.** `loader.rs` *deliberately* runs f32 on CPU
(`:108`, `:210`): the bf16→f32 weight upcast is lossless and f32 is the parity reference; bf16
is the Metal/real-time path. Wiring bf16 through CPU means replacing `candle_nn::Linear` at
dozens of sites across the backbone, depthformer, conformer, and detokenizer — a model-wide
change to a hot path whose parity is verified at **6.558e-6**. So #25 must be done **with the
model in the loop** (run `mic_chat`/the parity harness and confirm the numerics) — not as a
blind sweep — and only if bf16-on-CPU is actually wanted over the faithful f32 path.

## 4. Realtime pipeline threading — DONE (worker pipeline + barge-in), task #24

**Python**: `demo/chat.py` runs the sync generator on a **producer `Thread`** → `queue.Queue`
→ main relays to WebRTC playback (overlap gen+play; half-duplex via `ReplyOnPause`,
`can_interrupt=False`). `moshi/server.py` is true full-duplex: 3 asyncio coroutines
(recv/inference/send) + PortAudio callback threads.

**Was**: `mic_chat.rs` ran `generate_interleaved` on **main**; cpal callback threads did I/O;
`Arc<Mutex<VecDeque>>` ring for playback (gen overlapped play, but gen blocked main and the
mic was dropped during generation — half-duplex).

**Now** (`src/realtime.rs`): a **persistent inference worker thread** ([`RealtimePipeline`])
*owns* the model + processor + detokenizer (the [`VoiceEngine`] trait; real impl
[`Lfm2VoiceEngine`]) and loops `recv `[`Utterance`]` → respond (emit text + decode audio →
emit PCM) → TurnComplete`, talking to the consumer over **`crossbeam-channel`** ([`VoiceEvent`]
stream). Because the model lives off the main thread, capture/playback stay live (full-duplex).
**Barge-in** is an `AtomicBool` the generate loop polls — `generate_interleaved_cancellable`
(`lfm2_audio.rs`) breaks the decode loop the moment `cancel` is set, so an interrupting
utterance aborts the in-flight reply instead of running to `max_new_tokens`. The worker can own
the model because `LFM2AudioModel`/`LFM2AudioProcessor` are now `Send` (the MLP + `AudioDetokenizer`
`Send` fixes); nothing is shared by `&` across threads. `Drop` closes the channel and joins.

The threading is **unit-tested with a fake engine** (`realtime::tests`): event ordering,
worker persistence across turns, barge-in aborts the in-flight turn, engine errors are reported
without killing the worker, and `Drop` joins + drops the engine. End-to-end full-duplex (live
cpal capture, VAD-driven utterance boundaries, voice-onset-during-reply ⇒ barge-in + flush) is
the **`duplex_chat`** example. (No acoustic echo cancellation yet — the assistant's own audio
can re-trigger the mic VAD; headphones / higher `LFM_VAD_THRESHOLD` mitigate.)

## 5. Python ↔ Rust threading-model comparison

Full dissection in **`glm-version/threading.md`** (torch + `chat.py` + `moshi/server.py` vs the
Rust port). Concurrency separates into **four layers** — keeping them apart is what prevents
"why is Rust different" confusion:

| Layer | torch / Python | Rust port | Verdict |
|---|---|---|---|
| **A · intra-op** (one matmul across cores) | ATen pool, P-cores (`intraop_default_num_threads`) | rayon global pool, sized by `threads.rs` | **parity** (verified 8 P-cores) |
| **B · inter-op** (independent ops in parallel) | separate `at::launch` pool (default #CPUs) | none (candle is eager-sequential) | **N/A** — LFM2's graph is sequential, torch's pool is idle too |
| **C · GIL** | GIL serializes Python; torch releases it inside C++ ops ⇒ threads overlap **only during** kernels | **no GIL** — worker ‖ consumer ‖ cpal callbacks overlap **always** | **Rust is more parallel** (overlap is unconditional) |
| **D · pipeline** | `chat.py`: producer `Thread`+`queue.Queue`, half-duplex, no barge-in · `moshi/server.py`: 3 asyncio coroutines, 1 loop, cooperative, continuous-duplex | `realtime.rs`: worker thread owns model, `crossbeam` channels, **full-duplex + explicit `AtomicBool` barge-in** (`*_cancellable`) | **matched + extended** |
| **E · audio I/O** | PortAudio/fastrtc/WebSocket callback threads | cpal real-time callback threads + ring | **equivalent** |

The one remaining *structural* difference is a **model** difference, not a threading gap:
Moshi's server is *continuously* full-duplex (its architecture processes in/out every frame),
while LFM2-Audio is **turn-based interleaved** — the Rust pipeline is faithful to LFM2's turn
model and adds explicit barge-in, rather than imposing Moshi's frame loop on it.

## Files
`src/threads.rs`, `src/bf16_gemm.rs`, `csrc/bf16_gemm.c`, `build.rs`, `src/realtime.rs`,
`examples/duplex_chat.rs`, `Cargo.toml` (`rayon`/`num_cpus`/`libc`/`half`/`crossbeam-channel`
deps, `cc` build-dep, `accelerate` feature), `src/model/lfm2_audio.rs`
(`generate_interleaved_cancellable`), `src/loader.rs` (calls `configure_intraop_threads`;
precise bf16 note), `src/lib.rs`.
