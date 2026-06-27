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

## 4. Realtime pipeline threading — DESIGN (remaining, task #24)

**Python**: `demo/chat.py` runs the sync generator on a **producer `Thread`** → `queue.Queue`
→ main relays to WebRTC playback (overlap gen+play; half-duplex via `ReplyOnPause`,
`can_interrupt=False`). `moshi/server.py` is true full-duplex: 3 asyncio coroutines
(recv/inference/send) + PortAudio callback threads.

**Rust now**: `mic_chat.rs` runs `generate_interleaved` on **main**; cpal callback threads do
I/O; `Arc<Mutex<VecDeque>>` ring for playback (gen overlaps play, but gen blocks main and the
mic is dropped during generation — half-duplex).

**Target**: a **persistent inference worker thread** (moshi's inference coroutine) that *owns*
`model` + `mimi` and loops `recv utterance → prefill → generate (decode → send PCM)`; the
capture stream stays **live** (full-duplex); barge-in via an `AtomicBool` the generate loop
checks (needs the callback to return `ControlFlow`). Channels: `crossbeam-channel`.
**Constraint:** the worker must *own* the model/decoder because `MimiDetokenizer` holds a
`RefCell` (`!Sync`) — share nothing by `&` across threads; move them in, talk over channels.
`AudioDetokenizer` then needs a `+ Send` bound. This belongs in the (not-yet-built) native
voice loop (`voice_start`) more than the CLI example.

## Files
`src/threads.rs`, `src/bf16_gemm.rs`, `csrc/bf16_gemm.c`, `build.rs`, `Cargo.toml`
(`rayon`/`num_cpus`/`libc`/`half` deps, `cc` build-dep, `accelerate` feature),
`src/loader.rs` (calls `configure_intraop_threads`; precise bf16 note), `src/lib.rs`.
