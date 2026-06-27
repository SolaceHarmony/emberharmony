# Threading & concurrency — Python (torch + liquid_audio + moshi) ↔ Rust

> **Rust-side companion:** `liquid-audio-rs/THREADING_PARITY.md` (the implementer's view +
> verification). This doc is the reference-side dissection: what torch and the Python demos
> actually do, and where the Rust port matches, differs, or deliberately improves.

Concurrency lives at **four** distinct layers. Conflating them is the usual source of
"why is the Rust slower / different" confusion, so they are separated here.

---

## Layer A — Intra-op (one tensor op spread across cores)

A single `matmul`/`conv` is parallelized internally across worker threads.

| | Mechanism | Pool size |
|---|---|---|
| **torch** | ATen `at::parallel_for` over a global intra-op pool | `intraop_default_num_threads()` (`ParallelCommon.cpp`): `OMP_NUM_THREADS`→`MKL_NUM_THREADS`, else `TaskThreadPoolBase::defaultNumThreads()` ⇒ on Apple Silicon **`hw.perflevel0.physicalcpu` (P-cores only)** |
| **Rust/candle** | the `gemm` crate + candle kernels call **rayon's global pool** | default `num_cpus::get()` = **all logical cores** (incl. slow E-cores) |

**Parity: ACHIEVED.** `liquid-audio-rs/src/threads.rs` (`configure_intraop_threads`)
replicates torch's policy exactly and installs it as rayon's global pool, so candle + `gemm`
inherit it. Verified on M2 Max: candle would pick 12 (all logical); we pick **8** (P-cores),
byte-matching `sysctl hw.perflevel0.physicalcpu`. This is the single biggest compute-parity
lever — it affects **every** matmul.

## Layer B — Inter-op (independent ops/subgraphs in parallel)

| | Mechanism |
|---|---|
| **torch** | a **separate** inter-op pool (`at::launch`, `torch.set_num_interop_threads`, default = #CPUs), used when a graph forks independent branches (e.g. `torch.jit` fork, parallel module legs) |
| **Rust/candle** | no inter-op-pool concept; candle is eager and sequential |

**Parity: N/A — no gap.** The LFM2-Audio forward is a **sequential** eager graph (embed →
*N* decoder layers → norm → heads); it never forks independent ops, so torch's inter-op pool
is idle for this model. Nothing to replicate. (Pipeline-level overlap — gen vs playback — is a
*different* concern handled in Layer D, not by an inter-op pool.)

## Layer C — The GIL (Python-only) ⇒ Rust is strictly more parallel

This is the most important *conceptual* difference and the easiest to get wrong.

- **Python**: the **GIL** serializes Python bytecode — two Python threads never run bytecode
  at once. torch sidesteps it by **releasing the GIL inside its C++ dispatched ops** (the
  kernel runs GIL-free). So in `chat.py`, the producer `Thread` and the main relay thread
  overlap **only while the producer is inside a GIL-releasing torch op** (the `generate_*`
  compute). Between ops — Python glue, queue puts, token bookkeeping — they serialize.
- **Rust**: **no GIL.** The inference worker thread, the consumer, and the cpal audio
  callbacks run in **true parallel at all times**, not just during "C++ ops."

**Consequence:** the gen↔play overlap that Python achieves *conditionally* (only during
GIL-released kernels) is **unconditional** in the Rust port. Faithful in spirit, stronger in
practice. Any "Python does X with threads" reasoning that implicitly relies on the GIL does
**not** transfer 1:1 — Rust has more real concurrency, not less.

## Layer D — Pipeline / application threading (the producer/consumer structure)

Three reference shapes, two Rust shapes:

### Python · `demo/chat.py` (the LFM2-Audio demo) — producer Thread + Queue
`chat_response` spawns `chat_producer` on a `threading.Thread`; it owns the model + Mimi,
runs the sync `generate_interleaved` generator, decodes each audio frame to PCM, and
`q.put()`s tokens **and** PCM onto a `queue.Queue`. The main thread drains the queue and
relays text to the UI + PCM to fastrtc/WebRTC. **Half-duplex** (`ReplyOnPause`,
`can_interrupt=False`) — the mic is gated during a reply; **no barge-in**. Overlap of
gen+relay is the Layer-C GIL-release effect.

### Python · `moshi/server.py` (the Moshi model) — 3 asyncio coroutines, 1 loop
`asyncio.gather(opus_loop(), recv_loop(), send_loop())` on a **single event loop**:
- `recv_loop` — WebSocket → opus reader buffer.
- `opus_loop` — the inference loop: `mimi.encode` → `lm_gen.step` (per code) → `mimi.decode`
  → opus writer. **Runs inline and blocks the event loop during compute**, yielding only at
  `await asyncio.sleep(0.001)` / `await ws.send_bytes`.
- `send_loop` — opus writer → WebSocket.

So this is **cooperative** concurrency, *not* thread-level parallelism — the only real
parallelism is Layer-A intra-op inside the torch ops. It is **continuously full-duplex**
because the **Moshi model architecture** processes input and output streams every frame
(barge-in is implicit in the model, not the plumbing). Note this serves *Moshi*, not
LFM2-Audio — LFM2-Audio's shipped demo is `chat.py` above.

### Rust · `examples/mic_chat.rs` — single thread + callback (the older path)
`generate_interleaved` runs **inline on main**; its callback decodes audio → `Arc<Mutex<
VecDeque>>` ring; cpal callback threads do I/O. Gen overlaps playback (different threads), but
gen **blocks main** and the mic is dropped during generation — **half-duplex**, no
producer/consumer split. This is the closest analog to `chat.py` *minus* the producer thread.

### Rust · `src/realtime.rs` + `examples/duplex_chat.rs` — worker thread + channels (new)
A **persistent inference worker thread** (`RealtimePipeline`) **owns** the model + processor +
detokenizer (`VoiceEngine` trait; real impl `Lfm2VoiceEngine`) and loops `recv Utterance →
respond (emit text + decode audio → emit PCM) → TurnComplete`, over **`crossbeam-channel`**.
The consumer drains `VoiceEvent`s → cpal ring + stdout. **True parallel** (worker ‖ consumer ‖
cpal callbacks, no GIL), **full-duplex** (live mic), with **explicit barge-in**: an
`AtomicBool` that `generate_interleaved_cancellable` polls each decode step to abort the reply
mid-stream (`chat.py` cannot do this; it is a turn-based take on Moshi's interruptibility).

### Mapping

| Aspect | `chat.py` | `moshi/server.py` | Rust `realtime.rs` |
|---|---|---|---|
| concurrency unit | producer `Thread` + main | asyncio coroutines (1 loop) | worker + consumer + cpal **OS threads** |
| real parallelism | GIL-released overlap only | none at coroutine level | **always** (no GIL) |
| gen↔play decouple | `queue.Queue` (unbounded) | shared `sphn` opus buffers | `crossbeam` channel (unbounded) |
| duplex | half (`ReplyOnPause`) | full (continuous frames) | full (live mic + VAD) |
| barge-in | **none** | implicit (model arch) | **explicit** `AtomicBool` / `*_cancellable` |
| streaming state | `mimi.streaming(1)` | `streaming_forever(1)` | `reset_stream()` + `decode_step()` |
| audio I/O | fastrtc/WebRTC threads | WebSocket on loop + `sphn` (Rust) | cpal real-time callback threads |

## Layer E — Audio I/O callback threads

Real-time audio callbacks run on their own OS threads in **every** stack (PortAudio/fastrtc in
Python via the OS audio HAL; cpal in Rust). This layer was never a parity question — both
sides hand the device a ring/buffer and let the HAL thread drain it. The Rust ring is
`Arc<Mutex<VecDeque<f32>>>`; barge-in flushes it so playback stops the instant the user speaks.

---

## Net: where we stand vs torch/Python

| Layer | Verdict |
|---|---|
| A · intra-op pool | **Parity** — `threads.rs` matches torch's P-core policy (verified). |
| B · inter-op pool | **N/A** — LFM2-Audio's sequential graph uses none; no gap. |
| C · GIL | **Rust is more parallel** — overlap is unconditional, not GIL-gated. |
| D · pipeline | **Matched + extended** — `realtime.rs` = `chat.py`'s producer/consumer (worker + channel) **plus** full-duplex + explicit barge-in. |
| E · audio I/O | **Equivalent** — HAL callback threads + a ring, both sides. |

**The one true structural difference that remains is a model difference, not a threading
deficiency:** Moshi's server is *continuously* full-duplex (it encodes input and emits output
every frame because the Moshi architecture is inherently duplex), whereas LFM2-Audio is
**turn-based interleaved** (utterance → interleaved text/audio reply). The Rust pipeline is
faithful to LFM2-Audio's turn model and layers explicit barge-in on top — it does not (and
should not) impose Moshi's continuous-duplex frame loop on a turn-based model.

## Sources dissected
- torch: `aten/src/ATen/ParallelCommon.cpp` (`intraop_default_num_threads`), `at::launch` /
  `set_num_interop_threads` (inter-op pool), pybind GIL-release on dispatched ops.
- `upstream-liquid-audio/src/liquid_audio/demo/chat.py` (`chat_producer`/`chat_response`).
- `upstream-liquid-audio/src/liquid_audio/moshi/server.py` (`handle_chat`'s 3 coroutines).
- Rust: `src/threads.rs`, `src/realtime.rs`, `src/model/lfm2_audio.rs`
  (`generate_interleaved_cancellable`), `examples/mic_chat.rs`, `examples/duplex_chat.rs`.
