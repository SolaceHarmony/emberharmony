# 14 — Whole-Chain Coroutine-Driven, Zero-Copy, Zero-Wait Inference

Status: **authoritative target, with the LFM2 ownership cutover implemented.**
The working tree now has the immutable combined main+codec image, exact typed
views, native frontend/Conformer/backbone/Depthformer/Mimi, native tokenizer and
recurrence, per-conversation state/rollover, fixed PCM leases, reliable ticketed
events, interruption epochs, and expected-value parking. Two follow-ons remain
explicit: the engine still advances passes through a synchronous native
coordinator over its single request slot rather than a capacity-2 completion
continuation queue, and the physical mic/speaker adapter still bridges the native
dock into the legacy Rust `VoiceEvent` surface. The full Moshi port and physical
kcoro device dock are subsequent tranches; neither permits a Candle fallback.

This is the convergence target for specs 02, 03, 07, and 10 —
the picture they are each a slice of. It describes the end state where the entire
inference chain, microphone PCM to speaker PCM, runs as native passes on the
fixed Flashkern lane team, clocked by completion doorbells rather than a Rust
loop, over one zero-copy weight pool.

The load-bearing observation: **the substrate for this already exists.** Flashkern
is already a GPU-threadgroup engine — a fixed P-core lane team, generation-fence
barriers, atomic tile-claim, one dispatcher, expected-value doorbells, no spin
tier. The SQ/CQ bridge with descriptor leases exists. The safetensors loader
already demonstrates the required one-ingress-write discipline: a byte-exact
resident image with immutable views. Mimi's private folded arena is not the
model-image precedent and must not be generalized into a second weight pool.
At the design baseline, what was missing was not primitives but three ownership
cuts (retained here as historical rationale):

1. Rust drove recurrence by **blocking** on a **single-slot** pass.
2. The graph was **Candle above the assembly leaves** (prefill, modality
   scatter, the token/frame loop, KV ownership).
3. The model was **resident twice** — a byte-exact native image *and* a ~2.94 GB
   Candle copy that the backbone/depthformer passes actually ran off of.

This document plans the collapse of all three.

---

## 0. The north star (job 1)

**Divorce Rust from inference entirely — no math, no memory allocation, no
threading — and move all of it into Flashkern, where it belongs.** That is job 1;
everything else is downstream of it.

After the migration, Rust does exactly two things with audio, and nothing else on
the inference path:

- **grab PCM from the microphone**, and
- **grab PCM from the model.**

There is no `.wav` generation anywhere, ever, as a principle — PCM is a live
stream, not a rendered file. (A `.wav` render is the tell of a TTS mindset; this
is an interleaved real-time model, not a text-to-speech renderer.)

And the Rust audio dock uses **kcoro-rs** (`crates/kcoro`) — the same
non-blocking, park-on-wake mechanism as the native layers — so that **Rust std
channels are dumped entirely** at that layer. No `mpsc`, no `crossbeam`, no
polling loop. The dock's rings and promises wake on the same expected-value
substrate the lanes use. (Today the voice runtime holds ~57 std-channel sites;
those are the debt this retires.)

---

## 1. Requirements

### 1.1 Functional

- Every stage of the chain runs as a lane-uniform native pass on the resident
  lane team: resample, mel, Conformer, adapter, prefill + modality scatter,
  backbone prefill, backbone decode, text sampling, Depthformer, Mimi decode.
- **Recurrence is native.** The token/frame loop is a native session state
  machine advanced by completion continuations: a pass completion enqueues the
  next pass directly, without a host round-trip. This is the "device recurrence"
  row of the Flashkern GPU-equivalence table, realized.
- **Rust's only production roles** are the two docks and the observer:
  - dock microphone PCM in as a borrowed descriptor lease,
  - drain speaker PCM out of a borrowed descriptor lease,
  - submit control tickets (start turn, interrupt, configure),
  - observe transcript/telemetry on a lossy side channel.
  Rust never blocks on a numerical completion and never owns model state.
- Multiple conversations share one model image and one lane team; completions
  route to the correct session by conversation id and epoch.

### 1.2 Non-functional

- **Zero-copy after ingress.** Weights are bound from the single byte-exact
  resident image in checkpoint-native `(N,K)` bf16; tensor starts may be
  unaligned and kernels must accommodate that. The Candle duplicate is deleted.
  Activations live in engine-owned scratch planes and descriptor-leased rings and
  are passed between passes by pointer. No stage materializes a `Tensor`.
- **Zero-wait.** No polling, no bounded spin, no host thread blocked on the
  progress path. Every wait is an expected-value doorbell. The idle lane team
  stays under the existing `< 0.1%` CPU gate (`engine_idle_zero_spin`).
- **Real-time.** The pipeline overlaps stages, so wall-clock is the critical path,
  not the sum of stages. Per-frame Mimi decode (~14 ms) must keep pace with
  playback; backpressure is a doorbell park on the speaker dock, never a sleep.
- **Faithful numerics.** bf16 bit-matched to the captured fixtures across every
  ported stage; `-ffp-contract=off`; Accelerate/AMX permitted for matmul-shaped
  stages on Apple. Seeded CSPRNG native; fixed-seed byte-identity per turn.

### 1.3 Constraints

- **No tensors in the production data plane** — buffers with pointers. Candle
  survives only as an *offline* capture/parity oracle, never wired into the
  shipped path.
- **No math, memory allocation, or threading in Rust for inference.** All three
  belong to Flashkern (see §0). Rust owns only the two PCM docks and control.
- **No `.wav` generation, ever.** PCM is a live stream out of the model; there is
  no file-render step on the audio path.
- **Rust channels are dumped** at the audio dock in favour of kcoro-rs rings /
  promises, which park on the native wait-word substrate.
- **No fallback chains.** A native gate failure is a terminal completion with a
  cause, not a silent drop to Candle. (This is a real sequencing constraint —
  see Trade-off 4.)
- Rust host; C++ owns plans, sessions, and recurrence; assembly owns all math.
- Target hardware: M2 Max — 8 performance-core lanes (E-cores excluded by
  policy), 400 GB/s, bandwidth-bound at decode.

---

## 2. High-level design

The whole chain is **one native session state machine that submits a sequence of
passes to the existing lane team, where each pass's completion continuation
chooses the next pass.** A native dataflow graph clocked by doorbells.

### 2.1 Three planes (do not merge them)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PLANE 1 — native model SQ/CQ (compute)                              │
  │   fixed lane team · pass descriptors · generation fences · doorbells │
  │   THE progress path. exact-once completions. no host on it.          │
  ├─────────────────────────────────────────────────────────────────────┤
  │ PLANE 2 — PCM / control dock (I/O)                                   │
  │   mic lease in · speaker lease out · control tickets                 │
  │   Rust lives here, on kcoro-rs rings/promises — NOT std channels.    │
  │   borrowed descriptor regions. zero-copy. park-on-wake.              │
  ├─────────────────────────────────────────────────────────────────────┤
  │ PLANE 3 — reliable events + lossy observer (two sub-planes)         │
  │   text/transcript events: RELIABLE, ticketed, exactly-once.          │
  │   telemetry + waveform-viz ONLY: lossy, coalescible, sampled.        │
  │   neither drives numerical progress; the reliable half must not drop.│
  └─────────────────────────────────────────────────────────────────────┘
```

Text and transcript are **not** telemetry: a dropped token is a corrupted
conversation, so text/transcript events ride a reliable ticketed channel
(exactly-once delivery, like the completion path). Only sampled telemetry and
waveform visualization may be lossy/coalescible.

### 2.2 The shift, in one picture

**Today — Rust drives, blocking, one slot:**

```
Rust generate_with_cache loop  (holds a thread the entire turn):
  loop over tokens:
    pass_lock.lock()
    submit_pass(TOKEN_PASS) ─▶ [lane team] ─▶ CQ ─▶ unblock   ← thread blocked
    submit_pass(DEPTH_FRAME)─▶ [lane team] ─▶ CQ ─▶ unblock
    mimi decode_step ───────▶ [lane team] ─▶ CQ ─▶ unblock
    (Rust owns: cursor, KV cache, sampling loop, Candle prefill + scatter)
  SQ capacity = 1 · one pass in flight · no overlap · no native recurrence
```

**Target — native recurrence, continuation-driven:**

```
Rust: submit TURN ticket  (borrowed mic PCM lease) ──┐
                                                      ▼
                          ┌──────────────────────────────────────────┐
                          │        NATIVE SESSION STATE MACHINE        │
                          │  cursor · KV/conv planes · CSPRNG · epoch  │
                          └──────────────────────────────────────────┘
   on TURN accepted ─▶ PASS(resample→mel→conformer→adapter→prefill) ─▶ CQ ─┐
                                                                            │ completion
   ┌────────────────────────────────────────────────────────────────────┐ │ continuation
   │ on prefill done ─▶ PASS(decode token t) ─▶ lanes ─▶ CQ ─────────────┼─┤
   │ on token done   ─▶ (native sample) ─▶ PASS(depth frame t) ─▶ CQ ────┼─┤
   │ on frame done   ─▶ PASS(mimi decode t) ─▶ lanes ─▶ CQ ──────────────┼─┤
   │ on pcm ready    ─▶ publish PCM lease to speaker dock ───────────────┼─┤
   │ if not EOS      ─▶ PASS(decode token t+1)   (native loop, no host) ─┘ │
   └──────────────────────────────────────────────────────────────────────┘
Rust only: fills the mic lease, drains the speaker lease, reads the transcript.
SQ capacity ≥ 2 · next pass queued by the continuation · zero host wait.
```

### 2.3 Components

| Component | What it is | Exists? |
|---|---|---|
| **Weight image** | One allocation containing byte-exact main+codec source files; tensor views are `base + offset` and may be unaligned. | **Landed; page-table read-only after validation.** |
| **Scratch arenas** | Per-plan/per-conversation storage sized before readiness; zero steady-state growth. | **Landed for the complete LFM2 chain.** |
| **Session state machine** | One per conversation. Owns cursor, KV/conv planes, sampler CSPRNG, codec state, epoch, and recurrence. | **Landed natively; Rust no longer drives model progress.** |
| **Pass program set** | Native resample, mel, Conformer, prefill, token, Depthformer, and Mimi stages. | **Landed for LFM2.** Multi-row prefill optimization remains open. |
| **SQ/CQ (capacity ≥ 2) + completion continuation** | A completion may enqueue its follow-on without a synchronous coordinator wait. | **Open.** Current native coordinator parks correctly on a capacity-1 engine slot. |
| **Docks** | Generation-checked mic/speaker PCM leases and bounded control/events. | **Native dock landed.** Physical Rust device adapter remains a later tranche. |
| **Host collapse** | Rust submits tickets, services PCM, and observes events; it owns no model state. | **Landed in the desktop production path; oracle rims are non-release.** |

---

## 3. Deep dive

### 3.1 The zero-copy weight pool

At the design baseline the backbone and Depthformer ran off `PtrLen` views into
Candle-owned tensors while the native image sat beside them. The production path
now binds every LFM2 and Mimi weight directly from the one image; `PtrLen` and
Candle ownership survive only inside the offline oracle feature.

Target: make the resident image the sole weight owner for every plan.

- **One byte-exact image.** Complete source bytes land once in the final
  allocation. Alignment padding exists only between sources; tensors remain at
  their safetensors offsets. Alignment is never repaired by copying a weight.
- **Binding.** Plans carry compact byte-addressed `{base/offset, bytes, dtype,
  shape, layout=NK}` descriptors, not Candle tensors or unaligned C++ typed
  pointers. `lfm_model.cpp` already performs most of the name/shape binding.
- **Consumption.** Architecture leaves load BF16 words from `(N,K)` views and
  unlift them in registers. No `.t().contiguous()`, packed RHS, F32 shadow, or
  per-call whole-weight widening is admitted on Apple or non-Apple paths.
- **Derived storage.** Only formula-changing immutable values such as rope
  tables, window/FFT tables, BN denominators, or required weight-normalization
  folds may persist. Their bytes are reported separately from the model image.
- **Deletion.** `candle_builder` / `CandleBridge` and the ~2.94 GB copy go away;
  the loader stops copying; the working set halves — which matters at decode,
  where M=1 GEMV streams the whole model per token and cache thrash is the enemy.

### 3.2 Scratch arena discipline

One arena per pass-program, sized at ctx/plan build to a high-water bound,
bump-allocated, **zero allocation in steady state**, abort on overflow. This is
already true for the engine ctx scratch, `DepthPlan`, `BackbonePlan`, the Mimi
256 MiB arena, and the Conformer workspace. Two extensions:

- Fold the Conformer's per-call `create/destroy` workspace into the resident
  engine scratch so even audio-in prefill is allocation-free.
- Add frontend (resample, mel) and prefill scratch to the same discipline.

Activations never become `Tensor`s. The mel plane, Conformer rows, adapter
output, hidden state, logits, depth codes, and Mimi PCM all live in engine
scratch or caller-owned buffers and pass between stages by pointer. The three
transport round-trips that exist today — mel→`u16` blob→`Tensor`, adapter
out→`Tensor`, Mimi codes→`Tensor`→`Vec` — are deleted; the session holds the
pointers across the pass boundary instead.

### 3.3 The native recurrence loop (the heart)

Replace the Rust `generate_with_cache` state machine with a native one. On each
token-pass completion the session continuation, running on the bridge/dispatcher
side with no host involvement:

1. reads the sampled token (sampling already native, folded into the pass),
2. checks stop / EOS,
3. advances the token cursor and the KV/short-conv cursor,
4. submits the next pass — decode `t+1`, or the Depthformer frame, or the Mimi
   frame — per the interleave schedule.

```
          ┌──────── token-pass CQ ────────┐
          ▼                                │
   [session.on_token] ── EOS? ──▶ yes ──▶ finish turn, publish terminal CQ
          │  no                            ▲
          ├─▶ submit DEPTH_FRAME ─▶ CQ ─▶ [session.on_frame]
          │                                   │
          │                                   ├─▶ submit MIMI_FRAME ─▶ CQ ─▶ [session.on_pcm]
          │                                   │                                  │
          │                                   │                        publish PCM lease ─▶ speaker dock
          └─▶ submit TOKEN_PASS(t+1) ◀────────┘  (interleave per schedule)
```

- **Interrupt / barge-in.** An interrupt marks the publication epoch stale
  immediately. The in-flight pass reaches its fence, finds it cannot publish stale
  audio, and rings a terminal completion; the session rolls back any speculative
  branch. No assembly instruction is preempted mid-flight; correctness comes from
  the epoch check at the fence, not from cancellation. (Self-interruption / echo
  cancellation is a separate concern tracked elsewhere.)
- **Backpressure.** If the speaker dock's lease ring is full (playback is behind),
  the continuation parks the next Mimi pass on the dock's expected-value word —
  zero-spin backpressure, resumed by the drain doorbell.

### 3.4 Prefill + modality scatter as a native pass

This is the largest remaining Candle island and the hardest "no tensors" code.
Today `prefill_suffix` / `prefill_inputs` build the embedding plane by scattering
text embeddings, native-Conformer audio-in rows, and audio-out codebook
embeddings by modality flag — pure Candle `index`/`cat`/scatter — and the
multi-token backbone walk takes the Candle path (only `seq==1` decode is native).

Target: the scatter becomes an assembly gather leaf writing the embedding plane
into scratch; the multi-token walk reuses the existing per-layer stages over
`seq > 1`; the KV append is already a native plane, so ownership of the cache
moves from the Candle `Cache` struct to the native session.

Because prefill is **per-turn, not per-token**, its latency payoff is smaller than
the decode loop's — so it is sequenced late (P4), after the hot loop is fully
native. It is the last thing that lets the chain be *called* native.

### 3.5 The docks (I/O plane) — kcoro-rs, no std channels

The dock is where Rust std channels are dumped. Every hand-off below is a
kcoro-rs `ring` (bounded SPSC, `SendFuture`/`RecvFuture` that park on wake) or a
kcoro-rs `promise` (exact-once completion), so the audio path suspends and
resumes on the same expected-value substrate the lanes use — never on `mpsc`,
`crossbeam`, or a polling loop. kcoro-rs owns policy and lifecycle only; it never
touches PCM/weights/math and never runs on a compute lane or an audio callback
(its own contract).

- **Mic in — native chunked capture, NOT turn-batched.** The device callback
  writes PCM into a bounded ring; small fixed **chunks** are leased
  (`kc_descriptor` BORROWED) to the native session *as they arrive*, and the
  native session runs VAD, resample, and mel on the streaming chunks and detects
  turn boundaries itself. Rust does **not** accumulate a whole utterance and hand
  it over at turn-close — that turn-batching is the defect this replaces; Rust
  only moves chunks and holds the lease until the native side signals consumed.
  The callback stays a thin writer and does not run kcoro-rs.
- **Speaker out — native chunked playback.** Each Mimi frame pass publishes its
  PCM chunk into a descriptor-leased ring as it is produced; the Rust output dock
  drains chunks via a kcoro-rs `recv` future → `StreamingPcmResampler` (host
  rate-match, a permanent Rust surface) → device. Playback is continuous per
  chunk, not a whole-reply buffer. No `Tensor`, no `.wav`, no std channel crosses
  this seam.

The ~57 std-channel sites in the current voice runtime are retired here; the
`ThreadManager`/`done_rx` polling that motivated the coroutine work in the first
place is replaced by ring/promise waits.

### 3.6 Host collapse

`NativeEngine.pass_lock` exists solely because the C engine is single-slot and
Rust blocks per pass. Once the session drives recurrence natively and the SQ has
depth, the lock and every blocking `submit_pass` rim are deleted. Rust's engine
surface collapses to: create session, submit TURN ticket with a mic lease,
receive PCM leases, submit control tickets. That is spec 10's end state.

---

## 4. Scale & reliability

- **Many conversations, one image.** One weight pool, one lane team. Completions
  route by `conversation_id` / `epoch` to the right session continuation. Fairness
  uses the existing service classes (`DEADLINE` / `INTERACTIVE` / `BACKGROUND`)
  and ticket hierarchy; the dispatcher round-robins passes across live sessions.
  The lane team still runs **one pass at a time** (it is a threadgroup) — SQ depth
  buys queueing and overlap, not parallel passes.
- **Bandwidth is the decode ceiling.** M2 Max is 400 GB/s; the engine currently
  realizes ~66 of a ~250 GB/s practical bound. Decode is M=1 GEMV — every token
  streams the whole model, so tok/s ≈ model_bytes / bandwidth. The pool does not
  change that arithmetic, but deleting the duplicate keeps the working set from
  thrashing and keeps weights bf16 (half the bytes) in `(N,K)` (no repack). The
  win from zero-copy is measured in GB/s of avoided **activation** traffic.
- **Deadlines.** Doorbell waits take an absolute `deadline_ns`. A missed real-time
  deadline is a soft ticket cause — observable, not a crash.
- **Failure is terminal, not degraded.** No fallback. A rejected pass (stale
  epoch, unmet gate) is a terminal completion with a cause; the session decides
  (abort the turn), it never silently drops to Candle.
- **Reliability harness** (already specified in 03 / 11): the 1M-pass soak, stop
  during every submit/dispatch/fence/CQ phase, zero allocation after readiness,
  two conversations scheduled fairly over one image, p50/p95/p99/max latency, and
  ASan/UBSan/TSan across aarch64 / x86_64 / Rosetta.

---

## 5. Trade-offs (explicit)

1. **Resident image vs the Candle duplicate.** Binding the resident image
   directly (no pool, no repack — spec 02) halves RAM, but `candle_builder` cannot
   die until *every* consumer is native, and Candle owns prefill today. The
   Depthformer share has already dropped; the remaining backbone/embedding copy
   drops atomically when production adopts the completed native model. Adding a
   second native model beside the Rust model is forbidden because it would create
   a third main-checkpoint image. Candle remains an offline parity oracle only.
2. **Native recurrence vs the Rust loop.** The whole point: removes the blocked
   host thread and enables overlap. Cost: the hardest code in the project — a
   native state machine replacing a readable Rust loop, harder to debug.
   Mitigation: the per-phase-stop and soak gates; fixture-first parity per pass.
3. **SQ capacity ≥ 2 vs 1.** Enables recurrence-driven overlap, but multiple
   in-flight passes over one scratch arena require per-slot arenas. Start at
   capacity 2 (double-buffer), one arena per in-flight slot.
4. **No-fallback law vs incremental migration.** The law forbids a silent Candle
   `.or_else`, yet the migration needs Candle alive until the native path is
   complete. Resolution: Candle is a *build-time / offline* oracle, never wired as
   a runtime fallback. The runtime gate is "native or terminal error," consistent
   with the Mimi-required rule.
5. **Apple direct BF16 kernels vs Accelerate staging.** The current M>1 path
   widens the complete RHS into `gemm_amx_*`; that is weight materialization and
   is forbidden by the image contract. Replace it with a checkpoint-layout BF16
   kernel. Activation scratch may change precision when the numerical contract
   requires it, but resident weights are loaded and unlifted only in registers.
6. **Prefill native vs leaving it Candle.** Prefill is per-turn, but it is the
   ownership gate for deleting the remaining compatibility image. Develop it
   offline against Candle fixtures; do not ship a hybrid native/Candle fallback.
7. **Moshi.** Moshi stays a **supported model** — it is not dropped. It is
   partially on Flashkern already and gets ported the rest of the way, but as its
   own later phase (P5), because it is a second whole model and would otherwise
   stall the LFM2 hot-loop work. Decision: flip the shipped default to LFM2 only
   in the atomic native cutover. Moshi remains buildable and exercised offline
   until its native port lands.

---

## 6. Build order

### 6.0 What already exists — and why it reorders the plan

This section records the pre-cutover sequencing decision. The ownership work it
describes is now landed for LFM2:

- **A native LFM2 model exists** — `native/src/model/lfm_model.cpp` binds
  the whole backbone by name off the resident image (every layer's norms, FFN,
  short-conv, attention + qk-norms), plus embeddings, head, and Depthformer, all
  zero-copy. The product surface is now opaque runtime/model/conversation/session
  lifecycle plus PCM/control/event docks; numerical direct calls are oracle-only.
- **No weight pool needs to be built.** Spec 02 is explicit: kernels bind the
  resident image *unaligned* and must not repack. The resident image is the pool.
  The earlier "build a re-aligned pool" framing was wrong; drop it.
- **Production voice now uses this path exclusively.** It constructs the native
  runtime/model/conversation/session and fails hard for unsupported engines or
  devices. Frontend, Conformer, Mimi, modality assembly, tokenizer, sampling,
  recurrence, and context rollover are native-owned.

The consequence: **`compatibility_copied_bytes == 0` cannot precede native
prefill.** The Candle model is what performs prefill today, so its weight copy
cannot be deleted until prefill is native. A standalone "weight-pool P1" is
therefore both unnecessary (no repack) and not isolatable (Candle owns prefill).
The right shape is not "build a pool," it is **"close the native model's two gaps
and adopt it"** — which deletes the Candle copy, the Candle prefill, and the Rust
recurrence loop together, because native recurrence (`lfm_conversation_step` /
`_audio_frame`) already exists.

### 6.1 Phases (reordered)

Each phase ends at a gate and deletes the Rust/Candle owner it replaces.

- **P0 — done.** Mel, resample, Conformer + adapter native behind rims.
- **P1 — Adopt the native model where it is already complete; first copy drop.**
  Unwire Moshi from the default (below). Rebind the **fully-native-consumed**
  weights — the Depthformer, whose only consumer is the native depth-frame pass —
  from the resident image instead of `PtrLen`-into-Candle-storage, and stop
  copying them. Route the text / audio-out discrete-token path through
  `lfm_conversation_*`. *Gate:* Depthformer `compatibility_copied_bytes`
  contribution → 0; parity holds; native discrete-token recurrence drives a turn.

  **Depthformer cut — LANDED.** `build_depth_decode_resident`
  (`model/lfm2_audio.rs`) binds the depth plan straight from the resident image by
  name, with rope from the native `lfm_rope_table_f32` kernel — the same one
  `lfm_model.cpp` uses. It is now the production depth path; the Candle depth
  modules (`depthformer` / `depth_linear` / `depth_embeddings`) are built only on
  the non-resident training path (now `Option`, guarded in the training `forward`).
  Verified: `depth_resident_binder_matches_candle_binder` proves byte-identical
  greedy tokens vs the Candle-bound plan; the production load's Candle-copy ledger
  fell **231 → 151 tensors, 2.711 → 2.475 GB** (~236 MB / 80 depth tensors no
  longer duplicated). The remaining ~2.475 GB is the backbone + embeddings, whose
  copy is coupled to native prefill (P2/P3) — Candle owns prefill until then.
- **P2 — Native audio-in prefill + modality scatter (close gap a).** Extend the
  conversation ABI with a continuous-embedding prefill input so the already-native
  mel → Conformer → adapter rows scatter into the backbone natively, by modality
  flag, with no Candle. This is the unlock: once prefill is native, nothing on the
  input side needs the Candle model. *Gate:* audio-in prefill parity vs the Candle
  reference; no `Tensor` at the mel/adapter seam.

  **This is C++-owned, not a Rust rim.** The native prefill lives in
  `lfm_conversation_prefill` (`lfm_model.cpp`): C++ owns the prefill recurrence,
  and Rust only hands the Conformer output over as a *view* and submits a ticket.
  `native_engine.rs` stays a transitional **parity rim** — inference is never
  wired through it. (Guardrail: growing the Rust rim to drive prefill would keep
  Rust as the inference driver, which the whole migration exists to end.)

  **Native audio-in prefill — LANDED (capability; adoption pending).**
  - `lfm_engine_token_pass` gained an `embed_kind == 2` "provided embedding" path
    (`flashkern_engine.cpp`): a bf16 `[H]` hidden view fed verbatim into the pass
    scratch, skipping the table lookup — the point-and-stride way to feed a
    Conformer row (no weight copy). ABI carries a trailing `provided_embed`
    pointer, `nullptr` on every discrete-token caller.
  - `lfm_conversation_prefill_audio` (`lfm_model.cpp`, exposed as
    `NativeConversation::prefill_audio` in `handles.rs`) prefills `[n, hidden]`
    Conformer rows (a borrowed view) into KV, one provided-embedding pass per row —
    same sequential-per-position shape the discrete `lfm_conversation_prefill`
    already uses, so the "sequential vs parallel" worry was moot for the first cut.
  - **Verified:** `native_audio_prefill_matches_discrete_for_the_same_embedding`
    proves `embed_kind == 2` fed a token's own `embed_tokens` row yields the
    identical greedy next-token as the discrete `embed_kind == 0` path — i.e. the
    provided-embedding path produces byte-identical backbone state. 167 lib + 7
    native_safetensors green; decode unaffected.

  Still **C++-owned via the native `LfmModel` conversation** (per the steer:
  `native_engine.rs` stays a parity rim). What remains to drop the backbone copy:
  production voice must *adopt* this native conversation (route the Conformer
  output view + the interleave schedule through it, retire the Candle
  `forward_embeds`). A native *parallel* multi-token prefill pass is the perf
  follow-up for long context.
- **P3 — Adopt the native model for the whole turn; delete the Candle path (close
  gap b).** Move the interleaved generate schedule into the native session so
  `generate_with_cache` / `generate_interleaved` and the Candle
  `LFM2AudioModel` / `Lfm2Model` construction and `candle_builder` all delete
  together; the SQ gains depth (capacity 2) and completion continuations chain
  decode → depth → mimi; `pass_lock` and the blocking rims go. *Gate:*
  `compatibility_copied_bytes == 0` for LFM2; RAM halves; 1M-pass soak;
  two-conversations-fair; zero-alloc-after-ready; idle `< 0.1%`; Rust no longer
  blocks a numerical CQ. (This is the old P1+P2+P4 collapsed, because the native
  vehicle already exists.)
- **P4 — Zero-copy docks (kcoro-rs).** Mic/speaker descriptor leases over kcoro-rs
  rings/promises; dump the std channels and the `ThreadManager`/`done_rx` polling;
  delete the remaining `Tensor` round-trip at the Mimi seam. *Gate:* no `Tensor`
  in the data plane and no std channel on the audio path (static audit).
- **P5 — Finish the Moshi port to Flashkern.** Moshi is a supported model; carry
  its partially-native pipeline the rest of the way onto the lane team and the
  resident image, then delete Candle from the shipped graph entirely. Until then
  Moshi is unwired from the default but remains buildable / exercised offline.

**Moshi default switch (atomic-cutover gate):** flip the default engine from
`MoshiRealtime` to `Lfm2Interleaved` only when the native LFM2 session passes the
product gate. Moshi remains selectable/offline until its native port replaces
the Candle implementation; production never falls back between them.

**Revisit as it grows:** SQ capacity (2 → N as multi-conversation load rises);
arena high-water sizing under long contexts; an E-core `BACKGROUND` lane for
speculative decode / telemetry (currently P-core only); direct-BF16 kernel tile
geometry and prefetch distance when profiling identifies the bottleneck.

---

## 7. Assumptions

- The checkpoint stays bf16 `(N,K)`; no retrain, no requant.
- One model image serves all conversations; there is no per-conversation weight
  specialization.
- Apple and non-Apple production weight paths consume checkpoint BF16 directly;
  no backend may require a complete RHS conversion or repack.
- Real-time targets follow the Sesame latency bands already encoded in the voice
  runtime.
- Candle can be reduced to an offline oracle — nothing in the shipped product
  requires a Candle tensor at runtime once P4/P5 land.
