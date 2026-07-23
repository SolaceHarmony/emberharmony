# 09 — Responsive voice turns

What Sesame's recovered CSM client actually measures and rates, why our voice
pipeline self-interrupts, and the ordered plan to close both gaps in the native
voice runtime.

Sources examined 2026-07-05: the recovered Sesame demo client, Kyutai's moshi
repo, SesameAILabs/csm + the HF Transformers CSM port, and our own pipeline.

---

## 1. The mental model

A speech model neither hears nor speaks. It reads one flat token sequence — a
transcript of text tokens and audio-codec tokens — and predicts the next token.
Everything conversational rests on two facts:

1. **Speaker identity is provenance, not acoustics.** The runtime stamps a
   label when it commits audio to the sequence (`[0]` in CSM,
   `<|im_start|>user` in LFM2, stream position in Moshi). The model trusts the
   label completely; nothing ever verifies it against the sound.
2. **Turn boundaries are tokens, not timing.** `<eos>` / `<|audio_eos|>` /
   `<|im_end|>` are read and emitted by the model. The runtime decides *when*
   to close a turn; the model decides *that* a turn reads as closed.

Our self-interruption bug is a **labeling lie at the acoustic layer**: the
model's own voice crosses speaker→room→mic and gets committed under the user's
label. The model then correctly yields the floor to "the user" — itself.

Our latency gap is a **work-scheduling error**: too much turn work remains after
the user stops talking, plus a 500ms endpoint-silence policy before response
generation. Speech policy and scheduler mechanics must not be conflated.

### 1.1 Three meanings of time

1. **Sesame telemetry:** the recovered client samples spectral telemetry every
   20ms, measures last user voice to first agent voice, averages observations,
   and assigns the `<300/<500/<1000/<3000ms` ratings below. The 20ms interval is
   observer sampling only. It is never evidence for a scheduler heartbeat.
2. **Speech policy:** endpoint silence, sustained barge-in, speculative
   preparation, minimum utterance, and echo-tail durations describe acoustic
   behavior. Production represents them as sample/frame counts at the capture
   rate.
3. **Scheduler mechanics:** timeout receives, periodic probes, drain heartbeats,
   and retry sleeps are not speech policy. They must not cause progress.

The governing rule is:

> Human and acoustic durations are sample-clock state-machine thresholds.
> Wall-clock observation may measure latency or detect a stalled device, but it
> never advances speech or inference state. Every computational transition is
> caused by capture, completion, capacity, control, or an explicit fault edge.

Callbacks advance the capture cursor for voiced and silent device frames. An
endpoint is therefore `capture_cursor - last_voiced_cursor >= endpoint_frames`;
sustained barge-in is `consecutive_voiced_frames >= sustain_frames`. If device
callbacks stop, that is a liveness fault—not synthetic acoustic silence.

## 2. What each system actually does (evidence)

### 2.1 Sesame demo client — the client knows nothing

Recovered from the vim swapfile of the production bundle
(`/Volumes/stuff/sesame_demo/assets/.index-CplABjlX.js.swp`, recovered copy
with stable line numbers: `/Volumes/stuff/sesame_demo/index-CplABjlX.recovered.js`).

The client contains **zero** turn-taking, VAD-gating, or interruption logic.
It is a telephone plus a report card:

| Finding | Location (recovered.js) |
|---|---|
| Stock WebRTC call: `new RTCPeerConnection({ iceTransportPolicy: "relay", ... })`, mic track up, model track back | 50212 |
| Mic capture `getUserMedia({ audio: true })` — bare defaults ⇒ browser AEC/NS/AGC on | 47784 |
| Mic starts muted, unmutes after a warmup timer (no connection garbage reaches the model) | 47787, 47798, 47754 |
| Playback = `new Audio(); el.srcObject = remoteStream` — browser NetEq jitter buffer, no custom buffering | 47837–47859 |
| WebSocket is signaling only; base64 audio path exists as fallback | 27074–27090, 47651 |
| Telemetry state machine: 20ms snapshots, dual spectral VAD (600–2400Hz, adaptive min/max) over user mic **and** agent playback; states `user_talking / user_paused / agent_talking / agent_paused` | 49822–49930 |
| Measures last-user-speech → first-agent-voice per turn | 49873–49895 |
| **Latency rating: <300ms = 5, <500 = 4, <1000 = 3, <3000 = 2, else 1** | `getAgentResponseLatencyRating`, 49909 |
| First-word latency rating: <2s = 5, <3s = 4, <4s = 3, <6s = 2 | 49913 |
| Buffer-underrun and WebRTC jitter monitors (quality telemetry) | 49935+, 49960+ |

The `vad`/`interrupt` strings in the bundle are all telemetry or React
internals. Every intelligent decision runs server-side, colocated with the
model, fed by a continuous mic stream. The rating table is Sesame's latency
spec, leaked in minified JS.

### 2.2 Kyutai Moshi — echo was never solved in anything we ported

Upstream `github.com/kyutai-labs/moshi` (cloned shallow, 2026-07-05):

| Finding | Location |
|---|---|
| Web client requests browser AEC: `echoCancellation: true, noiseSuppression: true, autoGainControl: true` — the **entire** Moshi echo solution | `client/src/pages/Conversation/components/UserAudio/UserAudio.tsx:37` |
| Python CLI client: "barebones: it does not perform any echo cancellation" | `README.md:146` |
| MLX client: same disclaimer | `README.md:164` |
| "We recommend using the web UI as it provides additional echo cancellation that helps the overall model quality" | `README.md:196`, `rust/README.md:61` |
| Rust CLI is raw cpal in/out, no processing | `rust/moshi-cli/src/audio_io.rs` |
| `moshi-core` is pure model math; Candle's Mimi (`candle-transformers/src/models/mimi/`) is codec-only | — |

Every Moshi client that skips the browser ships the self-hearing problem as a
documented limitation. The model was always fed pre-cleaned audio by a browser.

### 2.3 CSM — turn discipline as a ten-line grammar

`github.com/SesameAILabs/csm` + HF Transformers port
(`src/transformers/models/csm/`):

| Finding | Location |
|---|---|
| Context = `Segment{speaker, text, audio}` list, both speakers, one interleaved sequence | `generator.py:98–131` |
| Speaker tag is literal text `[0]` through the Llama tokenizer | `generator.py:64` |
| Chat template: role compiles to `'[' + role + ']'` + text + `<eos>` + `<\|AUDIO\|>…<\|audio_eos\|>` — the whole turn system | `convert_csm.py:280` |
| Roles are stringified **integers** — speakers are symmetric, no privileged assistant | template validation, same location |
| Every context turn must carry both text and audio — the tag is a binding variable; voice identity comes from audio bound to it earlier | template validation; README FAQ (random voice without context) |
| End of speech = all-zero / all-EOS codebook frame | `generator.py:148`; `generation_csm.py:301–306` |
| Watermarking is applied once post-generation (provenance for third parties); plays no role in speaker handling | `generator.py:165`, `watermarking.py` |

CSM never listens while speaking — full-duplex self-hearing is designed out,
not solved. Its lesson is the context shape: everything both speakers said,
speaker-tagged at commit time, audio-grounded.

### 2.4 Our pipeline today

`crates/liquid-audio/src/` and
`packages/desktop/src-tauri/src/voice/runtime.rs`:

**Already right:**

| What | Where |
|---|---|
| CSM-shaped context: `<\|im_start\|>role` turns + modality flags (Text/AudioIn/AudioOut) | `processor.rs:333–398` |
| Model's generated audio committed to context **unconditionally**, barge-in or not (context = the model's thoughts; also anchors voice identity, per §2.3) | `realtime.rs:861–889` |
| Dedicated output thread decouples inference from blocking speaker writes (dd6e02e) | `voice_runtime.rs:795–822` |
| App-layer AEC exists: mic and speaker both routed through libwebrtc's platform ADM; `echo_cancellation: true, noise_suppression: true, auto_gain_control: true` | `src/voice/runtime.rs:1829–1837`; mic loopback :3082+, output loopback + ADM playout :2823+, :2947 |
| The two `PlatformAudio` handles share one refcounted ADM ⇒ the APM can see both capture and render | `livekit-0.7.49/src/platform_audio/mod.rs:408–418`; `libwebrtc-0.3.38/src/peer_connection_factory.rs:148–158` |

**Broken or missing:**

| Gap | Where | Consequence |
|---|---|---|
| Echo identity guessed from loudness: barge-in requires mic RMS > max(3× base, 2.5× playback RMS) | `voice_runtime.rs:1267–1277` | Loud echo self-interrupts; quiet user can't interrupt |
| Model goes deaf during own turn: `mic.clear()` while assistant speaks (default `can_interrupt=false`) | `voice_runtime.rs:1012–1018` | Overlapped user speech destroyed — never reaches context |
| VAD uses 200ms analysis windows; sustained barge-in policy is 400ms | `voice_runtime.rs` VAD policy constants and `window = in_rate / 5` | Policy is sample-clock state; callbacks advance it and a kcoro doorbell resumes the consumer |
| End-of-turn = 500ms silence (`silence_ms` default, reduced from 800ms after live testing) | `voice_runtime.rs` defaults | The duration is converted to `silence_frames`; the former 40ms timeout heartbeat is deleted |
| Full-utterance mel + prefill happens **after** end-of-turn | `realtime.rs:722–892` | O(utterance) work in the response-latency critical path |
| AEC never verified; macOS uses software AEC3 (VPIO is iOS-only), AGC may amplify residual echo | `libwebrtc audio_source.rs` platform notes | Self-interruption persists despite AEC being "on" |
| Moshi path: hard interrupt resets LM state entirely | `realtime.rs:433–437, 1022–1026` | Contradicts context-is-thoughts for the Moshi engine |

**Latency budget, ours vs Sesame's rating.** The recovered client rates an
averaged pause→first-agent-voice interval below 300ms as excellent and below
500ms as good; it does not prove what the Sesame server itself always achieves.
Our runtime reports individual turns. The current path still pays the 500ms
endpoint policy plus remaining post-endpoint work and first-frame generation.
We have no relay-network requirement, so structurally the local path should be
able to score well against those bands.

## 3. The plan

Ordering: measure first (W1), then the two latency workstreams (W2, W3), then
conversational feel (W4, W5). W6 runs in parallel — it is cheap and
de-risks everything else.

**Cross-spec dependency (added 2026-07-17, refined after code read).** This
ordering is internal to spec 09; W2 and W5 are additionally coupled to spec 11's
native migration and are not independent of it. Two facts from the tree pin what
actually blocks W2:
- Mel math already has a native, parity-gated path (commit `a1b06fc7`), but the
  Rust rim still materializes a full `Vec<f32>` + Candle `Tensor` per call
  (`FilterbankFeatures`, `processor.rs`). Streaming/chunked mel needs that
  transport rim cut (spec-11 doc 06, the Conformer cutover) — the math is not the
  blocker; the materialization is.
- Incremental/suffix prefill is **already native-owned** (`RUST_DELETION_PLAN.md`,
  "Native model chain"). What prevents pipelining prefill *during* listen is not
  prefill — it is that the C++ **session coordinator still parks once for the
  route's terminal result** (F1 in `RUST_DELETION_PLAN.md`: convert `run_action`
  to a session-owned asynchronous state machine). Until F1 lands, command/control
  cannot advance while a prefill route is outstanding.

So W2 is gated on **F1 (session async state machine) + the mel transport-rim
cut**, not on prefill nativeness. Its Rust anchors (`ChatState::add_audio`,
`Lfm2VoiceEngine::respond` in `realtime.rs`) are on spec 11's deletion list;
build W2 against the native seam, not by extending them. Only W1
(instrumentation) and W6 (AEC verification) are free of this dependency and can
run now.

### W1 — Instrument Sesame's metric (baseline before touching anything)

Implement exactly their measurement in `voice_runtime.rs`:
`last_voice` timestamp already exists in `vad_loop` (:1040); record the wall
time when the first PCM of a reply hits the output ring (consumer,
`voice_runtime.rs:869`), log `turn_latency_ms` per turn plus first-word
latency per session, and rate against 300/500/1000/3000ms bands.
Emit via the existing stats path so both CLI examples and the app see it.

*Acceptance:* every turn logs pause→first-audio; baselines captured for LFM2
and Moshi paths.

### W2 — Prefill during listen (the big one)

Move O(utterance) work inside the user's turn, CSM-style commit-as-you-go:

- Stream mic audio into the user turn incrementally: chunked mel computation
  (mind the conformer frontend's window/lookahead — see `wiki/CF04-Mel-Frontend.md`)
  and incremental prefill of the model on partial user audio.
- Split `Lfm2VoiceEngine::respond` (`realtime.rs:722`) into
  `ingest_user_chunk()` (during turn) + `finish_turn()` (at EOT: close user
  turn, open assistant turn, generate). `ChatState::add_audio`
  (`processor.rs`) grows an in-progress user turn instead of receiving one
  finished utterance.

*Acceptance:* post-EOT model work ≈ first-frame generation only; W1 metric
shows the gap collapsing toward `silence_ms` + generation.

### W3 — Sample-clock adaptive end-of-turn

- Express the current 500ms policy as `endpoint_frames` and tune toward a
  measured 300–400ms adaptive range only when real-session false-cutoff data
  supports it. The earlier 350ms value was too eager in live testing.
- Optional second stage: tentative EOT at ~250ms → begin generating
  speculatively; if the user resumes before commit, cancel and **do not
  commit** the speculative output (it was never spoken nor a completed
  thought). Any revival of preemptive generation must re-check the mode gate
  at commit time — we've been bitten by preemptive reads of stale mode state
  before.

*Acceptance:* no timed progress wake remains; median W1 latency is <500ms
without a rise in false end-of-turn (user cut-offs), measured over real
sessions.

### W4 — Never deaf: buffer overlap, commit as context

Delete the `mic.clear()` suppression (`voice_runtime.rs:1012`). During
playback, keep VAD-ing the post-AEC mic; buffer voiced spans; at the next
commit point append them as a user turn (audio-only user turns already work —
`respond()` builds them today) **even when no interrupt fired**. Overlap
("mm-hm", "no—Tuesday") becomes shared context instead of vanishing.

*Acceptance:* speech spoken over the assistant appears in the next turn's
context; chat_multiturn-style test covers it.

### W5 — Two-stage barge-in (duck → yield)

Replace the one-window reflex (`voice_runtime.rs:1025`):
sustained voiced input ~250–400ms → duck output gain (output thread already
owns the writes); continued speech → interrupt via the existing path (partial
assistant turn is already committed by `realtime.rs:861`), then commit the
W4 buffer as the user turn. Humans duck and listen for a beat before yielding;
so should we.

*Acceptance:* echo blips and coughs no longer stop the assistant; a real
interjection stops it within ~400ms with the overlap preserved in context.

### W6 — Verify the AEC actually cancels (parallel, cheap)

The AEC is enabled but unproven. Behind a debug flag, dump per-session:
(a) post-APM mic frames, (b) the playback reference (output thread knows the
exact samples and timing). Compute ERLE during assistant speech.

- ERLE ≈ 0dB → render reference isn't reaching the APM through the
  double-loopback topology; fix the wiring before trusting anything upstream.
- ERLE 15–30dB → AEC converges; residual echo is the problem — first
  experiment: `auto_gain_control: false` (AGC amplifying residual echo is the
  classic self-trigger); later option: macOS VPIO instead of software AEC3.

Optional diagnostic from the watermark discussion: silentcipher-style
detection over recorded sessions to quantify echo leakage offline. Measurement
tool only — never a real-time gate (chunk latency, and it would discard
genuine barge-in overlapping our playback).

## 4. What we are deliberately NOT doing

- **No acoustic speaker ID / watermark gating in the loop.** Identity is
  provenance at commit time (every system in §2 agrees); for self-recognition
  we hold the exact reference waveform — subtraction (AEC) strictly beats
  detection.
- **No new token grammar.** LFM2's turn grammar already expresses everything
  W2–W5 need, including audio-only user turns.
- **No client/server split.** Sesame's colocation lesson is already our
  architecture — in-process beats their relay transport. The gap is scheduling
  and end-of-turn policy, not plumbing.
