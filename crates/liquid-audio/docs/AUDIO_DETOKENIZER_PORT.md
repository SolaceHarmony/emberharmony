# Native LFM2.5 audio-detokenizer contract

Status: **native production output path, required for released
LFM2.5-Audio-1.5B.** This is not Mimi and has no fallback to Mimi.

## Pinned authority

- Model: `LiquidAI/LFM2.5-Audio-1.5B` at
  `c362a0625dfe45aa588dce5f0ada28a7e5707628`.
- Reference: `Liquid4All/liquid-audio` at
  `19e65845923a7f136442c95137884ec61eb386aa`.
- Vendored source/config/hash manifest:
  `native/vendor/liquid_audio/`.
- Implementation:
  `native/src/detokenizer/lfm_detokenizer.cpp`.

The production loader does not execute Python and does not download through
Rust. It reads the main checkpoint and
`audio_detokenizer/model.safetensors` directly into one native, sealed,
byte-exact image. The vendored Python is an auditable source contract only.

## Exact released graph

```text
8 code values in [0, 2047]
  → codebook-offset embedding lookup, mean across 8 (512 F32)
  → repeat each row six times
  → 8-layer causal LFM2 detokenizer
       conv, conv, attention, conv, attention, conv, attention, conv
       hidden 512, adjusted FFN 2304, 16 Q heads, 8 KV heads
       head dimension 32, local attention window 30
  → direct F32 projection 512 → 1282
  → 641 log-magnitude + 641 phase values
  → polar spectrum
  → inverse real DFT, n_fft 1280
  → Hann window, hop 320, same-trim overlap-add
  → mono F32 PCM at 24 kHz
```

One complete code frame represents 1,920 output samples. The stateful stream
publishes 1,440 finalized samples after its first frame, 1,920 after every
subsequent frame, and the correlated EOAudio flush publishes the final 480.
The complete stream therefore contains exactly `1920 × code_frames` samples;
the tail is never dropped merely because generation ended.

The packaged config says `bfloat16`, but every field in the released
detokenizer safetensors payload is F32. Safetensors metadata is authoritative.
The plan binds exactly 79 required F32 byte views with exact names, ranks, and
shapes. It validates the packaged-but-unused token embedding field without
counting it as a numerical consumer.

## Ownership and memory

- `LfmModel` solely owns the combined main-plus-detokenizer image and one
  immutable `LfmAudioDetokenizerPlan`.
- A conversation owns one `LfmAudioDetokenizerState`: ShortConv carry, three
  bounded GQA K/V rings, activation scratch, and ISTFT overlap-add/envelope
  state in one setup-time arena.
- A field called a tensor in upstream metadata becomes only a byte-addressed
  view: base, offset, extent, dtype, shape, and derived byte strides. No
  framework numerical object exists in production.
- No weight is widened, aligned, transposed, packed, relocated, or copied.
  Directly bound, formula-derived, and compatibility-copied bytes are tallied
  separately. Production requires `compatibility_copied_bytes == 0`.
- Formula-derived inverse-DFT and RoPE tables are permitted and accounted.
  They change the formula representation; they are not compatibility weight
  images.
- All per-frame storage is preallocated before readiness. The step and flush
  paths allocate nothing and write PCM directly into the retained playback
  reservation at 24 kHz. A device-rate mismatch uses conversation-owned
  detokenizer scratch and the same route's native streaming resampler.

## Route and isolation

The only released LFM2.5 output route is:

```text
REQ_TOKEN_PASS
  → REQ_DEPTH_FRAME
  → REQ_AUDIO_DETOKENIZE
  → optional native stream resample
  → retained playback lease
```

EOAudio routes to `lfm_detokenizer_state_flush`; it is not an artificial
silence timer and it cannot publish twice. Ticket/epoch checks remain attached
through final device consumption.

Mimi is retained as `liblfm_mimi.a` for a future native Moshi model. Its plan
can bind only `LFM_WEIGHT_COMPONENT_MIMI`. The LFM2.5 loader loads only
`MAIN` and `DETOKENIZER`, so it cannot construct Mimi accidentally. There is no
`REQ_MIMI_DECODE` in the LFM2.5 engine, no Mimi field in `LfmModel` or
`LfmConversation`, and no missing-detokenizer fallback.

## Numerical execution

Large six-row dense stages use Accelerate/AMX on Apple. Other current stages
use architecture SIMD and direct byte views, with accumulators kept in
registers and only stage results published. The implementation is already
native and tensor-free, but it is not yet the final Flashkern arithmetic form:
several value-producing loops remain in C++/intrinsics and the complete
detokenizer currently executes on lane zero.

Remaining performance work is therefore explicit:

1. Transliterate remaining scalar/intrinsic numerical bodies into paired
   AArch64/x86_64 assembly leaves without introducing a fallback.
2. Fuse projection → polar conversion → inverse DFT → window/overlap-add where
   liveness permits so intermediate spectrum/time planes disappear.
3. Partition ShortConv, GQA heads, dense rows, and spectral bins across the
   fixed kcoro/Flashkern team. Coroutines sequence coarse stages; assembly owns
   complete tiles and never suspends inside a kernel.
4. Preserve fixed accumulation/rounding contracts and direct destination
   writes while reducing the streaming underflow count in the real speaker
   gate.

## Gates

- Exact schema rejection: missing field, wrong dtype, wrong rank/shape,
  topology mismatch, or extra detokenizer view fails model readiness.
- Single-image accounting: one main-plus-detokenizer image, no open model files
  after readiness, no compatibility copies.
- Stateful stream accounting: `1440 + 1920 × (N - 1) + 480 == 1920 × N`.
- Native two-agent speech-to-speech gate: generated code frames pass through
  Depthformer and this detokenizer in memory; transcript, PCM, ticket, epoch,
  deterministic seed, and complete retirement are checked.
- Real speaker gate: the same in-memory stream is drained by the native device
  callback. Audible clarity does not waive underflow, overlap, truncation, or
  terminal-order failures.

