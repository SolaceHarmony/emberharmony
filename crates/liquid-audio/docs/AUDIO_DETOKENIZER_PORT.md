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

EOAudio begins a flush-form `LfmAudioDetokenizerProgram`. Its only numerical
phase is the remaining overlap emission, mounted on the same fixed team as a
normal code frame; it is not an artificial silence timer and it cannot publish
twice. Ticket/epoch checks remain attached through final device consumption.

Mimi is retained as `liblfm_mimi.a` for a future native Moshi model. Its plan
can bind only `LFM_WEIGHT_COMPONENT_MIMI`. The LFM2.5 loader loads only
`MAIN` and `DETOKENIZER`, so it cannot construct Mimi accidentally. There is no
`REQ_MIMI_DECODE` in the LFM2.5 engine, no Mimi field in `LfmModel` or
`LfmConversation`, and no missing-detokenizer fallback.

## Numerical execution

The detokenizer is one retained `LfmAudioDetokenizerProgram` carried by its
admitted pass ticket. Each causal phase is one Flashkern team generation. The
last logical lane to return publishes the quorum edge; that callback resumes
the bridge continuation, which advances the durable phase/layer cursor and
dispatches the successor. No host thread waits for a phase, and no lane waits
for another lane.

Embedding columns, row norms, ShortConv channels, GQA head groups, SwiGLU and
residual bands, spectral bins, overlap-add ring ranges, and final PCM ranges
are disjoint fixed-team partitions. A lane carries residual→RMSNorm through one
row-owned generation and overlap-add→normalization/emit through one ring-owned
generation; those pairs have no cross-lane dependency, so materializing a
quorum between them would only eject hot state. This removes 17 artificial
team generations per code frame. Large six-row dense stages and the inverse
DFT use Accelerate/AMX on Apple as explicit serial resources inside the same
ticket; peers return immediately and remain available to the shared kcoro
pool. An optional output-rate conversion is one final native generation.

Every non-opaque payload operation is now a paired
`flashkern_detokenizer.S` leaf on AArch64 and x86_64: embedding, residuals,
RMSNorm, SwiGLU arithmetic, scaled attention dots, softmax reductions and
normalization, RoPE, ShortConv, weighted values, polar conversion,
overlap-add, envelope normalization, PCM emission, and both immutable derived
table constructors. C++23 binds views and advances phases; it contains no
detokenizer payload formula or architecture intrinsic. The only opaque
numerical seams are measured Accelerate/AMX dense calls and vForce
exponential/sine/cosine calls.

The FIFO boundary is explicit. Values remain in architecture registers for a
complete assembly tile. They materialize only when the next consumer is an
opaque AMX/vForce call, a causal state owner, or a cross-lane quorum. The final
AMX projection writes once into its destination; bias, exponential/trigonometric
conversion, and polar conversion proceed in place in the same retained
continuation. The inverse DFT writes into the now-dead FFN scratch plane, and
overlap-add plus emission share one generation. There is no separate IFFT
plane, magnitude copy, phase copy, or bridge-side numerical handoff.

All stages consume direct byte views and preallocated conversation scratch.
The implementation is native and tensor-free. It also fixes the former AArch64
ShortConv tap scratch overrun: each of four NEON channels now owns exactly its
three taps instead of indexing a three-row array with four channel indices.
SIMD-producing partitions are expressed in four-element quanta, so changing
the fixed-team width cannot create new scalar boundary cells or change the
rounding contract. The real-checkpoint gate runs one fixed code trace through
three- and eight-lane programs, including causal state and final flush, and
requires byte-identical PCM.

The physical callback gate measured the change in two steps at 24 kHz in the
same two-agent test. Removing the former lane-zero whole-graph execution cut
streaming starvation from 11,744 frames across 47 callbacks to 640/3. Fusing
the row- and ring-local successors then reached **0 underflow frames across 0
callbacks**, with the same fixed-seed transcripts, PCM digests, frame counts,
and one ordered source transition. This is a measured gate result, not a claim
that arbitrary hardware can never underrun.

There is no known detokenizer numerical-ownership debt. Further fusion is
profile-driven only: it may remove a materialization boundary when liveness
proves the consumer can remain in the same register tile, but it may not cross
an opaque AMX/vForce seam, conceal a cross-lane dependency, add a fallback, or
change the fixed rounding contract merely to reduce a phase count.

## Verification

- Exact schema rejection: missing field, wrong dtype, wrong rank/shape,
  topology mismatch, or extra detokenizer view fails model readiness.
- Single-image accounting: one main-plus-detokenizer image, no open model files
  after readiness, no compatibility copies.
- Stateful stream accounting: `1440 + 1920 × (N - 1) + 480 == 1920 × N`.
- Native two-agent speech-to-speech gate: generated code frames pass through
  Depthformer and this detokenizer in memory; transcript, PCM, ticket, epoch,
  deterministic seed, and complete retirement are checked.
- Fixed-team invariance: the same released weights and audio-code trace must
  produce byte-identical stateful PCM at three and eight lanes. The native test
  receives the lane count as an explicit command-line argument.
- Real speaker gate: the same in-memory stream is drained by the native device
  callback. Audible clarity does not waive underflow, overlap, truncation, or
  terminal-order failures.
