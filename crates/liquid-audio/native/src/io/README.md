# Native Weight Image

`safetensors.cpp` is the native checkpoint boundary for the CPU inference stack.
It has no Rust or Candle dependency. A host supplies one path through the C ABI in
`native/include/lfm_safetensors.h`; C++ owns file discovery, file reads, parsing,
validation, tensor indexing, and lifetime.

## Memory Contract

```text
checkpoint directory / file
          |
          | one elected build, before inference
          v
+------------------------------------------------------------------+
| named, wired, read-only shared segment (64 KiB granules)          |
|                                                                  |
| 64 KiB header | shard 0 complete bytes | pad | shard 1 | ...     |
| [8-byte N][JSON][payload]     [8-byte N][JSON][payload]           |
+------------------------------------------------------------------+
      ^                                ^
      | base + tensor.offset           | base + tensor.offset
      |                                |
   BF16/F32 view                    BF16/F32 view
```

- `shm_open(O_CREAT|O_EXCL)` / `CreateFileMapping` elects one builder for an
  identity derived from the ordered source `FileState` tuples. Every selected
  shard is read directly into its final slice of that one named segment.
- The builder publishes `INVALID -> INITIALIZING -> BUILDING -> READY | POISONED`.
  `INITIALIZING` publishes the owner PID/start/UID and generation before any
  layout field, so a live initializer is correlated and a dead initializer can
  be poisoned and replaced without misclassifying the zero-state creation
  window. Same-process contenders
  dehydrate literal kcoro continuations; `READY` resumes their exact ticket
  after the process registry owns the completed lease. Synchronous callers
  receive `LFM_WEIGHT_IN_PROGRESS`; no thread waits, polls, or sleeps beside
  the header.
- Every mapping is mandatorily `mlock`ed / `VirtualLock`ed. Failure is terminal
  and names the operating-system limit; there is no unwired fallback.
- After source identity, metadata, span, and index validation, the complete
  region is published read-only with `mprotect(PROT_READ)` / `VirtualProtect`.
  An accidental write faults instead of corrupting every sharing conversation.
- Tensor payload bytes are never copied, cast, repacked, or materialized as host
  tensor objects.
- A `LfmTensorView` carries both a direct pointer and a base-relative offset.
- Names and shapes are small init-time descriptors parsed from JSON; kernels bind
  payload pointers once and perform no lookup in the inference loop.
- All view pointers remain valid until the final refcounted segment lease
  closes. Closing a handle detaches and never unlinks the machine-wide name.
  `lfm_weights_evict` is the only reclamation operation; live mappings survive
  it by POSIX/section semantics.
- A fresh process validates and attaches a `READY` segment read-only with zero
  tensor-payload reads. Header provenance retains the original build time,
  worker count, task count, generation, identity digest, and content-tree
  digest.

For sharded Hugging Face checkpoints, the loader validates the index against the
actual tensor names and source shards. Without an index, a directory resolves
`model.safetensors`, then sorted `model-*.safetensors`; unrelated tokenizer
checkpoints are not folded into the model image.

`lfm_weights_open_bundle` resolves the main model and LFM2.5 audio-detokenizer
source separately,
then sends both source sets through the same segment and read team. Its
catalog key is `(Main|Detokenizer, tensor name)`: cross-component duplicate
names are legal; duplicates within one component fail. The legacy lookup
functions are Main-scoped, while native model construction uses the
component-scoped forms.

## Validation

The loader rejects malformed JSON, unsupported dtypes, shape/bit-count overflow,
incorrect byte counts, non-contiguous or overlapping spans, payload bytes not
described by the header, duplicate names across shards, unsafe shard paths, and
index-to-shard disagreement. No C++ exception crosses the C ABI.

## Current Migration State

The shipped desktop opens the image only through the opaque native runtime. It
does not construct `ResidentWeights`, a Candle builder, or a Rust LFM2 model.
The old Rust model, training code, fixture capture, and compatibility adapters
were deleted after native ownership landed; no callable alternative loader or
model graph remains.

`LfmModelMemoryV2` reports source bytes, segment bytes, bytes constructed vs
attached by this process, wired bytes, mapping-attributed resident bytes,
weight-payload read calls/bytes, identity/content digests, directly
bound tensor bytes, formula-derived immutable bytes, compatibility-copy bytes,
load time, worker count, and task count. Production rejects a model unless
`compatibility_copied_bytes == 0`.

The standalone C++23 keeper is built without a Rust launcher:

```sh
make -C crates/liquid-audio/native/tools
crates/liquid-audio/native/tools/build/lfm-weight-segment verify /absolute/checkpoint
crates/liquid-audio/native/tools/build/lfm-weight-segment \
  host /absolute/checkpoint EPOCH:SEQUENCE:GENERATION:KIND
crates/liquid-audio/native/tools/build/lfm-native-speech-gate \
  /absolute/checkpoint 8 silent
```

It holds the wired lease across arbitrary client exits. On macOS GCD owns its
signal dormancy and invokes teardown; there is no sleeping host loop. `open`
performs one attach-or-build and exits, while `evict IDENTITY_SHA256` removes
only the named object. The optional canonical ticket is echoed in the fixed
readiness record so a spawning client resumes the exact logical continuation.
`verify` proves, with the real checkpoint, that one keeper build is followed by
a simultaneous zero-payload client attach and a fresh-process post-keeper
attach. It also samples macOS's wired-page counter: the keeper must wire roughly
one segment, a second mapping must not wire a second physical model, and keeper
retirement must drop roughly one segment's pages.

Cross-process multi-client request routing remains the next host seam; direct
continuation admission against a foreign `INITIALIZING` or `BUILDING`
generation is rejected
unless it arrived through the host's correlated readiness edge. No client
invents a retry loop or suspends without an owner that can resume it. The
blocking pipe/read and child reap inside `verify` are administrative test
coordination, not model execution or keeper dormancy.

`lfm-native-speech-gate` is the standalone production-path output gate. It is
linked directly from the C++23 runtime/model, kcoro C runtime, architecture
`.S` leaves, and Apple frameworks; it has no Rust launcher or Rust runtime. Two
complete native agents exchange through in-memory audio-token/PCM buffers. The
gate runs the fixed-seed exchange twice and hashes every terminal PCM sample in
memory, so build-vs-attach identity is tested without writing or rereading a WAV
file. `buffered` and `stream` replace `silent` only when a human is listening.

## Load benchmark

The real-checkpoint gate is an opt-in native example and never downloads or
silently substitutes a fixture:

```sh
LFM_MODEL_DIR=/absolute/checkpoint \
  cargo run --release -p liquid-audio --example bench_native_load
```

It explicitly evicts before every build sample, alternates the exact builder
with one and four I/O workers, and separately measures persisted attaches. It
validates that every run publishes the same content-tree digest and emits
cold/warm-build and attach p50/p95, build GiB/s, RSS, ownership bytes, payload
reads, worker count, and task count as JSON. Attach samples must report zero
tensor-payload reads. Cold
samples use a platform cache-bypass/eviction facility; when none is available,
the cold report is `null` rather than warm data under a misleading label. The
process exits unsuccessfully if the four-worker p50 or p95 regresses the serial
baseline. `LFM_LOAD_BENCH_RUNS` changes the default five samples per mode.

Native LFM2.5 output binds the Detokenizer catalog of the model-owned combined
image. It does not reopen the submodel and never constructs the legacy Mimi
decoder. Each conversation owns only detokenizer KV, convolution carry, ISTFT
overlap state, and scratch. Formula-derived RoPE/ISTFT tables live once in the
plan. Checkpoint-layout F32 weights remain immutable resident-image views; no
alignment, layout, or dtype copy is admitted.

## Provenance

The whole-file resident-block and span-planning approach was adapted from the
local `ember-ml` safetensors loader. Numerical UKM ingress was intentionally not
ported because model weights must remain byte-exact. JSON parsing uses the
MIT-licensed nlohmann/json 3.11.3 header vendored under `native/vendor/nlohmann/`.
