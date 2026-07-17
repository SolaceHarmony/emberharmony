# Native Weight Image

`safetensors.cpp` is the native checkpoint boundary for the CPU inference stack.
It has no Rust or Candle dependency. A host supplies one path through the C ABI in
`native/include/lfm_safetensors.h`; C++ owns file discovery, file reads, parsing,
validation, tensor indexing, and lifetime.

## Memory Contract

```text
checkpoint directory / file
          |
          | one blocking load, before inference
          v
+------------------------------------------------------------------+
| page-aligned, read-only LfmWeightImage virtual-memory region      |
|                                                                  |
| shard 0 complete bytes | pad | shard 1 complete bytes | ...      |
| [8-byte N][JSON][payload]     [8-byte N][JSON][payload]           |
+------------------------------------------------------------------+
      ^                                ^
      | base + tensor.offset           | base + tensor.offset
      |                                |
   BF16/F32 view                    BF16/F32 view
```

- Every selected shard is read directly into its final slice of one allocation.
- After source identity, metadata, span, and index validation, the complete
  region is published read-only with `mprotect(PROT_READ)` / `VirtualProtect`.
  An accidental write faults instead of corrupting every sharing conversation.
- Tensor payload bytes are never copied, cast, repacked, or materialized as host
  tensor objects.
- A `LfmTensorView` carries both a direct pointer and a base-relative offset.
- Names and shapes are small init-time descriptors parsed from JSON; kernels bind
  payload pointers once and perform no lookup in the inference loop.
- All view pointers remain valid until `lfm_weights_close`.
- Loading is synchronous. The loader performs no disk work after it returns.

For sharded Hugging Face checkpoints, the loader validates the index against the
actual tensor names and source shards. Without an index, a directory resolves
`model.safetensors`, then sorted `model-*.safetensors`; unrelated tokenizer
checkpoints are not folded into the model image.

`lfm_weights_open_bundle` resolves the main model and Mimi source separately,
then sends both source sets through the same allocation and read team. Its
catalog key is `(Main|Codec, tensor name)`: cross-component duplicate names are
legal; duplicates within one component fail. The legacy lookup functions are
Main-scoped, while native model construction uses the component-scoped forms.

## Validation

The loader rejects malformed JSON, unsupported dtypes, shape/bit-count overflow,
incorrect byte counts, non-contiguous or overlapping spans, payload bytes not
described by the header, duplicate names across shards, unsafe shard paths, and
index-to-shard disagreement. No C++ exception crosses the C ABI.

## Current Migration State

The shipped desktop opens the image only through the opaque native runtime. It
does not construct `ResidentWeights`, a Candle builder, or a Rust LFM2 model.
The old Rust model, training code, fixture capture, and compatibility adapters
are isolated behind the workspace-only `liquid-audio-oracle` package. They are
not in the production dependency graph.

`LfmModelMemoryV1` reports source bytes, logical resident-image bytes, directly
bound tensor bytes, formula-derived immutable bytes, compatibility-copy bytes,
load time, worker count, and task count. Production rejects a model unless
`compatibility_copied_bytes == 0`.

## Load benchmark

The real-checkpoint gate is an opt-in native example and never downloads or
silently substitutes a fixture:

```sh
LFM_MODEL_DIR=/absolute/checkpoint \
  cargo run --release -p liquid-audio --example bench_native_load
```

It alternates the exact loader with one and four I/O workers, validates that
every run publishes the same SHA-256 image and accounting, and emits cold/warm
p50 and p95 load time, GiB/s, RSS, worker count, and task count as JSON. Cold
samples use a platform cache-bypass/eviction facility; when none is available,
the cold report is `null` rather than warm data under a misleading label. The
process exits unsuccessfully if the four-worker p50 or p95 regresses the serial
baseline. `LFM_LOAD_BENCH_RUNS` changes the default five samples per mode.

Native Mimi now binds the Codec catalog of the model-owned combined image through
one model-lifetime `MimiDecodePlan`; it neither reopens the codec file nor owns a
duplicate image. Each conversation gets a `MimiDecodeState` containing only KV,
convolution carry, and scratch. Formula-derived codebooks and RoPE data live once
in the sealed plan. `mimi_decoder_new_from_file` remains only for isolated parity
tests. Mimi consumes checkpoint-layout F32 bytes directly, including unaligned
views, and reports formula-derived immutable bytes separately from its always-zero
compatibility-copy count.

## Provenance

The whole-file resident-block and span-planning approach was adapted from the
local `ember-ml` safetensors loader. Numerical UKM ingress was intentionally not
ported because model weights must remain byte-exact. JSON parsing uses the
MIT-licensed nlohmann/json 3.11.3 header vendored under `native/vendor/nlohmann/`.
