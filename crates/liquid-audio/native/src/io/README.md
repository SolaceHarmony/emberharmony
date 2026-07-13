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
| 64-byte-aligned LfmWeightImage allocation                         |
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

## Validation

The loader rejects malformed JSON, unsupported dtypes, shape/bit-count overflow,
incorrect byte counts, non-contiguous or overlapping spans, payload bytes not
described by the header, duplicate names across shards, unsafe shard paths, and
index-to-shard disagreement. No C++ exception crosses the C ABI.

## Current Migration State

`src/loader.rs` accepts an explicit checkpoint path from its host; the desktop host
gets that path and the selected device from persisted Tauri `VoiceSettings`. The
loader does not inspect `LFM_*` environment variables. It opens one
`ResidentWeights`, and `LFM2AudioModel` retains that owner for its complete life.

`src/compute/weights.rs` is the safe Rust boundary around `LfmWeightImage`. It
provides lifetime-bound immutable tensor views and one deliberately named
`candle_builder` compatibility adapter. Every tensor and byte copied through that
adapter is counted and reported at model load. On the current 1.5B checkpoint,
the native image is 2,940,724,032 bytes across 931 tensors; the not-yet-ported
Candle modules still copy 912 tensors / 2,940,616,960 bytes. Those numbers are
migration debt, not a zero-copy claim.

Native Mimi owns its own image inside `MimiDecoder`; Rust passes only the explicit
checkpoint pathname. The next model migration step is to bind the main image
directly into the Flashkern model schema, then remove compatibility copies as each
backbone, conformer, adapter, depthformer, and detokenizer component becomes native.

## Provenance

The whole-file resident-block and span-planning approach was adapted from the
local `ember-ml` safetensors loader. Numerical UKM ingress was intentionally not
ported because model weights must remain byte-exact. JSON parsing uses the
MIT-licensed nlohmann/json 3.11.3 header vendored under `native/vendor/nlohmann/`.
