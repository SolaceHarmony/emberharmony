# Parity goldens — provenance matters

**A parity golden is only meaningful if it was dumped from the SAME model + SAME
input the Rust test loads.** The Rust `*_parity` tests load `LFM_MODEL_DIR` (the
HF snapshot, `c362a0625…`). Several committed goldens were instead dumped from
`../model` (a *different* checkpoint — the `tokenizer-…checkpoint125` family), so
they encoded a different model's activations. The tests then "compared" the
snapshot-loaded Rust against a different checkpoint and reported spurious 5–22 %
"parity failures" that are NOT port bugs.

This was caught because `conformer_stages_parity` also had an **input** mismatch:
`dump_conformer_stages.py` generates its own `torch.manual_seed(0)` mel, but the
test feeds `refs.safetensors[mel]`. So it never actually compared the Rust and
Python conformer on the same input.

## Verified against Python, from the snapshot (real numbers)

Regenerating the goldens from the snapshot (weights the Rust loads) + the exact
test input shows the port is faithful:

| component   | dump script                          | result vs Python |
|-------------|--------------------------------------|------------------|
| mel         | `dump_mel_reference.py` (librosa)    | 1.18e-5 (already passing — weight-independent) |
| conformer   | `dump_conformer_on_refmel.py`        | conv_out 5.4e-7, sub 1.4e-6, final 4.0e-7 |
| depthformer | `dump_depthformer_from_snapshot.py`  | greedy tokens EXACT `[213,836,182,416,782,1111,1790,660]` |

Both new dump scripts load only the **pure-torch** module chains (conformer:
`encoder`/`mha`/`modules`/`subsampling`/`utils`; depthformer: `transformer.py`) via
a synthetic package, so they run with just `torch`+`safetensors` (no
`transformers`/`librosa`/`accelerate`, which the snapshot's `Lfm2Model` needs).

## All goldens regenerated from the snapshot — full suite green vs Python

After installing the HF Lfm2 stack (`transformers` 5.x + `accelerate` + `librosa` +
`sentencepiece`) and regenerating every golden from the snapshot, all 12 `*_parity`
tests pass at f32 precision — confirming the earlier reds were 100% golden-source
mismatches, not port bugs:

| test                | vs Python |
|---------------------|-----------|
| mel_parity          | 1.18e-5 |
| conformer_stages    | conv_out 5.4e-7 … final 4.0e-7 |
| front_end_parity    | conformer 4.76e-7 |
| prefill_parity      | 1.07e-6 |
| backbone_parity     | 6.3e-6 (text_logits 5.1e-6) — the HF `Lfm2Model` vs the moshi port |
| depthformer_parity  | greedy tokens EXACT |

### Regenerate from the snapshot
```
SNAP=~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B/snapshots/c362a0625…
PY=<python with torch + transformers(Lfm2) + accelerate + librosa + sentencepiece>
$PY parity/dump_reference.py  "$SNAP" parity/golden   # refs (mel + conformer)
$PY parity/dump_prefill.py    "$SNAP" parity/golden   # prefill_refs[in_emb]
$PY parity/dump_backbone.py   "$SNAP" parity/golden   # backbone_refs
$PY parity/dump_depthformer.py "$SNAP" parity/golden  # depthformer_refs
$PY parity/dump_conformer_on_refmel.py "$SNAP"        # conformer_stages (reads new refs[mel])
```
The conformer/depthformer pure-torch dumps (`dump_conformer_on_refmel.py`,
`dump_depthformer_from_snapshot.py`) also run with just `torch`+`safetensors`.

**Always regenerate the goldens from the SAME model dir (`LFM_MODEL_DIR`) the tests
load.** The committed goldens here are from snapshot `c362a06…`.

## Realtime Moshi trace parity

The desktop Moshi path is frame-fed rather than prompt-fed, so its parity check is
a trace pair instead of a safetensors golden. Dump the upstream Python
`server.py` order and the native Rust order from the same checkpoint files and
the same 24 kHz PCM input:

```
conda activate py312
python parity/dump_moshi_realtime.py <python-moshi-model> input-24khz.wav /tmp/py-moshi.json --greedy --frames 16
MOSHI_GREEDY=1 MOSHI_TRACE_FRAMES=16 cargo run --release --example moshi_realtime_trace -- <candle-moshi-dir> input-24khz.wav /tmp/rs-moshi.json
python parity/compare_moshi_realtime.py /tmp/py-moshi.json /tmp/rs-moshi.json
```

For a cheap checkpoint/remap smoke test, use `--verify-remap-only`; it validates
the Candle-to-Python depformer key mapping from the safetensors header and writes
cheap file metadata without loading model tensors or computing full-file hashes.
Use `--load-only` when you intentionally want to instantiate the vendored Python
model, apply the remapped weights, and write exact FNV fingerprints. For a single
stepping smoke test, pass `--warmup-frames 0 --frames 1`; the canonical
`server.py` parity path keeps the default four warmup frames.

The comparator requires matching Moshi/Mimi/tokenizer byte fingerprints by
default. The current Rust side supports the unconditioned Candle Moshi layout
only (for example `kyutai/moshiko-candle-bf16`). The Python trace dumper remaps
that Candle depformer layout into the vendored Python module names before
loading, so same-file parity can run against the local Rust checkpoint. If you
intentionally compare a PyTorch/Candle converted pair, pass
`--allow-converted-checkpoints`; do not call that a same-checkpoint parity run.

If a config advertises Liquid's conditioning/CFG fuser path, the native loader
must reject it rather than run with missing `condition_tensors` or CFG
semantics.
