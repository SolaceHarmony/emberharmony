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

## Still sourced from `../model` (need regeneration from the snapshot)

`backbone_refs` (5.6 %), `prefill_refs[in_emb]` (20 %), `refs[conformer]` used by
`front_end_parity` (22 %). These touch the HF `Lfm2Model` backbone, so regenerating
them needs the full Python stack (`transformers` with Lfm2 support + `accelerate` +
`librosa`) loading the **snapshot**. The conformer + depthformer evidence (1e-6 /
exact once the golden matches the loaded weights) strongly indicates these are the
same golden-source mismatch, not port regressions — but that should be confirmed by
regenerating, not assumed.

To regenerate everything from the snapshot once the stack is available:
```
<py-with-Lfm2> parity/dump_reference.py   <snapshot> parity/golden   # refs + conformer + backbone
<py-with-Lfm2> parity/dump_prefill.py     <snapshot> parity/golden
<py-with-torch> parity/dump_conformer_on_refmel.py       <snapshot>  # conformer_stages (pure torch)
<py-with-torch> parity/dump_depthformer_from_snapshot.py <snapshot>  # depthformer  (pure torch)
```
