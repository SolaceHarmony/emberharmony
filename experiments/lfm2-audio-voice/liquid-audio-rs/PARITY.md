# Numerical parity harness

The port compiles and is structurally faithful (module-for-module, same function
lists / API). **Faithful = numerically matching**, which this harness verifies
against the Python `liquid_audio` with shared weights + fixed inputs.

## Workflow

1. Download the model locally (the Rust loader takes a local dir):
   ```bash
   huggingface-cli download LiquidAI/LFM2-Audio-1.5B --local-dir ./model
   ```
2. Dump Python reference tensors (needs `pip install liquid-audio`):
   ```bash
   python parity/dump_reference.py ./model parity/refs
   ```
3. Run the Rust parity test (loads the same weights + input, compares):
   ```bash
   LFM_MODEL_DIR=./model cargo test --test parity -- --ignored --nocapture
   ```

The test asserts relative error bounds per stage (mel ≤ 5e-3, conformer ≤ 2e-2).

## Coverage + how to extend

Currently the harness gates the deterministic **front-end**: the mel featurizer
and the FastConformer encoder. To widen coverage, add tensors in
`dump_reference.py` (prefill `in_emb`, `lfm` last-hidden, first text logits /
audio frame) and matching assertions in `tests/parity.rs` against the public
accessors (`conformer_encode`, and add accessors for prefill / step as needed).

## Known faithfulness gaps to validate/close with parity

- **Sampling**: text + audio use greedy (argmax); temperature/top-k (multinomial)
  is not yet ported — affects generation diversity, not the deterministic forward.
- **dtype**: weights load as f32 (Python runs bf16); expect small (≤1e-2) diffs
  from bf16. A bf16 load path would tighten parity at some op-support cost.
- **STFT/ISTFT**: rustfft vs torch.fft — verify mel/ISTFT bounds first; centering
  and window-padding conventions are the usual culprits.
- **RoPE**: backbone uses standard `rope`; the depthformer/`transformer.rs` uses
  interleaved `rope_i` — confirm against Python per-module.
