# Numerical parity harness

The port compiles and is structurally faithful (module-for-module, same function
lists / API). **Faithful = numerically matching**, which this harness verifies
against the Python `liquid_audio` with shared weights + fixed inputs.

## Tier 1 — mel featurizer (no weights, no big download) ✅ VERIFIED

The mel featurizer computes its window + slaney filterbank at init, so it needs
only the tiny `config.json` (committed at `parity/cfg/config.json`), not the
~3 GB checkpoint. The reference is dumped from the **real upstream NeMo code**
(loaded by file path so the torchaudio-importing package `__init__` is skipped):

```bash
pip install librosa                       # 0.11 installs without numba
python parity/dump_mel_reference.py       # writes parity/golden/mel_refs.safetensors
cargo test --test parity mel_parity -- --ignored --nocapture
```

Result: shapes match `(1, 128, 101)`, **rel-err 1.08e-5**. Finding: this caught a
real off-by-one — `torch.stft(center=True)` emits `1 + L/hop` frames (the trailing
one a masked pad column); the port now matches frame count, valid-range
normalization, and tail masking. The committed `mel_refs.safetensors` (116 KB)
makes the test re-runnable with no Python/network.

## Tier 2 — conformer encoder + backbone (needs the weights)

1. Download the model locally (the Rust loader takes a local dir):
   ```bash
   huggingface-cli download LiquidAI/LFM2-Audio-1.5B --local-dir ./model
   ```
2. Dump Python reference tensors (needs the full `liquid_audio` env):
   ```bash
   python parity/dump_reference.py ./model parity/golden
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

- **STFT/mel**: ✅ verified (rel-err 1.08e-5) against the upstream NeMo featurizer
  — centering + window-padding + valid-range normalization confirmed.
- **Sampling**: ✅ ported (greedy + temperature/top-k multinomial); deterministic
  greedy is what parity exercises, sampling adds diversity on top.
- **dtype**: the checkpoint is bf16; `DType::F32` loads those exact bf16 values
  upcast (lossless) — the parity reference is dumped at f32, so no dtype gap. CPU
  has no bf16 matmul kernel; bf16-in-memory is CUDA/Metal-only.
- **conformer + backbone**: still to run (Tier 2 — needs the ~3 GB weights). The
  encoder ports the manual-attention path; RoPE is standard `rope` on the backbone
  vs interleaved `rope_i` on the depthformer — confirm per-module with Tier 2.
