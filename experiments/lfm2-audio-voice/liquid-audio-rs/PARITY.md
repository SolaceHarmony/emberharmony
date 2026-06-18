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

## Tier 2 — FastConformer encoder (needs the weights) ✅ VERIFIED

1. Download the model locally (the Rust loader takes a local dir). The repo is
   public and Python-env-light here (torchaudio is import-only on this path; the
   dump scripts register a spec'd stub, so no torchaudio wheel is required):
   ```bash
   python -m pip install einops sentencepiece librosa   # no torchaudio needed
   python -c "from huggingface_hub import snapshot_download as d; d('LiquidAI/LFM2-Audio-1.5B', local_dir='model')"
   ```
2. Dump Python reference tensors (full encoder + stage intermediates):
   ```bash
   python parity/dump_reference.py ./model parity/golden
   python parity/dump_conformer_stages.py ./model parity/golden   # optional, for localization
   ```
3. Run the Rust parity tests (load the same weights + input, compare):
   ```bash
   LFM_MODEL_DIR=./model cargo test --test parity -- --ignored --nocapture
   ```

Result (LFM2-Audio-1.5B, f32, CPU): **mel 1.1e-5, conformer 8.3e-7**, and every
stage near-exact — conv-stack 5.6e-7, subsampling-out 1.0e-6, pos-emb 9.5e-7,
after-layer-0 1.1e-6, final 1.6e-6. Two bugs the harness caught and closed:
- the `lfm.*` weight keys (bare HF `Lfm2Model`, no `.model.` wrapper; final norm
  is `embedding_norm`) — fixed in `lfm2_hf.rs`;
- the conformer **length** the model actually feeds is the full mel width
  (`ChatState.add_audio` ⇒ `audio_in_lens = mel.shape[1]`), not the
  preprocessor's valid `mel_len`; so a single clip gets no intermediate
  `MaskedConvSequential` masking — which is exactly how the port encodes each
  segment individually. (The earlier reference used `mel_len` and showed a
  spurious 7% gap.)

The stage test asserts ≤ 5e-3 per stage; the front-end test asserts mel ≤ 5e-3,
conformer ≤ 2e-2.

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
- **FastConformer encoder**: ✅ verified end-to-end and per-stage (≤ 1.6e-6) — the
  manual rel-pos attention path, dw_striding subsampling, conv (batch_norm), and
  macaron FFN all match.
- **lfm backbone**: ✅ verified (6.6e-6) — `forward_embeds` vs `lfm(inputs_embeds)`
  over a 24-token sequence; hybrid short-conv + GQA attention + standard RoPE match.
- **text head**: ✅ verified (5.5e-6) — tied-embedding `text_logits` for the last
  position matches `F.linear(hidden, embed_tokens.weight)`.
- **depthformer**: ✅ verified token-exact — greedy 8-codebook audio frame for a
  fixed lfm-hidden vector; interleaved `rope_i` + per-codebook autoregression match.
  (This run found a latent 1-D `Linear` bug in the sampler.)
- **detokenizer (audio-out)**: not yet run — needs the `audio_detokenizer/` weights,
  which the 1.5B repo omits (it ships the v1 Mimi codec path instead). Deferred
  alongside the v1 `processor.mimi` decode.
