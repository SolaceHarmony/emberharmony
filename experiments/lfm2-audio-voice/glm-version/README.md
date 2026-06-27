# `glm-version/` — Rust-side architecture docs for `liquid-audio-rs`

This folder mirrors the `ARCH/` tree (which documents the **Python** source of
Liquid AI's `liquid_audio`) with **Rust-first** companions for the
`liquid-audio-rs` port. Each file documents the Rust source file(s), how the
port works in Rust, and — most importantly — **where the port deliberately
diverges from the Python and why**.

The original `ARCH/` files were created by Claude on the Python code. These
companions are the Rust-side view: same topic, different language, focused on
the differences a Rust reader (or future GLM session) needs to know. They live
under `glm-version/` so they are safe from touches to the `ARCH/` tree.

## Layout

```
glm-version/
├── utils.md                         # LFMModality, mel2emb_len, module_exists, get_model_dir
├── processor.md                     # LFM2AudioProcessor + ChatState (the I/O container)
├── detokenizer.md                   # LFM2AudioDetokenizer + Istft (the LFM2 ISTFT vocoder)
├── trainer.md                       # Trainer (the supervised fine-tuning driver)
├── model/
│   ├── mlp.md                       # MLP (the audio_adapter 512→2048)
│   ├── transformer.md               # RawLmBackbone (the depthformer)
│   ├── lfm2_audio.md                # LFM2AudioModel (the top-level orchestrator)
│   ├── lfm2_backbone.md             # lfm2_hf::Model (the HF Lfm2Model backbone)
│   └── conformer/
│       ├── utils.md                 # CacheAwareStreamingConfig + stochastic-depth (off-path)
│       ├── subsampling.md           # ConvSubsampling (the 8× pre-encoder)
│       ├── mha.md                    # RelPositionalEncoding + RelPositionMultiHeadAttention
│       ├── modules.md               # ConformerLayer + ConformerConvolution + CausalConv1D
│       ├── encoder.md               # ConformerEncoder (the audio-IN front-end)
│       └── processor.md             # FilterbankFeatures (the mel front-end)
├── data/
│   ├── types.md                     # ChatMessage + the six-tensor bundles
│   ├── dataloader.md                 # LFM2DataLoader + lfm2_collator
│   ├── mapper.md                     # LFM2AudioChatMapper (chat → training sample)
│   └── preprocess.md                # preprocess_dataset + arrow_io (the Arrow build)
├── moshi/
│   ├── README.md                     # overview: reused via the `moshi` crate, not re-ported
│   └── STATUS.md                     # per-file status table (on-path codec vs off-path LM/transport)
└── demo/
    ├── chat.md                       # mic_chat.rs (the realtime speech-to-speech demo)
    └── model.md                      # demo/model.py — NOT ported (loader.rs/mic_chat.rs replace it)
```

## As-built changes by Claude

[`AS_BUILT_claude_changes.md`](AS_BUILT_claude_changes.md) documents the
threading, bf16 BFMMLA kernel, `Send` fixes, mask memoization, and `to_vec4`
extension Claude made to `liquid-audio-rs` across multiple sessions. These are
**not** part of the original Python port — they are execution-model parity work
(matching torch's intra-op thread policy, closing candle's CPU bf16-matmul gap,
and memoizing causal masks to eliminate per-call construction cost). A prior
zero-copy `KvCache` swap was reverted as a deviation from the reference. The
relevant per-module docs (`mlp.md`, `lfm2_backbone.md`) have been updated with
cross-references to the as-built doc.

## What these docs are (and aren't)

- **Are:** Rust-first architecture docs. Each file documents the Rust source
  file's role, how it works in Rust, its dtypes/shapes, its wiring, and a
  Python↔Rust diff table explaining every deliberate divergence and why.
- **Aren't:** a rehash of the Python. The `ARCH/` tree already covers the
  Python in depth; these docs cross-reference it and focus on the Rust
  differences. Read the `ARCH/` file for the Python's full mechanism; read the
  `glm-version/` file for what the Rust port actually does and where it
  diverges.

## The recurring Rust divergences

Most Rust files share the same set of deliberate divergences from the Python.
They are documented per-file but summarized here so the pattern is visible:

1. **Device-agnostic (§2.1).** Nothing in `src/` hardcodes a device. Every
   loader takes `device: &Device` + `dtype: DType`; examples default to
   `(Cpu, F32)`, Metal is opt-in (`LFM_DEVICE=metal` → bf16). The Python
   hard-codes `device="cuda"`/`dtype=bf16` and won't boot CPU-only. This is what
   makes the Rust port actually deliver LFM2's "runs on CPU" design point.
2. **Kernel-free (§2.2).** CUDA-gated kernels (`flash_attention_2`, `sdpa`,
   `causal_conv1d`) → portable candle ops (eager matmul + additive causal mask
   + softmax; `Conv1d` + gather-mul-sum). The eager SDPA matches the
   `sdpa`/no-flash math the f32 goldens were dumped from — *not* flash-attn's
   reordered online-softmax.
3. **Differentiable basic ops over fused no-bwd kernels.** `candle_nn::RmsNorm`,
   `softmax_last_dim`, `rope`, and `ops::sdpa` all take a fused
   `apply_op*_no_bwd` path that **severs autograd**. The port uses the
   basic-op differentiable equivalents (`ops::layer_norm_slow`,
   `ops::softmax`, `rope_slow`/`rope_i_slow`, hand-rolled SDPA) wherever the
   graph is trained (backbone, depthformer, conformer). Same forward values;
   only the backward path differs.
4. **Inheritance → composition; ABCs → traits; string literals → enums.**
   Python's class hierarchy (`PositionalEncoding ← RelPositionalEncoding`,
   `MultiHeadAttention ← RelPositionMultiHeadAttention`, `SequenceModel` ABC)
   becomes Rust structs holding their base + traits. `Literal["text"]` becomes
   `SegmentKind::Text`. The `AudioDetokenizer` trait is the codec seam.
5. **`@dataclass(frozen=True)` → plain structs with `pub` fields.** Rust has
   no `@dataclass`; the immutability becomes "owning constructor + read-only
   `pub` fields."
6. **`@cache` / `@property` / `register_buffer` → plain fields / recomputed
   tables.** Rust has no `@cache` (callers resolve once and hold the `PathBuf`),
   no `@property` (backends are `Option<Box<dyn …>>` built at load), no
   `register_buffer` (config is plain fields, tables are recomputed per call).
7. **Exception → `Result`/`panic!`.** Python `ValueError`/`RuntimeError`
   become `Result<T>` (recoverable) or `panic!`/`assert!` (construction-time
   invariants).
8. **Generator → callback stream.** `generate_interleaved`/`generate_sequential`
   take `FnMut(GenToken)` instead of `yield`ing (sync streaming; async lives
   only at the transport, per the design).
9. **Moshi reuse (§2.3).** The vendored `liquid_audio/moshi/**` is **reused
   via the `moshi` crate** (Kyutai's own Rust port), not re-ported in-tree.
   `audio_out.rs::MimiDetokenizer` is the thin adapter; the `AudioDetokenizer`
   trait is the seam.
10. **Cross-library f32 floor (§1.4).** The ~1e-6 residual vs Python is
    irreducible (candle gemm reduction order, libm transcendentals, FFT
    algorithm). The depthformer audio frame is **token-exact** (no float
    reduction in argmax/gather). Bit-exact where there is no float reduction;
    f32-floor where there is.

## Parity (from PYTHON_VS_RUST.md §1.2 + PARITY.md)

| Stage | Rust vs Python | Shape |
|---|---|---|
| LFM2 backbone hidden state | **6.558e-6** | `[1,24,2048]` |
| Text logits (tied head) | **5.505e-6** | `[65536]` |
| Conformer conv-subsampling | 5.611e-7 | `[1,256,13,16]` |
| Conformer post-subsample / pos-enc | 1.019e-6 | `[1,13,512]` |
| Conformer final | **8.25e-7** | `[1,512,13]` |
| Mel spectrogram (front-end) | **9.31e-6** (FFT-library floor) | `[1,128,101]` |
| Prefill embeddings (modality scatter) | **1.118e-6** | `[1,50,2048]` |
| **Depthformer audio frame** | **token-EXACT** `[213,836,182,416,782,1796,202,578]` | — |

## How to read these docs

1. Start with [`utils.md`](utils.md) — it's the smallest and establishes the
   `LFMModality`/`mel2emb_len`/`get_model_dir` vocabulary every other file
   uses.
2. Read [`model/lfm2_audio.md`](model/lfm2_audio.md) for the top-level
   orchestrator (prefill + generate + the depthformer inner loop).
3. Drill into the conformer (`model/conformer/`) for the audio-IN front-end,
   the backbone (`model/lfm2_backbone.md`) for the brain, and the depthformer
   (`model/transformer.md`) for the audio-OUT head.
4. [`moshi/README.md`](moshi/README.md) explains why the moshi tree is reused
   not re-ported.
5. The `ARCH/` originals are the Python-side companions — same topic, different
   language.

## Cross-references

- `ARCH/` — the Python-first architecture docs (Claude's originals).
- `liquid-audio-rs/PYTHON_VS_RUST.md` — the port report (where we are the same,
  where we differ, why).
- `liquid-audio-rs/PORT_STATUS.md` — the 38/38 class + 170/170 symbol
  inventory.
- `liquid-audio-rs/parity/PARITY.md` — the numerical parity harness workflow +
  results.