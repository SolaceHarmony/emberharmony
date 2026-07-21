# Architecture 1 — The Mimi Codec

> **Current product boundary:** Mimi remains a native codec implementation for
> the future Moshi tranche. Released LFM2.5 uses its required
> `audio_detokenizer/` graph and has no Mimi loader, request, state, or fallback.
> Sections below preserve upstream and former Rust-port archaeology; current
> native ownership is specified by `MIMI_PORT.md` and
> `AUDIO_DETOKENIZER_PORT.md`.

Scope: what Mimi *is*, how it's built and loaded, the encode/decode signal path, its
device/CUDA story, and the exact Python→Rust mapping. Hand-traced from the source
(`upstream-liquid-audio/src/liquid_audio/...` and `liquid-audio-rs/...`), not summarized.

---

## 1. What Mimi is, and where it sits in LFM2-Audio

Mimi is **Kyutai's neural audio codec** — a learned waveform⇄discrete-token transform.
It is **not** the LFM2-Audio model; it is a *peripheral* the processor owns. The
processor (`processor.py`) exposes **two independent audio-out facilities**:

| Field | Class | Role | Python prop |
|---|---|---|---|
| `_mimi` | `MimiModel` (Kyutai) | RVQ codec: waveform → codes, codes → waveform | `processor.py:101-119` |
| `_audio_detokenizer` | `LFM2AudioDetokenizer` | LFM2 ISTFT vocoder: codes → waveform (LFM2.5) | `processor.py:121-163` |

**Three jobs Mimi actually does** (and the one it does *not*):

1. **Audio-OUT encode for training data** — `LFM2AudioChatMapper._encode_audio_out`
   calls `processor.mimi.encode(wav)` to turn reference speech into the 8-codebook
   `audio_out` target codes (`data/mapper.py:229`). *This is the canonical source of
   the model's audio vocabulary.*
2. **Audio-OUT decode in the demo** — `demo/chat.py:34` `mimi.decode(t[None,:,None])`
   turns each generated 8-code frame back into 24 kHz audio, inside `mimi.streaming(1)`.
3. **Historical upstream fallback** — older processor/client paths used Mimi
   when a model shipped no `audio_detokenizer/` weights. The native product
   deliberately does not preserve this cross-model fallback.
4. **NOT audio-IN.** Mic audio entering the model goes through the **conformer mel
   front-end** (`processor.py:226-250`), never Mimi. Mimi-encode is only for building
   `audio_out` *targets*, not for feeding the model's *input*.

For LFM2.5 the preferred audio-OUT path is the LFM2 detokenizer:
`processor.decode()` (`processor.py:165-177`) dispatches to `self.audio_detokenizer`,
**not** Mimi — and rejects codes outside `[0, 2047]`.

---

## 2. MimiModel architecture (the codec internals)

Built by `moshi.models.loaders.get_mimi` (`loaders.py:296-333`). All hyperparameters
are the module-level dicts `_seanet_kwargs` / `_quantizer_kwargs` / `_transformer_kwargs`
(`loaders.py:38-80`). Rates: `SAMPLE_RATE=24000`, `FRAME_RATE=12.5` (`loaders.py:28-29`).

```
 waveform (B,1,T) @ 24 kHz
   │
   ▼  SEANetEncoder            _seanet_kwargs: dim=512, n_filters=64, kernel=7,
   │   ratios [8,6,5,4]         residual_kernel=3, ELU, compress=2, causal, norm="none"
   │   ⇒ hop_length = 8·6·5·4 = 960  ⇒ encoder rate = 24000/960 = 25 Hz
   ▼  latent (B,512, T/960)  @ 25 Hz
   │
   ▼  encoder_transformer      ProjectedTransformer: d_model=512, 8 heads, 8 layers,
   │   (causal, RoPE)           context=250, layer_scale=0.01, dim_ff=2048
   ▼  (B,512, T/960)  @ 25 Hz
   │
   ▼  _to_framerate → downsample   ConvDownsample1d, stride = 25/12.5 = 2  (learnt conv)
   ▼  latent (B,512, T/1920) @ 12.5 Hz          ── compression.py:267-278
   │
   ▼  SplitResidualVectorQuantizer  dim=256, n_q=32, bins=2048; active = set_num_codebooks(8)
   ▼  CODES (B, 8, T/1920)  ints ∈ [0,2047]      ── quantizer.encode, compression.py:387
══════════════════════════ encode │ decode ══════════════════════════
   ▼  quantizer.decode(codes) → latent (B,512, ·) @ 12.5 Hz   ── compression.py:431-433
   ▼  _to_encoder_framerate → upsample   ConvTrUpsample1d, stride 2 (channel-wise)
   ▼  (B,512, ·) @ 25 Hz                         ── compression.py:280-291
   ▼  decoder_transformer (causal, RoPE)
   ▼  SEANetDecoder (mirror ratios)  ⇒ ×960 upsample
   ▼  waveform (B,1,T') @ 24 kHz
```

**The 1920.** `frame_size = sample_rate / frame_rate = 24000 / 12.5 = 1920` samples
(`compression.py:244-246`). One Mimi frame = one 8-code column = **1920 audio samples**.
That is exactly the `t.numel() == 1920` chunk the demo plays (`chat.py:85`) and the
`MIMI_RATE` chunking in the Rust `mic_chat.rs`.

**The quantizer (SplitResidualVectorQuantizer).** `n_q=32` total codebooks of
`bins=2048` each, latent projected `512→256` for VQ. It is *split*: `rvq_first`
(the **semantic** codebook, `n_q_semantic=1`) + `rvq_rest` (the **acoustic** codebooks).
`set_num_codebooks(8)` (`loaders.py:332`) activates 8 → **1 semantic + 7 acoustic**,
giving codes `(B,8,T)`. The `rvq_first`/`rvq_rest` split is also why the **weight-key
naming** is `quantizer.rvq_first.*` / `quantizer.rvq_rest.*` — load-bearing for Rust §5.

**Codebook delays.** Note the `_lm_kwargs.delays = [0,0,1,1,...]` in `loaders.py:110`
is the **Moshi 7B LM's** acoustic-delay pattern — it belongs to `get_moshi_lm`, **not**
to LFM2-Audio and **not** to Mimi itself. LFM2-Audio's interleave/codebook cadence is
its own (`config.json: depthformer`, 6 layers/1024 dim). Do not conflate the two.

---

## 3. Loading

**Python** (`processor.py:101-119`): lazy. First access to `.mimi` →
`get_mimi(None, device=self.device)` builds the modules **uninitialized**
(`loaders.py:325` skips load when `filename is None`), then
`safetensors.torch.load_file(tokenizer-e351c8d8-checkpoint125.safetensors)` +
`load_state_dict(strict=True)`. The checkpoint filename is the constant
`MIMI_NAME` (`loaders.py:34`), pointed at by `processor.py:67`. Weights are loaded
**onto `self.device`** (`processor.py:114`).

**Training note:** SEANet `norm="none"` (`loaders.py:53`) — weight-norm is folded
into plain convs at export, so inference loads ordinary conv weights.

---

## 4. Device & CUDA

| Aspect | Python | Rust |
|---|---|---|
| Default device | `from_pretrained(device="cuda")` (`processor.py:61`); LFM2 detok hard-`.cuda()` (`processor.py:151`) | `device: &Device` everywhere; persistent weight dtype from safetensors; CPU BF16 via NEON, Metal opt-in |
| CUDA graphs | `_MimiState` wraps encoder/decoder/transformers in `CUDAGraphed` (`compression.py:99-102,219-230`) — **`disable = device.type != 'cuda'`** (`compression.py:221`), so graphs engage **only on CUDA** | none — candle ops directly |
| Attention | `F.scaled_dot_product_attention` + `torch.compile` (CUDA-gated) in the codec transformer | eager matmul + mask + softmax (`moshi` crate) |
| Custom kernels | none in the vendored codec (no `causal_conv1d`/`flash_attn`/triton) — stock torch SDPA | none |

So "the CUDA kernels involved" in the codec are **CUDA graph capture**
(`torch.cuda.CUDAGraph` via `CUDAGraphed`, `utils/compile.py`) + cuDNN/SDPA backends —
not bespoke `.cu` kernels. On CPU all of that is disabled and it runs eager.
**`candle-flashfftconv` (bf16×2 FFT kernels) is unrelated and unwired** — zero refs in
`liquid-audio-rs`.

---

## 5. Streaming (the real-time decode contract)

`MimiModel` is a `StreamingModule`. Two modes:

- **One-shot** (`_streaming_state is None`): `decode`/`encode` run the full module;
  `encode` first `pad_for_conv1d(x, frame_size, frame_size)` (`compression.py:358`).
- **Streaming** (`with mimi.streaming(1)`): conv/transformer state persists across
  calls; **input length must be an exact multiple of `frame_size=1920`**
  (`compression.py:361-365`) — "you are responsible for buffering." The demo decodes
  exactly one frame at a time inside the context (`chat.py:21,34`).

**Rust mirrors this precisely** (`audio_out.rs:74-119`):
- `MimiDetokenizer { inner: RefCell<moshi::mimi::Mimi> }` — the `RefCell` is the
  interior-mutability analog of the Python streaming-state mutation.
- `reset_stream()` → `Mimi::reset_state()` (the turn boundary, = entering
  `mimi.streaming(1)` fresh).
- `decode_step(frame (1,8,1))` → `Mimi::decode_step(StreamTensor, StreamMask)` keeps
  state across calls (the warmup latency means first call(s) return `None`) — this is
  the gapless real-time path `mic_chat.rs` uses.
- `decode()`/`encode()` `reset_state()` first → independent one-shot calls.

---

## 6. Python → Rust mapping (function-level)

| Concern | Python (file:line / symbol) | Rust (file:line / symbol / crate) |
|---|---|---|
| Codec model | `MimiModel` (`compression.py:105`) | `moshi::mimi::Mimi` (the `moshi` crate) |
| Build/config | `get_mimi` + `_seanet/_quantizer/_transformer_kwargs` (`loaders.py:296,38-80`) | `moshi::mimi::Config::v0_1(codebooks)` + `moshi::mimi::load` |
| Load weights | `load_file` + `load_state_dict(strict=True)` (`processor.py:111-115`) | `load_mimi` → `moshi::mimi::load(path, Some(cb), dev)` (`loader.rs:296-303`) |
| Checkpoint name | `MIMI_NAME="tokenizer-e351c8d8-checkpoint125.safetensors"` (`loaders.py:34`) | same filename string (`loader.rs`) |
| encode | `MimiModel.encode` (`compression.py:376-388`) | `MimiDetokenizer::encode` → `Mimi::encode` (`audio_out.rs:98-102`) |
| decode (one-shot) | `MimiModel.decode` (`compression.py:406-429`) | `MimiDetokenizer::decode` → `Mimi::decode` (`audio_out.rs:88-93`) |
| decode (streaming) | `mimi.streaming(1)` + per-frame `decode` (`chat.py:21,34`) | `decode_step` → `Mimi::decode_step` (`audio_out.rs:113-118`) |
| reset stream | `mimi.streaming(...)` ctx / `reset_streaming` | `reset_stream` → `Mimi::reset_state` (`audio_out.rs:105-107`) |
| codebooks | `set_num_codebooks(8)` (`loaders.py:332`) | `Some(codebooks)` to `moshi::mimi::load` |
| quantizer keys | `quantizer.rvq_first.*` / `rvq_rest.*` | matched natively by `moshi::mimi` (candle-transformers' Mimi can't load these) |
| backend dispatch | `processor.decode` → `audio_detokenizer` else `.mimi` | `Box<dyn AudioDetokenizer>`; `processor.rs` picks `audio_out.or(mimi)` |

The Rust `AudioDetokenizer` trait (`audio_out.rs:25-62`) is the design seam: `decode`
required; `encode` defaults to an error (only Mimi is an encoder — faithful to Python,
where only `MimiModel` exposes `encode`); `decode_step` defaults to one-shot.

---

## 7. Divergences (honest)

1. **Codec source.** Python uses the *vendored* `liquid_audio/moshi` `MimiModel`; Rust
   reuses the **published `moshi` crate** (Kyutai's own port, same algorithm + weight
   keys), not a candle-transformers vendor — chosen specifically because this
   checkpoint uses `rvq_first`/`rvq_rest` naming.
2. **Device.** Python is CUDA-coupled (won't boot CPU-only); Rust is device-agnostic.
3. **CUDA graphs.** Python captures CUDA graphs for the codec on GPU; Rust has no graph
   layer (candle eager). Numerically irrelevant; latency-relevant only.
4. **bf16 vs f32.** Python codec runs in module fp32/bf16 on CUDA; Rust uses F32 on CPU,
   bf16 on Metal.

---

## 8. How the codes connect to the LFM2-Audio token flow

The model never sees waveforms on the output side — it emits **8-codebook frames**
(`audio frame (8,)`, values `[0,2048]`, where `2048` = EOAudio). Those frames are
compatible with Mimi's `(B,8,T)` code layout (one frame = one timestep column).
Mimi defines the training/older-codec side of that vocabulary; released LFM2.5
inference uses its separately trained audio detokenizer:

```
  training:  reference wav ──mimi.encode──► audio_out codes (8,L) ──► model targets
  LFM2.5:    model audio head ──► frame (8,) ──audio detokenizer──► wav @24k
  Moshi:     future native Moshi route ────────mimi.decode───────► wav @24k
```

The semantic codebook (index 0, `rvq_first`) is why LFM2-Audio's audio loss upweights
codebook 0 (`audio_loss_weights`, `lfm2_audio.py:104`) — losing the semantic code costs
the most.

---

*Next: ARCH_2 (LFM2 weights, loading, inference path, audio libs), ARCH_3 (turn
detection), ARCH_4 (concurrency / full-duplex) — each hand-traced the same way.*
