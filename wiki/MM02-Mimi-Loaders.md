<!-- topic: Mimi Codec — Models -->
# MM02 · get_mimi factory + CheckpointInfo
**Code:** `MM02` · **Source:** `moshi/models/loaders.py` · **Rust:** `moshi crate mimi::load / Config::v0_1` · **On the LFM2-Audio inference path:** yes

## Role
`loaders.py` is the **factory + checkpoint-resolution layer** for the Kyutai Mimi neural audio codec (and, off-path, the Moshi 7B LM). It owns the frozen hyperparameter dicts (`_seanet_kwargs`/`_quantizer_kwargs`/`_transformer_kwargs`, `loaders.py:38-80`) and the rate constants `SAMPLE_RATE=24000` / `FRAME_RATE=12.5` (`loaders.py:28-29`). On the LFM2-Audio path only **`get_mimi`** (`loaders.py:296-333`) matters: it assembles a `MimiModel` from SEANet enc/dec + two `ProjectedTransformer`s + a `SplitResidualVectorQuantizer`, then optionally loads weights. Everything else here (`CheckpointInfo`, `get_moshi_lm`, the conditioner/LoRA/fuser helpers) services the standalone Moshi 7B and is **not** reached by LFM2-Audio, which uses its own backbone + depthformer and only borrows Mimi as a peripheral codec for audio-out codes.

## How it works
**`get_mimi(filename, device, num_codebooks=8)` — the only on-path entry (`loaders.py:296-333`).** Construction is config-driven from three module dicts; nothing is computed at call time except the encoder/decoder rate bridge.

1. **Build the modules** (`loaders.py:300-310`):
   - `SEANetEncoder(**_seanet_kwargs)` / `SEANetDecoder(**_seanet_kwargs)` — `dimension=512`, `n_filters=64`, `kernel_size=7`, `residual_kernel_size=3`, `last_kernel_size=3`, `ratios=[8,6,5,4]`, `n_residual_layers=1`, `activation="ELU"`, `compress=2`, `dilation_base=2`, `causal=True`, `pad_mode="constant"`, `true_skip=True`, **`norm="none"`** (`loaders.py:38-57`). `norm="none"` is load-bearing: training used weight-norm but the **exported checkpoint has weight-norm folded into plain conv weights**, so inference instantiates ordinary `Conv1d`s (`loaders.py:51-53`). The ratios product `8·6·5·4 = 960` is the encoder `hop_length` ⇒ encoder framerate `24000/960 = 25 Hz`.
   - `ProjectedTransformer(device=device, **_transformer_kwargs)` ×2 (encoder + decoder side) — `d_model=512`, `num_heads=8`, `num_layers=8`, `causal=True`, `layer_scale=0.01`, `context=250`, `conv_layout=True`, `max_period=10000`, `gating="none"`, `norm="layer_norm"`, `positional_embedding="rope"`, `dim_feedforward=2048` (`loaders.py:65-80`). LayerNorm (not RMSNorm) + RoPE(θ=10000) + LayerScale-0.01; SDPA attention with `1/sqrt(head_dim)` scale, `head_dim = 512/8 = 64`. These run at **25 Hz** between SEANet and the framerate resampler.
   - `SplitResidualVectorQuantizer(**_quantizer_kwargs)` — `dimension=256`, `n_q=32`, `bins=2048`, `input_dimension=output_dimension=512` (`loaders.py:58-64`). Latent is projected `512→256` for VQ. "Split" = `rvq_first` (1 **semantic** codebook) + `rvq_rest` (acoustic codebooks); that split is why the checkpoint keys are `quantizer.rvq_first.*` / `quantizer.rvq_rest.*`.

2. **Assemble `MimiModel`** (`loaders.py:311-323`): wires enc/dec/quantizer + both transformers, `channels=1`, `sample_rate=24000`, `frame_rate=12.5`, **`encoder_frame_rate = SAMPLE_RATE / encoder.hop_length = 24000/960 = 25`**, `causal=True`, `resample_method="conv"`. The 25↔12.5 Hz mismatch makes `MimiModel` insert a **learnt strided conv** framerate bridge (`ConvDownsample1d` stride 2 on encode, `ConvTrUpsample1d` stride 2 on decode). `.to(device)` then `.eval()`.

3. **Load weights, or not** (`loaders.py:325-332`): if `filename is None` the modules stay **uninitialized** (random) — this is exactly how the LFM2-Audio processor calls it (`processor.py:113` `get_mimi(None, device=...)`), then loads weights itself via `safetensors.torch.load_file(...)` + `load_state_dict(strict=True)` (`processor.py:114-115`). If `filename` is given and is `.safetensors`, `load_file(..., device=str(device))` + `load_state_dict`; else `torch.load(...)["model"]`. Either way, `set_num_codebooks(num_codebooks)` (`loaders.py:332`) caps the active RVQ depth to **8** (1 semantic + 7 acoustic) out of the 32 trained, yielding code columns `(B,8,T)`.

**`num_codebooks` derivation (`CheckpointInfo.get_mimi`, `loaders.py:257-264`).** Off the LFM2-Audio path: defaults to 8 with no config, else `max(dep_q, n_q − dep_q)` from the Moshi LM config (8 for the default `_lm_kwargs`), halved if `tts_config.multistream`. LFM2-Audio never goes through `CheckpointInfo`; its processor passes the literal 8.

**`get_mimi` is a pure builder** — there is no forward pass here. The actual signal path (SEANet → enc-transformer → downsample → RVQ → upsample → dec-transformer → SEANet⁻¹) lives in [compression.py](MM01-Mimi-Codec); `loaders.py` only fixes the geometry. The one piece of math it pins is the framerate bridge stride `encoder_frame_rate / frame_rate = 25/12.5 = 2`.

**Frame geometry that the rest of the system depends on:** `frame_size = sample_rate/frame_rate = 24000/12.5 = 1920` samples per Mimi frame = one 8-code column. (`frame_size` itself is computed in `compression.py`, but it follows directly from these two constants.)

**Off-path (do not implement for LFM2-Audio):** `CheckpointInfo.from_hf_repo` (`loaders.py:169-255`) HF-downloads config + sub-checkpoints; `get_moshi_lm` (`loaders.py:336-416`) builds the Moshi 7B `LMModel` with meta-device init + per-key dtype casting (`condition_provider.*`/`fuser.*` forced to f32, rest to model dtype — `loaders.py:386-391`); `get_conditioner*`/`get_condition_fuser`/`get_lora_moshi` are Moshi-LM conditioning/LoRA. The `_lm_kwargs.delays=[0,0,1,1,...]` (`loaders.py:110`) is the **Moshi LM's** acoustic-delay pattern, **not** Mimi's and **not** LFM2-Audio's — do not conflate.

## Dtypes & shapes
`get_mimi` is a constructor; "shapes" here are the module geometry it fixes and the tensors that subsequently flow through the built `MimiModel`.

| Stage | dtype | shape / value |
|---|---|---|
| `get_mimi` inputs | — | `filename: Path\|None`, `device`, `num_codebooks=8` (int) |
| Mimi weights on disk | **bf16** (`DEFAULT_REPO=...-bf16`) | safetensors state dict |
| Mimi compute (Python) | module dtype (default **bf16** on cuda; f32 on cpu) | — |
| Mimi compute (Rust) | **f32** on CPU (no CPU bf16 matmul), **bf16** on Metal | — |
| encode input waveform | **f32** | `(B,1,T)` @ 24 kHz |
| SEANet latent | model dtype | `(B,512,T/960)` @ 25 Hz |
| after downsample (12.5 Hz) | model dtype | `(B,512,T/1920)` |
| RVQ codes (encode out) | **int** (u32 in Rust) ∈ `[0,2047]` | `(B,8,T/1920)` |
| decode input codes | **int / u32** ∈ `[0,2047]` | `(B,8,T)` |
| decode output waveform | **f32** | `(B,1,T')` @ 24 kHz |
| rate constants | — | `SAMPLE_RATE=24000`, `FRAME_RATE=12.5`, `frame_size=1920`, `hop_length=960`, `encoder_frame_rate=25`, downsample/upsample stride `=2` |

Internal promotions inside the built codec: LayerNorm/softmax upcast to f32 then cast back; RVQ `cdist`/argmin and the residual loop run in the latent dtype; embedding/gather/index ops are exact (no float reduction). EOAudio (`2048`) is **not** a Mimi code — Mimi only emits `0..2047`; `2048` is the LFM2-Audio audio-head end token added downstream.

## Wiring
**Upstream (what feeds this):**
- [core_processor](CO01-Processor-ChatState) calls `get_mimi(None, device)` lazily on first `.mimi` access (`processor.py:113`), then injects the loaded bf16→model-dtype state dict. `get_mimi` itself consumes only the hyperparameter dicts + `device` + `num_codebooks=8` (int). The checkpoint filename `MIMI_NAME="tokenizer-e351c8d8-checkpoint125.safetensors"` (`loaders.py:34`) is resolved by the processor, not here.
- [core_utils](CO03-Utils) `get_model_dir` (snapshot_download) provides the cache dir the processor reads the Mimi checkpoint from.

**Downstream (what consumes the built `MimiModel`):**
- [moshi_compression](MM01-Mimi-Codec) — this factory *is* the constructor of `MimiModel`; everything the codec does (encode/decode/streaming) lives there. Edge: a configured-and-(optionally)-loaded `MimiModel` (bf16/f32 weights, geometry as in the table).
- [core_processor](CO01-Processor-ChatState) holds the returned `MimiModel` as `_mimi`; `processor.mimi.encode(wav f32 (B,1,T)) → codes int (B,8,T)` builds audio-out training targets and `mimi.decode(codes int (B,8,T)) → wav f32 (B,1,1920)` is the demo/v1 streaming vocoder.
- [data_mapper](DA02-Chat-Mapper) `LFM2AudioChatMapper._encode_audio_out` calls `processor.mimi.encode(wav)` → 8-codebook `audio_out` target codes (the canonical audio vocabulary).
- [demo_chat](DM01-Realtime-Chat) decodes generated frames `mimi.decode(frame int (1,8,1)) → wav f32 (1,1,1920)` inside `mimi.streaming(1)`.
- [moshi_vq](QZ01-Split-RVQ) / [moshi_seanet](MO01-SEANet) / [moshi_transformer](MO03-Codec-Transformer) are the sub-modules this factory instantiates with the kwargs dicts.

## Python ↔ Rust
Symbol mapping (py → rust):

| Concern | Python (`loaders.py`) | Rust (`moshi` crate) |
|---|---|---|
| Build + config | `get_mimi` + `_seanet/_quantizer/_transformer_kwargs` (`:296,38-80`) | `mimi::Config::v0_1(num_codebooks)` (sets the same SEANet/transformer/quantizer fields) |
| Load weights | `load_file` + `load_state_dict(strict)` (or via processor) | `mimi::load(path, Some(codebooks), dev)` → `Mimi::new(cfg, vb)` |
| `num_codebooks` | `set_num_codebooks(8)` (`:332`) | `Some(8)` → `cfg.quantizer_n_q` (Rust default-if-`None` is **16**, not 8 — see gotchas) |
| Rates | `SAMPLE_RATE=24000`, `FRAME_RATE=12.5` (`:28-29`) | `sample_rate: 24_000.`, `frame_rate: 12.5` in `v0_1` |
| Framerate bridge | `encoder_frame_rate=24000/960`, `resample_method="conv"` (`:318-320`) | `encoder_frame_rate = sample_rate/Πratios`; `downsample_stride = (enc_fr/frame_rate) as usize = 2`; `ResampleMethod::Conv` |
| Checkpoint name | `MIMI_NAME` (`:34`) | same string in `loader.rs` |
| Off-path Moshi LM | `get_moshi_lm` (`:336-416`) | not ported (Moshi LM is reference-only) |

**Deliberate divergences** (see `PYTHON_VS_RUST.md §2.3 "Upstream reuse" / §2.1 device / §2.2 kernels):**
1. **Crate reuse, not re-port.** Rust reuses Kyutai's published **`moshi` crate** (`moshi = "0.6"`, mimi 0.6.4) rather than re-porting `liquid_audio/moshi`. Chosen because this checkpoint uses `quantizer.rvq_first.*` / `rvq_rest.*` naming, which **candle-transformers' Mimi cannot load** (it expects the Encodec-style `encoder.layers.N` weight-norm layout). `PYTHON_VS_RUST.md §2.3`.
2. **`norm` field.** Python `_seanet_kwargs["norm"]="none"` (weight-norm pre-folded at export); the moshi crate's `Config::v0_1` declares `norm: conv::Norm::WeightNorm` (`mimi.rs:48`) but its `load` consumes the **same folded checkpoint** — i.e. both end at plain-conv inference weights; the enum label differs, the math is identical. Not a bug.
3. **Device-agnostic.** Python is GPU-coupled (default `device="cuda"`); the Rust `mimi::load(..., dev)` honors `&Device` (CPU/Metal/CUDA). `PYTHON_VS_RUST.md §2.1`.
4. **Attention/kernels.** Python codec transformers use `F.scaled_dot_product_attention` + `torch.compile` (CUDA-gated) and `CUDAGraphed` capture; Rust runs eager candle matmul+mask+softmax, no CUDA-graph layer. Numerically irrelevant, latency-only. `PYTHON_VS_RUST.md §2.2`, `ARCH_1_MIMI_CODEC.md §4`.

## Precision / gotchas
- **`get_mimi(None)` is the real call site.** LFM2-Audio never uses the file-loading branch of `get_mimi`; it builds uninitialized then `load_state_dict(strict=True)` in `processor.py:113-115`. A spec that loads weights *inside* `get_mimi` would diverge from the actual flow.
- **Rust `num_codebooks` default differs.** `Config::v0_1(None)` defaults `quantizer_n_q` to **16** (`mimi.rs:86`); Python `get_mimi`'s default is **8** (`loaders.py:297`). The processor path always passes `Some(8)` explicitly, so the loaded model matches — but never rely on the bare default.
- **Cross-library f32 floor.** Mimi decode is only smoke-tested for parity (waveform `(1,1,30720)`, peak 0.7395); like the rest of the port it sits at the ~1e-6 f32 cross-library floor (candle gemm/FFT order ≠ torch), not bit-exact. `PYTHON_VS_RUST.md §1.4`.
- **bf16 weights, f32/bf16 compute.** Checkpoint is bf16; CPU upcasts to f32 (lossless from bf16), Metal stays bf16. The RMSNorm bf16-order subtlety from the LFM2 backbone does **not** apply here — Mimi's transformers use **LayerNorm**, not RMSNorm.
- **EOAudio is not a Mimi token.** Mimi codes are strictly `0..2047`. `2048`=EOAudio is appended by the LFM2-Audio audio head; feeding `2048` into `mimi.decode` is out-of-range. `processor.decode` rejects codes outside `[0,2047]` before the codec (`ARCH_1_MIMI_CODEC.md §1`).
- **Mimi is audio-OUT only.** Mic input never touches Mimi — it goes through the conformer mel front-end. Mimi-encode exists solely to build `audio_out` *targets* (training) and to decode generated frames (v1/demo streaming vocoder). The preferred LFM2.5 audio-out is the LFM2 ISTFT detokenizer, not Mimi.
- **Streaming multiple-of-1920 contract.** The built `MimiModel`'s streaming mode requires input length to be an exact multiple of `frame_size=1920`; the geometry that enforces this (`frame_size = SAMPLE_RATE/FRAME_RATE`) is fixed here by the two rate constants. Caller buffers.
