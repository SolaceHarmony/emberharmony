<!-- topic: Overview -->
# LFM2.5-Audio — Architecture Map (Python reference → Rust port)

This is the architecture of the **vendored Python** `liquid_audio` package (the reference
implementation), mirrored as a documentation tree: every folder has a `README.md`, every
source file has a per-component spec, and every box below **hyperlinks** to its spec. The
Rust port (`liquid-audio-rs`) is mapped inside each spec. Hand-built from reading the source;
the wiring/dtype metadata was extracted from a per-file read pass.

> Companion docs: [ARCHAEOLOGY.md](Runtime-Questions) (the 4 runtime questions),
> [ARCH_1_MIMI_CODEC.md](Codec-Deep-Dive) (the codec deep-dive),
> [PYTHON_VS_RUST.md](https://github.com/SolaceHarmony/emberharmony/blob/explore/lfm2-audio-voice/experiments/lfm2-audio-voice/liquid-audio-rs/PYTHON_VS_RUST.md) (deliberate port divergences).

> **Codes:** every node is tagged with its architecture code (e.g. `MD04`); the full code↔Python↔doc-section map is in [CROSSREF.md](Cross-Reference).

## How to read it
- **Solid green node** = on the LFM2-Audio **inference tensor path**. **Dashed grey node** =
  off-path: training, the Moshi-7B LM (a *different* model), the websocket transport
  (unported), conditioners, utils.
- **Edge label = the dtype + shape that flows on that wire** (truncated; full type in each
  spec's *Dtypes & shapes* section).
- **Click any node** to open its detailed spec.
- Two views: **(A)** the inference-path *spine* (macro forward flow, readable); **(B)** the
  *full system graph* (all 50 components, folder-clustered, every wire).

## The global dtype through-line
Weights are **bf16 on disk** (`config.lfm.torch_dtype`). Compute dtype = **bf16** on CUDA/Metal,
**f32** on CPU (candle has no CPU bf16 matmul). The **mel front-end is f32/f64 regardless**
(precision-sensitive). The path:

```
mic wav f32 -> mel (computed f32, stored bf16) -> conformer (model dtype)
 -> audio_adapter -> 2048-d embeds (model dtype) -> scatter -> LFM2 backbone (model dtype,
 with f32-upcast islands for RMSNorm + attention softmax) -> hidden (1,L,2048)
 |- text head  -> logits (65536) -> sampled text id (int64)
 |- depthformer -> 8 codebooks coarse->fine -> audio frame int (8,) codes 0..2048 (2048=EOAudio)
                                            -> Mimi / LFM2-detok -> wav f32 @24 kHz
```
Token ids are **int64**; audio codes are **int** (u32 in Rust); `EOS=7`, `EOAudio=2048`.

---

## (A) Inference-path spine

```mermaid
flowchart TB
  subgraph core["core (package root)"]
    core_processor["CO01 · LFM2AudioProcessor + ChatState (processor.py)"]:::onp
    core_detokenizer["CO02 · LFM2AudioDetokenizer (detokenizer.py)"]:::onp
  end
  subgraph model["model/"]
    model_mlp["MD03 · MLP audio_adapter (model/mlp.py)"]:::onp
    model_lfm2_audio["MD01 · LFM2AudioModel (model/lfm2_audio.py)"]:::onp
    model_lfm2_backbone["MD02 · Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model)"]:::onp
    model_transformer["MD04 · RawLMBackbone depthformer (model/transformer.py)"]:::onp
  end
  subgraph conformer["model/conformer/"]
    conformer_processor["CF04 · FilterbankFeatures mel front-end (model/conformer/processor.py)"]:::onp
    conformer_subsampling["CF05 · ConvSubsampling (model/conformer/subsampling.py)"]:::onp
    conformer_encoder["CF01 · ConformerEncoder (conformer/encoder.py)"]:::onp
  end
  subgraph moshi_models["moshi/models/"]
    moshi_compression["MM01 · MimiModel codec (moshi/models/compression.py)"]:::onp
  end
  subgraph moshi_modules["moshi/modules/"]
    moshi_seanet["MO01 · SEANetEncoder/Decoder (moshi/modules/seanet.py)"]:::onp
  end
  subgraph moshi_quant["moshi/quantization/"]
    moshi_vq["QZ01 · SplitResidualVectorQuantizer (moshi/quantization/vq.py)"]:::onp
  end
  core_processor -->|"ChatState bundle: text int64 (1,L) + a…"| model_lfm2_audio
  core_processor -->|"int codes (u32) (1,8,T) in (0,2047)"| core_detokenizer
  core_processor -->|"int codes (u32) (1,8,T) for decode"| moshi_compression
  core_processor -->|"f32 (1,L') @16kHz wav"| conformer_processor
  core_detokenizer -->|"f32"| core_processor
  model_lfm2_audio -->|"GenToken stream: text id int64 (1,) + …"| core_processor
  model_lfm2_audio -->|"audio frame u32 codes (routed via proc…"| core_detokenizer
  model_lfm2_audio -->|"audio frame u32 codes → Mimi decode"| moshi_compression
  model_lfm2_backbone -->|"model-dtype last_hidden_state"| model_lfm2_audio
  model_lfm2_backbone -->|"model-dtype hidden (same backbone arch…"| core_detokenizer
  model_mlp -->|"model dtype (bf16/f32)"| model_lfm2_audio
  model_transformer -->|"int audio frame (8,), codes 0..2048 (2…"| model_lfm2_audio
  model_transformer -->|"int/u32 audio codes 0..2047 per frame …"| core_processor
  model_transformer -->|"int/u32 codes → f32 waveform @24kHz (L…"| core_detokenizer
  model_transformer -->|"int/u32 codes → f32 waveform @24kHz (M…"| moshi_compression
  conformer_encoder -->|"model dtype (bf16/f32)"| model_lfm2_audio
  conformer_encoder -->|"model dtype (bf16/f32)"| model_mlp
  conformer_processor -->|"bf16 (model dtype)"| conformer_encoder
  conformer_processor -->|"f32→bf16"| core_processor
  conformer_subsampling -->|"model dtype (bf16/f32)"| conformer_encoder
  moshi_compression -->|"f32 waveform (1,1,T·1920)@24kHz (fallb…"| core_processor
  moshi_seanet -->|"decoder out: waveform (B,1,t*960) f32 …"| moshi_compression
  moshi_seanet -->|"encoder latent (B,512,*) model-dtype (…"| moshi_vq
  moshi_vq -->|"encode: codes (B,8,T) int64 (0..2047)"| moshi_compression
  moshi_vq -->|"reconstructed waveform downstream of d…"| core_processor
  click core_processor "CO01-Processor-ChatState"
  click conformer_processor "CF04-Mel-Frontend"
  click conformer_subsampling "CF05-Subsampling"
  click conformer_encoder "CF01-Conformer-Encoder"
  click model_mlp "MD03-Audio-Adapter-MLP"
  click model_lfm2_audio "MD01-LFM2AudioModel"
  click model_lfm2_backbone "MD02-LFM2-Backbone"
  click model_transformer "MD04-Depthformer"
  click core_detokenizer "CO02-Detokenizer"
  click moshi_compression "MM01-Mimi-Codec"
  click moshi_vq "QZ01-Split-RVQ"
  click moshi_seanet "MO01-SEANet"
  classDef onp fill:#0b3d2e,stroke:#19a974,color:#e8fff4,stroke-width:1px;
  classDef offp fill:#2a2a33,stroke:#6b7280,color:#cbd5e1,stroke-dasharray:4 3;```

> Conformer and codec internals are collapsed here — open
> [model/conformer/](CF00-Overview) and [moshi/](Mimi-Codec-Overview) for their
> internal graphs, or see the full graph below.

---


> Continue → **[Full system graph](System-Graph)** · **[Component index](Component-Index)**
