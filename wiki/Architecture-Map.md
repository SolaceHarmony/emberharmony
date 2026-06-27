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

## (B) Full system graph (all 50 components)

Dashed clusters are off the LFM2-Audio inference path. Every node hyperlinks to its spec.

```mermaid
flowchart LR
  subgraph core["core (package root)"]
    core_processor["CO01 · LFM2AudioProcessor + ChatState (processor.py)"]:::onp
    core_detokenizer["CO02 · LFM2AudioDetokenizer (detokenizer.py)"]:::onp
    core_utils["CO03 · core_utils (utils.py)"]:::onp
  end
  subgraph model["model/"]
    model_lfm2_audio["MD01 · LFM2AudioModel (model/lfm2_audio.py)"]:::onp
    model_lfm2_backbone["MD02 · Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model)"]:::onp
    model_mlp["MD03 · MLP audio_adapter (model/mlp.py)"]:::onp
    model_transformer["MD04 · RawLMBackbone depthformer (model/transformer.py)"]:::onp
  end
  subgraph conformer["model/conformer/"]
    conformer_encoder["CF01 · ConformerEncoder (conformer/encoder.py)"]:::onp
    conformer_mha["CF02 · RelPositionMultiHeadAttention + RelPositionalEncoding (model/conformer/mha.py)"]:::onp
    conformer_modules["CF03 · ConformerLayer/Convolution/FeedForward/CausalConv1D (model/conformer/modules.py)"]:::onp
    conformer_processor["CF04 · FilterbankFeatures mel front-end (model/conformer/processor.py)"]:::onp
    conformer_subsampling["CF05 · ConvSubsampling (model/conformer/subsampling.py)"]:::onp
    conformer_utils["CF06 · conformer_utils (model/conformer/utils.py)"]:::onp
  end
  subgraph moshi_models["moshi/models/"]
    moshi_compression["MM01 · MimiModel codec (moshi/models/compression.py)"]:::onp
    moshi_loaders["MM02 · get_mimi factory + CheckpointInfo (moshi/models/loaders.py)"]:::onp
    moshi_lm["MM03 · LMModel + LMGen (moshi/models/lm.py)"]:::offp
    moshi_lm_utils["MM04 · ScaledEmbedding + delay/init helpers (moshi/models/lm_utils.py)"]:::offp
    moshi_tts["MM05 · TTSModel (moshi/models/tts.py)"]:::offp
  end
  subgraph moshi_modules["moshi/modules/"]
    moshi_seanet["MO01 · SEANetEncoder/Decoder (moshi/modules/seanet.py)"]:::onp
    moshi_conv["MO02 · StreamingConv1d/ConvTranspose1d (moshi/modules/conv.py)"]:::onp
    moshi_transformer["MO03 · ProjectedTransformer/StreamingTransformer (moshi/modules/transformer.py)"]:::onp
    moshi_resample["MO04 · ConvDownsample1d / ConvTrUpsample1d (moshi/modules/resample.py)"]:::onp
    moshi_rope["MO05 · RotaryEmbedding (moshi/modules/rope.py)"]:::onp
    moshi_streaming["MO06 · StreamingModule(State) (moshi/modules/streaming.py)"]:::onp
    moshi_gating["MO07 · ActivationGating / make_gating (moshi/modules/gating.py)"]:::onp
    moshi_lora["MO08 · LoRALinear (moshi/modules/lora.py)"]:::offp
  end
  subgraph moshi_quant["moshi/quantization/"]
    moshi_vq["QZ01 · SplitResidualVectorQuantizer (moshi/quantization/vq.py)"]:::onp
    moshi_core_vq["QZ02 · EuclideanCodebook / ResidualVectorQuantization (moshi/quantization/core_vq.py)"]:::onp
    moshi_quant_base["QZ03 · BaseQuantizer (moshi/quantization/base.py)"]:::onp
  end
  subgraph demo["demo/"]
    demo_chat["DM01 · demo_chat (demo/chat.py)"]:::offp
    demo_model["DM02 · demo singletons + CUDA warmup (demo/model.py)"]:::offp
  end
  subgraph transport["moshi/ (transport)"]
    moshi_server["TR01 · moshi_server (moshi/server.py)"]:::offp
    moshi_client["TR02 · moshi client (moshi/client.py)"]:::offp
    moshi_client_utils["TR03 · AnyPrinter/RawPrinter/Printer (moshi/client_utils.py)"]:::offp
    moshi_run_inference["TR04 · moshi_run_inference (moshi/run_inference.py)"]:::offp
    moshi_run_tts["TR05 · moshi_run_tts (moshi/run_tts.py)"]:::offp
    moshi_client_gradio["TR06 · MoshiHandler gradio WebRTC client (moshi/client_gradio.py)"]:::offp
  end
  subgraph data["data/"]
    data_dataloader["DA01 · LFM2DataLoader + lfm2_collator (data/dataloader.py)"]:::offp
    data_mapper["DA02 · LFM2AudioChatMapper (data/mapper.py)"]:::offp
    data_preprocess["DA03 · preprocess_dataset (data/preprocess.py)"]:::offp
    data_types["DA04 · data_types (data/types.py)"]:::offp
  end
  subgraph training["training"]
    core_trainer["CO04 · Trainer (trainer.py)"]:::offp
  end
  subgraph moshi_cond["moshi/conditioners/"]
    moshi_cond_base["CN01 · ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py)"]:::offp
    moshi_cond_text["CN02 · LUTConditioner (moshi/conditioners/text.py)"]:::offp
    moshi_cond_tensors["CN03 · TensorConditioner (moshi/conditioners/tensors.py)"]:::offp
  end
  subgraph moshi_utils["moshi/utils/"]
    moshi_util_sampling["MU01 · sample_token top-k/top-p multinomial (moshi/utils/sampling.py)"]:::offp
    moshi_util_compile["MU02 · CUDAGraphed + torch_compile gating (moshi/utils/compile.py)"]:::onp
    moshi_util_autocast["MU03 · TorchAutocast (moshi/utils/autocast.py)"]:::offp
    moshi_util_quantize["MU04 · QLinear int8 weight-quantize helper (moshi/utils/quantize.py)"]:::offp
    moshi_util_utils["MU05 · cross_entropy (moshi/utils/utils.py)"]:::offp
  end
  core_processor -->|"ChatState bundle: text int64 (1,L) + a…"| model_lfm2_audio
  core_processor -->|"int codes (u32) (1,8,T) in (0,2047)"| core_detokenizer
  core_processor -->|"int codes (u32) (1,8,T) for decode"| moshi_compression
  core_processor -->|"f32 (1,L') @16kHz wav"| conformer_processor
  core_detokenizer -->|"f32"| core_processor
  core_utils -->|"LFMModality int → modality_flag int64 …"| core_processor
  core_utils -->|"module_exists bool (attn select)"| model_lfm2_audio
  core_utils -->|"LFMModality int + mel2emb_len int → mo…"| data_mapper
  core_utils -->|"LFMModality.TEXT int → int64 pad value"| data_dataloader
  core_utils -->|"LFMModality.TEXT/AUDIO_OUT enum int pe…"| demo_chat
  core_trainer -->|"bf16 param grads in / f32 scalar out.l…"| model_lfm2_audio
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
  conformer_mha -->|"model dtype (bf16/f32)"| conformer_modules
  conformer_mha -->|"model dtype (bf16/f32)"| conformer_encoder
  conformer_modules -->|"model dtype (bf16/f32)"| conformer_encoder
  conformer_modules -->|"model dtype (bf16/f32)"| model_mlp
  conformer_processor -->|"bf16 (model dtype)"| conformer_encoder
  conformer_processor -->|"f32→bf16"| core_processor
  conformer_subsampling -->|"model dtype (bf16/f32)"| conformer_encoder
  conformer_utils -->|"context manager guarding attention com…"| conformer_mha
  conformer_utils -->|"list(float) layer_drop_probs (all 0.0 …"| conformer_encoder
  data_dataloader -->|"LFM2AudioModelInput {text int64 (1,B·c…"| core_trainer
  data_dataloader -->|"LFM2AudioModelInput (via trainer): fla…"| model_lfm2_audio
  data_mapper -->|"LFM2AudioTrainingSample (text/audio_ou…"| data_preprocess
  data_mapper -->|"Arrow rows re-read into LFM2AudioRow t…"| data_dataloader
  data_preprocess -->|"HF Arrow row → text int64 (1,L), audio…"| data_dataloader
  data_types -->|"LFM2AudioModelInput: text int64 (B,L),…"| model_lfm2_audio
  data_types -->|"LFM2AudioModelInput bundle (moved via …"| core_trainer
  data_types -->|"LFM2AudioRow → LFM2AudioModelInput (co…"| data_dataloader
  data_types -->|"LFM2AudioTrainingSample six fields → A…"| data_preprocess
  moshi_compression -->|"f32 waveform (1,1,1920)@24kHz per-fram…"| demo_chat
  moshi_compression -->|"f32 waveform (1,1,T·1920)@24kHz (fallb…"| core_processor
  moshi_compression -->|"int codes (B,8,T) values (0,2047) = au…"| data_mapper
  moshi_loaders -->|"constructed MimiModel"| moshi_compression
  moshi_loaders -->|"MimiModel held as _mimi"| core_processor
  moshi_loaders -->|"processor.mimi.encode(wav f32 (B,1,T))…"| data_mapper
  moshi_loaders -->|"mimi.decode(frame int (1,8,1)) → wav f…"| demo_chat
  moshi_loaders -->|"instantiated SplitResidualVectorQuanti…"| moshi_vq
  moshi_loaders -->|"instantiated SEANetEncoder/Decoder(dim…"| moshi_seanet
  moshi_loaders -->|"instantiated ProjectedTransformer x2 (…"| moshi_transformer
  moshi_lm -->|"audio frame int (B,8,1) (codes 0..card…"| moshi_compression
  moshi_lm -->|"LMModel+LMGen wrapped for script-drive…"| moshi_tts
  moshi_lm -->|"LMGen.step audio frame int (B,8,1) per…"| moshi_server
  moshi_lm -->|"LMGen.step audio frame int (B,8,1) (im…"| moshi_run_inference
  moshi_lm_utils -->|"model-dtype (bf16/f32) embeddings (B,K…"| moshi_lm
  moshi_lm_utils -->|"ScaledEmbedding handle (weight rows ze…"| moshi_tts
  moshi_tts -->|"int64 frame (B,1+Q,1) (audio codes via…"| moshi_compression
  moshi_tts -->|"int64 input_tokens (B,missing,1) (zero…"| moshi_lm
  moshi_seanet -->|"decoder out: waveform (B,1,t*960) f32 …"| moshi_compression
  moshi_seanet -->|"encoder latent (B,512,T/960) model-dty…"| moshi_resample
  moshi_seanet -->|"encoder latent (B,512,*) model-dtype (…"| moshi_vq
  moshi_conv -->|"model dtype (bf16/f32) (B,C_out,T') co…"| moshi_seanet
  moshi_conv -->|"encode: codes int/u32 (B,8,T/1920)"| moshi_compression
  moshi_conv -->|"model dtype (bf16/f32) latent (B,512,T…"| moshi_resample
  moshi_conv -->|"decode of audio frame (8,) int codes 0…"| core_processor
  moshi_transformer -->|"model-dtype (B,512,T') (encoder_transf…"| moshi_vq
  moshi_transformer -->|"model-dtype (B,512,T') (decoder_transf…"| moshi_seanet
  moshi_transformer -->|"model-dtype (B,512,T') (single-element…"| moshi_compression
  moshi_resample -->|"model dtype (f32/bf16)"| moshi_vq
  moshi_resample -->|"model dtype (f32/bf16)"| moshi_transformer
  moshi_resample -->|"model dtype (f32/bf16)"| moshi_compression
  moshi_rope -->|"rotated q/k model dtype (bf16/f32) (B,…"| moshi_transformer
  moshi_rope -->|"model dtype (B,8,T,64) within MimiMode…"| moshi_compression
  moshi_streaming -->|"streaming-state API (mixin)"| moshi_compression
  moshi_streaming -->|"StreamingContainer mixin"| moshi_seanet
  moshi_streaming -->|"_StreamingConv1dState: causal ring-buf…"| moshi_conv
  moshi_streaming -->|"_MHAState/_LayerState/_TransformerStat…"| moshi_transformer
  moshi_streaming -->|"LMModel/LMGen reuse State API (off LFM…"| moshi_lm
  moshi_gating -->|"model dtype (bf16/f32)"| moshi_transformer
  moshi_gating -->|"model dtype (bf16/f32)"| moshi_compression
  moshi_gating -->|"model dtype (bf16)"| moshi_lm
  moshi_lora -->|"bf16"| moshi_transformer
  moshi_lora -->|"bf16"| moshi_lm
  moshi_lora -->|"bf16"| moshi_loaders
  moshi_vq -->|"encode: codes (B,8,T) int64 (0..2047)"| moshi_compression
  moshi_vq -->|"audio_out codes (8,L) int64 via MimiMo…"| data_mapper
  moshi_vq -->|"reconstructed waveform downstream of d…"| core_processor
  moshi_core_vq -->|"decode: latent model-dtype (B,256,T) →…"| moshi_vq
  moshi_core_vq -->|"via SplitResidualVectorQuantizer: summ…"| moshi_compression
  moshi_quant_base -->|"codes (B,8,T) int64 (Rust u32) ∈(0,204…"| moshi_compression
  moshi_cond_base -->|"sum_condition (B,1,C) + cross_attentio…"| moshi_lm
  moshi_cond_base -->|"BaseConditioner base class (subclassed…"| moshi_cond_text
  moshi_cond_base -->|"BaseConditioner base class (subclassed…"| moshi_cond_tensors
  moshi_cond_base -->|"BaseConditioner/ConditionProvider/Cond…"| moshi_loaders
  moshi_cond_text -->|"ConditionType(condition bf16/f32 (B,1,…"| moshi_cond_base
  moshi_cond_text -->|"fused condition bf16/f32 (sum offset /…"| moshi_lm
  moshi_cond_text -->|"tokenizer.possible_values list(str) + …"| moshi_tts
  moshi_cond_tensors -->|"ConditionType(condition (B,T,D) model/…"| moshi_cond_base
  moshi_cond_tensors -->|"ConditionType(cond (B,T,output_dim) mo…"| moshi_lm
  moshi_cond_tensors -->|"constructed via get_conditioner (no te…"| moshi_tts
  moshi_util_sampling -->|"int64"| moshi_lm
  moshi_util_compile -->|"passthrough replay of encoder/decoder/…"| moshi_compression
  moshi_util_compile -->|"graphed _set_exec_mask: bool exec-mask…"| moshi_streaming
  moshi_util_compile -->|"off-path: graphed forward_text/depform…"| moshi_lm
  moshi_util_quantize -->|"in-place nn.Linear→QLinear swap"| moshi_lm
  moshi_util_quantize -->|"per-layer in/out-proj nn.Linear→QLinea…"| moshi_transformer
  moshi_server -->|"encode: f32 (1,1,1920) in"| moshi_compression
  moshi_server -->|"int codes (1,n_q,1) per code column → …"| moshi_lm
  moshi_server -->|"websocket frames: x01 Opus audio bytes…"| moshi_client
  moshi_client -->|"ws binary b'x01'+opus (Opus-encoded mi…"| moshi_server
  moshi_client_utils -->|"str (print_token/print_lag/print_pendi…"| moshi_client
  moshi_client_utils -->|"str (log lines)"| moshi_server
  moshi_client_utils -->|"str (print_token text + log"| moshi_run_inference
  moshi_run_inference -->|"f32 (B,1,1920) @ 24kHz → mimi.encode"| moshi_compression
  moshi_run_inference -->|"int codes (B, n_q, 1) into LMGen.step"| moshi_lm
  moshi_run_inference -->|"int64 text token id → SentencePiece pi…"| moshi_client_utils
  moshi_run_tts -->|"int audio codes (B, nq, 1) (u32 in Rus…"| moshi_compression
  moshi_client_gradio -->|"tagged Opus bytes (b'x01'+opus) over /…"| moshi_server
  demo_chat -->|"prefill bundle: text int64 (1,L), audi…"| model_lfm2_audio
  demo_chat -->|"writeback: text (1,·) int64, audio_out…"| core_processor
  demo_chat -->|"audio codes (1,8,1) int (u32 in Rust) …"| moshi_compression
  demo_model -->|"module-level singletons via `from .mod…"| demo_chat
  click core_processor "CO01-Processor-ChatState"
  click core_detokenizer "CO02-Detokenizer"
  click core_utils "CO03-Utils"
  click core_trainer "CO04-Trainer"
  click model_lfm2_audio "MD01-LFM2AudioModel"
  click model_lfm2_backbone "MD02-LFM2-Backbone"
  click model_mlp "MD03-Audio-Adapter-MLP"
  click model_transformer "MD04-Depthformer"
  click conformer_encoder "CF01-Conformer-Encoder"
  click conformer_mha "CF02-RelPos-MHA"
  click conformer_modules "CF03-Conformer-Layer"
  click conformer_processor "CF04-Mel-Frontend"
  click conformer_subsampling "CF05-Subsampling"
  click conformer_utils "CF06-Conformer-Utils"
  click data_dataloader "DA01-DataLoader"
  click data_mapper "DA02-Chat-Mapper"
  click data_preprocess "DA03-Preprocess-Arrow"
  click data_types "DA04-Data-Types"
  click moshi_compression "MM01-Mimi-Codec"
  click moshi_loaders "MM02-Mimi-Loaders"
  click moshi_lm "MM03-Moshi-LM"
  click moshi_lm_utils "MM04-Moshi-LM-Utils"
  click moshi_tts "MM05-Moshi-TTS"
  click moshi_seanet "MO01-SEANet"
  click moshi_conv "MO02-Streaming-Conv"
  click moshi_transformer "MO03-Codec-Transformer"
  click moshi_resample "MO04-Framerate-Resample"
  click moshi_rope "MO05-RoPE"
  click moshi_streaming "MO06-Streaming-Module"
  click moshi_gating "MO07-Gating"
  click moshi_lora "MO08-LoRA"
  click moshi_vq "QZ01-Split-RVQ"
  click moshi_core_vq "QZ02-VQ-Core"
  click moshi_quant_base "QZ03-Quantizer-Base"
  click moshi_cond_base "CN01-Conditioner-Base"
  click moshi_cond_text "CN02-Text-Conditioner"
  click moshi_cond_tensors "CN03-Tensor-Conditioner"
  click moshi_util_sampling "MU01-Sampling"
  click moshi_util_compile "MU02-CUDA-Graphs"
  click moshi_util_autocast "MU03-Autocast"
  click moshi_util_quantize "MU04-Int8-Quantize"
  click moshi_util_utils "MU05-Moshi-Utils"
  click moshi_server "TR01-WS-Server"
  click moshi_client "TR02-WS-Client"
  click moshi_client_utils "TR03-Client-Utils"
  click moshi_run_inference "TR04-Run-Inference"
  click moshi_run_tts "TR05-Run-TTS"
  click moshi_client_gradio "TR06-Gradio-Client"
  click demo_chat "DM01-Realtime-Chat"
  click demo_model "DM02-Demo-Singletons"
  classDef onp fill:#0b3d2e,stroke:#19a974,color:#e8fff4,stroke-width:1px;
  classDef offp fill:#2a2a33,stroke:#6b7280,color:#cbd5e1,stroke-dasharray:4 3;```

---

## Component index

### core (package root)

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| LFM2AudioProcessor + ChatState (processor.py) | wav f32 (1,L); text str; generated text int64  → ChatState bundle: text int64 (1,L), audio_in b | ✅ | [`processor.md`](CO01-Processor-ChatState) |
| LFM2AudioDetokenizer (detokenizer.py) | int64/int audio codes (1,8,T) values [0,2047] → f32 waveform (1,L) @ 24kHz | ✅ | [`detokenizer.md`](CO02-Detokenizer) |
| core_utils (utils.py) | int/int64 mel widths; str repo id or Path; str → enum int 1/2/3 (materialized int64 (1,L) by ca | ✅ | [`utils.md`](CO03-Utils) |

### model/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| LFM2AudioModel (model/lfm2_audio.py) | ChatState: text int64 (1,L_t); mel bf16 (128,Σ → stream of text id int64 (1,) and audio frame i | ✅ | [`model/lfm2_audio.md`](MD01-LFM2AudioModel) |
| Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model) | inputs_embeds model-dtype (bf16 cuda/Metal, f3 → last_hidden_state model-dtype (1,L,2048) post  | ✅ | [`model/lfm2_backbone.md`](MD02-LFM2-Backbone) |
| MLP audio_adapter (model/mlp.py) | model dtype (bf16 GPU / f32 CPU), (ΣT',512) → model dtype (bf16/f32), (ΣT',2048) | ✅ | [`model/mlp.md`](MD03-Audio-Adapter-MLP) |
| RawLMBackbone depthformer (model/transformer.py) | model dtype (bf16/f32) hidden [1,1,1024] per d → audio frame (8,) int codes 0..2048 (2048=EOAud | ✅ | [`model/transformer.md`](MD04-Depthformer) |

### model/conformer/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| ConformerEncoder (conformer/encoder.py) | model dtype (bf16/f32) mel features (B,128,T)  → model dtype (bf16/f32) audio_enc (B,512,T') +  | ✅ | [`model/conformer/encoder.md`](CF01-Conformer-Encoder) |
| RelPositionMultiHeadAttention + RelPositionalEncoding (model/conformer/mha.py) | model dtype (bf16 cuda / f32 Rust-CPU / bf16 M → model dtype, attention output (B,T',512); soft | ✅ | [`model/conformer/mha.md`](CF02-RelPos-MHA) |
| ConformerLayer/Convolution/FeedForward/CausalConv1D (model/conformer/modules.py) | model dtype (bf16 cuda / f32 CPU / bf16 Metal) → model dtype; (B,T',512) | ✅ | [`model/conformer/modules.md`](CF03-Conformer-Layer) |
| FilterbankFeatures mel front-end (model/conformer/processor.py) | f32 (1,L) mic PCM @16kHz → f32 mel (1,128,T) [T=1+L/160, padded to mult o | ✅ | [`model/conformer/processor.md`](CF04-Mel-Frontend) |
| ConvSubsampling (model/conformer/subsampling.py) | model dtype (bf16/f32) (B,T,128) mel features; → model dtype (bf16/f32) (B,T',512); int64 (B,)  | ✅ | [`model/conformer/subsampling.md`](CF05-Subsampling) |
| conformer_utils (model/conformer/utils.py) | config scalars + autocast-dtype enum (no activ → context manager (autocast bf16/f32 or nullcont | ✅ | [`model/conformer/utils.md`](CF06-Conformer-Utils) |

### moshi/models/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| MimiModel codec (moshi/models/compression.py) | decode: int codes (B,8,T) values [0,2047] (Rus → decode: f32 waveform (B,1,T·1920)@24kHz · enco | ✅ | [`moshi/models/compression.md`](MM01-Mimi-Codec) |
| get_mimi factory + CheckpointInfo (moshi/models/loaders.py) | filename:Path/None, device, num_codebooks=8 (i → configured MimiModel (bf16/f32 weights); geome | ✅ | [`moshi/models/loaders.md`](MM02-Mimi-Loaders) |
| LMModel + LMGen (moshi/models/lm.py) | codes int64 [B, n_q+1, T] (train forward); LMG → LMOutput.logits model-dtype [B, dep_q, T, card | — | [`moshi/models/lm.md`](MM03-Moshi-LM) |
| ScaledEmbedding + delay/init helpers (moshi/models/lm_utils.py) | int64 token ids (...,) for ScaledEmbedding; in → model-dtype (bf16/f32) embeddings (...,D); (un | — | [`moshi/models/lm_utils.md`](MM04-Moshi-LM-Utils) |
| TTSModel (moshi/models/tts.py) | script list[str] + voice safetensors speaker_w → TTSResult.frames: list of int64 [B,1+Q,1] (aco | — | [`moshi/models/tts.md`](MM05-Moshi-TTS) |

### moshi/modules/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| SEANetEncoder/Decoder (moshi/modules/seanet.py) | decoder: latent (B,512,t) model-dtype(bf16/f32 → decoder: waveform (B,1,t*960) f32 @24kHz; enco | ✅ | [`moshi/modules/seanet.md`](MO01-SEANet) |
| StreamingConv1d/ConvTranspose1d (moshi/modules/conv.py) | model dtype (bf16 cuda/Metal, f32 Rust CPU) (B → model dtype (B,C_out,T'); decoder transpose-co | ✅ | [`moshi/modules/conv.md`](MO02-Streaming-Conv) |
| ProjectedTransformer/StreamingTransformer (moshi/modules/transformer.py) | model-dtype (bf16 cuda / f32 cpu) SEANet laten → model-dtype refined latent [B,512,T'] (conv_la | ✅ | [`moshi/modules/transformer.md`](MO03-Codec-Transformer) |
| ConvDownsample1d / ConvTrUpsample1d (moshi/modules/resample.py) | model dtype (CPU f32 / Metal bf16 / Python cud → model dtype, (B,512,T/2@12.5Hz) downsample-out | ✅ | [`moshi/modules/resample.md`](MO04-Framerate-Resample) |
| RotaryEmbedding (moshi/modules/rope.py) | q/k: model dtype (bf16 cuda/Metal, f32 CPU) [B → qo/ko: model dtype (bf16/f32), rotated, [B,8,T | ✅ | [`moshi/modules/rope.md`](MO05-RoPE) |
| StreamingModule[State] (moshi/modules/streaming.py) | no tensor input; allocates exec_mask bool (B,) → no tensor output; state tree {name: State} wit | ✅ | [`moshi/modules/streaming.md`](MO06-Streaming-Module) |
| ActivationGating / make_gating (moshi/modules/gating.py) | model dtype (bf16 Metal/cuda, f32 Rust-CPU), ( → model dtype, (B,T,dim) | ✅ | [`moshi/modules/gating.md`](MO07-Gating) |
| LoRALinear (moshi/modules/lora.py) | bf16 (B,T,in_features); LoRA state_dict bf16 l → bf16 (B,T,out_features); merged weight bf16 (o | — | [`moshi/modules/lora.md`](MO08-LoRA) |

### moshi/quantization/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| SplitResidualVectorQuantizer (moshi/quantization/vq.py) | latent (B,512,T) model dtype (bf16 cuda/metal, → codes (B,8,T) int64 values 0..2047 (encode); l | ✅ | [`moshi/quantization/vq.md`](QZ01-Split-RVQ) |
| EuclideanCodebook / ResidualVectorQuantization (moshi/quantization/core_vq.py) | decode: codes int (u32 in Rust) [n_q,B,T]; enc → decode: latent model-dtype (bf16 CUDA/Metal, f | ✅ | [`moshi/quantization/core_vq.md`](QZ02-VQ-Core) |
| BaseQuantizer (moshi/quantization/base.py) | latent [B,512→256,T] model dtype (bf16 cuda /  → QuantizedResult{ x: latent [B,512,T] model dty | ✅ | [`moshi/quantization/base.md`](QZ03-Quantizer-Base) |

### demo/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| demo_chat (demo/chat.py) | mic audio (rate:int, np.ndarray int16-range) - → streamed: text token (1,) int64; audio code fr | — | [`demo/chat.md`](DM01-Realtime-Chat) |
| demo singletons + CUDA warmup (demo/model.py) | HF_DIR str + snapshot weights (bf16 backbone m → three eval() singletons: proc (LFM2AudioProces | — | [`demo/model.md`](DM02-Demo-Singletons) |

### moshi/ (transport)

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| moshi_server (moshi/server.py) | Opus bytes over websocket -> sphn-decoded f32  → Opus audio bytes (ws x01) from f32 (1,1,1920) | — | [`moshi/server.md`](TR01-WS-Server) |
| moshi client (moshi/client.py) | f32 PCM (1920,1) mono @24kHz from PortAudio mi → ws binary b"x01"+opus (mic uplink) to server; | — | [`moshi/client.md`](TR02-WS-Client) |
| AnyPrinter/RawPrinter/Printer (moshi/client_utils.py) | str (decoded text token per LM step); level:st → str bytes to stdout (ANSI SGR-wrapped in Print | — | [`moshi/client_utils.md`](TR03-Client-Utils) |
| moshi_run_inference (moshi/run_inference.py) | f32 PCM (B,1,N) @ 24kHz (from sphn.read, broad → list[(text_tokens int64 (Ttok,), audio_tokens  | — | [`moshi/run_inference.md`](TR04-Run-Inference) |
| moshi_run_tts (moshi/run_tts.py) | JSONL str -> TTSRequest{turns:list[str], voice → f32 waveform (B,1,total_samples) @ 24kHz -> cl | — | [`moshi/run_tts.md`](TR05-Run-TTS) |
| MoshiHandler gradio WebRTC client (moshi/client_gradio.py) | f32 PCM (N,) @24kHz from mic (int16-scaled, /3 → tagged Opus bytes b"x01"+opus to server; f32  | — | [`moshi/client_gradio.md`](TR06-Gradio-Client) |

### data/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| LFM2DataLoader + lfm2_collator (data/dataloader.py) | Arrow columns per row: text int(→int64) (1,n), → LFM2AudioModelInput: text int64 (1,B·ctx), aud | — | [`data/dataloader.md`](DA01-DataLoader) |
| LFM2AudioChatMapper (data/mapper.py) | list[ChatMessage] (str text + bytes audio) → LFM2AudioTrainingSample: text i64 (1,L_text);  | — | [`data/mapper.md`](DA02-Chat-Mapper) |
| preprocess_dataset (data/preprocess.py) | LFM2AudioTrainingSample: text int64 (1,L), aud → HF Arrow dataset on disk: List&lt;List&lt;Int6 | — | [`data/preprocess.md`](DA03-Preprocess-Arrow) |
| data_types (data/types.py) | list[ChatMessage] (role + Text/Audio/Interleav → LFM2AudioModelInput bundle: text int64 (B,L);  | — | [`data/types.md`](DA04-Data-Types) |

### training

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| Trainer (trainer.py) | LFM2AudioModelInput batch: text int64 (B,L), a → f32 scalar loss (per step) + bf16 safetensors  | — | [`trainer.md`](CO04-Trainer) |

### moshi/conditioners/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py) | ConditionAttributes (text str/None + TensorCon → ConditionType(condition model-dtype [B,T,outpu | — | [`moshi/conditioners/base.md`](CN01-Conditioner-Base) |
| LUTConditioner (moshi/conditioners/text.py) | list[str/None] (len B); after tokenize tokens  → ConditionType(condition model-dtype bf16/f32 [ | — | [`moshi/conditioners/text.md`](CN02-Text-Conditioner) |
| TensorConditioner (moshi/conditioners/tensors.py) | TensorCondition(tensor model-dtype/f32 [B/1,T, → ConditionType(condition model-dtype/f32 [B,T,D | — | [`moshi/conditioners/tensors.md`](CN03-Tensor-Conditioner) |

### moshi/utils/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| sample_token top-k/top-p multinomial (moshi/utils/sampling.py) | f32 logits (…, Card) — Moshi callers upcast vi → int64 token id (…,) squeezed | — | [`moshi/utils/sampling.md`](MU01-Sampling) |
| CUDAGraphed + torch_compile gating (moshi/utils/compile.py) | passthrough: model-dtype (B,512,·) latent @25H → passthrough: wrapped module output unchanged — | ✅ | [`moshi/utils/compile.md`](MU02-CUDA-Graphs) |
| TorchAutocast (moshi/utils/autocast.py) | no tensor input — ctor args: enabled:bool + to → no tensor output — __enter__ returns None; amb | — | [`moshi/utils/autocast.md`](MU03-Autocast) |
| QLinear int8 weight-quantize helper (moshi/utils/quantize.py) | nn.Linear.weight bf16/f32 [out,in] (module tre → weight int8 [out,in] + weight_scb fp32 [out];  | — | [`moshi/utils/quantize.md`](MU04-Int8-Quantize) |
| cross_entropy (moshi/utils/utils.py) | logits bf16 [B,K,T,card]; targets int64 [B,K,T → ce f32 [B,K,T] (per-codebook, masked positions | — | [`moshi/utils/utils.md`](MU05-Moshi-Utils) |

---

## Folder READMEs
- [model/](MD00-Overview) — the LFM2-Audio model graph
- [model/conformer/](CF00-Overview) — the FastConformer audio encoder
- [data/](DA00-Overview) — the training data pipeline
- [moshi/](Mimi-Codec-Overview) — the Kyutai Moshi stack (codec on-path; LM + transport off-path)
- [moshi/models/](MM00-Overview) · [moshi/modules/](MO00-Overview) ·
  [moshi/quantization/](QZ00-Overview) ·
  [moshi/conditioners/](CN00-Overview) · [moshi/utils/](MU00-Overview)
- [demo/](DM00-Overview) — realtime runtime + turn-taking

*Off-path note:* the **Moshi-7B LM** (`moshi/models/lm.py`), its **TTS**, the **websocket
transport** (`moshi/server.py`/`client.py`), **conditioners**, and **training** are documented
for completeness but are **not** part of the LFM2-Audio inference graph — LFM2-Audio uses its
own backbone + depthformer head and (for audio-out) the Mimi codec / LFM2 detokenizer only.
