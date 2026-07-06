<!-- topic: Overview -->
# Component index

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
| moshi_server (moshi/server.py) | Opus bytes over websocket -> sphn-decoded f32  → Opus audio bytes (ws x01) from f32 (1,1,1920) | — | [`moshi/server.md`](Moshi-Transport) |
| moshi client (moshi/client.py) | f32 PCM (1920,1) mono @24kHz from PortAudio mi → ws binary b"x01"+opus (mic uplink) to server; | — | [`moshi/client.md`](Moshi-Transport) |
| AnyPrinter/RawPrinter/Printer (moshi/client_utils.py) | str (decoded text token per LM step); level:st → str bytes to stdout (ANSI SGR-wrapped in Print | — | [`moshi/client_utils.md`](Moshi-Transport) |
| moshi_run_inference (moshi/run_inference.py) | f32 PCM (B,1,N) @ 24kHz (from sphn.read, broad → list[(text_tokens int64 (Ttok,), audio_tokens  | — | [`moshi/run_inference.md`](Moshi-Transport) |
| moshi_run_tts (moshi/run_tts.py) | JSONL str -> TTSRequest{turns:list[str], voice → f32 waveform (B,1,total_samples) @ 24kHz -> cl | — | [`moshi/run_tts.md`](Moshi-Transport) |
| MoshiHandler gradio WebRTC client (moshi/client_gradio.py) | f32 PCM (N,) @24kHz from mic (int16-scaled, /3 → tagged Opus bytes b"x01"+opus to server; f32  | — | [`moshi/client_gradio.md`](Moshi-Transport) |

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
| ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py) | ConditionAttributes (text str/None + TensorCon → ConditionType(condition model-dtype [B,T,outpu | — | [`moshi/conditioners/base.md`](Moshi-Conditioners) |
| LUTConditioner (moshi/conditioners/text.py) | list[str/None] (len B); after tokenize tokens  → ConditionType(condition model-dtype bf16/f32 [ | — | [`moshi/conditioners/text.md`](Moshi-Conditioners) |
| TensorConditioner (moshi/conditioners/tensors.py) | TensorCondition(tensor model-dtype/f32 [B/1,T, → ConditionType(condition model-dtype/f32 [B,T,D | — | [`moshi/conditioners/tensors.md`](Moshi-Conditioners) |

### moshi/utils/

| Component | dtype in → out | on-path | spec |
|---|---|---|---|
| sample_token top-k/top-p multinomial (moshi/utils/sampling.py) | f32 logits (…, Card) — Moshi callers upcast vi → int64 token id (…,) squeezed | — | [`moshi/utils/sampling.md`](Moshi-Utilities) |
| CUDAGraphed + torch_compile gating (moshi/utils/compile.py) | passthrough: model-dtype (B,512,·) latent @25H → passthrough: wrapped module output unchanged — | ✅ | [`moshi/utils/compile.md`](Moshi-Utilities) |
| TorchAutocast (moshi/utils/autocast.py) | no tensor input — ctor args: enabled:bool + to → no tensor output — __enter__ returns None; amb | — | [`moshi/utils/autocast.md`](Moshi-Utilities) |
| QLinear int8 weight-quantize helper (moshi/utils/quantize.py) | nn.Linear.weight bf16/f32 [out,in] (module tre → weight int8 [out,in] + weight_scb fp32 [out];  | — | [`moshi/utils/quantize.md`](Moshi-Utilities) |
| cross_entropy (moshi/utils/utils.py) | logits bf16 [B,K,T,card]; targets int64 [B,K,T → ce f32 [B,K,T] (per-codebook, masked positions | — | [`moshi/utils/utils.md`](Moshi-Utilities) |

---

## Folder READMEs
- [model/](MD00-Overview) — the LFM2-Audio model graph
- [model/conformer/](CF00-Overview) — the FastConformer audio encoder
- [data/](DA00-Overview) — the training data pipeline
- [moshi/](Mimi-Codec-Overview) — the Kyutai Moshi stack (codec on-path; LM + transport off-path)
- [moshi/models/](MM00-Overview) · [moshi/modules/](MO00-Overview) ·
  [moshi/quantization/](QZ00-Overview) ·
  [moshi/conditioners/](Moshi-Conditioners) · [moshi/utils/](Moshi-Utilities)
- [demo/](DM00-Overview) — realtime runtime + turn-taking

*Off-path note:* the **Moshi-7B LM** (`moshi/models/lm.py`), its **TTS**, the **websocket
transport** (`moshi/server.py`/`client.py`), **conditioners**, and **training** are documented
for completeness but are **not** part of the LFM2-Audio inference graph — LFM2-Audio uses its
own backbone + depthformer head and (for audio-out) the Mimi codec / LFM2 detokenizer only.
