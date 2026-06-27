<!-- topic: Overview -->
# Full system graph (all 50 components)

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
  click moshi_cond_base "Moshi-Conditioners"
  click moshi_cond_text "Moshi-Conditioners"
  click moshi_cond_tensors "Moshi-Conditioners"
  click moshi_util_sampling "Moshi-Utilities"
  click moshi_util_compile "Moshi-Utilities"
  click moshi_util_autocast "Moshi-Utilities"
  click moshi_util_quantize "Moshi-Utilities"
  click moshi_util_utils "Moshi-Utilities"
  click moshi_server "Moshi-Transport"
  click moshi_client "Moshi-Transport"
  click moshi_client_utils "Moshi-Transport"
  click moshi_run_inference "Moshi-Transport"
  click moshi_run_tts "Moshi-Transport"
  click moshi_client_gradio "Moshi-Transport"
  click demo_chat "DM01-Realtime-Chat"
  click demo_model "DM02-Demo-Singletons"
  classDef onp fill:#0b3d2e,stroke:#19a974,color:#e8fff4,stroke-width:1px;
  classDef offp fill:#2a2a33,stroke:#6b7280,color:#cbd5e1,stroke-dasharray:4 3;```

---

