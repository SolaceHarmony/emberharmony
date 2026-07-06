<!-- topic: Overview -->
# LFM2.5-Audio — Cross-Reference / Traceability Matrix

Each architecture part has a stable **code** (2 letters = group, 2 digits = sequence). The matrix maps each code to the Python `file:line` ranges its documentation cites, grouped by doc **section** (RO=Role, HW=How it works, DT=Dtypes & shapes, WI=Wiring, PR=Python↔Rust, PG=Precision/gotchas). Generated from the doc citations; the vendored Python under `upstream-liquid-audio/src/liquid_audio/` is the referent.

## 1. Registry

| Code | Component | Python source | Doc spec |
|---|---|---|---|
| `CO01` | LFM2AudioProcessor + ChatState (processor.py) | `processor.py` | [processor.md](CO01-Processor-ChatState) |
| `CO02` | LFM2AudioDetokenizer (detokenizer.py) | `detokenizer.py` | [detokenizer.md](CO02-Detokenizer) |
| `CO03` | core_utils (utils.py) | `utils.py` | [utils.md](CO03-Utils) |
| `CO04` | Trainer (trainer.py) | `trainer.py` | [trainer.md](CO04-Trainer) |
| `MD01` | LFM2AudioModel (model/lfm2_audio.py) | `model/lfm2_audio.py` | [model/lfm2_audio.md](MD01-LFM2AudioModel) |
| `MD02` | Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model) | `model/lfm2_audio.py (transformers.Lfm2Model)` | [model/lfm2_backbone.md](MD02-LFM2-Backbone) |
| `MD03` | MLP audio_adapter (model/mlp.py) | `model/mlp.py` | [model/mlp.md](MD03-Audio-Adapter-MLP) |
| `MD04` | RawLMBackbone depthformer (model/transformer.py) | `model/transformer.py` | [model/transformer.md](MD04-Depthformer) |
| `CF01` | ConformerEncoder (conformer/encoder.py) | `model/conformer/encoder.py` | [model/conformer/encoder.md](CF01-Conformer-Encoder) |
| `CF02` | RelPositionMultiHeadAttention + RelPositionalEncoding (model/conformer/mha.py) | `model/conformer/mha.py` | [model/conformer/mha.md](CF02-RelPos-MHA) |
| `CF03` | ConformerLayer/Convolution/FeedForward/CausalConv1D (model/conformer/modules.py) | `model/conformer/modules.py` | [model/conformer/modules.md](CF03-Conformer-Layer) |
| `CF04` | FilterbankFeatures mel front-end (model/conformer/processor.py) | `model/conformer/processor.py` | [model/conformer/processor.md](CF04-Mel-Frontend) |
| `CF05` | ConvSubsampling (model/conformer/subsampling.py) | `model/conformer/subsampling.py` | [model/conformer/subsampling.md](CF05-Subsampling) |
| `CF06` | conformer_utils (model/conformer/utils.py) | `model/conformer/utils.py` | [model/conformer/utils.md](CF06-Conformer-Utils) |
| `DA01` | LFM2DataLoader + lfm2_collator (data/dataloader.py) | `data/dataloader.py` | [data/dataloader.md](DA01-DataLoader) |
| `DA02` | LFM2AudioChatMapper (data/mapper.py) | `data/mapper.py` | [data/mapper.md](DA02-Chat-Mapper) |
| `DA03` | preprocess_dataset (data/preprocess.py) | `data/preprocess.py` | [data/preprocess.md](DA03-Preprocess-Arrow) |
| `DA04` | data_types (data/types.py) | `data/types.py` | [data/types.md](DA04-Data-Types) |
| `MM01` | MimiModel codec (moshi/models/compression.py) | `moshi/models/compression.py` | [moshi/models/compression.md](MM01-Mimi-Codec) |
| `MM02` | get_mimi factory + CheckpointInfo (moshi/models/loaders.py) | `moshi/models/loaders.py` | [moshi/models/loaders.md](MM02-Mimi-Loaders) |
| `MM03` | LMModel + LMGen (moshi/models/lm.py) | `moshi/models/lm.py` | [moshi/models/lm.md](MM03-Moshi-LM) |
| `MM04` | ScaledEmbedding + delay/init helpers (moshi/models/lm_utils.py) | `moshi/models/lm_utils.py` | [moshi/models/lm_utils.md](MM04-Moshi-LM-Utils) |
| `MM05` | TTSModel (moshi/models/tts.py) | `moshi/models/tts.py` | [moshi/models/tts.md](MM05-Moshi-TTS) |
| `MO01` | SEANetEncoder/Decoder (moshi/modules/seanet.py) | `moshi/modules/seanet.py` | [moshi/modules/seanet.md](MO01-SEANet) |
| `MO02` | StreamingConv1d/ConvTranspose1d (moshi/modules/conv.py) | `moshi/modules/conv.py` | [moshi/modules/conv.md](MO02-Streaming-Conv) |
| `MO03` | ProjectedTransformer/StreamingTransformer (moshi/modules/transformer.py) | `moshi/modules/transformer.py` | [moshi/modules/transformer.md](MO03-Codec-Transformer) |
| `MO04` | ConvDownsample1d / ConvTrUpsample1d (moshi/modules/resample.py) | `moshi/modules/resample.py` | [moshi/modules/resample.md](MO04-Framerate-Resample) |
| `MO05` | RotaryEmbedding (moshi/modules/rope.py) | `moshi/modules/rope.py` | [moshi/modules/rope.md](MO05-RoPE) |
| `MO06` | StreamingModule[State] (moshi/modules/streaming.py) | `moshi/modules/streaming.py` | [moshi/modules/streaming.md](MO06-Streaming-Module) |
| `MO07` | ActivationGating / make_gating (moshi/modules/gating.py) | `moshi/modules/gating.py` | [moshi/modules/gating.md](MO07-Gating) |
| `MO08` | LoRALinear (moshi/modules/lora.py) | `moshi/modules/lora.py` | [moshi/modules/lora.md](MO08-LoRA) |
| `QZ01` | SplitResidualVectorQuantizer (moshi/quantization/vq.py) | `moshi/quantization/vq.py` | [moshi/quantization/vq.md](QZ01-Split-RVQ) |
| `QZ02` | EuclideanCodebook / ResidualVectorQuantization (moshi/quantization/core_vq.py) | `moshi/quantization/core_vq.py` | [moshi/quantization/core_vq.md](QZ02-VQ-Core) |
| `QZ03` | BaseQuantizer (moshi/quantization/base.py) | `moshi/quantization/base.py` | [moshi/quantization/base.md](QZ03-Quantizer-Base) |
| `CN01` | ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py) | `moshi/conditioners/base.py` | [moshi/conditioners/base.md](Moshi-Conditioners) |
| `CN02` | LUTConditioner (moshi/conditioners/text.py) | `moshi/conditioners/text.py` | [moshi/conditioners/text.md](Moshi-Conditioners) |
| `CN03` | TensorConditioner (moshi/conditioners/tensors.py) | `moshi/conditioners/tensors.py` | [moshi/conditioners/tensors.md](Moshi-Conditioners) |
| `MU01` | sample_token top-k/top-p multinomial (moshi/utils/sampling.py) | `moshi/utils/sampling.py` | [moshi/utils/sampling.md](Moshi-Utilities) |
| `MU02` | CUDAGraphed + torch_compile gating (moshi/utils/compile.py) | `moshi/utils/compile.py` | [moshi/utils/compile.md](Moshi-Utilities) |
| `MU03` | TorchAutocast (moshi/utils/autocast.py) | `moshi/utils/autocast.py` | [moshi/utils/autocast.md](Moshi-Utilities) |
| `MU04` | QLinear int8 weight-quantize helper (moshi/utils/quantize.py) | `moshi/utils/quantize.py` | [moshi/utils/quantize.md](Moshi-Utilities) |
| `MU05` | cross_entropy (moshi/utils/utils.py) | `moshi/utils/utils.py` | [moshi/utils/utils.md](Moshi-Utilities) |
| `TR01` | moshi_server (moshi/server.py) | `moshi/server.py` | [moshi/server.md](Moshi-Transport) |
| `TR02` | moshi client (moshi/client.py) | `moshi/client.py` | [moshi/client.md](Moshi-Transport) |
| `TR03` | AnyPrinter/RawPrinter/Printer (moshi/client_utils.py) | `moshi/client_utils.py` | [moshi/client_utils.md](Moshi-Transport) |
| `TR04` | moshi_run_inference (moshi/run_inference.py) | `moshi/run_inference.py` | [moshi/run_inference.md](Moshi-Transport) |
| `TR05` | moshi_run_tts (moshi/run_tts.py) | `moshi/run_tts.py` | [moshi/run_tts.md](Moshi-Transport) |
| `TR06` | MoshiHandler gradio WebRTC client (moshi/client_gradio.py) | `moshi/client_gradio.py` | [moshi/client_gradio.md](Moshi-Transport) |
| `DM01` | demo_chat (demo/chat.py) | `demo/chat.py` | [demo/chat.md](DM01-Realtime-Chat) |
| `DM02` | demo singletons + CUDA warmup (demo/model.py) | `demo/model.py` | [demo/model.md](DM02-Demo-Singletons) |


> Full per-code matrix → **[Traceability-Matrix](Traceability-Matrix)**

## 3. Reverse index (Python file → documenting code·section)

| Python file | Documented by |
|---|---|
| `autocast.py` | MU03·HW, MU03·PG |
| `base.py` | CN01·DT, CN01·HW, CN01·PG, CN02·DT, CN02·HW, CN02·PG, CN02·WI, CN03·DT, CN03·HW, CN03·PG, CN03·WI, QZ03·HW, QZ03·PG |
| `chat.py` | DM01·HW, DM01·PG, DM01·PR, DM02·DT, DM02·PG, DM02·WI, MM01·HW, MM01·WI |
| `client.py` | TR02·HW, TR02·PG, TR03·HW, TR03·WI |
| `client_gradio.py` | TR06·HW |
| `client_utils.py` | TR02·HW |
| `compile.py` | MU02·DT, MU02·HW, MU02·PG |
| `compression.py` | MM01·HW, MO03·HW, MO04·HW, MO04·PR, MO06·DT, MO06·HW, MO06·WI, MU02·DT, MU02·HW, MU02·PR, MU02·RO, QZ03·HW, QZ03·PG, QZ03·WI |
| `conv.py` | MO01·HW, MO02·HW, MO02·PG, MO06·DT, MO06·HW, MO06·PG, MO06·WI |
| `core_vq.py` | QZ01·HW, QZ01·PG, QZ02·HW, QZ02·PG |
| `dataloader.py` | CO03·HW, DA04·HW, DA04·PG |
| `detokenizer.py` | CO01·HW, CO02·HW, CO02·RO |
| `encoder.py` | CF01·HW, CF02·HW, CF05·HW, CF06·HW, CF06·WI |
| `gating.py` | MO07·HW, MO07·PG, MO07·PR, MU02·HW |
| `lfm2_audio.py` | CF01·PG, CF01·WI, CF03·WI, CF04·WI, CF05·DT, CF05·HW, CF05·PG, CF05·WI, CO01·HW, CO03·HW, CO03·PG, CO04·HW, DA04·HW, DA04·PG, DA04·WI, DM01·HW, DM01·PG, MD02·HW, MD02·PR, MD02·RO, MD02·WI, MD03·HW, MD03·PG, MD03·RO, MD03·WI, MD04·HW, MD04·PG, MD04·RO, MU01·HW, MU05·PG, MU05·PR, MU05·RO, MU05·WI |
| `lm.py` | CN01·DT, CN01·HW, CN01·PG, CN01·WI, CN03·DT, CN03·PG, CN03·WI, MM03·DT, MM03·HW, MM03·PG, MM04·HW, MM04·WI, MO03·RO, MO06·HW, MO06·WI, MU01·DT, MU01·HW, MU01·WI, MU04·PG, MU04·WI, TR04·HW |
| `lm_utils.py` | MM04·HW |
| `loaders.py` | CN01·WI, CN03·WI, MM01·HW, MM01·PR, MM02·HW, MM02·PG, MM02·RO, MM02·WI, MM04·HW, MO01·HW, MO03·HW, MO03·RO, MO04·HW, MO05·RO, MO07·PG, MO07·RO, MO08·DT, MO08·HW, MO08·PG, MO08·WI |
| `lora.py` | MO08·DT, MO08·HW, MO08·PG |
| `mapper.py` | CO03·HW, DA02·DT, DA02·HW, DA02·PG, DA03·HW, DA03·PG, DA04·HW, MM01·DT, QZ01·DT, QZ01·PG |
| `mha.py` | CF01·DT, CF01·HW, CF01·PG, CF02·DT, CF02·HW, CF02·PG, CF02·PR, CF06·HW, CF06·PG, CF06·WI, MU03·HW |
| `mlp.py` | MD03·HW, MD03·PG |
| `model.py` | DM01·PR, DM02·HW, DM02·PG |
| `modules.py` | CF01·HW, CF03·HW |
| `preprocess.py` | DA03·HW, DA04·DT, DA04·WI |
| `processor.py` | CF04·DT, CF04·HW, CF04·PG, CF04·RO, CF04·WI, CO01·HW, CO01·PG, CO01·PR, CO02·DT, CO02·HW, CO02·PG, CO02·PR, CO02·RO, CO03·HW, DM01·HW, DM01·PG, DM02·PR, MM01·PG, MM02·HW, MM02·PG, MM02·WI, MU03·HW, MU03·PG, QZ01·PG, QZ02·PG |
| `quantize.py` | MU04·HW |
| `resample.py` | MM01·HW, MO04·HW, MO04·PR |
| `rope.py` | MO03·PG, MO05·DT, MO05·HW, MO05·PG, MO05·PR, MU02·HW |
| `run_inference.py` | MO06·HW, TR03·HW, TR03·WI, TR04·HW |
| `run_tts.py` | TR05·HW |
| `sampling.py` | MU01·HW |
| `seanet.py` | MO01·HW, MO06·WI |
| `server.py` | MO06·HW, MO06·PG, MO06·RO, MO08·HW, TR01·HW, TR02·HW, TR02·PG |
| `streaming.py` | MO02·HW, MO06·DT, MO06·HW, MO06·PG, MO06·PR, MO06·RO, MU02·WI |
| `subsampling.py` | CF01·HW, CF05·DT, CF05·HW |
| `tensors.py` | CN03·DT, CN03·HW, CN03·PG |
| `text.py` | CN02·DT, CN02·HW, CN02·PG, CN02·RO, CN02·WI |
| `trainer.py` | CO04·HW, DA04·WI, MU03·HW |
| `transformer.py` | CN01·HW, CN01·PG, MD01·PG, MD04·HW, MD04·PG, MM04·HW, MM04·PG, MO03·HW, MO03·PG, MO05·HW, MO06·WI, MO07·HW, MO07·PR, MO07·WI, MO08·HW, MO08·WI, MU02·HW, MU04·HW, MU04·PG, MU04·WI |
| `tts.py` | CN02·WI, MM04·WI, MM05·HW, MM05·PG |
| `types.py` | DA01·HW, DA03·HW |
| `utils.py` | CF04·WI, CF06·HW, CF06·PG, CO01·HW, CO01·PG, CO03·HW, DA01·HW, DA01·PG, DA02·HW, DA02·PG, DA03·HW, MU03·HW, MU05·HW, MU05·RO |
| `vq.py` | MM01·HW, MM01·PG, QZ01·HW, QZ01·PG, QZ02·HW, QZ02·PG, QZ03·HW, QZ03·PG |
