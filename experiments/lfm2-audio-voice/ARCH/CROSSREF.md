# LFM2.5-Audio — Cross-Reference / Traceability Matrix

Each architecture part has a stable **code** (2 letters = group, 2 digits = sequence). The matrix maps each code to the Python `file:line` ranges its documentation cites, grouped by doc **section** (RO=Role, HW=How it works, DT=Dtypes & shapes, WI=Wiring, PR=Python↔Rust, PG=Precision/gotchas). Generated from the doc citations; the vendored Python under `upstream-liquid-audio/src/liquid_audio/` is the referent.

## 1. Registry

| Code | Component | Python source | Doc spec |
|---|---|---|---|
| `CO01` | LFM2AudioProcessor + ChatState (processor.py) | `processor.py` | [processor.md](./processor.md) |
| `CO02` | LFM2AudioDetokenizer (detokenizer.py) | `detokenizer.py` | [detokenizer.md](./detokenizer.md) |
| `CO03` | core_utils (utils.py) | `utils.py` | [utils.md](./utils.md) |
| `CO04` | Trainer (trainer.py) | `trainer.py` | [trainer.md](./trainer.md) |
| `MD01` | LFM2AudioModel (model/lfm2_audio.py) | `model/lfm2_audio.py` | [model/lfm2_audio.md](./model/lfm2_audio.md) |
| `MD02` | Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model) | `model/lfm2_audio.py (transformers.Lfm2Model)` | [model/lfm2_backbone.md](./model/lfm2_backbone.md) |
| `MD03` | MLP audio_adapter (model/mlp.py) | `model/mlp.py` | [model/mlp.md](./model/mlp.md) |
| `MD04` | RawLMBackbone depthformer (model/transformer.py) | `model/transformer.py` | [model/transformer.md](./model/transformer.md) |
| `CF01` | ConformerEncoder (conformer/encoder.py) | `model/conformer/encoder.py` | [model/conformer/encoder.md](./model/conformer/encoder.md) |
| `CF02` | RelPositionMultiHeadAttention + RelPositionalEncoding (model/conformer/mha.py) | `model/conformer/mha.py` | [model/conformer/mha.md](./model/conformer/mha.md) |
| `CF03` | ConformerLayer/Convolution/FeedForward/CausalConv1D (model/conformer/modules.py) | `model/conformer/modules.py` | [model/conformer/modules.md](./model/conformer/modules.md) |
| `CF04` | FilterbankFeatures mel front-end (model/conformer/processor.py) | `model/conformer/processor.py` | [model/conformer/processor.md](./model/conformer/processor.md) |
| `CF05` | ConvSubsampling (model/conformer/subsampling.py) | `model/conformer/subsampling.py` | [model/conformer/subsampling.md](./model/conformer/subsampling.md) |
| `CF06` | conformer_utils (model/conformer/utils.py) | `model/conformer/utils.py` | [model/conformer/utils.md](./model/conformer/utils.md) |
| `DA01` | LFM2DataLoader + lfm2_collator (data/dataloader.py) | `data/dataloader.py` | [data/dataloader.md](./data/dataloader.md) |
| `DA02` | LFM2AudioChatMapper (data/mapper.py) | `data/mapper.py` | [data/mapper.md](./data/mapper.md) |
| `DA03` | preprocess_dataset (data/preprocess.py) | `data/preprocess.py` | [data/preprocess.md](./data/preprocess.md) |
| `DA04` | data_types (data/types.py) | `data/types.py` | [data/types.md](./data/types.md) |
| `MM01` | MimiModel codec (moshi/models/compression.py) | `moshi/models/compression.py` | [moshi/models/compression.md](./moshi/models/compression.md) |
| `MM02` | get_mimi factory + CheckpointInfo (moshi/models/loaders.py) | `moshi/models/loaders.py` | [moshi/models/loaders.md](./moshi/models/loaders.md) |
| `MM03` | LMModel + LMGen (moshi/models/lm.py) | `moshi/models/lm.py` | [moshi/models/lm.md](./moshi/models/lm.md) |
| `MM04` | ScaledEmbedding + delay/init helpers (moshi/models/lm_utils.py) | `moshi/models/lm_utils.py` | [moshi/models/lm_utils.md](./moshi/models/lm_utils.md) |
| `MM05` | TTSModel (moshi/models/tts.py) | `moshi/models/tts.py` | [moshi/models/tts.md](./moshi/models/tts.md) |
| `MO01` | SEANetEncoder/Decoder (moshi/modules/seanet.py) | `moshi/modules/seanet.py` | [moshi/modules/seanet.md](./moshi/modules/seanet.md) |
| `MO02` | StreamingConv1d/ConvTranspose1d (moshi/modules/conv.py) | `moshi/modules/conv.py` | [moshi/modules/conv.md](./moshi/modules/conv.md) |
| `MO03` | ProjectedTransformer/StreamingTransformer (moshi/modules/transformer.py) | `moshi/modules/transformer.py` | [moshi/modules/transformer.md](./moshi/modules/transformer.md) |
| `MO04` | ConvDownsample1d / ConvTrUpsample1d (moshi/modules/resample.py) | `moshi/modules/resample.py` | [moshi/modules/resample.md](./moshi/modules/resample.md) |
| `MO05` | RotaryEmbedding (moshi/modules/rope.py) | `moshi/modules/rope.py` | [moshi/modules/rope.md](./moshi/modules/rope.md) |
| `MO06` | StreamingModule[State] (moshi/modules/streaming.py) | `moshi/modules/streaming.py` | [moshi/modules/streaming.md](./moshi/modules/streaming.md) |
| `MO07` | ActivationGating / make_gating (moshi/modules/gating.py) | `moshi/modules/gating.py` | [moshi/modules/gating.md](./moshi/modules/gating.md) |
| `MO08` | LoRALinear (moshi/modules/lora.py) | `moshi/modules/lora.py` | [moshi/modules/lora.md](./moshi/modules/lora.md) |
| `QZ01` | SplitResidualVectorQuantizer (moshi/quantization/vq.py) | `moshi/quantization/vq.py` | [moshi/quantization/vq.md](./moshi/quantization/vq.md) |
| `QZ02` | EuclideanCodebook / ResidualVectorQuantization (moshi/quantization/core_vq.py) | `moshi/quantization/core_vq.py` | [moshi/quantization/core_vq.md](./moshi/quantization/core_vq.md) |
| `QZ03` | BaseQuantizer (moshi/quantization/base.py) | `moshi/quantization/base.py` | [moshi/quantization/base.md](./moshi/quantization/base.md) |
| `CN01` | ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py) | `moshi/conditioners/base.py` | [moshi/conditioners/base.md](./moshi/conditioners/base.md) |
| `CN02` | LUTConditioner (moshi/conditioners/text.py) | `moshi/conditioners/text.py` | [moshi/conditioners/text.md](./moshi/conditioners/text.md) |
| `CN03` | TensorConditioner (moshi/conditioners/tensors.py) | `moshi/conditioners/tensors.py` | [moshi/conditioners/tensors.md](./moshi/conditioners/tensors.md) |
| `MU01` | sample_token top-k/top-p multinomial (moshi/utils/sampling.py) | `moshi/utils/sampling.py` | [moshi/utils/sampling.md](./moshi/utils/sampling.md) |
| `MU02` | CUDAGraphed + torch_compile gating (moshi/utils/compile.py) | `moshi/utils/compile.py` | [moshi/utils/compile.md](./moshi/utils/compile.md) |
| `MU03` | TorchAutocast (moshi/utils/autocast.py) | `moshi/utils/autocast.py` | [moshi/utils/autocast.md](./moshi/utils/autocast.md) |
| `MU04` | QLinear int8 weight-quantize helper (moshi/utils/quantize.py) | `moshi/utils/quantize.py` | [moshi/utils/quantize.md](./moshi/utils/quantize.md) |
| `MU05` | cross_entropy (moshi/utils/utils.py) | `moshi/utils/utils.py` | [moshi/utils/utils.md](./moshi/utils/utils.md) |
| `TR01` | moshi_server (moshi/server.py) | `moshi/server.py` | [moshi/server.md](./moshi/server.md) |
| `TR02` | moshi client (moshi/client.py) | `moshi/client.py` | [moshi/client.md](./moshi/client.md) |
| `TR03` | AnyPrinter/RawPrinter/Printer (moshi/client_utils.py) | `moshi/client_utils.py` | [moshi/client_utils.md](./moshi/client_utils.md) |
| `TR04` | moshi_run_inference (moshi/run_inference.py) | `moshi/run_inference.py` | [moshi/run_inference.md](./moshi/run_inference.md) |
| `TR05` | moshi_run_tts (moshi/run_tts.py) | `moshi/run_tts.py` | [moshi/run_tts.md](./moshi/run_tts.md) |
| `TR06` | MoshiHandler gradio WebRTC client (moshi/client_gradio.py) | `moshi/client_gradio.py` | [moshi/client_gradio.md](./moshi/client_gradio.md) |
| `DM01` | demo_chat (demo/chat.py) | `demo/chat.py` | [demo/chat.md](./demo/chat.md) |
| `DM02` | demo singletons + CUDA warmup (demo/model.py) | `demo/model.py` | [demo/model.md](./demo/model.md) |

## 2. Traceability matrix (code · section → Python `file:line`)


### `CO01` — LFM2AudioProcessor + ChatState (processor.py)  ·  [processor.md](./processor.md)
| Section | Python lines |
|---|---|
| HW | `processor.py`:55-79,101-163,137-149,151,165-177,194,220-224,226-250,252-256,258-269; `detokenizer.py`:23; `lfm2_audio.py`:328-330,358-359; `utils.py`:15,40 |
| PR | `processor.py`:151 |
| PG | `processor.py`:238; `utils.py`:19 |

### `CO02` — LFM2AudioDetokenizer (detokenizer.py)  ·  [detokenizer.md](./detokenizer.md)
| Section | Python lines |
|---|---|
| RO | `detokenizer.py`:120; `processor.py`:165 |
| HW | `detokenizer.py`:6-24,27-107,35,82-83,86-92,95-101,104-105,118,121-130,126-128,131-134; `processor.py`:140-149 |
| DT | `processor.py`:165 |
| PR | `processor.py`:151 |
| PG | `processor.py`:140-149,165 |

### `CO03` — core_utils (utils.py)  ·  [utils.md](./utils.md)
| Section | Python lines |
|---|---|
| HW | `utils.py`:9,15,24,32,40; `dataloader.py`:45; `lfm2_audio.py`:144,162,330,335; `mapper.py`:156,203; `processor.py`:63,199,242 |
| PG | `lfm2_audio.py`:330 |

### `CO04` — Trainer (trainer.py)  ·  [trainer.md](./trainer.md)
| Section | Python lines |
|---|---|
| HW | `trainer.py`:21-130; `lfm2_audio.py`:104-113,453-478,463-464 |

### `MD01` — LFM2AudioModel (model/lfm2_audio.py)  ·  [model/lfm2_audio.md](./model/lfm2_audio.md)
| Section | Python lines |
|---|---|
| PG | `transformer.py`:77-78 |

### `MD02` — Lfm2Model HF backbone (model/lfm2_audio.py imports transformers.Lfm2Model)  ·  [model/lfm2_backbone.md](./model/lfm2_backbone.md)
| Section | Python lines |
|---|---|
| RO | `lfm2_audio.py`:14 |
| HW | `lfm2_audio.py`:162-165,199-205,208 |
| WI | `lfm2_audio.py`:208,366-372 |
| PR | `lfm2_audio.py`:162 |

### `MD03` — MLP audio_adapter (model/mlp.py)  ·  [model/mlp.md](./model/mlp.md)
| Section | Python lines |
|---|---|
| RO | `lfm2_audio.py`:87 |
| HW | `mlp.py`:6-37,17,20-21,23-35,32,39-40; `lfm2_audio.py`:339-355,346,350,353,355,369 |
| WI | `lfm2_audio.py`:87,369 |
| PG | `mlp.py`:32; `lfm2_audio.py`:350,355 |

### `MD04` — RawLMBackbone depthformer (model/transformer.py)  ·  [model/transformer.md](./model/transformer.md)
| Section | Python lines |
|---|---|
| RO | `lfm2_audio.py`:115-121,501-534 |
| HW | `transformer.py`:38-62,65-82,84-134,140-341,378-390,473-507,510; `lfm2_audio.py`:121,226-227,501-534 |
| PG | `transformer.py`:215-216; `lfm2_audio.py`:226-227 |

### `CF01` — ConformerEncoder (conformer/encoder.py)  ·  [model/conformer/encoder.md](./model/conformer/encoder.md)
| Section | Python lines |
|---|---|
| HW | `encoder.py`:491,641,737,850; `mha.py`:67,108,119,139,204,227,315,362,451; `modules.py`:28,84,229,393; `subsampling.py`:399,545 |
| DT | `mha.py`:71 |
| WI | `lfm2_audio.py`:87,339-346,349-350 |
| PG | `lfm2_audio.py`:330; `mha.py`:71,146,240-241 |

### `CF02` — RelPositionMultiHeadAttention + RelPositionalEncoding (model/conformer/mha.py)  ·  [model/conformer/mha.md](./model/conformer/mha.md)
| Section | Python lines |
|---|---|
| HW | `mha.py`:43,45,67,70-75,71,76,80,108,119,129,155,191-194,196-199,204,227,240-241,246-249,307,348,352-353,362,375,397,401-402,405-407,416,419,439-443,449,450,451; `encoder.py`:737 |
| DT | `mha.py`:71 |
| PR | `mha.py`:270 |
| PG | `mha.py`:43,146,241,450,451 |

### `CF03` — ConformerLayer/Convolution/FeedForward/CausalConv1D (model/conformer/modules.py)  ·  [model/conformer/modules.md](./model/conformer/modules.md)
| Section | Python lines |
|---|---|
| HW | `modules.py`:84,104-144,153,153-226,167-170,172-174,185,198-202,204-206,208,229-344,251,271-278,290-302,304,305-312,315,319-320,324-325,340,360-381,366,376-380,393-471,420-433,424-425,452-454,455-463,465-471 |
| WI | `lfm2_audio.py`:87 |

### `CF04` — FilterbankFeatures mel front-end (model/conformer/processor.py)  ·  [model/conformer/processor.md](./model/conformer/processor.md)
| Section | Python lines |
|---|---|
| RO | `processor.py`:62-67 |
| HW | `processor.py`:58,60-68,325,385-395,412-416,422,434,438-441,444,450-460,468-470,472-474,488-500,503-537,532 |
| DT | `processor.py`:238 |
| WI | `processor.py`:226-250,233; `lfm2_audio.py`:346; `utils.py`:15 |
| PG | `processor.py`:64,168,238,287,444,532 |

### `CF05` — ConvSubsampling (model/conformer/subsampling.py)  ·  [model/conformer/subsampling.md](./model/conformer/subsampling.md)
| Section | Python lines |
|---|---|
| HW | `subsampling.py`:62,108-120,122-181,324-336,351,366-392,397-399,406,545,558-586,561,594-600,603-605; `encoder.py`:324; `lfm2_audio.py`:341 |
| DT | `subsampling.py`:550,555; `lfm2_audio.py`:346 |
| WI | `lfm2_audio.py`:341,346 |
| PG | `lfm2_audio.py`:343 |

### `CF06` — conformer_utils (model/conformer/utils.py)  ·  [model/conformer/utils.md](./model/conformer/utils.md)
| Section | Python lines |
|---|---|
| HW | `utils.py`:25-40,31,32-33,35-36,38,40,42-64,66-112,93-96,99,102,105,107,109-111; `encoder.py`:429-431,891; `mha.py`:266-267,270,395 |
| WI | `encoder.py`:429,891; `mha.py`:270 |
| PG | `utils.py`:31,95; `mha.py`:266-267 |

### `DA01` — LFM2DataLoader + lfm2_collator (data/dataloader.py)  ·  [data/dataloader.md](./data/dataloader.md)
| Section | Python lines |
|---|---|
| HW | `types.py`:48,59,69; `utils.py`:9,15 |
| PG | `utils.py`:9 |

### `DA02` — LFM2AudioChatMapper (data/mapper.py)  ·  [data/mapper.md](./data/mapper.md)
| Section | Python lines |
|---|---|
| HW | `mapper.py`:30,38,47,55,56,67,75,102,110,130,153,166,181,192,196,203,207,219,223,226,229,230,231,234; `utils.py`:21 |
| DT | `mapper.py`:191 |
| PG | `mapper.py`:230; `utils.py`:21 |

### `DA03` — preprocess_dataset (data/preprocess.py)  ·  [data/preprocess.md](./data/preprocess.md)
| Section | Python lines |
|---|---|
| HW | `preprocess.py`:13-50; `mapper.py`:149-164,229-232; `types.py`:37-45; `utils.py`:15-21 |
| PG | `mapper.py`:231 |

### `DA04` — data_types (data/types.py)  ·  [data/types.md](./data/types.md)
| Section | Python lines |
|---|---|
| HW | `dataloader.py`:30-35,44-46,48,59-66,68; `lfm2_audio.py`:317-331; `mapper.py`:56,110,111,112,113-117,118,119,121 |
| DT | `preprocess.py`:26-29 |
| WI | `lfm2_audio.py`:393-413; `preprocess.py`:24-30; `trainer.py`:171 |
| PG | `dataloader.py`:46,59-66; `lfm2_audio.py`:326,398-413,398 |

### `MM01` — MimiModel codec (moshi/models/compression.py)  ·  [moshi/models/compression.md](./moshi/models/compression.md)
| Section | Python lines |
|---|---|
| HW | `compression.py`:105,387; `chat.py`:21; `loaders.py`:296-333,318,320; `resample.py`:68-109; `vq.py`:126-139,141-150,170,269-280 |
| DT | `mapper.py`:226-229 |
| WI | `chat.py`:34 |
| PR | `loaders.py`:332 |
| PG | `processor.py`:174; `vq.py`:144 |

### `MM02` — get_mimi factory + CheckpointInfo (moshi/models/loaders.py)  ·  [moshi/models/loaders.md](./moshi/models/loaders.md)
| Section | Python lines |
|---|---|
| RO | `loaders.py`:28-29,38-80,296-333 |
| HW | `loaders.py`:38-57,51-53,58-64,65-80,110,169-255,257-264,296-333,300-310,311-323,325-332,332,336-416,386-391; `processor.py`:113,114-115 |
| WI | `loaders.py`:34; `processor.py`:113 |
| PG | `loaders.py`:297; `processor.py`:113-115 |

### `MM03` — LMModel + LMGen (moshi/models/lm.py)  ·  [moshi/models/lm.md](./moshi/models/lm.md)
| Section | Python lines |
|---|---|
| HW | `lm.py`:31,76,134,138,140,145,158,178,188,198,212,218,224,300,316,343,365,373,404,444,599,662,692,708,730,771,797,803 |
| DT | `lm.py`:618,731 |
| PG | `lm.py`:31,696,731 |

### `MM04` — ScaledEmbedding + delay/init helpers (moshi/models/lm_utils.py)  ·  [moshi/models/lm_utils.md](./moshi/models/lm_utils.md)
| Section | Python lines |
|---|---|
| HW | `lm_utils.py`:88,102-124; `lm.py`:185,344; `loaders.py`:110; `transformer.py`:116-117 |
| WI | `lm.py`:127-138,344,500-513; `tts.py`:427-429 |
| PG | `transformer.py`:117 |

### `MM05` — TTSModel (moshi/models/tts.py)  ·  [moshi/models/tts.md](./moshi/models/tts.md)
| Section | Python lines |
|---|---|
| HW | `tts.py`:34-54,112-118,157-249,171-172,174-179,180-182,202-208,211-221,229-246,252-314,404,486-618,543-573,594-597,607,629-670,672-678 |
| PG | `tts.py`:464-468 |

### `MO01` — SEANetEncoder/Decoder (moshi/modules/seanet.py)  ·  [moshi/modules/seanet.md](./moshi/modules/seanet.md)
| Section | Python lines |
|---|---|
| HW | `seanet.py`:38-93,169-236,315-388; `conv.py`:42-45,223-231,240-243,248,261; `loaders.py`:41,53 |

### `MO02` — StreamingConv1d/ConvTranspose1d (moshi/modules/conv.py)  ·  [moshi/modules/conv.md](./moshi/modules/conv.md)
| Section | Python lines |
|---|---|
| HW | `conv.py`:25,29-39,42-49,52-76,79-101,91-99,113-158,161-169,223-231,233-243,240-243,245-274,248,253-259,260-261,263-267,268-273,308,340-362,352,353-360,365-419; `streaming.py`:35-42 |
| PG | `conv.py`:62-76,248,353-356 |

### `MO03` — ProjectedTransformer/StreamingTransformer (moshi/modules/transformer.py)  ·  [moshi/modules/transformer.md](./moshi/modules/transformer.md)
| Section | Python lines |
|---|---|
| RO | `lm.py`:145; `loaders.py`:302 |
| HW | `transformer.py`:932-943; `compression.py`:313; `loaders.py`:65 |
| PG | `transformer.py`:277; `rope.py`:50-66 |

### `MO04` — ConvDownsample1d / ConvTrUpsample1d (moshi/modules/resample.py)  ·  [moshi/modules/resample.md](./moshi/modules/resample.md)
| Section | Python lines |
|---|---|
| HW | `resample.py`:14-65,43-52,53-56,58-65,68-119,95-103,109-119,114-117; `compression.py`:141,189-217,267-278,280-291,314,324; `loaders.py`:320 |
| PR | `resample.py`:114-117; `compression.py`:202 |

### `MO05` — RotaryEmbedding (moshi/modules/rope.py)  ·  [moshi/modules/rope.md](./moshi/modules/rope.md)
| Section | Python lines |
|---|---|
| RO | `loaders.py`:76 |
| HW | `rope.py`:11,12,37-38,39-43,45-47,50-62,64-68,82; `transformer.py`:528,547-548,548,550,569-573 |
| DT | `rope.py`:37,50-54,65-66 |
| PR | `rope.py`:11 |
| PG | `rope.py`:37,38,39,46 |

### `MO06` — StreamingModule[State] (moshi/modules/streaming.py)  ·  [moshi/modules/streaming.md](./moshi/modules/streaming.md)
| Section | Python lines |
|---|---|
| RO | `streaming.py`:25,54,214; `server.py`:59 |
| HW | `streaming.py`:25-48,35,42,44-48,75,78-86,88-108,110,113-115,119-123,126,128,131-137,139-156,153,158-181,183-211,193,207-211,208,214-217; `compression.py`:98; `conv.py`:162,166-169; `lm.py`:529; `run_inference.py`:89; `server.py`:59,134 |
| DT | `streaming.py`:35,153; `compression.py`:222-227; `conv.py`:238,240,242 |
| WI | `compression.py`:40; `conv.py`:172; `lm.py`:49; `seanet.py`:20; `transformer.py`:328 |
| PR | `streaming.py`:207-209 |
| PG | `streaming.py`:128-129,193,208; `conv.py`:164,240; `server.py`:134 |

### `MO07` — ActivationGating / make_gating (moshi/modules/gating.py)  ·  [moshi/modules/gating.md](./moshi/modules/gating.md)
| Section | Python lines |
|---|---|
| RO | `loaders.py`:74,96 |
| HW | `gating.py`:13,14-22,17,19,20,21,50-58,60-61,67-82,70-72,85-93,110-114; `transformer.py`:75,670-699,737,743 |
| WI | `transformer.py`:75 |
| PR | `gating.py`:13,19-20,39-82,55-58,85-93; `transformer.py`:677-737 |
| PG | `gating.py`:19-20,56,60-61,63; `loaders.py`:74 |

### `MO08` — LoRALinear (moshi/modules/lora.py)  ·  [moshi/modules/lora.md](./moshi/modules/lora.md)
| Section | Python lines |
|---|---|
| HW | `lora.py`:5-22,9-16,19,25-41,30-31,35,37-38,44-97,65,71,74,76-82,83-89,91-95,97,99-107,109-114,116-118; `loaders.py`:468,471-476,482-483; `server.py`:195; `transformer.py`:409-433,437-439 |
| DT | `lora.py`:65; `loaders.py`:370 |
| WI | `loaders.py`:456-483; `transformer.py`:25 |
| PG | `lora.py`:65,71,104; `loaders.py`:371,398-401,476,482 |

### `QZ01` — SplitResidualVectorQuantizer (moshi/quantization/vq.py)  ·  [moshi/quantization/vq.md](./moshi/quantization/vq.md)
| Section | Python lines |
|---|---|
| HW | `vq.py`:21–167,73–84,95–124,114,126–139,132,141–151,170–322,195–204,269–279,281–287,286,289–322,315–317,319–322; `core_vq.py`:77–97,105–337,178–186,196–337,270–276,289–297,340–434,399,437–528,507–519,521–528 |
| DT | `mapper.py`:230,231 |
| PG | `vq.py`:144,274–277,319–322; `core_vq.py`:181–183,274–275,293–295,493,512–516; `mapper.py`:230,231; `processor.py`:174 |

### `QZ02` — EuclideanCodebook / ResidualVectorQuantization (moshi/quantization/core_vq.py)  ·  [moshi/quantization/core_vq.md](./moshi/quantization/core_vq.md)
| Section | Python lines |
|---|---|
| HW | `core_vq.py`:34-35,77-97,178-186,229-260,262-265,270-287,274-276,282,289-297,317-335,335,399-405,407-419,425-429,496-497,507-519,521-528; `vq.py`:121 |
| PG | `core_vq.py`:162-176; `processor.py`:165-177; `vq.py`:141-146 |

### `QZ03` — BaseQuantizer (moshi/quantization/base.py)  ·  [moshi/quantization/base.md](./moshi/quantization/base.md)
| Section | Python lines |
|---|---|
| HW | `base.py`:22-28,31-97,36,38-45,47-49,51-53,55-58,60-63,65-68,70-84,86-88,90-93,95-97,100-170,115-126,128-133,161-165; `compression.py`:166-167,249-265,382-384,433; `vq.py`:114 |
| WI | `compression.py`:132 |
| PG | `base.py`:129; `compression.py`:258-260; `vq.py`:144-146,315-317 |

### `CN01` — ConditionProvider/ConditionFuser/BaseConditioner (moshi/conditioners/base.py)  ·  [moshi/conditioners/base.md](./moshi/conditioners/base.md)
| Section | Python lines |
|---|---|
| HW | `base.py`:25,46,53-59,105,118,125-127,139,151,153-156,160-164,176-222,225,238-244,246,273,293,325,343,349,379-381,392,402-408,411,423; `lm.py`:392-393; `transformer.py`:130 |
| DT | `base.py`:160-164; `lm.py`:228,393,618-621 |
| WI | `lm.py`:104-105,354-357,616-621; `loaders.py`:437-453 |
| PG | `base.py`:153-156,162,176-222,379-381,407,416; `lm.py`:228,393,618-621; `transformer.py`:154 |

### `CN02` — LUTConditioner (moshi/conditioners/text.py)  ·  [moshi/conditioners/text.md](./moshi/conditioners/text.md)
| Section | Python lines |
|---|---|
| RO | `text.py`:26 |
| HW | `text.py`:18-31,34-44,85,98,101,102,118,119,125,131; `base.py`:93,118-122,120,125-127,151-165,151,160-164,325,349,379-381,392,411,423 |
| DT | `text.py`:101; `base.py`:160 |
| WI | `text.py`:26; `base.py`:293; `tts.py`:441-443 |
| PG | `text.py`:34,78,101,118; `base.py`:120,160,162 |

### `CN03` — TensorConditioner (moshi/conditioners/tensors.py)  ·  [moshi/conditioners/tensors.md](./moshi/conditioners/tensors.md)
| Section | Python lines |
|---|---|
| HW | `tensors.py`:11-13,12,13,15-16,16; `base.py`:25,40-44,46-59,118-127,118-122,131-137,151-165,153-156,160-164,319-322 |
| DT | `tensors.py`:11,15; `base.py`:158,160,160-164; `lm.py`:228 |
| WI | `base.py`:392-421; `lm.py`:354-357; `loaders.py`:424,429-430,437-447 |
| PG | `tensors.py`:12,13,16; `base.py`:32; `lm.py`:228 |

### `MU01` — sample_token top-k/top-p multinomial (moshi/utils/sampling.py)  ·  [moshi/utils/sampling.md](./moshi/utils/sampling.md)
| Section | Python lines |
|---|---|
| HW | `sampling.py`:15,51,67,86; `lfm2_audio.py`:486-497,519-529; `lm.py`:25,730,827 |
| DT | `lm.py`:730 |
| WI | `lm.py`:730,827 |

### `MU02` — CUDAGraphed + torch_compile gating (moshi/utils/compile.py)  ·  [moshi/utils/compile.md](./moshi/utils/compile.md)
| Section | Python lines |
|---|---|
| RO | `compression.py`:225-229 |
| HW | `compile.py`:24-34,37-54,57-146,169-175,190-280,283-287; `compression.py`:218-229; `gating.py`:13; `rope.py`:11; `transformer.py`:36 |
| DT | `compile.py`:243-250; `compression.py`:225,228,229 |
| WI | `streaming.py`:36 |
| PR | `compression.py`:220 |
| PG | `compile.py`:243-249 |

### `MU03` — TorchAutocast (moshi/utils/autocast.py)  ·  [moshi/utils/autocast.md](./moshi/utils/autocast.md)
| Section | Python lines |
|---|---|
| HW | `autocast.py`:26-27,29-40,34-40,42-45; `mha.py`:266; `processor.py`:444,468; `trainer.py`:176,194; `utils.py`:25-38 |
| PG | `autocast.py`:34-40; `processor.py`:444 |

### `MU04` — QLinear int8 weight-quantize helper (moshi/utils/quantize.py)  ·  [moshi/utils/quantize.md](./moshi/utils/quantize.md)
| Section | Python lines |
|---|---|
| HW | `quantize.py`:13-22,24-40,43-57; `transformer.py`:422 |
| WI | `lm.py`:24,237; `transformer.py`:20-21,443,862 |
| PG | `lm.py`:106; `transformer.py`:823 |

### `MU05` — cross_entropy (moshi/utils/utils.py)  ·  [moshi/utils/utils.md](./moshi/utils/utils.md)
| Section | Python lines |
|---|---|
| RO | `utils.py`:46-47; `lfm2_audio.py`:460 |
| HW | `utils.py`:7-52 |
| WI | `lfm2_audio.py`:460 |
| PR | `lfm2_audio.py`:460 |
| PG | `lfm2_audio.py`:455-470 |

### `TR01` — moshi_server (moshi/server.py)  ·  [moshi/server.md](./moshi/server.md)
| Section | Python lines |
|---|---|
| HW | `server.py`:47-60,62-72,74-173 |

### `TR02` — moshi client (moshi/client.py)  ·  [moshi/client.md](./moshi/client.md)
| Section | Python lines |
|---|---|
| HW | `client.py`:26,35-47,50,52,66,79,81,102,122,126,130,137-141,144,184; `client_utils.py`:11,205; `server.py`:148 |
| PG | `client.py`:117,123,130; `server.py`:148 |

### `TR03` — AnyPrinter/RawPrinter/Printer (moshi/client_utils.py)  ·  [moshi/client_utils.md](./moshi/client_utils.md)
| Section | Python lines |
|---|---|
| HW | `client.py`:185; `run_inference.py`:18,93 |
| WI | `client.py`:16; `run_inference.py`:18 |

### `TR04` — moshi_run_inference (moshi/run_inference.py)  ·  [moshi/run_inference.md](./moshi/run_inference.md)
| Section | Python lines |
|---|---|
| HW | `run_inference.py`:66-95,121-127,128-135,138-202,164-170,171-174,175-201,196-201,208-217,220-315; `lm.py`:344-369 |

### `TR05` — moshi_run_tts (moshi/run_tts.py)  ·  [moshi/run_tts.md](./moshi/run_tts.md)
| Section | Python lines |
|---|---|
| HW | `run_tts.py`:39-79,84-101,103-195,197-209 |

### `TR06` — MoshiHandler gradio WebRTC client (moshi/client_gradio.py)  ·  [moshi/client_gradio.md](./moshi/client_gradio.md)
| Section | Python lines |
|---|---|
| HW | `client_gradio.py`:21 |

### `DM01` — demo_chat (demo/chat.py)  ·  [demo/chat.md](./demo/chat.md)
| Section | Python lines |
|---|---|
| HW | `chat.py`:14,30-35,40,41-44,51-54,59,72-89,91-95,122-128; `lfm2_audio.py`:234,256-305,307; `processor.py`:184,226 |
| PR | `chat.py`:94; `model.py`:18 |
| PG | `chat.py`:31-32,80,94; `lfm2_audio.py`:276,300-301; `processor.py`:238 |

### `DM02` — demo singletons + CUDA warmup (demo/model.py)  ·  [demo/model.md](./demo/model.md)
| Section | Python lines |
|---|---|
| HW | `model.py`:13,16,18,20,23-26 |
| DT | `chat.py`:31 |
| WI | `chat.py`:11 |
| PR | `processor.py`:151 |
| PG | `model.py`:20,25; `chat.py`:31 |

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
