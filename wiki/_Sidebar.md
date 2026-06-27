**[Home](Home)**

**[Architecture map](Architecture-Map)** · **[Cross-reference](Cross-Reference)**

<details>
<summary><strong>Overview</strong></summary>

- [LFM2.5-Audio — Architecture Map (Python reference → Rust port)](Architecture-Map)
- [Architecture 1 — The Mimi Codec](Codec-Deep-Dive)
- [Component index](Component-Index)
- [LFM2.5-Audio — Cross-Reference / Traceability Matrix](Cross-Reference)
- [Kyutai Moshi stack (codec on-path; LM + transport off-path)](Mimi-Codec-Overview)
- [LFM2-Audio — Archaeological Map (Python source → Rust port)](Runtime-Questions)
- [Full system graph (all 50 components)](System-Graph)
- [Traceability matrix (code · section → Python `file:line`)](Traceability-Matrix)

</details>

<details>
<summary><strong>Model</strong></summary>

- [LFM2-Audio model graph](MD00-Overview)
- [MD01 · LFM2AudioModel (prefill + generate)](MD01-LFM2AudioModel)
- [MD02 · Lfm2Model HF backbone](MD02-LFM2-Backbone)
- [MD03 · MLP audio_adapter](MD03-Audio-Adapter-MLP)
- [MD04 · RawLMBackbone depthformer](MD04-Depthformer)

</details>

<details>
<summary><strong>Conformer Encoder</strong></summary>

- [FastConformer audio encoder](CF00-Overview)
- [CF01 · ConformerEncoder](CF01-Conformer-Encoder)
- [CF02 · RelPosition MultiHeadAttention](CF02-RelPos-MHA)
- [CF03 · ConformerLayer / Conv / FeedForward](CF03-Conformer-Layer)
- [CF04 · FilterbankFeatures mel front-end](CF04-Mel-Frontend)
- [CF05 · ConvSubsampling (dw_striding 8x)](CF05-Subsampling)
- [CF06 · Conformer utils (streaming/SD)](CF06-Conformer-Utils)

</details>

<details>
<summary><strong>Core</strong></summary>

- [CO01 · LFM2AudioProcessor + ChatState](CO01-Processor-ChatState)
- [CO02 · LFM2AudioDetokenizer (ISTFT vocoder)](CO02-Detokenizer)
- [CO03 · utils — LFMModality, mel2emb_len, get_model_dir](CO03-Utils)

</details>

<details>
<summary><strong>Mimi Codec — Models</strong></summary>

- [Mimi codec model + Moshi LM](MM00-Overview)
- [MM01 · MimiModel codec](MM01-Mimi-Codec)
- [MM02 · get_mimi factory + CheckpointInfo](MM02-Mimi-Loaders)

</details>

<details>
<summary><strong>Mimi Codec — Modules</strong></summary>

- [Codec / transformer building blocks](MO00-Overview)
- [MO01 · SEANetEncoder/Decoder](MO01-SEANet)
- [MO02 · StreamingConv1d / ConvTranspose1d](MO02-Streaming-Conv)
- [MO03 · ProjectedTransformer (codec)](MO03-Codec-Transformer)
- [MO04 · ConvDownsample/Upsample 25<->12.5Hz](MO04-Framerate-Resample)
- [MO05 · RotaryEmbedding](MO05-RoPE)
- [MO06 · StreamingModule base](MO06-Streaming-Module)
- [MO07 · ActivationGating](MO07-Gating)
- [MO08 · LoRALinear (off-path)](MO08-LoRA)

</details>

<details>
<summary><strong>Mimi Codec — Quantization</strong></summary>

- [Residual vector quantization (RVQ)](QZ00-Overview)
- [QZ01 · SplitResidualVectorQuantizer](QZ01-Split-RVQ)
- [QZ02 · EuclideanCodebook + VQ core](QZ02-VQ-Core)
- [QZ03 · BaseQuantizer / QuantizedResult](QZ03-Quantizer-Base)

</details>

<details>
<summary><strong>Data & Training</strong></summary>

- [CO04 · Trainer](CO04-Trainer)
- [Training data pipeline](DA00-Overview)
- [DA01 · LFM2DataLoader + collator](DA01-DataLoader)
- [DA02 · LFM2AudioChatMapper](DA02-Chat-Mapper)
- [DA03 · preprocess_dataset (Arrow writer)](DA03-Preprocess-Arrow)
- [DA04 · Data types](DA04-Data-Types)

</details>

<details>
<summary><strong>Runtime & Demo</strong></summary>

- [Realtime runtime + turn-taking](DM00-Overview)
- [DM01 · LFM2-Audio realtime chat](DM01-Realtime-Chat)
- [DM02 · demo singletons + warmup](DM02-Demo-Singletons)

</details>

<details>
<summary><strong>Moshi LM (off-path)</strong></summary>

- [MM03 · LMModel + LMGen (Moshi 7B LM)](MM03-Moshi-LM)
- [MM04 · Moshi LM utils](MM04-Moshi-LM-Utils)
- [MM05 · Moshi TTSModel](MM05-Moshi-TTS)

</details>

<details>
<summary><strong>Moshi Conditioners (off-path)</strong></summary>

- [Moshi Conditioners (off path)](Moshi-Conditioners)

</details>

<details>
<summary><strong>Moshi Utilities</strong></summary>

- [Moshi Utilities (off path)](Moshi-Utilities)

</details>

<details>
<summary><strong>Transport (off-path)</strong></summary>

- [Moshi Transport (off path)](Moshi-Transport)

</details>
