# Python → Rust signature & return-type audit

- Scope: `core`  ·  Python root: `/Volumes/stuff/Projects/agentsdevelopment/emberharmony/experiments/lfm2-audio-voice/upstream-liquid-audio/src/liquid_audio`
- **170/170** Python functions matched to a Rust fn  ·  **0** missing
- Flags among matched: **39** arity-mismatch, **5** return-presence-mismatch

Legend: `∅` = no annotation. Flags — MISSING / ARITY py_n/rs_n / RET-py-returns-rust-unit.
Type identity is shown side-by-side for review, not auto-asserted.

### Findings (the flags are idiomatic, verified against source)
- **0 missing** — every Python function/method has a Rust counterpart.
- **ARITY** flags are arg-grouping, not dropped logic: Python's many `__init__`/
  out-params collapse into Rust **config structs** (e.g. `ConformerEncoder::new(&Config, VarBuilder)`)
  and **`&mut Acc`** accumulators (the data mapper). `ISTFT.forward(spec)`→`(&re,&im)`
  is the no-complex-dtype split. The `forward_for_export`/`streaming_*`/`change_*` ones
  are the off-path NeMo stubs (PYTHON_VS_RUST.md §2.5).
- **RET-py-returns-rust-unit**: `generate_sequential`/`generate_interleaved` are Python
  **generators** → Rust **callback** (`F`) streaming (tokens still produced); `to`/`eval`/`train`
  are torch mode-toggles implemented as documented no-ops (inference is always eval).

## `data/dataloader.py` → `data/dataloader.rs`  (4 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 15 | `LFM2DataLoader.__init__(dataset_path: str, context_length: int) -> None` | `LFM2DataLoader::new(impl Into < std :: path :: PathBuf >, usize, Vec < RawRow >, Device) -> Self` | ok |
| 24 | `LFM2DataLoader.__len__() -> int` | `LFM2DataLoader::len() -> usize` | ok |
| 27 | `LFM2DataLoader.__getitem__(idx: int) -> LFM2AudioRow` | `LFM2DataLoader::get(usize) -> Result < LFM2AudioRow >` | ok |
| 58 | `lfm2_collator(batch: list[LFM2AudioRow]) -> LFM2AudioModelInput` | `lfm2_collator(& [LFM2AudioRow]) -> Result < LFM2AudioModelInput >` | ok |

## `data/mapper.py` → `data/mapper.rs`  (8 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 17 | `LFM2AudioChatMapper.__init__(processor: LFM2AudioProcessor, codebooks: int, interleaved_text_tokens: int, interleaved_audio_tokens: int) -> None` | `LFM2AudioChatMapper < 'a >::new(& 'a LFM2AudioProcessor, usize, usize, usize) -> Self` | ok |
| 30 | `LFM2AudioChatMapper.__call__(messages: list[ChatMessage]) -> LFM2AudioTrainingSample` | `LFM2AudioChatMapper < 'a >::call(& [ChatMessage]) -> Result < LFM2AudioTrainingSample >` | ok |
| 130 | `LFM2AudioChatMapper._append_interleaved_out(text: str, audio: bytes, text_parts: list[torch.Tensor], audio_out_parts: list[torch.Tensor], modality_seq: list[int], supervision_seq: list[bool]) -> None` | `LFM2AudioChatMapper < 'a >::append_interleaved_out(& InterleavedSegment, & mut Acc) -> Result < () >` | ARITY py6/rs2 |
| 166 | `LFM2AudioChatMapper._append_text(text: str, supervised: bool, text_parts: list[torch.Tensor], modality_seq: list[int], supervision_seq: list[bool]) -> None` | `LFM2AudioChatMapper < 'a >::append_text(& str, bool, & mut Acc) -> Result < () >` | ARITY py5/rs3 |
| 181 | `LFM2AudioChatMapper._append_audio_in(wav: torch.Tensor, sampling_rate: int, mel_parts: list[torch.Tensor], audio_in_lens: list[int], modality_seq: list[int], supervision_seq: list[bool]) -> None` | `LFM2AudioChatMapper < 'a >::append_audio_in(& Tensor, u32, & mut Acc) -> Result < () >` | ARITY py6/rs3 |
| 207 | `LFM2AudioChatMapper._append_audio_out(wav: torch.Tensor, sampling_rate: int, audio_out_parts: list[torch.Tensor], modality_seq: list[int], supervision_seq: list[bool]) -> None` | `LFM2AudioChatMapper < 'a >::append_audio_out(& Tensor, u32, & mut Acc) -> Result < () >` | ARITY py5/rs3 |
| 223 | `LFM2AudioChatMapper._encode_audio_out(wav: torch.Tensor, sampling_rate: int) -> torch.Tensor` | `LFM2AudioChatMapper < 'a >::encode_audio_out(& Tensor, u32) -> Result < Tensor >` | ok |
| 235 | `LFM2AudioChatMapper._load_audio_bytes(audio: bytes) -> tuple[torch.Tensor, int]` | `LFM2AudioChatMapper < 'a >::load_audio_bytes(& [u8]) -> Result < (Tensor , u32) >` | ok |

## `data/preprocess.py` → `data/preprocess.rs`  (1 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 13 | `preprocess_dataset(data: Iterable[list[ChatMessage]], output_path: str \| Path, mapper: LFM2AudioChatMapper, max_context_length: int) -> None` | `preprocess_dataset(impl IntoIterator < Item = Vec < ChatMessage > >, impl AsRef < Path >, & impl ChatMapper, i64) -> Result < usize >` | ok |

## `data/types.py` → `data/types.rs`  (1 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 69 | `LFM2AudioModelInput.to(device: torch.device \| str) -> LFM2AudioModelInput` | `LFM2AudioModelInput::to(& Device) -> Result < Self >` | ok |

## `detokenizer.py` → `detokenizer.rs`  (6 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 9 | `FusedEmbedding.__init__(dim: int, codeboooks: int, vocab_size: int) -> ∅` | `FusedEmbedding::new(VarBuilder) -> Result < Self >` | ok |
| 21 | `FusedEmbedding.forward(x: torch.Tensor) -> torch.Tensor` | `FusedEmbedding::forward(& Tensor) -> Result < Tensor >` | ok |
| 43 | `ISTFT.__init__(n_fft: int, hop_length: int, win_length: int, padding: str) -> ∅` | `Istft::new(usize, usize, usize, VarBuilder) -> Result < Self >` | ok |
| 54 | `ISTFT.forward(spec: torch.Tensor) -> torch.Tensor` | `Istft::forward(& Tensor, & Tensor) -> Result < Tensor >` | ARITY py1/rs2 |
| 111 | `LFM2AudioDetokenizer.__init__(backbone_config: Lfm2Config) -> ∅` | `LFM2AudioDetokenizer::new(Lfm2Config, usize, VarBuilder) -> Result < Self >` | ok |
| 120 | `LFM2AudioDetokenizer.forward(x: torch.Tensor) -> torch.Tensor` | `LFM2AudioDetokenizer::forward(& Tensor) -> Result < Tensor >` | ok |

## `model/conformer/encoder.py` → `model/conformer/encoder.rs`  (18 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 176 | `ConformerEncoder.input_example(max_batch, max_dim) -> ∅` | `ConformerEncoder::input_example(usize, usize) -> ()` | ok |
| 219 | `ConformerEncoder.disabled_deployment_input_names() -> ∅` | `ConformerEncoder::disabled_deployment_input_names() -> Vec < & 'static str >` | ok |
| 226 | `ConformerEncoder.disabled_deployment_output_names() -> ∅` | `ConformerEncoder::disabled_deployment_output_names() -> Vec < & 'static str >` | ok |
| 232 | `ConformerEncoder.__init__(feat_in, n_layers, d_model, feat_out, causal_downsampling, subsampling, subsampling_factor, subsampling_conv_chunking_factor, subsampling_conv_channels, reduction, reduction_position, reduction_factor, ff_expansion_factor, self_attention_model, n_heads, att_context_size, att_context_probs, att_context_style, xscaling, untie_biases, pos_emb_max_len, conv_kernel_size, conv_norm_type, conv_context_size, use_bias, dropout, dropout_pre_encoder, dropout_emb, dropout_att, stochastic_depth_drop_prob: float, stochastic_depth_mode: str, stochastic_depth_start_layer: int, global_tokens: int, global_tokens_spacing: int, global_attn_separate: bool, use_pytorch_sdpa: bool, use_pytorch_sdpa_backends, sync_max_audio_length: bool) -> ∅` | `ConformerEncoder::new(& ConformerEncoderConfig, VarBuilder) -> Result < Self >` | ok |
| 435 | `ConformerEncoder.forward_for_export(audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len) -> ∅` | `ConformerEncoder::forward_for_export(& Tensor) -> Result < Tensor >` | ARITY py5/rs1 |
| 466 | `ConformerEncoder.streaming_post_process(rets, keep_all_outputs) -> ∅` | `ConformerEncoder::streaming_post_process(Tensor) -> Tensor` | ARITY py2/rs1 |
| 491 | `ConformerEncoder.forward(audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len, bypass_pre_encode) -> ∅` | `ConformerEncoder::forward(& Tensor) -> Result < Tensor >` | ARITY py6/rs1 |
| 537 | `ConformerEncoder.forward_internal(audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len, bypass_pre_encode) -> ∅` | `ConformerEncoder::forward_internal(& Tensor) -> Result < Tensor >` | ARITY py6/rs1 |
| 704 | `ConformerEncoder.update_max_seq_length(seq_length: int, device) -> ∅` | `ConformerEncoder::update_max_seq_length(usize, & candle_core :: Device) -> ()` | ok |
| 724 | `ConformerEncoder.set_max_audio_length(max_audio_length) -> ∅` | `ConformerEncoder::set_max_audio_length(usize) -> ()` | ok |
| 737 | `ConformerEncoder._create_masks(att_context_size, padding_length, max_audio_length, offset, device) -> ∅` | `ConformerEncoder::create_masks() -> (Option < Tensor > , Option < Tensor >)` | ARITY py5/rs0 |
| 793 | `ConformerEncoder.enable_pad_mask(on) -> ∅` | `ConformerEncoder::enable_pad_mask(bool) -> bool` | ok |
| 805 | `ConformerEncoder._calc_context_sizes(att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size) -> ∅` | `ConformerEncoder::calc_context_sizes(Option < Vec < i64 > >, Option < Vec < Vec < i64 > > >, Option < Vec < f64 > >, & str, Option < ConvContextSize >, i64) -> Result < (Vec < Vec < i64 > > , Vec < i64 > , Vec < f64 > , ConvContextSize) >` | ARITY py5/rs6 |
| 853 | `ConformerEncoder.set_default_att_context_size(att_context_size) -> ∅` | `ConformerEncoder::set_default_att_context_size((i64 , i64)) -> ()` | ok |
| 870 | `ConformerEncoder.setup_streaming_params(chunk_size: int, shift_size: int, left_chunks: int, att_context_size: list, max_context: int) -> ∅` | `ConformerEncoder::setup_streaming_params() -> ()` | ARITY py5/rs0 |
| 977 | `ConformerEncoder.get_initial_cache_state(batch_size, dtype, device, max_dim) -> ∅` | `ConformerEncoder::get_initial_cache_state() -> Option < Tensor >` | ARITY py4/rs0 |
| 1017 | `ConformerEncoder.change_attention_model(self_attention_model: str, att_context_size: list[int], update_config: bool, device: torch.device) -> ∅` | `ConformerEncoder::change_attention_model(& str) -> ()` | ARITY py4/rs1 |
| 1146 | `ConformerEncoder.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor: int) -> ∅` | `ConformerEncoder::change_subsampling_conv_chunking_factor(i64) -> Result < () >` | ok |

## `model/conformer/mha.py` → `model/conformer/mha.rs`  (14 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 55 | `PositionalEncoding.__init__(d_model, dropout_rate, max_len, xscale, dropout_rate_emb) -> ∅` | `PositionalEncoding::new(usize, usize, Option < f64 >) -> Self` | ok |
| 67 | `PositionalEncoding.create_pe(positions, dtype) -> ∅` | `PositionalEncoding::create_pe(& Tensor, DType) -> Result < Tensor >` | ok |
| 82 | `PositionalEncoding.extend_pe(length, device, dtype) -> ∅` | `PositionalEncoding::extend_pe(usize, & Device, DType) -> Result < Tensor >` | ok |
| 89 | `PositionalEncoding.forward(x: torch.Tensor, cache_len) -> ∅` | `PositionalEncoding::forward(& Tensor, usize) -> Result < (Tensor , Tensor) >` | ok |
| 119 | `RelPositionalEncoding.extend_pe(length, device, dtype) -> ∅` | `RelPositionalEncoding::extend_pe(usize, & Device, DType) -> Result < Tensor >` | ok |
| 129 | `RelPositionalEncoding.forward(x, cache_len) -> ∅` | `RelPositionalEncoding::forward(& Tensor) -> Result < (Tensor , Tensor) >` | ARITY py2/rs1 |
| 166 | `MultiHeadAttention.__init__(n_head, n_feat, dropout_rate, max_cache_len, use_bias, use_pytorch_sdpa, use_pytorch_sdpa_backends) -> ∅` | `MultiHeadAttention::new(usize, usize, bool, VarBuilder) -> Result < Self >` | ok |
| 204 | `MultiHeadAttention.forward_qkv(query, key, value) -> ∅` | `MultiHeadAttention::forward_qkv(& Tensor, & Tensor, & Tensor) -> Result < (Tensor , Tensor , Tensor) >` | ok |
| 227 | `MultiHeadAttention.forward_attention(value, scores, mask) -> ∅` | `MultiHeadAttention::forward_attention(& Tensor, & Tensor, Option < & Tensor >) -> Result < Tensor >` | ok |
| 251 | `MultiHeadAttention.forward(query, key, value, mask, pos_emb, cache) -> ∅` | `MultiHeadAttention::forward(& Tensor, & Tensor, & Tensor, Option < & Tensor >) -> Result < Tensor >` | ARITY py6/rs4 |
| 307 | `MultiHeadAttention.update_cache(key, value, query, cache) -> ∅` | `MultiHeadAttention::update_cache(& Tensor, & Tensor, & Tensor, Option < & Tensor >) -> Result < (Tensor , Tensor , Tensor , Option < Tensor >) >` | ok |
| 325 | `RelPositionMultiHeadAttention.__init__(n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, max_cache_len, use_bias, use_pytorch_sdpa, use_pytorch_sdpa_backends) -> ∅` | `RelPositionMultiHeadAttention::new(usize, usize, bool, VarBuilder) -> Result < Self >` | ok |
| 362 | `RelPositionMultiHeadAttention.rel_shift(x) -> ∅` | `RelPositionMultiHeadAttention::rel_shift(& Tensor) -> Result < Tensor >` | ok |
| 375 | `RelPositionMultiHeadAttention.forward(query, key, value, mask, pos_emb, cache) -> ∅` | `RelPositionMultiHeadAttention::forward(& Tensor, & Tensor, & Tensor, Option < & Tensor >, & Tensor) -> Result < Tensor >` | ARITY py6/rs5 |

## `model/conformer/modules.py` → `model/conformer/modules.rs`  (11 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 55 | `ConformerLayer.__init__(d_model, d_ff, self_attention_model, global_tokens, global_tokens_spacing, global_attn_separate, n_heads, conv_kernel_size, conv_norm_type, conv_context_size, dropout, dropout_att, pos_bias_u, pos_bias_v, att_context_size, use_bias, use_pytorch_sdpa, use_pytorch_sdpa_backends) -> ∅` | `ConformerLayer::new(usize, usize, usize, usize, bool, VarBuilder) -> Result < Self >` | ok |
| 153 | `ConformerLayer.forward(x, att_mask, pos_emb, pad_mask, cache_last_channel, cache_last_time) -> ∅` | `ConformerLayer::forward(& Tensor, Option < & Tensor >, & Tensor, Option < & Tensor >) -> Result < Tensor >` | ARITY py6/rs4 |
| 241 | `ConformerConvolution.__init__(d_model, kernel_size, norm_type, conv_context_size, pointwise_activation, use_bias) -> ∅` | `ConformerConvolution::new(usize, usize, bool, VarBuilder) -> Result < Self >` | ok |
| 314 | `ConformerConvolution.forward(x, pad_mask, cache) -> ∅` | `ConformerConvolution::forward(& Tensor, Option < & Tensor >) -> Result < Tensor >` | ARITY py3/rs2 |
| 346 | `ConformerConvolution.reset_parameters_conv() -> ∅` | `ConformerConvolution::reset_parameters_conv() -> ()` | ok |
| 366 | `ConformerFeedForward.__init__(d_model, d_ff, dropout, activation, use_bias) -> ∅` | `ConformerFeedForward::new(usize, usize, VarBuilder) -> Result < Self >` | ok |
| 376 | `ConformerFeedForward.forward(x) -> ∅` | `ConformerFeedForward::forward(& Tensor) -> Result < Tensor >` | ok |
| 383 | `ConformerFeedForward.reset_parameters_ff() -> ∅` | `ConformerFeedForward::reset_parameters_ff() -> ()` | ok |
| 405 | `CausalConv1D.__init__(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: str \| int, dilation: int, groups: int, bias: bool, padding_mode: str, device, dtype) -> None` | `CausalConv1D::new(usize, usize, usize, usize, CausalPadding, usize, VarBuilder) -> Result < Self >` | ok |
| 451 | `CausalConv1D.update_cache(x, cache) -> ∅` | `CausalConv1D::update_cache(& Tensor, Option < & Tensor >) -> Result < (Tensor , Option < Tensor >) >` | ok |
| 465 | `CausalConv1D.forward(x, cache) -> ∅` | `CausalConv1D::forward(& Tensor, Option < & Tensor >) -> Result < (Tensor , Option < Tensor >) >` | ok |

## `model/conformer/processor.py` → `model/conformer/processor.rs`  (16 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 34 | `AudioPreprocessor.__init__(win_length, hop_length) -> ∅` | `AudioPreprocessor::new(usize, usize) -> Self` | ok |
| 61 | `AudioPreprocessor.forward(input_signal, length) -> ∅` | `AudioPreprocessor::forward(& Tensor) -> Result < Tensor >` | ARITY py2/rs1 |
| 71 | `AudioPreprocessor.get_features(input_signal, length) -> ∅` | `AudioPreprocessor::get_features(& Tensor, Option < & Tensor >) -> Result < (Tensor , Option < Tensor >) >` | ok |
| 145 | `AudioToMelSpectrogramPreprocessor.save_to(save_path: str) -> ∅` | `AudioToMelSpectrogramPreprocessor::save_to(& str) -> ()` | ok |
| 149 | `AudioToMelSpectrogramPreprocessor.restore_from(restore_path: str) -> ∅` | `AudioToMelSpectrogramPreprocessor::restore_from(& str) -> ()` | ok |
| 152 | `AudioToMelSpectrogramPreprocessor.__init__(sample_rate, window_size, window_stride, n_window_size, n_window_stride, window, normalize, n_fft, preemph, features, lowfreq, highfreq, log, log_zero_guard_type, log_zero_guard_value, dither, pad_to, frame_splicing, exact_pad, pad_value, mag_power, rng, nb_augmentation_prob, nb_max_freq, use_torchaudio: bool, mel_norm, stft_exact_pad, stft_conv) -> ∅` | `AudioToMelSpectrogramPreprocessor::new(FilterbankFeatures) -> Self` | ok |
| 229 | `AudioToMelSpectrogramPreprocessor.input_example(max_batch: int, max_dim: int, min_length: int) -> ∅` | `AudioToMelSpectrogramPreprocessor::input_example(usize, usize, usize) -> ()` | ok |
| 237 | `AudioToMelSpectrogramPreprocessor.get_features(input_signal, length) -> ∅` | `AudioToMelSpectrogramPreprocessor::get_features(& Tensor, Option < & Tensor >) -> Result < (Tensor , Option < Tensor >) >` | ok |
| 241 | `AudioToMelSpectrogramPreprocessor.filter_banks() -> ∅` | `AudioToMelSpectrogramPreprocessor::filter_banks() -> & Tensor` | ok |
| 250 | `FilterbankFeatures.__init__(sample_rate, n_window_size, n_window_stride, window, normalize, n_fft, preemph, nfilt, lowfreq, highfreq, log, log_zero_guard_type, log_zero_guard_value, dither, pad_to, max_duration, frame_splicing, exact_pad, pad_value, mag_power, use_grads, rng, nb_augmentation_prob, nb_max_freq, mel_norm, stft_exact_pad, stft_conv) -> ∅` | `FilterbankFeatures::new(MelConfig, & Device) -> Result < Self >` | ok |
| 385 | `FilterbankFeatures.stft(x) -> ∅` | `FilterbankFeatures::stft(& Tensor) -> Result < (Tensor , Tensor) >` | ok |
| 397 | `FilterbankFeatures.log_zero_guard_value_fn(x) -> ∅` | `FilterbankFeatures::log_zero_guard_value_fn(& Tensor) -> f64` | ok |
| 412 | `FilterbankFeatures.get_seq_len(seq_len) -> ∅` | `FilterbankFeatures::get_seq_len(usize) -> usize` | ok |
| 419 | `FilterbankFeatures.filter_banks() -> ∅` | `FilterbankFeatures::filter_banks() -> & Tensor` | ok |
| 422 | `FilterbankFeatures.forward(x, seq_len, linear_spec) -> ∅` | `FilterbankFeatures::forward(& Tensor) -> Result < Tensor >` | ARITY py3/rs1 |
| 503 | `normalize_batch(x, seq_len, normalize_type) -> ∅` | `normalize_batch(& Tensor, usize) -> Result < Tensor >` | ARITY py3/rs2 |

## `model/conformer/subsampling.py` → `model/conformer/subsampling.rs`  (14 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 43 | `ConvSubsampling.__init__(subsampling, subsampling_factor, feat_in, feat_out, conv_channels, subsampling_conv_chunking_factor, activation, is_causal) -> ∅` | `ConvSubsampling::new(usize, usize, usize, usize, VarBuilder) -> Result < Self >` | ok |
| 345 | `ConvSubsampling.get_sampling_frames() -> ∅` | `ConvSubsampling::get_sampling_frames() -> [usize ; 2]` | ok |
| 348 | `ConvSubsampling.get_streaming_cache_size() -> ∅` | `ConvSubsampling::get_streaming_cache_size() -> [usize ; 2]` | ok |
| 351 | `ConvSubsampling.forward(x, lengths) -> ∅` | `ConvSubsampling::forward(& Tensor) -> Result < Tensor >` | ARITY py2/rs1 |
| 406 | `ConvSubsampling.reset_parameters() -> ∅` | `ConvSubsampling::reset_parameters() -> ()` | ok |
| 429 | `ConvSubsampling.conv_split_by_batch(x, lengths) -> ∅` | `ConvSubsampling::conv_split_by_batch(& Tensor, Vec < usize >) -> Result < (Tensor , Vec < usize > , bool) >` | ok |
| 462 | `ConvSubsampling.conv_split_by_channel(x) -> ∅` | `ConvSubsampling::conv_split_by_channel(& Tensor) -> Result < Tensor >` | ok |
| 501 | `ConvSubsampling.channel_chunked_conv(conv, chunk_size, x) -> ∅` | `ConvSubsampling::channel_chunked_conv(& Conv2d, usize, & Tensor) -> Result < Tensor >` | ok |
| 535 | `ConvSubsampling.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor: int) -> ∅` | `ConvSubsampling::change_subsampling_conv_chunking_factor(i64) -> Result < () >` | ok |
| 545 | `calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num) -> ∅` | `calc_length(usize, i64, i64, i64, bool, usize) -> usize` | ok |
| 559 | `MaskedConvSequential.forward(x, lengths) -> ∅` | `MaskedConvSequential::forward(& Tensor, & [usize], i64, i64, (i64 , i64)) -> Result < (Tensor , Vec < usize >) >` | ARITY py2/rs5 |
| 588 | `MaskedConvSequential._create_mask(tensor, lengths) -> ∅` | `MaskedConvSequential::create_mask(& Tensor, & [usize]) -> Result < Tensor >` | ok |
| 594 | `apply_channel_mask(tensor, mask) -> ∅` | `apply_channel_mask(& Tensor, & Tensor) -> Result < Tensor >` | ok |
| 603 | `calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]) -> ∅` | `calculate_conv_output_size(i64, i64, i64, (i64 , i64)) -> i64` | ok |

## `model/conformer/utils.py` → `model/conformer/utils.rs`  (2 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 25 | `avoid_float16_autocast_context() -> ∅` | `avoid_float16_autocast_context() -> ()` | ok |
| 66 | `compute_stochastic_depth_drop_probs(num_layers: int, stochastic_depth_drop_prob: float, stochastic_depth_mode: str, stochastic_depth_start_layer: int) -> list[float]` | `compute_stochastic_depth_drop_probs(usize, f64, & str, usize) -> Vec < f64 >` | ok |

## `model/lfm2_audio.py` → `model/lfm2_audio.rs`  (9 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 73 | `LFM2AudioModel.__init__(conf: LFM2AudioConfig) -> ∅` | `LFM2AudioModel::new(Lfm2Config, & ConformerEncoderConfig, & DepthformerConfig, usize, usize, usize, & LossConf, VarBuilder) -> Result < Self >` | ok |
| 136 | `LFM2AudioModel.from_pretrained(repo_id: str \| Path, revision: str \| None, dtype: torch.dtype, device: torch.device \| str) -> Self` | `LFM2AudioModel::from_pretrained(& std :: path :: Path, DType, & candle_core :: Device) -> Result < (Self , crate :: processor :: LFM2AudioProcessor) >` | ARITY py4/rs3 |
| 172 | `LFM2AudioModel.generate_sequential(text: torch.Tensor, audio_in: torch.Tensor, audio_in_lens: torch.Tensor, audio_out: torch.Tensor, modality_flag: torch.Tensor, max_new_tokens: int, text_temperature: float \| None, text_top_k: int \| None, audio_temperature: float \| None, audio_top_k: int \| None) -> Generator[torch.Tensor, None, None]` | `LFM2AudioModel::generate_sequential(& ChatState, & GenParams, F) -> Result < () >` | ARITY py10/rs3 RET-py-returns-rust-unit |
| 234 | `LFM2AudioModel.generate_interleaved(text: torch.Tensor, audio_in: torch.Tensor, audio_in_lens: torch.Tensor, audio_out: torch.Tensor, modality_flag: torch.Tensor, max_new_tokens: int, text_temperature: float \| None, text_top_k: int \| None, audio_temperature: float \| None, audio_top_k: int \| None) -> Generator[torch.Tensor, None, None]` | `LFM2AudioModel::generate_interleaved(& ChatState, & GenParams, F) -> Result < () >` | ARITY py10/rs3 RET-py-returns-rust-unit |
| 307 | `LFM2AudioModel._prefill(text: torch.Tensor, audio_in: torch.Tensor, audio_in_lens: torch.Tensor, audio_out: torch.Tensor, modality_flag: torch.Tensor) -> torch.Tensor` | `LFM2AudioModel::prefill(& ChatState) -> Result < Tensor >` | ARITY py5/rs1 |
| 374 | `LFM2AudioModel.logits(batch: LFM2AudioModelInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` | `LFM2AudioModel::logits(& LFM2AudioModelInput) -> Result < (Tensor , Tensor , Tensor , Tensor) >` | ok |
| 453 | `LFM2AudioModel.forward(batch: LFM2AudioModelInput) -> LFM2AudioModelOutput` | `LFM2AudioModel::forward(& LFM2AudioModelInput) -> Result < LFM2AudioModelOutput >` | ok |
| 483 | `LFM2AudioModel._sample_text_token(logits: torch.Tensor, temperature: float \| None, top_k: int \| None) -> torch.Tensor` | `LFM2AudioModel::sample_text_token(& Tensor, & mut Sampler) -> Result < u32 >` | ARITY py3/rs2 |
| 501 | `LFM2AudioModel._sample_audio_frame(embedding: torch.Tensor, temperature: float \| None, top_k: int \| None) -> torch.Tensor` | `LFM2AudioModel::sample_audio_frame(& Tensor, & mut Sampler) -> Result < Vec < u32 > >` | ARITY py3/rs2 |

## `model/mlp.py` → `model/mlp.rs`  (2 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 6 | `MLP.__init__(in_channels: int, out_channels: int, hidden_dim: list[int], bias: bool, use_layer_norm: bool, dropout: float) -> ∅` | `MLP::new(usize, usize, & [usize], bool, bool, f64, VarBuilder) -> Result < Self >` | ok |
| 39 | `MLP.forward(x: torch.Tensor) -> torch.Tensor` | `MLP::forward(& Tensor) -> Result < Tensor >` | ok |

## `model/transformer.py` → `model/transformer.rs`  (33 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 28 | `SequenceModel.__init__(*args, **kwargs) -> None` | `SequenceModel::new() -> ()` | ok |
| 32 | `SequenceModel.forward(x: torch.Tensor, cache: CacheType) -> torch.Tensor` | `SequenceModel::forward(& Tensor, Option < & mut [LayerKvCache] >) -> Result < Tensor >` | ok |
| 35 | `SequenceModel.forward_cached(x: torch.Tensor, cache: CacheType) -> tuple[torch.Tensor, CacheType]` | `SequenceModel::forward_cached(& Tensor, Option < Vec < LayerCache > >) -> Result < (Tensor , Vec < LayerCache >) >` | ok |
| 43 | `LayerKVCache.__init__(cache: tuple[torch.Tensor, torch.Tensor] \| None) -> None` | `LayerKvCache::new() -> Self` | ok |
| 48 | `LayerKVCache.update(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` | `LayerKvCache::update(& Tensor, & Tensor) -> Result < (Tensor , Tensor) >` | ok |
| 58 | `LayerKVCache.get_cache_size() -> int` | `LayerKvCache::get_cache_size() -> usize` | ok |
| 66 | `RMSNorm.__init__(dim: int, eps: float) -> None` | `RmsNorm::new(usize, f64, VarBuilder) -> Result < Self >` | ok |
| 71 | `RMSNorm._norm(x: torch.Tensor) -> torch.Tensor` | `RmsNorm::norm(& Tensor) -> Result < Tensor >` | ok |
| 74 | `RMSNorm.forward(x: torch.Tensor, cache: CacheType) -> torch.Tensor` | `RmsNorm::forward(& Tensor) -> Result < Tensor >` | ARITY py2/rs1 |
| 80 | `RMSNorm.forward_cached(x: torch.Tensor, cache: CacheType) -> tuple[torch.Tensor, CacheType]` | `RmsNorm::forward_cached(& Tensor) -> Result < (Tensor , LayerCache) >` | ARITY py2/rs1 |
| 85 | `GLU.__init__(dim: int, ff_dim: int \| None, mlp_init_scale: float, out_init_scale: float, use_swiglu: bool, multiple_of: int, ffn_dim_multiplier: float) -> ∅` | `Glu::new(usize, Option < usize >, bool, usize, f64, VarBuilder) -> Result < Self >` | ok |
| 129 | `GLU.forward(x: torch.Tensor, cache: CacheType) -> torch.Tensor` | `Glu::forward(& Tensor) -> Result < Tensor >` | ARITY py2/rs1 |
| 136 | `GLU.forward_cached(x: torch.Tensor, cache: CacheType) -> tuple[torch.Tensor, CacheType]` | `Glu::forward_cached(& Tensor) -> Result < (Tensor , LayerCache) >` | ARITY py2/rs1 |
| 141 | `BoundedAttention.__init__(dim: int, num_heads: int, head_style: Literal['mha', 'gqa', 'mqa'], gqa_dim: int \| None, qk_layernorm: bool, norm_eps: float) -> None` | `BoundedAttention::new(usize, usize, HeadStyle, usize, bool, f64, VarBuilder) -> Result < Self >` | ok |
| 171 | `BoundedAttention.forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, freqs_cis: torch.Tensor \| None, cache: LayerKVCache \| None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]` | `BoundedAttention::forward(& Tensor, & Tensor, & Tensor, & Tensor, & Tensor, Option < & mut LayerKvCache >) -> Result < Tensor >` | ARITY py5/rs6 |
| 230 | `MHA.__init__(dim: int, num_heads: int, head_style: Literal['mha', 'gqa', 'mqa'], out_init_scale: float, proj_init_scale: float, qk_layernorm: bool, norm_eps: float, gqa_dim: int, freqs_cis: torch.Tensor \| None, max_seq_len: int, theta: float) -> ∅` | `Mha::new(usize, usize, HeadStyle, bool, f64, usize, usize, f64, VarBuilder) -> Result < Self >` | ok |
| 295 | `MHA._validate_cache(cache: CacheType) -> TypeGuard[tuple[torch.Tensor, torch.Tensor]]` | `Mha::validate_cache(& LayerCache) -> bool` | ok |
| 303 | `MHA.forward(x: torch.Tensor, cache: CacheType) -> torch.Tensor` | `Mha::forward(& Tensor, Option < & mut LayerKvCache >) -> Result < Tensor >` | ok |
| 306 | `MHA.forward_cached(x: torch.Tensor, cache: CacheType) -> tuple[torch.Tensor, CacheType]` | `Mha::forward_cached(& Tensor, LayerCache) -> Result < (Tensor , LayerCache) >` | ok |
| 347 | `StandardBlock.__init__(operator: SequenceModel, ff_dim: int \| None, mlp_init_scale: float, out_init_scale: float, use_swiglu: bool, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float) -> ∅` | `StandardBlock::new(Mha, Option < usize >, bool, usize, f64, f64, VarBuilder) -> Result < Self >` | ok |
| 378 | `StandardBlock.forward(x: torch.Tensor, cache: CacheType) -> torch.Tensor` | `StandardBlock::forward(& Tensor, Option < & mut LayerKvCache >) -> Result < Tensor >` | ok |
| 385 | `StandardBlock.forward_cached(x: torch.Tensor, cache: CacheType \| None) -> tuple[torch.Tensor, CacheType]` | `StandardBlock::forward_cached(& Tensor, LayerCache) -> Result < (Tensor , LayerCache) >` | ok |
| 393 | `apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` | `apply_rotary_emb(& Tensor, & Tensor, & Tensor, & Tensor) -> Result < (Tensor , Tensor) >` | ARITY py3/rs4 |
| 425 | `reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor` | `reshape_for_broadcast(& Tensor, & Tensor) -> Result < Tensor >` | ok |
| 450 | `precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor` | `precompute_freqs_cis(usize, usize, f64, & Device) -> Result < (Tensor , Tensor) >` | ARITY py3/rs4 |
| 474 | `SharedEmbedding.__init__(dim: int, vocab_size: int, embed_init_scale: float, norm_eps: float, tie_embedding: bool) -> None` | `SharedEmbedding::new(usize, usize, f64, VarBuilder) -> Result < Self >` | ok |
| 500 | `SharedEmbedding.forward(tokens: torch.Tensor) -> torch.Tensor` | `SharedEmbedding::forward(& Tensor) -> Result < Tensor >` | ok |
| 503 | `SharedEmbedding.embed(tokens: torch.Tensor) -> torch.Tensor` | `SharedEmbedding::embed(& Tensor) -> Result < Tensor >` | ok |
| 506 | `SharedEmbedding.get_logits(embeddings: torch.Tensor) -> torch.Tensor` | `SharedEmbedding::get_logits(& Tensor) -> Result < Tensor >` | ok |
| 517 | `RawLMBackbone.__init__(layers: Iterable[SequenceModel], vocab_size: int, norm_eps: float, embed_init_scale: float, has_embedding: bool, tie_embedding: bool) -> None` | `RawLmBackbone::new(Vec < StandardBlock >, Option < SharedEmbedding >, usize) -> Self` | ok |
| 542 | `RawLMBackbone.forward(x: torch.Tensor, cache: CacheType \| None) -> torch.Tensor` | `RawLmBackbone::forward(& Tensor, Option < & mut [LayerKvCache] >) -> Result < Tensor >` | ok |
| 554 | `RawLMBackbone.forward_cached(x: torch.Tensor, cache: CacheType \| None) -> tuple[torch.Tensor, CacheType]` | `RawLmBackbone::forward_cached(& Tensor, Option < Vec < LayerCache > >) -> Result < (Tensor , Vec < LayerCache >) >` | ok |
| 569 | `wrap_activation_checkpoint(mod: T) -> T` | `wrap_activation_checkpoint(T) -> T` | ok |

## `processor.py` → `processor.rs`  (22 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 37 | `LFM2AudioProcessor.__init__(text_tokenizer_path: str, audio_processor_config: PreprocessorConfig, mimi_weights_path: str \| None, detokenizer_path: str \| None, name: str \| None) -> None` | `LFM2AudioProcessor::new(Tokenizer, FilterbankFeatures, Option < Box < dyn AudioDetokenizer > >, Device) -> Self` | ok |
| 56 | `LFM2AudioProcessor.from_pretrained(repo_id: str \| Path, revision: str \| None, device: torch.device \| str) -> Self` | `LFM2AudioProcessor::from_pretrained(& Path, candle_core :: DType, & Device) -> Result < Self >` | ok |
| 81 | `LFM2AudioProcessor.to(device: str \| torch.device \| None, dtype: torch.dtype \| None) -> Self` | `LFM2AudioProcessor::to() -> ()` | ARITY py2/rs0 RET-py-returns-rust-unit |
| 85 | `LFM2AudioProcessor.eval() -> Self` | `LFM2AudioProcessor::eval() -> ()` | RET-py-returns-rust-unit |
| 89 | `LFM2AudioProcessor.train() -> Self` | `LFM2AudioProcessor::train() -> ()` | RET-py-returns-rust-unit |
| 94 | `LFM2AudioProcessor.text() -> PreTrainedTokenizer` | `LFM2AudioProcessor::text() -> & Tokenizer` | ok |
| 98 | `LFM2AudioProcessor.audio() -> AudioToMelSpectrogramPreprocessor` | `LFM2AudioProcessor::audio() -> & FilterbankFeatures` | ok |
| 102 | `LFM2AudioProcessor.mimi() -> MimiModel` | `LFM2AudioProcessor::mimi() -> Option < & dyn AudioDetokenizer >` | ok |
| 122 | `LFM2AudioProcessor.audio_detokenizer() -> LFM2AudioDetokenizer` | `LFM2AudioProcessor::audio_detokenizer() -> Option < & dyn AudioDetokenizer >` | ok |
| 166 | `LFM2AudioProcessor.decode(audio_codes: torch.Tensor) -> torch.Tensor` | `LFM2AudioProcessor::decode(& Tensor) -> Result < Tensor >` | ok |
| 180 | `LFM2AudioProcessor.device() -> torch.device` | `LFM2AudioProcessor::device() -> & Device` | ok |
| 187 | `ChatState.__init__(processor: LFM2AudioProcessor, codebooks: int, dtype: torch.dtype) -> None` | `ChatState < 'a >::new(& 'a LFM2AudioProcessor, usize) -> Result < Self >` | ok |
| 201 | `ChatState.__repr__() -> str` | `ChatState < '_ >::fmt(& mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` | ARITY py0/rs1 |
| 205 | `ChatState.__getitem__(name: str) -> Any` | `ChatState < '_ >::get(& str) -> Result < & Tensor >` | ok |
| 210 | `ChatState.__iter__() -> Iterator[str]` | `ChatState < '_ >::iter() -> impl Iterator < Item = & 'static str >` | ok |
| 213 | `ChatState.__len__() -> int` | `ChatState < '_ >::len() -> usize` | ok |
| 217 | `ChatState.device() -> torch.device` | `ChatState < '_ >::device() -> & Device` | ok |
| 220 | `ChatState.add_text(text: str) -> None` | `ChatState < 'a >::add_text(& str) -> Result < () >` | ok |
| 226 | `ChatState.add_audio(wave: torch.Tensor, sampling_rate: int) -> None` | `ChatState < 'a >::add_audio(& Tensor, u32) -> Result < () >` | ok |
| 252 | `ChatState.end_turn() -> None` | `ChatState < 'a >::end_turn() -> Result < () >` | ok |
| 255 | `ChatState.new_turn(role: Literal['system', 'user', 'assistant']) -> None` | `ChatState < 'a >::new_turn(& str) -> Result < () >` | ok |
| 258 | `ChatState.append(text: torch.Tensor, audio_out: torch.Tensor, modality_flag: torch.Tensor) -> ∅` | `ChatState < 'a >::append(& Tensor, & Tensor, & Tensor) -> Result < () >` | ok |

## `trainer.py` → `trainer.rs`  (5 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 21 | `Trainer.__init__(model_id: str, train_data: LFM2DataLoader \| None, val_data: LFM2DataLoader \| None, lr: float, betas: tuple[float, float], weight_decay: float, min_ratio: float, max_steps: int, warmup_steps: int, batch_size: int, dataloader_num_workers: int, logging_interval: int, save_interval: int, val_interval: int, output_dir: str) -> None` | `Trainer::new(& Path, TrainerConfig, & Device, Box < dyn DataIter >, Option < Box < dyn DataIter > >) -> Result < Self >` | ok |
| 132 | `Trainer.train() -> None` | `Trainer::train() -> Result < () >` | ok |
| 171 | `Trainer.train_step(batch: LFM2AudioModelInput) -> LFM2AudioModelOutput` | `Trainer::train_step(& LFM2AudioModelInput) -> Result < LFM2AudioModelOutput >` | ok |
| 185 | `Trainer.validate() -> None` | `Trainer::validate() -> Result < () >` | ok |
| 209 | `Trainer.log(model_output: LFM2AudioModelOutput) -> None` | `Trainer::log(& LFM2AudioModelOutput) -> Result < () >` | ok |

## `utils.py` → `utils.rs`  (4 fns, 0 missing)

| Py:line | Python signature → return | Rust signature → return | flag |
|--:|---|---|---|
| 15 | `mel2emb_len(l: T) -> T` | `mel2emb_len(i64) -> i64` | ok |
| 24 | `emb2mel_len(l: T) -> T` | `emb2mel_len(i64) -> i64` | ok |
| 32 | `module_exists(name: str) -> bool` | `module_exists(& str) -> bool` | ok |
| 41 | `get_model_dir(repo_id: str \| Path, revision: str \| None) -> Path` | `get_model_dir(& str, Option < & str >) -> std :: io :: Result < std :: path :: PathBuf >` | ok |
