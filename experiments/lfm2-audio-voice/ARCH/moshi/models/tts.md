# moshi_tts
**Code:** `MM05` · **Source:** `moshi/models/tts.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
`TTSModel` is Kyutai's Delayed-Streams-Modeling (DSM) text-to-speech wrapper around the **Moshi multi-stream LM** ([`moshi/models/lm.py`](lm.md)), the **Mimi** codec ([`moshi/models/compression.py`](compression.md)) and a SentencePiece tokenizer. It exists to drive *script → padded-aligned text + audio codes* generation: the model is fed pure text words and itself signals (via a `new_word` special token) when the next acoustic frame should start the next word; a `StateMachine` pops words off a queue in response and force-feeds their tokens. This is **vendored Moshi code, reused as the Kyutai `moshi` crate** — it is a *different* model from LFM2-Audio (which has its own backbone + depthformer and never tokenizes a "script" nor uses acoustic delays), so it is **off the LFM2-Audio inference path** and has **no Rust port in `liquid-audio-rs/src/`** (the `core` parity scope excludes all of `moshi/**`; see PYTHON_VS_RUST.md §4).

## How it works
The file is a **state machine + delay bookkeeping** layer; the actual transformer math lives in `lm.py`/`compression.py`. The novel mechanism is the alignment co-generation and acoustic-delay handling.

**Special tokens (`TokenIds`, tts.py:34-54).** `card = text_card + 1` (input text cardinality including the initial token, used to multiplex two text streams). `new_word=0`, `pad=3`, `main=1`/`other=2` (speaker-turn markers), `zero=-1` (embeds to exactly the 0 vector), `ungenerated=-2` (a value not yet produced, used as a fill sentinel in delayed prefixes).

**Script → entries (`script_to_entries`, tts.py:252-314).** Per turn, text is cleaned (`’→'`, `:→space`, strip `()`), split on whitespace and on a single SSML tag `<break time="Xs"/>`. Each word is SentencePiece-encoded to `tokens`; the first content word of a turn gets a speaker marker prepended when `multi_speaker` and the speaker changed (`idx % 2`). `<break>` becomes a tokenless `Entry` whose `padding = round(seconds * frame_rate)` (frame_rate = Mimi's 12.5 Hz), i.e. it just forces that many pad steps. `padding_between` adds `max(0, padding_between + len(tokens) - 1)` forced pads after each word to slow articulation.

**Acoustic delay (`_delayed`, tts.py:112-118).** Given codes `[K, T]` and per-codebook `delays`, builds `[K, T + max(delays)]` filled with `fill_value`, then writes `out[k, delay : delay+T] = codes[k]`. This is the RVQ acoustic-delay pattern (semantic codebook leads, acoustic codebooks lag) that Moshi uses so each step predicts only freshly-revealed codes. `delay_steps = round(audio_delay * mimi.frame_rate)` (tts.py:404) is the *additional* text→audio delay layered on top of the per-codebook delays.

**The step loop (`generate`, tts.py:486-618).** Construction sets `self.lm.dep_q = self.n_q` for non-multistream so the depformer emits exactly `n_q` audio codebooks. An `LMGen` (lm.py) is built with three hooks; the loop runs up to `max_gen_length` steps, each calling `lm_gen.step(input_tokens, depformer_replace_tokens=…)`. `input_tokens` is `[B, missing, 1]` of `zero`, where `missing = n_q - dep_q` (the forced-input audio codebooks the depformer does *not* generate). For the first `delay_steps` steps the audio stream is still all-`zero`, so `depformer_replace_tokens = no_depformer_tokens` short-circuits the depformer entirely (tts.py:607). The loop terminates once every batch item has `end_step` set and `offset >= max(end_step) + delay_steps + final_padding` (tts.py:594-597). Each non-`None` returned `frame` (`[B, 1+Q, 1]` long, acoustic delay already corrected by `LMGen`) is cloned into `frames` and optionally passed to `on_frame`.

**The state machine (`StateMachine.process`, tts.py:157-249) — the core.** Called from the `on_text_hook` once the LM has *sampled* a text token but before the depformer runs. Decision order:
1. Sampled token is coerced to `pad` unless it is exactly `new_word` or `pad` (tts.py:171-172).
2. **Override to PAD** if `state.queued` still has word tokens to feed, *or* `forced_padding > 0` (tts.py:174-179).
3. **Override to NEW_WORD** if `remaining_padding <= 0` (we have exhausted the max run of pads, tts.py:180-182) — this bounds silence between words.
4. On **NEW_WORD**: pop the next `Entry`; record `consumption_times`/`transcript`; `queued.extend(entry.tokens)`; reset `remaining_padding = max_padding`; set `forced_padding = entry.padding`. A tokenless (break) entry degrades to `pad`. If `entries` is empty → emit `pad` and latch `end_step = step` (the generation-end signal, tts.py:202-208).
5. On **PAD**: decrement both pad counters (floored at 0); the actual `output` token is `queued.popleft()` if word tokens are pending, else `pad` (tts.py:211-221).

**Second-stream lookahead multiplexing (tts.py:229-246).** When `second_stream_ahead > 0` the model has two muxed text streams. The lookahead word's tokens are queued separately (`lookahead_queued`, via `get_tokens_ahead`). The two per-step tokens are combined by a **cartesian-product encode**: `output = (second + 1) * card + output` (the `+1` lets `second = -1` mean an all-zeros embedding). lm.py's `EmbeddingFactory(demux_second_stream=True)` de-multiplexes this back (`text_card + 1` = `card`).

**The three `LMGen` hooks (tts.py:543-573).**
- `on_text_logits_hook`: adds `padding_bonus` to the `pad` logit (positive ⇒ slower speech).
- `on_audio_hook`: forces audio codebook `q` to `zero` while `offset < delays[q+audio_offset] + delay_steps` (the still-delayed region), then overlays any `audio_prefix` codes via `torch.where(mask, audio_codes, audio_tokens)` where `mask = (audio_codes != ungenerated)`.
- `on_text_hook`: for each batch item, either pops a forced `text_prefix` token or runs `machine.process(...)`, writing the result back into `text_tokens` in place; logs `(sampled, fed)`.

**Prefix / voice conditioning.** `get_prefix` (tts.py:672-678) Mimi-encodes a wav (`mimi.encode(...)[0,:,:-2]`, dropping the last 2 frames), prepends a `zero` text row. `make_condition_attributes` (tts.py:629-670) packs up to `max_speakers` voice embeddings (loaded from safetensors `speaker_wavs`, transposed and view-flattened) into a `TensorCondition`, plus a `control='ok'` text condition and an optional discrete `cfg` value. CFG (`cfg_coef != 1.0`) appends null-dropout conditions (`_make_null`/`dropout_all_conditions`) so the batch carries both conditioned and null branches.

There is **no normalization, attention, RoPE, convolution, or RVQ math in this file** — all of that is delegated to `lm.py` (Moshi 7B backbone + depformer), `compression.py` (Mimi), and the quantizers. This component is pure Python control flow over those.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `script_to_entries` | `list[str]` (script turns) | `list[Entry]` (`tokens: list[int]`, `padding: int`) |
| `_delayed` | `codes [K,T]` int64 | `[K, T+max(delays)]` int64, `fill_value`-padded |
| `LMGen.step` input | `input_tokens [B, missing, 1]` int64 (all `zero`) | `frame [B, 1+Q, 1]` int64 (delay-corrected) or `None` |
| `on_text_hook` rewrite | sampled `text_tokens [B]` int64 | fed text tokens `[B]` int64 (machine output) |
| second-stream mux | `(second, output)` ints | `(second+1)*card + output` int64 |
| `get_prefix` | wav f32 @ `mimi.sample_rate` (24 kHz) | prefix `[1+K_audio, T]` int64 codes (+ `zero` text row) |
| `make_condition_attributes` | voice safetensors `speaker_wavs [1, S, D]` f32 | `TensorCondition` (`[1, max_speakers*S, D]` f32 + bool mask) |
| `generate` result | — | `TTSResult.frames: list[[B,1+Q,1]]` int64 |

`temp`/`cfg_coef`/`padding_bonus` are f32 scalars. Token ids throughout are **int64** (`torch.long`). Model/codec weights are bf16 (Python default `dtype=torch.bfloat16`, `device='cpu'` overridable to cuda); no f32-upcast norm/softmax happens *in this file* (those live in `lm.py`/`compression.py`). Mimi output waveform (downstream of `frames`) is f32 @ 24 kHz.

## Wiring
**Upstream (feeds this):**
- A `loaders.CheckpointInfo` → `TTSModel.from_checkpoint_info` builds the Moshi `LMModel`, `MimiModel` and tokenizer. See [moshi_loaders](loaders.md) — provides `get_mimi`/`get_moshi`/`get_text_tokenizer`.
- A user **script** (`list[str]`) and **voice safetensors** (`speaker_wavs` embeddings) are the data inputs.

**Internal calls (consumed by this, not "downstream" of its tensor output):**
- [moshi_lm](lm.md) — `LMGen.step` does the actual transformer forward; this file feeds it `input_tokens [B,missing,1]` int64 and reads back `frame [B,1+Q,1]` int64. The text-stream mux (`(second+1)*card+output`) is de-muxed by lm.py's `EmbeddingFactory`.
- [moshi_compression](compression.md) — `mimi.encode` (prefix path, wav f32 → codes int) and `mimi.decode(frame[:,1:,:])` (in `warmup`, codes int → waveform f32 @ 24 kHz).
- [moshi_cond_text](../conditioners/text.md) / [moshi_cond_tensors](../conditioners/tensors.md) — `LUTConditioner` (cfg) and `TensorCondition` (`speaker_wavs`) consumed via `lm.condition_provider.prepare_and_provide`.

**Downstream (consumes this component's output):**
- `TTSResult.frames` (`list[[B,1+Q,1]]` int64) → [moshi_compression](compression.md) `MimiModel.decode` to produce the f32 @ 24 kHz waveform (the standalone `moshi/run_tts.py` driver does this). On the LFM2-Audio path nothing consumes this — it is a self-contained Moshi-TTS subsystem.

## Python ↔ Rust
**No Rust counterpart.** `moshi/models/tts.py` is part of the vendored `liquid_audio/moshi/**`, which `liquid-audio-rs` **reuses as the Kyutai `moshi` crate** rather than re-porting (PYTHON_VS_RUST.md §2.3, §4; PORT_STATUS.md line 68: "♻ reuse the `moshi` crate"). `compare_symbols.py --scope core` deliberately excludes `moshi/**`, so there is no `TTSModel`/`StateMachine`/`script_to_entries`/`_delayed` symbol in `src/`. The only Moshi pieces actually wired into LFM2-Audio are the **Mimi codec** (`moshi::mimi`, chosen because its `rvq_first`/`rvq_rest` weight names match the checkpoint) and the `LogitsProcessor` sampler — *not* this TTS state machine. If `liquid-audio-rs` ever needs DSM-TTS it would call the upstream `moshi` crate's equivalent, which mirrors this file 1:1.

## Precision / gotchas
- **Off-path, different model.** Do not conflate Moshi's `delay_steps`/acoustic-delay/`new_word`-driven alignment with LFM2-Audio generation — LFM2-Audio has no script tokenization, no acoustic delay, and no `StateMachine`. Treating this file as part of the LFM2 path is the main mis-read to avoid.
- **`EOAudio` is *not* here.** The `2048` end-of-audio sentinel belongs to the LFM2-Audio depthformer head ([model_lfm2_audio](../../model/lfm2_audio.md)); this Moshi file ends generation via `state.end_step` (model sampled `new_word` with empty `entries`) plus the `delay_steps + final_padding` drain, not via an EOAudio code.
- **`zero = -1` vs `ungenerated = -2`.** `zero` is a real embeddable sentinel (maps to the all-zeros embedding in `lm.py`); `ungenerated = -2` is only a *fill* marker inside delayed prefix tensors and is masked out (`audio_codes != ungenerated`) before being written — it must never reach the embedding table.
- **Pad-counter coupling.** `remaining_padding` (how many pads are *allowed* in a row, bounds silence) and `forced_padding` (how many pads are *required*, from `entry.padding`/breaks) are independent; both decrement on a pad step but only `remaining_padding` is reset (`= max_padding`) on a new word. An off-by-one here changes word timing, not correctness of codes.
- **Second-stream mux is a cartesian product, not addition of logits.** `(second+1)*card + output` is reversible only because `0 ≤ output < card`; the `+1` reserves `second = -1 ⇒ all-zeros`. Mis-ordering the factors silently corrupts the second stream.
- **CFG batch doubling.** When `cfg_coef != 1.0` the conditioned + null branches are concatenated, doubling the effective batch through `LMGen`; downstream slicing must account for it. Models trained with CFG *distillation* must instead pass `cfg` as a discrete conditioning value (raises otherwise, tts.py:464-468/509-513).
- **`get_prefix` drops the last 2 Mimi frames** (`[..., :-2]`) — a deliberate trim, not a bug.
