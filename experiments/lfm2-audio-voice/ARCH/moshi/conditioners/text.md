# moshi_cond_text
**Code:** `CN02` · **Source:** `moshi/conditioners/text.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
`LUTConditioner` is Moshi's lookup-table **text conditioner**: it maps a small set of categorical *attribute strings* (genre, key, speaker label, classifier-free-guidance flag, etc.) into a learned dense embedding `[B, T, output_dim]` plus a validity mask, so the Moshi 7B LM and the Moshi `TTSModel` can be steered by metadata. It is a `BaseConditioner` subclass driven by a `NoopTokenizer` (one index per whole string, not per word). **It is vendored Moshi machinery and is *not* part of LFM2.5-Audio:** the only importers are the off-path `moshi/models/lm.py` (via `ConditionProvider`/`ConditionFuser`) and `moshi/models/tts.py` (`text.py:26`). LFM2-Audio builds its own backbone + depthformer and never constructs a `ConditionProvider`, so nothing in `processor.py` / `model/lfm2_audio.py` reaches this code. It is documented here for inventory completeness; the Rust port deliberately omits it (reuse-the-`moshi`-crate, off-path; PYTHON_VS_RUST.md §4).

## How it works
Two-phase contract inherited from `BaseConditioner` (`base.py:93`): a sync-point `prepare()` (CPU tokenize → device transfer) and a tensor-only `forward()` (`base.py:151`). The phases exist so BPE/host work happens before GPU work to avoid a CUDA sync mid-step.

- **Tokenize — `NoopTokenizer.__call__` (`text.py:85`).** For each input string in the batch list:
  - `None` (attribute absent) → emit `pad_idx = n_bins` and length `0`.
  - present, no `possible_values` → `hash_trick(text, n_bins)` = `sha256(utf-8) mod n_bins` (`text.py:34-44`), a deterministic hash bucket. There is **no** collision handling — two distinct strings can alias to the same bin.
  - present, with `possible_values` → exact dict index (`text.py:98`), raising on an unknown value.
  Each present item has length `1` (a single "token" per attribute — this is a *global/categorical* conditioner, not a sequence). Result: `tokens = int tensor [B, 1]` (`text.py:101`, `.int()[:, None]`) and `mask = length_to_mask(lengths)` (`text.py:102`). `length_to_mask` (`text.py:18-31`) builds a bool `[B, Lmax]` via `arange(Lmax)[None,:] < lengths[:,None]`; with all lengths ∈ {0,1} and `final_length = max(maxlen, 1)`, the mask is `[B, 1]` — `True` for present, `False` for absent/padded.

- **prepare — `LUTConditioner.prepare` (`text.py:125`).** Tokenize on CPU, then move both tokens and mask to the embedding's device (`self.embed.weight.device`). Returns a `TokenizedText` NamedTuple. The `.to(device)` is double-applied (lines 128 and 129) — harmless idempotent transfer, no functional effect.

- **embed — `LUTConditioner._get_condition` (`text.py:131`).** A single `nn.Embedding(n_bins + 1, dim)` lookup (`text.py:118`; `+1` row reserved for `pad_idx`). `embeds = self.embed(tokens)` → `[B, 1, dim]`. Constructor scales the LUT once at init by `init_scale` (`text.py:119`, `.weight.data *= init_scale`). Returns `ConditionType(embeds, mask)`.

- **project + pad-fill — `BaseConditioner.forward` (`base.py:151-165`).** This is where the actual numerics finish (the subclass only supplies the raw LUT lookup):
  1. If `T == 0` and `pad_empty`, replace with a zero tensor + zero mask (defensive; the noop path always yields `T == 1`).
  2. `cond = self.output_proj(cond)` — a **biasless** `nn.Linear(dim → output_dim)` when `force_linear or dim != output_dim`, else `Identity` (`base.py:118-122`). `output_bias` is asserted `False` (`base.py:120`). This is the only matmul; there is **no normalization, no attention, no activation, no RoPE, no convolution** in this component.
  3. **Learned-padding blend (`base.py:160-164`):** `maskf = mask.float()[..., None]`; `cond = cond * maskf + learnt_padding * (1 - maskf)`. So *present* attributes pass through the projected embedding; *absent* ones (`mask == 0`) are overwritten by a learned per-feature vector `learnt_padding` (`[1,1,output_dim]`, init `randn * 0.2`, `base.py:125-127`). With `learn_padding=False`, absent positions are zeroed instead.

There is no streaming state, no causal mask, no sampling — a single embedding gather + linear projection + masked blend, applied once per attribute per forward.

**Fusion (downstream of this module, in `base.py`).** `ConditionProvider.forward` (`base.py:325`) runs each conditioner and returns `{name: ConditionType}`. `ConditionFuser` (`base.py:349`) then combines them by method: `prepend` (concat onto the transformer sequence, `get_prepend` `base.py:423`), `sum` (a shared per-step offset, asserts `cond.shape[1]==1`, `get_sum` `base.py:411`), or `cross` (concat into the cross-attention KV, `get_cross` `base.py:392`, optionally adding a sinusoidal pos-emb via `create_sin_embedding`). Only `sum`/`cross` are accepted at construction time (`base.py:379-381`).

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `NoopTokenizer.__call__` | `list[str | None]` (len B) | `tokens int [B,1]`, `mask bool [B,1]` |
| `prepare` | same list | `TokenizedText(tokens int64→on-device, mask bool)` |
| `_get_condition` (LUT) | `tokens int [B,1]` | `embeds (model dtype, bf16/f32) [B,1,dim]`, `mask bool [B,1]` |
| `output_proj` (Linear, biasless) | `[B,1,dim]` | `[B,1,output_dim]` |
| learned-pad blend | `cond [B,1,output_dim]`, `mask bool [B,1]` | `cond [B,1,output_dim]` |
| `ConditionType` (final) | — | `condition (model dtype) [B,1,output_dim]`, `mask bool [B,1]` |

Dtype notes: token ids are `int` (`text.py:101` `.int()` — int32, *not* int64; the LUT `nn.Embedding` accepts either). The embedding weight and the `output_proj` weight follow the module dtype — **bf16** under Moshi's CUDA default, **f32** on CPU. The mask→`float()` blend (`base.py:160`) promotes the `[B,1]` mask to f32 for the multiply, broadcasting against the (bf16/f32) cond — a mixed-dtype multiply that follows torch promotion. There is **no** f32/f64 upcast for norm/softmax here (this module has neither). `hash_trick` is a pure-Python `int` (256-bit sha256 → `mod n_bins`), never a tensor.

## Wiring
Off-path; none of these edges touch the LFM2-Audio tensor flow.

- **Upstream:** `ConditionProvider.prepare` (`base.py:293`) collates per-attribute string batches from `ConditionAttributes.text` dicts and calls `LUTConditioner.prepare`; in the Moshi TTS path the attributes come from [moshi_tts](../models/tts.md) script/speaker metadata. Edge: `list[str|None]` (len B) → this module. See [moshi_cond_base](base.md).
- **Downstream:** the projected `ConditionType [B,1,output_dim]` (model dtype) is consumed by [moshi_cond_base](base.md)'s `ConditionFuser` (`get_sum`/`get_cross`/`get_prepend`), which injects it into the [moshi_lm](../models/lm.md) Moshi-7B transformer (sum offset, prepended token, or cross-attention KV) — *not* into the LFM2-Audio backbone. `moshi/models/tts.py` (`text.py:26`) also reads `LUTConditioner.tokenizer.possible_values` directly to validate CFG conditioning values ([moshi_tts](../models/tts.md), `tts.py:441-443`).

## Python ↔ Rust
No Rust counterpart exists. `liquid-audio-rs/src/` contains **zero** references to `LUTConditioner`, `NoopTokenizer`, `hash_trick`, `length_to_mask`, `TextConditioner`, or `conditioners/text` (grep-verified). This is a **deliberate omission**, not a gap:

- The whole vendored `liquid_audio/moshi/**` tree is "reused as the `moshi` crate, not re-ported" and is excluded from the port's `core` parity scope by design (PYTHON_VS_RUST.md §4; PORT_STATUS.md table row `moshi/* → ♻ reuse the moshi crate`).
- The conditioning subsystem (`ConditionProvider`/`ConditionFuser`/`LUTConditioner`) is only wired into the Moshi 7B LM and `TTSModel`, which are themselves off-path (`moshi_lm`, `moshi_tts` are "reference only / off-path"). LFM2-Audio's Rust pipeline ([model_lfm2_audio](../../model/lfm2_audio.md)) never instantiates a conditioner.
- Even the Mimi codec path actually used by the Rust port (Kyutai's `moshi::mimi` crate) does not pull in the conditioners module.

Symbol map: `LUTConditioner` / `NoopTokenizer` / `BaseConditioner.forward` / `ConditionFuser` → *(none)*. If ever needed, the natural candle shape would be `candle_nn::Embedding` (LUT) + biasless `candle_nn::Linear` (output_proj) + a masked `where`-blend for `learnt_padding`, mirroring the candle-ops-over-CUDA-kernels strategy used elsewhere (PYTHON_VS_RUST.md §2.2).

## Precision / gotchas
- **Not on the inference path** — the single most important fact: changing this file cannot affect LFM2-Audio output. It steers only the Moshi-7B LM / TTS reference models.
- **`hash_trick` collisions (`text.py:34`).** With no `possible_values`, distinct strings can map to the same of `n_bins` buckets and share an embedding row. Silent by design — fine for the "robust hashing of free-form tags" use case, surprising if you expected injective ids.
- **`pad_idx == n_bins` (`text.py:78`)** is the extra `+1` embedding row (`text.py:118`). Absent attributes (`None`) tokenize to this index *and* get `mask=0`, so the masked blend (`base.py:162`) overwrites whatever that row produced with `learnt_padding` — the pad row's embedding is effectively dead weight on the forward (it matters only if `learn_padding=False`, where absent → zero).
- **`output_proj` is biasless** (asserted, `base.py:120`); a mask of all-zeros yields a clean `learnt_padding`-only (or zero) output with no bias leakage.
- **Mask dtype crossing (`base.py:160`).** `mask.float()` × bf16 `cond` is a mixed-precision multiply; under torch promotion it lands in the wider dtype. Order is *multiply present, add learned-pad* — there is no normalize/cast subtlety like the RMSNorm-bf16 ordering elsewhere in the model (this module has no norm).
- **`.int()` not `.int64()` (`text.py:101`).** Token ids are int32; the embedding lookup is dtype-agnostic so this is harmless, but it differs from the int64 token ids used on the real LFM2-Audio text path ([core_processor](../../processor.md)).
- **EOAudio / special tokens are irrelevant here** — this conditioner has no audio codes and no autoregressive vocabulary; the `2048=EOAudio` and `65536` text-vocab facts apply to the on-path heads, not to this LUT.
