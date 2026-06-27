<!-- topic: Moshi Conditioners (off-path) -->
# CN01 · ConditionProvider / Fuser
**Code:** `CN01` · **Source:** `moshi/conditioners/base.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
This file is the conditioning framework vendored from Kyutai's Moshi / Meta's AudioCraft: it defines how *external attributes* (a text genre string, a raw tensor) are embedded into dense vectors and *fused* into a generative LM's stream. It supplies three things — `BaseConditioner` (per-attribute embed + pad + project), `ConditionProvider` (collate a batch of `ConditionAttributes` and run every conditioner), and `ConditionFuser` (combine the resulting `(embedding, mask)` pairs by `sum` / `prepend` / `cross`). It is the conditioning backbone of the Moshi 7B multi-stream LM (`moshi_lm`), not of LFM2-Audio. **LFM2.5-Audio never instantiates any of it** — its inputs are scattered by modality flag in `model/lfm2_audio.py`, not conditioned/fused — so this whole module is reference-only (off-path) and has no Rust port.

## How it works
The module is three cooperating `nn.Module`s plus a `ConditionAttributes` data container.

**`BaseConditioner.__init__` (`base.py:105`).** Generic over a `Prepared` type. Holds an `output_proj` that is `nn.Linear(dim, output_dim, bias=output_bias)` when `force_linear` (default `True`) or `dim != output_dim`, else `nn.Identity` (`base.py:118`). There is an `assert not output_bias` guard so the linear is bias-free in the forced case. An optional learnt padding vector `learnt_padding` is a `Parameter` of shape `(1,1,output_dim)`, initialized `randn` then scaled in-place by `0.2` (`base.py:125-127`); `None` if `learn_padding=False`.

**`BaseConditioner.forward` (`base.py:151`) — the per-attribute pipeline:**
1. `cond, mask = self._get_condition(inputs)` — abstract; subclasses (`LUTConditioner`, `TensorConditioner`) produce `cond` of shape `[B,T,dim]` and a bool `mask` of `[B,T]`. The base raises `NotImplementedError` (`base.py:139,149`).
2. Empty-pad guard (`base.py:153-156`): if `T==0` and `pad_empty`, it rebuilds a zero `cond` of `[B,T,C]` and a zero bool mask. (Note: this branch keeps `T==0`; it is a degenerate guard, not a length-1 re-pad despite the docstring.)
3. **Projection:** `cond = self.output_proj(cond)` — a single dense projection `dim → output_dim`, no normalization, no activation.
4. **Masked padding blend (`base.py:160-164`):** `maskf = mask.float()[..., None]` upcasts the bool mask to a `[B,T,1]` float gate; then
   ```python
   cond = cond * maskf + self.learnt_padding * (1 - maskf)   # learnt pad
   cond = cond * maskf                                        # zero pad
   ```
   i.e. valid positions keep the projected embedding, padded positions are replaced by the (broadcast) learnt-padding vector or zeroed. Returns a `ConditionType(cond, mask)` NamedTuple (`base.py:25`). There is **no attention, no RoPE, no RMSNorm/LayerNorm, no convolution, no quantization** here — it is embed → linear → masked-select.

**`ConditionProvider` (`base.py:225`)** wraps a `nn.ModuleDict` of named conditioners and partitions them via `isinstance` into `text_conditions` (`_BaseTextConditioner`) and `tensor_conditions` (`_BaseTensorConditioner`) (`base.py:238-244`).
- `_collate_text` (`base.py:246`) transposes a list of `ConditionAttributes` into `{attr: [str|None, ...]}` via a `defaultdict(list)`.
- `_collate_tensors` (`base.py:273`) stacks per-attribute `TensorCondition`s with `TensorCondition.cat` (`base.py:46`), which right-pads each `[1,T,D]` tensor to `T=max` into a `[B,T,D]` zero buffer and copies the per-item bool masks — classic right-padding collation (`base.py:53-59`).
- `prepare` (`base.py:293`) collates, asserts the attribute keyset is a subset of the registered conditioners, raises if any conditioner got no input, then calls each conditioner's `prepare(batch)` (the "sync-point" stage: BPE tokenize + host→device transfer, separated so the GPU forward has no stalls). `forward` (`base.py:325`) then runs each conditioner module on its prepared batch and returns `{name: ConditionType(condition, mask)}`. `prepare_and_provide` (`base.py:343`) chains the two.

**`ConditionFuser` (`base.py:349`)** maps each condition name to one of `FUSING_METHODS = ["sum","prepend","cross"]` via `cond2fuse`, and at construction it actually *rejects* `prepend` (`base.py:379-381` raises unless the method is `sum` or `cross`) — so the live fuse paths are sum and cross.
- `get_sum` (`base.py:411`): for every `sum`-tagged condition, assert `cond.shape[1] == 1` (a single time step) and accumulate `sum = sum + cond`. This is a per-step **additive offset** broadcast over the whole sequence — used by the LM as `input_ = input_ + sum_condition` (`moshi/models/lm.py:392-393`).
- `get_cross` (`base.py:392`): concatenate all `cross`-tagged conditions along `dim=1` (time) to build the cross-attention key/value source. Optionally adds a sinusoidal positional embedding: `positions = arange(T).view(1,-1,1)`, `pos_emb = create_sin_embedding(positions, C).to(cross.dtype)`, `cross = cross + scale * pos_emb` (`base.py:402-408`). `create_sin_embedding` (`moshi/modules/transformer.py:130`) is the standard half-split table: `half_dim = dim//2`, `phase = positions / (max_period ** (arange(half_dim)/(half_dim-1)))` with `max_period=10000`, returning `cat([cos(phase), sin(phase)], dim=-1)` — i.e. a **concatenated (half-split) cos/sin layout, computed in f32 by default** then cast to the cross dtype.
- `get_prepend` (`base.py:423`): concatenates `prepend`-tagged conditions and, if any, folds in `get_sum`. Present in the API but unreachable through the constructor's guard.

**Dropout utilities** (`dropout_tensor`/`dropout_condition_`/`dropout_all_conditions`, `base.py:176-222`) zero out a condition's tensor+mask (the AudioCraft classifier-free-guidance nullification), used by the LM's CFG path, not by any embedding math here.

## Dtypes & shapes
| Stage | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `BaseConditioner._get_condition` (subclass) | prepared batch (token ids int64 / raw tensor) | `cond` model dtype `[B,T,dim]`, `mask` bool `[B,T]` |
| `output_proj` (Linear `dim→output_dim`) | model dtype `[B,T,dim]` | model dtype `[B,T,output_dim]` |
| masked blend (`base.py:160-164`) | `cond` model dtype `[B,T,output_dim]`, `mask` bool `[B,T]` → `maskf` f32 `[B,T,1]` | model dtype `[B,T,output_dim]` |
| `TensorCondition.cat` collate | list of `[1,Tᵢ,D]` (mask bool `[1,Tᵢ]`) | `[B,max T,D]`, mask bool `[B,max T]` |
| `ConditionFuser.get_sum` | each `[B,1,C]` model dtype | `[B,1,C]` model dtype (additive offset) |
| `ConditionFuser.get_cross` | each `[B,Tᵢ,C]` model dtype | `[B,ΣTᵢ,C]` model dtype (+ f32 sin pos-emb cast to cross dtype) |
| `create_sin_embedding` | `positions` long `[1,T,1]` | f32 (default) `[1,T,dim]`, then `.to(cross.dtype)` |

Internal promotions: the mask is upcast bool→f32 (`mask.float()`) for the blend; `create_sin_embedding` builds its table in **f32** then casts to the cross dtype. Moshi keeps `condition_provider` pinned to **float32** at the LM level (`moshi/models/lm.py:228` "We always keep the condition provider as float32"), and the LM casts `sum`/`cross` back to model dtype before use (`lm.py:393`, `lm.py:618-621`). No int64 token-id math, no u32 codes, no f64 here.

## Wiring
**Upstream (feeds this):** a batch of `ConditionAttributes` (text dict + `TensorCondition` dict) assembled by a Moshi dataset/runner; the concrete embedders are [moshi_cond_text](CN02-Text-Conditioner) (`LUTConditioner`) and [moshi_cond_tensors](CN03-Tensor-Conditioner) (`TensorConditioner`), both subclasses of `BaseConditioner` here. The sinusoidal helper comes from [moshi_transformer](MO03-Codec-Transformer) (`create_sin_embedding`). Loaders [moshi_loaders](MM02-Mimi-Loaders) (`get_conditioner_provider`/`get_condition_fuser`, `loaders.py:437-453`) build the `ConditionProvider`/`ConditionFuser` instances.

**Downstream (consumes this output):** [moshi_lm](MM03-Moshi-LM) is the only real consumer — `LMModel.__init__` stores `condition_provider`/`fuser` (`lm.py:104-105,229-234`); `LMModel.forward` calls `fuser.get_sum`/`fuser.get_cross` to produce `sum_condition` `[B,1,C]` and `cross_attention_src` `[B,ΣT,C]` (model dtype after cast), feeding the transformer as an additive input offset and cross-attention KV source (`lm.py:354-357,392-396`); `LMGen` does the same on the streaming path (`lm.py:616-621`). **No LFM2.5-Audio component consumes this** — neither [core_processor](CO01-Processor-ChatState) nor [model_lfm2_audio](MD01-LFM2AudioModel) nor [model_lfm2_backbone](MD01-LFM2AudioModel) imports the conditioners; LFM2-Audio fuses modalities by index-scatter, not by this provider/fuser.

## Python ↔ Rust
No Rust counterpart. The Rust port's `compare_symbols.py` core scope **excludes all of `moshi/`** (PYTHON_VS_RUST.md §4: "the vendored Python `liquid_audio/moshi/**` is reused as the `moshi` crate, not re-ported"). The `moshi` crate that `liquid-audio-rs` depends on is used **only for the Mimi codec** (`moshi::mimi`, ARCHAEOLOGY.md Q1) — its conditioner/LM modules are never loaded by this project. So there is no `ConditionProvider`/`ConditionFuser`/`BaseConditioner` symbol in `liquid-audio-rs/src/` (grep for "condition" returns nothing), and no deliberate divergence to record because the component is never on the port surface. This is the same off-path status PYTHON_VS_RUST.md §4 assigns to the whole `moshi/**` LM/conditioning stack, distinct from the on-path Mimi reuse in §2.3.

## Precision / gotchas
- **f32-pinned conditioning.** The provider is deliberately kept in float32 by its only consumer (`lm.py:228`); `sum`/`cross` are cast to the LM's compute dtype at the boundary (`lm.py:393`, `lm.py:618-621`). The sinusoidal pos-emb is also built in f32 then cast (`base.py:407`, `create_sin_embedding` default `dtype=torch.float32`) — this is the "extended precision until the boundary" pattern, here at the conditioning edge rather than the mel front-end.
- **`get_sum` requires length-1.** `assert cond.shape[1] == 1` (`base.py:416`) — a `sum` condition must be a single broadcastable step; a multi-step sum condition is a hard error.
- **`prepend` is dead.** Despite being in `FUSING_METHODS` and having a full `get_prepend`, the `ConditionFuser` constructor raises on any method other than `sum`/`cross` (`base.py:379-381`), so `has_prepend`/`get_prepend` are unreachable through normal construction.
- **Empty-pad branch is a no-op resize.** The `T==0` guard (`base.py:153-156`) rebuilds zero tensors that are still length-0, contrary to the docstring's "padded to have length 1"; downstream collation/`get_sum` is what enforces real lengths.
- **Sinusoidal layout is half-split, not interleaved.** `create_sin_embedding` concatenates `[cos, sin]` blocks (`transformer.py:154`) — different from the *interleaved* RoPE (`rope_i`) used in the LFM2 depthformer ([model_transformer](MD04-Depthformer)); do not conflate the two.
- **No norm/attention here.** Unlike most components in this codebase, `base.py` has no RMSNorm/LayerNorm order question, no `1/sqrt(d)` attention scale, no causal mask — the masked blend (`base.py:162`) is the only numerically interesting op, and it is exact (multiply + add of a bool-derived f32 gate). The classifier-free-guidance dropout (`base.py:176-222`) is the only thing that mutates condition values, and it zeroes them exactly.
