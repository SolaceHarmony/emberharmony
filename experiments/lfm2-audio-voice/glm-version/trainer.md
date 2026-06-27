# core_trainer (Rust port)
**Source:** `liquid-audio-rs/src/trainer.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/trainer.py` · **On the LFM2-Audio inference path:** no

> Companion to [`ARCH/trainer.md`](../ARCH/trainer.md). The original is already
> notably Rust-aware (it references `trainer.rs` line numbers throughout); this
> version is Rust-first and consolidates the Rust-specific divergences.

## Role
`Trainer` (`trainer.rs` struct, `:240`-ish) is the supervised fine-tuning driver
for `LFM2AudioModel` in the Rust port. It owns the optimizer, LR schedule,
dataloaders, and the step/epoch/checkpoint loop, and runs the model's *training*
`forward(batch) -> LFM2AudioModelOutput`. It exists purely for training; it is
**not** on the realtime inference/generation path (that is
`LFM2AudioModel::generate_*` driven by the processor). Critically, the trainer
holds **no loss of its own** — the cross-entropy + per-codebook weighting lives
in the model (`lfm2_audio.rs::forward`) — so `train_step` and `validate` are
thin wrappers around `self.model.forward(batch)`.

## How it works (Rust)

**Construction (`Trainer::new` / `Trainer::with_model`, `:262`/`:279`).** Wires
four things:
1. **Model** (`:269`): `from_pretrained_trainable(model_dir, cfg.dtype, device)`
  loads the model + its `VarMap` (the trainable `Var` set). `cfg.model_id`
  records the Python `model_id` for parity.
2. **Optimizer** (`:287-296`): `candle_nn::AdamW` with `ParamsAdamW { lr, beta1,
  beta2, eps: 1e-8, weight_decay }` over `varmap.all_vars()`. AdamW filters to
  float `Var`s internally, so integer buffers (e.g. `codebook_offsets`) are
  skipped. **No `fused=True`** — candle's AdamW is an un-fused kernel sequence;
  the math is identical, only the kernel fusion differs (§2.6).
3. **LR schedule** (`Trainer::lr_at`, `:332`): the identical piecewise schedule
   computed directly per step — `LinearLR` warmup (`start_factor=1e-8 → 1.0`
   over `warmup_steps`) chained into `CosineAnnealingLR`
   (`T_max = max(1, max_steps - warmup_steps)`, `eta_min = lr * min_ratio`).
   Applied via `set_learning_rate(lr_at(step))` **before** the optimizer step
   (`:437`) to match torch's post-step `scheduler.step()` ordering.
4. **Dataloaders** (`:266-267`): `Box<dyn DataIter>` — the `DataIter` trait
   (`:47`) is the Rust analog of `torch.utils.data.DataLoader`. `LoaderDataIter`
   (`:98`) wraps `crate::data::dataloader::LFM2DataLoader`, batching consecutive
   rows through `lfm2_collator`. `new_shuffled` re-permutes rows each epoch via
   splitmix64 Fisher–Yates (`seed` makes the run reproducible). `VecDataIter`
   (`:61`) is the minimal in-memory loader.

**Train loop (`train`, the `while self.step < self.max_steps` loop).** Pulls
`next_batch()`; on `Ok(None)` it bumps `self.epoch`, calls `reset()`, and pulls
again. Each iteration: `train_step(batch)` → `self.step += 1` → log. Then
interval-gated side effects: `save_state` every `save_interval`;
`model.eval()`→`validate()`→`model.train()` every `val_interval`. On exit:
`save_model` to `{output_dir}/model.safetensors`.

**`train_step`.** `set_learning_rate(lr_at(step))` → `optimizer.backward_step(&out.loss)`
(`:439`) — candle builds a fresh `GradStore` each backward, so `zero_grad` is
implicit (no `optimizer.zero_grad()` call needed). The model's `forward(batch)`
returns `LFM2AudioModelOutput { loss, … }`; `out.loss` is the f32 scalar the
optimizer steps on.

**The loss is the model's, not the trainer's.** `out.loss` is computed inside
`LFM2AudioModel::forward` (`lfm2_audio.rs:575`): per-token
`cross_entropy_none` separately on text logits and depthformer audio logits;
audio loss is reshaped `(L·C) → (L, C)` over `C=codebooks` and weighted by
`audio_loss_weights`; final scalar is the modality-weighted token-mean. The
trainer never sees codebooks or modality multipliers — it only reads `out.loss`
(and `out.audio_loss`/`out.text_loss` for logging). The earlier Rust had a
duplicate trainer-side `Trainer::forward`/`LossConfig`/`ce_none`; these were
**removed** so both `train_step` and `validate` route through
`LFM2AudioModel::forward` and cannot diverge (§2.6).

**`validate`** (under no-grad): accumulates `loss_sum`/`loss_count` across the
whole val loader (same `model.forward(batch)`), then `Trainer::reduce`
(single-process identity) and `mean = sum / count.clamp_min(1)`. The Rust
upcasts the val/log loss to f32 explicitly before reducing (`:468`/`:489`).

**`Trainer::reduce`** (`:355-357`): single-process identity (the all-reduce of a
1-process group) — faithful, not a stub.

**Checkpointing:** `VarMap::save` → `state_step_{N}.safetensors` (state) and
`model.safetensors` (final model). Single file; no sharding analog, tensors
identical.

There is no normalization scheme, attention, RoPE, conv, or sampling logic in
this component — those live in the model/codec it drives.

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| Batch | `LFM2AudioModelInput`: text I64 `(B,L)`, audio_in f32 mel, audio_in_lens I64 `(B,)`, audio_out I64 codes `(B,L,C)`, modality_flag I64 `(B,L)`, supervision_mask U8 `(B,L)` | same |
| `model.forward(batch)` | batch; weights bf16 (Metal) / f32 (CPU) | `LFM2AudioModelOutput { loss, audio_loss, text_loss: f32 scalars }` |
| `backward_step(&out.loss)` | f32 scalar loss | grads on the `Var`s |
| `lr_at`/scheduler | step `usize` | LR `f64` |
| `reduce(loss, "mean"/"sum")` | f32 scalar | f32 scalar (identity, single-proc) |
| `VarMap::save` | the `Var` set | safetensors on disk (bf16/f32) |

Internal promotions: cross-entropy and all loss reductions run in **f32** (the
model upcasts logits/loss off the bf16 activations); token ids stay **I64**;
audio codes are **I64** (0..2048, 2048=EOAudio); LR math is **f64**. Weights are
bf16 on disk/Metal; f32 on CPU (no CPU bf16 matmul).

## Wiring (Rust)
**Upstream:**
- `data/dataloader.rs` — `LFM2DataLoader` + `lfm2_collator` produce the collated
  `LFM2AudioModelInput` batches. See
  [`glm-version/data/dataloader.md`](data/dataloader.md).
- `data/types.rs` — the `LFM2AudioModelInput`/`LFM2AudioModelOutput` structs
  that cross every trainer boundary. See
  [`glm-version/data/types.md`](data/types.md).

**Driven (the model the trainer optimizes):**
- `model/lfm2_audio.rs` — `train_step`/`validate` call its `forward(batch)`; it
  returns `out.loss` (f32 scalar) and holds the `audio_loss_weights` buffer +
  per-modality multipliers. See
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).

**Downstream (consumers of the trainer's *output* — produced weights):**
- `model/lfm2_audio.rs` — the `VarMap::save`'d safetensors are what
  `LFM2AudioModel::from_pretrained` later loads for inference.

## Python ↔ Rust — where the port differs

| Python (`trainer.py`) | Rust (`trainer.rs`) | Difference | Why |
|---|---|---|---|
| `Trainer.__init__` | `Trainer::new` / `Trainer::with_model` (`:262`/`:279`) | **`train_data is None ⇒ ValueError` enforced by types** | `train_loader: Box<dyn DataIter>` is non-`Option`; Rust's type system enforces the Python runtime check. |
| `torch.optim.AdamW(..., fused=True)` | `candle_nn::AdamW` + `ParamsAdamW` (`:287-296`) | **deliberate: no fused kernel** | §2.6. Same math; candle's AdamW is an un-fused kernel sequence. |
| `LinearLR ⇒ CosineAnnealingLR (SequentialLR)` | `Trainer::lr_at` (`:332-350`) — the schedule computed directly per step | **deliberate: direct formula** | no `SequentialLR` in candle; the piecewise schedule is computed inline. Applied via `set_learning_rate(lr_at(step))` *before* the optimizer step (`:437`) to match torch's post-step `scheduler.step()` ordering. |
| `accelerator.autocast()` (bf16) | implicit — weights carry the load dtype, loss math upcast to f32 | **deliberate: no cast ctx** | candle has no autocast; dtype is explicit at load. |
| `accelerator.backward(loss)` + `optimizer.step()` | `optimizer.backward_step(&out.loss)` (`:439`) | **deliberate: combined** | candle builds a fresh `GradStore` each backward, so `zero_grad` is implicit (no `optimizer.zero_grad()` call). |
| `accelerator.reduce(t, "mean"/"sum")` | `Trainer::reduce` (`:355-357`) — identity | **deliberate: single-process** | the all-reduce of a 1-process group is identity. Faithful, not a stub. |
| `accelerator.save_state` | `VarMap::save → state_step_{N}.safetensors` (`:502-506`) | **deliberate: safetensors** | no `accelerate` project config; a single safetensors file. |
| `accelerator.save_model(max_shard_size="5GB")` | `VarMap::save → model.safetensors` (`:512-516`) | **deliberate: single file** | no sharding analog; tensors identical. |
| `model.parameters()` | `varmap.all_vars()` (`:296`) | identical | AdamW filters to float `Var`s, skipping int buffers. |
| `DataLoader(shuffle=True, collate_fn=lfm2_collator)` | `LoaderDataIter` over `LFM2DataLoader` (`:98-177`) | **deliberate: splitmix64 Fisher–Yates** | a different PRNG can't reproduce torch's permutation anyway, so a seeded reproducible shuffle is equally faithful. `num_workers`/`pin_memory`/`prefetch_factor` are throughput-only kwargs with no single-process candle referent. |
| `train_data is None ⇒ ValueError` | `train_loader: Box<dyn DataIter>` non-`Option` | **deliberate: type-enforced** | Rust's type system enforces the Python runtime check. |
| `DataIter` (Python iterator protocol) | `DataIter` trait (`:47`) with `next_batch() -> Result<Option<…>>` + `reset()` | **deliberate: trait** | `Ok(None)` = epoch end; `Err` = load/collate failure (must surface, not be swallowed as epoch end). Unit-tested at `:539`. |
| device/dtype hardcoded `cuda`/`bf16` | `TrainerConfig.dtype` defaults to `F32` | **deliberate** | §2.1. CPU has no bf16 matmul in candle; bf16 only on CUDA/Metal. |

**Deliberate divergences** (PYTHON_VS_RUST §2.6): the earlier Rust carried a
duplicate trainer-side `Trainer::forward`/`LossConfig`/`ce_none`; these were
**removed** so both `train_step` and `validate` route through
`LFM2AudioModel::forward` and cannot diverge from the Python (which also has no
trainer-side loss). Loaders are stored on `self` to match
`self.train_loader`/`val_loader`. Single-process `reduce` is identity.

## Precision / gotchas (Rust-specific)
- **The loss lives in the model, not the trainer.** Any per-codebook weighting,
  modality multiplier, or `ignore_index=-100` masking question must be answered
  at [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md); the trainer only
  sums/means the returned f32 scalar. The Rust upcasts the val/log loss to f32
  explicitly (`to_dtype(F32)`, `:468`/`:489`) before reducing.
- **Scheduler-after-optimizer ordering is load-bearing.** Step 0 runs at the
  LinearLR `1e-8` floor, not at peak `lr`; the Rust reproduces this by
  `set_learning_rate(lr_at(step))` *before* the in-place AdamW step (`:437`) —
  same value torch's `SequentialLR` would hold after the post-step
  `scheduler.step()`. Off-by-one here would shift the entire warmup curve by one
  step.
- **`backward_step` builds a fresh `GradStore`.** candle creates a new
  `GradStore` each backward, so `zero_grad` is implicit — there's no
  `optimizer.zero_grad()` call. Don't add one; it's a no-op that would only hide
  the implicit-clear behavior.
- **Cosine floor:** `eta_min = lr * min_ratio` (default `3e-6`), reached at
  `max_steps`; `T_max = max(1, max_steps - warmup_steps)` guards
  `warmup_steps == max_steps` (`:345`).
- **`DataIter` error vs exhaustion (`:53`, `:163-168`, unit-tested `:539`):**
  a row-load/collate failure must surface as `Err`, **not** `Ok(None)` —
  `None` is treated as epoch end, so swallowing the error would silently
  shorten training or cut validation short. Python's `DataLoader` raises the
  exception; the Rust `?`-propagates it.
- **`find_unused_parameters=True` (Python DDP) has no Rust single-proc
  analog.** The modality-scatter leaves some params gradient-less on a given
  batch; in single-process candle this is fine (no DDP sync). If the Rust
  trainer ever goes multi-device, this becomes relevant.
- **bf16 numerics:** AdamW master-weight copies are managed by `accelerate`
  autocast in Python; the cross-library f32 floor (~1e-6, §1.4) and the bf16
  RMSNorm multiply-order subtlety (§2.4) belong to the model/codec, not this
  loop — the trainer's only float op is the f32 loss reduction.
- **`TrainerConfig.dtype` defaults to `F32`.** CPU has no bf16 matmul in
  candle; bf16 only on CUDA/Metal. The Python hard-codes `dtype=torch.bfloat16`
  — the Rust port is device-agnostic (§2.1).
- **`VarMap::save` is the checkpoint.** No `accelerate` project config; a single
  safetensors file per save. `state_step_{N}.safetensors` for state,
  `model.safetensors` for the final model.

## Cross-references
- [`ARCH/trainer.md`](../ARCH/trainer.md) — Python original (already
  Rust-aware).
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.6 (trainer —
  `accelerate`/torch → candle, loss on the model, de-duplicated).
- `liquid-audio-rs/src/data/dataloader.rs` — `LFM2DataLoader` + `lfm2_collator`.
- `liquid-audio-rs/src/loader.rs` — `from_pretrained_trainable` (the trainable
  load path).