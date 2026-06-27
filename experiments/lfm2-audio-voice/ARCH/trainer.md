# core_trainer
**Code:** `CO04` · **Source:** `trainer.py` · **Rust:** `trainer.rs` · **On the LFM2-Audio inference path:** no

## Role
`Trainer` is the supervised fine-tuning driver for `LFM2AudioModel`. It owns the optimizer, LR schedule, dataloaders, and the step/epoch/checkpoint loop, and runs the model's *training* `forward(batch) -> LFM2AudioModelOutput` under HuggingFace `accelerate` (bf16 mixed precision, fused AdamW, distributed-aware reductions). It exists purely for training; it is **not** on the realtime inference/generation path (that is `LFM2AudioModel.generate_*` driven by the processor/demo). Critically, the trainer holds **no loss of its own** — the cross-entropy + per-codebook weighting lives in the model — so `train_step` and `validate` are thin wrappers around `self.model(batch)`.

## How it works

**Construction (`trainer.py:21-130`).** `__init__` wires four things:
1. **Accelerator (`:47-58`):** `mixed_precision="bf16"`, `DistributedDataParallelKwargs(find_unused_parameters=True)` (the modality-scatter means some params get no grad on a given batch, so DDP must tolerate unused params), `dispatch_batches=False` (each process pulls its own batches rather than broadcasting from rank 0), and `ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=30)` (rolling 30-checkpoint window under `output_dir`).
2. **Model (`:61-65`):** `LFM2AudioModel.from_pretrained(model_id, device=accelerator.device, dtype=torch.bfloat16)` — weights load bf16.
3. **Optimizer + schedule (`:68-91`):** `AdamW(lr, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1, fused=True)` over `model.parameters()`. The schedule is a `SequentialLR` of two phases chained at `milestones=[warmup_steps]`:
   - `LinearLR(start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)` — LR ramps from `lr·1e-8` to `lr` linearly over warmup.
   - `CosineAnnealingLR(T_max=max(1, max_steps-warmup_steps), eta_min=lr·min_ratio)` — cosine decay from `lr` down to the floor `lr·0.1`.
4. **Dataloaders (`:96-117`):** `DataLoader(train_data, batch_size, shuffle=True, collate_fn=lfm2_collator, pin_memory=True, …)`; optional val loader with `shuffle=False`. Then `accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)` (`:119`) wraps everything for the distributed/autocast context. `optimizer.zero_grad()` once, counters zeroed.

**Train loop (`train`, `:132-169`).** A `while self.step < self.max_steps` step-budget loop (not epoch-budget). It pulls `next(train_iter)`; on `StopIteration` it bumps `self.epoch`, rebuilds the iterator (`iter(self.train_loader)` re-draws the shuffle permutation), and pulls again (`:140-145`). Each iteration: `train_step(batch)` → `self.step += 1` → `log(out)`. Then interval-gated side effects: `accelerator.save_state()` every `save_interval` (`:151`); `model.eval(); validate(); model.train()` every `val_interval` (`:154-157`). On exit: `wait_for_everyone()` barrier, then `accelerator.save_model(unwrap_model(model), f"{output_dir}/final", max_shard_size="5GB", safe_serialization=True)` and `end_training()` (`:159-166`).

**train_step (`:171-182`).** `optimizer.zero_grad()` → `batch.to(device)` → `with accelerator.autocast(): out = self.model(batch)` → `accelerator.backward(out.loss)` → `optimizer.step()` → `scheduler.step()`. Note ordering: `scheduler.step()` fires **after** `optimizer.step()`, so optimizer step `s` consumes the LR the scheduler last set (step 0 therefore runs at the `1e-8` LinearLR floor).

**The loss is the model's, not the trainer's.** `out.loss` is computed inside `LFM2AudioModel.forward` (`lfm2_audio.py:453-478`): per-token `cross_entropy(reduction="none", ignore_index=-100)` separately on text logits → `text_loss` and on the depthformer audio logits → `audio_loss`. Audio loss is reshaped `(L·C)->(L,C)` over `C=codebooks` codebooks and weighted by the `audio_loss_weights` buffer: `(audio_loss * w).sum(-1) / w.sum()` (`lfm2_audio.py:463-464`). That buffer (`lfm2_audio.py:104-113`) is either `ones` with codebook-0 scaled by `semantic_codebook_factor`, or a `log`-spaced `linspace(1,0,C)·log(factor)` then `exp` — i.e. the semantic (first) codebook is up-weighted vs the acoustic residuals. Final scalar (`:470`) is the modality-weighted token-mean: `(text_mult·text_loss.sum() + audio_mult·audio_loss.sum()) / (text_mult·text_tokens + audio_mult·audio_tokens)`. The trainer never sees codebooks or modality multipliers — it only reads `out.loss` (and `out.audio_loss`/`out.text_loss` for logging hooks).

**validate (`:184-207`, `@torch.no_grad`).** Accumulates `loss_sum`/`loss_count` device tensors across the whole val loader (same `model(batch)` forward under autocast, `:194-197`), then `accelerator.reduce(..., "sum")` across processes and `mean = sum / count.clamp_min(1)`. Prints `val_loss` with elapsed `[mm:ss]`.

**log (`:209-218`).** On `logging_interval`: `accelerator.reduce(loss.detach(), "mean")` (cross-process mean of the per-rank loss), reads `optimizer.param_groups[0]["lr"]`, prints `loss`/`lr`.

There is no normalization scheme, attention, RoPE, conv, or sampling logic in this component — those live in the model/codec it drives. The only numeric ops here are the LR schedule formulas and the loss reductions; everything else is control flow and `accelerate` plumbing.

## Dtypes & shapes

| Stage | Input | Output |
|---|---|---|
| Batch `to(device)` (`:174`) | `LFM2AudioModelInput`: text int64 `(B,L)`, audio_in bf16 mel `(B,ΣT',128)`-ish, audio_in_lens int64 `(B,)`, audio_out int codes `(B,L,C)`, modality_flag int `(B,L)`, supervision_mask bool `(B,L)` | same, on device |
| `model(batch)` under autocast (`:177`) | batch above; weights bf16 | `LFM2AudioModelOutput{loss, audio_loss, text_loss: f32 scalars}` (CE upcasts to f32 internally) |
| `backward(out.loss)` (`:179`) | f32 scalar loss | grads on bf16 params (autocast manages master copies) |
| `lr_at`/scheduler | step int | LR float64 |
| `reduce(loss, "mean"/"sum")` | f32 scalar (per-rank) | f32 scalar (global) |
| `save_state`/`save_model` | bf16 param set | safetensors on disk (bf16) |

Internal promotions: cross-entropy and all loss reductions run in **f32** (the model upcasts logits/loss off the bf16 activations); token ids stay **int64**; audio codes are **int** (0..2048, 2048=EOAudio); LR math is **float64**. Weights are **bf16** throughout (disk and compute on CUDA).

## Wiring

**Upstream:**
- [data_dataloader](data/dataloader.md) — `DataLoader` + `lfm2_collator` produce the collated `LFM2AudioModelInput` batches (int64 text `(B,L)`, bf16 mel audio_in, int audio_out codes `(B,L,C)`, bool supervision_mask) fed to `train_step`/`validate`.
- [data_types](data/types.md) — the `LFM2AudioModelInput`/`LFM2AudioModelOutput` dataclasses (and `.to(device)`) that cross every trainer boundary.

**Driven (the model the trainer optimizes — the loss source, not "downstream output"):**
- [model_lfm2_audio](model/lfm2_audio.md) — `train_step`/`validate` call its `forward(batch)`; it returns `out.loss` (f32 scalar) and holds the `audio_loss_weights` buffer + per-modality multipliers. The trainer's gradients flow back into this model's bf16 params.

**Downstream (consumers of the trainer's *output* — produced weights):**
- [model_lfm2_audio](model/lfm2_audio.md) — the checkpointed/`save_model`'d bf16 safetensors are what `LFM2AudioModel.from_pretrained` later loads for inference.

## Python ↔ Rust

| Python (`trainer.py`) | Rust (`trainer.rs`) | Note |
|---|---|---|
| `Trainer.__init__` | `Trainer::new` / `Trainer::with_model` (`:262-309`) | `train_data is None ⇒ ValueError` enforced by the type system (`train_loader` non-`Option`) |
| `torch.optim.AdamW(..., fused=True)` | `candle_nn::AdamW` + `ParamsAdamW` (`:287-296`) | same math; no fused kernel (deliberate, PYTHON_VS_RUST §2.6) |
| `LinearLR ⇒ CosineAnnealingLR (SequentialLR)` | `Trainer::lr_at` (`:332-350`) | the identical piecewise schedule computed directly per step; scheduler counter folded in by `set_learning_rate(lr_at(step))` **before** the optimizer step (`:437`) to match torch's post-step `scheduler.step()` |
| `accelerator.autocast()` | implicit — weights already at load dtype, loss in f32 | no separate cast ctx (`:429-431`) |
| `accelerator.backward(loss)` + `optimizer.step()` | `optimizer.backward_step(&out.loss)` (`:439`) | candle builds a fresh `GradStore` each backward, so `zero_grad` is implicit |
| `accelerator.reduce(t, "mean"/"sum")` | `Trainer::reduce` (`:355-357`) | single-process identity (the all-reduce of a 1-proc group); faithful, not a stub |
| `accelerator.save_state` | `VarMap::save → state_step_{N}.safetensors` (`:502-506`) | |
| `accelerator.save_model(max_shard_size="5GB")` | `VarMap::save → model.safetensors` (`:512-516`) | single file; no sharding analog, tensors identical |
| `model.parameters()` | `varmap.all_vars()` | AdamW filters to float `Var`s, skipping int buffers |
| `DataLoader(shuffle=True, collate_fn=lfm2_collator)` | `LoaderDataIter` over `LFM2DataLoader` (`:98-177`) | splitmix64 Fisher–Yates (no `rand` dep); a different PRNG can't reproduce torch's permutation anyway, so a seeded reproducible shuffle is equally faithful |

**Deliberate divergences (PYTHON_VS_RUST §2.6):** the earlier Rust carried a duplicate trainer-side `Trainer::forward`/`LossConfig`/`ce_none`; these were **removed** so both `train_step` and `validate` route through `LFM2AudioModel::forward` and cannot diverge from the Python (which also has no trainer-side loss). Loaders are stored on `self` to match `self.train_loader`/`val_loader`. Single-process `reduce` is identity. Device-agnostic vs Python's GPU-coupled `dtype=bf16` (§2.1): the Rust `TrainerConfig.dtype` defaults to `F32` (CPU has no bf16 matmul in candle), bf16 only on CUDA/Metal.

## Precision / gotchas

- **The loss lives in the model, not the trainer.** Any per-codebook weighting, modality multiplier, or `ignore_index=-100` masking question must be answered at [model_lfm2_audio](model/lfm2_audio.md); the trainer only sums/means the returned f32 scalar. The Rust upcasts the val/log loss to f32 explicitly (`to_dtype(F32)`, `:468`/`:489`) before reducing.
- **Scheduler-after-optimizer ordering is load-bearing.** Step 0 runs at the LinearLR `1e-8` floor, not at peak `lr`; the Rust reproduces this by `set_learning_rate(lr_at(step))` *before* the in-place AdamW step (`:437`) — same value torch's `SequentialLR` would hold after the post-step `scheduler.step()`. Off-by-one here would shift the entire warmup curve by one step.
- **Cosine floor:** `eta_min = lr·min_ratio` (default `3e-6`), reached at `max_steps`; `T_max = max(1, max_steps-warmup_steps)` guards `warmup_steps == max_steps`.
- **DataIter error vs exhaustion (`trainer.rs:53,163-168`, unit-tested `:539`):** a row-load/collate failure must surface as `Err`, **not** `Ok(None)` — `None` is treated as epoch end, so swallowing the error would silently shorten training or cut validation short. Python's `DataLoader` raises the exception; the Rust `?`-propagates it.
- **`find_unused_parameters=True`** is required because modality-scatter leaves some params gradient-less on a given batch; without it DDP would error on the unused-param sync.
- **bf16 numerics:** AdamW master-weight copies are managed by `accelerate` autocast in Python; the cross-library f32 floor (~1e-6, PYTHON_VS_RUST §1.4) and the bf16 RMSNorm multiply-order subtlety (§2.4) belong to the model/codec, not this loop — the trainer's only float op is the f32 loss reduction.
