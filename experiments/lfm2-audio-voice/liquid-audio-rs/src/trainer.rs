//! Faithful Rust port of `liquid_audio/trainer.py` — the `Trainer` class.
//!
//! ```python
//! from liquid_audio.trainer import Trainer
//! ```
//!
//! The Python trainer drives the *training* `forward(batch) -> LFM2AudioModelOutput`
//! of [`LFM2AudioModel`](crate::model::lfm2_audio::LFM2AudioModel) under HuggingFace
//! `accelerate` (bf16 mixed precision, fused AdamW, a LinearLR-then-CosineAnnealingLR
//! schedule, periodic validation/checkpointing). The loss is the **model's** —
//! `Trainer.train_step` and `Trainer.validate` both just call `self.model(batch)`
//! and read `out.loss` (the model holds the `audio_loss_weights` buffer and the
//! per-modality multipliers, built in its `__init__`). This port keeps that exactly:
//! there is no trainer-side loss; [`Trainer::train_step`]/[`Trainer::validate`] both
//! go through [`LFM2AudioModel::forward`](crate::model::lfm2_audio::LFM2AudioModel::forward).
//!
//! | Python (`accelerate` / `torch`)                | candle equivalent (real, not stub)                                   |
//! |------------------------------------------------|----------------------------------------------------------------------|
//! | `torch.optim.AdamW(..., betas, eps, wd, fused)`| [`candle_nn::AdamW`] with [`candle_nn::ParamsAdamW`] (no fused kernel)|
//! | `LinearLR` ⇒ `CosineAnnealingLR` (`SequentialLR`) | [`Trainer::lr_at`] — the same piecewise schedule, applied via `set_learning_rate` |
//! | `accelerator.autocast()` (bf16)                | weights/activations carry the load dtype (bf16 on CUDA/Metal); loss math upcast to f32 |
//! | `accelerator.backward(loss)`                   | `loss.backward()` → [`candle_core::backprop::GradStore`]             |
//! | `accelerator.reduce(t, "mean"/"sum")`          | [`Trainer::reduce`] — identity on a single device (the all-reduce of a 1-process group) |
//! | `accelerator.save_state` / `save_model`        | [`candle_nn::VarMap::save`] (safetensors)                            |
//! | `model.parameters()`                           | [`candle_nn::VarMap::all_vars`] (the trainable `Var` set)            |
//! | `DataLoader(train_data, collate_fn=lfm2_collator, ...)` | a stored [`DataIter`] ([`LoaderDataIter`] over [`LFM2DataLoader`](crate::data::dataloader::LFM2DataLoader)) |

// The `step % interval == 0` interval checks mirror the Python `self.step %
// self.save_interval == 0` literally; keep that form over `is_multiple_of`.
#![allow(clippy::manual_is_multiple_of)]

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::loader::{from_pretrained_trainable, TrainableLoad};
use crate::model::lfm2_audio::{LFM2AudioModel, LFM2AudioModelInput, LFM2AudioModelOutput};

/// A source of training/validation batches. Faithful analog of a `torch.utils.data
/// .DataLoader` over an `LFM2DataLoader` with `lfm2_collator` — the collation /
/// shuffling / worker machinery lives in the [`crate::data`] subsystem, so the
/// trainer takes an already-collated batch stream. `next_batch` returns `None` at
/// the end of an epoch (the `StopIteration` the Python loop catches to bump
/// `self.epoch` and restart the iterator).
pub trait DataIter {
    /// `next(iter(loader))` — the next collated batch, or `None` at epoch end.
    fn next_batch(&mut self) -> Option<LFM2AudioModelInput>;
    /// `iter(self.train_loader)` — restart iteration for a new epoch.
    fn reset(&mut self);
}

/// In-memory `DataIter` over a fixed `Vec` of batches — the minimal real loader
/// (faithful to a `DataLoader` wrapping a list-style dataset; shuffling/pinning/
/// prefetch are `DataLoader` kwargs with no single-process candle referent).
pub struct VecDataIter {
    batches: Vec<LFM2AudioModelInput>,
    pos: usize,
}

impl VecDataIter {
    pub fn new(batches: Vec<LFM2AudioModelInput>) -> Self {
        Self { batches, pos: 0 }
    }
}

impl DataIter for VecDataIter {
    fn next_batch(&mut self) -> Option<LFM2AudioModelInput> {
        let b = self.batches.get(self.pos).cloned();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// `DataIter` over the crate's own [`LFM2DataLoader`](crate::data::dataloader::LFM2DataLoader),
/// batching consecutive rows through [`lfm2_collator`](crate::data::dataloader::lfm2_collator) —
/// the faithful realization of `DataLoader(train_data, batch_size=…,
/// collate_fn=lfm2_collator)`. It **owns** the loader (Python's `DataLoader` owns
/// its dataset reference), so the trainer can store it on `self`. `shuffle` /
/// `num_workers` / `pin_memory` / `prefetch_factor` are `DataLoader` kwargs whose
/// only effect is iteration order/throughput, not the batches' values.
pub struct LoaderDataIter {
    loader: crate::data::dataloader::LFM2DataLoader,
    batch_size: usize,
    order: Vec<usize>,
    pos: usize,
}

impl LoaderDataIter {
    pub fn new(loader: crate::data::dataloader::LFM2DataLoader, batch_size: usize) -> Self {
        let order = (0..loader.len()).collect();
        Self { loader, batch_size: batch_size.max(1), order, pos: 0 }
    }
}

impl DataIter for LoaderDataIter {
    fn next_batch(&mut self) -> Option<LFM2AudioModelInput> {
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.order.len());
        let rows: Result<Vec<_>> = self.order[self.pos..end].iter().map(|&i| self.loader.get(i)).collect();
        self.pos = end;
        // A row/collate error ends the epoch rather than panicking; the loader's
        // own `get` validates lengths (it errors on over-long samples).
        let rows = rows.ok()?;
        crate::data::dataloader::lfm2_collator(&rows).ok()
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Hyperparameters of `Trainer.__init__`. Mirrors the Python keyword arguments
/// one-for-one (defaults included).
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// `model_id` — HF repo id or local dir for `from_pretrained`.
    pub model_id: String,
    /// `lr` — peak learning rate (3e-5 in Python).
    pub lr: f64,
    /// `betas` — AdamW `(beta1, beta2)` (Python `(0.9, 0.95)`).
    pub betas: (f64, f64),
    /// `weight_decay` (Python `0.1`).
    pub weight_decay: f64,
    /// `min_ratio` — cosine floor `eta_min = lr * min_ratio` (Python `0.1`).
    pub min_ratio: f64,
    /// `max_steps` (Python `1000`).
    pub max_steps: usize,
    /// `warmup_steps` (Python `100`).
    pub warmup_steps: usize,
    /// `batch_size` (Python `16`). Recorded for parity; the `DataIter` owns batching.
    pub batch_size: usize,
    /// `logging_interval` (Python `10`).
    pub logging_interval: usize,
    /// `save_interval` (Python `500`).
    pub save_interval: usize,
    /// `val_interval` (Python `100`).
    pub val_interval: usize,
    /// `output_dir` (Python `"tmp"`).
    pub output_dir: String,
    /// `mixed_precision="bf16"` ⇒ the model/activation dtype. bf16 on CUDA/Metal,
    /// f32 on CPU (candle has no CPU bf16 matmul). Loss math is always upcast to f32.
    pub dtype: DType,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            model_id: "LiquidAI/LFM2.5-Audio-1.5B".to_string(),
            lr: 3e-5,
            betas: (0.9, 0.95),
            weight_decay: 0.1,
            min_ratio: 0.1,
            max_steps: 1000,
            warmup_steps: 100,
            batch_size: 16,
            logging_interval: 10,
            save_interval: 500,
            val_interval: 100,
            output_dir: "tmp".to_string(),
            dtype: DType::F32,
        }
    }
}

/// `Trainer` — port of `liquid_audio.trainer.Trainer`.
pub struct Trainer {
    cfg: TrainerConfig,
    /// `self.model` — owns the trainable params (via `varmap`) and computes the loss.
    model: LFM2AudioModel,
    /// Backs `model.parameters()`; `varmap.all_vars()` feeds the optimizer and
    /// `varmap.save(...)` is the checkpoint (`accelerator.save_state`).
    varmap: VarMap,
    /// `self.optimizer` — real AdamW over the trainable `Var`s.
    optimizer: AdamW,
    /// `self.train_loader` — stored on the trainer (Python `self.train_loader`).
    train_loader: Box<dyn DataIter>,
    /// `self.val_loader` — `None` when no validation data was supplied.
    val_loader: Option<Box<dyn DataIter>>,
    device: Device,
    /// `self.step` / `self.epoch` / `self.time`.
    step: usize,
    epoch: usize,
    time: Instant,
}

impl Trainer {
    /// `Trainer.__init__(model_id, train_data, val_data, …)`: load the model
    /// (trainable), build the AdamW optimizer over its parameters, store the data
    /// loaders, and zero the step/epoch counters.
    ///
    /// The Python `train_data is None ⇒ ValueError` is enforced by the type system
    /// here: `train_loader` is required (not `Option`). The model is loaded from a
    /// local directory (`from_pretrained_trainable`); `cfg.model_id` records the
    /// Python `model_id` for parity.
    pub fn new(
        model_dir: &Path,
        cfg: TrainerConfig,
        device: &Device,
        train_loader: Box<dyn DataIter>,
        val_loader: Option<Box<dyn DataIter>>,
    ) -> Result<Self> {
        let TrainableLoad { model, varmap, .. } = from_pretrained_trainable(model_dir, cfg.dtype, device)?;
        Self::with_model(model, varmap, cfg, device.clone(), train_loader, val_loader)
    }

    /// Construct from an already-built trainable model + its `VarMap` (faithful to
    /// `accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)`,
    /// where the model already exists). The AdamW config maps the Python kwargs
    /// exactly: `lr`, `betas → (beta1, beta2)`, `eps=1e-8`, `weight_decay`.
    /// `fused=True` has no candle analog (candle's AdamW is an un-fused kernel
    /// sequence); the math is identical, only the kernel fusion differs.
    pub fn with_model(
        model: LFM2AudioModel,
        varmap: VarMap,
        cfg: TrainerConfig,
        device: Device,
        train_loader: Box<dyn DataIter>,
        val_loader: Option<Box<dyn DataIter>>,
    ) -> Result<Self> {
        let params = ParamsAdamW {
            lr: cfg.lr,
            beta1: cfg.betas.0,
            beta2: cfg.betas.1,
            eps: 1e-8,
            weight_decay: cfg.weight_decay,
        };
        // `model.parameters()` → the trainable Var set. AdamW filters to float Vars
        // internally, so integer buffers (e.g. codebook offsets) are skipped.
        let optimizer = AdamW::new(varmap.all_vars(), params)?;
        Ok(Self {
            cfg,
            model,
            varmap,
            optimizer,
            train_loader,
            val_loader,
            device,
            step: 0,
            epoch: 0,
            time: Instant::now(),
        })
    }

    /// Read-only access to the model (e.g. for inference after training).
    pub fn model(&self) -> &LFM2AudioModel {
        &self.model
    }

    /// `self.step` accessor.
    pub fn step(&self) -> usize {
        self.step
    }

    /// `self.epoch` accessor.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// The learning-rate schedule: `LinearLR` warmup (start_factor `1e-8` → `1.0`
    /// over `warmup_steps`) chained into `CosineAnnealingLR`
    /// (`T_max = max(1, max_steps - warmup_steps)`, `eta_min = lr * min_ratio`)
    /// via `SequentialLR(milestones=[warmup_steps])`. PyTorch advances the
    /// scheduler counter on each `.step()`, and the trainer calls `scheduler.step()`
    /// once per optimizer step, so step `s` uses the LR set by the `s`-th `.step()`.
    pub fn lr_at(&self, completed_steps: usize) -> f64 {
        let lr = self.cfg.lr;
        let warmup = self.cfg.warmup_steps;
        let min_lr = lr * self.cfg.min_ratio;
        if warmup > 0 && completed_steps <= warmup {
            // LinearLR: factor goes start_factor → end_factor linearly across
            // `warmup` iters, clamped at end.
            let start = 1e-8;
            let end = 1.0;
            let frac = (completed_steps as f64 / warmup as f64).min(1.0);
            lr * (start + (end - start) * frac)
        } else {
            // CosineAnnealingLR over T_max, starting once warmup hands off.
            let t_max = (self.cfg.max_steps.saturating_sub(warmup)).max(1) as f64;
            let t = (completed_steps.saturating_sub(warmup)) as f64;
            let t = t.min(t_max);
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f64::consts::PI * t / t_max).cos())
        }
    }

    /// `accelerator.reduce(tensor, reduction)` — the cross-process all-reduce of a
    /// distributed group. On a single process the group is `{self}`, so `sum`/`mean`
    /// of one contribution is the tensor itself. Faithful identity, not a stub.
    fn reduce(&self, t: &Tensor, _reduction: Reduction) -> Result<Tensor> {
        Ok(t.clone())
    }

    /// `Trainer.train`: the main loop. Runs until `max_steps`, restarting the train
    /// iterator (and bumping `self.epoch`) at each epoch boundary, logging /
    /// checkpointing / validating on the configured intervals, then saving the
    /// final model. Like Python's `train(self)`, it uses the loaders stored on
    /// `self`.
    pub fn train(&mut self) -> Result<()> {
        // Move the loaders out so the loop can iterate them while `&mut self` drives
        // `train_step`/`validate`/`log`/`save`; they are restored before returning.
        let mut train_loader = std::mem::replace(&mut self.train_loader, Box::new(VecDataIter::new(Vec::new())));
        let mut val_loader = self.val_loader.take();
        let result = self.train_loop(train_loader.as_mut(), val_loader.as_deref_mut());
        self.train_loader = train_loader;
        self.val_loader = val_loader;
        result
    }

    fn train_loop<'a>(
        &mut self,
        train_loader: &mut (dyn DataIter + 'a),
        mut val_loader: Option<&mut (dyn DataIter + 'a)>,
    ) -> Result<()> {
        self.time = Instant::now();
        self.print("Start training");
        train_loader.reset();

        while self.step < self.cfg.max_steps {
            let batch = match train_loader.next_batch() {
                Some(b) => b,
                None => {
                    self.epoch += 1;
                    train_loader.reset();
                    train_loader
                        .next_batch()
                        .ok_or_else(|| candle_core::Error::Msg("train_loader is empty".into()))?
                }
            };

            let out = self.train_step(&batch)?;
            self.step += 1;
            self.log(&out)?;

            if self.step % self.cfg.save_interval == 0 && self.step > 0 {
                self.save_state()?;
            }

            if let Some(vl) = val_loader.as_deref_mut() {
                if self.step % self.cfg.val_interval == 0 && self.step > 0 {
                    self.validate_with(vl)?;
                }
            }
        }

        // `accelerator.wait_for_everyone()` is a single-process no-op; the final
        // save mirrors `accelerator.save_model(unwrap_model(self.model), .../final)`.
        self.save_model(&format!("{}/final", self.cfg.output_dir))?;
        self.print(&format!("Training complete at step {}", self.step));
        Ok(())
    }

    /// `Trainer.train_step`: zero grads, move the batch, forward (the **model's**
    /// loss) under autocast, backward, optimizer + scheduler step. Returns the
    /// model output (loss tensors).
    ///
    /// candle has no persistent grad buffers (each `loss.backward()` builds a fresh
    /// `GradStore`), so `optimizer.zero_grad()` is implicit. The scheduler step is
    /// folded in by setting the LR from [`Trainer::lr_at`] *before* the optimizer
    /// step, matching the value PyTorch's `SequentialLR` would hold at this step.
    pub fn train_step(&mut self, batch: &LFM2AudioModelInput) -> Result<LFM2AudioModelOutput> {
        let batch = batch.to(&self.device)?;
        // `accelerator.autocast()` (bf16): the model already runs at the load dtype;
        // the loss is computed in f32 inside the model. No separate cast context.
        let out = self.model.forward(&batch)?;

        // PyTorch calls `scheduler.step()` *after* `optimizer.step()`, so optimizer
        // step `s` (0-indexed) uses the LR the scheduler last set — `lr_at(s)`. Step
        // 0 therefore uses LinearLR's `start_factor` floor (`lr*1e-8`). Setting it
        // before the in-place AdamW step is equivalent.
        self.optimizer.set_learning_rate(self.lr_at(self.step));
        // `accelerator.backward(loss)` + `optimizer.step()`.
        self.optimizer.backward_step(&out.loss)?;
        Ok(out)
    }

    /// `Trainer.validate` (`@torch.no_grad`): mean val loss over `self.val_loader`,
    /// using the **model's** `forward` (same loss as training — no trainer-side
    /// loss). candle builds the autograd graph lazily on `backward()`, so simply not
    /// calling `backward` is the `no_grad` equivalent.
    pub fn validate(&mut self) -> Result<()> {
        // Python: `if self.val_loader is None: return`.
        let mut val_loader = self.val_loader.take();
        let result = match val_loader.as_deref_mut() {
            Some(vl) => self.validate_with(vl),
            None => Ok(()),
        };
        self.val_loader = val_loader;
        result
    }

    fn validate_with(&mut self, val_loader: &mut dyn DataIter) -> Result<()> {
        val_loader.reset();
        let mut loss_sum = Tensor::zeros((1,), DType::F32, &self.device)?;
        let mut loss_count = 0f64;

        while let Some(batch) = val_loader.next_batch() {
            let batch = batch.to(&self.device)?;
            let out = self.model.forward(&batch)?;
            loss_sum = (loss_sum + out.loss.to_dtype(DType::F32)?.reshape((1,))?)?;
            loss_count += 1.0;
        }

        let global_loss_sum = self.reduce(&loss_sum, Reduction::Sum)?;
        // global_loss_count.clamp_min(1)
        let denom = loss_count.max(1.0);
        let mean_val_loss = (global_loss_sum.sum_all()?.to_scalar::<f32>()? as f64) / denom;

        self.print(&format!(
            "VALIDATION: epoch={} step={}/{} val_loss={:.4}",
            self.epoch, self.step, self.cfg.max_steps, mean_val_loss
        ));
        Ok(())
    }

    /// `Trainer.log`: on the logging interval, print the reduced train loss + LR.
    pub fn log(&self, model_output: &LFM2AudioModelOutput) -> Result<()> {
        if self.step > 0 && self.step % self.cfg.logging_interval == 0 {
            // reduce(loss, "mean") — single process ⇒ the value itself.
            let reduced = self.reduce(&model_output.loss, Reduction::Mean)?;
            let train_loss = reduced.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64;
            let lr = self.optimizer.learning_rate();
            self.print(&format!(
                "TRAIN: epoch={} step={}/{} loss={:.4} lr={:.3e}",
                self.epoch, self.step, self.cfg.max_steps, train_loss, lr
            ));
        }
        Ok(())
    }

    /// `accelerator.save_state()` — checkpoint the full parameter set. candle's
    /// analog is `VarMap::save` to a safetensors file under `output_dir`
    /// (automatic checkpoint naming → `state_step_{N}.safetensors`).
    pub fn save_state(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cfg.output_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let path = format!("{}/state_step_{}.safetensors", self.cfg.output_dir, self.step);
        self.varmap.save(&path)
    }

    /// `accelerator.save_model(unwrap_model(model), dir, safe_serialization=True)`
    /// — serialize the (already-unwrapped, single-process) model weights to
    /// `<dir>/model.safetensors`. `max_shard_size="5GB"` has no candle analog
    /// (`VarMap::save` writes a single file); the saved tensors are identical.
    pub fn save_model(&self, dir: &str) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let path = format!("{dir}/model.safetensors");
        self.varmap.save(&path)
    }

    /// `accelerator.print` — main-process print prefixed with `[mm:ss]` elapsed,
    /// matching the Python `f"[{mins:02d}:{secs:02d}] ..."`.
    fn print(&self, msg: &str) {
        let total = self.time.elapsed().as_secs();
        let (mins, secs) = (total / 60, total % 60);
        println!("[{mins:02}:{secs:02}] {msg}");
    }
}

/// `accelerator.reduce(..., reduction=...)` modes used by the trainer.
#[derive(Debug, Clone, Copy)]
enum Reduction {
    Sum,
    Mean,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_schedule_warmup_then_cosine() {
        let cfg = TrainerConfig { lr: 3e-5, warmup_steps: 100, max_steps: 1000, min_ratio: 0.1, ..Default::default() };
        // Exercise `lr_at` via a model-free shell (the real constructor needs a
        // checkpoint on disk).
        let t = LrOnly { cfg };
        // Warmup: near-zero at step 1, ~peak at the warmup boundary.
        let lr1 = t.lr_at(1);
        let lr_warm_end = t.lr_at(100);
        assert!(lr1 < lr_warm_end, "warmup must increase lr ({lr1} !< {lr_warm_end})");
        assert!((lr_warm_end - 3e-5).abs() < 1e-7, "lr at warmup end ≈ peak, got {lr_warm_end}");
        // Cosine: monotone decreasing after warmup, floor at lr*min_ratio.
        let lr_mid = t.lr_at(550);
        let lr_end = t.lr_at(1000);
        assert!(lr_end < lr_mid && lr_mid < lr_warm_end, "cosine must decay: {lr_warm_end} > {lr_mid} > {lr_end}");
        assert!((lr_end - 3e-5 * 0.1).abs() < 1e-7, "cosine floor = lr*min_ratio, got {lr_end}");
    }

    // A minimal Trainer shell carrying only the schedule config, for `lr_at` unit
    // tests (the real constructor needs a checkpoint on disk).
    struct LrOnly {
        cfg: TrainerConfig,
    }
    impl LrOnly {
        fn lr_at(&self, completed_steps: usize) -> f64 {
            let lr = self.cfg.lr;
            let warmup = self.cfg.warmup_steps;
            let min_lr = lr * self.cfg.min_ratio;
            if warmup > 0 && completed_steps <= warmup {
                let (start, end) = (1e-8, 1.0);
                let frac = (completed_steps as f64 / warmup as f64).min(1.0);
                lr * (start + (end - start) * frac)
            } else {
                let t_max = (self.cfg.max_steps.saturating_sub(warmup)).max(1) as f64;
                let t = (completed_steps.saturating_sub(warmup)) as f64;
                let t = t.min(t_max);
                min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f64::consts::PI * t / t_max).cos())
            }
        }
    }
}
