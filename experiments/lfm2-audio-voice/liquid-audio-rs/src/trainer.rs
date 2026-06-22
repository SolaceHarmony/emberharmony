//! Faithful Rust port of `liquid_audio/trainer.py` — the `Trainer` class.
//!
//! ```python
//! from liquid_audio.trainer import Trainer
//! ```
//!
//! The Python trainer drives the *training* `forward(batch) -> LFM2AudioModelOutput`
//! of [`LFM2AudioModel`](crate::model::lfm2_audio::LFM2AudioModel) under HuggingFace
//! `accelerate` (bf16 mixed precision, fused AdamW, a LinearLR-then-CosineAnnealingLR
//! schedule, periodic validation/checkpointing). This port keeps the same control
//! flow and hyperparameters, expressed against the existing crate types and candle:
//!
//! | Python (`accelerate` / `torch`)                | candle equivalent (real, not stub)                                   |
//! |------------------------------------------------|----------------------------------------------------------------------|
//! | `torch.optim.AdamW(..., betas, eps, wd, fused)`| [`candle_nn::AdamW`] with [`candle_nn::ParamsAdamW`] (no fused kernel)|
//! | `LinearLR` ⇒ `CosineAnnealingLR` (`SequentialLR`) | [`lr_at`] — the same piecewise schedule, applied via `set_learning_rate` |
//! | `accelerator.autocast()` (bf16)                | weights/activations carry the load dtype (bf16 on CUDA/Metal); loss math upcast to f32 |
//! | `accelerator.backward(loss)`                   | `loss.backward()` → [`candle_core::backprop::GradStore`]             |
//! | `accelerator.reduce(t, "mean"/"sum")`          | [`Trainer::reduce`] — identity on a single device (the all-reduce of a 1-process group) |
//! | `accelerator.save_state` / `save_model`        | [`candle_nn::VarMap::save`] (safetensors)                            |
//! | `model.parameters()`                           | [`candle_nn::VarMap::all_vars`] (the trainable `Var` set)            |
//! | `DataLoader` (shuffle, collate, workers)       | the [`DataIter`] trait — the caller supplies batches (data pipeline is out of port scope) |
//!
//! `nn.functional.cross_entropy(..., ignore_index=-100, reduction="none")` is
//! reproduced as a manual `log_softmax`+gather (candle's [`candle_nn::loss::cross_entropy`]
//! mean-reduces, which the weighted normalization here cannot use). The `-100`
//! ignore index never appears: [`LFM2AudioModel::logits`](crate::model::lfm2_audio::LFM2AudioModel::logits)
//! already restricts labels to supervised positions, so "ignore" is by construction.

// The `step % interval == 0` interval checks mirror the Python `self.step %
// self.save_interval == 0` literally; keep that form over `is_multiple_of`.
#![allow(clippy::manual_is_multiple_of)]

use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::log_softmax;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::loader::{from_pretrained_trainable, TrainableLoad};
use crate::model::lfm2_audio::{LFM2AudioModel, LFM2AudioModelInput, LFM2AudioModelOutput};
use crate::utils::mel2emb_len;

/// A source of training/validation batches. Faithful analog of a `torch.utils.data
/// .DataLoader` over an `LFM2DataLoader` with `lfm2_collator` — the collation /
/// shuffling / worker machinery lives in the (unported) `liquid_audio.data`
/// subsystem, so the trainer takes an already-collated batch stream. `next_batch`
/// returns `None` at the end of an epoch (the `StopIteration` the Python loop
/// catches to bump `self.epoch` and restart the iterator).
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
/// the faithful realization of the Python
/// `DataLoader(train_data, batch_size=…, collate_fn=lfm2_collator)`. `shuffle` /
/// `num_workers` / `pin_memory` / `prefetch_factor` are `DataLoader` kwargs whose
/// only effect is iteration order and throughput, not the batches' values; the
/// optional `shuffle` here reorders row indices for parity with `shuffle=True`.
pub struct LoaderDataIter<'a> {
    loader: &'a crate::data::dataloader::LFM2DataLoader,
    batch_size: usize,
    order: Vec<usize>,
    pos: usize,
}

impl<'a> LoaderDataIter<'a> {
    pub fn new(loader: &'a crate::data::dataloader::LFM2DataLoader, batch_size: usize) -> Self {
        Self { loader, batch_size: batch_size.max(1), order: (0..loader.len()).collect(), pos: 0 }
    }
}

impl DataIter for LoaderDataIter<'_> {
    fn next_batch(&mut self) -> Option<LFM2AudioModelInput> {
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.order.len());
        let rows: Result<Vec<_>> = self.order[self.pos..end].iter().map(|&i| self.loader.get(i)).collect();
        self.pos = end;
        // A row/collate error is surfaced as an empty batch end rather than a panic;
        // the loader's own `get` validates lengths (it errors on over-long samples).
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

/// The per-codebook loss weights and modality multipliers the training `forward`
/// applies (Python `LFM2AudioModel.audio_loss_weights` + `conf.*_loss_multiplier`).
/// Built from the model config at load time.
#[derive(Debug, Clone)]
pub struct LossConfig {
    pub codebooks: usize,
    pub text_loss_multiplier: f64,
    pub audio_loss_multiplier: f64,
    /// `audio_loss_weights` — `(C,)`, the per-codebook weighting (`log`/`linear`).
    pub audio_loss_weights: Tensor,
}

impl LossConfig {
    /// Build `audio_loss_weights` exactly as `LFM2AudioModel.__init__`:
    /// ```python
    /// if codebook_weight == "log":
    ///     weights = (linspace(1, 0, C) * log(semantic_codebook_factor)).exp()
    /// else:
    ///     weights = ones(C); weights[0] *= semantic_codebook_factor
    /// ```
    pub fn new(
        codebooks: usize,
        codebook_weight: &str,
        semantic_codebook_factor: f64,
        text_loss_multiplier: f64,
        audio_loss_multiplier: f64,
        device: &Device,
    ) -> Result<Self> {
        let weights: Vec<f32> = if codebook_weight == "log" {
            let log_factor = semantic_codebook_factor.ln();
            (0..codebooks)
                .map(|i| {
                    // linspace(1, 0, C)[i] = 1 - i/(C-1)  (C==1 ⇒ the single point 1.0)
                    let t = if codebooks > 1 { 1.0 - i as f64 / (codebooks as f64 - 1.0) } else { 1.0 };
                    (t * log_factor).exp() as f32
                })
                .collect()
        } else {
            let mut w = vec![1.0f32; codebooks];
            if let Some(first) = w.first_mut() {
                *first *= semantic_codebook_factor as f32;
            }
            w
        };
        let audio_loss_weights = Tensor::from_vec(weights, (codebooks,), device)?;
        Ok(Self { codebooks, text_loss_multiplier, audio_loss_multiplier, audio_loss_weights })
    }
}

/// `Trainer` — port of `liquid_audio.trainer.Trainer`.
pub struct Trainer {
    cfg: TrainerConfig,
    /// `self.model` — owns the trainable params (via `varmap`).
    model: LFM2AudioModel,
    /// Backs `model.parameters()`; `varmap.all_vars()` feeds the optimizer and
    /// `varmap.save(...)` is the checkpoint (`accelerator.save_state`).
    varmap: VarMap,
    /// `self.optimizer` — real AdamW over the trainable `Var`s.
    optimizer: AdamW,
    loss: LossConfig,
    device: Device,
    /// `self.step` / `self.epoch` / `self.time`.
    step: usize,
    epoch: usize,
    time: Instant,
}

impl Trainer {
    /// `Trainer.__init__`: load the model + processor (trainable), build the
    /// AdamW optimizer over its parameters, and zero the step/epoch counters.
    ///
    /// The data loaders are passed to [`Trainer::train`] (Python stores them on
    /// `self`; here they stream in, matching the `DataIter` design and avoiding a
    /// self-referential borrow of the model's device). `train_data is None ⇒
    /// ValueError` is enforced by `train` taking the loader by value.
    pub fn from_pretrained(model_dir: &std::path::Path, cfg: TrainerConfig, device: &Device) -> Result<Self> {
        let TrainableLoad {
            model,
            varmap,
            processor: _processor,
            codebooks,
            codebook_weight,
            semantic_codebook_factor,
            text_loss_multiplier,
            audio_loss_multiplier,
        } = from_pretrained_trainable(model_dir, cfg.dtype, device)?;

        let loss = LossConfig::new(
            codebooks,
            &codebook_weight,
            semantic_codebook_factor,
            text_loss_multiplier,
            audio_loss_multiplier,
            device,
        )?;
        let trainer = Self::with_model(model, varmap, loss, cfg, device.clone())?;
        Ok(trainer)
    }

    /// Construct directly from an already-built trainable model + its `VarMap`
    /// (faithful to `accelerator.prepare(model, optimizer, ...)`, where the model
    /// already exists). The AdamW config maps the Python kwargs exactly:
    /// `lr`, `betas → (beta1, beta2)`, `eps=1e-8`, `weight_decay`. `fused=True`
    /// has no candle analog (candle's AdamW is a single un-fused kernel sequence);
    /// the math is identical, only the kernel fusion differs.
    pub fn with_model(model: LFM2AudioModel, varmap: VarMap, loss: LossConfig, cfg: TrainerConfig, device: Device) -> Result<Self> {
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
            loss,
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
    /// via `SequentialLR(milestones=[warmup_steps])`. Returns the LR *after*
    /// `step+1` scheduler steps — PyTorch's `LinearLR`/`CosineAnnealingLR` advance
    /// their internal counter on each `.step()` call, and the trainer calls
    /// `scheduler.step()` once per optimizer step (so step `s` uses the LR set by
    /// the `s`-th `.step()`).
    pub fn lr_at(&self, completed_steps: usize) -> f64 {
        let lr = self.cfg.lr;
        let warmup = self.cfg.warmup_steps;
        let min_lr = lr * self.cfg.min_ratio;
        if warmup > 0 && completed_steps <= warmup {
            // LinearLR: factor goes start_factor → end_factor linearly across
            // `warmup` iters. After `completed_steps` steps the factor is
            // start + (end - start) * completed_steps / total_iters, clamped at end.
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
    /// distributed group. On a single process (the only mode without a real
    /// distributed backend) the group is `{self}`, so `sum`/`mean` of one
    /// contribution is the tensor itself. Faithful identity, not a stub.
    fn reduce(&self, t: &Tensor, _reduction: Reduction) -> Result<Tensor> {
        Ok(t.clone())
    }

    /// `Trainer.train`: the main loop. Runs until `max_steps`, restarting the
    /// train iterator (and bumping `self.epoch`) at each epoch boundary, logging /
    /// checkpointing / validating on the configured intervals, then saving the
    /// final model.
    pub fn train<T: DataIter, V: DataIter>(&mut self, train_loader: &mut T, val_loader: Option<&mut V>) -> Result<()> {
        self.time = Instant::now();
        self.print("Start training");
        train_loader.reset();
        let mut val_loader = val_loader;

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
                    self.validate(vl)?;
                }
            }
        }

        // `accelerator.wait_for_everyone()` is a single-process no-op; the final
        // save mirrors `accelerator.save_model(unwrap_model(self.model), .../final)`.
        self.save_model(&format!("{}/final", self.cfg.output_dir))?;
        self.print(&format!("Training complete at step {}", self.step));
        Ok(())
    }

    /// `Trainer.train_step`: zero grads, move the batch, forward under autocast,
    /// backward, optimizer + scheduler step. Returns the model output (loss tensors).
    ///
    /// candle has no persistent grad buffers (each `loss.backward()` builds a fresh
    /// `GradStore`), so `optimizer.zero_grad()` is implicit — there is nothing to
    /// zero. The scheduler step is folded in by setting the LR from [`lr_at`]
    /// *before* the optimizer step, matching the value PyTorch's `SequentialLR`
    /// would hold at this step.
    pub fn train_step(&mut self, batch: &LFM2AudioModelInput) -> Result<LFM2AudioModelOutput> {
        let batch = batch.to(&self.device)?;
        // `accelerator.autocast()` (bf16): the model already runs at the load dtype;
        // the loss is computed in f32 (see `forward`), which is what AMP's loss
        // accumulation does. No separate cast context is needed.
        let out = self.forward(&batch)?;

        // `scheduler.step()`: the LR PyTorch would set on the (step+1)-th call.
        self.optimizer.set_learning_rate(self.lr_at(self.step + 1));
        // `accelerator.backward(loss)` + `optimizer.step()`.
        self.optimizer.backward_step(&out.loss)?;
        Ok(out)
    }

    /// The training `forward(batch) -> LFM2AudioModelOutput` of
    /// [`LFM2AudioModel`](crate::model::lfm2_audio::LFM2AudioModel). Lives here (not
    /// on the model) because the loss weights/multipliers are training config; the
    /// model exposes the parity-verified `logits(batch)` that this consumes.
    /// Faithful to `LFM2AudioModel.forward`:
    /// ```python
    /// text_logits, audio_logits, text_labels, audio_labels = self.logits(batch)
    /// text_loss = cross_entropy(text_logits, text_labels, ignore_index=-100, reduction="none")
    /// audio_loss = cross_entropy(audio_logits, audio_labels, ignore_index=-100, reduction="none")
    /// audio_loss = rearrange(audio_loss, "(L C) -> L C", C=codebooks)
    /// audio_loss = (audio_loss * audio_loss_weights).sum(-1) / audio_loss_weights.sum()
    /// text_tokens = text_loss.numel(); audio_tokens = audio_loss.numel()
    /// weighted_tokens = t_mult * text_tokens + a_mult * audio_tokens
    /// loss = (t_mult * text_loss.sum() + a_mult * audio_loss.sum()) / (weighted_tokens + 1e-6)
    /// ```
    pub fn forward(&self, batch: &LFM2AudioModelInput) -> Result<LFM2AudioModelOutput> {
        let (text_logits, audio_logits, text_labels, audio_labels) = self.model.logits(batch)?;
        let dev = text_logits.device();

        // cross_entropy(reduction="none"): per-row -log p(label). `logits`/`labels`
        // already cover only supervised positions, so `ignore_index=-100` is moot.
        let text_loss = ce_none(&text_logits, &text_labels)?; // (n_text,)
        let audio_loss_flat = ce_none(&audio_logits, &audio_labels)?; // (n_audio * C,)

        let c = self.loss.codebooks;
        let n_audio = audio_loss_flat.dim(0)? / c.max(1);
        // rearrange "(L C) -> L C"; weight per-codebook then sum / weight-sum.
        let aw = self.loss.audio_loss_weights.to_dtype(DType::F32)?; // (C,)
        let aw_sum = aw.sum_all()?.to_scalar::<f32>()? as f64;
        let audio_loss = if n_audio == 0 {
            Tensor::zeros((0,), DType::F32, dev)?
        } else {
            let al = audio_loss_flat.reshape((n_audio, c))?; // (L, C)
            let weighted = al.broadcast_mul(&aw)?.sum(1)?; // (L,)
            (weighted / aw_sum)?
        };

        let text_tokens = text_loss.dim(0)?; // numel of a 1-D tensor
        let audio_tokens = audio_loss.dim(0)?;
        let (tm, am) = (self.loss.text_loss_multiplier, self.loss.audio_loss_multiplier);
        let weighted_tokens = tm * text_tokens as f64 + am * audio_tokens as f64;

        let text_sum = sum_all_f32(&text_loss)?;
        let audio_sum = sum_all_f32(&audio_loss)?;
        let loss = ((&text_sum * tm)? + (&audio_sum * am)?)?;
        let loss = (loss / (weighted_tokens + 1e-6))?;

        // Per-modality mean losses (output diagnostics).
        let audio_loss_mean = (&audio_sum / (audio_tokens as f64 + 1e-6))?;
        let text_loss_mean = (&text_sum / (text_tokens as f64 + 1e-6))?;

        // audio_out_tokens = batch.audio_out.shape[1]
        let audio_out_tokens = Tensor::new(batch.audio_out.dim(1)? as i64, dev)?;
        // text_tokens = (batch.text[0] > 0).sum()
        let text_row = batch.text.i(0)?.to_dtype(DType::I64)?;
        let n_text_pos = text_row.ge(1i64)?.to_dtype(DType::I64)?.sum_all()?;
        // audio_in_tokens = mel2emb_len(batch.audio_in_lens).sum()
        let lens: Vec<i64> = batch.audio_in_lens.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let audio_in_tokens_val: i64 = lens.iter().map(|&l| mel2emb_len(l)).sum();
        let audio_in_tokens = Tensor::new(audio_in_tokens_val, dev)?;

        Ok(LFM2AudioModelOutput {
            loss,
            audio_loss: audio_loss_mean,
            text_loss: text_loss_mean,
            audio_out_tokens,
            text_tokens: n_text_pos,
            audio_in_tokens,
        })
    }

    /// `Trainer.validate` (`@torch.no_grad`): mean val loss over the val loader.
    /// candle builds the autograd graph lazily on `backward()`, so simply not
    /// calling `backward` is the `no_grad` equivalent (and `out.loss.detach()` is a
    /// plain forward tensor here).
    pub fn validate<V: DataIter>(&mut self, val_loader: &mut V) -> Result<()> {
        val_loader.reset();
        let mut loss_sum = Tensor::zeros((1,), DType::F32, &self.device)?;
        let mut loss_count = 0f64;

        while let Some(batch) = val_loader.next_batch() {
            let batch = batch.to(&self.device)?;
            let out = self.forward(&batch)?;
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

/// `nn.functional.cross_entropy(logits, labels, reduction="none")` — per-row
/// negative log-likelihood. Returns `(N,)`; for an empty input returns `(0,)`.
/// (candle's `candle_nn::loss::cross_entropy` mean-reduces and so can't be used
/// for the custom weighted normalization the model's loss needs.)
fn ce_none(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let n = logits.dim(0)?;
    if n == 0 {
        return Tensor::zeros((0,), DType::F32, logits.device());
    }
    let logp = log_softmax(&logits.to_dtype(DType::F32)?, 1)?; // (N, V)
    let labels = labels.to_dtype(DType::U32)?;
    // gather the log-prob of each row's true class, negate → per-row NLL.
    let picked = logp.gather(&labels.unsqueeze(1)?, 1)?.squeeze(1)?; // (N,)
    picked.neg()
}

/// `tensor.sum()` → scalar tensor, computed in f32. Kept as a `(1,)`-free scalar
/// so it stays in the autograd graph (the loss flows through `text_sum`/`audio_sum`).
fn sum_all_f32(t: &Tensor) -> Result<Tensor> {
    t.to_dtype(DType::F32)?.sum_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn lr_schedule_warmup_then_cosine() {
        let cfg = TrainerConfig { lr: 3e-5, warmup_steps: 100, max_steps: 1000, min_ratio: 0.1, ..Default::default() };
        // Build a Trainer shell without a model: exercise lr_at via a stub.
        let t = lr_only(cfg);
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

    #[test]
    fn log_codebook_weights_match_formula() {
        // log schedule: weights[i] = exp(linspace(1,0,C)[i] * ln(factor)).
        // i=0 → factor, i=C-1 → 1.0.
        let lc = LossConfig::new(4, "log", 8.0, 1.0, 1.0, &Device::Cpu).unwrap();
        let w: Vec<f32> = lc.audio_loss_weights.to_vec1().unwrap();
        assert!((w[0] - 8.0).abs() < 1e-4, "w[0] should be the factor, got {}", w[0]);
        assert!((w[3] - 1.0).abs() < 1e-4, "w[C-1] should be 1.0, got {}", w[3]);
    }

    #[test]
    fn linear_codebook_weights_scale_first() {
        let lc = LossConfig::new(3, "linear", 5.0, 1.0, 1.0, &Device::Cpu).unwrap();
        let w: Vec<f32> = lc.audio_loss_weights.to_vec1().unwrap();
        assert_eq!(w, vec![5.0, 1.0, 1.0]);
    }

    #[test]
    fn ce_none_matches_manual_nll() {
        // 2 rows, 3 classes; labels [0, 2].
        let logits = Tensor::from_vec(vec![2.0f32, 1.0, 0.1, 0.0, 0.0, 3.0], (2, 3), &Device::Cpu).unwrap();
        let labels = Tensor::from_vec(vec![0u32, 2u32], (2,), &Device::Cpu).unwrap();
        let loss: Vec<f32> = ce_none(&logits, &labels).unwrap().to_vec1().unwrap();
        // Row 0: -log softmax([2,1,0.1])[0]; Row 1: -log softmax([0,0,3])[2].
        let nll = |v: &[f32], k: usize| {
            let m = v.iter().cloned().fold(f32::MIN, f32::max);
            let denom: f32 = v.iter().map(|x| (x - m).exp()).sum();
            -((v[k] - m).exp() / denom).ln()
        };
        assert!((loss[0] - nll(&[2.0, 1.0, 0.1], 0)).abs() < 1e-5);
        assert!((loss[1] - nll(&[0.0, 0.0, 3.0], 2)).abs() < 1e-5);
    }

    #[test]
    fn ce_none_empty_is_empty() {
        let logits = Tensor::zeros((0, 5), DType::F32, &Device::Cpu).unwrap();
        let labels = Tensor::zeros((0,), DType::U32, &Device::Cpu).unwrap();
        assert_eq!(ce_none(&logits, &labels).unwrap().dim(0).unwrap(), 0);
    }

    // A minimal Trainer carrying only the schedule config, for `lr_at` unit tests
    // (the real constructor needs a checkpoint on disk).
    fn lr_only(cfg: TrainerConfig) -> LrOnly {
        LrOnly { cfg }
    }
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
