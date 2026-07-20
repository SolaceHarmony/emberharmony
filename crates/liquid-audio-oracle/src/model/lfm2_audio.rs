//! Differentiable Candle reference for offline LFM2 training.
//!
//! Production inference, generation, Conformer, and codec execution are native.
//! This module retains only teacher-forced backbone/depthformer loss math. Audio-in
//! batches fail explicitly until a native differentiable Conformer boundary exists.

use crate::model::lfm2_hf::{Cache as LfmCache, Lfm2Config, Model as Lfm2Model};
use crate::model::transformer::{HeadStyle, Mha, RawLmBackbone, SharedEmbedding, StandardBlock};
use crate::utils::{mel2emb_len, LFMModality};
use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// +1 over 2048 for the EOAudio token.
const AUDIO_VOCAB_SIZE: usize = 2048 + 1;

#[derive(Debug, Clone)]
pub struct DepthformerConfig {
    pub layers: usize,
    pub dim: usize,
    pub tie: bool,
}

/// Loss hyperparameters consumed by `LFM2AudioModel::new` to build the
/// `audio_loss_weights` buffer (Python `__init__` 104-113) and to stash the
/// loss multipliers (`self.conf.text_loss_multiplier` / `audio_loss_multiplier`).
/// Bundled into one struct to keep `new`'s signature clean; these are construction
/// inputs only and never affect any generation/forward computation path.
#[derive(Debug, Clone)]
pub struct LossConf {
    /// `Literal["log", "linear"]` — the per-codebook loss-weight schedule.
    pub codebook_weight: String,
    pub semantic_codebook_factor: f64,
    pub text_loss_multiplier: f64,
    pub audio_loss_multiplier: f64,
}

impl Default for LossConf {
    fn default() -> Self {
        // Mirrors the `LFM2AudioConfig` field defaults (`text/audio_loss_multiplier
        // = 1.0`) and the `from_pretrained` config fallbacks (`codebook_weight =
        // "linear"`, `semantic_codebook_factor = 1.0`).
        Self {
            codebook_weight: "linear".to_string(),
            semantic_codebook_factor: 1.0,
            text_loss_multiplier: 1.0,
            audio_loss_multiplier: 1.0,
        }
    }
}

/// `LFM2AudioModelOutput` — output of the **training** `forward` (cross-entropy
/// losses + token counts).
///
/// PORT: the training `forward(batch) -> LFM2AudioModelOutput` (see
/// [`LFM2AudioModel::forward`]) and its `logits(batch)` consume a
/// `LFM2AudioModelInput` training batch assembled by the `liquid_audio.data`
/// pipeline (`data/types.py`). The loss lives on the model (it holds the
/// `audio_loss_weights` buffer + loss multipliers built in `new`); the trainer
/// just calls `model.forward` — there is no separate trainer-side loss.
#[derive(Debug, Clone)]
pub struct LFM2AudioModelOutput {
    pub loss: Tensor,
    pub audio_loss: Tensor,
    pub text_loss: Tensor,
    pub audio_out_tokens: Tensor,
    pub text_tokens: Tensor,
    pub audio_in_tokens: Tensor,
}

// `LFM2AudioModelInput` (the batched training input) is defined in its Python home
// `data/types.py` → `crate::data::types`; re-export it here, where `logits`/`forward`
// consume it.
pub use crate::data::types::LFM2AudioModelInput;

// `nn.functional.cross_entropy(..., reduction="none")` lives in
// [`crate::candle_ext::loss::cross_entropy_none`] — the reduction candle's
// mean-only `cross_entropy` lacks. `forward` (below) calls it for the text/audio
// per-token NLL before the per-codebook weighting.
use crate::candle_ext::loss::cross_entropy_none;

pub struct LFM2AudioModel {
    lfm: Lfm2Model,
    lfm_cfg: Lfm2Config,
    audio_embedding: SharedEmbedding,
    depthformer: RawLmBackbone,
    depth_linear: Linear,
    depth_embeddings: Vec<SharedEmbedding>,
    codebooks: usize,
    codebook_offsets: Vec<i64>,
    depthformer_dim: usize,
    /// `audio_loss_weights` buffer (Python `__init__` 104-113): the per-codebook
    /// loss weighting, `(C,)`. Construction-only (not used by any generation path);
    /// consumed by the training `forward`.
    audio_loss_weights: Tensor,
    /// `self.conf.text_loss_multiplier` / `audio_loss_multiplier` — training-loss
    /// scalars (Python `LFM2AudioConfig`). Read only by `forward`.
    text_loss_multiplier: f64,
    audio_loss_multiplier: f64,
}

impl LFM2AudioModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lfm_cfg: Lfm2Config,
        depth_cfg: &DepthformerConfig,
        codebooks: usize,
        loss_conf: &LossConf,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = lfm_cfg.hidden_size;
        let lfm = Lfm2Model::new(&lfm_cfg, vb.pp("lfm"))?;
        let audio_embedding = SharedEmbedding::new(
            hidden,
            AUDIO_VOCAB_SIZE * codebooks,
            1e-5,
            vb.pp("audio_embedding"),
        )?;

        // Depthformer: RawLMBackbone(has_embedding=False) of StandardBlock(MHA(dim)).
        let df_vb = vb.pp("depthformer").pp("layers");
        let mut layers = Vec::with_capacity(depth_cfg.layers);
        for i in 0..depth_cfg.layers {
            let lvb = df_vb.pp(i.to_string());
            let mha = Mha::new(
                depth_cfg.dim,
                32,
                HeadStyle::Gqa,
                true,
                1e-5,
                8,
                128_000,
                1_000_000.0,
                lvb.pp("operator"),
            )?;
            let block = StandardBlock::new(mha, None, true, 256, 1.0, 1e-5, lvb)?;
            layers.push(block);
        }
        let depthformer = RawLmBackbone::new(layers, None, depth_cfg.dim);

        let depth_linear = linear(hidden, depth_cfg.dim * codebooks, vb.pp("depth_linear"))?;
        let de_vb = vb.pp("depth_embeddings");
        let mut depth_embeddings = Vec::with_capacity(codebooks);
        for i in 0..codebooks {
            depth_embeddings.push(SharedEmbedding::new(
                depth_cfg.dim,
                AUDIO_VOCAB_SIZE,
                1e-5,
                de_vb.pp(i.to_string()),
            )?);
        }

        let codebook_offsets = (0..codebooks as i64)
            .map(|i| i * AUDIO_VOCAB_SIZE as i64)
            .collect();

        // `audio_loss_weights` buffer — Python `__init__` 104-113:
        // ```python
        // if codebook_weight == "log":
        //     weights = (linspace(1, 0, C) * log(semantic_codebook_factor)).exp()
        // else:
        //     weights = ones(C); weights[0] *= semantic_codebook_factor
        // ```
        // A registered buffer loaded from / co-located with the checkpoint; built
        // here from config (additive, no effect on any forward/generation path).
        let dev = vb.device();
        let weights: Vec<f32> = if loss_conf.codebook_weight == "log" {
            let log_factor = loss_conf.semantic_codebook_factor.ln();
            (0..codebooks)
                .map(|i| {
                    // linspace(1, 0, C)[i] = 1 - i/(C-1)  (C==1 ⇒ the single point 1.0)
                    let t = if codebooks > 1 {
                        1.0 - i as f64 / (codebooks as f64 - 1.0)
                    } else {
                        1.0
                    };
                    (t * log_factor).exp() as f32
                })
                .collect()
        } else {
            let mut w = vec![1.0f32; codebooks];
            if let Some(first) = w.first_mut() {
                *first *= loss_conf.semantic_codebook_factor as f32;
            }
            w
        };
        let audio_loss_weights = Tensor::from_vec(weights, (codebooks,), dev)?;

        Ok(Self {
            lfm,
            lfm_cfg,
            audio_embedding,
            depthformer,
            depth_linear,
            depth_embeddings,
            codebooks,
            codebook_offsets,
            depthformer_dim: depth_cfg.dim,
            audio_loss_weights,
            text_loss_multiplier: loss_conf.text_loss_multiplier,
            audio_loss_multiplier: loss_conf.audio_loss_multiplier,
        })
    }

    /// Full causal backbone forward used by the teacher-forced loss path. of the `lfm` backbone over `embeds` (1,L,H),
    /// returning the normed all-position hidden state — for backbone parity.
    #[doc(hidden)]
    pub fn backbone_forward_embeds(&self, embeds: &Tensor) -> Result<Tensor> {
        let mut cache = LfmCache::new(true, embeds.dtype(), &self.lfm_cfg, embeds.device())?;
        self.lfm.forward_embeds(embeds, 0, &mut cache, None)
    }

    /// `logits(batch)` — training logits + labels, faithful to
    /// `LFM2AudioModel.logits`: `(text_logits (n_t,V), audio_logits (n_a·C,Va),
    /// text_labels (n_t,), audio_labels (n_a·C,))`. Teacher-forced; the depthformer
    /// runs the C-codebook sequence in parallel (causally masked — see
    /// transformer.rs). Reuses prefill / backbone / tied text head / depth
    /// embeddings, all parity-verified, so correctness is by composition.
    pub fn logits(&self, batch: &LFM2AudioModelInput) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let dev = batch.text.device();
        let in_emb = self.prefill_inputs(
            &batch.text,
            &batch.audio_in,
            &batch.audio_in_lens,
            &batch.audio_out,
            &batch.modality_flag,
        )?;
        let out_emb = self.backbone_forward_embeds(&in_emb)?; // (B, L, D)
        let (b, ll, d) = out_emb.dims3()?;
        // out_emb_shifted = out_emb[:, :-1] flattened to (B*(L-1), D): the Python
        // selects supervised rows with a 2-D boolean mask over the whole batch, so
        // the per-row shift drops each row's last step. For inference (B=1) this is
        // out_emb.i(0)[..L-1] as before.
        let out_emb_shifted = out_emb
            .narrow(1, 0, ll - 1)?
            .reshape((b * (ll - 1), d))?
            .contiguous()?; // (B*(L-1), D)

        // Read ids as i64 (torch.long) regardless of the input's int dtype — the
        // dataloader feeds I64, ChatState feeds U32; the cast handles both. Read the
        // FULL (B, L) modality/supervision, not row 0: the Python builds 2-D masks
        // over the whole batch. For inference (B=1) this is identical to row 0.
        let modality: Vec<Vec<i64>> = batch.modality_flag.to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let sup: Vec<Vec<u8>> = batch
            .supervision_mask
            .to_dtype(DType::U8)?
            .to_vec2::<u8>()?;
        let (text_id, audio_id) = (LFMModality::Text as i64, LFMModality::AudioOut as i64);

        // Supervised, non-first text / audio-out positions. Row index into the
        // flattened out_emb_shifted = bi*(L-1) + (p-1) (per-row shift `out_emb[:, :-1]`
        // paired with `mask[:, 1:]`); label index is the GLOBAL row-major counter into
        // the flat text (1,n_text) / audio_out (C, n_ao) token tensors.
        let (mut text_rows, mut text_lbl) = (Vec::<u32>::new(), Vec::<u32>::new());
        let (mut audio_rows, mut audio_lbl) = (Vec::<u32>::new(), Vec::<u32>::new());
        let (mut ti, mut ai) = (0u32, 0u32);
        for bi in 0..b {
            for p in 0..ll {
                if modality[bi][p] == text_id {
                    if p >= 1 && sup[bi][p] != 0 {
                        text_rows.push((bi * (ll - 1) + (p - 1)) as u32);
                        text_lbl.push(ti);
                    }
                    ti += 1;
                } else if modality[bi][p] == audio_id {
                    if p >= 1 && sup[bi][p] != 0 {
                        audio_rows.push((bi * (ll - 1) + (p - 1)) as u32);
                        audio_lbl.push(ai);
                    }
                    ai += 1;
                }
            }
        }

        // ---- text head (tied embedding): F.linear(text_out_emb, embed_tokens.weight) ----
        let ew = self.lfm.embed_weight().to_dtype(DType::F32)?; // (V, D)
        let vocab = ew.dim(0)?;
        let text_logits = if text_rows.is_empty() {
            Tensor::zeros((0, vocab), DType::F32, dev)?
        } else {
            let idx = Tensor::from_vec(text_rows.clone(), (text_rows.len(),), dev)?;
            let rows = out_emb_shifted
                .index_select(&idx, 0)?
                .to_dtype(DType::F32)?;
            rows.matmul(&ew.t()?.contiguous()?)?
        };
        let text_labels = {
            let t = batch.text.i(0)?;
            if text_lbl.is_empty() {
                Tensor::zeros((0,), t.dtype(), dev)?
            } else {
                t.index_select(
                    &Tensor::from_vec(text_lbl.clone(), (text_lbl.len(),), dev)?,
                    0,
                )?
            }
        };

        // ---- audio head (teacher-forced depthformer over the C codebooks) ----
        let (c, dd) = (self.codebooks, self.depthformer_dim);
        let (audio_logits, audio_labels) = if audio_rows.is_empty() {
            (
                Tensor::zeros((0, AUDIO_VOCAB_SIZE), DType::F32, dev)?,
                Tensor::zeros((0,), DType::U32, dev)?,
            )
        } else {
            let n_a = audio_rows.len();
            let aemb = out_emb_shifted
                .index_select(&Tensor::from_vec(audio_rows.clone(), (n_a,), dev)?, 0)?; // (n_a, D)
            let mut din = self.depth_linear.forward(&aemb)?.reshape((n_a, c, dd))?; // (n_a, C, dd)

            // teacher tokens: audio_out[:C, audio_lbl] → (C, n_a); per-codebook embed → (n_a, C, dd)
            let albl = Tensor::from_vec(audio_lbl.clone(), (audio_lbl.len(),), dev)?;
            let codes = batch
                .audio_out
                .narrow(0, 0, c)?
                .index_select(&albl, 1)?
                .to_dtype(DType::I64)?; // (C, n_a)
            let mut tok_rows = Vec::with_capacity(c);
            for ci in 0..c {
                tok_rows.push(
                    self.depth_embeddings[ci]
                        .embed(&codes.i(ci)?)?
                        .reshape((n_a, 1, dd))?,
                );
            }
            let dtok = Tensor::cat(&tok_rows.iter().collect::<Vec<_>>(), 1)?; // (n_a, C, dd)
                                                                              // dtok[:, -1] *= 0 ; roll(+1) along C → codebook c sees c-1's token, c0 sees zero.
            let zero_last = Tensor::zeros((n_a, 1, dd), dtok.dtype(), dev)?;
            let dtok = Tensor::cat(&[&dtok.narrow(1, 0, c - 1)?, &zero_last], 1)?;
            let dtok = Tensor::cat(&[&dtok.narrow(1, c - 1, 1)?, &dtok.narrow(1, 0, c - 1)?], 1)?;
            din = (din + dtok)?;

            // Split (Python 435-440): the depthformer cannot handle a batch dim
            // > 16k (= 2**14). When `n` along dim 0 reaches 2**14, run it on
            // `2**k` near-equal chunks (`torch.chunk`) and concat. For the parity
            // inputs `n < 16384` ⇒ `k = 0` ⇒ `num_chunks = 1` ⇒ a single call
            // identical to the unsplit forward, so this is a no-op for parity.
            let n = din.dim(0)?;
            let should_split = n >= 16384; // 2**14
            let k: i64 = if should_split {
                (n as f64).log2().floor() as i64 - 14 + 1
            } else {
                0
            };
            let num_chunks = 1usize << (k.max(0) as usize);
            let dout = if num_chunks <= 1 {
                self.depthformer.forward(&din, None)? // (n_a, C, dd), causally masked
            } else {
                // `torch.chunk(num_chunks)` along dim 0: ceil(n/num_chunks)-sized
                // pieces (the last may be smaller / chunks may be fewer). Run each
                // through the depthformer and concat back along dim 0.
                let chunk = n.div_ceil(num_chunks);
                let mut outs: Vec<Tensor> = Vec::new();
                let mut start = 0usize;
                while start < n {
                    let cur = chunk.min(n - start);
                    let part = din.narrow(0, start, cur)?;
                    outs.push(self.depthformer.forward(&part, None)?);
                    start += cur;
                }
                Tensor::cat(&outs.iter().collect::<Vec<_>>(), 0)?
            };
            let mut clog = Vec::with_capacity(c);
            for ci in 0..c {
                let logits_c =
                    self.depth_embeddings[ci].get_logits(&dout.narrow(1, ci, 1)?.squeeze(1)?)?; // (n_a, Va)
                clog.push(logits_c.unsqueeze(0)?);
            }
            let stacked = Tensor::cat(&clog.iter().collect::<Vec<_>>(), 0)?; // (C, n_a, Va)
            let va = stacked.dim(2)?;
            let audio_logits = stacked
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n_a * c, va))?; // (L C) V
            let audio_labels = codes
                .to_dtype(DType::U32)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n_a * c,))?; // (L C)
            (audio_logits, audio_labels)
        };

        Ok((text_logits, audio_logits, text_labels, audio_labels))
    }

    /// `forward(batch) -> LFM2AudioModelOutput` — the **training** cross-entropy
    /// loss (Python `LFM2AudioModel.forward`, 453-481). Faithful:
    /// ```python
    /// text_logits, audio_logits, text_labels, audio_labels = self.logits(batch)
    /// text_loss  = cross_entropy(text_logits, text_labels, ignore_index=-100, reduction="none")
    /// audio_loss = cross_entropy(audio_logits, audio_labels, ignore_index=-100, reduction="none")
    /// audio_loss = rearrange(audio_loss, "(L C) -> L C", C=codebooks)
    /// audio_loss = (audio_loss * audio_loss_weights).sum(-1) / audio_loss_weights.sum()
    /// text_tokens = text_loss.numel(); audio_tokens = audio_loss.numel()
    /// weighted_tokens = t_mult * text_tokens + a_mult * audio_tokens
    /// loss = (t_mult * text_loss.sum() + a_mult * audio_loss.sum()) / (weighted_tokens + 1e-6)
    /// ```
    /// Self-contained on the model (loss weights/multipliers are stored fields);
    /// [`crate::trainer::Trainer`] drives it via `train_step`/`validate`.
    pub fn forward(&self, batch: &LFM2AudioModelInput) -> Result<LFM2AudioModelOutput> {
        let (text_logits, audio_logits, text_labels, audio_labels) = self.logits(batch)?;
        let dev = text_logits.device();

        // cross_entropy(reduction="none"): per-row -log p(label). `logits`/`labels`
        // already cover only supervised positions, so `ignore_index=-100` is moot.
        let text_loss = cross_entropy_none(&text_logits, &text_labels)?; // (n_text,)
        let audio_loss_flat = cross_entropy_none(&audio_logits, &audio_labels)?; // (n_audio * C,)

        let c = self.codebooks;
        let n_audio = audio_loss_flat.dim(0)? / c.max(1);
        // rearrange "(L C) -> L C"; weight per-codebook then sum / weight-sum.
        let aw = self.audio_loss_weights.to_dtype(DType::F32)?; // (C,)
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
        let (tm, am) = (self.text_loss_multiplier, self.audio_loss_multiplier);
        let weighted_tokens = tm * text_tokens as f64 + am * audio_tokens as f64;

        let text_sum = text_loss.to_dtype(DType::F32)?.sum_all()?;
        let audio_sum = audio_loss.to_dtype(DType::F32)?.sum_all()?;
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

    /// Teacher-forced modality scatter for the training logits path.
    pub fn prefill_inputs(
        &self,
        text: &Tensor,
        audio_in: &Tensor,
        audio_in_lens: &Tensor,
        audio_out: &Tensor,
        modality_flag: &Tensor,
    ) -> Result<Tensor> {
        let dev = text.device();
        // Read ids as i64 (torch.long) regardless of input int dtype (I64 from the
        // dataloader. Reading it as u32 would silently lose valid I64 ids.
        //
        // Read the FULL (B, L) modality_flag, not row 0: the Python `_prefill` uses 2-D
        // boolean masks over the whole batch (`in_emb[modality==TEXT] = text_emb`), so
        // the flat text/audio embeddings scatter across all batch rows in row-major
        // order.
        let (b, ll) = modality_flag.dims2()?;
        let modality: Vec<i64> = modality_flag
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let l = modality.len(); // b*ll

        // text embeddings (n_text, D)
        let text_emb = self.lfm.embed(text)?.i(0)?; // (n_text, D)

        // Full Conformer training is intentionally unavailable: the corrected
        // implementation is native and exposes no differentiable backward ABI.
        // Silently dropping these rows would train a different model.
        let lens: Vec<i64> = audio_in_lens.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        if lens.iter().any(|&len| len != 0)
            || modality
                .iter()
                .any(|&kind| kind == LFMModality::AudioIn as i64)
        {
            return Err(candle_core::Error::Msg(
                "audio-in training requires a native differentiable Conformer backward path; \
                 the obsolete Rust Conformer is not retained"
                    .into(),
            ));
        }
        let _ = audio_in;
        let audio_in_emb: Option<Tensor> = None;

        // audio-out embeddings (n_ao, D)
        let audio_out_emb = {
            let m = audio_out.dim(1)?;
            if m == 0 {
                None
            } else {
                let codes = audio_out
                    .narrow(0, 0, self.codebooks)?
                    .to_dtype(DType::I64)?;
                let offs =
                    Tensor::from_vec(self.codebook_offsets.clone(), (self.codebooks, 1), dev)?;
                let offset_codes = codes.broadcast_add(&offs)?; // (codebooks, m)
                let emb = self.audio_embedding.embed(&offset_codes)?; // (codebooks, m, D)
                Some(emb.sum(0)?.to_dtype(text_emb.dtype())?) // (m, D)
            }
        };

        // combined = [text; audio_in; audio_out]; build index per position.
        let n_text = text_emb.dim(0)?;
        let n_ai = audio_in_emb
            .as_ref()
            .map(|a| a.dim(0).unwrap_or(0))
            .unwrap_or(0);
        let mut parts = vec![text_emb.clone()];
        if let Some(a) = &audio_in_emb {
            parts.push(a.clone());
        }
        if let Some(a) = &audio_out_emb {
            parts.push(a.clone());
        }
        let combined = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?; // (n_total, D)

        let (mut ct, mut cai, mut cao) = (0usize, 0usize, 0usize);
        let text_base = 0usize;
        let ai_base = n_text;
        let ao_base = n_text + n_ai;
        let mut index = Vec::with_capacity(l);
        for m in &modality {
            let idx = if *m == LFMModality::Text as i64 {
                let v = text_base + ct;
                ct += 1;
                v
            } else if *m == LFMModality::AudioIn as i64 {
                let v = ai_base + cai;
                cai += 1;
                v
            } else if *m == LFMModality::AudioOut as i64 {
                let v = ao_base + cao;
                cao += 1;
                v
            } else {
                // An unknown modality flag must error, not silently bucket as
                // AudioOut (the Python asserts the flag is one of the 3 modalities).
                return Err(candle_core::Error::Msg(format!(
                    "prefill: unknown modality flag {m} (expected 1/2/3)"
                )));
            };
            index.push(idx as u32);
        }
        // The scatter consumes exactly the rows of each part; a count mismatch
        // means a malformed modality_flag (mirrors the Python _prefill asserts).
        if ct != n_text
            || cai != n_ai
            || cao
                != audio_out_emb
                    .as_ref()
                    .map(|a| a.dim(0).unwrap_or(0))
                    .unwrap_or(0)
        {
            return Err(candle_core::Error::Msg(format!(
                "prefill: modality_flag counts (text {ct}, audio_in {cai}, audio_out {cao}) \
                 do not match inputs (text {n_text}, audio_in {n_ai})"
            )));
        }
        let index = Tensor::from_vec(index, (l,), dev)?;
        let in_emb = combined.index_select(&index, 0)?; // (B*L, D)
        let d = in_emb.dim(1)?;
        in_emb.reshape((b, ll, d)) // (B, L, D) — Python `text_emb.new_empty((B, L, D))`
    }
}
