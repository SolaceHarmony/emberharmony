//! Port of `liquid_audio/model/lfm2_audio.py` — `LFM2AudioModel` + generation.
//!
//! Assembly: HF `Lfm2Model` backbone (`lfm2_hf`) + FastConformer encoder +
//! audio-adapter MLP + audio-token `SharedEmbedding` + a depthformer
//! (`RawLmBackbone` of `StandardBlock(MHA)`) predicting the 8 Mimi codebooks per
//! audio frame. `generate_interleaved` is the streaming loop the usage example
//! drives; it is exposed here as a synchronous callback stream (faithful to the
//! Python generator — async lives only at the transport, per the design).
//!
//! Sampling: faithful to the upstream `_sample_text_token` / `_sample_audio_frame`
//! — greedy (argmax) when `temperature` is None/≤0 or `top_k == 1`, otherwise
//! `logits /= temperature`, top-k mask (keep ≥ the k-th largest, rest → -inf),
//! softmax, and `torch.multinomial`-equivalent draw via a seeded `StdRng`.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::model::conformer::encoder::{ConformerEncoder, ConformerEncoderConfig};
use crate::model::lfm2_hf::{Cache as LfmCache, Lfm2Config, Model as Lfm2Model};
use crate::model::linear::{linear_forward, linear_logits};
use crate::model::mlp::MLP;
use crate::model::transformer::{
    HeadStyle, LayerKvCache, Mha, RawLmBackbone, SharedEmbedding, StandardBlock,
};
use crate::processor::{ChatState, SpecialTokenIds};
use crate::utils::{mel2emb_len, LFMModality};

/// +1 over 2048 for the EOAudio token.
const AUDIO_VOCAB_SIZE: usize = 2048 + 1;
/// End-of-audio code: the per-codebook vocab's final entry (2048). Derived from
/// `AUDIO_VOCAB_SIZE`, which the checkpoint validates at load — `audio_embedding`
/// must have exactly `AUDIO_VOCAB_SIZE * codebooks` rows or `VarBuilder::get` fails.
const END_OF_AUDIO: u32 = (AUDIO_VOCAB_SIZE - 1) as u32;

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

/// `LFM2_HFConfig` — locates the HF backbone checkpoint (dataclass). The loader
/// resolves this into the concrete [`crate::model::lfm2_hf::Lfm2Config`].
#[allow(non_camel_case_types)] // mirror the Python class name exactly
#[derive(Debug, Clone)]
pub struct LFM2_HFConfig {
    pub pretrained_model_name_or_path: String,
    pub revision: Option<String>,
}

/// `LFM2AudioConfig` — the top-level model config (Python dataclass, parsed from
/// `config.json`). `loader.rs` reads the same JSON into the concrete sub-configs;
/// this is the faithful aggregate type for the 1:1 inventory.
#[derive(Debug, Clone)]
pub struct LFM2AudioConfig {
    pub architectures: Vec<String>,
    pub codebooks: usize,
    pub tie_audio_embeddings: bool,
    pub semantic_codebook_factor: f64,
    /// `Literal["log", "linear"]` — the per-codebook loss-weight schedule.
    pub codebook_weight: String,
    pub text_loss_multiplier: f64,
    pub audio_loss_multiplier: f64,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    pub preprocessor: crate::model::conformer::processor::MelConfig,
    pub encoder: crate::model::conformer::encoder::ConformerEncoderConfig,
    pub lfm: crate::model::lfm2_hf::Lfm2Config,
    pub depthformer: DepthformerConfig,
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

/// One streamed token: a text id, or one audio frame (codebooks codes).
#[derive(Debug, Clone)]
pub enum GenToken {
    Text(u32),
    Audio(Vec<u32>),
}

/// What a persistent cross-turn cache already contains (spec 09, W2a): the prefix of
/// the conversation context that has been forwarded through the backbone. Everything
/// past the cursor is the suffix [`LFM2AudioModel::prefill_suffix`] must build next
/// turn. A zero cursor means "nothing cached" — the suffix is the whole context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillCursor {
    /// Sequence positions forwarded (the cache's length / next `index_pos`).
    pub positions: usize,
    /// Text tokens consumed from `ChatState::text`.
    pub text: usize,
    /// Whole audio-in segments consumed (entries of `audio_in_lens`).
    pub audio_segments: usize,
    /// Audio-out frames consumed (columns of `ChatState::audio_out`).
    pub audio_out: usize,
}

/// Generation knobs — mirrors the kwargs of `generate_interleaved` /
/// `generate_sequential` in Python, plus a `seed` for the multinomial RNG
/// (Python relies on the global `torch` generator; we make it explicit and
/// reproducible). All `None` (the default) ⇒ greedy, matching the Python.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new_tokens: usize,
    pub text_temperature: Option<f64>,
    pub text_top_k: Option<usize>,
    pub audio_temperature: Option<f64>,
    pub audio_top_k: Option<usize>,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 20, // Python default
            text_temperature: None,
            text_top_k: None,
            audio_temperature: None,
            audio_top_k: None,
            seed: 42,
        }
    }
}

impl GenParams {
    /// The HF getting-started defaults for the interleaved demo: text greedy (no temperature
    /// or top-k), audio sampled at `temperature=1.0`/`top_k=4`, `max_new_tokens=1024`, fixed
    /// seed. Mirrors the README two-turn example's `generate_interleaved(..., max_new_tokens=512,
    /// audio_temperature=1.0, audio_top_k=4)` — but with the demo UI's 1024 interleaved budget.
    ///
    /// **Audio is sampled, not greedy, deliberately:** greedy audio (audio temp/top-k `None`)
    /// is degenerate for the Depthformer — the model is trained for sampled audio, so greedy
    /// produces an unintelligible reply. `Default` stays greedy (the Python kwargs default);
    /// this is the realtime/voice-path default.
    pub fn demo_defaults() -> Self {
        Self {
            max_new_tokens: 1024,
            text_temperature: None,
            text_top_k: None,
            audio_temperature: Some(1.0),
            audio_top_k: Some(4),
            seed: 0,
        }
    }
}

/// `Sampler` — the next-token sampler, built on `candle_transformers`'
/// [`LogitsProcessor`] (the same sampler `moshi` uses for depformer decoding)
/// rather than a private softmax+multinomial. Faithful to `_sample_text_token` and
/// the per-codebook step of `_sample_audio_frame`:
/// ```python
/// greedy = temperature is None or temperature <= 0 or top_k == 1
/// if greedy: next = logits.argmax()
/// else:
///     logits /= temperature
///     if top_k is not None:
///         min_score = torch.topk(logits, top_k).values[-1]
///         logits[logits < min_score] = -inf       # threshold-style: ties kept
///     next = torch.multinomial(logits.softmax(0), 1)
/// ```
/// Greedy ⇒ [`Sampling::ArgMax`]; `LogitsProcessor::sample_argmax` is
/// `logits.argmax(-1)`, byte-identical to the previous greedy path, so generation
/// parity (incl. the token-exact depthformer) is preserved. Stochastic ⇒
/// [`Sampling::All`] (temperature softmax + multinomial), with Torch's *threshold*
/// top-k injected through [`LogitsProcessor::sample_f`]: candle's built-in
/// `Sampling::TopK` keeps exactly `k` tokens, whereas Torch keeps every token `≥`
/// the k-th largest (ties included), so the mask is applied via the `sample_f`
/// extension hook rather than forking the sampler.
struct Sampler {
    processor: LogitsProcessor,
    /// Torch threshold top-k bound, applied on the stochastic path only.
    top_k: Option<usize>,
    greedy: bool,
}

impl Sampler {
    fn new(seed: u64, temperature: Option<f64>, top_k: Option<usize>) -> Self {
        let greedy = match (temperature, top_k) {
            (None, _) => true,
            (Some(t), _) if t <= 0.0 => true,
            (_, Some(1)) => true,
            _ => false,
        };
        let sampling = if greedy {
            Sampling::ArgMax
        } else {
            // non-greedy ⇒ temperature is Some(>0).
            Sampling::All {
                temperature: temperature.expect("non-greedy ⇒ temperature is Some(>0)"),
            }
        };
        Self {
            processor: LogitsProcessor::from_sampling(seed, sampling),
            top_k,
            greedy,
        }
    }

    /// Sample one token from 1-D `logits` (`V,`).
    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        match (self.greedy, self.top_k) {
            // Stochastic + top-k: inject Torch's threshold mask via the sample_f hook.
            (false, Some(k)) => self
                .processor
                .sample_f(logits, move |prs| torch_topk_mask(prs, k)),
            // Greedy (argmax), or stochastic without top-k (plain multinomial).
            _ => self.processor.sample(logits),
        }
    }
}

/// Torch `topk(logits, k).values[-1]` threshold applied to the probability vector:
/// zero every probability below the k-th largest. softmax is monotonic, so the
/// logit threshold and the probability threshold select the same tokens; ties at
/// the boundary are *kept*, matching `logits[logits < min_score] = -inf`. The kept
/// probabilities need no renormalization — multinomial (`WeightedIndex`) samples
/// proportionally, so the resulting distribution equals Python's softmax-after-mask.
fn torch_topk_mask(prs: &mut [f32], k: usize) {
    let k = k.min(prs.len());
    if k == 0 {
        return;
    }
    let mut sorted: Vec<f32> = prs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let min_score = sorted[k - 1];
    for p in prs.iter_mut() {
        if *p < min_score {
            *p = 0.0;
        }
    }
}

// `nn.functional.cross_entropy(..., reduction="none")` lives in
// [`crate::candle_ext::loss::cross_entropy_none`] — the reduction candle's
// mean-only `cross_entropy` lacks. `forward` (below) calls it for the text/audio
// per-token NLL before the per-codebook weighting.
use crate::candle_ext::loss::cross_entropy_none;
use crate::weights::{NativeWeightImage, ResidentWeights};

pub struct LFM2AudioModel {
    lfm: Lfm2Model,
    lfm_cfg: Lfm2Config,
    conformer: ConformerEncoder,
    audio_adapter: MLP,
    audio_embedding: SharedEmbedding,
    depthformer: RawLmBackbone,
    depth_linear: Linear,
    depth_embeddings: Vec<SharedEmbedding>,
    codebooks: usize,
    codebook_offsets: Vec<i64>,
    depthformer_dim: usize,
    interleaved_n_text: usize,
    interleaved_n_audio: usize,
    /// Generation-control token ids resolved from the model's own tokenizer at load
    /// (never literals — the model defines them). See `SpecialTokenIds`.
    special: SpecialTokenIds,
    hidden: usize,
    /// `audio_loss_weights` buffer (Python `__init__` 104-113): the per-codebook
    /// loss weighting, `(C,)`. Construction-only (not used by any generation path);
    /// consumed by the training `forward`.
    audio_loss_weights: Tensor,
    /// `self.conf.text_loss_multiplier` / `audio_loss_multiplier` — training-loss
    /// scalars (Python `LFM2AudioConfig`). Read only by `forward`.
    text_loss_multiplier: f64,
    audio_loss_multiplier: f64,
    /// Pure-NEON depthformer frame decoder (flashkern): zero-copy views of the depthformer
    /// weights + persistent scratch, built once at load when the CPU kernels are available.
    /// `None` ⇒ the candle op-chain path runs (also the parity reference).
    depth_flash: Option<crate::flashkern::decode::DepthDecode>,
    /// Byte-parity reference chain (DECODE_ENGINE.md §5): pins every ulp-tier decode
    /// deviation off — `grouped_gqa_decode=false` on each internally-built cache and
    /// depth-flash disabled — so greedy text + seeded audio reproduces the recorded
    /// wav-hash baseline bit-for-bit. Token-exact tiers (fused conv/MLP) stay on.
    reference_numerics: bool,
    /// Owns the canonical checkpoint image for native binders. Candle-backed
    /// components still own their measured compatibility copies. `None` is
    /// reserved for trainable/tests.
    resident: Option<ResidentWeights>,
}

impl LFM2AudioModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lfm_cfg: Lfm2Config,
        enc_cfg: &ConformerEncoderConfig,
        depth_cfg: &DepthformerConfig,
        codebooks: usize,
        interleaved_n_text: usize,
        interleaved_n_audio: usize,
        special: SpecialTokenIds,
        loss_conf: &LossConf,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::build(
            lfm_cfg,
            enc_cfg,
            depth_cfg,
            codebooks,
            interleaved_n_text,
            interleaved_n_audio,
            special,
            loss_conf,
            vb,
            None,
        )
    }

    /// Production constructor: the native image is the checkpoint owner and
    /// the Candle builder is only a measured compatibility adapter over it.
    #[allow(clippy::too_many_arguments)]
    pub fn new_resident(
        lfm_cfg: Lfm2Config,
        enc_cfg: &ConformerEncoderConfig,
        depth_cfg: &DepthformerConfig,
        codebooks: usize,
        interleaved_n_text: usize,
        interleaved_n_audio: usize,
        special: SpecialTokenIds,
        loss_conf: &LossConf,
        resident: ResidentWeights,
        device: &Device,
    ) -> Result<Self> {
        let vb = resident.candle_builder(device);
        Self::build(
            lfm_cfg,
            enc_cfg,
            depth_cfg,
            codebooks,
            interleaved_n_text,
            interleaved_n_audio,
            special,
            loss_conf,
            vb,
            Some(resident),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        lfm_cfg: Lfm2Config,
        enc_cfg: &ConformerEncoderConfig,
        depth_cfg: &DepthformerConfig,
        codebooks: usize,
        interleaved_n_text: usize,
        interleaved_n_audio: usize,
        special: SpecialTokenIds,
        loss_conf: &LossConf,
        vb: VarBuilder,
        resident: Option<ResidentWeights>,
    ) -> Result<Self> {
        let hidden = lfm_cfg.hidden_size;
        let lfm = Lfm2Model::new(&lfm_cfg, vb.pp("lfm"))?;
        let conformer = ConformerEncoder::new(enc_cfg, vb.pp("conformer"))?;
        let feat_out = if enc_cfg.feat_out > 0 && enc_cfg.feat_out != enc_cfg.d_model {
            enc_cfg.feat_out
        } else {
            enc_cfg.d_model
        };
        let audio_adapter = MLP::new(
            feat_out,
            hidden,
            &[hidden],
            true,
            true,
            0.0,
            vb.pp("audio_adapter"),
        )?;
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

        let mut model = Self {
            lfm,
            lfm_cfg,
            conformer,
            audio_adapter,
            audio_embedding,
            depthformer,
            depth_linear,
            depth_embeddings,
            codebooks,
            codebook_offsets,
            depthformer_dim: depth_cfg.dim,
            interleaved_n_text,
            interleaved_n_audio,
            special,
            hidden,
            audio_loss_weights,
            text_loss_multiplier: loss_conf.text_loss_multiplier,
            audio_loss_multiplier: loss_conf.audio_loss_multiplier,
            depth_flash: None,
            reference_numerics: false,
            resident,
        };
        // Built AFTER assembly: the ctx captures raw views into tensors the model now owns
        // (Arc-heap storages — stable across moves of `model`).
        model.depth_flash = model.build_depth_flash();
        if model.depth_flash.is_some() {
            eprintln!("[voice] flashkern depthformer decoder active (pure-NEON audio frames)");
        }
        // Resident native-engine layer table (same capture contract as depth_flash:
        // Arc-heap storages, guard clears before the weights drop).
        model
            .lfm
            .install_native_ctx(model.lfm_cfg.max_position_embeddings);
        model.install_native_heads();
        Ok(model)
    }

    pub fn resident_weights(&self) -> Option<&NativeWeightImage> {
        self.resident.as_ref().map(ResidentWeights::image)
    }

    pub fn compatibility_copies(&self) -> crate::weights::CompatibilityCopies {
        self.resident
            .as_ref()
            .map(ResidentWeights::compatibility_copies)
            .unwrap_or_default()
    }

    /// Capture the depthformer as a flashkern [`DepthDecode`] — every weight a zero-copy
    /// bf16 view in checkpoint layout. Any non-conforming tensor (wrong device/dtype/
    /// layout, non-swiglu Glu, missing qk-norms) ⇒ `None`, and the candle path runs.
    fn build_depth_flash(&self) -> Option<crate::flashkern::decode::DepthDecode> {
        use crate::flashkern::decode::{DepthDecode, DepthHead, DepthLayer, PtrLen};
        if !crate::bf16_gemm::bf16_gemm_nt_available() {
            return None;
        }
        // Depth-flash rides the native engine's lane team. The engine is the
        // SUBSTRATE — process_engine() panics at init if it can't stand up, so
        // there is no "engine absent" branch here; `None` below means only
        // tensor nonconformance (wrong dtype/layout), never a missing runtime.
        let mut layers = Vec::with_capacity(self.depthformer.layers.len());
        let mut geom = None;
        let mut cos_sin = None;
        for blk in &self.depthformer.layers {
            let (mha, glu, opn, ffnn) = blk.flash_parts();
            let (qkv, out, ba, cos, sin) = mha.flash_parts();
            let (heads, kvh, hd, qln, kln) = ba.flash_parts();
            let (w1, w2, w3, swiglu) = glu.flash_parts();
            if !swiglu || heads * hd != self.depthformer_dim {
                return None;
            }
            let w3 = w3?;
            geom = Some((heads, kvh, hd, w1.weight().dim(0).ok()?, opn.eps() as f32));
            cos_sin = Some((PtrLen::f32(cos)?, PtrLen::f32(sin)?));
            layers.push(DepthLayer {
                qkv_w: PtrLen::bf16(qkv.weight())?,
                out_w: PtrLen::bf16(out.weight())?,
                q_ln: PtrLen::bf16(qln?.weight())?,
                k_ln: PtrLen::bf16(kln?.weight())?,
                opnorm: PtrLen::bf16(opn.weight())?,
                ffnnorm: PtrLen::bf16(ffnn.weight())?,
                w1: PtrLen::bf16(w1.weight())?,
                w3: PtrLen::bf16(w3.weight())?,
                w2: PtrLen::bf16(w2.weight())?,
            });
        }
        let (heads, kvh, _hd, ff, eps) = geom?;
        let (cos, sin) = cos_sin?;
        let mut heads_w = Vec::with_capacity(self.codebooks);
        for se in &self.depth_embeddings {
            let (emb, norm, logits) = se.flash_parts();
            let vocab = emb.dim(0).ok()?;
            if logits.dim(0).ok()? != vocab || emb.dim(1).ok()? != self.depthformer_dim {
                return None;
            }
            heads_w.push(DepthHead {
                emb: PtrLen::bf16(emb)?,
                norm: PtrLen::bf16(norm.weight())?,
                logits: PtrLen::bf16(logits)?,
                vocab,
            });
        }
        Some(DepthDecode::new(
            self.depthformer_dim,
            heads,
            kvh,
            ff,
            self.codebooks,
            self.hidden,
            eps,
            layers,
            heads_w,
            PtrLen::bf16(self.depth_linear.weight())?,
            PtrLen::bf16(self.depth_linear.bias()?)?,
            cos,
            sin,
        ))
    }

    /// Test/parity seam: disable the flashkern depthformer so the candle op chain runs.
    /// Select the byte-parity reference chain (see the `reference_numerics` field).
    pub fn set_reference_numerics(&mut self, on: bool) {
        self.reference_numerics = on;
        self.set_depth_flash_enabled(!on);
    }

    /// Internally-built caches route through this so the reference chain pins the
    /// ulp-tier flags in ONE place.
    fn new_cache(&self, dtype: DType, device: &Device) -> Result<LfmCache> {
        let mut cache = LfmCache::new(true, dtype, &self.lfm_cfg, device)?;
        if self.reference_numerics {
            cache.grouped_gqa_decode = false;
        }
        Ok(cache)
    }

    pub fn set_depth_flash_enabled(&mut self, on: bool) {
        if !on {
            self.depth_flash = None;
        } else if self.depth_flash.is_none() {
            self.depth_flash = self.build_depth_flash();
        }
    }

    /// `from_pretrained(dir, device)` — load the model + processor from a
    /// local model directory (Python `LFM2AudioModel.from_pretrained`, 135-169).
    /// A thin delegation to [`crate::loader::from_pretrained`], which parses
    /// `config.json` (including `codebook_weight` / `semantic_codebook_factor` /
    /// `text_loss_multiplier` / `audio_loss_multiplier`), memory-maps the
    /// safetensors, and constructs both the model and its [`LFM2AudioProcessor`]
    /// (Python returns just the model; the processor is loaded alongside here, as
    /// the rest of this crate's entry points do). No loader logic is duplicated.
    pub fn from_pretrained(
        dir: &std::path::Path,
        device: &candle_core::Device,
    ) -> Result<(Self, crate::processor::LFM2AudioProcessor)> {
        crate::loader::from_pretrained(dir, device)
    }

    /// Run the FastConformer encoder over mel features `(B, feat_in, T)` →
    /// `(B, d, T')`. Exposed for parity testing.
    pub fn conformer_encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.conformer.forward(mel)
    }

    /// The `ConformerEncoder` (Python `self.conformer`). Read access for the
    /// streaming/export inventory methods (`get_initial_cache_state`,
    /// `streaming_post_process`, `input_example`, …).
    pub fn conformer(&self) -> &crate::model::conformer::encoder::ConformerEncoder {
        &self.conformer
    }

    /// Debug: conformer stage intermediates for parity localization.
    #[doc(hidden)]
    pub fn conformer_stages(
        &self,
        mel: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.conformer.forward_stages(mel)
    }

    /// Debug: conformer subsampling conv-stack output (pre flatten+linear).
    #[doc(hidden)]
    pub fn conformer_sub_conv(&self, mel: &Tensor) -> Result<Tensor> {
        self.conformer.subsampling_conv_out(mel)
    }

    /// Debug: full causal forward of the `lfm` backbone over `embeds` (1,L,H),
    /// returning the normed all-position hidden state — for backbone parity.
    #[doc(hidden)]
    pub fn backbone_forward_embeds(&self, embeds: &Tensor) -> Result<Tensor> {
        let mut cache = self.new_cache(embeds.dtype(), embeds.device())?;
        self.lfm.forward_embeds(embeds, 0, &mut cache, None)
    }

    /// Debug: prefill input embeddings for a `ChatState` (the modality-scatter
    /// assembly: text embed + conformer/adapter audio-in + audio-out embedding).
    #[doc(hidden)]
    pub fn prefill_chat(&self, chat: &ChatState) -> Result<Tensor> {
        self.prefill(chat)
    }

    /// Debug: tied-embedding text logits for a single hidden vector (H,) — the
    /// text head used in generation.
    #[doc(hidden)]
    pub fn text_logits_of(&self, hidden_last: &Tensor) -> Result<Tensor> {
        self.text_logits(hidden_last)
    }

    /// One backbone forward over `in_emb` at `index_pos` through an external cache,
    /// no sampling. Production use: the engine's speculative prefill (forward the
    /// utterance suffix during the VAD pause window, roll back on false pause);
    /// also lets tests assert chunked-continuation forward equals one full forward
    /// (the numerical contract behind the persistent cross-turn cache, spec 09 W2a).
    pub fn forward_embeds(
        &self,
        in_emb: &Tensor,
        index_pos: usize,
        cache: &mut LfmCache,
    ) -> Result<Tensor> {
        self.lfm.forward_embeds(in_emb, index_pos, cache, None)
    }

    /// Debug: greedy depthformer audio frame (8 codebook tokens) for a fixed
    /// `embedding` (H,) — for depthformer parity (token-exact vs Python greedy).
    #[doc(hidden)]
    pub fn audio_frame_greedy(&self, embedding: &Tensor) -> Result<Vec<u32>> {
        let mut sampler = Sampler::new(0, None, None); // greedy ⇒ argmax (seed unused)
        self.sample_audio_frame(embedding, &mut sampler)
    }

    /// Build the prefill input embeddings, scattering text / audio-in / audio-out
    /// embeddings into sequence order by `modality_flag` (index_select instead of
    /// PyTorch boolean assignment).
    fn prefill(&self, chat: &ChatState) -> Result<Tensor> {
        self.prefill_inputs(
            &chat.text,
            &chat.audio_in,
            &chat.audio_in_lens,
            &chat.audio_out,
            &chat.modality_flag,
        )
    }

    /// A fresh backbone KV/conv cache for the externally-owned generation path
    /// ([`Self::generate_with_cache`]). The engine keeps this alive across turns
    /// (spec 09, W2a) so each turn only forwards the context *suffix* it has not
    /// seen yet, instead of re-prefilling the whole conversation.
    pub fn make_cache(&self, dtype: DType, device: &Device) -> Result<LfmCache> {
        self.new_cache(dtype, device)
    }

    /// Build prefill embeddings for the context SUFFIX past `cursor` — the positions a
    /// persistent cross-turn cache has not forwarded yet. `cursor` counts what the cache
    /// already contains: sequence positions, text tokens, whole audio-in segments (a
    /// segment is added atomically with its turn, so it is never split), and audio-out
    /// frames. With a zero cursor this is exactly [`prefill_chat`](Self::prefill_chat).
    ///
    /// Hard-errors if the cursor's per-modality counts do not match the prefix
    /// `modality_flag[..cursor.positions]` — a desynced cursor must invalidate the cache
    /// (full re-prefill), never silently misalign the scatter.
    pub fn prefill_suffix(&self, chat: &ChatState, cursor: &PrefillCursor) -> Result<Tensor> {
        let dev = chat.text.device();
        let modality: Vec<i64> = chat
            .modality_flag
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let n = modality.len();
        if cursor.positions > n {
            return Err(candle_core::Error::Msg(format!(
                "prefill_suffix: cursor positions {} beyond context length {n}",
                cursor.positions
            )));
        }

        // Verify the cursor against the actual prefix (cheap CPU walk) — the cache and
        // the context tensors must describe the same prefix or continuation is garbage.
        let (mut pt, mut pai, mut pao) = (0usize, 0usize, 0usize);
        for m in &modality[..cursor.positions] {
            if *m == LFMModality::Text as i64 {
                pt += 1;
            } else if *m == LFMModality::AudioIn as i64 {
                pai += 1;
            } else if *m == LFMModality::AudioOut as i64 {
                pao += 1;
            }
        }
        let lens: Vec<i64> = chat.audio_in_lens.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        if cursor.audio_segments > lens.len() {
            return Err(candle_core::Error::Msg(format!(
                "prefill_suffix: cursor segments {} beyond {} audio-in segments",
                cursor.audio_segments,
                lens.len()
            )));
        }
        // Cached audio-in positions must cover whole segments: the conformer output
        // length per segment is what occupies AudioIn positions.
        if pt != cursor.text || pao != cursor.audio_out {
            return Err(candle_core::Error::Msg(format!(
                "prefill_suffix: cursor (text {}, audio_out {}) does not match prefix \
                 counts (text {pt}, audio_out {pao})",
                cursor.text, cursor.audio_out
            )));
        }

        // text suffix embeddings
        let n_text_total = chat.text.dim(1)?;
        let text_suffix = chat
            .text
            .narrow(1, cursor.text, n_text_total - cursor.text)?;
        let text_emb = self.lfm.embed(&text_suffix)?.i(0)?; // (n_text_suffix, D)

        // audio-in: run the conformer ONLY over segments the cache has not seen.
        let mut frame_cursor: usize = lens[..cursor.audio_segments]
            .iter()
            .map(|&l| l as usize)
            .sum();
        let mut audio_in_rows: Vec<Tensor> = Vec::new();
        for &len in &lens[cursor.audio_segments..] {
            let seg = chat.audio_in.narrow(1, frame_cursor, len as usize)?;
            frame_cursor += len as usize;
            let seg = seg.unsqueeze(0)?.to_dtype(text_emb.dtype())?;
            let enc = self.conformer.forward(&seg)?;
            let enc = enc.i(0)?.transpose(0, 1)?.contiguous()?;
            let adapted = self.audio_adapter.forward(&enc)?;
            audio_in_rows.push(adapted);
        }
        let audio_in_emb = if audio_in_rows.is_empty() {
            None
        } else {
            Some(Tensor::cat(&audio_in_rows.iter().collect::<Vec<_>>(), 0)?)
        };
        // The prefix AudioIn positions must equal the summed conformer output lengths of
        // the cached segments — the same subsampling arithmetic `add_audio` used to lay
        // down the modality flags. A mismatch means cursor/context desync.
        let cached_emb_len: usize = lens[..cursor.audio_segments]
            .iter()
            .map(|&l| mel2emb_len(l) as usize)
            .sum();
        if pai != cached_emb_len {
            return Err(candle_core::Error::Msg(format!(
                "prefill_suffix: prefix AudioIn positions {pai} do not match cached \
                 segments' embed length {cached_emb_len} ({} segments)",
                cursor.audio_segments
            )));
        }

        // audio-out suffix embeddings
        let m_total = chat.audio_out.dim(1)?;
        let audio_out_emb = {
            let m = m_total - cursor.audio_out;
            if m == 0 {
                None
            } else {
                let codes = chat
                    .audio_out
                    .narrow(0, 0, self.codebooks)?
                    .narrow(1, cursor.audio_out, m)?
                    .to_dtype(DType::I64)?;
                let offs =
                    Tensor::from_vec(self.codebook_offsets.clone(), (self.codebooks, 1), dev)?;
                let offset_codes = codes.broadcast_add(&offs)?;
                let emb = self.audio_embedding.embed(&offset_codes)?;
                Some(emb.sum(0)?.to_dtype(text_emb.dtype())?)
            }
        };

        // Scatter the suffix parts into sequence order by the suffix modality flags.
        let n_text = text_emb.dim(0)?;
        let n_ai = audio_in_emb
            .as_ref()
            .map(|a| a.dim(0).unwrap_or(0))
            .unwrap_or(0);
        let n_ao = audio_out_emb
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
        let combined = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?;

        let (mut ct, mut cai, mut cao) = (0usize, 0usize, 0usize);
        let ai_base = n_text;
        let ao_base = n_text + n_ai;
        let suffix = &modality[cursor.positions..];
        let mut index = Vec::with_capacity(suffix.len());
        for m in suffix {
            let idx = if *m == LFMModality::Text as i64 {
                let v = ct;
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
                return Err(candle_core::Error::Msg(format!(
                    "prefill_suffix: unknown modality flag {m} (expected 1/2/3)"
                )));
            };
            index.push(idx as u32);
        }
        if ct != n_text || cai != n_ai || cao != n_ao {
            return Err(candle_core::Error::Msg(format!(
                "prefill_suffix: suffix modality counts (text {ct}, audio_in {cai}, \
                 audio_out {cao}) do not match suffix inputs (text {n_text}, audio_in \
                 {n_ai}, audio_out {n_ao})"
            )));
        }
        let index = Tensor::from_vec(index, (suffix.len(),), dev)?;
        combined.index_select(&index, 0)?.unsqueeze(0) // (1, n_suffix, D)
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
        let ew = self.lfm.embed_weight(); // (V, D)
        let vocab = ew.dim(0)?;
        let text_logits = if text_rows.is_empty() {
            Tensor::zeros((0, vocab), DType::F32, dev)?
        } else {
            let idx = Tensor::from_vec(text_rows.clone(), (text_rows.len(),), dev)?;
            let rows = out_emb_shifted.index_select(&idx, 0)?;
            linear_logits(ew, &rows)?
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
            let mut din = linear_forward(&self.depth_linear, &aemb)?.reshape((n_a, c, dd))?; // (n_a, C, dd)

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

    /// `_sample_text_token(logits, *, temperature, top_k)` (Python 483-499) — the
    /// text head's next-token draw. The temperature/top-k policy is held by the
    /// [`Sampler`] (built once from `GenParams`), so this just delegates to it.
    fn sample_text_token(&self, logits: &Tensor, sampler: &mut Sampler) -> Result<u32> {
        sampler.sample(logits)
    }

    /// `_prefill(text, audio_in, audio_in_lens, audio_out, modality_flag)` from
    /// raw fields — the modality-scatter shared by `prefill_chat` (inference) and
    /// `logits` (training). Faithful to Python `LFM2AudioModel._prefill`.
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
        // dataloader, U32 from ChatState). NB: reading an I64 tensor as u32 here would
        // silently return an empty `lens` (`unwrap_or_default`) and drop audio-in.
        //
        // Read the FULL (B, L) modality_flag, not row 0: the Python `_prefill` uses 2-D
        // boolean masks over the whole batch (`in_emb[modality==TEXT] = text_emb`), so
        // the flat text/audio embeddings scatter across all batch rows in row-major
        // order. For inference (B=1) this is identical to row 0.
        let (b, ll) = modality_flag.dims2()?;
        let modality: Vec<i64> = modality_flag
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let l = modality.len(); // b*ll

        // text embeddings (n_text, D)
        let text_emb = self.lfm.embed(text)?.i(0)?; // (n_text, D)

        // audio-in embeddings (n_ai, D): encode each segment, adapt, concat.
        // PROPAGATE a malformed-lens error rather than silently dropping ALL audio-in:
        // the legitimate no-audio / text-only case is a 1-D `(0,)` tensor whose
        // `to_vec1` is `Ok([])`, so `?` only fires on a genuinely malformed lens
        // (wrong rank/dtype). The old `unwrap_or_default()` here swallowed those into
        // an empty `lens`, which would scatter zero audio-in and then trip the count
        // check below with a confusing message instead of the real cause.
        let lens: Vec<i64> = audio_in_lens.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let mut audio_in_rows: Vec<Tensor> = Vec::new();
        let mut frame_cursor = 0usize;
        for &len in &lens {
            let seg = audio_in.narrow(1, frame_cursor, len as usize)?; // (128, frames)
            frame_cursor += len as usize;
            // Mel is built in f32 (the NeMo preprocessor runs in f32); cast to the
            // model dtype before the conformer, matching Python's
            // `self.conformer(padded_audio_in.mT.to(text_emb.dtype), …)`. No-op on
            // the f32 CPU parity path; on bf16 (Metal) it avoids a conv dtype clash.
            let seg = seg.unsqueeze(0)?.to_dtype(text_emb.dtype())?; // (1, 128, frames)
            let enc = self.conformer.forward(&seg)?; // (1, d, T')
            let enc = enc.i(0)?.transpose(0, 1)?.contiguous()?; // (T', d)
            let adapted = self.audio_adapter.forward(&enc)?; // (T', hidden)
            audio_in_rows.push(adapted);
        }
        let audio_in_emb = if audio_in_rows.is_empty() {
            None
        } else {
            Some(Tensor::cat(&audio_in_rows.iter().collect::<Vec<_>>(), 0)?)
        };

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

    fn text_logits(&self, h_last: &Tensor) -> Result<Tensor> {
        // nn.functional.linear(h, embed_weight): (V,D) @ (D,) -> (V,)
        let h = h_last.reshape((1, self.hidden))?;
        linear_logits(self.lfm.embed_weight(), &h)?.squeeze(0)
    }

    /// Depthformer audio-frame sampler → `codebooks` codes. Faithful to
    /// `_sample_audio_frame`: per-codebook draw via the audio [`Sampler`]
    /// (greedy/temperature/top-k held by the sampler).
    fn sample_audio_frame(&self, embedding: &Tensor, sampler: &mut Sampler) -> Result<Vec<u32>> {
        // Pure-NEON path: the whole frame (8 codebook steps × 6 blocks) as one flashkern
        // dispatch — no candle op runs. Same rounding ladder; the seeded Sampler is shared,
        // so the RNG stream matches the candle path.
        if let Some(ctx) = &self.depth_flash {
            if embedding.device().is_cpu() && embedding.dtype() == DType::BF16 {
                return self.sample_audio_frame_flash(ctx, embedding, sampler);
            }
        }
        // depth_linear(embedding) → (codebooks, depthformer_dim). `embedding` is a
        // 1-D (D,) lfm hidden; candle's Linear needs a 2-D input, so add a row dim
        // (Python's nn.Linear accepts the 1-D vector directly).
        let emb2d = embedding.flatten_all()?.unsqueeze(0)?; // (1, D)
        let din = linear_forward(&self.depth_linear, &emb2d)?
            .reshape((self.codebooks, self.depthformer_dim))?;
        let mut df_token = Tensor::zeros((self.depthformer_dim,), din.dtype(), din.device())?;
        let mut caches: Vec<LayerKvCache> = (0..self.depthformer.layers.len())
            .map(|_| LayerKvCache::new())
            .collect();
        let mut out = Vec::with_capacity(self.codebooks);
        for i in 0..self.codebooks {
            let cur = (din.i(i)? + &df_token)?.reshape((1, 1, self.depthformer_dim))?;
            let dout = self
                .depthformer
                .forward(&cur, Some(caches.as_mut_slice()))?; // (1,1,dim)
            let dout = dout.reshape((1, self.depthformer_dim))?;
            let logits = self.depth_embeddings[i].get_logits(&dout)?.i(0)?; // (vocab,)
            let token = sampler.sample(&logits)?;
            out.push(token);
            let tok = Tensor::from_vec(vec![token], (1,), din.device())?;
            df_token = self.depth_embeddings[i]
                .embed(&tok)?
                .reshape((self.depthformer_dim,))?;
        }
        Ok(out)
    }

    /// One frame through [`crate::flashkern::decode::DepthDecode::frame`]: extract the
    /// backbone hidden's bf16 bits, run the dispatch, and sample per codebook via the
    /// SAME seeded sampler over the rb'd logits (rebuilt as a tiny bf16 tensor so the
    /// LogitsProcessor sees byte-identical inputs to the candle path's `get_logits`).
    fn sample_audio_frame_flash(
        &self,
        ctx: &crate::flashkern::decode::DepthDecode,
        embedding: &Tensor,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        use candle_core::Storage;
        let flat = embedding.flatten_all()?.contiguous()?;
        let bits: Vec<u16> = {
            let (st, l) = flat.storage_and_layout();
            match &*st {
                Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                    let (a, b) = l
                        .contiguous_offsets()
                        .ok_or_else(|| candle_core::Error::Msg("hidden not contiguous".into()))?;
                    v[a..b].iter().map(|x| x.to_bits()).collect()
                }
                _ => candle_core::bail!("depth flash: hidden must be CPU bf16"),
            }
        };
        let err_cell = std::sync::Mutex::new(None);
        let dev = embedding.device().clone();
        let toks = ctx.frame(&bits, |logits_bits| {
            let v: Vec<half::bf16> = logits_bits
                .iter()
                .map(|&b| half::bf16::from_bits(b))
                .collect();
            match Tensor::from_vec(v, (logits_bits.len(),), &dev).and_then(|t| sampler.sample(&t)) {
                Ok(t) => t,
                Err(e) => {
                    *err_cell.lock().unwrap() = Some(e);
                    0
                }
            }
        });
        if let Some(e) = err_cell.lock().unwrap().take() {
            return Err(e);
        }
        Ok(toks)
    }

    /// Install the head tables on the native engine (text embed = tied logits head,
    /// audio embed table, final embedding-norm). No-op when captures fail — the token
    /// pass simply stays unserved.
    fn install_native_heads(&self) {
        use crate::flashkern::decode::PtrLen;
        let engine = crate::flashkern::native_engine::process_engine();
        let embed = self.lfm.embed_weight();
        let audio = self.audio_embedding.flash_parts().0;
        let norm = self.lfm.embedding_norm();
        let (Some(ep), Some(ap), Some(np)) = (
            PtrLen::bf16(embed),
            PtrLen::bf16(audio),
            PtrLen::bf16(norm.weight()),
        ) else {
            return;
        };
        let (Ok(vocab), Ok(arows)) = (embed.dim(0), audio.dim(0)) else {
            return;
        };
        let _ = engine.set_heads(
            ep.addr() as *const u16,
            vocab,
            ap.addr() as *const u16,
            arows,
            np.addr() as *const u16,
            norm.eps() as f32,
        );
    }

    /// The native token pass for the generate loop: ids in, `(h_last, logits?)` out.
    /// `Ok(None)` = unserved (any gate failed) — caller builds `in_emb` and takes the
    /// candle path, bit-identical. On success `index_pos` has been advanced.
    fn native_token_step(
        &self,
        cache: &mut LfmCache,
        index_pos: &mut usize,
        ids: &[u32],
        embed_kind: u32,
        want_logits: bool,
    ) -> Result<Option<(Tensor, Option<Tensor>)>> {
        use half::slice::HalfFloatSliceExt;
        let vocab = self.lfm.embed_weight().dim(0)?;
        // Audio ids arrive RAW (per-codebook tokens); the engine's table is the flat
        // audio-embedding matrix, so apply the codebook offsets here — the same
        // `t + offset` audio_frame_embed applies. At most 8 codebooks: stack buffer.
        let mut idbuf = [0u32; 8];
        let ids = if embed_kind == 1 {
            if ids.len() > idbuf.len() {
                return Ok(None);
            }
            for (slot, (t, o)) in idbuf.iter_mut().zip(ids.iter().zip(&self.codebook_offsets)) {
                *slot = (*t as i64 + o) as u32;
            }
            &idbuf[..ids.len()]
        } else {
            ids
        };
        // The hidden Tensor's own storage doubles as the engine's out plane (bf16 is
        // bit-transparent over u16) — the one allocation this step makes, and it is
        // the Tensor the depthformer consumes. The Tensor boundary itself dies when
        // the depthformer folds onto the lane team / ST_LOGITS revives.
        let mut h_bits: Vec<half::bf16> = vec![half::bf16::ZERO; self.hidden];
        let mut logits_buf: Vec<f32> = if want_logits { vec![0f32; vocab] } else { Vec::new() };
        let served = self.lfm.native_token_pass(
            cache,
            *index_pos,
            ids,
            embed_kind,
            h_bits.as_mut_slice().reinterpret_cast_mut(),
            if want_logits { Some(&mut logits_buf) } else { None },
        )?;
        if !served {
            return Ok(None);
        }
        *index_pos += 1;
        let h_last = Tensor::from_vec(h_bits, (self.hidden,), &candle_core::Device::Cpu)?;
        let logits = if want_logits {
            Some(Tensor::from_vec(logits_buf, (vocab,), &candle_core::Device::Cpu)?)
        } else {
            None
        };
        Ok(Some((h_last, logits)))
    }

    fn audio_frame_embed(&self, tokens: &[u32]) -> Result<Tensor> {
        // audio_embedding(tokens + offsets).sum(0) → (D,) → (1,1,D)
        let dev = self.lfm.embed_weight().device();
        let codes: Vec<i64> = tokens
            .iter()
            .zip(&self.codebook_offsets)
            .map(|(t, o)| *t as i64 + o)
            .collect();
        let codes = Tensor::from_vec(codes, (self.codebooks,), dev)?;
        let emb = self.audio_embedding.embed(&codes)?; // (codebooks, D)
        emb.sum(0)?.reshape((1, 1, self.hidden))
    }

    /// `generate_sequential` as a synchronous callback stream — text is emitted
    /// in full, then (after `<|audio_start|>`) audio frames until EOAudio.
    /// Faithful to the Python generator (ASR/TTS path).
    pub fn generate_sequential<F: FnMut(GenToken)>(
        &self,
        chat: &ChatState,
        params: &GenParams,
        mut on_token: F,
    ) -> Result<()> {
        let mut in_emb = self.prefill(chat)?;
        let mut index_pos = 0usize;
        let mut cache = self.new_cache(in_emb.dtype(), in_emb.device())?;
        // Text and audio carry independent samplers (the Python uses one generator;
        // for greedy — the default — both are argmax and the RNG is unused).
        let mut text_sampler =
            Sampler::new(params.seed, params.text_temperature, params.text_top_k);
        let mut audio_sampler =
            Sampler::new(params.seed, params.audio_temperature, params.audio_top_k);

        let mut current = LFMModality::Text;

        let mut ended = false;
        for _ in 0..params.max_new_tokens {
            let seq_len = in_emb.dim(1)?;
            let h = self
                .lfm
                .forward_embeds(&in_emb, index_pos, &mut cache, None)?; // (1, seq, D)
            index_pos += seq_len;
            let h_last = h.i((0, seq_len - 1))?.contiguous()?; // (D,)

            match current {
                LFMModality::Text => {
                    let logits = self.text_logits(&h_last)?;
                    let next = self.sample_text_token(&logits, &mut text_sampler)?;
                    on_token(GenToken::Text(next));
                    if next == self.special.audio_start {
                        current = LFMModality::AudioOut; // <|audio_start|>
                    }
                    if next == self.special.im_end {
                        ended = true;
                        break; // <|im_end|>
                    }
                    let tok = Tensor::from_vec(vec![next], (1,), in_emb.device())?;
                    in_emb = self.lfm.embed(&tok)?.reshape((1, 1, self.hidden))?;
                }
                LFMModality::AudioOut => {
                    let mut frame = self.sample_audio_frame(&h_last, &mut audio_sampler)?;
                    if frame[0] == END_OF_AUDIO {
                        for c in frame.iter_mut() {
                            *c = END_OF_AUDIO; // next_token[:] = 2048
                        }
                        current = LFMModality::Text;
                    }
                    on_token(GenToken::Audio(frame.clone()));
                    in_emb = self.audio_frame_embed(&frame)?;
                }
                LFMModality::AudioIn => unreachable!(),
            }
        }
        if !ended {
            eprintln!(
                "\n[voice] max_new_tokens ({}) EXHAUSTED mid-reply — the turn was \
                 truncated before <|im_end|>; raise the token budget",
                params.max_new_tokens
            );
        }
        Ok(())
    }

    /// `generate_interleaved` as a synchronous callback stream — interleaves runs
    /// of text and audio (real-time S2S). Faithful to the Python generator.
    pub fn generate_interleaved<F: FnMut(GenToken)>(
        &self,
        chat: &ChatState,
        params: &GenParams,
        on_token: F,
    ) -> Result<()> {
        let in_emb = self.prefill(chat)?;
        self.generate_from_embeds(in_emb, params, on_token)
    }

    /// `generate_interleaved` with a **barge-in** signal: the loop polls `cancel` at the
    /// top of every decode step and returns early once it is set. This is the
    /// consumer-stops-the-generator semantics of the Python path (whose generator simply
    /// stops being iterated) made explicit for the worker-thread pipeline — when a new
    /// utterance arrives the realtime worker flips `cancel` and generation aborts mid-stream
    /// instead of running to `max_new_tokens` and wasting the P-cores.
    pub fn generate_interleaved_cancellable<F: FnMut(GenToken)>(
        &self,
        chat: &ChatState,
        params: &GenParams,
        cancel: &AtomicBool,
        on_token: F,
    ) -> Result<()> {
        let in_emb = self.prefill(chat)?;
        self.generate_from_embeds_cancellable(in_emb, params, cancel, on_token)
    }

    /// The interleaved generation loop given the prefill embeds directly (Python
    /// `generate_interleaved` after `_prefill`). Exposed so it can be driven from raw
    /// model inputs (`prefill_inputs`) for the end-to-end `generate_interleaved_parity`
    /// golden, not just a `ChatState`.
    pub fn generate_from_embeds<F: FnMut(GenToken)>(
        &self,
        in_emb: Tensor,
        params: &GenParams,
        on_token: F,
    ) -> Result<()> {
        // Never-set flag ⇒ identical behavior to before (one relaxed atomic load per step
        // is negligible next to a transformer block).
        self.generate_from_embeds_cancellable(in_emb, params, &AtomicBool::new(false), on_token)
    }

    /// [`generate_from_embeds`] with a barge-in `cancel` signal — see
    /// [`generate_interleaved_cancellable`](Self::generate_interleaved_cancellable).
    pub fn generate_from_embeds_cancellable<F: FnMut(GenToken)>(
        &self,
        in_emb: Tensor,
        params: &GenParams,
        cancel: &AtomicBool,
        on_token: F,
    ) -> Result<()> {
        let mut cache = self.new_cache(in_emb.dtype(), in_emb.device())?;
        let mut index_pos = 0usize;
        self.generate_with_cache(&mut cache, &mut index_pos, in_emb, params, cancel, on_token)
    }

    /// The interleaved generation loop over an EXTERNALLY-owned cache and position —
    /// the persistent cross-turn path (spec 09, W2a). `in_emb` is the context suffix
    /// the cache has not seen ([`prefill_suffix`](Self::prefill_suffix)); `index_pos`
    /// is the cache's current length and is left at the new cache length on return,
    /// so the caller can account exactly which generated tokens were forwarded (all
    /// emitted tokens except the last one are in the cache — the loop forwards the
    /// previous token before sampling the next).
    pub fn generate_with_cache<F: FnMut(GenToken)>(
        &self,
        cache: &mut LfmCache,
        index_pos: &mut usize,
        mut in_emb: Tensor,
        params: &GenParams,
        cancel: &AtomicBool,
        mut on_token: F,
    ) -> Result<()> {
        let mut text_sampler =
            Sampler::new(params.seed, params.text_temperature, params.text_top_k);
        let mut audio_sampler =
            Sampler::new(params.seed, params.audio_temperature, params.audio_top_k);

        let mut current = LFMModality::Text;
        let mut modality_left = self.interleaved_n_text as i64;
        let mut text_done = false;
        // Sampled ids waiting to become the next step's input: `(raw ids, embed_kind)`.
        // The native token pass consumes ids directly (embed absorbed into the engine);
        // the candle path derives `in_emb` from them on demand.
        let mut pending: Option<(Vec<u32>, u32)> = None;
        // Distinguishes a natural end (<|im_end|> or barge-in) from silently
        // exhausting the token budget — a truncated reply must be LOUD, never
        // "the model just stopped talking".
        let mut ended = false;

        for _ in 0..params.max_new_tokens {
            // Barge-in: a new utterance asked us to stop — drop this reply mid-stream.
            // (The pass-boundary doorbell: never checked inside a token.)
            if cancel.load(Ordering::Acquire) {
                ended = true;
                break;
            }
            modality_left -= 1;
            // Text logits ride the engine (kcoro audit finding): ST_LOGITS runs
            // the SAME lfm_bf16_gemm_nt_f32 as linear_logits and now emits raw
            // f32 — the extra bf16 round that flipped the perf hash is gone, so
            // the native head is bit-identical to the candle head (PERF oracle
            // = the proof). Audio steps skip the head (the depthformer wants
            // hidden, not logits).
            let want_logits = matches!(current, LFMModality::Text);
            let stepped = match pending.as_ref() {
                Some((ids, kind)) => {
                    self.native_token_step(cache, index_pos, ids, *kind, want_logits)?
                }
                None => None,
            };
            let (h_last, native_logits) = match stepped {
                Some((h, lg)) => (h, lg),
                None => {
                    // The candle path (prefill step, or any native gate failed):
                    // derive in_emb from the pending ids first if there are any.
                    if let Some((ids, kind)) = pending.take() {
                        in_emb = if kind == 0 {
                            let tok = Tensor::from_vec(vec![ids[0]], (1,), in_emb.device())?;
                            self.lfm.embed(&tok)?.reshape((1, 1, self.hidden))?
                        } else {
                            self.audio_frame_embed(&ids)?
                        };
                    }
                    let seq_len = in_emb.dim(1)?;
                    let h = self.lfm.forward_embeds(&in_emb, *index_pos, cache, None)?;
                    *index_pos += seq_len;
                    (h.i((0, seq_len - 1))?.contiguous()?, None)
                }
            };

            match current {
                LFMModality::Text => {
                    let logits = match native_logits {
                        Some(l) => l,
                        None => self.text_logits(&h_last)?,
                    };
                    let next = self.sample_text_token(&logits, &mut text_sampler)?;
                    if next == self.special.im_end {
                        ended = true;
                        break; // <|im_end|>
                    }
                    on_token(GenToken::Text(next));
                    if next == self.special.text_end {
                        text_done = true; // <|text_end|>
                    }
                    if modality_left <= 0 || text_done {
                        current = LFMModality::AudioOut;
                        modality_left = self.interleaved_n_audio as i64;
                    }
                    pending = Some((vec![next], 0));
                }
                LFMModality::AudioOut => {
                    let mut frame = self.sample_audio_frame(&h_last, &mut audio_sampler)?;
                    if modality_left <= 0 && !text_done {
                        current = LFMModality::Text;
                        modality_left = self.interleaved_n_text as i64;
                    }
                    if frame[0] == END_OF_AUDIO {
                        for c in frame.iter_mut() {
                            *c = END_OF_AUDIO; // next_token[:] = 2048
                        }
                        current = LFMModality::Text;
                    }
                    on_token(GenToken::Audio(frame.clone()));
                    pending = Some((frame, 1));
                }
                LFMModality::AudioIn => unreachable!(),
            }
        }
        if !ended {
            eprintln!(
                "\n[voice] max_new_tokens ({}) EXHAUSTED mid-reply — the turn was \
                 truncated before <|im_end|>; raise the token budget",
                params.max_new_tokens
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn logits(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (v.len(),), &Device::Cpu).unwrap()
    }

    #[test]
    fn greedy_when_no_temperature() {
        let l = logits(&[0.1, 5.0, 0.2, 3.0]);
        assert_eq!(Sampler::new(0, None, None).sample(&l).unwrap(), 1);
    }

    #[test]
    fn greedy_when_temp_nonpositive_or_topk_one() {
        let l = logits(&[0.1, 5.0, 0.2, 3.0]);
        // temperature <= 0 ⇒ greedy
        assert_eq!(Sampler::new(0, Some(0.0), Some(50)).sample(&l).unwrap(), 1);
        // top_k == 1 ⇒ greedy even with a temperature
        assert_eq!(Sampler::new(0, Some(1.5), Some(1)).sample(&l).unwrap(), 1);
    }

    #[test]
    fn topk_restricts_support() {
        // With top_k=2 the only reachable tokens are the two largest logits (1, 3).
        let l = logits(&[0.1, 5.0, 0.2, 3.0, -2.0]);
        let mut s = Sampler::new(7, Some(1.0), Some(2));
        for _ in 0..200 {
            let t = s.sample(&l).unwrap();
            assert!(t == 1 || t == 3, "top-k=2 sampled out-of-support token {t}");
        }
    }

    #[test]
    fn seed_is_reproducible() {
        let l = logits(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let draw = || {
            let mut s = Sampler::new(123, Some(1.0), None);
            (0..16).map(|_| s.sample(&l).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn sampling_can_pick_nonargmax() {
        // A flat-ish distribution with temperature should not always return argmax.
        let l = logits(&[2.0, 1.9, 1.8, 1.7]);
        let mut s = Sampler::new(1, Some(1.0), None);
        let mut seen_non_zero = false;
        for _ in 0..200 {
            if s.sample(&l).unwrap() != 0 {
                seen_non_zero = true;
                break;
            }
        }
        assert!(seen_non_zero, "temperature sampling never left the argmax");
    }
}
