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

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::model::conformer::encoder::{ConformerEncoder, ConformerEncoderConfig};
use crate::model::lfm2_hf::{Cache as LfmCache, Lfm2Config, Model as Lfm2Model};
use crate::model::mlp::MLP;
use crate::model::transformer::{HeadStyle, LayerKvCache, Mha, RawLmBackbone, SharedEmbedding, StandardBlock};
use crate::processor::ChatState;
use crate::utils::LFMModality;

/// +1 over 2048 for the EOAudio token.
const AUDIO_VOCAB_SIZE: usize = 2048 + 1;

#[derive(Debug, Clone)]
pub struct DepthformerConfig {
    pub layers: usize,
    pub dim: usize,
    pub tie: bool,
}

/// One streamed token: a text id, or one audio frame (codebooks codes).
#[derive(Debug, Clone)]
pub enum GenToken {
    Text(u32),
    Audio(Vec<u32>),
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

/// Faithful port of the sampling body shared by `_sample_text_token` and the
/// per-codebook step of `_sample_audio_frame`:
/// ```python
/// greedy = temperature is None or temperature <= 0 or top_k == 1
/// if greedy: next = logits.argmax()
/// else:
///     logits /= temperature
///     if top_k is not None:
///         min_score = torch.topk(logits, top_k).values[-1]
///         logits[logits < min_score] = -inf
///     next = torch.multinomial(logits.softmax(0), 1)
/// ```
fn sample_token(logits: &Tensor, temperature: Option<f64>, top_k: Option<usize>, rng: &mut StdRng) -> Result<u32> {
    let greedy = match (temperature, top_k) {
        (None, _) => true,
        (Some(t), _) if t <= 0.0 => true,
        (_, Some(1)) => true,
        _ => false,
    };
    let logits = logits.to_dtype(DType::F32)?;
    if greedy {
        return logits.argmax(0)?.to_scalar::<u32>();
    }
    let temp = temperature.expect("non-greedy ⇒ temperature is Some(>0)") as f32;
    let mut v: Vec<f32> = logits.to_vec1::<f32>()?.into_iter().map(|x| x / temp).collect();
    if let Some(k) = top_k {
        // min_score = the k-th largest logit; keep ≥ it, mask the rest to -inf.
        let k = k.min(v.len());
        let mut sorted = v.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let min_score = sorted[k - 1];
        for x in v.iter_mut() {
            if *x < min_score {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    // softmax (numerically stable) → multinomial(probs, 1) via inverse-CDF draw.
    let maxv = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in v.iter_mut() {
        *x = (*x - maxv).exp();
        sum += *x;
    }
    let r: f32 = rng.gen::<f32>() * sum;
    let mut acc = 0f32;
    for (i, p) in v.iter().enumerate() {
        acc += *p;
        if r < acc {
            return Ok(i as u32);
        }
    }
    Ok((v.len() - 1) as u32) // float-rounding guard: fall back to the last bin
}

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
    hidden: usize,
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
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = lfm_cfg.hidden_size;
        let lfm = Lfm2Model::new(&lfm_cfg, vb.pp("lfm"))?;
        let conformer = ConformerEncoder::new(enc_cfg, vb.pp("conformer"))?;
        let feat_out = if enc_cfg.feat_out > 0 && enc_cfg.feat_out != enc_cfg.d_model { enc_cfg.feat_out } else { enc_cfg.d_model };
        let audio_adapter = MLP::new(feat_out, hidden, &[hidden], true, true, 0.0, vb.pp("audio_adapter"))?;
        let audio_embedding = SharedEmbedding::new(hidden, AUDIO_VOCAB_SIZE * codebooks, 1e-5, vb.pp("audio_embedding"))?;

        // Depthformer: RawLMBackbone(has_embedding=False) of StandardBlock(MHA(dim)).
        let df_vb = vb.pp("depthformer").pp("layers");
        let mut layers = Vec::with_capacity(depth_cfg.layers);
        for i in 0..depth_cfg.layers {
            let lvb = df_vb.pp(i.to_string());
            let mha = Mha::new(depth_cfg.dim, 32, HeadStyle::Gqa, true, 1e-5, 8, 128_000, 1_000_000.0, lvb.pp("operator"))?;
            let block = StandardBlock::new(mha, None, true, 256, 1.0, 1e-5, lvb)?;
            layers.push(block);
        }
        let depthformer = RawLmBackbone { layers, embedding: None, dim: depth_cfg.dim };

        let depth_linear = linear(hidden, depth_cfg.dim * codebooks, vb.pp("depth_linear"))?;
        let de_vb = vb.pp("depth_embeddings");
        let mut depth_embeddings = Vec::with_capacity(codebooks);
        for i in 0..codebooks {
            depth_embeddings.push(SharedEmbedding::new(depth_cfg.dim, AUDIO_VOCAB_SIZE, 1e-5, de_vb.pp(i.to_string()))?);
        }

        let codebook_offsets = (0..codebooks as i64).map(|i| i * AUDIO_VOCAB_SIZE as i64).collect();

        Ok(Self {
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
            hidden,
        })
    }

    /// Run the FastConformer encoder over mel features `(B, feat_in, T)` →
    /// `(B, d, T')`. Exposed for parity testing.
    pub fn conformer_encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.conformer.forward(mel)
    }

    /// Debug: conformer stage intermediates for parity localization.
    #[doc(hidden)]
    pub fn conformer_stages(&self, mel: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
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
        let mut cache = LfmCache::new(true, embeds.dtype(), &self.lfm_cfg, embeds.device())?;
        self.lfm.forward_embeds(embeds, 0, &mut cache, None)
    }

    /// Debug: tied-embedding text logits for a single hidden vector (H,) — the
    /// text head used in generation.
    #[doc(hidden)]
    pub fn text_logits_of(&self, hidden_last: &Tensor) -> Result<Tensor> {
        self.text_logits(hidden_last)
    }

    /// Debug: greedy depthformer audio frame (8 codebook tokens) for a fixed
    /// `embedding` (H,) — for depthformer parity (token-exact vs Python greedy).
    #[doc(hidden)]
    pub fn audio_frame_greedy(&self, embedding: &Tensor) -> Result<Vec<u32>> {
        let mut rng = StdRng::seed_from_u64(0); // unused on the greedy path
        self.sample_audio_frame(embedding, None, None, &mut rng)
    }

    /// Build the prefill input embeddings, scattering text / audio-in / audio-out
    /// embeddings into sequence order by `modality_flag` (index_select instead of
    /// PyTorch boolean assignment).
    fn prefill(&self, chat: &ChatState) -> Result<Tensor> {
        let dev = chat.text.device();
        let modality: Vec<u32> = chat.modality_flag.i(0)?.to_vec1::<u32>()?;
        let l = modality.len();

        // text embeddings (n_text, D)
        let text_emb = self.lfm.embed(&chat.text)?.i(0)?; // (n_text, D)

        // audio-in embeddings (n_ai, D): encode each segment, adapt, concat.
        let lens: Vec<u32> = chat.audio_in_lens.to_vec1::<u32>().unwrap_or_default();
        let mut audio_in_rows: Vec<Tensor> = Vec::new();
        let mut frame_cursor = 0usize;
        for &len in &lens {
            let seg = chat.audio_in.narrow(1, frame_cursor, len as usize)?; // (128, frames)
            frame_cursor += len as usize;
            let seg = seg.unsqueeze(0)?; // (1, 128, frames)
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
            let m = chat.audio_out.dim(1)?;
            if m == 0 {
                None
            } else {
                let codes = chat.audio_out.narrow(0, 0, self.codebooks)?.to_dtype(DType::I64)?;
                let offs = Tensor::from_vec(self.codebook_offsets.clone(), (self.codebooks, 1), dev)?;
                let offset_codes = codes.broadcast_add(&offs)?; // (codebooks, m)
                let emb = self.audio_embedding.embed(&offset_codes)?; // (codebooks, m, D)
                Some(emb.sum(0)?.to_dtype(text_emb.dtype())?) // (m, D)
            }
        };

        // combined = [text; audio_in; audio_out]; build index per position.
        let n_text = text_emb.dim(0)?;
        let n_ai = audio_in_emb.as_ref().map(|a| a.dim(0).unwrap_or(0)).unwrap_or(0);
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
            let idx = if *m == LFMModality::Text as u32 {
                let v = text_base + ct;
                ct += 1;
                v
            } else if *m == LFMModality::AudioIn as u32 {
                let v = ai_base + cai;
                cai += 1;
                v
            } else {
                let v = ao_base + cao;
                cao += 1;
                v
            };
            index.push(idx as u32);
        }
        let index = Tensor::from_vec(index, (l,), dev)?;
        let in_emb = combined.index_select(&index, 0)?; // (L, D)
        in_emb.unsqueeze(0) // (1, L, D)
    }

    fn text_logits(&self, h_last: &Tensor) -> Result<Tensor> {
        // nn.functional.linear(h, embed_weight): (V,D) @ (D,) -> (V,)
        let w = self.lfm.embed_weight().to_dtype(DType::F32)?;
        let h = h_last.to_dtype(DType::F32)?.reshape((self.hidden, 1))?;
        w.matmul(&h)?.squeeze(1)
    }

    /// Depthformer audio-frame sampler → `codebooks` codes. Faithful to
    /// `_sample_audio_frame`: per-codebook greedy/temperature/top-k via
    /// [`sample_token`].
    fn sample_audio_frame(&self, embedding: &Tensor, temperature: Option<f64>, top_k: Option<usize>, rng: &mut StdRng) -> Result<Vec<u32>> {
        // depth_linear(embedding) → (codebooks, depthformer_dim). `embedding` is a
        // 1-D (D,) lfm hidden; candle's Linear needs a 2-D input, so add a row dim
        // (Python's nn.Linear accepts the 1-D vector directly).
        let emb2d = embedding.flatten_all()?.unsqueeze(0)?; // (1, D)
        let din = self.depth_linear.forward(&emb2d)?.reshape((self.codebooks, self.depthformer_dim))?;
        let mut df_token = Tensor::zeros((self.depthformer_dim,), din.dtype(), din.device())?;
        let mut caches: Vec<LayerKvCache> = (0..self.depthformer.layers.len()).map(|_| LayerKvCache::new()).collect();
        let mut out = Vec::with_capacity(self.codebooks);
        for i in 0..self.codebooks {
            let cur = (din.i(i)? + &df_token)?.reshape((1, 1, self.depthformer_dim))?;
            let dout = self.depthformer.forward(&cur, Some(caches.as_mut_slice()))?; // (1,1,dim)
            let dout = dout.reshape((1, self.depthformer_dim))?;
            let logits = self.depth_embeddings[i].get_logits(&dout)?.i(0)?; // (vocab,)
            let token = sample_token(&logits, temperature, top_k, rng)?;
            out.push(token);
            let tok = Tensor::from_vec(vec![token], (1,), din.device())?;
            df_token = self.depth_embeddings[i].embed(&tok)?.reshape((self.depthformer_dim,))?;
        }
        Ok(out)
    }

    fn audio_frame_embed(&self, tokens: &[u32]) -> Result<Tensor> {
        // audio_embedding(tokens + offsets).sum(0) → (D,) → (1,1,D)
        let dev = self.lfm.embed_weight().device();
        let codes: Vec<i64> = tokens.iter().zip(&self.codebook_offsets).map(|(t, o)| *t as i64 + o).collect();
        let codes = Tensor::from_vec(codes, (self.codebooks,), dev)?;
        let emb = self.audio_embedding.embed(&codes)?; // (codebooks, D)
        emb.sum(0)?.reshape((1, 1, self.hidden))
    }

    /// `generate_sequential` as a synchronous callback stream — text is emitted
    /// in full, then (after `<|audio_start|>`) audio frames until EOAudio.
    /// Faithful to the Python generator (ASR/TTS path).
    pub fn generate_sequential<F: FnMut(GenToken)>(&self, chat: &ChatState, params: &GenParams, mut on_token: F) -> Result<()> {
        let mut in_emb = self.prefill(chat)?;
        let mut index_pos = 0usize;
        let mut cache = LfmCache::new(true, in_emb.dtype(), &self.lfm_cfg, in_emb.device())?;
        let mut rng = StdRng::seed_from_u64(params.seed);

        let mut current = LFMModality::Text;

        for _ in 0..params.max_new_tokens {
            let seq_len = in_emb.dim(1)?;
            let h = self.lfm.forward_embeds(&in_emb, index_pos, &mut cache, None)?; // (1, seq, D)
            index_pos += seq_len;
            let h_last = h.i((0, seq_len - 1))?.contiguous()?; // (D,)

            match current {
                LFMModality::Text => {
                    let logits = self.text_logits(&h_last)?;
                    let next = sample_token(&logits, params.text_temperature, params.text_top_k, &mut rng)?;
                    on_token(GenToken::Text(next));
                    if next == 128 {
                        current = LFMModality::AudioOut; // <|audio_start|>
                    }
                    if next == 7 {
                        break; // <|im_end|>
                    }
                    let tok = Tensor::from_vec(vec![next], (1,), in_emb.device())?;
                    in_emb = self.lfm.embed(&tok)?.reshape((1, 1, self.hidden))?;
                }
                LFMModality::AudioOut => {
                    let mut frame = self.sample_audio_frame(&h_last, params.audio_temperature, params.audio_top_k, &mut rng)?;
                    if frame[0] == 2048 {
                        for c in frame.iter_mut() {
                            *c = 2048; // next_token[:] = 2048
                        }
                        current = LFMModality::Text;
                    }
                    on_token(GenToken::Audio(frame.clone()));
                    in_emb = self.audio_frame_embed(&frame)?;
                }
                LFMModality::AudioIn => unreachable!(),
            }
        }
        Ok(())
    }

    /// `generate_interleaved` as a synchronous callback stream — interleaves runs
    /// of text and audio (real-time S2S). Faithful to the Python generator.
    pub fn generate_interleaved<F: FnMut(GenToken)>(&self, chat: &ChatState, params: &GenParams, mut on_token: F) -> Result<()> {
        let mut in_emb = self.prefill(chat)?;
        let mut index_pos = 0usize;
        let mut cache = LfmCache::new(true, in_emb.dtype(), &self.lfm_cfg, in_emb.device())?;
        let mut rng = StdRng::seed_from_u64(params.seed);

        let mut current = LFMModality::Text;
        let mut modality_left = self.interleaved_n_text as i64;
        let mut text_done = false;

        for _ in 0..params.max_new_tokens {
            modality_left -= 1;
            let seq_len = in_emb.dim(1)?;
            let h = self.lfm.forward_embeds(&in_emb, index_pos, &mut cache, None)?; // (1, seq, D)
            index_pos += seq_len;
            let h_last = h.i((0, seq_len - 1))?.contiguous()?; // (D,)

            match current {
                LFMModality::Text => {
                    let logits = self.text_logits(&h_last)?;
                    let next = sample_token(&logits, params.text_temperature, params.text_top_k, &mut rng)?;
                    if next == 7 {
                        break; // <|im_end|>
                    }
                    on_token(GenToken::Text(next));
                    if next == 130 {
                        text_done = true; // <|text_end|>
                    }
                    if modality_left <= 0 || text_done {
                        current = LFMModality::AudioOut;
                        modality_left = self.interleaved_n_audio as i64;
                    }
                    let tok = Tensor::from_vec(vec![next], (1,), in_emb.device())?;
                    in_emb = self.lfm.embed(&tok)?.reshape((1, 1, self.hidden))?;
                }
                LFMModality::AudioOut => {
                    let mut frame = self.sample_audio_frame(&h_last, params.audio_temperature, params.audio_top_k, &mut rng)?;
                    if modality_left <= 0 && !text_done {
                        current = LFMModality::Text;
                        modality_left = self.interleaved_n_text as i64;
                    }
                    if frame[0] == 2048 {
                        for c in frame.iter_mut() {
                            *c = 2048; // next_token[:] = 2048
                        }
                        current = LFMModality::Text;
                    }
                    on_token(GenToken::Audio(frame.clone()));
                    in_emb = self.audio_frame_embed(&frame)?;
                }
                LFMModality::AudioIn => unreachable!(),
            }
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
        let mut rng = StdRng::seed_from_u64(0);
        let l = logits(&[0.1, 5.0, 0.2, 3.0]);
        assert_eq!(sample_token(&l, None, None, &mut rng).unwrap(), 1);
    }

    #[test]
    fn greedy_when_temp_nonpositive_or_topk_one() {
        let mut rng = StdRng::seed_from_u64(0);
        let l = logits(&[0.1, 5.0, 0.2, 3.0]);
        // temperature <= 0 ⇒ greedy
        assert_eq!(sample_token(&l, Some(0.0), Some(50), &mut rng).unwrap(), 1);
        // top_k == 1 ⇒ greedy even with a temperature
        assert_eq!(sample_token(&l, Some(1.5), Some(1), &mut rng).unwrap(), 1);
    }

    #[test]
    fn topk_restricts_support() {
        // With top_k=2 the only reachable tokens are the two largest logits (1, 3).
        let l = logits(&[0.1, 5.0, 0.2, 3.0, -2.0]);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let t = sample_token(&l, Some(1.0), Some(2), &mut rng).unwrap();
            assert!(t == 1 || t == 3, "top-k=2 sampled out-of-support token {t}");
        }
    }

    #[test]
    fn seed_is_reproducible() {
        let l = logits(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let draw = || {
            let mut rng = StdRng::seed_from_u64(123);
            (0..16).map(|_| sample_token(&l, Some(1.0), None, &mut rng).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn sampling_can_pick_nonargmax() {
        // A flat-ish distribution with temperature should not always return argmax.
        let l = logits(&[2.0, 1.9, 1.8, 1.7]);
        let mut rng = StdRng::seed_from_u64(1);
        let mut seen_non_zero = false;
        for _ in 0..200 {
            if sample_token(&l, Some(1.0), None, &mut rng).unwrap() != 0 {
                seen_non_zero = true;
                break;
            }
        }
        assert!(seen_non_zero, "temperature sampling never left the argmax");
    }
}
