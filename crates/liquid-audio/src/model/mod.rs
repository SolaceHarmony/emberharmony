//! Port of `liquid_audio/model/`.

pub mod conformer; // model/conformer/      (FastConformer audio encoder)
pub mod lfm2_audio; // model/lfm2_audio.py   (LFM2AudioModel + generate_interleaved)
pub mod lfm2_hf; // HF Lfm2Model backbone (main sequence model + detokenizer)
pub mod linear;
pub mod mlp;
pub mod norm; // differentiable LayerNorm (candle_nn::LayerNorm forward is no_bwd)
pub mod transformer; // model/transformer.py  (depthformer + shared embeddings)
