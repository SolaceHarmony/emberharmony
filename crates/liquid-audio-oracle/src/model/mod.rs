//! Port of `liquid_audio/model/`.

pub mod lfm2_audio; // teacher-forced LFM2 loss model
pub mod lfm2_hf; // differentiable HF LFM2 backbone reference
pub mod linear;
pub mod transformer; // model/transformer.py  (depthformer + shared embeddings)
