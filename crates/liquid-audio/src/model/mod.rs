//! Port of `liquid_audio/model/`.

pub mod lfm2_audio; // model/lfm2_audio.py   (LFM2AudioModel + generate_interleaved)
pub mod lfm2_hf; // HF Lfm2Model backbone (main sequence model + detokenizer)
pub mod linear;
pub mod native_conformer; // Rust rim over the native Conformer encoder + adapter
pub mod transformer; // model/transformer.py  (depthformer + shared embeddings)
