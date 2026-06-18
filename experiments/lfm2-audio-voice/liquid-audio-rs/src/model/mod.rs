//! Port of `liquid_audio/model/`.

pub mod conformer; // model/conformer/      (FastConformer audio encoder)
pub mod mlp;
pub mod transformer; // model/transformer.py  (LFM2 backbone)
// pub mod lfm2_audio;    // model/lfm2_audio.py   (LFM2AudioModel + generate_interleaved)
