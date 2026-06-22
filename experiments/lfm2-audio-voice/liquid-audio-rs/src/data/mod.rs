//! Port of `liquid_audio/data/` — the data-pipeline value types + loader.

pub mod dataloader; // data/dataloader.py (LFM2DataLoader + lfm2_collator)
pub mod mapper; // data/mapper.py (LFM2AudioChatMapper: chat → training sample)
pub mod preprocess; // data/preprocess.py (preprocess_dataset: chats → on-disk dataset)
pub mod types; // data/types.py (chat-content segments + pre-packed tensor bundles)

pub use mapper::LFM2AudioChatMapper;
pub use preprocess::preprocess_dataset;
