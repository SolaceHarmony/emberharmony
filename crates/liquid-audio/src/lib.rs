//! Desktop control data and model snapshot download support.
//!
//! Inference, scheduling, checkpoint interpretation, PCM, and model/session
//! lifetime are owned by the standalone C++23 host. This crate has no native
//! linkage and no in-process inference surface.

mod control;
pub mod utils;

pub use control::AudioStatsSnapshot;
#[cfg(feature = "download")]
pub use utils::{snapshot_download_to, snapshot_download_with, DownloadProgress};
