//! Port of `liquid_audio/utils.py`.

/// Modality flag for interleaved generation. Mirrors `LFMModality(IntEnum)` —
/// Python `IntEnum` + `auto()` numbers from 1, so TEXT=1, AUDIO_IN=2, AUDIO_OUT=3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum LFMModality {
    Text = 1,
    AudioIn = 2,
    AudioOut = 3,
}

/// Python floor division (`a // b`): rounds toward negative infinity, unlike
/// Rust's `/` which truncates toward zero. Needed for faithful `mel2emb_len`.
fn floordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

/// Convert log-mel feature length to final LFM embedding length.
///
/// This is just floor division. Faithful to `-(l // -8)` (i.e. ceil(l/8)).
/// Note: smallest mel-length for encoder is 9.
pub fn mel2emb_len(l: i64) -> i64 {
    -floordiv(l, -8)
}

/// Convert LFM embedding length to log-mel feature length.
///
/// Note: this is an upper bound. Faithful to `l * 8`.
pub fn emb2mel_len(l: i64) -> i64 {
    l * 8
}

/// Faithful to Python's importlib-based `module_exists`. Rust has no runtime
/// module lookup, so this maps to compile-time Cargo features (the analog used
/// here is the optional flash-attn path).
pub fn module_exists(name: &str) -> bool {
    match name {
        "flash_attn" => cfg!(feature = "flash-attn"),
        _ => false,
    }
}

/// Resolve a model directory. Faithful to `get_model_dir(repo_id, *, revision)`:
/// an existing local path is returned as-is (and a `revision` alongside a path is
/// an error, as in Python); otherwise `repo_or_path` is treated as a HF repo id
/// and snapshot-downloaded (the `huggingface_hub.snapshot_download` analog), the
/// snapshot directory being returned. The download branch requires the `download`
/// feature (on by default). Application hosts should use [`snapshot_download_to`]
/// so cache and credential policy are explicit.
pub fn get_model_dir(
    repo_or_path: &str,
    revision: Option<&str>,
) -> std::io::Result<std::path::PathBuf> {
    let p = std::path::PathBuf::from(repo_or_path);
    if p.is_dir() {
        if revision.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot use `revision` when given a path", // RuntimeError in Python
            ));
        }
        return Ok(p);
    }
    download_snapshot(repo_or_path, revision)
}

/// Per-file progress for [`snapshot_download_with`]. `index` is 0-based; `total` is the
/// sibling count. Byte-level progress is **not** available from hf-hub's sync API (it only
/// offers a terminal indicatif bar), so progress is honestly reported per file, not per byte.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub index: usize,
    pub total: usize,
    pub file: String,
}

/// `snapshot_download(repo_id, revision=...)` — fetch every file in the repo into
/// hf-hub's conventional user cache and return the snapshot directory. This convenience
/// path is always unauthenticated; application hosts use [`snapshot_download_to`].
#[cfg(feature = "download")]
fn download_snapshot(repo_id: &str, revision: Option<&str>) -> std::io::Result<std::path::PathBuf> {
    snapshot_download_with(repo_id, revision, None, |_| {})
}

/// Convenience download using hf-hub's conventional user cache. Credentials are
/// explicit: `None` means unauthenticated and never falls back to a token file.
#[cfg(feature = "download")]
pub fn snapshot_download_with(
    repo_id: &str,
    revision: Option<&str>,
    token: Option<&str>,
    progress: impl FnMut(DownloadProgress),
) -> std::io::Result<std::path::PathBuf> {
    let cache = hf_hub::Cache::default();
    snapshot_download_to(repo_id, revision, cache.path(), token, progress)
}

/// Download a complete snapshot into a host-selected cache directory. Cache,
/// revision, and credential are all explicit inputs; this function never consults
/// process environment or hf-hub's ambient token file.
#[cfg(feature = "download")]
pub fn snapshot_download_to(
    repo_id: &str,
    revision: Option<&str>,
    cache: &std::path::Path,
    token: Option<&str>,
    mut progress: impl FnMut(DownloadProgress),
) -> std::io::Result<std::path::PathBuf> {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    let to_io = |e: hf_hub::api::sync::ApiError| std::io::Error::other(format!("hf-hub: {e}"));

    let api = ApiBuilder::new()
        .with_cache_dir(cache.to_path_buf())
        .with_token(token.map(str::to_owned))
        .build()
        .map_err(to_io)?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(repo_id.to_string()),
    };

    fn snapshot_root(path: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
        let mut root = path.to_path_buf();
        for _ in std::path::Path::new(name).components() {
            if !root.pop() {
                return None;
            }
        }
        Some(root)
    }

    // List then fetch every sibling (snapshot_download grabs the whole repo).
    let info = repo.info().map_err(to_io)?;
    let total = info.siblings.len();
    let mut root: Option<std::path::PathBuf> = None;
    for (index, sib) in info.siblings.iter().enumerate() {
        progress(DownloadProgress {
            index,
            total,
            file: sib.rfilename.clone(),
        });
        let path = repo.get(&sib.rfilename).map_err(to_io)?;
        let candidate =
            snapshot_root(&path, &sib.rfilename).or_else(|| path.parent().map(|p| p.to_path_buf()));
        if root.is_none() || sib.rfilename == "config.json" {
            root = candidate;
        }
    }
    root.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("repo {repo_id} has no files"),
        )
    })
}

#[cfg(not(feature = "download"))]
fn download_snapshot(
    repo_id: &str,
    _revision: Option<&str>,
) -> std::io::Result<std::path::PathBuf> {
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{repo_id} is not a local dir and the `download` feature is disabled — clone the HF repo and pass its path"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_values_match_intenum() {
        assert_eq!(LFMModality::Text as i64, 1);
        assert_eq!(LFMModality::AudioIn as i64, 2);
        assert_eq!(LFMModality::AudioOut as i64, 3);
    }

    #[test]
    fn mel_emb_len_roundtrip() {
        // -(9 // -8) == 2 ; -(16 // -8) == 2 ; -(17 // -8) == 3
        assert_eq!(mel2emb_len(9), 2);
        assert_eq!(mel2emb_len(16), 2);
        assert_eq!(mel2emb_len(17), 3);
        assert_eq!(emb2mel_len(2), 16);
    }
}
