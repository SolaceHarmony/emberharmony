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
/// feature (on by default).
pub fn get_model_dir(repo_or_path: &str, revision: Option<&str>) -> std::io::Result<std::path::PathBuf> {
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

/// `snapshot_download(repo_id, revision=...)` — fetch every file in the repo into
/// the HF cache and return the snapshot directory (parent of `config.json`).
#[cfg(feature = "download")]
fn download_snapshot(repo_id: &str, revision: Option<&str>) -> std::io::Result<std::path::PathBuf> {
    use hf_hub::{api::sync::Api, Repo, RepoType};
    let to_io = |e: hf_hub::api::sync::ApiError| std::io::Error::new(std::io::ErrorKind::Other, format!("hf-hub: {e}"));

    let api = Api::new().map_err(to_io)?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string())),
        None => api.model(repo_id.to_string()),
    };

    // List then fetch every sibling (snapshot_download grabs the whole repo).
    let info = repo.info().map_err(to_io)?;
    let mut root: Option<std::path::PathBuf> = None;
    for sib in &info.siblings {
        let path = repo.get(&sib.rfilename).map_err(to_io)?;
        if sib.rfilename == "config.json" {
            root = path.parent().map(|p| p.to_path_buf());
        }
    }
    root.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, format!("repo {repo_id} has no config.json"))
    })
}

#[cfg(not(feature = "download"))]
fn download_snapshot(repo_id: &str, _revision: Option<&str>) -> std::io::Result<std::path::PathBuf> {
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
