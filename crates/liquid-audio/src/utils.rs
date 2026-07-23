//! Host-owned model snapshot download support.

/// Per-file progress for [`snapshot_download_with`]. `index` is 0-based; `total`
/// is the sibling count.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub index: usize,
    pub total: usize,
    pub file: String,
}

/// Convenience download using hf-hub's conventional user cache. Credentials
/// are explicit: `None` means unauthenticated and never falls back to a token
/// file.
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

/// Download a complete snapshot into a host-selected cache directory.
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
        // No guessing at the parent directory: an unresolvable snapshot root
        // falls through to the NotFound below rather than a plausible wrong path.
        let candidate = snapshot_root(&path, &sib.rfilename);
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
