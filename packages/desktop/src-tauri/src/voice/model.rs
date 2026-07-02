//! Model management for the native LFM2-Audio voice provider: explicit download
//! (HF repo id + revision + keychain-stored token, with per-file progress), a native
//! folder picker for a local snapshot, and HF-token storage in the OS keychain.
//!
//! Decoupled + fail-hard: this layer only *acquires* a model into a local directory.
//! Nothing here ever runs at session start — `voice_start` loads the local dir or fails
//! (see [`super::control::plan`] / [`super::runtime`]). Secrets never cross into the
//! webview: the token is set/queried-for-presence here and read natively at download time.

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::{AppHandle, State};

use super::threads::ThreadManager;

/// Keychain coordinates for the Hugging Face token. The token lives ONLY here — never in
/// the settings JSON, never sent to the webview.
const KEYRING_SERVICE: &str = "ai.ofharmony.emberharmony.voice";
const KEYRING_USER: &str = "huggingface";

/// Streamed download progress (mirrors the tagged shape of [`super::control::VoiceEvent`]).
/// `index`/`total` are 1-based file counts. No faked byte progress — hf-hub's sync API
/// reports per file, not per byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DownloadEvent {
    /// total file count is known (repo listing succeeded)
    Started { total: u32 },
    /// fetching file `index` of `total`
    File {
        index: u32,
        total: u32,
        name: String,
    },
    /// finished; `dir` is the local snapshot directory to set as the active model
    Done { dir: String },
    /// hard failure — the download did not complete; no model is marked active
    Error { message: String },
}

fn token_entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(|e| format!("keychain: {e}"))
}

/// The stored HF token, or `None` if unset/empty. `Err` only on a real keychain fault.
fn hf_token() -> Result<Option<String>, String> {
    match token_entry()?.get_password() {
        Ok(t) if !t.trim().is_empty() => Ok(Some(t)),
        Ok(_) => Ok(None),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(format!("keychain: {e}")),
    }
}

/// Normalize a user-entered model reference to a bare `owner/name` repo id, extracting any
/// revision from a pasted Hub URL (`.../tree/<rev>`, `/blob/<rev>`, `/resolve/<rev>`).
/// Fail-hard: returns `Err` for anything that is not `owner/name`.
fn normalize_repo_id(input: &str) -> Result<(String, Option<String>), String> {
    let invalid =
        || format!("`{input}` is not a valid Hugging Face repo id (expected `owner/name`).");
    let s = input.trim();
    if s.is_empty() {
        return Err("Enter a Hugging Face model id (owner/name) or a model URL.".into());
    }
    // Strip scheme + known Hub hosts.
    let s = s.strip_prefix("https://").unwrap_or(s);
    let s = s.strip_prefix("http://").unwrap_or(s);
    let s = s.strip_prefix("huggingface.co/").unwrap_or(s);
    let s = s.strip_prefix("hf.co/").unwrap_or(s);
    let segments: Vec<&str> = s.split('/').filter(|x| !x.is_empty()).collect();
    if segments.len() < 2 {
        return Err(invalid());
    }
    let (owner, name) = (segments[0], segments[1]);
    let valid = |t: &str| {
        !t.is_empty()
            && t.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    };
    if !valid(owner) || !valid(name) {
        return Err(invalid());
    }
    // A pasted `.../tree|blob|resolve/<rev>` carries the revision.
    let revision = if segments.len() >= 4 && matches!(segments[2], "tree" | "blob" | "resolve") {
        Some(segments[3].to_string())
    } else {
        None
    };
    Ok((format!("{owner}/{name}"), revision))
}

/// Append a token hint to auth-shaped errors (the raw error is always preserved).
fn annotate(err: &std::io::Error) -> String {
    let msg = err.to_string();
    let lower = msg.to_lowercase();
    if lower.contains("401") || lower.contains("403") || lower.contains("unauthorized") {
        format!(
            "{msg} — this repo may be gated or private; add your Hugging Face token in Settings."
        )
    } else {
        msg
    }
}

#[derive(Default)]
pub struct ModelDownloadRuntime {
    threads: ThreadManager,
}

impl ModelDownloadRuntime {
    fn spawn(&self, f: impl FnOnce() + Send + 'static) -> Result<(), String> {
        self.threads.spawn_if_idle(
            "voice-model-download",
            "voice model download already running",
            f,
        )
    }
}

/// Download a model snapshot into the HF cache with per-file progress.
///
/// `source` is a repo id or a pasted Hub URL; `revision` (if non-empty) overrides any
/// revision parsed from the URL. The token is read natively from the keychain — it is never
/// passed from the webview. Returns after the worker thread is spawned; the terminal
/// `Done { dir }` / `Error { message }` over `channel` is authoritative. On `Done`, the
/// caller persists `dir` as the active `model_dir`. Fail-hard: a partial/failed download
/// never yields a `Done`.
#[tauri::command]
pub async fn voice_model_download(
    runtime: State<'_, ModelDownloadRuntime>,
    source: String,
    revision: Option<String>,
    channel: Channel<DownloadEvent>,
) -> Result<(), String> {
    let (repo_id, url_rev) = normalize_repo_id(&source)?;
    let revision = revision
        .map(|r| r.trim().to_string())
        .filter(|r| !r.is_empty())
        .or(url_rev);
    let token = hf_token()?;

    runtime.spawn(move || {
        let result = liquid_audio::snapshot_download_with(
            &repo_id,
            revision.as_deref(),
            token.as_deref(),
            |p| {
                if p.index == 0 {
                    let _ = channel.send(DownloadEvent::Started {
                        total: p.total as u32,
                    });
                }
                let _ = channel.send(DownloadEvent::File {
                    index: p.index as u32 + 1,
                    total: p.total as u32,
                    name: p.file,
                });
            },
        );
        let terminal = match result {
            Ok(dir) => DownloadEvent::Done {
                dir: dir.to_string_lossy().into_owned(),
            },
            Err(e) => DownloadEvent::Error {
                message: annotate(&e),
            },
        };
        let _ = channel.send(terminal);
    })
}

/// Open a native folder picker; returns the chosen directory path (or `None` if cancelled).
/// `async` so the blocking dialog runs off the main thread (the documented pattern).
#[tauri::command]
pub async fn voice_pick_model_dir(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let picked = app
        .dialog()
        .file()
        .set_title("Select local voice model directory")
        .blocking_pick_folder();
    match picked {
        Some(path) => Ok(Some(
            path.into_path()
                .map_err(|e| format!("invalid folder path: {e}"))?
                .to_string_lossy()
                .into_owned(),
        )),
        None => Ok(None),
    }
}

/// Store (non-empty) or clear (empty) the Hugging Face token in the OS keychain.
#[tauri::command]
pub async fn voice_hf_token_set(token: String) -> Result<(), String> {
    let entry = token_entry()?;
    let token = token.trim();
    if token.is_empty() {
        return match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(format!("keychain: {e}")),
        };
    }
    entry
        .set_password(token)
        .map_err(|e| format!("keychain: {e}"))
}

/// Whether a Hugging Face token is stored (presence only — the value is never returned).
#[tauri::command]
pub async fn voice_hf_token_status() -> Result<bool, String> {
    Ok(hf_token()?.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_plain_repo_id() {
        assert_eq!(
            normalize_repo_id("LiquidAI/LFM2.5-Audio-1.5B").unwrap(),
            ("LiquidAI/LFM2.5-Audio-1.5B".to_string(), None)
        );
    }

    #[test]
    fn normalize_strips_url_and_extracts_revision() {
        assert_eq!(
            normalize_repo_id("https://huggingface.co/owner/name/tree/my-rev").unwrap(),
            ("owner/name".to_string(), Some("my-rev".to_string()))
        );
        assert_eq!(
            normalize_repo_id("huggingface.co/owner/name").unwrap(),
            ("owner/name".to_string(), None)
        );
        // /tree with no revision segment → still just the id, no revision.
        assert_eq!(
            normalize_repo_id("owner/name/tree").unwrap(),
            ("owner/name".to_string(), None)
        );
    }

    #[test]
    fn normalize_rejects_malformed() {
        assert!(normalize_repo_id("").is_err());
        assert!(normalize_repo_id("just-a-name").is_err());
        assert!(normalize_repo_id("owner/").is_err());
        assert!(normalize_repo_id("/name").is_err());
        assert!(normalize_repo_id("bad owner/name").is_err());
    }

    #[test]
    fn download_event_tagged_serialization() {
        let f = serde_json::to_value(DownloadEvent::File {
            index: 2,
            total: 9,
            name: "model.safetensors".into(),
        })
        .unwrap();
        assert_eq!(f["type"], "file");
        assert_eq!(f["index"], 2);
        assert_eq!(f["name"], "model.safetensors");
        let d = serde_json::to_value(DownloadEvent::Done { dir: "/x/y".into() }).unwrap();
        assert_eq!(d["type"], "done");
        assert_eq!(d["dir"], "/x/y");
    }
}
