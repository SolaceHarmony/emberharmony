//! LFM2.5-Audio runtime — thin wrapper over llama.cpp's `llama-liquid-audio-cli`.
//!
//! Treats LFM2.5-Audio as a plain local model: speech in, speech/text out, on
//! CPU/Metal via llama.cpp GGUF. The CLI is taken verbatim from the official GGUF
//! model card. Every call errors clearly if the binary or GGUFs are missing,
//! rather than pretending to work.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

pub struct Lfm {
    bin: PathBuf,
    model: PathBuf,
    mmproj: PathBuf,
    vocoder: PathBuf,
    tokenizer: PathBuf,
    voice: String,
}

impl Lfm {
    pub fn from_env() -> Result<Self> {
        let quant = std::env::var("LFM_QUANT").unwrap_or_else(|_| "Q4_0".into());
        let model_dir: PathBuf = std::env::var("LFM_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_model_dir());
        let bin = resolve_bin()?;
        let f = |prefix: &str| model_dir.join(format!("{prefix}LFM2.5-Audio-1.5B-{quant}.gguf"));
        let lfm = Self {
            bin,
            model: f(""),
            mmproj: f("mmproj-"),
            vocoder: f("vocoder-"),
            tokenizer: f("tokenizer-"),
            voice: std::env::var("LFM_VOICE").unwrap_or_default(),
        };
        let missing: Vec<String> = [&lfm.model, &lfm.mmproj, &lfm.vocoder, &lfm.tokenizer]
            .iter()
            .filter(|p| !p.exists())
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        if !missing.is_empty() {
            bail!(
                "missing GGUF files in {}: {} — run setup.sh",
                model_dir.display(),
                missing.join(", ")
            );
        }
        Ok(lfm)
    }

    fn base(&self) -> Vec<String> {
        vec![
            "-m".into(), self.model.display().to_string(),
            "-mm".into(), self.mmproj.display().to_string(),
            "-mv".into(), self.vocoder.display().to_string(),
            "--tts-speaker-file".into(), self.tokenizer.display().to_string(),
        ]
    }

    fn run(&self, args: &[String]) -> Result<String> {
        let out = Command::new(&self.bin)
            .args(args)
            .output()
            .with_context(|| format!("failed to spawn {}", self.bin.display()))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(anyhow!(
                "llama-liquid-audio-cli failed ({}):\n{}",
                out.status.code().unwrap_or(-1),
                &err[err.len().saturating_sub(2000)..]
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// Speech -> text.
    pub fn asr(&self, in_wav: &Path) -> Result<String> {
        let mut a = self.base();
        a.extend(["-sys".into(), "Perform ASR.".into(), "--audio".into(), in_wav.display().to_string()]);
        self.run(&a)
    }

    /// Text -> speech WAV at `out_wav`.
    pub fn tts(&self, text: &str, out_wav: &Path) -> Result<()> {
        let sys = if self.voice.is_empty() {
            "Perform TTS.".to_string()
        } else {
            format!("Perform TTS. {}", self.voice)
        };
        let mut a = self.base();
        a.extend(["-sys".into(), sys, "-p".into(), text.into(), "--output".into(), out_wav.display().to_string()]);
        self.run(&a)?;
        Ok(())
    }

    /// Speech in -> (text reply, speech WAV). The conversational mode: LFM answers
    /// in its own voice and on the text channel at once. The text is what the loop
    /// inspects for the DELEGATE marker.
    pub fn interleaved(&self, in_wav: &Path, out_wav: &Path, system: &str) -> Result<String> {
        let mut a = self.base();
        a.extend([
            "-sys".into(), system.into(),
            "--audio".into(), in_wav.display().to_string(),
            "--output".into(), out_wav.display().to_string(),
        ]);
        self.run(&a)
    }
}

fn default_model_dir() -> PathBuf {
    // crate_dir/models
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn resolve_bin() -> Result<PathBuf> {
    if let Ok(b) = std::env::var("LFM_BIN") {
        let p = PathBuf::from(&b);
        if !p.exists() {
            bail!("LFM_BIN={b} does not exist");
        }
        return Ok(p);
    }
    // search PATH
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let cand = Path::new(dir).join("llama-liquid-audio-cli");
            if cand.exists() {
                return Ok(cand);
            }
        }
    }
    bail!("llama-liquid-audio-cli not found. Build it from llama.cpp PR #18641 and set LFM_BIN or add it to PATH (see setup.sh).")
}
