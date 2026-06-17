//! Local voice loop — LFM2.5-Audio up front, GLM subagent for the hard work.
//!
//!   mic → LFM2.5-Audio (interleaved: converses in its own voice + emits text)
//!           ├─ no marker        → play LFM's own spoken reply        (small talk)
//!           └─ "DELEGATE: task" → GLM-5.1 subagent does the work → LFM speaks the result
//!
//! No LiveKit, no brain bridge. LFM2.5-Audio is the front-line intelligence; its one
//! primitive tool is the DELEGATE text-marker the loop watches for (the audio model
//! has no native function calling). Runs as a native Rust binary like the rest of
//! the native voice code.
//!
//! Env: LFM_ROUTE=marker|chat|delegate, LFM_ALLOW_EXEC=1, plus the LFM_*/GLM_* vars
//! documented in lfm.rs / glm.rs / audio.rs.

mod audio;
mod glm;
mod lfm;

use anyhow::Result;

const CONVERSE_SYSTEM: &str = "Respond with interleaved text and audio. You are a warm, brief voice assistant. \
Chat naturally and answer simple questions yourself in one or two short spoken sentences. But when the user asks \
for real engineering, coding, research, or file/system work, do NOT attempt it yourself. Instead, briefly say \
you'll get your engineer on it, and on the TEXT channel output exactly one line of the form: \
DELEGATE: <a clear, self-contained description of the task>. Only emit DELEGATE for genuine work, never for small talk.";

fn extract_delegation(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim_start();
        if t.len() >= 9 && t[..9].eq_ignore_ascii_case("DELEGATE:") {
            let task = t[9..].trim().to_string();
            if !task.is_empty() {
                return Some(task);
            }
        }
    }
    None
}

fn main() -> Result<()> {
    let route = std::env::var("LFM_ROUTE").unwrap_or_else(|_| "marker".into());
    let allow_exec = std::env::var("LFM_ALLOW_EXEC").as_deref() == Ok("1");

    let lfm = match lfm::Lfm::from_env() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("LFM2.5-Audio not ready: {e}");
            eprintln!("Run ./setup.sh to build llama-liquid-audio-cli (llama.cpp PR #18641) and download the GGUFs.");
            std::process::exit(1);
        }
    };

    // GLM is only required when delegation can happen.
    let glm = match glm::Glm::from_env() {
        Ok(g) => Some(g),
        Err(e) => {
            if route == "chat" {
                eprintln!("[lfm-voice] no GLM configured ({e}); running LFM-only (route=chat).");
                None
            } else {
                eprintln!("GLM subagent not configured: {e}");
                std::process::exit(1);
            }
        }
    };

    let work = std::env::temp_dir().join("lfm-voice");
    std::fs::create_dir_all(&work)?;
    let in_wav = work.join("in.wav");
    let out_wav = work.join("out.wav");
    let reply_wav = work.join("reply.wav");
    let cwd = std::env::current_dir()?.display().to_string();

    println!("[lfm-voice] route={route} allow_exec={allow_exec}. Ctrl-C to quit.");
    loop {
        if !audio::record_utterance(&in_wav)? {
            continue;
        }

        if route == "delegate" {
            let user_text = lfm.asr(&in_wav)?;
            println!("  you: {user_text}");
            let result = delegate(&glm, &user_text, allow_exec, &cwd)?;
            println!("  glm: {result}");
            lfm.tts(&result, &reply_wav)?;
            audio::play_wav(&reply_wav)?;
            continue;
        }

        // marker / chat: LFM converses and may flag work on the text channel.
        let text = lfm.interleaved(&in_wav, &out_wav, CONVERSE_SYSTEM)?;
        println!("  lfm(text): {text}");

        let task = if route == "chat" { None } else { extract_delegation(&text) };
        match task {
            Some(task) => {
                // Acknowledge in LFM's own voice first (if it produced audio), then delegate.
                if out_wav.exists() && std::fs::metadata(&out_wav).map(|m| m.len() > 0).unwrap_or(false) {
                    let _ = audio::play_wav(&out_wav);
                }
                println!("  → delegating to GLM: {task}");
                let result = delegate(&glm, &task, allow_exec, &cwd)?;
                println!("  glm: {result}");
                lfm.tts(&result, &reply_wav)?;
                audio::play_wav(&reply_wav)?;
            }
            None => {
                audio::play_wav(&out_wav)?;
            }
        }
    }
}

fn delegate(glm: &Option<glm::Glm>, task: &str, allow_exec: bool, cwd: &str) -> Result<String> {
    match glm {
        Some(g) => g.run_subagent(task, allow_exec, cwd, 8),
        None => Ok("I don't have my engineer hooked up right now.".to_string()),
    }
}
