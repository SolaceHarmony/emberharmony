//! GLM subagent — the "hard work" delegate, reached as a plain HTTP model.
//!
//! GLM-5.1 over the ollama-cloud OpenAI-compatible API. The key is read from
//! `$OLLAMA_API_KEY` or the EmberHarmony auth store and is never logged. An
//! optional `bash` tool (off by default) lets the subagent actually do work.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

const SYSTEM: &str = "You are a capable engineering subagent invoked by a voice assistant to do the hard work. \
You receive a task, do it, and return a SHORT, plain-spoken summary the voice assistant can read aloud — \
no markdown, no code fences, two or three sentences at most. If you have the bash tool, use it to actually \
inspect or change things; if you do not, reason it through and give the best concrete answer or plan.";

pub struct Glm {
    base_url: String,
    model: String,
    key: String,
    client: reqwest::blocking::Client,
}

impl Glm {
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("GLM_BASE_URL").unwrap_or_else(|_| "https://ollama.com/v1".into());
        let model = std::env::var("GLM_MODEL").unwrap_or_else(|_| "glm-5.1".into());
        Ok(Self {
            base_url,
            model,
            key: load_key()?,
            client: reqwest::blocking::Client::new(),
        })
    }

    fn chat(&self, messages: &Value, tools: Option<&Value>) -> Result<Value> {
        let mut body = json!({ "model": self.model, "messages": messages, "stream": false });
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.key)
            .json(&body)
            .send()
            .context("GLM request failed")?;
        if !resp.status().is_success() {
            let code = resp.status();
            let detail = resp.text().unwrap_or_default();
            return Err(anyhow!("GLM API HTTP {code}: {}", &detail[..detail.len().min(500)]));
        }
        let payload: Value = resp.json().context("GLM response not JSON")?;
        Ok(payload["choices"][0]["message"].clone())
    }

    /// Run GLM on `task`, returning a short, speakable result.
    pub fn run_subagent(&self, task: &str, allow_exec: bool, cwd: &str, max_steps: usize) -> Result<String> {
        let mut messages = json!([
            { "role": "system", "content": SYSTEM },
            { "role": "user", "content": task },
        ]);
        let tools = if allow_exec { Some(bash_tool()) } else { None };

        for _ in 0..max_steps {
            let msg = self.chat(&messages, tools.as_ref())?;
            let calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
            if calls.is_empty() {
                let content = msg["content"].as_str().unwrap_or("").trim().to_string();
                return Ok(if content.is_empty() {
                    "(the subagent returned nothing)".into()
                } else {
                    content
                });
            }
            // Echo the assistant turn (with its tool calls) into history.
            messages.as_array_mut().unwrap().push(json!({
                "role": "assistant",
                "content": msg["content"].as_str().unwrap_or(""),
                "tool_calls": calls,
            }));
            for call in &calls {
                let name = call["function"]["name"].as_str().unwrap_or("");
                let args: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
                let result = if name == "bash" && allow_exec {
                    let cmd = args["command"].as_str().unwrap_or("");
                    eprintln!("  [subagent bash] {cmd}");
                    run_bash(cmd, cwd)
                } else {
                    format!("tool '{name}' is not available (execution disabled)")
                };
                messages.as_array_mut().unwrap().push(json!({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }));
            }
        }
        Ok("(the subagent hit its step limit without finishing)".into())
    }
}

fn bash_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and get back stdout/stderr. Use to inspect or change files, run tests, git, etc.",
            "parameters": {
                "type": "object",
                "properties": { "command": { "type": "string", "description": "The shell command to run" } },
                "required": ["command"]
            }
        }
    }])
}

fn run_bash(command: &str, cwd: &str) -> String {
    let out = Command::new("bash").arg("-c").arg(command).current_dir(cwd).output();
    match out {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).to_string();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            if s.len() > 6000 {
                s.truncate(6000);
                s.push_str("\n…[truncated]");
            }
            format!("[exit {}]\n{}", o.status.code().unwrap_or(-1), s.trim())
        }
        Err(e) => format!("[failed to run: {e}]"),
    }
}

/// ollama-cloud API key from env or the EmberHarmony auth store. Never printed.
fn load_key() -> Result<String> {
    if let Ok(k) = std::env::var("OLLAMA_API_KEY") {
        if !k.trim().is_empty() {
            return Ok(k.trim().to_string());
        }
    }
    let path: PathBuf = std::env::var("EMBERHARMONY_AUTH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join(".local/share/emberharmony/auth.json")
        });
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("no OLLAMA_API_KEY and auth store unreadable at {}", path.display()))?;
    let v: Value = serde_json::from_str(&data).context("auth.json is not valid JSON")?;
    let entry = v.get("ollama-cloud").or_else(|| v.get("ollama_cloud"));
    entry
        .and_then(|e| e.get("key"))
        .and_then(|k| k.as_str())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| anyhow!("no ollama-cloud key in {}", path.display()))
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}
