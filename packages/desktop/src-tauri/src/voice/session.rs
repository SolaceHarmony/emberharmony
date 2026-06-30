//! Session bridge: turns one voice utterance into a session prompt and streams
//! the reply back out of the server's SSE feed. Rust port of the turn logic in
//! `packages/emberharmony/src/voice/bridge.ts` (see its test harness for the
//! behavioural contract this mirrors).
//!
//! The HTTP/SSE transport (reqwest) lands in Phase 1 alongside cpal/STT/TTS; the
//! load-bearing, fiddly part — the per-turn event state machine — is implemented
//! and tested here so the wiring on top is mechanical. A key win of doing this in
//! Rust: the async runner can wrap each SSE read in `tokio::time::timeout`, so the
//! staleness watchdog fires even on a fully-silent connection — closing the gap
//! the TS version still has (its check only runs when an event arrives).

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde_json::{Value, json};

pub const VOICE_SYSTEM_PROMPT: &str = concat!(
    "The user is speaking to you by voice and hears your replies as speech. ",
    "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations. ",
    "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two, ",
    "then ask whether to proceed -- they will confirm out loud."
);

/// Configuration for one bridged session (Phase 1 input for the async runner).
#[derive(Debug, Clone)]
pub struct SessionBridgeConfig {
    /// EmberHarmony server origin, e.g. `http://localhost:4096`.
    pub server_url: String,
    /// Project directory the session belongs to.
    pub directory: String,
    /// Session to bridge into.
    pub session_id: String,
    /// Basic auth, if the server is password-protected.
    pub username: Option<String>,
    pub password: Option<String>,
    /// `plan` / `build` for the turn — the safety gate decides this per utterance.
    pub agent: Option<String>,
    /// Model override used when the spoken turn delegates work to the session.
    pub model: Option<SessionBridgeModel>,
    /// Model variant for the delegated session prompt.
    pub variant: Option<String>,
    /// Extra per-message system instructions attached to every voice prompt.
    pub system: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBridgeModel {
    pub provider_id: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionBridgeEvent {
    Delta { reply_id: String, text: String },
    Done,
}

/// A parsed server SSE event, narrowed to the variants the turn loop cares about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    Connected,
    Heartbeat,
    PartUpdated {
        session_id: String,
        message_id: String,
        part_type: String,
        delta: Option<String>,
    },
    MessageUpdated {
        session_id: String,
    },
    Idle {
        session_id: String,
    },
    Error {
        session_id: Option<String>,
        error: String,
    },
    /// Any other / unknown event — ignored, never ends or branches the turn.
    Other,
}

/// What the reducer wants the runner to do with one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Emit a reply text chunk under a stable id (TTS speaks it as one utterance).
    Delta { reply_id: String, text: String },
    /// Turn finished cleanly (session went idle).
    Done,
    /// Session reported an error scoped to this turn.
    Failed(String),
    /// No session events for longer than the staleness window — dead connection.
    TimedOut,
    /// Nothing to do.
    Ignore,
}

/// Parse one `data:` SSE line into a [`SessionEvent`].
///
/// Returns `None` for non-`data:` lines, empty payloads, and — crucially — for a
/// malformed/truncated JSON frame: a single bad frame must never abort the turn
/// (mirrors the `try { JSON.parse } catch { continue }` guard in bridge.ts).
pub fn event_from_data_line(line: &str) -> Option<SessionEvent> {
    let data = line.strip_prefix("data:")?.trim();
    if data.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(data).ok()?;
    Some(parse_event(&value))
}

/// Map a decoded JSON event object to a [`SessionEvent`].
pub fn parse_event(v: &Value) -> SessionEvent {
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    let props = v.get("properties");
    match ty {
        "server.connected" => SessionEvent::Connected,
        "server.heartbeat" => SessionEvent::Heartbeat,
        "message.part.updated" => {
            let part = props.and_then(|p| p.get("part"));
            let session_id = part
                .and_then(|p| p.get("sessionID"))
                .and_then(Value::as_str);
            match (part, session_id) {
                (Some(part), Some(session_id)) => SessionEvent::PartUpdated {
                    session_id: session_id.to_string(),
                    message_id: part
                        .get("messageID")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    part_type: part
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    delta: props
                        .and_then(|p| p.get("delta"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                },
                _ => SessionEvent::Other,
            }
        }
        "message.updated" => match props
            .and_then(|p| p.get("info"))
            .and_then(|i| i.get("sessionID"))
            .and_then(Value::as_str)
        {
            Some(session_id) => SessionEvent::MessageUpdated {
                session_id: session_id.to_string(),
            },
            None => SessionEvent::Other,
        },
        "session.idle" => match props
            .and_then(|p| p.get("sessionID"))
            .and_then(Value::as_str)
        {
            Some(session_id) => SessionEvent::Idle {
                session_id: session_id.to_string(),
            },
            None => SessionEvent::Other,
        },
        "session.error" => {
            let session_id = props
                .and_then(|p| p.get("sessionID"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let error = props
                .and_then(|p| p.get("error"))
                // a JSON string error -> its clean unquoted text; non-strings -> JSON
                .map(|e| {
                    e.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| e.to_string())
                })
                .or_else(|| props.map(Value::to_string))
                .unwrap_or_default();
            SessionEvent::Error { session_id, error }
        }
        _ => SessionEvent::Other,
    }
}

/// The per-turn event state machine. Coalesces every assistant text part under
/// the FIRST part's message id (so TTS hears one continuous utterance), ends the
/// turn only on `session.idle`, scopes everything to this session, and bumps an
/// activity clock on every relevant event so the staleness window only trips on a
/// genuinely dead feed.
pub struct TurnReducer {
    session_id: String,
    reply_id: Option<String>,
    last_activity_ms: u64,
    stale_ms: u64,
}

impl TurnReducer {
    /// Default staleness window — matches bridge.ts (`STALE_MS = 120_000`,
    /// tolerating ~4 missed 30s heartbeats).
    pub const DEFAULT_STALE_MS: u64 = 120_000;

    pub fn new(session_id: impl Into<String>, start_ms: u64) -> Self {
        Self {
            session_id: session_id.into(),
            reply_id: None,
            last_activity_ms: start_ms,
            stale_ms: Self::DEFAULT_STALE_MS,
        }
    }

    /// Drive one event at logical time `now_ms`. The staleness check runs first,
    /// so a long-silent feed times out on the next event regardless of its kind.
    pub fn step(&mut self, ev: &SessionEvent, now_ms: u64) -> Step {
        if now_ms.saturating_sub(self.last_activity_ms) > self.stale_ms {
            return Step::TimedOut;
        }
        match ev {
            SessionEvent::PartUpdated {
                session_id,
                message_id,
                part_type,
                delta,
            } if *session_id == self.session_id => {
                self.last_activity_ms = now_ms;
                if part_type == "text" {
                    if let Some(text) = delta {
                        let reply_id = self
                            .reply_id
                            .get_or_insert_with(|| message_id.clone())
                            .clone();
                        return Step::Delta {
                            reply_id,
                            text: text.clone(),
                        };
                    }
                }
                Step::Ignore
            }
            SessionEvent::MessageUpdated { session_id } if *session_id == self.session_id => {
                self.last_activity_ms = now_ms;
                Step::Ignore
            }
            SessionEvent::Heartbeat => {
                self.last_activity_ms = now_ms;
                Step::Ignore
            }
            SessionEvent::Idle { session_id } if *session_id == self.session_id => Step::Done,
            // An error with no sessionID applies to us; one scoped to a different
            // session is ignored (mirrors bridge.ts's `if (sid && sid !== ours) continue`).
            SessionEvent::Error { session_id, error }
                if session_id.as_deref().map_or(true, |s| s == self.session_id) =>
            {
                Step::Failed(error.clone())
            }
            _ => Step::Ignore,
        }
    }

    /// Check the staleness window without waiting for the next SSE frame. The TS
    /// LiveKit bridge can only discover a dead silent socket when another event
    /// arrives; the native runner polls this once per second so a severed stream
    /// terminates on time.
    pub fn tick(&mut self, now_ms: u64) -> Step {
        if now_ms.saturating_sub(self.last_activity_ms) > self.stale_ms {
            return Step::TimedOut;
        }
        Step::Ignore
    }
}

pub async fn run_turn(
    cfg: SessionBridgeConfig,
    text: String,
    cancel: Arc<AtomicBool>,
    mut sink: impl FnMut(SessionBridgeEvent) -> bool + Send + 'static,
) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(150))
        .build()
        .map_err(|e| format!("session bridge client: {e}"))?;
    let mut events = open_events(&client, &cfg).await?;

    // Pull the server.connected frame before posting the prompt so the stream
    // cannot miss the first reply delta.
    let _ = next_event(&mut events, &cancel).await?;

    post_prompt(&client, &cfg, &text, &cancel).await?;

    let start = Instant::now();
    let mut reducer = TurnReducer::new(&cfg.session_id, 0);
    loop {
        if cancel.load(Ordering::SeqCst) {
            abort_prompt(&client, &cfg).await;
            return Ok(());
        }

        match tokio::time::timeout(Duration::from_secs(1), next_event(&mut events, &cancel)).await {
            Ok(Ok(Some(event))) => match reducer.step(&event, elapsed_ms(start)) {
                Step::Delta { reply_id, text } => {
                    if !sink(SessionBridgeEvent::Delta { reply_id, text }) {
                        abort_prompt(&client, &cfg).await;
                        return Ok(());
                    }
                }
                Step::Done => {
                    let _ = sink(SessionBridgeEvent::Done);
                    return Ok(());
                }
                Step::Failed(error) => return Err(format!("session error: {error}")),
                Step::TimedOut => return Err("session reply timed out".into()),
                Step::Ignore => {}
            },
            Ok(Ok(None)) => return Err("session event stream closed".into()),
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                if matches!(reducer.tick(elapsed_ms(start)), Step::TimedOut) {
                    abort_prompt(&client, &cfg).await;
                    return Err("session reply timed out".into());
                }
            }
        }
    }
}

type EventStream = futures::stream::BoxStream<'static, Result<Vec<u8>, String>>;

struct SseStream {
    chunks: EventStream,
    buffer: String,
}

async fn open_events(
    client: &reqwest::Client,
    cfg: &SessionBridgeConfig,
) -> Result<SseStream, String> {
    let response = with_auth(
        client
            .get(format!("{}/event", cfg.server_url.trim_end_matches('/')))
            .header("accept", "text/event-stream")
            .header("x-emberharmony-directory", url_encode(&cfg.directory)),
        cfg,
    )
    .send()
    .await
    .map_err(|e| format!("event stream failed: {e}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("event stream failed: {status} {body}"));
    }
    Ok(SseStream {
        chunks: response
            .bytes_stream()
            .map(|chunk| {
                chunk
                    .map(|bytes| bytes.to_vec())
                    .map_err(|e| format!("event stream read failed: {e}"))
            })
            .boxed(),
        buffer: String::new(),
    })
}

async fn next_event(
    stream: &mut SseStream,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<SessionEvent>, String> {
    loop {
        if let Some(event) = drain_event(&mut stream.buffer) {
            return Ok(Some(event));
        }
        if cancel.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let Some(chunk) = stream.chunks.next().await else {
            return Ok(None);
        };
        let chunk = chunk?;
        stream.buffer.push_str(&String::from_utf8_lossy(&chunk));
    }
}

fn drain_event(buffer: &mut String) -> Option<SessionEvent> {
    loop {
        let Some(boundary) = buffer.find("\n\n") else {
            return None;
        };
        let chunk = buffer[..boundary].to_string();
        buffer.replace_range(..boundary + 2, "");
        for line in chunk.lines() {
            if let Some(event) = event_from_data_line(line) {
                return Some(event);
            }
        }
    }
}

async fn post_prompt(
    client: &reqwest::Client,
    cfg: &SessionBridgeConfig,
    text: &str,
    cancel: &Arc<AtomicBool>,
) -> Result<(), String> {
    if cancel.load(Ordering::SeqCst) {
        return Ok(());
    }
    let mut body = json!({
        "parts": [{ "type": "text", "text": text }],
    });
    if let Some(model) = &cfg.model {
        body["model"] = json!({
            "providerID": model.provider_id.as_str(),
            "modelID": model.model_id.as_str(),
        });
    }
    if let Some(agent) = cfg.agent.as_deref().filter(|s| !s.trim().is_empty()) {
        body["agent"] = json!(agent);
    }
    if let Some(system) = cfg.system.as_deref().filter(|s| !s.trim().is_empty()) {
        body["system"] = json!(system);
    }
    if let Some(variant) = cfg.variant.as_deref().filter(|s| !s.trim().is_empty()) {
        body["variant"] = json!(variant);
    }
    let response = with_auth(
        client
            .post(format!(
                "{}/session/{}/prompt_async",
                cfg.server_url.trim_end_matches('/'),
                cfg.session_id.as_str()
            ))
            .header("content-type", "application/json")
            .header("x-emberharmony-directory", url_encode(&cfg.directory))
            .json(&body),
        cfg,
    )
    .send()
    .await
    .map_err(|e| format!("session prompt failed: {e}"))?;
    if response.status().is_success() {
        return Ok(());
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(format!("session prompt failed: {status} {body}"))
}

async fn abort_prompt(client: &reqwest::Client, cfg: &SessionBridgeConfig) {
    let _ = with_auth(
        client
            .post(format!(
                "{}/session/{}/abort",
                cfg.server_url.trim_end_matches('/'),
                cfg.session_id.as_str()
            ))
            .header("x-emberharmony-directory", url_encode(&cfg.directory)),
        cfg,
    )
    .send()
    .await;
}

fn with_auth(req: reqwest::RequestBuilder, cfg: &SessionBridgeConfig) -> reqwest::RequestBuilder {
    match cfg.password.as_deref() {
        Some(password) => req.basic_auth(
            cfg.username.as_deref().unwrap_or("emberharmony"),
            Some(password),
        ),
        None => req,
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

pub(crate) fn url_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                [b as char, '\0', '\0']
            }
            _ => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                [
                    '%',
                    HEX[(b >> 4) as usize] as char,
                    HEX[(b & 0x0F) as usize] as char,
                ]
            }
        })
        .filter(|c| *c != '\0')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SID: &str = "ses_test";

    fn part(message_id: &str, delta: &str) -> SessionEvent {
        SessionEvent::PartUpdated {
            session_id: SID.into(),
            message_id: message_id.into(),
            part_type: "text".into(),
            delta: Some(delta.into()),
        }
    }

    /// Run a script of (event, now_ms) through a reducer, collecting deltas and
    /// the terminal step. Mirrors `drain()` in the TS bridge test harness.
    fn run(script: &[(SessionEvent, u64)]) -> (Vec<(String, String)>, Step) {
        let mut r = TurnReducer::new(SID, 1_000_000);
        let mut deltas = Vec::new();
        for (ev, now) in script {
            match r.step(ev, *now) {
                Step::Delta { reply_id, text } => deltas.push((reply_id, text)),
                Step::Ignore => {}
                terminal => return (deltas, terminal),
            }
        }
        (deltas, Step::Ignore)
    }

    #[test]
    fn shipped_regression_tool_step_finalizes_but_reply_continues() {
        // tool-call step finalizes its assistant message (message.updated) mid-turn;
        // the continuation (m2) must still stream, under m1's id, ending on idle.
        let (deltas, terminal) = run(&[
            (SessionEvent::Connected, 1_000_000),
            (part("m1", "Let me check. "), 1_000_000),
            (
                SessionEvent::MessageUpdated {
                    session_id: SID.into(),
                },
                1_000_000,
            ),
            (part("m2", "The answer is 42."), 1_000_000),
            (
                SessionEvent::Idle {
                    session_id: SID.into(),
                },
                1_000_000,
            ),
        ]);
        let text: String = deltas.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, "Let me check. The answer is 42.");
        let ids: Vec<&str> = deltas.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["m1", "m1"]); // coalesced under the first message id
        assert_eq!(terminal, Step::Done);
    }

    #[test]
    fn multi_message_turn_coalesces_under_one_reply_id() {
        let (deltas, terminal) = run(&[
            (part("m1", "a"), 1_000_000),
            (part("m2", "b"), 1_000_000),
            (part("m3", "c"), 1_000_000),
            (
                SessionEvent::Idle {
                    session_id: SID.into(),
                },
                1_000_000,
            ),
        ]);
        let text: String = deltas.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, "abc");
        assert!(deltas.iter().all(|(id, _)| id == "m1"));
        assert_eq!(terminal, Step::Done);
    }

    #[test]
    fn ignores_parts_and_idle_for_a_different_session() {
        let foreign_part = SessionEvent::PartUpdated {
            session_id: "other".into(),
            message_id: "x1".into(),
            part_type: "text".into(),
            delta: Some("FOREIGN".into()),
        };
        let (deltas, terminal) = run(&[
            (SessionEvent::Connected, 1_000_000),
            (foreign_part, 1_000_000),
            (
                SessionEvent::Idle {
                    session_id: "other".into(),
                },
                1_000_000,
            ), // must NOT end our turn
            (part("m1", "ours"), 1_000_000),
            (
                SessionEvent::Idle {
                    session_id: SID.into(),
                },
                1_000_000,
            ),
        ]);
        let text: String = deltas.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, "ours");
        assert_eq!(terminal, Step::Done);
    }

    #[test]
    fn scoped_error_fails_turn_foreign_error_ignored() {
        let mut r = TurnReducer::new(SID, 0);
        assert_eq!(
            r.step(
                &SessionEvent::Error {
                    session_id: Some("other".into()),
                    error: "x".into()
                },
                0
            ),
            Step::Ignore
        );
        assert_eq!(
            r.step(
                &SessionEvent::Error {
                    session_id: None,
                    error: "no-scope".into()
                },
                0
            ),
            Step::Failed("no-scope".into()) // an unscoped error applies to us
        );
    }

    #[test]
    fn heartbeats_bump_activity_and_prevent_false_timeout() {
        // 270s of wall clock elapses, but each gap between activity is < 120s.
        let (deltas, terminal) = run(&[
            (SessionEvent::Connected, 1_000_000),
            (part("m1", "a"), 1_000_000),
            (SessionEvent::Heartbeat, 1_090_000),
            (SessionEvent::Heartbeat, 1_180_000),
            (part("m1", "b"), 1_270_000),
            (
                SessionEvent::Idle {
                    session_id: SID.into(),
                },
                1_270_000,
            ),
        ]);
        let text: String = deltas.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, "ab");
        assert_eq!(terminal, Step::Done);
    }

    #[test]
    fn staleness_fires_when_feed_goes_silent() {
        let (deltas, terminal) = run(&[
            (SessionEvent::Connected, 1_000_000),
            (part("m1", "a"), 1_000_000),
            (part("m1", "b"), 1_130_001), // 130s later, no heartbeat between
        ]);
        assert_eq!(
            deltas.iter().map(|(_, t)| t.as_str()).collect::<String>(),
            "a"
        );
        assert_eq!(terminal, Step::TimedOut);
    }

    #[test]
    fn parses_data_lines_and_skips_malformed_frames() {
        // valid frame
        assert_eq!(
            event_from_data_line(r#"data: {"type":"server.heartbeat"}"#),
            Some(SessionEvent::Heartbeat)
        );
        // text part
        assert_eq!(
            event_from_data_line(
                r#"data: {"type":"message.part.updated","properties":{"part":{"sessionID":"ses_test","type":"text","messageID":"m1"},"delta":"hi"}}"#
            ),
            Some(part("m1", "hi"))
        );
        // malformed JSON -> skipped, not fatal
        assert_eq!(event_from_data_line("data: {this is not json}"), None);
        // non-data line and empty data -> skipped
        assert_eq!(event_from_data_line(": keep-alive"), None);
        assert_eq!(event_from_data_line("data:"), None);
        // unknown event type -> Other (no-op, never ends the turn)
        assert_eq!(
            event_from_data_line(r#"data: {"type":"something.unknown"}"#),
            Some(SessionEvent::Other)
        );
    }
}
