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

use serde_json::Value;

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
    /// Extra per-message system instructions attached to every voice prompt.
    pub system: Option<String>,
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
            let session_id = part.and_then(|p| p.get("sessionID")).and_then(Value::as_str);
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
        "session.idle" => match props.and_then(|p| p.get("sessionID")).and_then(Value::as_str) {
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
                .map(|e| e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string()))
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
                if session_id
                    .as_deref()
                    .map_or(true, |s| s == self.session_id) =>
            {
                Step::Failed(error.clone())
            }
            _ => Step::Ignore,
        }
    }
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
            (SessionEvent::MessageUpdated { session_id: SID.into() }, 1_000_000),
            (part("m2", "The answer is 42."), 1_000_000),
            (SessionEvent::Idle { session_id: SID.into() }, 1_000_000),
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
            (SessionEvent::Idle { session_id: SID.into() }, 1_000_000),
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
            (SessionEvent::Idle { session_id: "other".into() }, 1_000_000), // must NOT end our turn
            (part("m1", "ours"), 1_000_000),
            (SessionEvent::Idle { session_id: SID.into() }, 1_000_000),
        ]);
        let text: String = deltas.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(text, "ours");
        assert_eq!(terminal, Step::Done);
    }

    #[test]
    fn scoped_error_fails_turn_foreign_error_ignored() {
        let mut r = TurnReducer::new(SID, 0);
        assert_eq!(
            r.step(&SessionEvent::Error { session_id: Some("other".into()), error: "x".into() }, 0),
            Step::Ignore
        );
        assert_eq!(
            r.step(&SessionEvent::Error { session_id: None, error: "no-scope".into() }, 0),
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
            (SessionEvent::Idle { session_id: SID.into() }, 1_270_000),
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
        assert_eq!(deltas.iter().map(|(_, t)| t.as_str()).collect::<String>(), "a");
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
