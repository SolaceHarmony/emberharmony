//! Load-time verification of `ChatState`'s turn format against the snapshot's own
//! `chat_template.jinja` — the model's authoritative prompting contract.
//!
//! `ChatState` cannot simply *render* the template at runtime: the template has no
//! audio content type (there is no Jinja syntax for "splice 63 conformer embeddings
//! here"), which is why LiquidAI's own Python `ChatState` and their transformers-js
//! demo both hand-build the turn strings and insert audio embeds in code. So the
//! turn strings live in [`crate::processor`] (`SEQUENCE_START` / `turn_header` /
//! `TURN_FOOTER`) — and THIS module proves, at load time, that those strings
//! tokenize identically to what the snapshot's template renders for a text-only
//! conversation. If LiquidAI ever ships a snapshot with a changed template, the
//! load fails loudly instead of prompting the model off-distribution.

use std::path::Path;

use candle_core::Result;
use tokenizers::Tokenizer;

use crate::processor::{turn_header, SEQUENCE_START, TURN_FOOTER};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("chat template: {e}"))
}

/// One text-only message for template rendering / verification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Render the snapshot's `chat_template.jinja` for a text-only conversation —
/// exactly what Python `tokenizer.apply_chat_template` would produce. `bos_token`
/// is supplied from the tokenizer config contract (`<|startoftext|>`).
pub fn render(
    template_source: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_source).map_err(err)?;
    let template = env.get_template("chat").map_err(err)?;
    template
        .render(minijinja::context! {
            messages => messages,
            bos_token => SEQUENCE_START,
            add_generation_prompt => add_generation_prompt,
        })
        .map_err(err)
}

/// The same conversation built the way `ChatState` builds it (BOS + headers/footers
/// + a trailing generation header). Kept string-level so the comparison covers the
/// exact literals `new_turn`/`end_turn` write.
pub fn chatstate_equivalent(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::from(SEQUENCE_START);
    for message in messages {
        out.push_str(&turn_header(&message.role));
        out.push_str(&message.content);
        out.push_str(TURN_FOOTER);
    }
    if add_generation_prompt {
        out.push_str(&turn_header("assistant"));
    }
    out
}

/// Verify the snapshot's template against `ChatState`'s format at TOKEN level.
/// A missing `chat_template.jinja` is skipped (older snapshots); a present one
/// that disagrees is a hard error — no fallback, per house rules.
pub fn verify_snapshot(dir: &Path, tokenizer: &Tokenizer) -> Result<()> {
    let path = dir.join("chat_template.jinja");
    let Ok(source) = std::fs::read_to_string(&path) else {
        return Ok(()); // no template shipped — nothing to verify against
    };

    let sample = [
        ChatMessage::new("system", "Respond with interleaved text and audio."),
        ChatMessage::new("user", "Hello there."),
        ChatMessage::new("assistant", "Hi! How can I help?"),
        ChatMessage::new("user", "Tell me a story."),
    ];
    let rendered = render(&source, &sample, true)?;
    let ours = chatstate_equivalent(&sample, true);

    let ids = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(err)?
            .get_ids()
            .to_vec())
    };
    let (rendered_ids, our_ids) = (ids(&rendered)?, ids(&ours)?);
    if rendered_ids != our_ids {
        return Err(err(format!(
            "snapshot chat_template.jinja disagrees with ChatState's turn format.\n\
             template render: {rendered:?}\n\
             chatstate build: {ours:?}\n\
             (token counts {} vs {})",
            rendered_ids.len(),
            our_ids.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LFM2.5-Audio template as shipped (c362a062 snapshot), inlined so the
    /// string-level contract is tested without a model download.
    const LFM2_TEMPLATE: &str = "{{- bos_token -}}{%- set system_prompt = \"\" -%}{%- set ns = namespace(system_prompt=\"\") -%}{%- if messages[0][\"role\"] == \"system\" -%} {%- set ns.system_prompt = messages[0][\"content\"] -%} {%- set messages = messages[1:] -%}{%- endif -%}{%- if tools -%} {%- set ns.system_prompt = ns.system_prompt + (\"\\n\" if ns.system_prompt else \"\") + \"List of tools: <|tool_list_start|>[\" -%} {%- for tool in tools -%} {%- if tool is not string -%} {%- set tool = tool | tojson -%} {%- endif -%} {%- set ns.system_prompt = ns.system_prompt + tool -%} {%- if not loop.last -%} {%- set ns.system_prompt = ns.system_prompt + \", \" -%} {%- endif -%} {%- endfor -%} {%- set ns.system_prompt = ns.system_prompt + \"]<|tool_list_end|>\" -%}{%- endif -%}{%- if ns.system_prompt -%} {{- \"<|im_start|>system\\n\" + ns.system_prompt + \"<|im_end|>\\n\" -}}{%- endif -%}{%- for message in messages -%} {{- \"<|im_start|>\" + message[\"role\"] + \"\\n\" -}} {%- set content = message[\"content\"] -%} {%- if content is not string -%} {%- set content = content | tojson -%} {%- endif -%} {%- if message[\"role\"] == \"tool\" -%} {%- set content = \"<|tool_response_start|>\" + content + \"<|tool_response_end|>\" -%} {%- endif -%} {{- content + \"<|im_end|>\\n\" -}}{%- endfor -%}{%- if add_generation_prompt -%} {{- \"<|im_start|>assistant\\n\" -}}{%- endif -%}";

    #[test]
    fn chatstate_format_matches_shipped_template_string() {
        let sample = [
            ChatMessage::new("system", "Respond with interleaved text and audio."),
            ChatMessage::new("user", "Hello there."),
            ChatMessage::new("assistant", "Hi! How can I help?"),
            ChatMessage::new("user", "Tell me a story."),
        ];
        let rendered = render(LFM2_TEMPLATE, &sample, true).expect("render");
        let ours = chatstate_equivalent(&sample, true);
        assert_eq!(rendered, ours, "template render != ChatState build");
    }

    #[test]
    fn generation_prompt_is_optional() {
        let sample = [ChatMessage::new("user", "hi")];
        let rendered = render(LFM2_TEMPLATE, &sample, false).expect("render");
        assert!(!rendered.ends_with("<|im_start|>assistant\n"));
        assert!(rendered.ends_with(TURN_FOOTER));
    }
}
