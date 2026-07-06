//! Port of `liquid_audio/data/types.py` — the data-pipeline value types.
//!
//! These are the plain dataclasses the training/data subsystem passes around:
//! the chat-content segments that describe a conversation turn
//! (`TextSegment` / `AudioSegment` / `InterleavedSegment` / `ChatMessage`) and
//! the three pre-packed tensor bundles (`LFM2AudioTrainingSample`,
//! `LFM2AudioRow`, `LFM2AudioModelInput`).
//!
//! The Python `Literal["text"|"audio"|"interleaved"]` `kind` discriminators and
//! the `Literal["user"|"system"|"assistant"]` `role` become real Rust enums
//! ([`SegmentKind`], [`Role`]) — the closest faithful equivalent to a string
//! literal type. The `ChatContentSegment = TextSegment | AudioSegment |
//! InterleavedSegment` union (a PEP 604 sum type) becomes the [`ChatContentSegment`]
//! enum.
//!
//! `audio: bytes` (raw encoded-audio bytes in Python) maps to `Vec<u8>`.
//!
//! The three tensor bundles each hold the same six `candle` [`Tensor`] fields the
//! Python dataclasses hold as `torch.Tensor`. `LFM2AudioModelInput` is **not**
//! redefined here: it already lives in
//! [`crate::model::lfm2_audio`] (with its `to(device)` method), and is re-exported
//! from this module so `liquid_audio.data.types.LFM2AudioModelInput` resolves to
//! the one canonical type.

use candle_core::{Device, Result, Tensor};

/// `LFM2AudioModelInput` — the batched training input (the third Python
/// `data/types.py` dataclass, assembled by the collator): the model inputs plus
/// the `supervision_mask` marking which positions contribute to the loss. Defined
/// here, where Python defines it; `model::lfm2_audio` (which consumes it in
/// `logits`/`forward`) re-exports it.
#[derive(Debug, Clone)]
pub struct LFM2AudioModelInput {
    pub text: Tensor,
    pub audio_in: Tensor,
    pub audio_in_lens: Tensor,
    pub audio_out: Tensor,
    pub modality_flag: Tensor,
    pub supervision_mask: Tensor,
}

impl LFM2AudioModelInput {
    /// `to(self, device) -> LFM2AudioModelInput` (py 69) — move every field to `device`.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            text: self.text.to_device(device)?,
            audio_in: self.audio_in.to_device(device)?,
            audio_in_lens: self.audio_in_lens.to_device(device)?,
            audio_out: self.audio_out.to_device(device)?,
            modality_flag: self.modality_flag.to_device(device)?,
            supervision_mask: self.supervision_mask.to_device(device)?,
        })
    }
}

/// The `kind` discriminator on a chat-content segment. Faithful to the Python
/// `Literal["text"]` / `Literal["audio"]` / `Literal["interleaved"]` default
/// fields — each `@dataclass` pins `kind` to a single string literal; the Rust
/// equivalent of that closed string set is this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// `"text"`
    Text,
    /// `"audio"`
    Audio,
    /// `"interleaved"`
    Interleaved,
}

impl SegmentKind {
    /// The wire string for this kind (the value the Python `Literal` carries).
    pub fn as_str(self) -> &'static str {
        match self {
            SegmentKind::Text => "text",
            SegmentKind::Audio => "audio",
            SegmentKind::Interleaved => "interleaved",
        }
    }
}

/// `role` of a [`ChatMessage`]. Faithful to the Python
/// `Literal["user", "system", "assistant"]` annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `"user"`
    User,
    /// `"system"`
    System,
    /// `"assistant"`
    Assistant,
}

impl Role {
    /// The wire string for this role.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::System => "system",
            Role::Assistant => "assistant",
        }
    }
}

/// `TextSegment` — a text-only chat-content segment.
///
/// Faithful to:
/// ```python
/// @dataclass(frozen=True, slots=True)
/// class TextSegment:
///     kind: Literal["text"] = "text"
///     text: str = ""
/// ```
/// The `frozen=True` immutability is expressed by exposing only owning
/// constructors and accessors (the fields stay private). `kind` is fixed to
/// [`SegmentKind::Text`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSegment {
    text: String,
}

impl TextSegment {
    /// `TextSegment(text=...)`.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// The fixed `kind` field (`"text"`).
    pub fn kind(&self) -> SegmentKind {
        SegmentKind::Text
    }

    /// The `text` field.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Default for TextSegment {
    /// `TextSegment()` — the Python default (`text=""`).
    fn default() -> Self {
        Self {
            text: String::new(),
        }
    }
}

/// `AudioSegment` — an audio-only chat-content segment carrying raw encoded
/// audio bytes.
///
/// Faithful to:
/// ```python
/// @dataclass(frozen=True, slots=True)
/// class AudioSegment:
///     kind: Literal["audio"] = "audio"
///     audio: bytes = b""
/// ```
/// `audio: bytes` → `Vec<u8>`. `kind` is fixed to [`SegmentKind::Audio`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSegment {
    audio: Vec<u8>,
}

impl AudioSegment {
    /// `AudioSegment(audio=...)`.
    pub fn new(audio: impl Into<Vec<u8>>) -> Self {
        Self {
            audio: audio.into(),
        }
    }

    /// The fixed `kind` field (`"audio"`).
    pub fn kind(&self) -> SegmentKind {
        SegmentKind::Audio
    }

    /// The `audio` field (raw encoded-audio bytes).
    pub fn audio(&self) -> &[u8] {
        &self.audio
    }
}

impl Default for AudioSegment {
    /// `AudioSegment()` — the Python default (`audio=b""`).
    fn default() -> Self {
        Self { audio: Vec::new() }
    }
}

/// `InterleavedSegment` — a chat-content segment carrying both text and audio
/// for interleaved (real-time S2S) turns.
///
/// Faithful to:
/// ```python
/// @dataclass(frozen=True, slots=True)
/// class InterleavedSegment:
///     kind: Literal["interleaved"] = "interleaved"
///     text: str = ""
///     audio: bytes = b""
/// ```
/// `kind` is fixed to [`SegmentKind::Interleaved`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterleavedSegment {
    text: String,
    audio: Vec<u8>,
}

impl InterleavedSegment {
    /// `InterleavedSegment(text=..., audio=...)`.
    pub fn new(text: impl Into<String>, audio: impl Into<Vec<u8>>) -> Self {
        Self {
            text: text.into(),
            audio: audio.into(),
        }
    }

    /// The fixed `kind` field (`"interleaved"`).
    pub fn kind(&self) -> SegmentKind {
        SegmentKind::Interleaved
    }

    /// The `text` field.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The `audio` field (raw encoded-audio bytes).
    pub fn audio(&self) -> &[u8] {
        &self.audio
    }
}

impl Default for InterleavedSegment {
    /// `InterleavedSegment()` — the Python defaults (`text=""`, `audio=b""`).
    fn default() -> Self {
        Self {
            text: String::new(),
            audio: Vec::new(),
        }
    }
}

/// `ChatContentSegment = TextSegment | AudioSegment | InterleavedSegment`.
///
/// Faithful to the Python PEP-604 union type alias. The `kind` field of each
/// variant is the discriminator; [`ChatContentSegment::kind`] reads it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatContentSegment {
    /// A [`TextSegment`].
    Text(TextSegment),
    /// An [`AudioSegment`].
    Audio(AudioSegment),
    /// An [`InterleavedSegment`].
    Interleaved(InterleavedSegment),
}

impl ChatContentSegment {
    /// The `kind` discriminator of the held segment.
    pub fn kind(&self) -> SegmentKind {
        match self {
            ChatContentSegment::Text(_) => SegmentKind::Text,
            ChatContentSegment::Audio(_) => SegmentKind::Audio,
            ChatContentSegment::Interleaved(_) => SegmentKind::Interleaved,
        }
    }
}

impl From<TextSegment> for ChatContentSegment {
    fn from(s: TextSegment) -> Self {
        ChatContentSegment::Text(s)
    }
}

impl From<AudioSegment> for ChatContentSegment {
    fn from(s: AudioSegment) -> Self {
        ChatContentSegment::Audio(s)
    }
}

impl From<InterleavedSegment> for ChatContentSegment {
    fn from(s: InterleavedSegment) -> Self {
        ChatContentSegment::Interleaved(s)
    }
}

/// `ChatMessage` — one message in a conversation: a [`Role`] plus an ordered
/// list of [`ChatContentSegment`]s.
///
/// Faithful to:
/// ```python
/// @dataclass(frozen=True, slots=True, kw_only=True)
/// class ChatMessage:
///     role: Literal["user", "system", "assistant"]
///     content: list[ChatContentSegment]
/// ```
/// `kw_only=True` (no positional construction) maps to the named-field struct
/// literal / [`ChatMessage::new`] constructor; `frozen=True` to private fields
/// with read-only accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    role: Role,
    content: Vec<ChatContentSegment>,
}

impl ChatMessage {
    /// `ChatMessage(role=..., content=...)`.
    pub fn new(role: Role, content: Vec<ChatContentSegment>) -> Self {
        Self { role, content }
    }

    /// The `role` field.
    pub fn role(&self) -> Role {
        self.role
    }

    /// The `content` field.
    pub fn content(&self) -> &[ChatContentSegment] {
        &self.content
    }
}

/// `LFM2AudioTrainingSample` — a pre-packed data item (the per-sample tensors
/// before padding/collation).
///
/// Faithful to:
/// ```python
/// @dataclass(slots=True, kw_only=True)
/// class LFM2AudioTrainingSample:
///     text: torch.Tensor
///     audio_in: torch.Tensor
///     audio_in_lens: torch.Tensor
///     audio_out: torch.Tensor
///     modality_flag: torch.Tensor
///     supervision_mask: torch.Tensor
/// ```
/// `torch.Tensor` → candle [`Tensor`]. This is structurally identical to
/// [`LFM2AudioRow`] and [`LFM2AudioModelInput`]; the three are distinct named
/// stages of the data pipeline (sample → padded row → collated batch), kept
/// separate to mirror the Python.
#[derive(Debug, Clone)]
pub struct LFM2AudioTrainingSample {
    /// `text` — text token ids.
    pub text: Tensor,
    /// `audio_in` — input mel features.
    pub audio_in: Tensor,
    /// `audio_in_lens` — per-segment input-audio frame lengths.
    pub audio_in_lens: Tensor,
    /// `audio_out` — output audio codebook tokens.
    pub audio_out: Tensor,
    /// `modality_flag` — per-position [`crate::utils::LFMModality`] flags.
    pub modality_flag: Tensor,
    /// `supervision_mask` — which positions contribute to the loss.
    pub supervision_mask: Tensor,
}

impl LFM2AudioTrainingSample {
    /// Move every tensor field to `device`. Faithful in spirit to the sibling
    /// `LFM2AudioModelInput.to(device)` (the Python `LFM2AudioTrainingSample`
    /// has no `to`, but candle's explicit-placement model makes the per-field
    /// move the real equivalent of a device transfer for this bundle).
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            text: self.text.to_device(device)?,
            audio_in: self.audio_in.to_device(device)?,
            audio_in_lens: self.audio_in_lens.to_device(device)?,
            audio_out: self.audio_out.to_device(device)?,
            modality_flag: self.modality_flag.to_device(device)?,
            supervision_mask: self.supervision_mask.to_device(device)?,
        })
    }
}

/// `LFM2AudioRow` — a single padded row produced from the dataset (a
/// [`LFM2AudioTrainingSample`] after padding, before batching).
///
/// Faithful to:
/// ```python
/// @dataclass(slots=True, kw_only=True)
/// class LFM2AudioRow:
///     text: torch.Tensor
///     audio_in: torch.Tensor
///     audio_in_lens: torch.Tensor
///     audio_out: torch.Tensor
///     modality_flag: torch.Tensor
///     supervision_mask: torch.Tensor
/// ```
/// Same six fields as [`LFM2AudioTrainingSample`] / [`LFM2AudioModelInput`];
/// kept as its own type to mirror the Python pipeline stages.
#[derive(Debug, Clone)]
pub struct LFM2AudioRow {
    /// `text` — text token ids.
    pub text: Tensor,
    /// `audio_in` — input mel features.
    pub audio_in: Tensor,
    /// `audio_in_lens` — per-segment input-audio frame lengths.
    pub audio_in_lens: Tensor,
    /// `audio_out` — output audio codebook tokens.
    pub audio_out: Tensor,
    /// `modality_flag` — per-position [`crate::utils::LFMModality`] flags.
    pub modality_flag: Tensor,
    /// `supervision_mask` — which positions contribute to the loss.
    pub supervision_mask: Tensor,
}

impl LFM2AudioRow {
    /// Move every tensor field to `device` (see
    /// [`LFM2AudioTrainingSample::to`]).
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            text: self.text.to_device(device)?,
            audio_in: self.audio_in.to_device(device)?,
            audio_in_lens: self.audio_in_lens.to_device(device)?,
            audio_out: self.audio_out.to_device(device)?,
            modality_flag: self.modality_flag.to_device(device)?,
            supervision_mask: self.supervision_mask.to_device(device)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_kinds_are_fixed() {
        assert_eq!(TextSegment::default().kind(), SegmentKind::Text);
        assert_eq!(AudioSegment::default().kind(), SegmentKind::Audio);
        assert_eq!(
            InterleavedSegment::default().kind(),
            SegmentKind::Interleaved
        );
        assert_eq!(SegmentKind::Text.as_str(), "text");
        assert_eq!(SegmentKind::Audio.as_str(), "audio");
        assert_eq!(SegmentKind::Interleaved.as_str(), "interleaved");
    }

    #[test]
    fn segment_defaults_match_python() {
        // text="" , audio=b""
        assert_eq!(TextSegment::default().text(), "");
        assert!(AudioSegment::default().audio().is_empty());
        assert_eq!(InterleavedSegment::default().text(), "");
        assert!(InterleavedSegment::default().audio().is_empty());
    }

    #[test]
    fn union_reports_held_kind() {
        let s: ChatContentSegment = TextSegment::new("hi").into();
        assert_eq!(s.kind(), SegmentKind::Text);
        let a: ChatContentSegment = AudioSegment::new(vec![1u8, 2, 3]).into();
        assert_eq!(a.kind(), SegmentKind::Audio);
        let i: ChatContentSegment = InterleavedSegment::new("hi", vec![9u8]).into();
        assert_eq!(i.kind(), SegmentKind::Interleaved);
    }

    #[test]
    fn roles_round_trip_to_strings() {
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn chat_message_holds_role_and_content() {
        let msg = ChatMessage::new(
            Role::Assistant,
            vec![
                TextSegment::new("hello").into(),
                AudioSegment::new(vec![0u8, 1]).into(),
            ],
        );
        assert_eq!(msg.role(), Role::Assistant);
        assert_eq!(msg.content().len(), 2);
        assert_eq!(msg.content()[0].kind(), SegmentKind::Text);
        assert_eq!(msg.content()[1].kind(), SegmentKind::Audio);
    }
}
