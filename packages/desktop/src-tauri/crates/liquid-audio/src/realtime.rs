//! Multi-threaded realtime speech-to-speech pipeline.
//!
//! Faithful port of the threading model in `liquid_audio/demo/chat.py`, restructured
//! toward `moshi/server.py`'s inference coroutine.
//!
//! **Python (`chat.py`)**: `chat_response` spawns a `chat_producer` `Thread` that owns the
//! model + Mimi codec, runs `generate_interleaved`, decodes each audio frame to PCM, and
//! `q.put()`s tokens + PCM onto a `queue.Queue`; the main thread drains the queue and
//! relays text to the UI and PCM to WebRTC playback. Generation overlaps playback because
//! they run on different threads.
//!
//! **Here**: a *persistent* inference worker thread owns the [`VoiceEngine`] and loops
//! `recv utterance → respond (emit text + decode audio → emit PCM) → TurnComplete`. The
//! consumer (UI / playback feeder) drains [`VoiceEvent`]s off a channel. Because the model
//! lives on its own thread, capture and playback are never blocked by generation
//! (full-duplex), and a newly-detected utterance can request **barge-in** — an
//! `AtomicBool` the generate loop polls (see
//! [`LFM2AudioModel::generate_interleaved_cancellable`]) to abort the in-flight reply
//! instead of running it to `max_new_tokens` and tying up the P-cores.
//!
//! The engine is a trait so the threading is unit-tested with a fake (no model needed);
//! [`Lfm2VoiceEngine`] is the real implementation that owns the model + processor.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{unbounded, Receiver, Sender};

const MIMI_RATE: u32 = 24_000; // Mimi/LFM2 detokenizer output rate.

/// A captured user utterance handed to the worker: mono f32 samples + their sample rate.
pub struct Utterance {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// One streamed reply item the worker emits — the Rust analog of `chat.py`'s `q.put`.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    /// A decoded text fragment (one or more detokenized text tokens).
    Text(String),
    /// A decoded PCM chunk (mono f32) at the pipeline's output rate.
    Audio(Vec<f32>),
    /// The reply for the current utterance finished normally (`chat.py`'s `q.put(None)`).
    TurnComplete,
    /// The reply was cut short by [`RealtimePipeline::interrupt`] (barge-in).
    Interrupted,
    /// The engine errored on this turn. The worker stays alive for the next utterance.
    Error(String),
}

/// The model side of the pipeline, abstracted so the worker-thread machinery can be
/// exercised with a fake. The real implementation ([`Lfm2VoiceEngine`]) owns the model +
/// processor + detokenizer; it must be `Send` to move onto the worker thread.
pub trait VoiceEngine: Send {
    /// Respond to one utterance, calling `emit` for each [`VoiceEvent`] in order. Poll
    /// `cancel` frequently and return `Ok(false)` promptly once it is set (barge-in);
    /// return `Ok(true)` when the reply ran to completion. `Err` is surfaced as
    /// [`VoiceEvent::Error`] and does not kill the worker.
    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String>;
}

/// Handle to the inference worker thread: submit utterances, receive reply events, request
/// barge-in. Dropping it closes the channel and joins the worker.
pub struct RealtimePipeline {
    utt_tx: Option<Sender<Utterance>>,
    event_rx: Receiver<VoiceEvent>,
    cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl RealtimePipeline {
    /// Spawn the worker thread; it owns `engine` for its lifetime and serves utterances
    /// until this handle is dropped (which closes the utterance channel).
    pub fn spawn<E: VoiceEngine + 'static>(mut engine: E) -> Self {
        let (utt_tx, utt_rx) = unbounded::<Utterance>();
        let (event_tx, event_rx) = unbounded::<VoiceEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();

        let worker = std::thread::Builder::new()
            .name("lfm2-inference".into())
            .spawn(move || {
                // The inference coroutine: serve utterances until the sender is dropped
                // (channel closed) ⇒ `iter()` ends ⇒ the thread returns and joins.
                for utt in utt_rx.iter() {
                    // A fresh turn clears any barge-in left set by the previous reply, so
                    // it cannot carry over and abort the new one before it starts.
                    cancel_worker.store(false, Ordering::SeqCst);
                    let mut emit = |ev: VoiceEvent| {
                        let _ = event_tx.send(ev);
                    };
                    let terminal = match engine.respond(&utt, &cancel_worker, &mut emit) {
                        Ok(true) => VoiceEvent::TurnComplete,
                        Ok(false) => VoiceEvent::Interrupted,
                        Err(e) => VoiceEvent::Error(e),
                    };
                    if event_tx.send(terminal).is_err() {
                        break; // consumer hung up — nothing left to serve.
                    }
                }
            })
            .expect("spawn lfm2-inference worker");

        Self { utt_tx: Some(utt_tx), event_rx, cancel, worker: Some(worker) }
    }

    /// Hand the worker a new utterance. Returns `false` if the worker has stopped.
    pub fn submit(&self, utt: Utterance) -> bool {
        self.utt_tx.as_ref().is_some_and(|tx| tx.send(utt).is_ok())
    }

    /// Request barge-in: abort the in-flight reply. The engine polls this and returns
    /// early, after which the worker emits [`VoiceEvent::Interrupted`]. Call this before
    /// submitting the interrupting utterance.
    pub fn interrupt(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// The stream of reply events; drain it in the consumer (UI / playback feeder).
    pub fn events(&self) -> &Receiver<VoiceEvent> {
        &self.event_rx
    }
}

impl Drop for RealtimePipeline {
    fn drop(&mut self) {
        // Abort any in-flight reply fast, then close the utterance channel so the worker's
        // `iter()` ends, then join — no detached thread, no leak.
        self.cancel.store(true, Ordering::SeqCst);
        drop(self.utt_tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// ---------------------------------------------------------------------------------------
// Real engine
// ---------------------------------------------------------------------------------------

use candle_core::{DType, Device, Tensor};

use crate::{ChatState, GenParams, GenToken, LFM2AudioModel, LFM2AudioProcessor, LFMModality};

/// The accumulated conversation tensors persisted across turns — the five model-input fields
/// of a [`ChatState`]. The engine holds these (not a `ChatState`, which borrows the processor
/// and so can't be stored beside it) and rebuilds a transient `ChatState` each turn via
/// [`ChatState::from_parts`]. This is the Rust analog of Python keeping ONE persistent
/// `ChatState` object across turns: the discrete `audio_out` generated each turn is fed back
/// here as context for the next prefill (`generate_interleaved`'s `audio_out` scatter).
#[derive(Clone)]
pub struct ConversationState {
    pub text: Tensor,
    pub audio_in: Tensor,
    pub audio_in_lens: Tensor,
    pub audio_out: Tensor,
    pub modality_flag: Tensor,
}

/// Build a `(rows, cols)` i64 tensor, using a zero-length view of a 1-col buffer when
/// `cols == 0` (candle can't allocate a zero-size buffer on Metal — the same guard
/// `ChatState::new` uses for its empty placeholders). Used to stack the per-turn generated
/// `text`/`audio_out`/`modality_flag` for [`ChatState::append`].
fn stack_i64(vals: Vec<i64>, rows: usize, cols: usize, dev: &Device) -> candle_core::Result<Tensor> {
    if cols == 0 {
        Tensor::zeros((rows, 1), DType::I64, dev)?.narrow(1, 0, 0)
    } else {
        Tensor::from_vec(vals, (rows, cols), dev)
    }
}

/// The real [`VoiceEngine`]: owns the LFM2-Audio model + processor and answers each
/// utterance via [`LFM2AudioModel::generate_interleaved_cancellable`], decoding audio
/// frames to PCM through the processor's streaming detokenizer — the headless equivalent
/// of `chat.py`'s `chat_producer` body.
pub struct Lfm2VoiceEngine {
    model: LFM2AudioModel,
    proc: LFM2AudioProcessor,
    params: GenParams,
    codebooks: usize,
    device: Device,
    /// Resample target for emitted PCM. `MIMI_RATE` (or 0) ⇒ emit the codec's native
    /// 24 kHz untouched; otherwise resample each chunk to this device rate.
    out_rate: u32,
    /// The assistant's system prompt, added once at the start of the conversation (Python adds
    /// the system turn once, before the first user turn). Defaults to the demo's interleaved
    /// prompt; the desktop layer's `TurnMode` sets the ASR/TTS variants via
    /// [`Self::with_system_prompt`]. (Per-mode `max_new_tokens` rides on `params`.)
    system_prompt: String,
    /// The persistent conversation: `None` until the first turn completes (cold start = fresh
    /// `ChatState::new` + the system turn), then `Some`, carrying the accumulated tensors so
    /// each later turn *continues* the conversation — feeding the prior turn's discrete
    /// `audio_out` back as context (the `audio_out` → prefill-scatter path). Cold-start-per-turn
    /// (the prior behavior) made every utterance context-free.
    conv: Option<ConversationState>,
}

impl Lfm2VoiceEngine {
    pub fn new(
        model: LFM2AudioModel,
        proc: LFM2AudioProcessor,
        params: GenParams,
        codebooks: usize,
        device: Device,
        out_rate: u32,
    ) -> Self {
        Self {
            model,
            proc,
            params,
            codebooks,
            device,
            out_rate,
            system_prompt: "Respond with interleaved text and audio.".to_string(),
            conv: None,
        }
    }

    /// Override the system prompt (the desktop `TurnMode` → ASR / TTS / Interleaved prompt,
    /// verbatim from the demo's `audio-model.js`). Builder form so the single `new` call site
    /// and the tests stay unchanged.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

impl VoiceEngine for Lfm2VoiceEngine {
    fn respond(
        &mut self,
        utt: &Utterance,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(VoiceEvent),
    ) -> Result<bool, String> {
        let s = |e: candle_core::Error| e.to_string();

        // `audio_detokenizer()` returns the LFM2 detokenizer when present, else falls back to
        // the Mimi codec (processor.rs: `audio_out.or(mimi)`) — so this streaming `decode_step`
        // path works on both model families, mirroring the one-shot `processor.decode` fallback.
        let detok = self.proc.audio_detokenizer().ok_or("no audio-out backend (Mimi) in this model")?;
        detok.reset_stream(); // turn boundary — independent streaming decode.

        let wave = Tensor::from_vec(utt.samples.clone(), (1, utt.samples.len()), &self.device).map_err(s)?;

        // Restore the persistent conversation (clone the prior tensors — cheap Arc bumps — so an
        // interrupted turn can be discarded without losing history; we never `take`). First turn
        // → fresh `ChatState` + the system turn (added once, like Python); later turns → seed
        // from the accumulated state so the prior discrete `audio_out` conditions this prefill.
        let prior = self.conv.clone();
        let mut chat = match &prior {
            None => {
                let mut c = ChatState::new(&self.proc, self.codebooks).map_err(s)?;
                c.new_turn("system").map_err(s)?;
                c.add_text(&self.system_prompt).map_err(s)?;
                c.end_turn().map_err(s)?;
                c
            }
            Some(conv) => ChatState::from_parts(
                &self.proc,
                self.codebooks,
                conv.text.clone(),
                conv.audio_in.clone(),
                conv.audio_in_lens.clone(),
                conv.audio_out.clone(),
                conv.modality_flag.clone(),
            )
            .map_err(s)?,
        };
        chat.new_turn("user").map_err(s)?;
        chat.add_audio(&wave, utt.rate).map_err(s)?; // CONTINUOUS audio-in (mel → Conformer)
        chat.end_turn().map_err(s)?;
        chat.new_turn("assistant").map_err(s)?;

        let text = self.proc.text();
        let device = &self.device;
        let codebooks = self.codebooks;
        let out_rate = self.out_rate;
        let mut cb_err: Option<String> = None;

        // Collect the generated stream for `chat.append` (the discrete `audio_out` → context
        // path). `text_ids` / `audio_frames` are the SEPARATED in-order streams (Python's
        // `text_out` / `audio_out` lists); `modality_out` preserves the INTERLEAVED generation
        // order (Python's `modality_out`). `audio_frames` keeps ALL frames including the
        // all-2048 EOAudio terminator — `append` needs the full `audio_out`
        // (`torch.stack(audio_out, 1)`); only the Mimi DECODE drops the last (`audio_out[:-1]`).
        let mut text_ids: Vec<i64> = Vec::new();
        let mut audio_frames: Vec<Vec<u32>> = Vec::new();
        let mut modality_out: Vec<i64> = Vec::new();

        self.model
            .generate_interleaved_cancellable(&chat, &self.params, cancel, |tok| {
                if cb_err.is_some() {
                    return;
                }
                match tok {
                    GenToken::Text(id) => {
                        text_ids.push(id as i64);
                        modality_out.push(LFMModality::Text as i64);
                        match text.decode(&[id], true) {
                            Ok(text) => emit(VoiceEvent::Text(text)),
                            Err(e) => cb_err = Some(e.to_string()),
                        }
                    }
                    GenToken::Audio(frame) => {
                        // Collect EVERY frame (incl. EOAudio) for `append` BEFORE the streaming
                        // skip — append needs the full audio_out; only PCM playback drops it.
                        modality_out.push(LFMModality::AudioOut as i64);
                        audio_frames.push(frame.clone());
                        if frame.contains(&2048) {
                            return; // EOAudio terminator — no waveform to stream.
                        }
                        // Decode the 8-code frame to PCM via the streaming detokenizer.
                        let decoded = (|| -> candle_core::Result<Option<Vec<f32>>> {
                            let codes = Tensor::from_vec(frame.clone(), (1, codebooks, 1), device)?;
                            match detok.decode_step(&codes)? {
                                Some(chunk) => {
                                    let mut pcm = chunk.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                                    if out_rate != MIMI_RATE && out_rate != 0 {
                                        pcm = crate::resample::resample_slice(&pcm, MIMI_RATE, out_rate);
                                    }
                                    Ok(Some(pcm))
                                }
                                None => Ok(None),
                            }
                        })();
                        match decoded {
                            Ok(Some(pcm)) => emit(VoiceEvent::Audio(pcm)),
                            Ok(None) => {}
                            Err(e) => cb_err = Some(e.to_string()),
                        }
                    }
                }
            })
            .map_err(s)?;

        if let Some(e) = cb_err {
            return Err(e);
        }

        // Completed unless the loop broke because barge-in was requested.
        let completed = !cancel.load(Ordering::Acquire);

        // Persist the turn ONLY on clean completion. `append` weaves the generated text +
        // discrete `audio_out` (with their interleaved modality flags) into the conversation,
        // then `end_turn` closes the assistant turn — exactly Python's `chat.append(...)` +
        // `chat.end_turn()`. An interrupted turn is discarded: `self.conv` keeps the prior
        // history (we cloned `prior`, never took it), so barge-in rewinds to before this turn.
        if completed {
            let n_text = text_ids.len();
            let n_audio = audio_frames.len();
            let n_flag = modality_out.len();
            let text_t = stack_i64(text_ids, 1, n_text, &self.device).map_err(s)?;
            // Stack frames into (codebooks, n_audio) column-major: flat[c*n_audio + f] =
            // frame[f][c] — the transpose of the (frame, codebook) collection, matching
            // `torch.stack(audio_out, 1)` (each frame is a column).
            let mut flat = Vec::with_capacity(codebooks * n_audio);
            for c in 0..codebooks {
                for f in &audio_frames {
                    flat.push(f[c] as i64);
                }
            }
            let audio_t = stack_i64(flat, codebooks, n_audio, &self.device).map_err(s)?;
            let mod_t = stack_i64(modality_out, 1, n_flag, &self.device).map_err(s)?;
            chat.append(&text_t, &audio_t, &mod_t).map_err(s)?;
            chat.end_turn().map_err(s)?;

            let saved = ConversationState {
                text: chat.text.clone(),
                audio_in: chat.audio_in.clone(),
                audio_in_lens: chat.audio_in_lens.clone(),
                audio_out: chat.audio_out.clone(),
                modality_flag: chat.modality_flag.clone(),
            };
            drop(chat); // end the `&self.proc` borrow before writing `self.conv`.
            self.conv = Some(saved);
        }

        Ok(completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn utt(n: usize) -> Utterance {
        Utterance { samples: vec![0.0; n], rate: 16_000 }
    }

    /// Drain events until a terminal one (TurnComplete / Interrupted / Error), bounded by
    /// a timeout so a wiring bug fails the test instead of hanging it.
    fn collect_turn(rx: &Receiver<VoiceEvent>) -> Vec<VoiceEvent> {
        let mut out = Vec::new();
        loop {
            let ev = rx.recv_timeout(Duration::from_secs(5)).expect("expected an event before timeout");
            let terminal = matches!(
                ev,
                VoiceEvent::TurnComplete | VoiceEvent::Interrupted | VoiceEvent::Error(_)
            );
            out.push(ev);
            if terminal {
                return out;
            }
        }
    }

    /// Emits a scripted (Text, Audio) pair then completes; counts the turns it served.
    struct ScriptEngine {
        turns: Arc<AtomicUsize>,
    }
    impl VoiceEngine for ScriptEngine {
        fn respond(&mut self, utt: &Utterance, _cancel: &AtomicBool, emit: &mut dyn FnMut(VoiceEvent)) -> Result<bool, String> {
            self.turns.fetch_add(1, Ordering::SeqCst);
            emit(VoiceEvent::Text(format!("got {}", utt.samples.len())));
            emit(VoiceEvent::Audio(vec![0.1, 0.2, 0.3]));
            Ok(true)
        }
    }

    #[test]
    fn emits_events_in_order_then_turn_complete() {
        let turns = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimePipeline::spawn(ScriptEngine { turns: turns.clone() });
        assert!(pipe.submit(utt(5)));
        assert_eq!(
            collect_turn(pipe.events()),
            vec![
                VoiceEvent::Text("got 5".into()),
                VoiceEvent::Audio(vec![0.1, 0.2, 0.3]),
                VoiceEvent::TurnComplete,
            ]
        );
        assert_eq!(turns.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn worker_persists_across_turns() {
        let turns = Arc::new(AtomicUsize::new(0));
        let pipe = RealtimePipeline::spawn(ScriptEngine { turns: turns.clone() });
        for n in [3usize, 7] {
            assert!(pipe.submit(utt(n)));
            let evs = collect_turn(pipe.events());
            assert_eq!(evs.last(), Some(&VoiceEvent::TurnComplete));
        }
        assert_eq!(turns.load(Ordering::SeqCst), 2, "the worker should serve every utterance");
    }

    /// Emits Audio forever until `cancel` is set — stands in for a long generation.
    struct LoopEngine;
    impl VoiceEngine for LoopEngine {
        fn respond(&mut self, _utt: &Utterance, cancel: &AtomicBool, emit: &mut dyn FnMut(VoiceEvent)) -> Result<bool, String> {
            for _ in 0..100_000 {
                if cancel.load(Ordering::Acquire) {
                    return Ok(false);
                }
                emit(VoiceEvent::Audio(vec![0.0]));
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(true)
        }
    }

    #[test]
    fn interrupt_aborts_in_flight_turn() {
        let pipe = RealtimePipeline::spawn(LoopEngine);
        assert!(pipe.submit(utt(1)));
        // Wait until generation is actually under way.
        let first = pipe.events().recv_timeout(Duration::from_secs(5)).expect("turn should start");
        assert!(matches!(first, VoiceEvent::Audio(_)));

        pipe.interrupt();

        // It must terminate with Interrupted (not TurnComplete), and promptly.
        let mut seen = 0;
        loop {
            let ev = pipe.events().recv_timeout(Duration::from_secs(5)).expect("expected terminal event");
            match ev {
                VoiceEvent::Interrupted => break,
                VoiceEvent::TurnComplete => panic!("barge-in should interrupt, not complete"),
                _ => {
                    seen += 1;
                    assert!(seen < 50_000, "engine did not stop after interrupt()");
                }
            }
        }
    }

    struct ErrEngine;
    impl VoiceEngine for ErrEngine {
        fn respond(&mut self, _utt: &Utterance, _cancel: &AtomicBool, _emit: &mut dyn FnMut(VoiceEvent)) -> Result<bool, String> {
            Err("boom".into())
        }
    }

    #[test]
    fn engine_error_is_reported_and_worker_survives() {
        let pipe = RealtimePipeline::spawn(ErrEngine);
        assert!(pipe.submit(utt(0)));
        assert_eq!(pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(), VoiceEvent::Error("boom".into()));
        // The worker must still be alive to serve a second utterance.
        assert!(pipe.submit(utt(0)));
        assert_eq!(pipe.events().recv_timeout(Duration::from_secs(5)).unwrap(), VoiceEvent::Error("boom".into()));
    }

    /// Real-model proof of the multi-turn persistence (Gap A): two `respond()` calls on the
    /// actual LFM2.5-Audio engine must GROW the persisted `conv` — turn 2 seeds from turn 1's
    /// state (`from_parts`) and appends again, so every field is strictly longer than after
    /// turn 1. This is the engine analog of `examples/chat_multiturn` and the only test that
    /// exercises `from_parts` + the collect→`append`→save path end-to-end.
    ///
    /// `#[ignore]` (needs the model + is slow); run with:
    ///   LFM_DEVICE=metal cargo test --features metal --lib -- --ignored engine_multiturn
    /// (or `LFM_MODEL_DIR=/abs/model cargo test --lib -- --ignored engine_multiturn` on CPU/f32).
    #[test]
    #[ignore = "needs the real LFM2.5-Audio model; slow"]
    fn engine_multiturn_grows_conv() {
        use crate::{from_pretrained, get_model_dir, GenParams};

        // Device: Metal bf16 when built with the feature + LFM_DEVICE=metal; else CPU f32.
        let (device, dtype) = match std::env::var("LFM_DEVICE").ok().as_deref() {
            #[cfg(feature = "metal")]
            Some("metal") => (Device::new_metal(0).expect("metal device"), DType::BF16),
            _ => (Device::Cpu, DType::F32),
        };

        let model_ref = std::env::var("LFM_MODEL")
            .or_else(|_| std::env::var("LFM_MODEL_DIR"))
            .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
        let dir = get_model_dir(&model_ref, None).expect("resolve model dir");
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let codebooks = cfg["codebooks"].as_u64().expect("codebooks") as usize;
        let (model, proc) = from_pretrained(&dir, dtype, &device).expect("load model");

        // Short budget keeps the test quick; demo audio sampling so frames are non-degenerate.
        let params = GenParams { max_new_tokens: 96, ..GenParams::demo_defaults() };
        let mut engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, MIMI_RATE);

        // Reuse the reference question clip for both turns (a smoke test of growth, not content).
        let wav_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../../experiments/lfm2-audio-voice/upstream-liquid-audio/assets/question.wav"
        );
        let bytes = std::fs::read(wav_path).expect("read question.wav");
        // PCM16 mono/stereo → mono f32; assume 16-bit (the reference asset is).
        let hdr = 44usize;
        let pcm: Vec<f32> = bytes[hdr..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let utt = || Utterance { samples: pcm.clone(), rate: 16_000 };

        let cancel = AtomicBool::new(false);
        let mut sink = |_ev: VoiceEvent| {};

        // Turn 1: cold start → fresh ChatState + system turn → generate → append → save.
        let done1 = engine.respond(&utt(), &cancel, &mut sink).expect("turn 1");
        assert!(done1, "turn 1 should complete");
        let c1 = engine.conv.clone().expect("conv persisted after turn 1");
        let (t1, a1, m1) = (
            c1.text.dim(1).unwrap(),
            c1.audio_out.dim(1).unwrap(),
            c1.modality_flag.dim(1).unwrap(),
        );
        assert!(t1 > 0 && a1 > 0, "turn 1 must produce text + discrete audio_out (got {t1}, {a1})");

        // Turn 2: seeds from c1 via from_parts → generate (prefill scatters c1's audio_out as
        // context) → append → save. Every field must be strictly longer than after turn 1.
        let done2 = engine.respond(&utt(), &cancel, &mut sink).expect("turn 2");
        assert!(done2, "turn 2 should complete");
        let c2 = engine.conv.clone().expect("conv persisted after turn 2");
        let (t2, a2, m2) = (
            c2.text.dim(1).unwrap(),
            c2.audio_out.dim(1).unwrap(),
            c2.modality_flag.dim(1).unwrap(),
        );
        assert!(t2 > t1, "turn 2 text must grow: {t1} -> {t2}");
        assert!(a2 > a1, "turn 2 audio_out must grow: {a1} -> {a2}");
        assert!(m2 > m1, "turn 2 modality_flag must grow: {m1} -> {m2}");
    }

    #[test]
    fn drop_joins_worker_and_drops_engine() {
        // A guard inside the engine flips a flag when the engine is dropped — which only
        // happens when the worker thread ends. If Drop didn't close the channel + join,
        // the flag would still be false (or the test would hang on a detached thread).
        struct Guard(Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        struct GuardedEngine {
            _g: Guard,
        }
        impl VoiceEngine for GuardedEngine {
            fn respond(&mut self, _utt: &Utterance, _cancel: &AtomicBool, emit: &mut dyn FnMut(VoiceEvent)) -> Result<bool, String> {
                emit(VoiceEvent::Text("x".into()));
                Ok(true)
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        {
            let pipe = RealtimePipeline::spawn(GuardedEngine { _g: Guard(dropped.clone()) });
            assert!(pipe.submit(utt(1)));
            let _ = collect_turn(pipe.events());
            // `pipe` drops here → channel closes → worker loop ends → engine (+ Guard) drop.
        }
        assert!(dropped.load(Ordering::SeqCst), "worker must drop the engine on shutdown");
    }
}
