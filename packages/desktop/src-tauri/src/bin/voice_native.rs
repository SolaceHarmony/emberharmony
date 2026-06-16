//! Standalone headless native voice client.
//!
//! Connects to a LiveKit voice room and bridges the OS microphone + speakers
//! via `cpal` — no webview, no Tauri window. This is the disembodied voice
//! path the TUI/CLI will use. It talks to a running EmberHarmony server (which
//! hosts the brain session + agent worker).
//!
//! Run:
//!   cargo run --bin voice-native --features voice
//!
//! Env:
//!   EH_SERVER    server base URL          (default http://127.0.0.1:4096)
//!   EH_DIR       project directory        (default: current dir)
//!   EH_SESSION   existing session id      (optional; one is created if absent)
//!   EH_PASSWORD  basic-auth password      (optional; unsecured server if unset)
//!
//! NOTE: this is a first cut for on-device testing. Per-buffer linear
//! resampling has boundary artifacts, and output assumes the device accepts
//! 48kHz; both are refined once we hear it.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::track::{LocalAudioTrack, LocalTrack, TrackSource};
use livekit::webrtc::audio_frame::AudioFrame;
use livekit::webrtc::{
    audio_source::native::NativeAudioSource,
    audio_stream::native::NativeAudioStream,
    prelude::{AudioSourceOptions, RtcAudioSource},
};

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u32 = 1;
const FRAME_SAMPLES: usize = (SAMPLE_RATE as usize) / 100; // 10ms @ 48kHz = 480
// f32 amplitude scale for the AEC. aec3's linear pipeline wants NORMALIZED
// floats in [-1,1]: feeding i16-range magnitudes (SCALE=1.0) saturated the
// pipeline and it output pure silence (meter showed raw_peak up to ~8800 but
// clean_peak=0 every frame). Dividing i16 by 32768 keeps it in [-1,1].
const SCALE: f32 = 32768.0;
// Cap the render (played-audio) backlog so the AEC reference can't drift far
// from the mic timeline (~100ms).
const RENDER_CAP: usize = SAMPLE_RATE as usize / 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = std::env::var("EH_SERVER").unwrap_or_else(|_| "http://127.0.0.1:4096".into());
    let dir = std::env::var("EH_DIR")
        .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().into_owned());
    let password = std::env::var("EH_PASSWORD").ok();

    let http = reqwest::Client::new();
    let auth = |rb: reqwest::RequestBuilder| match &password {
        Some(p) => rb.basic_auth("emberharmony", Some(p)),
        None => rb,
    };

    // 1. Session id — create one if not supplied.
    let session_id = match std::env::var("EH_SESSION") {
        Ok(s) => s,
        Err(_) => {
            let resp = auth(
                http.post(format!("{server}/session"))
                    .header("x-emberharmony-directory", &dir)
                    .json(&serde_json::json!({})),
            )
            .send()
            .await?;
            let v: serde_json::Value = resp.json().await?;
            v["id"]
                .as_str()
                .ok_or("no session id in /session response")?
                .to_string()
        }
    };
    println!("[voice-native] session: {session_id}");

    // 2. Voice token.
    let resp = auth(
        http.post(format!("{server}/voice/token"))
            .header("x-emberharmony-directory", &dir)
            .json(&serde_json::json!({ "sessionID": session_id })),
    )
    .send()
    .await?;
    let tok: serde_json::Value = resp.json().await?;
    let url = tok["url"].as_str().ok_or("no url in /voice/token")?.to_string();
    let token = tok["token"].as_str().ok_or("no token in /voice/token")?.to_string();
    println!("[voice-native] connecting to {url}");

    // 3. Connect to the LiveKit room.
    let (room, mut events) = Room::connect(&url, &token, RoomOptions::default()).await?;
    println!("[voice-native] connected to room: {}", room.name());

    // 4. Mic capture -> NativeAudioSource -> publish.
    let source = NativeAudioSource::new(AudioSourceOptions::default(), SAMPLE_RATE, CHANNELS, 200);
    let mic = LocalAudioTrack::create_audio_track("mic", RtcAudioSource::Native(source.clone()));
    room.local_participant()
        .publish_track(
            LocalTrack::Audio(mic),
            TrackPublishOptions {
                source: TrackSource::Microphone,
                ..Default::default()
            },
        )
        .await?;
    // Echo-cancellation render reference: the output callback pushes the mono
    // samples it plays here; the capture AEC pulls them to cancel the speaker
    // echo out of the mic (so the agent stops hearing — and interrupting —
    // itself). Shared mono i16 @ 48kHz.
    let render: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));

    start_capture(source, render.clone());

    // 5. Speaker playback ring buffer (filled from the agent's audio track).
    let playback: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));
    start_playback(playback.clone(), render.clone());

    // 6. Room events: pump the agent's audio into the playback buffer.
    let ev_loop = tokio::spawn(async move {
        while let Some(ev) = events.recv().await {
            match ev {
                RoomEvent::TrackSubscribed {
                    track: RemoteTrack::Audio(audio),
                    participant,
                    ..
                } => {
                    println!("[voice-native] agent audio subscribed: {}", participant.identity());
                    let pb = playback.clone();
                    tokio::spawn(async move {
                        let mut stream =
                            NativeAudioStream::new(audio.rtc_track(), SAMPLE_RATE as i32, CHANNELS as i32);
                        while let Some(frame) = stream.next().await {
                            let mut buf = pb.lock().unwrap();
                            buf.extend(frame.data.iter().copied());
                            let max = SAMPLE_RATE as usize * 2; // cap ~2s
                            while buf.len() > max {
                                buf.pop_front();
                            }
                        }
                    });
                }
                RoomEvent::Disconnected { reason } => {
                    println!("[voice-native] disconnected: {reason:?}");
                    break;
                }
                _ => {}
            }
        }
    });

    println!("[voice-native] listening — speak into your mic. Ctrl-C to quit.");
    tokio::signal::ctrl_c().await?;
    println!("[voice-native] shutting down");
    room.close().await.ok();
    ev_loop.abort();
    Ok(())
}

/// Crude per-buffer linear resample (mono i16). Fine for a first test.
fn resample(input: &[i16], from: u32, to: u32) -> Vec<i16> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f64;
        let a = input[idx.min(last)] as f64;
        let b = input[(idx + 1).min(last)] as f64;
        out.push((a + (b - a) * frac).round() as i16);
    }
    out
}

/// Capture the OS microphone and feed 48kHz mono 10ms frames into `source`.
/// cpal stream lives on its own thread (cpal streams are !Send); samples cross
/// to a tokio task via an unbounded channel where `capture_frame` is awaited.
fn start_capture(source: NativeAudioSource, render: Arc<Mutex<VecDeque<i16>>>) {
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<(Vec<i16>, u32)>();
    let (clean_tx, mut clean_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<i16>>();

    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                eprintln!("[voice-native] no input (microphone) device found");
                return;
            }
        };
        let config = device.default_input_config().expect("no default input config");
        let dev_rate = config.sample_rate().0;
        let dev_ch = config.channels() as usize;
        println!("[voice-native] mic: {} ch @ {} Hz", dev_ch, dev_rate);
        let err_fn = |e| eprintln!("[voice-native] input stream error: {e}");

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let mut mono = Vec::with_capacity(data.len() / dev_ch);
                    for ch in data.chunks(dev_ch) {
                        let avg = ch.iter().copied().sum::<f32>() / dev_ch as f32;
                        mono.push((avg.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
                    }
                    let _ = raw_tx.send((mono, dev_rate));
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let mut mono = Vec::with_capacity(data.len() / dev_ch);
                    for ch in data.chunks(dev_ch) {
                        let sum: i32 = ch.iter().map(|&s| s as i32).sum();
                        mono.push((sum / dev_ch as i32) as i16);
                    }
                    let _ = raw_tx.send((mono, dev_rate));
                },
                err_fn,
                None,
            ),
            other => {
                eprintln!("[voice-native] unsupported input sample format: {other:?}");
                return;
            }
        }
        .expect("failed to build input stream");

        stream.play().expect("failed to start input stream");
        std::thread::park(); // keep the stream alive
    });

    // --- AEC thread: resample to 48k, echo-cancel against the render
    // reference, chunk to 10ms. Runs on its own thread so the (sync) aec3
    // pipeline never has to be Send for tokio. ---
    std::thread::spawn(move || {
        let format = AudioFormat::ten_ms(SAMPLE_RATE, CHANNELS as u16);
        let mut pipeline = match linear::builder(format, format).initial_delay_ms(116).build() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!(
                    "[voice-native] AEC init failed ({e}); mic passes through WITHOUT echo cancellation"
                );
                None
            }
        };
        let mut acc: Vec<i16> = Vec::new();
        let mut render_f = vec![0.0f32; FRAME_SAMPLES];
        let mut cap_f = vec![0.0f32; FRAME_SAMPLES];
        let mut out_f = vec![0.0f32; FRAME_SAMPLES];
        let mut frame_count: u64 = 0;

        while let Ok((mono, dev_rate)) = raw_rx.recv() {
            acc.extend(resample(&mono, dev_rate, SAMPLE_RATE));
            while acc.len() >= FRAME_SAMPLES {
                let chunk: Vec<i16> = acc.drain(..FRAME_SAMPLES).collect();
                let raw_peak = chunk.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
                let cleaned: Vec<i16> = if let Some(pl) = pipeline.as_mut() {
                    {
                        let mut r = render.lock().unwrap();
                        for slot in render_f.iter_mut() {
                            *slot = r.pop_front().unwrap_or(0) as f32 / SCALE;
                        }
                    }
                    for (dst, &s) in cap_f.iter_mut().zip(chunk.iter()) {
                        *dst = s as f32 / SCALE;
                    }
                    let _ = pl.handle_render_frame(&render_f);
                    match pl.process_capture_frame(&cap_f, &mut out_f) {
                        Ok(_) => out_f
                            .iter()
                            .map(|&s| (s * SCALE).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                            .collect(),
                        Err(e) => {
                            eprintln!("[voice-native] AEC capture error: {e}");
                            chunk
                        }
                    }
                } else {
                    chunk
                };
                frame_count += 1;
                if frame_count % 100 == 0 {
                    let clean_peak = cleaned.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
                    eprintln!(
                        "[voice-native] mic meter: frames={frame_count} raw_peak={raw_peak} clean_peak={clean_peak}"
                    );
                }
                let _ = clean_tx.send(cleaned);
            }
        }
    });

    // --- tokio task: push cleaned frames into the LiveKit source ---
    tokio::spawn(async move {
        while let Some(chunk) = clean_rx.recv().await {
            let frame = AudioFrame {
                data: Cow::Owned(chunk),
                num_channels: CHANNELS,
                sample_rate: SAMPLE_RATE,
                samples_per_channel: FRAME_SAMPLES as u32,
            };
            if let Err(e) = source.capture_frame(&frame).await {
                eprintln!("[voice-native] capture_frame error: {e}");
            }
        }
    });
}

/// Play the agent's 48kHz mono audio from `buf` to the default output device.
fn start_playback(buf: Arc<Mutex<VecDeque<i16>>>, render: Arc<Mutex<VecDeque<i16>>>) {
    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_output_device() {
            Some(d) => d,
            None => {
                eprintln!("[voice-native] no output (speaker) device found");
                return;
            }
        };
        let config = device.default_output_config().expect("no default output config");
        let dev_ch = config.channels() as usize;
        println!(
            "[voice-native] speaker: {} ch @ {} Hz",
            dev_ch,
            config.sample_rate().0
        );
        let err_fn = |e| eprintln!("[voice-native] output stream error: {e}");

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config.into(),
                move |out: &mut [f32], _| {
                    let mut b = buf.lock().unwrap();
                    let mut r = render.lock().unwrap();
                    for frame in out.chunks_mut(dev_ch) {
                        let s = b.pop_front().unwrap_or(0);
                        r.push_back(s); // feed AEC render reference (mono)
                        let f = s as f32 / i16::MAX as f32;
                        for o in frame.iter_mut() {
                            *o = f; // upmix mono -> all channels
                        }
                    }
                    while r.len() > RENDER_CAP {
                        r.pop_front();
                    }
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_output_stream(
                &config.into(),
                move |out: &mut [i16], _| {
                    let mut b = buf.lock().unwrap();
                    let mut r = render.lock().unwrap();
                    for frame in out.chunks_mut(dev_ch) {
                        let s = b.pop_front().unwrap_or(0);
                        r.push_back(s); // feed AEC render reference (mono)
                        for o in frame.iter_mut() {
                            *o = s;
                        }
                    }
                    while r.len() > RENDER_CAP {
                        r.pop_front();
                    }
                },
                err_fn,
                None,
            ),
            other => {
                eprintln!("[voice-native] unsupported output sample format: {other:?}");
                return;
            }
        }
        .expect("failed to build output stream");

        stream.play().expect("failed to start output stream");
        std::thread::park();
    });
}
