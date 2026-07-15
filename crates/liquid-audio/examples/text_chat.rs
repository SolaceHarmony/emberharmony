//! Minimal **text → text** proof: send the model text, get text back. No audio path at
//! all — this exercises tokenizer → LFM2 backbone (`lfm2_hf`) → text head → sampler →
//! detokenize, the cleanest check that the assembled port generates coherent language.
//!
//! Run (CPU BF16 via the in-tree NEON kernel):
//!   LFM_MODEL_DIR=/abs/path/to/model \
//!     cargo run --release --example text_chat -- "Hello! Who are you, in one sentence?"
//! Use `LFM_DEVICE=metal` for Apple GPU BF16.

use candle_core::Device;
use liquid_audio::{from_pretrained, get_model_dir, ChatState, GenParams, GenToken};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// `LFM_DEVICE=metal` → Apple GPU BF16. Default/`cpu` → CPU BF16 through the
/// in-tree NEON kernel.
fn select_device() -> Res<Device> {
    match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("LFM_DEVICE=metal needs a build with `--features metal`".into())
            }
        }
        Some("cpu") | None => {
            if liquid_audio::flashkern::native_engine::bf16_gemm_available() {
                Ok(Device::Cpu)
            } else {
                Err("CPU BF16 needs the NEON BFMMLA kernel; use Metal on this Mac".into())
            }
        }
        Some(other) => Err(format!("unknown LFM_DEVICE={other}; use cpu or metal").into()),
    }
}

fn main() -> Res<()> {
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello! Who are you, in one sentence?".into());
    let max_new_tokens: usize = std::env::var("LFM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    // Default: BF16 inference, on CPU via NEON or on Metal when requested.
    let device = select_device()?;

    eprintln!("[load] resolving `{model_ref}`…");
    let dir = get_model_dir(&model_ref, None)?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"]
        .as_u64()
        .ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!(
        "[load] model + processor from {} (safetensor dtype, {device:?})…",
        dir.display()
    );
    let t0 = std::time::Instant::now();
    let (model, proc) = from_pretrained(&dir, &device)?;
    eprintln!("[load] done in {:.1}s.", t0.elapsed().as_secs_f32());

    // Text-only chat: helpful-assistant system prompt + a user TEXT turn + open assistant
    // turn. (The demo's `generateTextOnly` uses the same "You are a helpful assistant."
    // system prompt; here the input is text, not audio.)
    let mut chat = ChatState::new(&proc, codebooks)?;
    chat.new_turn("system")?;
    chat.add_text("You are a helpful assistant.")?;
    chat.end_turn()?;
    chat.new_turn("user")?;
    chat.add_text(&prompt)?;
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    let params = GenParams {
        max_new_tokens,
        text_temperature: None, // greedy ⇒ deterministic
        text_top_k: None,
        audio_temperature: None,
        audio_top_k: None,
        seed: 0,
    };

    eprintln!("[gen] generate_sequential (greedy, max {max_new_tokens})…");
    let mut text_ids: Vec<u32> = Vec::new();
    let mut audio_frames = 0usize;
    let tg = std::time::Instant::now();
    model.generate_sequential(&chat, &params, |tok| match tok {
        GenToken::Text(id) => text_ids.push(id),
        GenToken::Audio(_) => audio_frames += 1, // ignore audio for the text-only proof
    })?;
    let secs = tg.elapsed().as_secs_f32();
    eprintln!(
        "[gen] {} text tokens ({audio_frames} audio frames ignored) in {secs:.1}s = {:.1} tok/s",
        text_ids.len(),
        text_ids.len() as f32 / secs.max(1e-6)
    );

    let text = proc.text().decode(&text_ids, true)?;
    println!("\nUSER: {prompt}\nASSISTANT: {text}\n");
    Ok(())
}
