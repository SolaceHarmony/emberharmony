use std::time::Duration;

fn main() {
    let ms = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(700);
    let hz = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(660.0);
    let amp = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.12);

    let result = tauri::async_runtime::block_on(async {
        emberharmony_lib::voice::runtime::play_local_webrtc_probe(
            Duration::from_millis(ms),
            hz,
            amp,
        )
        .await
    });

    if let Err(error) = result {
        eprintln!("voice audio probe failed: {error}");
        std::process::exit(1);
    }
}
