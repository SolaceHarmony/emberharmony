//! Opt-in benchmark for the byte-exact native resident-image loader.
//!
//! This executable never resolves or downloads a model. It requires an explicit
//! local checkpoint so an absent real-checkpoint gate cannot be reported as a
//! pass:
//!
//! ```text
//! LFM_MODEL_DIR=/absolute/checkpoint \
//!   cargo run --release -p liquid-audio --example bench_native_load
//! ```
//!
//! `LFM_LOAD_BENCH_RUNS` controls the number of samples per cache/worker mode
//! (default 5). The benchmark alternates the one-worker serial baseline and the
//! production four-worker direct loader. Cold measurements use an OS cache
//! bypass/eviction hint and are emitted as unavailable when that cannot be done
//! honestly. `LFM_LOAD_BENCH_DETOKENIZER` may override the released
//! `audio_detokenizer` directory.

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::time::Instant;

use serde_json::{json, Value};
const ABI: u32 = 2;
const OK: i32 = 0;
const BUILT: u32 = 1;
const ATTACHED: u32 = 1 << 1;
const WIRED: u32 = 1 << 2;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[repr(C)]
struct WeightImage {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LoadStats {
    size: u32,
    abi_version: u32,
    source_bytes: u64,
    segment_bytes: u64,
    segment_constructed_bytes: u64,
    attached_shared_bytes: u64,
    wired_bytes: u64,
    process_resident_bytes: u64,
    build_ns: u64,
    attach_ns: u64,
    generation: u64,
    task_count: u32,
    worker_count: u32,
    flags: u32,
    source_count: u32,
    payload_read_calls: u64,
    payload_read_bytes: u64,
    identity_digest: [u8; 32],
    content_digest: [u8; 32],
}

#[link(name = "lfm_safetensors", kind = "static")]
#[cfg_attr(target_os = "macos", link(name = "c++"))]
#[cfg_attr(
    any(
        all(target_family = "unix", not(target_os = "macos")),
        all(target_os = "windows", target_env = "gnu")
    ),
    link(name = "stdc++")
)]
unsafe extern "C" {
    // Benchmark-private native entry point: intentionally absent from every
    // installed/product header and from liquid-audio's Rust API.
    fn lfm_internal_weights_open_bundle_benchmark(
        main_path: *const c_char,
        detokenizer_path: *const c_char,
        workers: u32,
        uncached: u32,
        out: *mut *mut WeightImage,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_internal_weights_benchmark_cold_supported() -> i32;
    fn lfm_weights_close(image: *mut WeightImage);
    fn lfm_weights_load_stats(image: *const WeightImage, out: *mut LoadStats) -> i32;
    fn lfm_weights_evict(identity: *const u8, error: *mut c_char, error_length: usize) -> i32;
}

struct Image(NonNull<WeightImage>);

impl Image {
    fn open(main: &Path, detokenizer: &Path, workers: u32, uncached: bool) -> Res<Self> {
        let main = CString::new(main.as_os_str().as_encoded_bytes())?;
        let detokenizer = CString::new(detokenizer.as_os_str().as_encoded_bytes())?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0i8; 1024];
        let status = unsafe {
            lfm_internal_weights_open_bundle_benchmark(
                main.as_ptr(),
                detokenizer.as_ptr(),
                workers,
                u32::from(uncached),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != OK {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(format!("native loader status {status}: {message}").into());
        }
        Ok(Self(
            NonNull::new(raw).ok_or("native loader returned a null image")?,
        ))
    }

    fn stats(&self) -> Res<LoadStats> {
        let mut stats = LoadStats {
            size: std::mem::size_of::<LoadStats>() as u32,
            abi_version: ABI,
            ..LoadStats::default()
        };
        let status = unsafe { lfm_weights_load_stats(self.0.as_ptr(), &mut stats) };
        if status != OK {
            return Err(format!("native load accounting status {status}").into());
        }
        if stats.size as usize != std::mem::size_of::<LoadStats>() || stats.abi_version != ABI {
            return Err("native load accounting ABI mismatch".into());
        }
        Ok(stats)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { lfm_weights_close(self.0.as_ptr()) };
    }
}

#[derive(Clone)]
struct Sample {
    built: bool,
    requested_workers: u32,
    elapsed_ms: f64,
    gib_per_second: Option<f64>,
    rss_after_open_bytes: Option<u64>,
    rss_delta_bytes: Option<i64>,
    stats: LoadStats,
    digest: String,
}

fn digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn evict(identity: &[u8; 32]) -> Res<()> {
    let mut error = [0i8; 1024];
    let status = unsafe { lfm_weights_evict(identity.as_ptr(), error.as_mut_ptr(), error.len()) };
    if status == OK {
        return Ok(());
    }
    Err(format!(
        "native segment eviction status {status}: {}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    )
    .into())
}

fn rss_bytes() -> Option<u64> {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn sample(
    main: &Path,
    detokenizer: &Path,
    workers: u32,
    uncached: bool,
    built: bool,
    identity: &[u8; 32],
) -> Res<Sample> {
    if built {
        evict(identity)?;
    }
    let rss_before = rss_bytes();
    let started = Instant::now();
    let image = Image::open(main, detokenizer, workers, uncached)?;
    let elapsed = started.elapsed();
    let rss_after = rss_bytes();
    let stats = image.stats()?;
    const DISPOSITION: u32 = BUILT | ATTACHED;
    const REQUIRED: u32 = WIRED;
    if stats.flags & REQUIRED != REQUIRED
        || stats.flags & DISPOSITION != if built { BUILT } else { ATTACHED }
        || stats.payload_read_calls
            != if built {
                u64::from(stats.task_count)
            } else {
                0
            }
        || stats.payload_read_bytes != if built { stats.source_bytes } else { 0 }
    {
        return Err(format!(
            "native loader returned the wrong {} accounting flags/counters",
            if built { "build" } else { "attach" }
        )
        .into());
    }
    let seconds = elapsed.as_secs_f64();
    Ok(Sample {
        built,
        requested_workers: workers,
        elapsed_ms: seconds * 1000.0,
        gib_per_second: built
            .then_some(stats.source_bytes as f64 / GIB / seconds.max(f64::MIN_POSITIVE)),
        rss_after_open_bytes: rss_after,
        rss_delta_bytes: rss_before
            .zip(rss_after)
            .map(|(before, after)| after as i64 - before as i64),
        stats,
        digest: digest(&stats.content_digest),
    })
}

fn percentile(values: impl IntoIterator<Item = f64>, quantile: f64) -> Option<f64> {
    let mut values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let rank = (quantile * values.len() as f64).ceil() as usize;
    Some(values[rank.saturating_sub(1).min(values.len() - 1)])
}

fn metric(samples: &[Sample], value: impl Fn(&Sample) -> Option<f64>) -> Value {
    let values = samples.iter().filter_map(value).collect::<Vec<_>>();
    json!({
        "p50": percentile(values.iter().copied(), 0.50),
        "p95": percentile(values, 0.95),
    })
}

fn summary(samples: &[Sample]) -> Value {
    let first = &samples[0];
    json!({
        "samples": samples.len(),
        "elapsed_ms": metric(samples, |sample| Some(sample.elapsed_ms)),
        "gib_per_second": metric(samples, |sample| sample.gib_per_second),
        "rss_after_open_bytes": metric(samples, |sample| sample.rss_after_open_bytes.map(|value| value as f64)),
        "rss_delta_bytes": metric(samples, |sample| sample.rss_delta_bytes.map(|value| value as f64)),
        "source_bytes": first.stats.source_bytes,
        "segment_bytes": first.stats.segment_bytes,
        "segment_constructed_bytes": first.stats.segment_constructed_bytes,
        "attached_shared_bytes": first.stats.attached_shared_bytes,
        "wired_bytes": first.stats.wired_bytes,
        "payload_read_calls": first.stats.payload_read_calls,
        "payload_read_bytes": first.stats.payload_read_bytes,
        "task_count": first.stats.task_count,
        "worker_count": first.stats.worker_count,
        "content_tree_sha256": first.digest,
    })
}

fn compare(serial: &[Sample], parallel: &[Sample]) -> Value {
    let serial_p50 = percentile(serial.iter().map(|sample| sample.elapsed_ms), 0.50).unwrap();
    let serial_p95 = percentile(serial.iter().map(|sample| sample.elapsed_ms), 0.95).unwrap();
    let parallel_p50 = percentile(parallel.iter().map(|sample| sample.elapsed_ms), 0.50).unwrap();
    let parallel_p95 = percentile(parallel.iter().map(|sample| sample.elapsed_ms), 0.95).unwrap();
    json!({
        "parallel_over_serial_p50": parallel_p50 / serial_p50,
        "parallel_over_serial_p95": parallel_p95 / serial_p95,
        "passes_no_regression_gate": parallel_p50 <= serial_p50 && parallel_p95 <= serial_p95,
    })
}

fn pair(
    main: &Path,
    detokenizer: &Path,
    runs: usize,
    uncached: bool,
    identity: &[u8; 32],
) -> Res<(Vec<Sample>, Vec<Sample>)> {
    let mut serial = Vec::with_capacity(runs);
    let mut parallel = Vec::with_capacity(runs);
    for run in 0..runs {
        let order = if run % 2 == 0 { [1, 4] } else { [4, 1] };
        for workers in order {
            let measured = sample(main, detokenizer, workers, uncached, true, identity)?;
            if workers == 1 {
                serial.push(measured);
            } else {
                parallel.push(measured);
            }
        }
    }
    Ok((serial, parallel))
}

fn attaches(main: &Path, detokenizer: &Path, runs: usize, identity: &[u8; 32]) -> Res<Vec<Sample>> {
    (0..runs)
        .map(|_| sample(main, detokenizer, 4, false, false, identity))
        .collect()
}

fn validate(samples: &[&[Sample]]) -> Res<()> {
    let first = samples
        .iter()
        .flat_map(|samples| samples.iter())
        .next()
        .ok_or("native loader benchmark produced no samples")?;
    for sample in samples.iter().flat_map(|samples| samples.iter()) {
        if sample.stats.source_bytes != first.stats.source_bytes
            || sample.stats.segment_bytes != first.stats.segment_bytes
            || sample.stats.task_count != first.stats.task_count
            || sample.digest != first.digest
        {
            return Err("native loader image/accounting changed between benchmark modes".into());
        }
        let expected = sample.stats.task_count.min(sample.requested_workers);
        if sample.built && sample.stats.worker_count != expected {
            return Err(format!(
                "loader reported {} workers for {} tasks",
                sample.stats.worker_count, sample.stats.task_count
            )
            .into());
        }
    }
    Ok(())
}

fn env_runs() -> Res<usize> {
    let runs = std::env::var("LFM_LOAD_BENCH_RUNS")
        .unwrap_or_else(|_| "5".into())
        .parse::<usize>()?;
    if runs < 2 {
        return Err("LFM_LOAD_BENCH_RUNS must be at least 2 for p50/p95".into());
    }
    Ok(runs)
}

fn main() -> Res<()> {
    /* The native loader now publishes correlated kcoro readiness edges. This
     * direct-FFI benchmark must retain the substrate crate even though Rust
     * owns no loader logic. */
    kcoro_sys::link_anchor();
    let main = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .ok_or("set LFM_MODEL_DIR to an explicit local LFM2-Audio checkpoint")?,
    );
    if !main.is_dir() {
        return Err(format!("LFM_MODEL_DIR is not a directory: {}", main.display()).into());
    }
    let detokenizer = std::env::var_os("LFM_LOAD_BENCH_DETOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| main.join("audio_detokenizer"));
    if !detokenizer.join("model.safetensors").is_file() {
        return Err(format!(
            "released audio detokenizer is missing: {}",
            detokenizer.display()
        )
        .into());
    }
    let runs = env_runs()?;
    let cold_supported = unsafe { lfm_internal_weights_benchmark_cold_supported() } == 1;
    let skip_cold = std::env::var_os("LFM_LOAD_BENCH_SKIP_COLD").is_some();

    /* Resolve the exact identity once, then make every build sample explicit
     * with evict. Without this preflight the persistent segment turns a
     * loader benchmark into an attach benchmark after its first sample. */
    let preflight = Image::open(&main, &detokenizer, 4, false)?;
    let identity = preflight.stats()?.identity_digest;
    drop(preflight);

    let cold = if cold_supported && !skip_cold {
        let (serial, parallel) = pair(&main, &detokenizer, runs, true, &identity)?;
        Some((serial, parallel))
    } else {
        None
    };

    /* One unreported build warms the source page cache. The segment is then
     * explicitly evicted by the first measured sample; only the file cache is
     * warm. */
    drop(sample(&main, &detokenizer, 4, false, true, &identity)?);
    let (warm_serial, warm_parallel) = pair(&main, &detokenizer, runs, false, &identity)?;
    /* The final warm build intentionally leaves READY behind. Every following
     * sample must attach with zero tensor-payload reads. */
    let attach = attaches(&main, &detokenizer, runs, &identity)?;
    let mut all = vec![
        warm_serial.as_slice(),
        warm_parallel.as_slice(),
        attach.as_slice(),
    ];
    if let Some((serial, parallel)) = &cold {
        all.push(serial);
        all.push(parallel);
    }
    validate(&all)?;

    let warm_compare = compare(&warm_serial, &warm_parallel);
    let cold_json = cold.as_ref().map(|(serial, parallel)| {
        json!({
            "serial": summary(serial),
            "four_worker": summary(parallel),
            "comparison": compare(serial, parallel),
        })
    });
    let report = json!({
        "checkpoint": main,
        "audio_detokenizer": detokenizer,
        "runs_per_mode": runs,
        "chunk_bytes": 8 * 1024 * 1024usize,
        "digest_algorithm": "LFM-WEIGHT-CONTENT-V1 sha256 tree",
        "cold_cache_control": if skip_cold {
            "skipped_by_LFM_LOAD_BENCH_SKIP_COLD"
        } else if cfg!(target_os = "macos") {
            "fcntl_F_NOCACHE"
        } else if cold_supported {
            "posix_fadvise_DONTNEED"
        } else {
            "unavailable"
        },
        "cold_build": cold_json,
        "warm_build": {
            "serial": summary(&warm_serial),
            "four_worker": summary(&warm_parallel),
            "comparison": warm_compare,
        },
        "attach": summary(&attach),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    let warm_pass = report["warm_build"]["comparison"]["passes_no_regression_gate"] == true;
    let cold_pass = report["cold_build"]
        .as_object()
        .map(|cold| cold["comparison"]["passes_no_regression_gate"] == true)
        .unwrap_or(true);
    if (!warm_pass || !cold_pass) && std::env::var_os("LFM_LOAD_BENCH_ALLOW_REGRESSION").is_none() {
        return Err("four-worker native loader regressed its serial baseline".into());
    }
    evict(&identity)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        assert_eq!(percentile([1.0, 4.0, 2.0, 3.0], 0.50), Some(2.0));
        assert_eq!(percentile([1.0, 4.0, 2.0, 3.0], 0.95), Some(4.0));
        assert_eq!(percentile([], 0.50), None);
    }
}
