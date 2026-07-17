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
//! honestly. `LFM_LOAD_BENCH_CODEC` may override the codec checkpoint path.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::NonNull;
use std::time::Instant;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ABI: u32 = 1;
const OK: i32 = 0;
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
    resident_bytes: u64,
    task_count: u32,
    worker_count: u32,
}

unsafe extern "C" {
    // Benchmark-private native entry point: intentionally absent from every
    // installed/product header and from liquid-audio's Rust API.
    fn lfm_internal_weights_open_bundle_benchmark(
        main_path: *const c_char,
        codec_path: *const c_char,
        workers: u32,
        uncached: u32,
        out: *mut *mut WeightImage,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_internal_weights_benchmark_cold_supported() -> i32;
    fn lfm_weights_close(image: *mut WeightImage);
    fn lfm_weights_data(image: *const WeightImage) -> *const c_void;
    fn lfm_weights_resident_bytes(image: *const WeightImage) -> u64;
    fn lfm_weights_load_stats(image: *const WeightImage, out: *mut LoadStats) -> i32;
}

struct Image(NonNull<WeightImage>);

impl Image {
    fn open(main: &Path, codec: &Path, workers: u32, uncached: bool) -> Res<Self> {
        let main = CString::new(main.as_os_str().as_encoded_bytes())?;
        let codec = CString::new(codec.as_os_str().as_encoded_bytes())?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0i8; 1024];
        let status = unsafe {
            lfm_internal_weights_open_bundle_benchmark(
                main.as_ptr(),
                codec.as_ptr(),
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

    fn digest(&self) -> Res<String> {
        let bytes = unsafe { lfm_weights_resident_bytes(self.0.as_ptr()) };
        let len = usize::try_from(bytes).map_err(|_| "resident image exceeds usize")?;
        let data = unsafe { lfm_weights_data(self.0.as_ptr()) }.cast::<u8>();
        if data.is_null() && len != 0 {
            return Err("native loader returned a null image base".into());
        }
        let image = unsafe { std::slice::from_raw_parts(data, len) };
        Ok(format!("{:x}", Sha256::digest(image)))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { lfm_weights_close(self.0.as_ptr()) };
    }
}

#[derive(Clone)]
struct Sample {
    requested_workers: u32,
    elapsed_ms: f64,
    gib_per_second: f64,
    rss_after_open_bytes: Option<u64>,
    rss_delta_bytes: Option<i64>,
    stats: LoadStats,
    digest: String,
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

fn sample(main: &Path, codec: &Path, workers: u32, uncached: bool) -> Res<Sample> {
    let rss_before = rss_bytes();
    let started = Instant::now();
    let image = Image::open(main, codec, workers, uncached)?;
    let elapsed = started.elapsed();
    let rss_after = rss_bytes();
    let stats = image.stats()?;
    let digest = image.digest()?;
    let seconds = elapsed.as_secs_f64();
    Ok(Sample {
        requested_workers: workers,
        elapsed_ms: seconds * 1000.0,
        gib_per_second: stats.source_bytes as f64 / GIB / seconds.max(f64::MIN_POSITIVE),
        rss_after_open_bytes: rss_after,
        rss_delta_bytes: rss_before
            .zip(rss_after)
            .map(|(before, after)| after as i64 - before as i64),
        stats,
        digest,
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
        "gib_per_second": metric(samples, |sample| Some(sample.gib_per_second)),
        "rss_after_open_bytes": metric(samples, |sample| sample.rss_after_open_bytes.map(|value| value as f64)),
        "rss_delta_bytes": metric(samples, |sample| sample.rss_delta_bytes.map(|value| value as f64)),
        "source_bytes": first.stats.source_bytes,
        "resident_image_bytes": first.stats.resident_bytes,
        "task_count": first.stats.task_count,
        "worker_count": first.stats.worker_count,
        "image_sha256": first.digest,
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

fn pair(main: &Path, codec: &Path, runs: usize, uncached: bool) -> Res<(Vec<Sample>, Vec<Sample>)> {
    let mut serial = Vec::with_capacity(runs);
    let mut parallel = Vec::with_capacity(runs);
    for run in 0..runs {
        let order = if run % 2 == 0 { [1, 4] } else { [4, 1] };
        for workers in order {
            let measured = sample(main, codec, workers, uncached)?;
            if workers == 1 {
                serial.push(measured);
            } else {
                parallel.push(measured);
            }
        }
    }
    Ok((serial, parallel))
}

fn validate(samples: &[&[Sample]]) -> Res<()> {
    let first = samples
        .iter()
        .flat_map(|samples| samples.iter())
        .next()
        .ok_or("native loader benchmark produced no samples")?;
    for sample in samples.iter().flat_map(|samples| samples.iter()) {
        if sample.stats.source_bytes != first.stats.source_bytes
            || sample.stats.resident_bytes != first.stats.resident_bytes
            || sample.stats.task_count != first.stats.task_count
            || sample.digest != first.digest
        {
            return Err("native loader image/accounting changed between benchmark modes".into());
        }
        let expected = sample.stats.task_count.min(sample.requested_workers);
        if sample.stats.worker_count != expected {
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
    let main = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .ok_or("set LFM_MODEL_DIR to an explicit local LFM2-Audio checkpoint")?,
    );
    if !main.is_dir() {
        return Err(format!("LFM_MODEL_DIR is not a directory: {}", main.display()).into());
    }
    let codec = std::env::var_os("LFM_LOAD_BENCH_CODEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| main.join("tokenizer-e351c8d8-checkpoint125.safetensors"));
    if !codec.is_file() {
        return Err(format!("Mimi codec checkpoint is missing: {}", codec.display()).into());
    }
    let runs = env_runs()?;
    let cold_supported = unsafe { lfm_internal_weights_benchmark_cold_supported() } == 1;
    let skip_cold = std::env::var_os("LFM_LOAD_BENCH_SKIP_COLD").is_some();

    let cold = if cold_supported && !skip_cold {
        let (serial, parallel) = pair(&main, &codec, runs, true)?;
        Some((serial, parallel))
    } else {
        None
    };

    // One unreported cached open makes the subsequent warm series explicit.
    drop(sample(&main, &codec, 4, false)?);
    let (warm_serial, warm_parallel) = pair(&main, &codec, runs, false)?;
    let mut all = vec![warm_serial.as_slice(), warm_parallel.as_slice()];
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
        "codec": codec,
        "runs_per_mode": runs,
        "chunk_bytes": 8 * 1024 * 1024usize,
        "digest_algorithm": "sha256",
        "cold_cache_control": if skip_cold {
            "skipped_by_LFM_LOAD_BENCH_SKIP_COLD"
        } else if cfg!(target_os = "macos") {
            "fcntl_F_NOCACHE"
        } else if cold_supported {
            "posix_fadvise_DONTNEED"
        } else {
            "unavailable"
        },
        "cold": cold_json,
        "warm": {
            "serial": summary(&warm_serial),
            "four_worker": summary(&warm_parallel),
            "comparison": warm_compare,
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    let warm_pass = report["warm"]["comparison"]["passes_no_regression_gate"] == true;
    let cold_pass = report["cold"]
        .as_object()
        .map(|cold| cold["comparison"]["passes_no_regression_gate"] == true)
        .unwrap_or(true);
    if (!warm_pass || !cold_pass) && std::env::var_os("LFM_LOAD_BENCH_ALLOW_REGRESSION").is_none() {
        return Err("four-worker native loader regressed its serial baseline".into());
    }
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
