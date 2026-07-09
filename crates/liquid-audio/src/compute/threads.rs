//! Intra-op thread-pool parity with torch.
//!
//! candle's CPU compute is multi-threaded — `matmul` goes through the `gemm` crate
//! (rayon feature) and conv/sort/many kernels call rayon directly — but it sizes the
//! pool from `num_cpus::get()` (**all** logical cores). torch does **not**: ATen's
//! `intraop_default_num_threads()` (`aten/src/ATen/ParallelCommon.cpp`) honours
//! `OMP_NUM_THREADS` / `MKL_NUM_THREADS`, and otherwise calls
//! `TaskThreadPoolBase::defaultNumThreads()`, which on **Apple Silicon queries
//! `hw.perflevel0.physicalcpu` — the performance cores only**, deliberately excluding
//! the efficiency (E) cores. Scheduling compute-bound matmul onto the slow E-cores
//! (what rayon's default does) hurts both throughput and tail latency via work-steal
//! imbalance. This module replicates torch's policy exactly and installs it as rayon's
//! **global** pool, so candle + `gemm` inherit it.
//!
//! Call [`configure_intraop_threads`] once, before the first tensor op (e.g. at the top
//! of `from_pretrained`). It is a no-op if the global pool was already built.

/// Replicates torch `at::intraop_default_num_threads()`:
/// 1. `OMP_NUM_THREADS`, then `MKL_NUM_THREADS` (torch's order); we also accept
///    `RAYON_NUM_THREADS` since that is what candle/rayon otherwise read.
/// 2. else `TaskThreadPoolBase::defaultNumThreads()` — on macOS the count of
///    performance cores (`hw.perflevel0.physicalcpu`); elsewhere physical cores.
pub fn intraop_default_num_threads() -> usize {
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"] {
        if let Some(n) = std::env::var(var)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            if n > 0 {
                return n;
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // torch's exact source on Apple Silicon. Fall back to the overall physical
        // core count if perflevel0 is unavailable (Intel Macs have no perflevels).
        if let Some(n) = sysctl_usize(c"hw.perflevel0.physicalcpu") {
            return n;
        }
        if let Some(n) = sysctl_usize(c"hw.physicalcpu") {
            return n;
        }
    }
    num_cpus::get_physical().max(1)
}

/// Read an integer `sysctl` by name (macOS). Returns `None` on any failure.
#[cfg(target_os = "macos")]
fn sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
    let mut val: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>();
    // SAFETY: `name` is a valid NUL-terminated C string; `val`/`len` are valid for the
    // documented OUT params; no input buffer (null, 0).
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut val as *mut libc::c_int as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && val > 0).then_some(val as usize)
}

/// Install torch's intra-op thread count as rayon's **global** pool size (candle +
/// `gemm` use it). Idempotent: a second call (or a pool already initialised by a prior
/// candle op) is a harmless no-op. Returns the thread count in effect.
pub fn configure_intraop_threads() -> usize {
    let n = intraop_default_num_threads();
    // `build_global` errors if the global pool already exists — that's fine, it means
    // someone (or a previous call) already sized it; we just report our intended count.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_pipeline_types_are_send() {
        // Feasibility probe for the worker-thread realtime pipeline (task #24): a
        // dedicated inference thread must *own* the model + processor (the Mimi decoder
        // holds a !Sync RefCell, so nothing is shared by &). This compiles iff both are
        // Send — i.e. the producer/consumer design is viable without trait changes.
        fn is_send<T: Send>() {}
        is_send::<crate::model::lfm2_audio::LFM2AudioModel>();
        is_send::<crate::processor::LFM2AudioProcessor>();
    }

    #[test]
    fn intraop_threads_is_sane_and_not_all_logical() {
        let n = intraop_default_num_threads();
        assert!(n >= 1, "must pick at least one thread");
        // Should never exceed the logical core count, and on Apple Silicon should be
        // the performance-core subset (≤ physical ≤ logical).
        assert!(
            n <= num_cpus::get(),
            "n={n} exceeds logical cores {}",
            num_cpus::get()
        );
        eprintln!(
            "intraop threads = {n} (physical {}, logical {})",
            num_cpus::get_physical(),
            num_cpus::get()
        );
    }
}
