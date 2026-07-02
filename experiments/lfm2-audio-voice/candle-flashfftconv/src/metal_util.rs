//! Shared Metal glue: compile + cache a named compute pipeline from inline source.
//!
//! candle 0.9 builds Metal pipelines through `candle_metal_kernels::metal` (its
//! objc2-metal shim), so external custom ops compile shaders against
//! `MetalDevice::metal_device()` and cache the resulting `ComputePipeline`.

#![cfg(feature = "metal")]

use candle_core::backend::BackendStorage;
use candle_core::{DType, Layout, MetalDevice, MetalStorage, Result, Shape};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Global, thread-safe cache of compiled pipelines, keyed by kernel name. A
/// `ComputePipeline` is the immutable **compiled kernel** (verified `Send + Sync`, see
/// `thread_safety`), so it is compiled **once process-wide** and shared across every
/// thread; each dispatch then builds its own cheap, per-thread command encoder against
/// the shared pipeline — compiled-code vs. instances-of-the-code.
static PIPELINES: OnceLock<Mutex<HashMap<&'static str, ComputePipeline>>> = OnceLock::new();

fn pipelines() -> &'static Mutex<HashMap<&'static str, ComputePipeline>> {
    PIPELINES.get_or_init(|| Mutex::new(HashMap::new()))
}

// Test-only per-thread compile counter. The pipeline cache is thread-local, so this
// mirrors it: a reused kernel compiles once on a thread and then dispatches from the
// cached `ComputePipeline`, so this stops incrementing on reuse. Thread-local (not
// global) so the count is unaffected by other tests compiling on other threads.
#[cfg(test)]
thread_local! {
    static PIPELINE_COMPILES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// MSL compiles (cache misses) on the **current thread**. Stops incrementing once a
/// kernel is cached — the check behind the "compiled once, then reused" test.
#[cfg(test)]
pub fn pipeline_compiles() -> u64 {
    PIPELINE_COMPILES.with(|c| c.get())
}

#[cfg(test)]
mod thread_safety {
    use super::ComputePipeline;
    // Compiles only if a compiled pipeline is safe to share across threads — i.e. the
    // compiled kernel can live in one global cache, not one-per-thread. Fails the BUILD
    // (not just at runtime) if `ComputePipeline` is not `Send + Sync`.
    #[test]
    fn compiled_pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComputePipeline>();
    }
}

/// Compile `source` and return the pipeline for `fn_name`, cached **process-wide**
/// (shared across threads) by `fn_name`. Assumes a single Metal device (the common case).
pub fn pipeline(dev: &MetalDevice, fn_name: &'static str, source: &str) -> Result<ComputePipeline> {
    // Fast path: the shared compiled kernel is already cached.
    if let Some(p) = pipelines().lock().unwrap().get(fn_name) {
        return Ok(p.clone());
    }
    // Miss: compile *without* holding the lock — an MSL compile is slow and other threads
    // must still be able to look up their (already-cached) kernels. A cold race may
    // compile the same kernel twice; that is benign (last insert wins).
    let mtl = dev.metal_device();
    let lib = mtl
        .new_library_with_source(source, None)
        .map_err(|e| candle_core::Error::Msg(format!("metal compile {fn_name}: {e}")))?;
    let func = lib
        .get_function(fn_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("metal get_function {fn_name}: {e}")))?;
    let p = mtl
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("metal pipeline {fn_name}: {e}")))?;
    #[cfg(test)]
    PIPELINE_COMPILES.with(|c| c.set(c.get() + 1));
    pipelines().lock().unwrap().insert(fn_name, p.clone());
    Ok(p)
}

/// Dispatch a butterfly phase: two f32 inputs (`x` at buffer 0, a matrix/twiddle
/// table at buffer 1), `[B,H,N,L]` scalars at buffers 3..7, one complex output
/// `[B,H,N,L,2]` at buffer 2. One thread per complex element; the kernel guards
/// `gid >= B*H*N*L`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dft(
    xs: &MetalStorage,
    xl: &Layout,
    ds: &MetalStorage,
    dl: &Layout,
    src: &str,
    fn_name: &'static str,
    b: usize,
    h: usize,
    n: usize,
    l: usize,
    out_complex: bool,
) -> Result<(MetalStorage, Shape)> {
    if xs.dtype() != DType::F32 || ds.dtype() != DType::F32 {
        candle_core::bail!("butterfly metal kernels support f32 only");
    }
    let total = b * h * n * l;
    let out_el = if out_complex { total * 2 } else { total };
    let out_shape = if out_complex {
        Shape::from((b, h, n, l, 2))
    } else {
        Shape::from((b, h, n, l))
    };
    let dev = xs.device();
    let p = pipeline(dev, fn_name, src)?;
    let dts = DType::F32.size_in_bytes();

    let out_buf = dev.new_buffer(out_el, DType::F32, fn_name)?;
    let enc = dev.command_encoder()?;
    enc.set_compute_pipeline_state(&p);
    enc.set_buffer(0, Some(xs.buffer()), xl.start_offset() * dts);
    enc.set_buffer(1, Some(ds.buffer()), dl.start_offset() * dts);
    enc.set_buffer(2, Some(&*out_buf), 0);
    enc.set_bytes(3, &(b as u32));
    enc.set_bytes(4, &(h as u32));
    enc.set_bytes(5, &(n as u32));
    enc.set_bytes(6, &(l as u32));

    let max_tg = p.max_total_threads_per_threadgroup().max(1);
    let tg = total.clamp(1, max_tg);
    let ng = total.div_ceil(tg);
    enc.dispatch_thread_groups(
        MTLSize {
            width: ng,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok((
        MetalStorage::new(out_buf, dev.clone(), out_el, DType::F32),
        out_shape,
    ))
}
