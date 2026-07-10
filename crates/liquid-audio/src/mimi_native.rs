//! Rust rim of the native Mimi decode kernel (native/src/mimi/*, C++/NEON/AMX).
//!
//! The streaming audio-out hot path: one latent frame of codes in, up to
//! 1920 f32 samples at 24 kHz out, ~14 ms/frame on the M2 Max — measured
//! against moshi-Rust at ≤ 4.2e-6 worst PCM error across the 250-slot KV wrap.
//!
//! Weights are a buffer (the engine discipline): the checkpoint safetensors is
//! mmap'd once and every decoder tensor is handed to C as a zero-copy f32 span
//! in checkpoint layout. The C side folds/re-arms what it needs ONCE at init
//! into its own arena; nothing repacks per step. This struct owns the mmap for
//! the decoder's whole life — the C side keeps pointers into it.
//!
//! The kernel is the SUBSTRATE for streaming audio-out (no-fallbacks): if it
//! can't initialize, loading fails loudly. There is no moshi fallback on the
//! per-frame path — moshi remains only for turn-level tooling (encode, and the
//! one-shot whole-clip decode the byte oracles pin).

use std::ffi::{c_char, c_void, CString};
use std::sync::Mutex;

/// Mirror of the C `MimiWeight` (mimi_kernel.h). Field order/types exact.
#[repr(C)]
struct FfiMimiWeight {
    name: *const c_char,
    data: *const f32,
    shape: *const i64,
    ndim: u32,
    len: u64,
}

/// Mirror of the C `MimiWeightTable`.
#[repr(C)]
struct FfiMimiWeightTable {
    entries: *const FfiMimiWeight,
    count: u32,
}

extern "C" {
    fn mimi_decoder_new(
        d: *mut *mut c_void,
        w: *const FfiMimiWeightTable,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn mimi_decoder_step(d: *mut c_void, codes: *const u32, pcm_out: *mut f32) -> i32;
    fn mimi_decoder_reset(d: *mut c_void);
    fn mimi_decoder_free(d: *mut c_void);
}

/// `MIMI_FRAME_OUT * 2` (mimi_kernel.h): pcm_out capacity with drain headroom.
const PCM_CAP: usize = 1920 * 2;

/// The native streaming Mimi decoder. Interior-mutex'd like the moshi wrapper
/// (the C decoder is single-slot; the lock IS the `Sync`).
pub struct NativeMimi {
    dec: Mutex<*mut c_void>,
    codebooks: usize,
    // Keep-alive for the zero-copy weight spans the C side holds pointers into.
    _mmap: candle_core::safetensors::MmapedSafetensors,
    _names: Vec<CString>,
    _shapes: Vec<Vec<i64>>,
}

// SAFETY: the decoder pointer is only ever used under `dec`'s lock (single-slot
// C state machine, same contract as NativeEngine's pass_lock).
unsafe impl Send for NativeMimi {}
unsafe impl Sync for NativeMimi {}

impl NativeMimi {
    /// Build from the Mimi checkpoint safetensors. Hard error on ANY missing /
    /// misshaped weight or init failure — the kernel is required, not optional.
    pub fn new(
        checkpoint: &std::path::Path,
        codebooks: usize,
    ) -> std::result::Result<Self, String> {
        // The C ABI is FIXED eight-code (MIMI_NQ): mimi_decoder_step always
        // reads 8 codes. Any other count would pass the per-call length check
        // against the WRONG expectation and let C read out of bounds
        // (review P1). Reject at construction.
        if codebooks != 8 {
            return Err(format!(
                "mimi native: kernel ABI is fixed at 8 codebooks (MIMI_NQ), config says {codebooks}"
            ));
        }
        // SAFETY: same contract as every other mmap'd-weights load in this
        // stack (the file must not be truncated/mutated while mapped — the
        // checkpoint is a read-only model artifact).
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(checkpoint) }
            .map_err(|e| format!("mimi native: mmap {checkpoint:?}: {e}"))?;
        let views = mmap.tensors();
        let mut names: Vec<CString> = Vec::with_capacity(views.len());
        let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(views.len());
        // Two passes so the entry pointers reference FINAL storage addresses
        // (names/shapes vectors must not reallocate after we take pointers).
        for (name, view) in &views {
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(format!(
                    "mimi native: tensor '{name}' is {:?}, expected F32 \
                     (this checkpoint ships f32; a converted export needs a rim cast)",
                    view.dtype()
                ));
            }
            names.push(
                CString::new(name.as_str())
                    .map_err(|_| format!("mimi native: NUL in tensor name '{name}'"))?,
            );
            shapes.push(view.shape().iter().map(|&d| d as i64).collect());
        }
        let entries: Vec<FfiMimiWeight> = views
            .iter()
            .enumerate()
            .map(|(i, (_, view))| FfiMimiWeight {
                name: names[i].as_ptr(),
                data: view.data().as_ptr() as *const f32,
                shape: shapes[i].as_ptr(),
                ndim: shapes[i].len() as u32,
                len: (view.data().len() / 4) as u64,
            })
            .collect();
        let table = FfiMimiWeightTable {
            entries: entries.as_ptr(),
            count: entries.len() as u32,
        };
        let mut dec: *mut c_void = std::ptr::null_mut();
        let mut err = [0i8; 512];
        // SAFETY: table/entries/names/shapes all outlive this call; the C side
        // copies what it needs at init and keeps only WEIGHT-DATA pointers,
        // which `_mmap` keeps alive for the decoder's lifetime.
        let rc = unsafe {
            mimi_decoder_new(
                &mut dec,
                &table,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 || dec.is_null() {
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr() as *const c_char) };
            return Err(format!(
                "mimi native: decoder init failed (rc {rc}): {}",
                msg.to_string_lossy()
            ));
        }
        Ok(Self {
            dec: Mutex::new(dec),
            codebooks,
            _mmap: mmap,
            _names: names,
            _shapes: shapes,
        })
    }

    /// One latent frame of codes → PCM samples at 24 kHz. Empty result is codec
    /// priming (never an error — errors are `Err`, per the C ABI's negative rc).
    pub fn decode_step(&self, codes: &[u32]) -> std::result::Result<Vec<f32>, String> {
        if codes.len() != self.codebooks {
            return Err(format!(
                "mimi native: {} codes, expected {}",
                codes.len(),
                self.codebooks
            ));
        }
        let dec = self.dec.lock().unwrap_or_else(|p| p.into_inner());
        let mut pcm = vec![0f32; PCM_CAP];
        // SAFETY: dec valid under the lock; codes length checked; pcm has the
        // header-contract capacity.
        let n = unsafe { mimi_decoder_step(*dec, codes.as_ptr(), pcm.as_mut_ptr()) };
        if n < 0 {
            return Err(format!("mimi native: decode_step failed (rc {n})"));
        }
        pcm.truncate(n as usize);
        Ok(pcm)
    }

    /// Turn boundary: re-arm all streaming state (conv carries, KV ring).
    pub fn reset(&self) {
        let dec = self.dec.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: dec valid under the lock.
        unsafe { mimi_decoder_reset(*dec) };
    }
}

impl Drop for NativeMimi {
    fn drop(&mut self) {
        let dec = self.dec.get_mut().unwrap_or_else(|p| p.into_inner());
        if !dec.is_null() {
            // SAFETY: owned pointer, dropped exactly once.
            unsafe { mimi_decoder_free(*dec) };
            *dec = std::ptr::null_mut();
        }
    }
}
