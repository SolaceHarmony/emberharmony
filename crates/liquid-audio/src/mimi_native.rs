//! Rust rim of the native Mimi decode kernel (native/src/mimi/*, C++/NEON/AMX).
//!
//! The streaming audio-out hot path: one latent frame of codes in, up to
//! 1920 f32 samples at 24 kHz out, ~14 ms/frame on the M2 Max — measured
//! against moshi-Rust at ≤ 4.2e-6 worst PCM error across the 250-slot KV wrap.
//!
//! Weights are a native-owned buffer (the engine discipline): C++ reads the
//! checkpoint directly into one aligned resident image, parses tensor spans in
//! place, and keeps that image alive inside the decoder. Rust passes one path;
//! it never constructs Candle tensors or safetensors descriptors for this path.
//!
//! The kernel is the SUBSTRATE for streaming audio-out (no-fallbacks): if it
//! can't initialize, loading fails loudly. There is no moshi fallback on the
//! per-frame path — moshi remains only for turn-level tooling (encode, and the
//! one-shot whole-clip decode the byte oracles pin).

use std::ffi::{c_char, c_void, CString};
use std::sync::Mutex;

extern "C" {
    fn mimi_decoder_new_from_file(
        d: *mut *mut c_void,
        checkpoint: *const c_char,
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
        let path = CString::new(checkpoint.as_os_str().as_encoded_bytes())
            .map_err(|_| format!("mimi native: NUL in checkpoint path {checkpoint:?}"))?;
        let mut dec: *mut c_void = std::ptr::null_mut();
        let mut err = [0i8; 512];
        // SAFETY: path is NUL-terminated for the call. The returned decoder owns
        // the native resident weight image and releases it in mimi_decoder_free.
        let rc = unsafe {
            mimi_decoder_new_from_file(
                &mut dec,
                path.as_ptr(),
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
