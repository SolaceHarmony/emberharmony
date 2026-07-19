//! Rust rim over the native Conformer encoder + audio adapter
//! (native/src/model/lfm_conformer.cpp + flashkern_conformer.S). The Rust
//! `ConformerEncoder`/`MLP` audio path is DELETED; the encoder+adapter math is
//! native, binding views into the resident safetensors image and running on
//! the Flashkern lane team. Parity is gated by native/tests/fixtures/conformer
//! (real checkpoint, BF16 production ladder, captured from the deleted Rust).
//!
//! This rim owns the native handle's lifetime and one transport at the seam:
//! it reads a BF16 mel segment tensor and returns the adapted embedding rows as
//! a BF16 tensor for the (still-Candle) prefill assembly. That tensor round-trip
//! dies at the doc 07 conversation cutover; it is transport, not math.

use candle_core::{DType, Device, Result, Tensor};

use crate::flashkern::native_engine::process_engine;
use crate::weights::ResidentWeights;

#[repr(C)]
struct RawConformer {
    _private: [u8; 0],
}
#[repr(C)]
struct RawWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct Geometry {
    size: u32,
    abi_version: u32,
    feat_in: u32,
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    d_ff: u32,
    conv_kernel: u32,
    subsampling: u32,
    conv_channels: u32,
    adapter_hidden: u32,
    adapter_out: u32,
    reserved: [u64; 4],
}

const CONFORMER_ABI: u32 = 1;

unsafe extern "C" {
    fn lfm_conformer_create(
        engine: *mut std::ffi::c_void,
        weights: *const std::ffi::c_void,
        geometry: *const Geometry,
        out: *mut *mut RawConformer,
        error: *mut std::ffi::c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_conformer_destroy(c: *mut RawConformer) -> i32;
    fn lfm_conformer_bound_weight_bytes(c: *const RawConformer) -> u64;
    fn lfm_conformer_derived_bytes(c: *const RawConformer) -> u64;
    fn lfm_conformer_materialized_weight_bytes(c: *const RawConformer) -> u64;
    fn lfm_conformer_direct_gemm_calls(c: *const RawConformer) -> u64;
    fn lfm_conformer_workspace_create(out: *mut *mut RawWorkspace) -> i32;
    fn lfm_conformer_workspace_destroy(w: *mut RawWorkspace) -> i32;
    fn lfm_conformer_out_rows(c: *const RawConformer, mel_frames: u64) -> u64;
    fn lfm_conformer_forward(
        c: *const RawConformer,
        w: *mut RawWorkspace,
        mel: *const u16,
        mel_frames: u64,
        out_rows: *mut u16,
        out_capacity_values: u64,
    ) -> i32;
}

/// Parsed `audio_encoder` config from config.json — the geometry source of
/// truth (was `conformer::encoder::ConformerEncoderConfig`; the Candle encoder
/// is deleted, the config lives here now).
#[derive(Debug, Clone)]
pub struct ConformerEncoderConfig {
    pub feat_in: usize,
    pub feat_out: usize, // -1 in Python → 0 here means "= d_model"
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize, // 0 → = d_model
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub xscaling: bool,
    /// `rel_pos` (the model's config) or `abs_pos`. Only `rel_pos` is supported.
    pub self_attention_model: String,
}

/// Geometry the native binder validates against the checkpoint tensors.
#[derive(Debug, Clone, Copy)]
pub struct ConformerGeometry {
    pub feat_in: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling: usize,
    pub conv_channels: usize,
    pub adapter_hidden: usize,
    pub adapter_out: usize,
}

/// Audit counters for the immutable checkpoint-view contract. A production
/// forward must leave `materialized_weight_bytes` at zero while increasing the
/// direct-GEMM witness count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformerMemory {
    pub bound_weight_bytes: u64,
    pub derived_bytes: u64,
    pub materialized_weight_bytes: u64,
    pub direct_gemm_calls: u64,
}

/// Native Conformer encoder + audio adapter. Holds the resident image alive
/// (so bound weight views stay valid). The scratch workspace is NOT a field:
/// `forward_segment` allocates and frees it per call, so this type carries no
/// shared mutable state.
pub struct NativeConformer {
    handle: *mut RawConformer,
    feat_in: usize,
    adapter_out: usize,
    device: Device,
    // Keeps the resident image (and its bound views) alive for our lifetime.
    _resident: ResidentWeights,
}

// SAFETY: the handle holds only immutable bound weight views and tables built
// at create time; `lfm_conformer_forward` takes it as `const` and writes only
// into a workspace that `forward_segment` allocates and frees privately per
// call. Nothing mutable is reachable through `&self`, so sharing across threads
// (Sync) and moving between them (Send) are both sound.
unsafe impl Send for NativeConformer {}
unsafe impl Sync for NativeConformer {}

impl Drop for NativeConformer {
    fn drop(&mut self) {
        unsafe {
            let _ = lfm_conformer_destroy(self.handle);
        }
    }
}

impl NativeConformer {
    pub fn new(
        resident: ResidentWeights,
        geometry: ConformerGeometry,
        device: &Device,
    ) -> Result<Self> {
        let g = Geometry {
            size: std::mem::size_of::<Geometry>() as u32,
            abi_version: CONFORMER_ABI,
            feat_in: geometry.feat_in as u32,
            d_model: geometry.d_model as u32,
            n_layers: geometry.n_layers as u32,
            n_heads: geometry.n_heads as u32,
            d_ff: geometry.d_ff as u32,
            conv_kernel: geometry.conv_kernel as u32,
            subsampling: geometry.subsampling as u32,
            conv_channels: geometry.conv_channels as u32,
            adapter_hidden: geometry.adapter_hidden as u32,
            adapter_out: geometry.adapter_out as u32,
            reserved: [0; 4],
        };
        let mut handle: *mut RawConformer = std::ptr::null_mut();
        let mut err = [0i8; 256];
        let rc = unsafe {
            lfm_conformer_create(
                process_engine().raw_ptr(),
                resident.raw_image_ptr(),
                &g,
                &mut handle,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 || handle.is_null() {
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(candle_core::Error::Msg(format!(
                "native conformer create failed (status {rc}): {msg}"
            )));
        }
        Ok(Self {
            handle,
            feat_in: geometry.feat_in,
            adapter_out: geometry.adapter_out,
            device: device.clone(),
            _resident: resident,
        })
    }

    pub fn memory(&self) -> ConformerMemory {
        // SAFETY: all four queries are read-only and `self.handle` remains live
        // for the duration of the calls.
        unsafe {
            ConformerMemory {
                bound_weight_bytes: lfm_conformer_bound_weight_bytes(self.handle),
                derived_bytes: lfm_conformer_derived_bytes(self.handle),
                materialized_weight_bytes: lfm_conformer_materialized_weight_bytes(self.handle),
                direct_gemm_calls: lfm_conformer_direct_gemm_calls(self.handle),
            }
        }
    }

    /// One audio-in segment: `mel` is `(1, feat_in, T)` or `(feat_in, T)`,
    /// consumed as BF16 (the prefill seam cast). Returns adapted embedding rows
    /// `(out_rows, adapter_out)` BF16 — the encoder+adapter output, ready to
    /// scatter into the prefill embedding plane.
    pub fn forward_segment(&self, mel: &Tensor) -> Result<Tensor> {
        let mel = mel.to_dtype(DType::BF16)?;
        let dims = mel.dims();
        let (feat_in, frames) = match dims {
            [c, t] => (*c, *t),
            [1, c, t] => (*c, *t),
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "native conformer: mel segment must be (feat_in, T), got {dims:?}"
                )))
            }
        };
        // The native reader indexes the mel buffer with the geometry's feat_in;
        // a segment with a different feature count would drive an out-of-bounds
        // read inside the C++ forward. Hard-gate it here (not a debug_assert).
        if feat_in != self.feat_in {
            return Err(candle_core::Error::Msg(format!(
                "native conformer: mel feat_in {feat_in} != geometry feat_in {}",
                self.feat_in
            )));
        }
        // BF16 bits, row-major (feat_in x frames) — the ChatState audio_in layout.
        let mel = mel.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mel_bits: Vec<u16> = mel.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

        let out_rows = unsafe { lfm_conformer_out_rows(self.handle, frames as u64) } as usize;
        if out_rows == 0 {
            return Err(candle_core::Error::Msg(format!(
                "native conformer: segment of {frames} mel frames yields no rows"
            )));
        }
        let values = out_rows * self.adapter_out;
        let mut out_bits = vec![0u16; values];

        // Per-call scratch: created and freed here so the encoder holds no
        // shared mutable state (this is what keeps &self + Send/Sync sound).
        // Audio-in segments are per-turn, not per-token — the alloc is off the
        // hot path.
        let mut workspace: *mut RawWorkspace = std::ptr::null_mut();
        let rc = unsafe { lfm_conformer_workspace_create(&mut workspace) };
        if rc != 0 || workspace.is_null() {
            return Err(candle_core::Error::Msg(format!(
                "native conformer workspace create failed (status {rc})"
            )));
        }
        let rc = unsafe {
            lfm_conformer_forward(
                self.handle,
                workspace,
                mel_bits.as_ptr(),
                frames as u64,
                out_bits.as_mut_ptr(),
                values as u64,
            )
        };
        unsafe {
            let _ = lfm_conformer_workspace_destroy(workspace);
        }
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!(
                "native conformer forward failed (status {rc}, {frames} frames)"
            )));
        }
        // Widen BF16 bits -> f32 -> BF16 tensor (transport; dies with doc 07).
        let f32s: Vec<f32> = out_bits
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        Tensor::from_vec(f32s, (out_rows, self.adapter_out), &self.device)?.to_dtype(DType::BF16)
    }
}
