//! Typed PCM -> adapted-row pass gate. The fixture is a small but complete
//! Conformer with real native weights/views and workspaces; no numerical mock or
//! duplicated implementation participates in the comparison.

use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(C)]
struct WeightImage {
    _private: [u8; 0],
}

#[repr(C)]
struct Frontend {
    _private: [u8; 0],
}

#[repr(C)]
struct FrontendWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct Resampler {
    _private: [u8; 0],
}

#[repr(C)]
struct ResamplerWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct Conformer {
    _private: [u8; 0],
}

#[repr(C)]
struct ConformerWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct FrontendConfig {
    size: u32,
    abi_version: u32,
    sample_rate: u32,
    n_window_size: u32,
    n_window_stride: u32,
    n_fft: u32,
    nfilt: u32,
    exact_pad: u32,
    pad_to: u32,
    reserved0: u32,
    preemph: f64,
    log_zero_guard_value: f64,
    mag_power: f64,
    reserved: [u64; 4],
}

#[repr(C)]
struct ConformerGeometry {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct F32Span {
    data: *const f32,
    length: u64,
}

#[repr(C)]
struct AudioPass {
    size: u32,
    abi_version: u32,
    resampler: *const Resampler,
    resampler_workspace: *mut ResamplerWorkspace,
    frontend: *const Frontend,
    frontend_workspace: *mut FrontendWorkspace,
    conformer: *const Conformer,
    conformer_workspace: *mut ConformerWorkspace,
    pcm: *const f32,
    sample_count: u64,
    resampled: *mut f32,
    resampled_capacity: u64,
    mel: *mut u16,
    mel_capacity: u64,
    adapted: *mut u16,
    adapted_capacity: u64,
    out_adapted_values: *mut u64,
}

unsafe extern "C" {
    fn lfm_weights_open(
        path: *const c_char,
        out: *mut *mut WeightImage,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_weights_close(image: *mut WeightImage);
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_free(engine: *mut c_void);
    fn lfm_engine_audio_encode(engine: *mut c_void, model_id: u64, pass: *const AudioPass) -> i32;
    fn lfm_engine_audio_encode_passes(engine: *const c_void) -> u64;
    fn lfm_frontend_create(config: *const FrontendConfig, out: *mut *mut Frontend) -> i32;
    fn lfm_frontend_destroy(frontend: *mut Frontend) -> i32;
    fn lfm_frontend_workspace_create(out: *mut *mut FrontendWorkspace) -> i32;
    fn lfm_frontend_workspace_destroy(workspace: *mut FrontendWorkspace) -> i32;
    fn lfm_frontend_workspace_reserve(
        frontend: *const Frontend,
        workspace: *mut FrontendWorkspace,
        max_sample_count: u64,
        flags: u32,
    ) -> i32;
    fn lfm_frontend_seq_len(frontend: *const Frontend, sample_count: u64) -> u64;
    fn lfm_frontend_forward_bf16_workspace(
        frontend: *const Frontend,
        workspace: *mut FrontendWorkspace,
        pcm: *const f32,
        sample_count: u64,
        out: *mut u16,
        capacity: u64,
    ) -> i32;
    fn lfm_resampler_create(orig: u32, new: u32, out: *mut *mut Resampler) -> i32;
    fn lfm_resampler_destroy(resampler: *mut Resampler) -> i32;
    fn lfm_resampler_workspace_create(out: *mut *mut ResamplerWorkspace) -> i32;
    fn lfm_resampler_workspace_destroy(workspace: *mut ResamplerWorkspace) -> i32;
    fn lfm_resampler_workspace_reserve(
        resampler: *const Resampler,
        workspace: *mut ResamplerWorkspace,
        max_sample_count: u64,
    ) -> i32;
    fn lfm_resampler_out_length(
        resampler: *const Resampler,
        sample_count: u64,
        out: *mut u64,
    ) -> i32;
    fn lfm_resampler_process(
        resampler: *const Resampler,
        workspace: *mut ResamplerWorkspace,
        input: *const f32,
        sample_count: u64,
        destination: *mut f32,
        capacity: u64,
        out: *mut F32Span,
    ) -> i32;
    fn lfm_conformer_create(
        engine: *mut c_void,
        weights: *const c_void,
        geometry: *const ConformerGeometry,
        out: *mut *mut Conformer,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_conformer_destroy(conformer: *mut Conformer) -> i32;
    fn lfm_conformer_workspace_create(out: *mut *mut ConformerWorkspace) -> i32;
    fn lfm_conformer_workspace_destroy(workspace: *mut ConformerWorkspace) -> i32;
    fn lfm_conformer_workspace_reserve(
        conformer: *const Conformer,
        workspace: *mut ConformerWorkspace,
        max_mel_frames: u64,
    ) -> i32;
    fn lfm_conformer_out_rows(conformer: *const Conformer, mel_frames: u64) -> u64;
    fn lfm_conformer_out_width(conformer: *const Conformer) -> u64;
    fn lfm_conformer_materialized_weight_bytes(conformer: *const Conformer) -> u64;
    fn lfm_conformer_direct_gemm_calls(conformer: *const Conformer) -> u64;
    fn lfm_conformer_forward(
        conformer: *const Conformer,
        workspace: *mut ConformerWorkspace,
        mel: *const u16,
        mel_frames: u64,
        out: *mut u16,
        capacity: u64,
    ) -> i32;
}

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "emberharmony-native-audio-pass-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap_or_default();
    }
}

struct Spec {
    name: String,
    shape: Vec<u64>,
    value: u16,
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn value(name: &str) -> u16 {
    if name.ends_with("running_var") || name.contains("norm") && name.ends_with("weight") {
        return bf16(1.0);
    }
    if name.ends_with("bias") || name.ends_with("running_mean") {
        return 0;
    }
    if name.contains("pos_bias") {
        return bf16(0.015625);
    }
    bf16(0.03125)
}

fn add(specs: &mut Vec<Spec>, name: impl Into<String>, shape: &[u64]) {
    let name = name.into();
    specs.push(Spec {
        value: value(&name),
        name,
        shape: shape.to_vec(),
    });
}

fn write_conformer(path: &Path) {
    let mut specs = Vec::new();
    for (name, shape) in [
        ("conformer.pre_encode.conv.0.weight", vec![1, 1, 3, 3]),
        ("conformer.pre_encode.conv.0.bias", vec![1]),
        ("conformer.pre_encode.conv.2.weight", vec![1, 1, 3, 3]),
        ("conformer.pre_encode.conv.2.bias", vec![1]),
        ("conformer.pre_encode.conv.3.weight", vec![1, 1, 1, 1]),
        ("conformer.pre_encode.conv.3.bias", vec![1]),
        ("conformer.pre_encode.conv.5.weight", vec![1, 1, 3, 3]),
        ("conformer.pre_encode.conv.5.bias", vec![1]),
        ("conformer.pre_encode.conv.6.weight", vec![1, 1, 1, 1]),
        ("conformer.pre_encode.conv.6.bias", vec![1]),
        ("conformer.pre_encode.out.weight", vec![8, 1]),
        ("conformer.pre_encode.out.bias", vec![8]),
    ] {
        add(&mut specs, name, &shape);
    }

    let root = "conformer.layers.0.";
    for (suffix, shape) in [
        ("norm_feed_forward1.weight", vec![8]),
        ("norm_feed_forward1.bias", vec![8]),
        ("feed_forward1.linear1.weight", vec![8, 8]),
        ("feed_forward1.linear1.bias", vec![8]),
        ("feed_forward1.linear2.weight", vec![8, 8]),
        ("feed_forward1.linear2.bias", vec![8]),
        ("norm_self_att.weight", vec![8]),
        ("norm_self_att.bias", vec![8]),
        ("self_attn.linear_q.weight", vec![8, 8]),
        ("self_attn.linear_q.bias", vec![8]),
        ("self_attn.linear_k.weight", vec![8, 8]),
        ("self_attn.linear_k.bias", vec![8]),
        ("self_attn.linear_v.weight", vec![8, 8]),
        ("self_attn.linear_v.bias", vec![8]),
        ("self_attn.linear_out.weight", vec![8, 8]),
        ("self_attn.linear_out.bias", vec![8]),
        ("self_attn.linear_pos.weight", vec![8, 8]),
        ("self_attn.pos_bias_u", vec![1, 8]),
        ("self_attn.pos_bias_v", vec![1, 8]),
        ("norm_conv.weight", vec![8]),
        ("norm_conv.bias", vec![8]),
        ("conv.pointwise_conv1.weight", vec![16, 8, 1]),
        ("conv.pointwise_conv1.bias", vec![16]),
        ("conv.depthwise_conv.weight", vec![8, 1, 3]),
        ("conv.depthwise_conv.bias", vec![8]),
        ("conv.pointwise_conv2.weight", vec![8, 8, 1]),
        ("conv.pointwise_conv2.bias", vec![8]),
        ("conv.batch_norm.weight", vec![8]),
        ("conv.batch_norm.bias", vec![8]),
        ("conv.batch_norm.running_mean", vec![8]),
        ("conv.batch_norm.running_var", vec![8]),
        ("norm_feed_forward2.weight", vec![8]),
        ("norm_feed_forward2.bias", vec![8]),
        ("feed_forward2.linear1.weight", vec![8, 8]),
        ("feed_forward2.linear1.bias", vec![8]),
        ("feed_forward2.linear2.weight", vec![8, 8]),
        ("feed_forward2.linear2.bias", vec![8]),
        ("norm_out.weight", vec![8]),
        ("norm_out.bias", vec![8]),
    ] {
        add(&mut specs, format!("{root}{suffix}"), &shape);
    }
    for (name, shape) in [
        ("audio_adapter.model.0.weight", vec![8]),
        ("audio_adapter.model.0.bias", vec![8]),
        ("audio_adapter.model.1.weight", vec![8, 8]),
        ("audio_adapter.model.1.bias", vec![8]),
        ("audio_adapter.model.3.weight", vec![8, 8]),
        ("audio_adapter.model.3.bias", vec![8]),
    ] {
        add(&mut specs, name, &shape);
    }

    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for spec in specs {
        let begin = payload.len();
        let count = spec.shape.iter().product::<u64>() as usize;
        for _ in 0..count {
            payload.extend_from_slice(&spec.value.to_le_bytes());
        }
        header.insert(
            spec.name,
            serde_json::json!({
                "dtype": "BF16",
                "shape": spec.shape,
                "data_offsets": [begin, payload.len()],
            }),
        );
    }
    let mut bytes = serde_json::to_vec(&header).unwrap();
    bytes.resize((bytes.len() + 7) & !7, b' ');
    let mut file = Vec::with_capacity(8 + bytes.len() + payload.len());
    file.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&bytes);
    file.extend_from_slice(&payload);
    std::fs::write(path, file).unwrap();
}

#[test]
fn typed_audio_encode_matches_stage_oracle_and_reuses_prepared_storage() {
    std::hint::black_box(std::mem::size_of::<liquid_audio::NativeVoiceSampling>());
    kcoro_sys::link_anchor();

    let temp = Temp::new();
    let path = temp.0.join("conformer.safetensors");
    write_conformer(&path);
    let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let mut image = std::ptr::null_mut();
    let mut error = [0i8; 512];
    assert_eq!(
        unsafe { lfm_weights_open(path.as_ptr(), &mut image, error.as_mut_ptr(), error.len()) },
        0
    );

    let engine = unsafe { lfm_engine_new(2) };
    assert!(!engine.is_null());
    let geometry = ConformerGeometry {
        size: std::mem::size_of::<ConformerGeometry>() as u32,
        abi_version: 1,
        feat_in: 8,
        d_model: 8,
        n_layers: 1,
        n_heads: 1,
        d_ff: 8,
        conv_kernel: 3,
        subsampling: 8,
        conv_channels: 1,
        adapter_hidden: 8,
        adapter_out: 8,
        reserved: [0; 4],
    };
    let mut conformer = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            lfm_conformer_create(
                engine,
                image.cast(),
                &geometry,
                &mut conformer,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        0,
        "{}",
        unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );

    let config = FrontendConfig {
        size: std::mem::size_of::<FrontendConfig>() as u32,
        abi_version: 1,
        sample_rate: 16_000,
        n_window_size: 8,
        n_window_stride: 4,
        n_fft: 8,
        nfilt: 8,
        exact_pad: 0,
        pad_to: 0,
        reserved0: 0,
        preemph: 0.97,
        log_zero_guard_value: 2f64.powi(-24),
        mag_power: 2.0,
        reserved: [0; 4],
    };
    let mut frontend = std::ptr::null_mut();
    let mut frontend_workspace = std::ptr::null_mut();
    let mut resampler = std::ptr::null_mut();
    let mut resampler_workspace = std::ptr::null_mut();
    let mut conformer_workspace = std::ptr::null_mut();
    unsafe {
        assert_eq!(lfm_frontend_create(&config, &mut frontend), 0);
        assert_eq!(lfm_frontend_workspace_create(&mut frontend_workspace), 0);
        assert_eq!(lfm_resampler_create(24_000, 16_000, &mut resampler), 0);
        assert_eq!(lfm_resampler_workspace_create(&mut resampler_workspace), 0);
        assert_eq!(lfm_conformer_workspace_create(&mut conformer_workspace), 0);
    }

    let pcm = (0..96)
        .map(|index| ((index % 19) as f32 - 9.0) / 9.0)
        .collect::<Vec<_>>();
    let mut sample_count = 0;
    unsafe {
        assert_eq!(
            lfm_resampler_out_length(resampler, pcm.len() as u64, &mut sample_count),
            0
        );
        assert_eq!(
            lfm_resampler_workspace_reserve(resampler, resampler_workspace, pcm.len() as u64),
            0
        );
        assert_eq!(
            lfm_frontend_workspace_reserve(frontend, frontend_workspace, sample_count, 1 | 2,),
            0
        );
    }
    let frames = unsafe { lfm_frontend_seq_len(frontend, sample_count) };
    assert_eq!(frames, 16);
    unsafe {
        assert_eq!(
            lfm_conformer_workspace_reserve(conformer, conformer_workspace, frames),
            0
        );
    }
    let rows = unsafe { lfm_conformer_out_rows(conformer, frames) };
    let width = unsafe { lfm_conformer_out_width(conformer) };
    assert_eq!((rows, width), (2, 8));

    let mut reference_resampled = vec![0.0f32; sample_count as usize];
    let mut reference_mel = vec![0u16; (frames * 8) as usize];
    let mut reference = vec![0u16; (rows * width) as usize];
    let mut span = F32Span {
        data: std::ptr::null(),
        length: 0,
    };
    unsafe {
        assert_eq!(
            lfm_resampler_process(
                resampler,
                resampler_workspace,
                pcm.as_ptr(),
                pcm.len() as u64,
                reference_resampled.as_mut_ptr(),
                reference_resampled.len() as u64,
                &mut span,
            ),
            0
        );
        assert_eq!(span.data, reference_resampled.as_ptr());
        assert_eq!(span.length, sample_count);
        assert_eq!(
            lfm_frontend_forward_bf16_workspace(
                frontend,
                frontend_workspace,
                span.data,
                span.length,
                reference_mel.as_mut_ptr(),
                reference_mel.len() as u64,
            ),
            0
        );
        assert_eq!(
            lfm_conformer_forward(
                conformer,
                conformer_workspace,
                reference_mel.as_ptr(),
                frames,
                reference.as_mut_ptr(),
                reference.len() as u64,
            ),
            0
        );
    }
    assert_eq!(
        unsafe { lfm_conformer_materialized_weight_bytes(conformer) },
        0
    );
    let direct_before = unsafe { lfm_conformer_direct_gemm_calls(conformer) };

    // One extra destination cell lets the oversized command reach the sealed
    // resampler-workspace admission check without ever pointing past storage.
    let mut resampled = vec![f32::NAN; sample_count as usize + 1];
    let mut mel = vec![u16::MAX; (frames * 8) as usize];
    let mut adapted = vec![u16::MAX; (rows * width) as usize];
    let mut values = 0;
    let pass = AudioPass {
        size: std::mem::size_of::<AudioPass>() as u32,
        abi_version: 1,
        resampler,
        resampler_workspace,
        frontend,
        frontend_workspace,
        conformer,
        conformer_workspace,
        pcm: pcm.as_ptr(),
        sample_count: pcm.len() as u64,
        resampled: resampled.as_mut_ptr(),
        resampled_capacity: resampled.len() as u64,
        mel: mel.as_mut_ptr(),
        mel_capacity: mel.len() as u64,
        adapted: adapted.as_mut_ptr(),
        adapted_capacity: adapted.len() as u64,
        out_adapted_values: &mut values,
    };
    let before = unsafe { lfm_engine_audio_encode_passes(engine) };
    for run in 1..=8 {
        assert_eq!(unsafe { lfm_engine_audio_encode(engine, 0, &pass) }, 0);
        assert_eq!(values as usize, reference.len());
        assert_eq!(adapted, reference, "run {run}: typed pass changed parity");
        assert_eq!(
            unsafe { lfm_engine_audio_encode_passes(engine) },
            before + run
        );
        assert_eq!(
            unsafe { lfm_conformer_direct_gemm_calls(conformer) },
            direct_before + run * 16,
            "run {run}: every Conformer linear must execute inside the typed ticket"
        );
        assert_eq!(
            unsafe { lfm_conformer_materialized_weight_bytes(conformer) },
            0
        );
    }

    // A command larger than the admitted high-water mark must fail through the
    // same ticket instead of growing any workspace in steady state.
    let oversized = AudioPass {
        sample_count: pcm.len() as u64 + 1,
        ..pass
    };
    assert_eq!(
        unsafe { lfm_engine_audio_encode(engine, 0, &oversized) },
        -libc::ENOBUFS
    );
    assert_eq!(
        unsafe { lfm_engine_audio_encode_passes(engine) },
        before + 9
    );
    assert_eq!(
        unsafe { lfm_conformer_direct_gemm_calls(conformer) },
        direct_before + 8 * 16
    );
    assert_eq!(unsafe { lfm_engine_audio_encode(engine, 0, &pass) }, 0);
    assert_eq!(
        unsafe { lfm_engine_audio_encode_passes(engine) },
        before + 10
    );
    assert_eq!(
        unsafe { lfm_conformer_direct_gemm_calls(conformer) },
        direct_before + 9 * 16
    );
    assert_eq!(
        adapted, reference,
        "typed pass did not recover after rejection"
    );
    assert_eq!(
        unsafe { lfm_conformer_materialized_weight_bytes(conformer) },
        0
    );

    unsafe {
        assert_eq!(lfm_conformer_workspace_destroy(conformer_workspace), 0);
        assert_eq!(lfm_resampler_workspace_destroy(resampler_workspace), 0);
        assert_eq!(lfm_resampler_destroy(resampler), 0);
        assert_eq!(lfm_frontend_workspace_destroy(frontend_workspace), 0);
        assert_eq!(lfm_frontend_destroy(frontend), 0);
        assert_eq!(lfm_conformer_destroy(conformer), 0);
        lfm_engine_free(engine);
        lfm_weights_close(image);
    }
}
