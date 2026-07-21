use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use liquid_audio::NativeVoiceSampling;

const OK: i32 = 0;
const PERMISSION: i32 = -1;
const IO: i32 = -2;
const FORMAT: i32 = -3;
const NOT_FOUND: i32 = -5;
const INVALID: i32 = -22;
const WEIGHT_ABI: u32 = 1;
const RUNTIME_ABI: u32 = 4;
const MODEL_ABI: u32 = 4;
const PAYLOAD_CONFIG: u32 = 1;
const PAYLOAD_WEIGHT_IMAGE: u32 = 1 << 1;
const PAYLOAD_WEIGHT_INDEX: u32 = 1 << 2;
const PAYLOAD_TOKENIZER: u32 = 1 << 3;
const PAYLOAD_READS_COMPLETE: u32 = 1;
const BF16: u32 = 13;
const F32: u32 = 16;

#[repr(C)]
struct WeightImage {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeModel {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeRuntime {
    _private: [u8; 0],
}

#[repr(C)]
struct RuntimeConfig {
    size: u32,
    abi_version: u32,
    coordination_workers: u32,
    kernel_lanes: u32,
    event_capacity: u32,
    session_capacity: u32,
    reserved0: u32,
    reserved1: u32,
    flags: u64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Default)]
struct RuntimeSnapshot {
    size: u32,
    abi_version: u32,
    runtime_epoch: u64,
    state: u32,
    kernel_lanes: u32,
    live_models: u32,
    live_sessions: u32,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Default)]
struct ModelInfo {
    size: u32,
    abi_version: u32,
    resident_bytes: u64,
    plan_id: u64,
    depth_plan_id: u64,
    hidden: u32,
    ffn: u32,
    layers: u32,
    vocab: u32,
    max_context: u32,
    codebooks: u32,
    capabilities: u32,
    reserved: [u32; 5],
}

#[repr(C)]
#[derive(Default)]
struct ModelMemory {
    size: u32,
    abi_version: u32,
    source_bytes: u64,
    resident_image_bytes: u64,
    directly_bound_bytes: u64,
    derived_immutable_bytes: u64,
    materialized_weight_bytes: u64,
    compatibility_copied_bytes: u64,
    payload_read_calls: u64,
    payload_read_bytes: u64,
    post_publication_read_calls: u64,
    post_publication_read_bytes: u64,
    post_publication_materialization_attempts: u64,
    post_publication_materialization_bytes: u64,
    publication_generation: u64,
    load_ns: u64,
    load_workers: u32,
    load_tasks: u32,
    payload_read_coverage: u32,
    accounting_flags: u32,
    post_readiness_allocation_attempts: u64,
    post_readiness_allocation_bytes: u64,
    reserved: [u64; 2],
}

#[repr(C)]
#[derive(Debug, Default)]
struct TensorView {
    size: u32,
    abi_version: u32,
    name: *const c_char,
    data: *const c_void,
    shape: *const u64,
    offset: u64,
    elements: u64,
    bytes: u64,
    rank: u32,
    dtype: u32,
    shard: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Debug, Default)]
struct LoadStats {
    size: u32,
    abi_version: u32,
    source_bytes: u64,
    resident_bytes: u64,
    task_count: u32,
    worker_count: u32,
}

extern "C" {
    fn lfm_internal_runtime_create_manual_deadlines_for_test(
        config: *const RuntimeConfig,
        out: *mut *mut NativeRuntime,
    ) -> i32;
    fn lfm_runtime_start(runtime: *mut NativeRuntime) -> i32;
    fn lfm_runtime_request_stop(runtime: *mut NativeRuntime);
    fn lfm_runtime_join(runtime: *mut NativeRuntime) -> i32;
    fn lfm_runtime_snapshot(runtime: *const NativeRuntime, out: *mut RuntimeSnapshot) -> i32;
    fn lfm_runtime_destroy(runtime: *mut NativeRuntime) -> i32;
    fn lfm_runtime_model_open(
        runtime: *mut NativeRuntime,
        path: *const c_char,
        out: *mut *mut NativeModel,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_weights_open(
        path: *const c_char,
        out: *mut *mut WeightImage,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn lfm_weights_open_files(
        paths: *const *const c_char,
        count: usize,
        out: *mut *mut WeightImage,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn lfm_weights_open_bundle(
        main_path: *const c_char,
        detokenizer_path: *const c_char,
        out: *mut *mut WeightImage,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn lfm_weights_close(image: *mut WeightImage);
    fn lfm_weights_data(image: *const WeightImage) -> *const c_void;
    fn lfm_weights_resident_bytes(image: *const WeightImage) -> u64;
    fn lfm_weights_count(image: *const WeightImage) -> usize;
    fn lfm_weights_component_count(image: *const WeightImage, component: u32) -> usize;
    fn lfm_weights_load_stats(image: *const WeightImage, out: *mut LoadStats) -> i32;
    fn lfm_weights_at(image: *const WeightImage, index: usize, out: *mut TensorView) -> i32;
    fn lfm_weights_find(
        image: *const WeightImage,
        name: *const c_char,
        out: *mut TensorView,
    ) -> i32;
    fn lfm_weights_find_component(
        image: *const WeightImage,
        component: u32,
        name: *const c_char,
        out: *mut TensorView,
    ) -> i32;
    fn lfm_weights_dtype_name(dtype: u32) -> *const c_char;
    fn lfm_internal_engine_new_manual_deadlines_for_test(workers: i32) -> *mut c_void;
    fn lfm_engine_free(engine: *mut c_void);
    fn lfm_model_open(
        engine: *mut c_void,
        path: *const c_char,
        out: *mut *mut NativeModel,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_model_close(model: *mut NativeModel) -> i32;
    fn lfm_model_info(model: *const NativeModel, out: *mut ModelInfo) -> i32;
    fn lfm_model_memory(model: *const NativeModel, out: *mut ModelMemory) -> i32;
    fn lfm_internal_model_accounting_fault_test(
        source: *const u8,
        loaded: *mut u8,
        rejected: *mut u8,
        bytes: usize,
        out: *mut ModelMemory,
        out_read_status: *mut i32,
        out_weight_status: *mut i32,
        out_policy_status: *mut i32,
    ) -> i32;
    fn lfm_internal_conversation_allocation_seal_test(
        after_compatible: *mut ModelMemory,
        after_rejected: *mut ModelMemory,
        out_growth_status: *mut i32,
        out_capture_rate_status: *mut i32,
        out_playback_rate_status: *mut i32,
    ) -> i32;
    fn lfm_internal_model_source_gate_test(
        path: *const c_char,
        config_status: *mut i32,
        weights_status: *mut i32,
        tokenizer_status: *mut i32,
    ) -> i32;
    fn lfm_bf16_unlift_bits(source_bytes: *const c_void) -> u32;
    fn lfm_internal_weights_open_bundle_benchmark(
        main_path: *const c_char,
        detokenizer_path: *const c_char,
        workers: u32,
        uncached: u32,
        out: *mut *mut WeightImage,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn lfm_internal_weights_open_fault_test(
        path: *const c_char,
        mode: u32,
        scheduled: *mut u32,
        completed: *mut u32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
}

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        std::hint::black_box(std::mem::size_of::<NativeVoiceSampling>());
        kcoro_sys::link_anchor();
        const BASE: &str = "emberharmony-native-safetensors";
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{BASE}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap_or_default();
    }
}

struct Tensor<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [u64],
    data: &'a [u8],
}

fn write_file(path: &Path, tensors: &[Tensor<'_>]) {
    let mut root = serde_json::Map::new();
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset + tensor.data.len();
        root.insert(
            tensor.name.into(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let data = tensors
        .iter()
        .flat_map(|tensor| tensor.data.iter().copied())
        .collect::<Vec<_>>();
    write_raw(path, serde_json::Value::Object(root), &data);
}

fn write_raw(path: &Path, header: serde_json::Value, data: &[u8]) {
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize((header.len() + 7) & !7, b' ');
    let mut file = Vec::with_capacity(8 + header.len() + data.len());
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(data);
    std::fs::write(path, file).unwrap();
}

fn payload_start(path: &Path) -> usize {
    let mut prefix = [0u8; 8];
    std::fs::File::open(path)
        .unwrap()
        .read_exact(&mut prefix)
        .unwrap();
    8 + u64::from_le_bytes(prefix) as usize
}

#[derive(Clone, Debug)]
struct TinyTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
}

fn write_zero_model(path: &Path, tensors: &[TinyTensor]) {
    let mut root = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let width = match tensor.dtype {
            "BF16" | "F16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            dtype => panic!("unsupported synthetic model dtype {dtype}"),
        };
        let bytes = tensor.shape.iter().product::<u64>() as usize * width;
        let begin = data.len();
        data.resize(begin + bytes, 0);
        root.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [begin, begin + bytes],
            }),
        );
    }
    write_raw(path, serde_json::Value::Object(root), &data);
}

fn tiny_model_tensors(layers: usize) -> Vec<TinyTensor> {
    const HIDDEN: u64 = 8;
    const FFN: u64 = 12;
    const VOCAB: u64 = 16;
    const CODEBOOKS: u64 = 2;
    const DEPTH_DIM: u64 = 8;
    const DEPTH_FFN: u64 = 256;
    const DEPTH_VOCAB: u64 = 2049;
    let tensor = |name: String, shape: Vec<u64>| TinyTensor {
        name,
        dtype: "BF16",
        shape,
    };
    let mut tensors = vec![
        tensor("lfm.embed_tokens.weight".into(), vec![VOCAB, HIDDEN]),
        tensor("lfm.embedding_norm.weight".into(), vec![HIDDEN]),
        tensor(
            "audio_embedding.embedding.weight".into(),
            vec![CODEBOOKS * 2049, HIDDEN],
        ),
    ];
    for layer in 0..layers {
        let root = format!("lfm.layers.{layer}.");
        tensors.extend([
            tensor(format!("{root}operator_norm.weight"), vec![HIDDEN]),
            tensor(format!("{root}ffn_norm.weight"), vec![HIDDEN]),
            tensor(format!("{root}feed_forward.w1.weight"), vec![FFN, HIDDEN]),
            tensor(format!("{root}feed_forward.w3.weight"), vec![FFN, HIDDEN]),
            tensor(format!("{root}feed_forward.w2.weight"), vec![HIDDEN, FFN]),
        ]);
    }
    let attention = "lfm.layers.0.self_attn.";
    tensors.extend([
        tensor(format!("{attention}q_proj.weight"), vec![HIDDEN, HIDDEN]),
        tensor(format!("{attention}k_proj.weight"), vec![4, HIDDEN]),
        tensor(format!("{attention}v_proj.weight"), vec![4, HIDDEN]),
        tensor(format!("{attention}out_proj.weight"), vec![HIDDEN, HIDDEN]),
        tensor(format!("{attention}q_layernorm.weight"), vec![4]),
        tensor(format!("{attention}k_layernorm.weight"), vec![4]),
    ]);
    for layer in 1..layers {
        let conv = format!("lfm.layers.{layer}.conv.");
        tensors.extend([
            tensor(format!("{conv}in_proj.weight"), vec![3 * HIDDEN, HIDDEN]),
            tensor(format!("{conv}conv.weight"), vec![HIDDEN, 1, 3]),
            tensor(format!("{conv}out_proj.weight"), vec![HIDDEN, HIDDEN]),
        ]);
    }
    let depth = "depthformer.layers.0.";
    tensors.extend([
        tensor(
            format!("{depth}operator.qkv_proj.weight"),
            vec![16, DEPTH_DIM],
        ),
        tensor(
            format!("{depth}operator.out_proj.weight"),
            vec![DEPTH_DIM, DEPTH_DIM],
        ),
        tensor(
            format!("{depth}operator.bounded_attention.q_layernorm.weight"),
            vec![4],
        ),
        tensor(
            format!("{depth}operator.bounded_attention.k_layernorm.weight"),
            vec![4],
        ),
        tensor(format!("{depth}operator_norm.weight"), vec![DEPTH_DIM]),
        tensor(format!("{depth}ffn_norm.weight"), vec![DEPTH_DIM]),
        tensor(
            format!("{depth}feed_forward.w1.weight"),
            vec![DEPTH_FFN, DEPTH_DIM],
        ),
        tensor(
            format!("{depth}feed_forward.w3.weight"),
            vec![DEPTH_FFN, DEPTH_DIM],
        ),
        tensor(
            format!("{depth}feed_forward.w2.weight"),
            vec![DEPTH_DIM, DEPTH_FFN],
        ),
        tensor(
            "depth_linear.weight".into(),
            vec![CODEBOOKS * DEPTH_DIM, HIDDEN],
        ),
        tensor("depth_linear.bias".into(), vec![CODEBOOKS * DEPTH_DIM]),
    ]);
    for codebook in 0..CODEBOOKS {
        let root = format!("depth_embeddings.{codebook}.");
        tensors.extend([
            tensor(
                format!("{root}embedding.weight"),
                vec![DEPTH_VOCAB, DEPTH_DIM],
            ),
            tensor(format!("{root}embedding_norm.weight"), vec![DEPTH_DIM]),
            tensor(
                format!("{root}to_logits.weight"),
                vec![DEPTH_VOCAB, DEPTH_DIM],
            ),
        ]);
    }
    tensors
}

fn write_tiny_model(temp: &Temp, layers: usize, mutate: impl FnOnce(&mut Vec<TinyTensor>)) {
    let mut tensors = tiny_model_tensors(layers);
    mutate(&mut tensors);
    write_zero_model(&temp.0.join("model.safetensors"), &tensors);
    let types = (0..layers)
        .map(|layer| if layer == 0 { "full_attention" } else { "conv" })
        .collect::<Vec<_>>();
    std::fs::write(
        temp.0.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "codebooks": 2,
            "depthformer": {
                "layers": 1,
                "dim": 8,
                "heads": 2,
                "kv_heads": 1
            },
            "lfm": {
                "vocab_size": 16,
                "hidden_size": 8,
                "num_hidden_layers": layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "norm_eps": 1e-5,
                "max_position_embeddings": 32,
                "conv_L_cache": 3,
                "layer_types": types,
                "block_ff_dim": 12,
                "block_auto_adjust_ff_dim": false
            }
        }))
        .unwrap(),
    )
    .unwrap();
}

fn open_tiny_model(temp: &Temp) -> (*mut c_void, *mut NativeModel, i32, String) {
    let engine = unsafe { lfm_internal_engine_new_manual_deadlines_for_test(2) };
    assert!(!engine.is_null());
    let path = CString::new(temp.0.as_os_str().as_encoded_bytes()).unwrap();
    let mut model = std::ptr::null_mut();
    let mut error = [0i8; 512];
    let status = unsafe {
        lfm_model_open(
            engine,
            path.as_ptr(),
            &mut model,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    (engine, model, status, message)
}

fn assert_tiny_model_rejected(temp: &Temp, message: &str) {
    let (engine, model, status, error) = open_tiny_model(temp);
    assert_ne!(status, 0, "invalid model unexpectedly opened");
    assert!(model.is_null(), "failed model open published a handle");
    assert!(
        error.contains(message),
        "expected error containing {message:?}, got status {status}: {error}"
    );
    unsafe { lfm_engine_free(engine) };
}

fn tiny_model_memory(temp: &Temp) -> ModelMemory {
    let (engine, model, status, message) = open_tiny_model(temp);
    assert_eq!(status, 0, "native model open failed: {message}");
    let mut memory = ModelMemory {
        size: std::mem::size_of::<ModelMemory>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    assert_eq!(unsafe { lfm_model_memory(model, &mut memory) }, 0);
    assert_eq!(unsafe { lfm_model_close(model) }, 0);
    unsafe { lfm_engine_free(engine) };
    memory
}

#[derive(Debug)]
struct Image(*mut WeightImage);

impl Image {
    fn open(path: &Path) -> Result<Self, (i32, String)> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let mut image = std::ptr::null_mut();
        let mut err = [0i8; 512];
        let rc =
            unsafe { lfm_weights_open(path.as_ptr(), &mut image, err.as_mut_ptr(), err.len()) };
        if rc != OK {
            let message = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err((rc, message));
        }
        assert!(!image.is_null());
        Ok(Self(image))
    }

    fn find(&self, name: &str) -> Result<TensorView, i32> {
        let name = CString::new(name).unwrap();
        let mut view = TensorView::default();
        let rc = unsafe { lfm_weights_find(self.0, name.as_ptr(), &mut view) };
        if rc != OK {
            return Err(rc);
        }
        Ok(view)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { lfm_weights_close(self.0) };
    }
}

#[test]
fn native_file_is_one_aligned_image_with_stable_views() {
    let temp = Temp::new();
    let path = temp.0.join("weights.safetensors");
    let bf16 = [0x80, 0x3f, 0x00, 0x40];
    let f32 = 1.5f32.to_le_bytes();
    write_file(
        &path,
        &[
            Tensor {
                name: "backbone.weight",
                dtype: "BF16",
                shape: &[2],
                data: &bf16,
            },
            Tensor {
                name: "head.scale",
                dtype: "F32",
                shape: &[],
                data: &f32,
            },
        ],
    );

    let image = Image::open(&path).unwrap();
    assert_eq!(unsafe { lfm_weights_count(image.0) }, 2);
    let base = unsafe { lfm_weights_data(image.0) } as usize;
    assert_ne!(base, 0);
    assert_eq!(base & 63, 0);
    assert!(
        unsafe { lfm_weights_resident_bytes(image.0) } >= std::fs::metadata(path).unwrap().len()
    );

    let view = image.find("backbone.weight").unwrap();
    assert_eq!(view.size as usize, std::mem::size_of::<TensorView>());
    assert_eq!(view.abi_version, WEIGHT_ABI);
    assert_eq!(view.dtype, BF16);
    assert_eq!(view.rank, 1);
    assert_eq!(view.elements, 2);
    assert_eq!(view.bytes, 4);
    assert_eq!(unsafe { std::slice::from_raw_parts(view.shape, 1) }, &[2]);
    assert_eq!(view.data as usize, base + view.offset as usize);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), 4) },
        bf16
    );
    assert_eq!(
        unsafe { CStr::from_ptr(lfm_weights_dtype_name(view.dtype)) }.to_bytes(),
        b"BF16"
    );
    assert_eq!(image.find("missing.weight").unwrap_err(), NOT_FOUND);
}

#[test]
#[cfg(unix)]
fn published_image_is_process_read_only() {
    let temp = Temp::new();
    let path = temp.0.join("weights.safetensors");
    write_file(
        &path,
        &[Tensor {
            name: "weight",
            dtype: "BF16",
            shape: &[2],
            data: &[0x80, 0x3f, 0x00, 0x40],
        }],
    );

    let image = Image::open(&path).unwrap();
    let base = unsafe { lfm_weights_data(image.0) };
    assert!(!base.is_null());

    // A child can probe the VM protection without taking down the test runner.
    // It performs no allocator or lock work after fork: the first write must be
    // rejected by the kernel because publication sealed the complete image.
    let child = unsafe { libc::fork() };
    assert!(child >= 0, "fork failed");
    if child == 0 {
        unsafe {
            libc::memset(base.cast_mut(), 0xa5, 1);
            libc::_exit(0);
        }
    }

    let mut status = 0;
    assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
    assert!(
        libc::WIFSIGNALED(status),
        "write unexpectedly succeeded: {status}"
    );
    let signal = libc::WTERMSIG(status);
    assert!(
        signal == libc::SIGSEGV || signal == libc::SIGBUS,
        "read-only write terminated with signal {signal}"
    );
}

#[test]
fn bundle_scopes_duplicate_names_and_uses_one_image() {
    const MAIN: u32 = 1;
    const DETOKENIZER: u32 = 2;

    let temp = Temp::new();
    let main = temp.0.join("model.safetensors");
    let detokenizer = temp.0.join("audio_detokenizer.safetensors");
    let main_data = 1.25f32.to_le_bytes();
    let detokenizer_data = (-3.5f32).to_le_bytes();
    write_file(
        &main,
        &[Tensor {
            name: "shared.weight",
            dtype: "F32",
            shape: &[1],
            data: &main_data,
        }],
    );
    write_file(
        &detokenizer,
        &[Tensor {
            name: "shared.weight",
            dtype: "F32",
            shape: &[1],
            data: &detokenizer_data,
        }],
    );

    let main_c = CString::new(main.as_os_str().as_encoded_bytes()).unwrap();
    let detokenizer_c = CString::new(detokenizer.as_os_str().as_encoded_bytes()).unwrap();
    let mut raw = std::ptr::null_mut();
    let mut err = [0i8; 512];
    assert_eq!(
        unsafe {
            lfm_weights_open_bundle(
                main_c.as_ptr(),
                detokenizer_c.as_ptr(),
                &mut raw,
                err.as_mut_ptr(),
                err.len(),
            )
        },
        OK,
        "{}",
        unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()
    );
    let image = Image(raw);
    assert_eq!(unsafe { lfm_weights_count(image.0) }, 1);
    assert_eq!(unsafe { lfm_weights_component_count(image.0, MAIN) }, 1);
    assert_eq!(
        unsafe { lfm_weights_component_count(image.0, DETOKENIZER) },
        1
    );

    let name = CString::new("shared.weight").unwrap();
    let mut main_view = TensorView::default();
    let mut detokenizer_view = TensorView::default();
    assert_eq!(
        unsafe { lfm_weights_find(image.0, name.as_ptr(), &mut main_view) },
        OK
    );
    assert_eq!(
        unsafe {
            lfm_weights_find_component(image.0, DETOKENIZER, name.as_ptr(), &mut detokenizer_view)
        },
        OK
    );
    assert_ne!(main_view.data, detokenizer_view.data);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(main_view.data.cast::<u8>(), 4) },
        main_data
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(detokenizer_view.data.cast::<u8>(), 4) },
        detokenizer_data
    );
    let base = unsafe { lfm_weights_data(image.0) } as usize;
    let resident = unsafe { lfm_weights_resident_bytes(image.0) } as usize;
    assert!((main_view.data as usize) >= base);
    assert!((detokenizer_view.data as usize) < base + resident);

    let open_benchmark = |workers| {
        let mut raw = std::ptr::null_mut();
        let mut error = [0i8; 512];
        assert_eq!(
            unsafe {
                lfm_internal_weights_open_bundle_benchmark(
                    main_c.as_ptr(),
                    detokenizer_c.as_ptr(),
                    workers,
                    0,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            OK,
            "{}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
        Image(raw)
    };
    let serial = open_benchmark(1);
    let parallel = open_benchmark(4);
    let mut serial_stats = LoadStats::default();
    let mut parallel_stats = LoadStats::default();
    assert_eq!(
        unsafe { lfm_weights_load_stats(serial.0, &mut serial_stats) },
        OK
    );
    assert_eq!(
        unsafe { lfm_weights_load_stats(parallel.0, &mut parallel_stats) },
        OK
    );
    assert_eq!(serial_stats.worker_count, 1);
    assert_eq!(parallel_stats.worker_count, 2);
    assert_eq!(serial_stats.task_count, 2);
    assert_eq!(parallel_stats.task_count, 2);
    let serial_bytes = unsafe {
        std::slice::from_raw_parts(
            lfm_weights_data(serial.0).cast::<u8>(),
            lfm_weights_resident_bytes(serial.0) as usize,
        )
    };
    let parallel_bytes = unsafe {
        std::slice::from_raw_parts(
            lfm_weights_data(parallel.0).cast::<u8>(),
            lfm_weights_resident_bytes(parallel.0) as usize,
        )
    };
    assert_eq!(serial_bytes, parallel_bytes);
}

#[test]
fn bf16_unlift_is_bit_exact_from_unaligned_checkpoint_bytes() {
    // +0, -0, the smallest subnormal, +/-infinity, and two NaN payloads.
    let words = [0x0000u16, 0x8000, 0x0001, 0x7f80, 0xff80, 0x7f81, 0xffff];
    let mut bytes = vec![0x5au8];
    bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));

    for (index, word) in words.into_iter().enumerate() {
        let source = unsafe { bytes.as_ptr().add(1 + index * 2) };
        assert_ne!((source as usize) & 1, 0);
        assert_eq!(
            unsafe { lfm_bf16_unlift_bits(source.cast()) },
            u32::from(word) << 16
        );
    }
}

#[test]
fn parallel_read_is_byte_exact_across_chunks_and_zeroes_only_padding() {
    const CHUNK: usize = 8 * 1024 * 1024;

    let temp = Temp::new();
    let first = temp.0.join("model-00001-of-00002.safetensors");
    let second = temp.0.join("model-00002-of-00002.safetensors");
    let a = (0..2 * CHUNK + 37)
        .map(|i| (i.wrapping_mul(31).wrapping_add(7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let b = (0..CHUNK + 113)
        .map(|i| (i.wrapping_mul(17).wrapping_add(19) & 0xff) as u8)
        .collect::<Vec<_>>();
    write_file(
        &first,
        &[Tensor {
            name: "model.a",
            dtype: "U8",
            shape: &[a.len() as u64],
            data: &a,
        }],
    );
    write_file(
        &second,
        &[Tensor {
            name: "model.b",
            dtype: "U8",
            shape: &[b.len() as u64],
            data: &b,
        }],
    );

    let first_file = std::fs::read(&first).unwrap();
    let second_file = std::fs::read(&second).unwrap();
    let image = Image::open(&temp.0).unwrap();
    let base = unsafe { lfm_weights_data(image.0) }.cast::<u8>();
    let resident = unsafe { lfm_weights_resident_bytes(image.0) } as usize;
    let bytes = unsafe { std::slice::from_raw_parts(base, resident) };
    let a_view = image.find("model.a").unwrap();
    let b_view = image.find("model.b").unwrap();
    let mut stats = LoadStats::default();
    assert_eq!(unsafe { lfm_weights_load_stats(image.0, &mut stats) }, OK);
    assert_eq!(stats.size as usize, std::mem::size_of::<LoadStats>());
    assert_eq!(stats.abi_version, WEIGHT_ABI);
    assert_eq!(
        stats.source_bytes,
        (first_file.len() + second_file.len()) as u64
    );
    assert_eq!(stats.resident_bytes, resident as u64);
    let tasks = (first_file.len() + CHUNK - 1) / CHUNK + (second_file.len() + CHUNK - 1) / CHUNK;
    assert_eq!(stats.task_count, tasks as u32);
    assert_eq!(stats.worker_count, tasks.min(4) as u32);

    assert!(
        unsafe { std::slice::from_raw_parts(a_view.data.cast::<u8>(), a.len()) } == a.as_slice(),
        "first payload changed across positioned-read chunks"
    );
    assert!(
        unsafe { std::slice::from_raw_parts(b_view.data.cast::<u8>(), b.len()) } == b.as_slice(),
        "second payload changed across positioned-read chunks"
    );
    assert!(
        &bytes[..first_file.len()] == first_file.as_slice(),
        "first complete source changed in the resident image"
    );

    let second_base = b_view.offset as usize - payload_start(&second);
    assert_eq!(second_base & 63, 0);
    assert!(bytes[first_file.len()..second_base]
        .iter()
        .all(|byte| *byte == 0));
    assert!(
        &bytes[second_base..second_base + second_file.len()] == second_file.as_slice(),
        "second complete source changed in the resident image"
    );
    assert!(bytes[second_base + second_file.len()..]
        .iter()
        .all(|byte| *byte == 0));
}

#[test]
fn concurrent_opens_publish_independent_complete_images() {
    let temp = Temp::new();
    let path = temp.0.join("model.safetensors");
    let payload = (0usize..1024 * 1024 + 19)
        .map(|index| (index.wrapping_mul(29).wrapping_add(11) & 0xff) as u8)
        .collect::<Vec<_>>();
    write_file(
        &path,
        &[Tensor {
            name: "weight",
            dtype: "U8",
            shape: &[payload.len() as u64],
            data: &payload,
        }],
    );

    let start = Arc::new(Barrier::new(8));
    let opened = Arc::new(Barrier::new(8));
    let workers = (0..8)
        .map(|_| {
            let start = Arc::clone(&start);
            let opened = Arc::clone(&opened);
            let path = path.clone();
            let expected = payload.clone();
            std::thread::spawn(move || {
                start.wait();
                let image = Image::open(&path).expect("concurrent image open");
                let view = image.find("weight").expect("concurrent tensor view");
                let actual = unsafe {
                    std::slice::from_raw_parts(view.data.cast::<u8>(), view.bytes as usize)
                };
                assert_eq!(actual, expected);
                let base = unsafe { lfm_weights_data(image.0) as usize };
                opened.wait();
                base
            })
        })
        .collect::<Vec<_>>();
    let bases = workers
        .into_iter()
        .map(|worker| worker.join().expect("concurrent loader worker"))
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(bases.len(), 8, "each open must own its own final image");
}

#[test]
fn changed_source_and_read_failure_join_the_complete_read_team() {
    const CHUNK: usize = 8 * 1024 * 1024;

    let temp = Temp::new();
    let failed = temp.0.join("failed.safetensors");
    let payload = vec![0x5au8; CHUNK + 1];
    write_file(
        &failed,
        &[Tensor {
            name: "weight",
            dtype: "U8",
            shape: &[payload.len() as u64],
            data: &payload,
        }],
    );

    let invoke = |path: &Path, mode| {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let mut scheduled = 0;
        let mut completed = 0;
        let mut error = [0i8; 512];
        let status = unsafe {
            lfm_internal_weights_open_fault_test(
                path.as_ptr(),
                mode,
                &mut scheduled,
                &mut completed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        (
            status,
            scheduled,
            completed,
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        )
    };

    let (status, scheduled, completed, message) = invoke(&failed, 1);
    assert_eq!(status, IO);
    assert_eq!(scheduled, 2, "fixture must cross one 8 MiB boundary");
    assert_eq!(
        completed, scheduled,
        "loader returned before every read task terminated"
    );
    assert!(message.contains("injected positioned-read failure"));

    let changed = temp.0.join("changed.safetensors");
    write_file(
        &changed,
        &[Tensor {
            name: "weight",
            dtype: "U8",
            shape: &[32],
            data: &[0x33; 32],
        }],
    );
    let bytes = std::fs::metadata(&changed).unwrap().len();
    let (status, scheduled, completed, message) = invoke(&changed, 2);
    assert_eq!(status, IO);
    assert_eq!(
        completed, scheduled,
        "source verification ran before the read team joined"
    );
    assert_eq!(std::fs::metadata(&changed).unwrap().len(), bytes + 1);
    assert!(message.contains("file changed while loading"));
}

#[test]
fn truncated_sources_and_shape_arithmetic_overflow_fail_closed() {
    let temp = Temp::new();
    let truncated = temp.0.join("truncated.safetensors");
    write_file(
        &truncated,
        &[Tensor {
            name: "weight",
            dtype: "BF16",
            shape: &[2],
            data: &[0, 0, 0, 0],
        }],
    );
    let bytes = std::fs::metadata(&truncated).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(bytes - 1)
        .unwrap();
    assert_eq!(
        Image::open(&truncated)
            .err()
            .expect("truncated source accepted")
            .0,
        FORMAT
    );

    let overflow = temp.0.join("overflow.safetensors");
    write_raw(
        &overflow,
        serde_json::json!({
            "weight": {
                "dtype": "BF16",
                "shape": [u64::MAX, 2],
                "data_offsets": [0, 0]
            }
        }),
        &[],
    );
    let (status, message) = Image::open(&overflow)
        .err()
        .expect("overflowing tensor shape accepted");
    assert_eq!(status, FORMAT);
    assert!(message.contains("overflow"), "unexpected error: {message}");
}

#[test]
fn model_schema_rejects_wrong_dtype_with_the_same_element_count() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |tensors| {
        tensors
            .iter_mut()
            .find(|tensor| tensor.name == "lfm.layers.0.operator_norm.weight")
            .expect("operator norm fixture")
            .dtype = "F16";
    });
    assert_tiny_model_rejected(
        &temp,
        "model tensor 'lfm.layers.0.operator_norm.weight' has the wrong dtype or rank",
    );
}

#[test]
fn model_schema_rejects_swapped_dimensions_with_the_same_element_count() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |tensors| {
        tensors
            .iter_mut()
            .find(|tensor| tensor.name == "lfm.layers.0.feed_forward.w1.weight")
            .expect("FFN w1 fixture")
            .shape = vec![8, 12];
    });
    assert_tiny_model_rejected(
        &temp,
        "model tensor 'lfm.layers.0.feed_forward.w1.weight' has the wrong shape",
    );
}

#[test]
fn model_schema_rejects_a_missing_middle_layer() {
    let temp = Temp::new();
    write_tiny_model(&temp, 3, |tensors| {
        let index = tensors
            .iter()
            .position(|tensor| tensor.name == "lfm.layers.1.operator_norm.weight")
            .expect("middle-layer norm fixture");
        tensors.remove(index);
    });
    assert_tiny_model_rejected(
        &temp,
        "missing model tensor 'lfm.layers.1.operator_norm.weight'",
    );
}

#[test]
fn model_schema_rejects_short_or_extra_layer_type_entries() {
    for types in [
        serde_json::json!(["full_attention"]),
        serde_json::json!(["full_attention", "conv", "conv"]),
    ] {
        let temp = Temp::new();
        write_tiny_model(&temp, 2, |_| {});
        let path = temp.0.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["lfm"]["layer_types"] = types;
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        assert_tiny_model_rejected(
            &temp,
            "lfm.layer_types length does not match num_hidden_layers",
        );
    }
}

#[test]
fn model_config_rejects_adjusted_ffn_rounding_overflow() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |_| {});
    let path = temp.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["lfm"]["block_ff_dim"] = serde_json::json!(i64::MAX);
    config["lfm"]["block_auto_adjust_ff_dim"] = serde_json::json!(true);
    config["lfm"]["block_multiple_of"] = serde_json::json!(i64::MAX);
    config["lfm"]["block_ffn_dim_multiplier"] = serde_json::json!(2.0);
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();

    assert_tiny_model_rejected(&temp, "adjusted FFN rounding overflows size_t");
}

#[test]
fn model_schema_rejects_audio_vocabulary_codebook_mismatch() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |tensors| {
        tensors
            .iter_mut()
            .find(|tensor| tensor.name == "audio_embedding.embedding.weight")
            .expect("audio embedding fixture")
            .shape = vec![2 * 2049 - 1, 8];
    });
    assert_tiny_model_rejected(
        &temp,
        "audio embedding vocabulary does not match configured codebooks",
    );
}

#[test]
fn runtime_rejects_incomplete_voice_model_without_retaining_a_child() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |_| {});
    let config = RuntimeConfig {
        size: std::mem::size_of::<RuntimeConfig>() as u32,
        abi_version: RUNTIME_ABI,
        coordination_workers: 1,
        kernel_lanes: 2,
        event_capacity: 2,
        session_capacity: 1,
        reserved0: 0,
        reserved1: 0,
        flags: 0,
        reserved: [0; 4],
    };
    let mut runtime = std::ptr::null_mut();
    assert_eq!(
        unsafe { lfm_internal_runtime_create_manual_deadlines_for_test(&config, &mut runtime) },
        0
    );
    assert_eq!(unsafe { lfm_runtime_start(runtime) }, 0);
    let path = CString::new(temp.0.as_os_str().as_encoded_bytes()).unwrap();
    let mut model = std::ptr::null_mut();
    let mut error = [0i8; 512];
    assert_eq!(
        unsafe {
            lfm_runtime_model_open(
                runtime,
                path.as_ptr(),
                &mut model,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        INVALID
    );
    assert!(model.is_null());
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    assert!(
        message.contains("complete native LFM2 voice model"),
        "unexpected lifecycle rejection: {message}"
    );
    let mut snapshot = RuntimeSnapshot {
        size: std::mem::size_of::<RuntimeSnapshot>() as u32,
        abi_version: RUNTIME_ABI,
        ..Default::default()
    };
    assert_eq!(unsafe { lfm_runtime_snapshot(runtime, &mut snapshot) }, 0);
    assert_eq!(snapshot.live_models, 0);
    unsafe { lfm_runtime_request_stop(runtime) };
    assert_eq!(unsafe { lfm_runtime_join(runtime) }, 0);
    assert_eq!(unsafe { lfm_runtime_destroy(runtime) }, 0);
}

#[test]
fn directly_bound_accounting_excludes_unused_checkpoint_tensors() {
    let baseline = Temp::new();
    write_tiny_model(&baseline, 2, |_| {});
    let baseline_memory = tiny_model_memory(&baseline);

    let extra = Temp::new();
    write_tiny_model(&extra, 2, |tensors| {
        tensors.push(TinyTensor {
            name: "unused.audit.weight".into(),
            dtype: "BF16",
            shape: vec![1024],
        });
    });
    let extra_memory = tiny_model_memory(&extra);

    assert!(extra_memory.source_bytes > baseline_memory.source_bytes);
    assert!(extra_memory.resident_image_bytes > baseline_memory.resident_image_bytes);
    assert_eq!(
        extra_memory.directly_bound_bytes, baseline_memory.directly_bound_bytes,
        "unused checkpoint tensors must not masquerade as schema-bound weights"
    );
}

#[test]
fn model_accounting_counts_before_and_rejects_after_publication() {
    const BYTES: usize = 7;
    let source = [3u8, 1, 4, 1, 5, 9, 2];
    let mut loaded = [0u8; BYTES];
    let mut rejected = [0xa5u8; BYTES];
    let untouched = rejected;
    let mut memory = ModelMemory {
        size: std::mem::size_of::<ModelMemory>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    let mut read = 0;
    let mut weight = 0;
    let mut policy = 0;
    assert_eq!(
        unsafe {
            lfm_internal_model_accounting_fault_test(
                source.as_ptr(),
                loaded.as_mut_ptr(),
                rejected.as_mut_ptr(),
                BYTES,
                &mut memory,
                &mut read,
                &mut weight,
                &mut policy,
            )
        },
        0
    );

    assert_eq!(loaded, source, "prepublication recorder skipped its copy");
    assert_eq!(
        rejected, untouched,
        "post-publication recorder touched the destination before rejecting"
    );
    assert_eq!(read, PERMISSION);
    assert_eq!(weight, PERMISSION);
    assert_eq!(
        policy, INVALID,
        "production weight-zero policy accepted a copy"
    );
    assert_eq!(memory.publication_generation, 1);
    assert_eq!(memory.payload_read_calls, 1);
    assert_eq!(memory.payload_read_bytes, BYTES as u64);
    assert_eq!(memory.post_publication_read_calls, 1);
    assert_eq!(memory.post_publication_read_bytes, BYTES as u64);
    assert_eq!(memory.materialized_weight_bytes, BYTES as u64);
    assert_eq!(memory.compatibility_copied_bytes, BYTES as u64);
    assert_eq!(memory.post_publication_materialization_attempts, 1);
    assert_eq!(memory.post_publication_materialization_bytes, BYTES as u64);
    assert_eq!(memory.post_readiness_allocation_attempts, 0);
    assert_eq!(memory.post_readiness_allocation_bytes, 0);
    assert_eq!(memory.payload_read_coverage, PAYLOAD_CONFIG);
    assert_eq!(
        memory.accounting_flags & PAYLOAD_READS_COMPLETE,
        PAYLOAD_READS_COMPLETE
    );
}

#[test]
fn conversation_readiness_seal_rejects_numerical_reallocation_before_mutation() {
    let mut compatible = ModelMemory {
        size: std::mem::size_of::<ModelMemory>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    let mut rejected = ModelMemory {
        size: std::mem::size_of::<ModelMemory>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    let mut growth = 0;
    let mut capture = 0;
    let mut playback = 0;
    assert_eq!(
        unsafe {
            lfm_internal_conversation_allocation_seal_test(
                &mut compatible,
                &mut rejected,
                &mut growth,
                &mut capture,
                &mut playback,
            )
        },
        0
    );

    assert_eq!(compatible.post_readiness_allocation_attempts, 0);
    assert_eq!(compatible.post_readiness_allocation_bytes, 0);
    assert_eq!(growth, PERMISSION, "capture high-water growth was admitted");
    assert_eq!(
        capture, PERMISSION,
        "capture-rate plan replacement was admitted"
    );
    assert_eq!(
        playback, PERMISSION,
        "playback-rate plan replacement was admitted"
    );
    assert_eq!(rejected.post_readiness_allocation_attempts, 3);
    assert_eq!(
        rejected.post_readiness_allocation_bytes,
        46_080,
        "logical bytes must cover 6,400-frame growth, 3,200-frame capture-plan replacement, and one 24 kHz detokenizer playback plane"
    );
}

#[test]
fn every_native_model_source_rejects_before_path_io_after_publication() {
    let temp = Temp::new();
    let path = temp.0.join("publication-must-not-touch-this-path");
    assert!(!path.exists());
    let sentinel = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let mut config = 0;
    let mut weights = 0;
    let mut tokenizer = 0;
    assert_eq!(
        unsafe {
            lfm_internal_model_source_gate_test(
                sentinel.as_ptr(),
                &mut config,
                &mut weights,
                &mut tokenizer,
            )
        },
        0
    );
    assert_eq!(config, PERMISSION);
    assert_eq!(weights, PERMISSION);
    assert_eq!(tokenizer, PERMISSION);
    assert!(!path.exists());
}

#[test]
fn opaque_native_model_reports_single_image_accounting() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |_| {});
    let (engine, model, status, message) = open_tiny_model(&temp);
    assert_eq!(status, 0, "native model open failed: {message}");
    assert!(!model.is_null());
    let mut info = ModelInfo {
        size: std::mem::size_of::<ModelInfo>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    assert_eq!(unsafe { lfm_model_info(model, &mut info) }, 0);
    assert_eq!(
        (info.hidden, info.ffn, info.layers, info.vocab),
        (8, 12, 2, 16)
    );
    assert_eq!(info.max_context, 32);
    assert_eq!(info.codebooks, 2);
    assert!(info.plan_id > 0);
    assert!(info.depth_plan_id > 0);
    assert_eq!(info.capabilities & 1, 1);
    assert!(info.resident_bytes > 0);
    let mut memory = ModelMemory {
        size: std::mem::size_of::<ModelMemory>() as u32,
        abi_version: MODEL_ABI,
        ..Default::default()
    };
    assert_eq!(unsafe { lfm_model_memory(model, &mut memory) }, 0);
    assert!(memory.source_bytes > 0);
    assert_eq!(memory.resident_image_bytes, info.resident_bytes);
    assert!(memory.directly_bound_bytes > 0);
    assert_eq!(memory.materialized_weight_bytes, 0);
    assert_eq!(memory.compatibility_copied_bytes, 0);
    assert_eq!(memory.publication_generation, 1);
    assert_eq!(
        memory.payload_read_calls,
        u64::from(memory.load_tasks) + 1,
        "config read plus resident-image read tasks must be accounted"
    );
    let config_bytes = std::fs::metadata(temp.0.join("config.json")).unwrap().len();
    assert_eq!(
        memory.payload_read_bytes,
        memory.source_bytes + config_bytes
    );
    assert_eq!(memory.post_publication_read_calls, 0);
    assert_eq!(memory.post_publication_read_bytes, 0);
    assert_eq!(memory.post_publication_materialization_attempts, 0);
    assert_eq!(memory.post_publication_materialization_bytes, 0);
    assert_eq!(memory.post_readiness_allocation_attempts, 0);
    assert_eq!(memory.post_readiness_allocation_bytes, 0);
    assert_eq!(
        memory.payload_read_coverage,
        PAYLOAD_CONFIG | PAYLOAD_WEIGHT_IMAGE
    );
    assert_eq!(
        memory.accounting_flags & PAYLOAD_READS_COMPLETE,
        PAYLOAD_READS_COMPLETE,
        "all applicable source implementations are installed on this owner"
    );
    assert!(memory.load_ns > 0);
    assert!(memory.load_workers > 0);
    assert!(memory.load_tasks > 0);

    assert_eq!(unsafe { lfm_model_close(model) }, 0);
    unsafe { lfm_engine_free(engine) };
}

#[test]
fn model_owned_index_read_is_counted_without_posthoc_loader_summation() {
    let temp = Temp::new();
    write_tiny_model(&temp, 2, |_| {});
    let shard = "model-00001-of-00001.safetensors";
    std::fs::rename(temp.0.join("model.safetensors"), temp.0.join(shard)).unwrap();
    let map = tiny_model_tensors(2)
        .into_iter()
        .map(|tensor| (tensor.name, serde_json::Value::String(shard.into())))
        .collect::<serde_json::Map<_, _>>();
    let index = serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap();
    std::fs::write(temp.0.join("model.safetensors.index.json"), &index).unwrap();

    let memory = tiny_model_memory(&temp);
    let config = std::fs::metadata(temp.0.join("config.json")).unwrap().len();
    assert_eq!(
        memory.payload_read_calls,
        u64::from(memory.load_tasks) + 2,
        "config and index are distinct owner-recorded reads"
    );
    assert_eq!(
        memory.payload_read_bytes,
        memory.source_bytes + config + index.len() as u64
    );
    assert_eq!(
        memory.payload_read_coverage,
        PAYLOAD_CONFIG | PAYLOAD_WEIGHT_IMAGE | PAYLOAD_WEIGHT_INDEX
    );
    assert_eq!(
        memory.accounting_flags & PAYLOAD_READS_COMPLETE,
        PAYLOAD_READS_COMPLETE
    );
    assert_eq!(memory.post_publication_read_calls, 0);
    assert_eq!(memory.post_publication_read_bytes, 0);
}

#[test]
#[ignore = "requires LFM_MODEL_DIR and its released audio_detokenizer component"]
fn complete_runtime_model_reports_lifecycle_only_memory_accounting() {
    let dir = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .expect("LFM_MODEL_DIR must name the complete LFM2-Audio checkpoint"),
    );
    let model = liquid_audio::NativeVoiceModel::open(&dir).expect("complete native voice model");
    let first = model.memory().expect("native lifecycle memory report");
    let second = model
        .memory()
        .expect("repeat native lifecycle memory report");
    assert_eq!(
        first, second,
        "immutable model accounting changed after open"
    );
    assert!(first.source_bytes > 0);
    assert!(first.resident_image_bytes >= first.source_bytes);
    assert!(first.directly_bound_bytes > 0);
    assert_eq!(first.materialized_weight_bytes, 0);
    assert_eq!(first.compatibility_copied_bytes, 0);
    assert!(first.payload_read_calls > 0);
    assert!(
        first.payload_read_calls >= u64::from(first.load_tasks) + 2,
        "voice accounting omitted its config or tokenizer read"
    );
    assert!(
        first.payload_read_bytes > first.source_bytes,
        "voice accounting omitted non-image payload bytes"
    );
    assert_eq!(
        first.payload_read_coverage & (PAYLOAD_CONFIG | PAYLOAD_WEIGHT_IMAGE | PAYLOAD_TOKENIZER),
        PAYLOAD_CONFIG | PAYLOAD_WEIGHT_IMAGE | PAYLOAD_TOKENIZER,
        "complete voice model did not account an actual config, shard, or tokenizer read"
    );
    assert_eq!(first.publication_generation, 1);
    assert_eq!(first.post_publication_read_calls, 0);
    assert_eq!(first.post_publication_read_bytes, 0);
    assert_eq!(first.post_publication_materialization_attempts, 0);
    assert_eq!(first.post_publication_materialization_bytes, 0);
    assert!(
        first.payload_read_accounting_complete,
        "real-checkpoint read gate refuses incomplete source coverage"
    );
    assert!(first.load_ns > 0);
    assert!((1..=4).contains(&first.load_workers));
    assert!(first.load_tasks > 0);
}

#[test]
fn checkpoint_index_loads_each_shard_into_the_same_image() {
    let temp = Temp::new();
    let first = temp.0.join("model-00001-of-00002.safetensors");
    let second = temp.0.join("model-00002-of-00002.safetensors");
    let a = 3.0f32.to_le_bytes();
    let b = 7.0f32.to_le_bytes();
    write_file(
        &first,
        &[Tensor {
            name: "model.a",
            dtype: "F32",
            shape: &[1],
            data: &a,
        }],
    );
    write_file(
        &second,
        &[Tensor {
            name: "model.b",
            dtype: "F32",
            shape: &[1],
            data: &b,
        }],
    );
    std::fs::write(
        temp.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 8},
            "weight_map": {
                "model.a": "model-00001-of-00002.safetensors",
                "model.b": "model-00002-of-00002.safetensors",
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let image = Image::open(&temp.0).unwrap();
    let first_view = image.find("model.a").unwrap();
    let second_view = image.find("model.b").unwrap();
    assert_eq!(first_view.dtype, F32);
    assert_eq!(second_view.dtype, F32);
    assert_ne!(first_view.shard, second_view.shard);
    assert_eq!(unsafe { *first_view.data.cast::<f32>() }, 3.0);
    assert_eq!(unsafe { *second_view.data.cast::<f32>() }, 7.0);

    let resident = unsafe {
        std::slice::from_raw_parts(
            lfm_weights_data(image.0).cast::<u8>(),
            lfm_weights_resident_bytes(image.0) as usize,
        )
    }
    .to_vec();
    std::fs::write(
        temp.0.join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":8},"weight_map":{"model.b":"model-00002-of-00002.safetensors","model.a":"model-00001-of-00002.safetensors"}}"#,
    )
    .unwrap();
    let reordered = Image::open(&temp.0).unwrap();
    let reordered_bytes = unsafe {
        std::slice::from_raw_parts(
            lfm_weights_data(reordered.0).cast::<u8>(),
            lfm_weights_resident_bytes(reordered.0) as usize,
        )
    };
    assert_eq!(
        reordered_bytes, resident,
        "index key order must not change the resident image digest"
    );
}

#[test]
fn malformed_spans_and_duplicate_shard_names_are_rejected() {
    let temp = Temp::new();
    let malformed = temp.0.join("malformed.safetensors");
    write_raw(
        &malformed,
        serde_json::json!({
            "bad": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}
        }),
        &[0; 4],
    );
    let (status, message) = match Image::open(&malformed) {
        Ok(_) => panic!("malformed safetensors unexpectedly loaded"),
        Err(error) => error,
    };
    assert_eq!(status, FORMAT);
    assert!(message.contains("does not match dtype and shape"));

    let first = temp.0.join("one.safetensors");
    let second = temp.0.join("two.safetensors");
    write_file(
        &first,
        &[Tensor {
            name: "duplicate",
            dtype: "F32",
            shape: &[1],
            data: &1.0f32.to_le_bytes(),
        }],
    );
    write_file(
        &second,
        &[Tensor {
            name: "duplicate",
            dtype: "F32",
            shape: &[1],
            data: &2.0f32.to_le_bytes(),
        }],
    );
    let paths = [
        CString::new(first.as_os_str().as_encoded_bytes()).unwrap(),
        CString::new(second.as_os_str().as_encoded_bytes()).unwrap(),
    ];
    let raw = [paths[0].as_ptr(), paths[1].as_ptr()];
    let mut image = std::ptr::null_mut();
    let mut err = [0i8; 512];
    let rc = unsafe {
        lfm_weights_open_files(
            raw.as_ptr(),
            raw.len(),
            &mut image,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    assert_eq!(rc, FORMAT);
    assert!(image.is_null());
    assert!(unsafe { CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .contains("duplicate tensor name"));
}

#[test]
fn indexed_view_iteration_uses_the_public_descriptor_surface() {
    let temp = Temp::new();
    let path = temp.0.join("weights.safetensors");
    write_file(
        &path,
        &[Tensor {
            name: "only",
            dtype: "F32",
            shape: &[1],
            data: &0.25f32.to_le_bytes(),
        }],
    );
    let image = Image::open(&path).unwrap();
    let mut view = TensorView::default();
    assert_eq!(unsafe { lfm_weights_at(image.0, 0, &mut view) }, OK);
    assert_eq!(unsafe { CStr::from_ptr(view.name) }.to_bytes(), b"only");
    assert_eq!(unsafe { lfm_weights_at(image.0, 1, &mut view) }, NOT_FOUND);
}
