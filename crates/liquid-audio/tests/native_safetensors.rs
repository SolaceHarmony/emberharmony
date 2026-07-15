use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use liquid_audio::weights::{ResidentWeights, WeightDType};

const OK: i32 = 0;
const FORMAT: i32 = -3;
const NOT_FOUND: i32 = -5;
const WEIGHT_ABI: u32 = 1;
const MODEL_ABI: u32 = 2;
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

extern "C" {
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
    fn lfm_weights_close(image: *mut WeightImage);
    fn lfm_weights_data(image: *const WeightImage) -> *const c_void;
    fn lfm_weights_resident_bytes(image: *const WeightImage) -> u64;
    fn lfm_weights_count(image: *const WeightImage) -> usize;
    fn lfm_weights_at(image: *const WeightImage, index: usize, out: *mut TensorView) -> i32;
    fn lfm_weights_find(
        image: *const WeightImage,
        name: *const c_char,
        out: *mut TensorView,
    ) -> i32;
    fn lfm_weights_dtype_name(dtype: u32) -> *const c_char;
    fn lfm_engine_new(workers: i32) -> *mut c_void;
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
}

static NEXT: AtomicU64 = AtomicU64::new(0);

fn workspace_model_dir() -> PathBuf {
    PathBuf::from("../../experiments/lfm2-audio-voice/model")
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
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

fn write_zero_bf16_model(path: &Path, tensors: &[(String, Vec<u64>)]) {
    let mut root = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() as usize * 2;
        let begin = data.len();
        data.resize(begin + bytes, 0);
        root.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [begin, begin + bytes],
            }),
        );
    }
    write_raw(path, serde_json::Value::Object(root), &data);
}

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
fn opaque_native_model_and_conversation_own_the_complete_token_state() {
    let temp = Temp::new();
    let hidden = 8u64;
    let ffn = 12u64;
    let vocab = 16u64;
    let codebooks = 2u64;
    let mut tensors = vec![
        ("lfm.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("lfm.embedding_norm.weight".into(), vec![hidden]),
    ];
    for layer in 0..2 {
        let root = format!("lfm.layers.{layer}.");
        tensors.extend([
            (format!("{root}operator_norm.weight"), vec![hidden]),
            (format!("{root}ffn_norm.weight"), vec![hidden]),
            (format!("{root}feed_forward.w1.weight"), vec![ffn, hidden]),
            (format!("{root}feed_forward.w3.weight"), vec![ffn, hidden]),
            (format!("{root}feed_forward.w2.weight"), vec![hidden, ffn]),
        ]);
    }
    let attention = "lfm.layers.0.self_attn.";
    tensors.extend([
        (format!("{attention}q_proj.weight"), vec![hidden, hidden]),
        (format!("{attention}k_proj.weight"), vec![4, hidden]),
        (format!("{attention}v_proj.weight"), vec![4, hidden]),
        (format!("{attention}out_proj.weight"), vec![hidden, hidden]),
        (format!("{attention}q_layernorm.weight"), vec![4]),
        (format!("{attention}k_layernorm.weight"), vec![4]),
    ]);
    let conv = "lfm.layers.1.conv.";
    tensors.extend([
        (format!("{conv}in_proj.weight"), vec![3 * hidden, hidden]),
        (format!("{conv}conv.weight"), vec![hidden, 1, 3]),
        (format!("{conv}out_proj.weight"), vec![hidden, hidden]),
    ]);
    let depth_dim = 8u64;
    let depth_ffn = 256u64;
    let depth_qkv = 16u64;
    let depth_vocab = 11u64;
    let depth = "depthformer.layers.0.";
    tensors.extend([
        (
            format!("{depth}operator.qkv_proj.weight"),
            vec![depth_qkv, depth_dim],
        ),
        (
            format!("{depth}operator.out_proj.weight"),
            vec![depth_dim, depth_dim],
        ),
        (
            format!("{depth}operator.bounded_attention.q_layernorm.weight"),
            vec![4],
        ),
        (
            format!("{depth}operator.bounded_attention.k_layernorm.weight"),
            vec![4],
        ),
        (format!("{depth}operator_norm.weight"), vec![depth_dim]),
        (format!("{depth}ffn_norm.weight"), vec![depth_dim]),
        (
            format!("{depth}feed_forward.w1.weight"),
            vec![depth_ffn, depth_dim],
        ),
        (
            format!("{depth}feed_forward.w3.weight"),
            vec![depth_ffn, depth_dim],
        ),
        (
            format!("{depth}feed_forward.w2.weight"),
            vec![depth_dim, depth_ffn],
        ),
        (
            "depth_linear.weight".into(),
            vec![codebooks * depth_dim, hidden],
        ),
        ("depth_linear.bias".into(), vec![codebooks * depth_dim]),
    ]);
    for codebook in 0..codebooks {
        let root = format!("depth_embeddings.{codebook}.");
        tensors.extend([
            (
                format!("{root}embedding.weight"),
                vec![depth_vocab, depth_dim],
            ),
            (format!("{root}embedding_norm.weight"), vec![depth_dim]),
            (
                format!("{root}to_logits.weight"),
                vec![depth_vocab, depth_dim],
            ),
        ]);
    }
    write_zero_bf16_model(&temp.0.join("model.safetensors"), &tensors);
    std::fs::write(
        temp.0.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "codebooks": codebooks,
            "depthformer": {
                "layers": 1,
                "dim": depth_dim,
                "heads": 2,
                "kv_heads": 1
            },
            "lfm": {
                "vocab_size": vocab,
                "hidden_size": hidden,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "norm_eps": 1e-5,
                "max_position_embeddings": 32,
                "conv_L_cache": 3,
                "layer_types": ["full_attention", "conv"],
                "block_ff_dim": ffn,
                "block_auto_adjust_ff_dim": false
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let engine = unsafe { lfm_engine_new(2) };
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
    assert_eq!(
        status,
        0,
        "native model open failed: {}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
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

    assert_eq!(unsafe { lfm_model_close(model) }, 0);
    unsafe { lfm_engine_free(engine) };

    let model = liquid_audio::NativeModel::open(&temp.0).expect("safe opaque model");
    let safe_info = model.info().expect("safe model info");
    assert_eq!((safe_info.hidden, safe_info.layers), (8, 2));
    assert!(safe_info.depthformer);
    let mut conversation = model
        .conversation(liquid_audio::NativeConversationConfig {
            seed: Some(7),
            temperature: None,
            top_k: None,
        })
        .expect("native conversation");
    let first = conversation
        .step(&[3], liquid_audio::EmbeddingKind::Text)
        .expect("first native token pass");
    assert_eq!((first.position, first.sampled_token), (0, 0));
    let second = conversation
        .step(&[first.sampled_token], liquid_audio::EmbeddingKind::Text)
        .expect("second native token pass");
    assert_eq!((second.position, second.sampled_token), (1, 0));
    conversation.reset().expect("native conversation reset");
    let reset = conversation
        .step(&[3], liquid_audio::EmbeddingKind::Text)
        .expect("reset native token pass");
    assert_eq!((reset.position, reset.sampled_token), (0, 0));
}

#[test]
fn rust_owner_drives_the_compatibility_builder_without_reopening_the_file() {
    let temp = Temp::new();
    let path = temp.0.join("weights.safetensors");
    let values = [1.25f32, -3.5f32];
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        &[Tensor {
            name: "model.weight",
            dtype: "F32",
            shape: &[2],
            data: &data,
        }],
    );

    let resident = ResidentWeights::open(&path).unwrap();
    assert_eq!(resident.dtype(), candle_core::DType::F32);
    assert_eq!(resident.image().len(), 1);
    let view = resident.image().find("model.weight").unwrap();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(
        view.data_ptr() as usize,
        resident.image().base() as usize + view.offset() as usize
    );

    let witness = resident.clone();
    let builder = resident.candle_builder(&candle_core::Device::Cpu);
    drop(resident);
    let tensor = builder.get((2,), "model.weight").unwrap();
    assert_eq!(tensor.to_vec1::<f32>().unwrap(), values);
    assert_eq!(
        witness.compatibility_copies(),
        liquid_audio::weights::CompatibilityCopies {
            tensors: 1,
            bytes: data.len() as u64,
        }
    );
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

#[test]
#[ignore = "needs the repository LFM2.5-Audio fixture and about 3 GB of free memory"]
fn real_model_checkpoint_loads_without_candle() {
    let dir = workspace_model_dir();
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing fixture at {}",
        dir.display()
    );
    let model = dir.join("model.safetensors");
    let bytes = std::fs::metadata(&model).unwrap().len();
    let resident = ResidentWeights::open(&dir).unwrap();
    assert_eq!(resident.dtype(), candle_core::DType::BF16);
    let image = resident.image();
    let count = image.len();
    assert!(count > 100, "real model exposed only {count} tensors");
    assert_eq!(image.resident_bytes(), (bytes + 63) & !63);

    let base = image.base() as usize;
    let end = base + image.resident_bytes() as usize;
    let mut bf16 = 0usize;
    for index in 0..count {
        let view = image.at(index).unwrap();
        assert!(!view.name().unwrap().is_empty());
        assert_eq!(view.data_ptr() as usize, base + view.offset() as usize);
        assert!(
            view.data_ptr() as usize >= base
                && view.data_ptr() as usize + view.bytes() as usize <= end
        );
        bf16 += usize::from(view.dtype().unwrap() == WeightDType::BF16);
    }
    eprintln!(
        "[native-weights] {} bytes, {count} tensors, {bf16} BF16 tensors",
        image.resident_bytes()
    );
    assert!(bf16 > 100, "real model exposed only {bf16} BF16 tensors");
}

#[test]
#[ignore = "needs the repository LFM2.5-Audio fixture and about 8 GB of free memory"]
fn real_production_loader_retains_the_native_image() {
    let dir = workspace_model_dir();
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing fixture at {}",
        dir.display()
    );
    let (model, _processor) =
        liquid_audio::from_pretrained(&dir, &candle_core::Device::Cpu).unwrap();
    let image = model
        .resident_weights()
        .expect("production model lost its native checkpoint owner");
    assert!(image.len() > 100);
    assert!(image.resident_bytes() > 2_000_000_000);

    let copies = model.compatibility_copies();
    assert!(
        copies.tensors > 500,
        "only {} tensors crossed the compatibility boundary",
        copies.tensors
    );
    assert!(
        copies.bytes > 2_000_000_000,
        "only {} bytes crossed the compatibility boundary",
        copies.bytes
    );
}
