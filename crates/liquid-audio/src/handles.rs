use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::ffi;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} (native status {})",
            self.message, self.status
        )
    }
}

impl std::error::Error for NativeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    pub resident_bytes: u64,
    pub hidden: u32,
    pub ffn: u32,
    pub layers: u32,
    pub vocab: u32,
    pub max_context: u32,
    pub codebooks: u32,
    pub depthformer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConversationConfig {
    pub seed: Option<u64>,
    pub temperature: Option<f64>,
    pub top_k: Option<u32>,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            seed: None,
            temperature: None,
            top_k: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingKind {
    Text,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenResult {
    pub position: u64,
    pub sampled_token: u32,
}

struct ModelInner(NonNull<ffi::Model>);

unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        // Arc ownership guarantees every conversation has already released its
        // model reference. A nonzero result leaves the native allocation live.
        let status = unsafe { ffi::lfm_model_close(self.0.as_ptr()) };
        if status != 0 {
            eprintln!("[flashkern] native model close refused with status {status}");
        }
    }
}

#[derive(Clone)]
pub struct NativeModel(Arc<ModelInner>);

impl NativeModel {
    pub fn open(path: &Path) -> Result<Self, NativeError> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| NativeError {
            status: -1,
            message: "model path contains a NUL byte".into(),
        })?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0i8; 512];
        let engine = crate::flashkern::native_engine::process_engine().raw_ptr();
        let status = unsafe {
            ffi::lfm_model_open(
                engine,
                path.as_ptr(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(status, &error));
        }
        let pointer = NonNull::new(pointer).ok_or_else(|| NativeError {
            status: -1,
            message: "native model open returned a null handle".into(),
        })?;
        Ok(Self(Arc::new(ModelInner(pointer))))
    }

    pub fn info(&self) -> Result<ModelInfo, NativeError> {
        let mut raw = ffi::ModelInfo {
            size: std::mem::size_of::<ffi::ModelInfo>() as u32,
            abi_version: ffi::ABI,
            ..Default::default()
        };
        let status = unsafe { ffi::lfm_model_info(self.0 .0.as_ptr(), &mut raw) };
        if status != 0 {
            return Err(status_error(status, "native model info failed"));
        }
        Ok(ModelInfo {
            resident_bytes: raw.resident_bytes,
            hidden: raw.hidden,
            ffn: raw.ffn,
            layers: raw.layers,
            vocab: raw.vocab,
            max_context: raw.max_context,
            codebooks: raw.codebooks,
            depthformer: raw.depth_plan_id != 0,
        })
    }

    pub fn conversation(
        &self,
        config: ConversationConfig,
    ) -> Result<NativeConversation, NativeError> {
        let greedy = config.temperature.is_none();
        let sampler = ffi::SamplerConfig {
            size: std::mem::size_of::<ffi::SamplerConfig>() as u32,
            abi_version: ffi::SAMPLE_ABI,
            flags: greedy.then_some(ffi::SAMPLE_GREEDY).unwrap_or(0),
            top_k: config.top_k.unwrap_or(0),
            temperature: config.temperature.unwrap_or(1.0),
            reserved: 0,
        };
        let raw = ffi::ConversationConfig {
            size: std::mem::size_of::<ffi::ConversationConfig>() as u32,
            abi_version: ffi::ABI,
            flags: config
                .seed
                .is_none()
                .then_some(ffi::SEED_SYSTEM)
                .unwrap_or(0),
            reserved0: 0,
            seed: config.seed.unwrap_or(0),
            text_sampler: sampler,
            audio_sampler: sampler,
            reserved: [0; 4],
        };
        let mut pointer = std::ptr::null_mut();
        let mut error = [0i8; 512];
        let status = unsafe {
            ffi::lfm_conversation_create(
                self.0 .0.as_ptr(),
                &raw,
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(native_error(status, &error));
        }
        let pointer = NonNull::new(pointer).ok_or_else(|| NativeError {
            status: -1,
            message: "native conversation create returned a null handle".into(),
        })?;
        Ok(NativeConversation {
            pointer,
            model: self.0.clone(),
        })
    }
}

pub struct NativeConversation {
    pointer: NonNull<ffi::Conversation>,
    model: Arc<ModelInner>,
}

unsafe impl Send for NativeConversation {}

impl NativeConversation {
    pub fn step(&mut self, ids: &[u32], kind: EmbeddingKind) -> Result<TokenResult, NativeError> {
        let mut result = ffi::TokenResult {
            size: std::mem::size_of::<ffi::TokenResult>() as u32,
            abi_version: ffi::ABI,
            position: 0,
            sampled_token: 0,
            input_count: 0,
            embedding_kind: 0,
            flags: 0,
            reserved: [0; 4],
        };
        let kind = match kind {
            EmbeddingKind::Text => 0,
            EmbeddingKind::Audio => 1,
        };
        let status = unsafe {
            ffi::lfm_conversation_step(
                self.pointer.as_ptr(),
                ids.as_ptr(),
                ids.len(),
                kind,
                &mut result,
            )
        };
        if status != 0 {
            return Err(status_error(status, "native conversation step failed"));
        }
        Ok(TokenResult {
            position: result.position,
            sampled_token: result.sampled_token,
        })
    }

    pub fn reset(&mut self) -> Result<(), NativeError> {
        let status = unsafe { ffi::lfm_conversation_reset(self.pointer.as_ptr()) };
        if status != 0 {
            return Err(status_error(status, "native conversation reset failed"));
        }
        Ok(())
    }

    pub fn model(&self) -> NativeModel {
        NativeModel(self.model.clone())
    }
}

impl Drop for NativeConversation {
    fn drop(&mut self) {
        let status = unsafe { ffi::lfm_conversation_close(self.pointer.as_ptr()) };
        if status != 0 {
            eprintln!("[flashkern] native conversation close refused with status {status}");
        }
    }
}

fn native_error(status: i32, error: &[i8]) -> NativeError {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    NativeError {
        status,
        message: if message.is_empty() {
            "native operation failed".into()
        } else {
            message
        },
    }
}

fn status_error(status: i32, message: &str) -> NativeError {
    NativeError {
        status,
        message: message.into(),
    }
}
