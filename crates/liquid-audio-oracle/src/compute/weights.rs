//! Rust ownership boundary for the native resident weight image.
//!
//! C++ parses safetensors and owns one immutable, aligned allocation. Rust owns
//! the opaque image handle and may borrow tensor descriptors from it. The
//! `candle_builder` adapter is deliberately isolated here: it copies a tensor
//! out of the resident image only for model components that have not yet moved
//! to the native inference engine.

use std::ffi::{c_char, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{VarBuilder, VarMap};

const ABI_VERSION: u32 = 1;
const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = -1;
const ERROR_BYTES: usize = 1024;

#[repr(C)]
struct RawWeightImage {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawTensorView {
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
    fn lfm_weights_open_bundle(
        main_path: *const c_char,
        codec_path: *const c_char,
        out: *mut *mut RawWeightImage,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn lfm_weights_close(image: *mut RawWeightImage);
    fn lfm_weights_component_count(image: *const RawWeightImage, component: u32) -> usize;
    fn lfm_weights_at_component(
        image: *const RawWeightImage,
        component: u32,
        index: usize,
        out: *mut RawTensorView,
    ) -> i32;
    fn lfm_weights_find_component(
        image: *const RawWeightImage,
        component: u32,
        name: *const c_char,
        out: *mut RawTensorView,
    ) -> i32;
}

/// The bundle namespace is part of tensor identity. Main and codec checkpoints
/// may legally contain the same tensor name, so no lookup is allowed to infer a
/// component from the spelling of that name.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightComponent {
    Main = 1,
    Codec = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightError {
    status: i32,
    message: String,
}

impl WeightError {
    fn new(status: i32, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (native weight status {})", self.message, self.status)
    }
}

impl std::error::Error for WeightError {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDType {
    Bool = 1,
    F4 = 2,
    F6E2M3 = 3,
    F6E3M2 = 4,
    U8 = 5,
    I8 = 6,
    F8E5M2 = 7,
    F8E4M3 = 8,
    F8E8M0 = 9,
    I16 = 10,
    U16 = 11,
    F16 = 12,
    BF16 = 13,
    I32 = 14,
    U32 = 15,
    F32 = 16,
    C64 = 17,
    F64 = 18,
    I64 = 19,
    U64 = 20,
}

impl WeightDType {
    pub fn is_floating(self) -> bool {
        matches!(
            self,
            Self::F4
                | Self::F6E2M3
                | Self::F6E3M2
                | Self::F8E5M2
                | Self::F8E4M3
                | Self::F8E8M0
                | Self::F16
                | Self::BF16
                | Self::F32
                | Self::F64
        )
    }

    pub fn candle(self) -> Result<DType, WeightError> {
        match self {
            Self::U8 => Ok(DType::U8),
            Self::U32 => Ok(DType::U32),
            Self::I16 => Ok(DType::I16),
            Self::I32 => Ok(DType::I32),
            Self::I64 => Ok(DType::I64),
            Self::BF16 => Ok(DType::BF16),
            Self::F16 => Ok(DType::F16),
            Self::F32 => Ok(DType::F32),
            Self::F64 => Ok(DType::F64),
            Self::F8E4M3 => Ok(DType::F8E4M3),
            Self::F6E2M3 => Ok(DType::F6E2M3),
            Self::F6E3M2 => Ok(DType::F6E3M2),
            Self::F4 => Ok(DType::F4),
            Self::F8E8M0 => Ok(DType::F8E8M0),
            _ => Err(WeightError::new(
                INVALID_ARGUMENT,
                format!("native dtype {self:?} has no Candle representation"),
            )),
        }
    }
}

impl TryFrom<u32> for WeightDType {
    type Error = WeightError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Bool),
            2 => Ok(Self::F4),
            3 => Ok(Self::F6E2M3),
            4 => Ok(Self::F6E3M2),
            5 => Ok(Self::U8),
            6 => Ok(Self::I8),
            7 => Ok(Self::F8E5M2),
            8 => Ok(Self::F8E4M3),
            9 => Ok(Self::F8E8M0),
            10 => Ok(Self::I16),
            11 => Ok(Self::U16),
            12 => Ok(Self::F16),
            13 => Ok(Self::BF16),
            14 => Ok(Self::I32),
            15 => Ok(Self::U32),
            16 => Ok(Self::F32),
            17 => Ok(Self::C64),
            18 => Ok(Self::F64),
            19 => Ok(Self::I64),
            20 => Ok(Self::U64),
            _ => Err(WeightError::new(
                INVALID_ARGUMENT,
                format!("unknown native weight dtype {value}"),
            )),
        }
    }
}

pub struct NativeWeightImage {
    raw: NonNull<RawWeightImage>,
}

// The native image is immutable after `open` returns. Its data, descriptor
// vectors, and name table are read-only until the final Rust owner drops it.
unsafe impl Send for NativeWeightImage {}
unsafe impl Sync for NativeWeightImage {}

impl NativeWeightImage {
    pub fn open_bundle(main: &Path, codec: &Path) -> Result<Self, WeightError> {
        let main = CString::new(main.as_os_str().as_encoded_bytes()).map_err(|_| {
            WeightError::new(
                INVALID_ARGUMENT,
                "main weight path contains an embedded NUL",
            )
        })?;
        let codec = CString::new(codec.as_os_str().as_encoded_bytes()).map_err(|_| {
            WeightError::new(
                INVALID_ARGUMENT,
                "codec weight path contains an embedded NUL",
            )
        })?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0i8; ERROR_BYTES];
        let status = unsafe {
            lfm_weights_open_bundle(
                main.as_ptr(),
                codec.as_ptr(),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        Self::finish_open(status, raw, &error)
    }

    fn finish_open(
        status: i32,
        raw: *mut RawWeightImage,
        error: &[i8],
    ) -> Result<Self, WeightError> {
        if status != OK {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(WeightError::new(status, message));
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            WeightError::new(INVALID_ARGUMENT, "native loader returned a null image")
        })?;
        Ok(Self { raw })
    }

    pub fn len(&self, component: WeightComponent) -> usize {
        unsafe { lfm_weights_component_count(self.raw.as_ptr(), component as u32) }
    }

    pub fn at(
        &self,
        component: WeightComponent,
        index: usize,
    ) -> Result<NativeTensor<'_>, WeightError> {
        let mut raw = RawTensorView::default();
        let status = unsafe {
            lfm_weights_at_component(self.raw.as_ptr(), component as u32, index, &mut raw)
        };
        self.finish_view(
            status,
            raw,
            format!("{component:?} weight index {index} was not found"),
        )
    }

    pub fn find(
        &self,
        component: WeightComponent,
        name: &str,
    ) -> Result<NativeTensor<'_>, WeightError> {
        let c_name = CString::new(name).map_err(|_| {
            WeightError::new(INVALID_ARGUMENT, "tensor name contains an embedded NUL")
        })?;
        let mut raw = RawTensorView::default();
        let status = unsafe {
            lfm_weights_find_component(
                self.raw.as_ptr(),
                component as u32,
                c_name.as_ptr(),
                &mut raw,
            )
        };
        self.finish_view(
            status,
            raw,
            format!("native {component:?} tensor `{name}` was not found"),
        )
    }

    pub fn contains(&self, component: WeightComponent, name: &str) -> bool {
        self.find(component, name).is_ok()
    }

    pub fn floating_dtype(&self, component: WeightComponent) -> Result<WeightDType, WeightError> {
        let mut found: Option<(WeightDType, String)> = None;
        for index in 0..self.len(component) {
            let tensor = self.at(component, index)?;
            let dtype = tensor.dtype()?;
            if !dtype.is_floating() {
                continue;
            }
            if let Some((previous, first)) = &found {
                if *previous != dtype {
                    return Err(WeightError::new(
                        INVALID_ARGUMENT,
                        format!(
                            "mixed floating safetensor dtypes: `{first}` is {previous:?}, `{}` is {dtype:?}",
                            tensor.name()?
                        ),
                    ));
                }
                continue;
            }
            found = Some((dtype, tensor.name()?.to_owned()));
        }
        found.map(|(dtype, _)| dtype).ok_or_else(|| {
            WeightError::new(
                INVALID_ARGUMENT,
                format!("{component:?} checkpoint has no floating tensors"),
            )
        })
    }

    fn finish_view(
        &self,
        status: i32,
        raw: RawTensorView,
        missing: String,
    ) -> Result<NativeTensor<'_>, WeightError> {
        if status != OK {
            return Err(WeightError::new(status, missing));
        }
        if raw.size as usize != std::mem::size_of::<RawTensorView>()
            || raw.abi_version != ABI_VERSION
        {
            return Err(WeightError::new(
                INVALID_ARGUMENT,
                format!(
                    "native tensor descriptor ABI mismatch: size {}, version {}",
                    raw.size, raw.abi_version
                ),
            ));
        }
        if raw.name.is_null() || raw.data.is_null() || (raw.rank != 0 && raw.shape.is_null()) {
            return Err(WeightError::new(
                INVALID_ARGUMENT,
                "native loader returned an incomplete tensor descriptor",
            ));
        }
        Ok(NativeTensor {
            raw,
            _image: PhantomData,
        })
    }
}

impl Drop for NativeWeightImage {
    fn drop(&mut self) {
        unsafe { lfm_weights_close(self.raw.as_ptr()) };
    }
}

#[derive(Clone, Copy)]
pub struct NativeTensor<'a> {
    raw: RawTensorView,
    _image: PhantomData<&'a NativeWeightImage>,
}

impl NativeTensor<'_> {
    pub fn name(&self) -> Result<&str, WeightError> {
        unsafe { CStr::from_ptr(self.raw.name) }
            .to_str()
            .map_err(|error| WeightError::new(INVALID_ARGUMENT, error.to_string()))
    }

    pub fn data(&self) -> &[u8] {
        if self.raw.bytes == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.raw.data.cast::<u8>(), self.raw.bytes as usize) }
    }

    pub fn shape(&self) -> &[u64] {
        if self.raw.rank == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.raw.shape, self.raw.rank as usize) }
    }

    fn candle_shape(&self) -> Result<Vec<usize>, WeightError> {
        self.shape()
            .iter()
            .map(|&dim| {
                usize::try_from(dim).map_err(|_| {
                    WeightError::new(INVALID_ARGUMENT, "tensor dimension exceeds usize")
                })
            })
            .collect()
    }

    pub fn dtype(&self) -> Result<WeightDType, WeightError> {
        self.raw.dtype.try_into()
    }

    pub fn bytes(&self) -> u64 {
        self.raw.bytes
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompatibilityCopies {
    pub tensors: usize,
    pub bytes: u64,
}

#[derive(Default)]
struct CopyCounters {
    tensors: AtomicUsize,
    bytes: AtomicU64,
}

#[derive(Clone)]
pub struct ResidentWeights {
    image: Arc<NativeWeightImage>,
    copies: Arc<CopyCounters>,
}

impl ResidentWeights {
    pub fn open_bundle(main: &Path, codec: &Path) -> Result<Self, WeightError> {
        Self::from_image(NativeWeightImage::open_bundle(main, codec)?)
    }

    pub fn from_image(image: NativeWeightImage) -> Result<Self, WeightError> {
        image.floating_dtype(WeightComponent::Main)?.candle()?;
        Ok(Self {
            image: Arc::new(image),
            copies: Arc::new(CopyCounters::default()),
        })
    }

    pub fn dtype(&self, component: WeightComponent) -> Result<DType, WeightError> {
        self.image.floating_dtype(component)?.candle()
    }

    /// Temporary adapter for components that still instantiate Candle modules.
    /// Every successful tensor request is a payload copy and is counted.
    pub fn candle_builder(
        &self,
        component: WeightComponent,
        device: &Device,
    ) -> Result<VarBuilder<'static>, WeightError> {
        let dtype = self.dtype(component)?;
        Ok(VarBuilder::from_backend(
            Box::new(CandleBridge {
                image: self.image.clone(),
                component,
                copies: self.copies.clone(),
            }),
            dtype,
            device.clone(),
        ))
    }

    pub fn compatibility_copies(&self) -> CompatibilityCopies {
        CompatibilityCopies {
            tensors: self.copies.tensors.load(Ordering::Relaxed),
            bytes: self.copies.bytes.load(Ordering::Relaxed),
        }
    }

    /// Initialize the mutable variables of the offline training oracle from
    /// validated native views. This is the only intentional weight copy: the
    /// model image remains byte-exact and immutable, while autograd requires
    /// mutable framework-owned storage. Every copied payload is accounted.
    pub fn copy_into_varmap(&self, component: WeightComponent, vars: &VarMap) -> CandleResult<()> {
        let mut data = vars.data().lock().map_err(|_| {
            candle_core::Error::Msg("training variable map lock was poisoned".into())
        })?;
        for (name, var) in data.iter_mut() {
            let view = self
                .image
                .find(component, name)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            let shape = view
                .candle_shape()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if var.dims() != shape {
                return Err(candle_core::Error::UnexpectedShape {
                    msg: format!("shape mismatch for native {component:?} weight {name}"),
                    expected: var.shape().clone(),
                    got: Shape::from(shape),
                }
                .bt());
            }
            let source = view
                .dtype()
                .and_then(WeightDType::candle)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            let tensor = Tensor::from_raw_buffer(view.data(), source, var.dims(), var.device())?
                .to_dtype(var.dtype())?;
            var.set(&tensor)?;
            self.copies.tensors.fetch_add(1, Ordering::Relaxed);
            self.copies.bytes.fetch_add(view.bytes(), Ordering::Relaxed);
        }
        Ok(())
    }
}

struct CandleBridge {
    image: Arc<NativeWeightImage>,
    component: WeightComponent,
    copies: Arc<CopyCounters>,
}

impl CandleBridge {
    fn load(
        &self,
        name: &str,
        expected: Option<&Shape>,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let view = self
            .image
            .find(self.component, name)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let shape = view
            .candle_shape()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if let Some(expected) = expected {
            if expected.dims() != shape {
                return Err(candle_core::Error::UnexpectedShape {
                    msg: format!("shape mismatch for native weight {name}"),
                    expected: expected.clone(),
                    got: Shape::from(shape),
                }
                .bt());
            }
        }
        let source = view
            .dtype()
            .and_then(WeightDType::candle)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let tensor =
            Tensor::from_raw_buffer(view.data(), source, &shape, device)?.to_dtype(dtype)?;
        self.copies.tensors.fetch_add(1, Ordering::Relaxed);
        self.copies.bytes.fetch_add(view.bytes(), Ordering::Relaxed);
        Ok(tensor)
    }
}

impl SimpleBackend for CandleBridge {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        self.load(name, Some(&shape), dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> CandleResult<Tensor> {
        self.load(name, None, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.image.contains(self.component, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dtype_mapping_is_explicit() {
        assert_eq!(WeightDType::BF16.candle().unwrap(), DType::BF16);
        assert!(WeightDType::I8.candle().is_err());
        assert!(WeightDType::F8E5M2.is_floating());
    }

    #[test]
    fn raw_tensor_view_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<RawTensorView>(), 72);
        assert_eq!(std::mem::align_of::<RawTensorView>(), 8);
    }
}
