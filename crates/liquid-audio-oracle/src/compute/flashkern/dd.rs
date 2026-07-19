//! ABI record for native double-double kernels.
//!
//! Arithmetic belongs to the architecture kernels. Rust carries this pair only
//! while the temporary conformance rim still exposes the native IRFFT entry point.

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f32,
    pub lo: f32,
}

impl Dd {
    #[inline]
    pub const fn from_f32(value: f32) -> Self {
        Self { hi: value, lo: 0.0 }
    }
}
