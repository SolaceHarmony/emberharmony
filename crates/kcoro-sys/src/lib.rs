//! Link anchor for the vendored kcoro_arena C runtime.
//!
//! This crate intentionally exposes no safe Rust wrapper. `liquid-audio` owns the
//! private FFI declarations for the native engine and only depends on this crate so
//! Cargo builds and links the stackless coordination runtime exactly once.

#[inline(always)]
pub fn link_anchor() {}
