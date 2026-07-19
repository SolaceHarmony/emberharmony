//! Native kcoro coordination substrate.
//!
//! The production-facing Rust surface is intentionally narrow. Numerical
//! payloads never pass through this crate; callers share fixed records and use
//! an expected-value doorbell only to resume a predicate-driven continuation.

use std::ffi::c_void;
use std::ptr::NonNull;

unsafe extern "C" {
    fn kc_doorbell_create(out: *mut *mut c_void) -> i32;
    fn kc_doorbell_observe(doorbell: *const c_void) -> u32;
    fn kc_doorbell_ring_one(doorbell: *mut c_void);
    fn kc_doorbell_ring_all(doorbell: *mut c_void);
    fn kc_doorbell_wait(doorbell: *mut c_void, expected: u32, deadline_ns: u64) -> i32;
    fn kc_doorbell_destroy(doorbell: *mut c_void);
}

/// Cache-isolated expected-value edge shared with native kcoro.
///
/// The doorbell is not the condition. A consumer snapshots `observe`, checks
/// every owned predicate, then calls `park` only while those predicates remain
/// false. Producers publish state before ringing. `park` has deliberately no
/// timeout: capture-frame thresholds drive speech policy; a separate named
/// device-liveness fault source owns any wall-clock deadline.
pub struct Doorbell {
    raw: NonNull<c_void>,
}

// The native object contains only a lock-free sequence, an immutable prepared
// wait registration, and its backend teardown accounting. Every operation is
// explicitly multi-thread safe; ownership remains with the Rust value.
unsafe impl Send for Doorbell {}
unsafe impl Sync for Doorbell {}

impl Doorbell {
    pub fn new() -> Result<Self, i32> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { kc_doorbell_create(&mut raw) };
        if status != 0 {
            return Err(status);
        }
        Ok(Self {
            raw: NonNull::new(raw).expect("kc_doorbell_create returned success without an object"),
        })
    }

    #[inline]
    pub fn observe(&self) -> u32 {
        unsafe { kc_doorbell_observe(self.raw.as_ptr()) }
    }

    #[inline]
    pub fn ring_one(&self) {
        unsafe { kc_doorbell_ring_one(self.raw.as_ptr()) }
    }

    #[inline]
    pub fn ring_all(&self) {
        unsafe { kc_doorbell_ring_all(self.raw.as_ptr()) }
    }

    /// Park until the sequence differs from `expected`.
    ///
    /// Callers must recheck their actual predicate after every return. Spurious
    /// wakes and unrelated edges are valid. This performs no timed progress.
    #[inline]
    pub fn park(&self, expected: u32) -> Result<(), i32> {
        match unsafe { kc_doorbell_wait(self.raw.as_ptr(), expected, 0) } {
            0 => Ok(()),
            status => Err(status),
        }
    }
}

impl Drop for Doorbell {
    fn drop(&mut self) {
        unsafe { kc_doorbell_destroy(self.raw.as_ptr()) }
    }
}

/// Preserve the explicit link anchor used by low-level ABI conformance tests.
#[inline(always)]
pub fn link_anchor() {}

#[cfg(test)]
mod tests {
    use super::Doorbell;
    use std::sync::Arc;

    #[test]
    fn publication_before_park_and_callback_resume_are_lost_wake_safe() {
        let doorbell = Arc::new(Doorbell::new().unwrap());
        let initial = doorbell.observe();
        doorbell.ring_all();
        assert_eq!(doorbell.park(initial), Ok(()));

        let expected = doorbell.observe();
        let parked = Arc::clone(&doorbell);
        let waiter = std::thread::spawn(move || parked.park(expected));
        doorbell.ring_all();
        assert_eq!(waiter.join().unwrap(), Ok(()));
    }
}
