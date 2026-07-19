//! Native kcoro coordination substrate.
//!
//! The production-facing Rust surface is intentionally narrow. Numerical
//! payloads never pass through this crate; callers share fixed records and use
//! an expected-value doorbell only to resume a predicate-driven continuation.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};
use std::sync::Arc;

unsafe extern "C" {
    fn kc_doorbell_create(out: *mut *mut c_void) -> i32;
    fn kc_doorbell_observe(doorbell: *const c_void) -> u32;
    fn kc_doorbell_ring_one(doorbell: *mut c_void);
    fn kc_doorbell_ring_all(doorbell: *mut c_void);
    fn kc_doorbell_wait(doorbell: *mut c_void, expected: u32, deadline_ns: u64) -> i32;
    fn kc_doorbell_realtime_safe(doorbell: *const c_void) -> i32;
    fn kc_doorbell_destroy(doorbell: *mut c_void);
}

const ABI_VERSION: u32 = 1;

#[repr(C)]
struct NativeRuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    arena_segment_size: usize,
    ticket_capacity: u32,
    reserved: u32,
}

type NativeCallback = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct NativeServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<NativeCallback>,
    context: *mut c_void,
    reserved: u64,
}

#[repr(C)]
#[derive(Default)]
struct NativeServiceSnapshot {
    size: u32,
    abi_version: u32,
    notifications: u64,
    handled_notifications: u64,
    callbacks: u64,
    run_state: u32,
    started: u32,
    stop_requested: u32,
    joined: u32,
}

unsafe extern "C" {
    fn kc_runtime_create(config: *const NativeRuntimeConfig, out: *mut *mut c_void) -> i32;
    fn kc_runtime_start(runtime: *mut c_void) -> i32;
    fn kc_runtime_run_until_idle(runtime: *mut c_void) -> i32;
    fn kc_runtime_request_stop(runtime: *mut c_void);
    fn kc_runtime_join(runtime: *mut c_void) -> i32;
    fn kc_runtime_destroy(runtime: *mut c_void) -> i32;

    fn kc_service_create(
        runtime: *mut c_void,
        config: *const NativeServiceConfig,
        out: *mut *mut c_void,
    ) -> i32;
    fn kc_service_start(service: *mut c_void) -> i32;
    fn kc_service_notify(service: *mut c_void) -> i32;
    fn kc_service_notifier_create(service: *mut c_void, out: *mut *mut c_void) -> i32;
    fn kc_service_notifier_notify(notifier: *mut c_void) -> i32;
    fn kc_service_notifier_destroy(notifier: *mut c_void) -> i32;
    fn kc_service_ready_again(service: *mut c_void) -> i32;
    fn kc_service_request_stop(service: *mut c_void);
    fn kc_service_join(service: *mut c_void) -> i32;
    fn kc_service_snapshot_get(service: *mut c_void, out: *mut NativeServiceSnapshot) -> i32;
    fn kc_service_destroy(service: *mut c_void) -> i32;
}

#[inline]
fn status(code: i32) -> Result<(), i32> {
    if code != 0 {
        return Err(code);
    }
    Ok(())
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

    /// Whether ringing this doorbell is admissible from a realtime callback on
    /// the current host. Callers must select a non-realtime producer path when
    /// this is false.
    #[inline]
    pub fn realtime_safe(&self) -> bool {
        unsafe { kc_doorbell_realtime_safe(self.raw.as_ptr()) != 0 }
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

/// Setup parameters for a [`Runtime`]. Zero-valued fields select the native
/// defaults (one worker, a one-MiB arena segment, and 256 tickets).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub workers: u32,
    pub segment: usize,
    pub tickets: u32,
}

struct RuntimeInner {
    raw: NonNull<c_void>,
}

// The native runtime serializes lifecycle changes with its mutex and uses
// atomics plus its doorbell for worker-facing state.
unsafe impl Send for RuntimeInner {}
unsafe impl Sync for RuntimeInner {}

impl RuntimeInner {
    #[inline]
    fn start(&self) -> Result<(), i32> {
        status(unsafe { kc_runtime_start(self.raw.as_ptr()) })
    }

    #[inline]
    fn stop(&self) {
        unsafe { kc_runtime_request_stop(self.raw.as_ptr()) }
    }

    #[inline]
    fn join(&self) -> Result<(), i32> {
        status(unsafe { kc_runtime_join(self.raw.as_ptr()) })
    }
}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        self.stop();
        if self.join().is_err() {
            return;
        }
        let _ = unsafe { kc_runtime_destroy(self.raw.as_ptr()) };
    }
}

/// An owning kcoro runtime.
///
/// Services retain the native runtime through an internal ownership token, so
/// a normal Rust owner can store a `Runtime` and its `Service` values side by
/// side. Dropping this public handle requests stop; native destruction waits
/// until every service has released its token.
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Create a runtime with the native defaults.
    pub fn new() -> Result<Self, i32> {
        Self::with_config(RuntimeConfig::default())
    }

    /// Create a runtime with explicit setup parameters.
    pub fn with_config(config: RuntimeConfig) -> Result<Self, i32> {
        let config = NativeRuntimeConfig {
            size: std::mem::size_of::<NativeRuntimeConfig>() as u32,
            abi_version: ABI_VERSION,
            worker_count: config.workers,
            arena_segment_size: config.segment,
            ticket_capacity: config.tickets,
            reserved: 0,
        };
        let mut raw = std::ptr::null_mut();
        status(unsafe { kc_runtime_create(&config, &mut raw) })?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                raw: NonNull::new(raw)
                    .expect("kc_runtime_create returned success without an object"),
            }),
        })
    }

    /// Start the fixed native worker set.
    #[inline]
    pub fn start(&self) -> Result<(), i32> {
        self.inner.start()
    }

    /// Wait until currently runnable service work is drained.
    #[inline]
    pub fn run_until_idle(&self) -> Result<(), i32> {
        status(unsafe { kc_runtime_run_until_idle(self.inner.raw.as_ptr()) })
    }

    /// Close admission and request every retained service to stop.
    #[inline]
    pub fn stop(&self) {
        self.inner.stop()
    }

    /// Join the worker set once all tracked services have retired.
    #[inline]
    pub fn join(&self) -> Result<(), i32> {
        self.inner.join()
    }

    /// Stop and join this handle. Native destruction occurs when services have
    /// released their internal runtime ownership tokens.
    pub fn destroy(self) -> Result<(), i32> {
        self.stop();
        self.join()
    }

    /// Mount a retained, serialized callback on this runtime.
    ///
    /// The closure is owned by the returned service and may run on any runtime
    /// worker. It must therefore be `Send` and `'static`. Panics are caught by
    /// the FFI trampoline and recorded by [`Service::callback_panicked`].
    pub fn service<F>(&self, callback: F) -> Result<Service, i32>
    where
        F: FnMut() + Send + 'static,
    {
        let mut callback = callback;
        self.polling_service(move || {
            callback();
            ServiceOutcome::Park
        })
    }

    /// Mount a bounded retained callback that can explicitly remain ready.
    ///
    /// Returning [`ServiceOutcome::ReadyAgain`] requeues this exact
    /// continuation after the callback yields. It does not wait for another
    /// producer, acquire the runtime mutex, ring a wait word, or create a timer.
    /// Use it when one invocation consumes a fixed quota from a still-ready
    /// predicate.
    pub fn polling_service<F>(&self, callback: F) -> Result<Service, i32>
    where
        F: FnMut() -> ServiceOutcome + Send + 'static,
    {
        let context = Box::new(Callback {
            callback: UnsafeCell::new(Box::new(callback)),
            panicked: AtomicBool::new(false),
            service: AtomicPtr::new(std::ptr::null_mut()),
            reschedule_error: AtomicI32::new(0),
        });
        let config = NativeServiceConfig {
            size: std::mem::size_of::<NativeServiceConfig>() as u32,
            abi_version: ABI_VERSION,
            callback: Some(invoke),
            context: (&*context as *const Callback).cast_mut().cast(),
            reserved: 0,
        };
        let mut raw = std::ptr::null_mut();
        status(unsafe { kc_service_create(self.inner.raw.as_ptr(), &config, &mut raw) })?;
        context.service.store(raw, Ordering::Release);
        Ok(Service {
            inner: Arc::new(ServiceInner {
                raw: NonNull::new(raw)
                    .expect("kc_service_create returned success without an object"),
                runtime: Arc::clone(&self.inner),
                context: Some(context),
            }),
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Callback {
    callback: UnsafeCell<Box<dyn FnMut() -> ServiceOutcome + Send + 'static>>,
    panicked: AtomicBool,
    service: AtomicPtr<c_void>,
    reschedule_error: AtomicI32,
}

// kc_service guarantees that one retained continuation invokes its callback
// serially. Rust never exposes the UnsafeCell or a reference to its closure.
unsafe impl Sync for Callback {}

unsafe extern "C" fn invoke(context: *mut c_void) {
    let context = unsafe { &*context.cast::<Callback>() };
    if context.panicked.load(Ordering::Acquire) {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let callback = unsafe { &mut *context.callback.get() };
        callback()
    }));
    match result {
        Ok(ServiceOutcome::Park) => {}
        Ok(ServiceOutcome::ReadyAgain) => {
            let status = unsafe { kc_service_ready_again(context.service.load(Ordering::Acquire)) };
            if status != 0 {
                context.reschedule_error.store(status, Ordering::Release);
            }
        }
        Err(payload) => {
            context.panicked.store(true, Ordering::Release);
            // A user-defined panic payload may itself panic from Drop. Leaking
            // this one exceptional payload keeps even that unwind from crossing
            // the C ABI.
            std::mem::forget(payload);
        }
    }
}

/// Outcome of one bounded retained-service callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceOutcome {
    /// The callback drained its predicate and may become dormant.
    Park,
    /// Work remains after this invocation's fixed quota; yield and re-enter.
    ReadyAgain,
}

struct ServiceInner {
    raw: NonNull<c_void>,
    runtime: Arc<RuntimeInner>,
    context: Option<Box<Callback>>,
}

// Native service operations lock or use atomics, and its continuation invokes
// the Send closure serially even when the runtime has several workers.
unsafe impl Send for ServiceInner {}
unsafe impl Sync for ServiceInner {}

impl ServiceInner {
    #[inline]
    fn stop(&self) {
        unsafe { kc_service_request_stop(self.raw.as_ptr()) }
    }

    #[inline]
    fn join(&self) -> Result<(), i32> {
        status(unsafe { kc_service_join(self.raw.as_ptr()) })
    }

    fn leak(&mut self) {
        if let Some(context) = self.context.take() {
            std::mem::forget(context);
        }
    }
}

impl Drop for ServiceInner {
    fn drop(&mut self) {
        self.stop();
        if self.join().is_err() {
            if self.runtime.start().is_err() {
                self.leak();
                return;
            }
            self.stop();
            if self.join().is_err() {
                self.leak();
                return;
            }
        }
        if unsafe { kc_service_destroy(self.raw.as_ptr()) } != 0 {
            self.leak();
        }
    }
}

/// A point-in-time view of service notification and callback progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceSnapshot {
    pub notifications: u64,
    pub handled_notifications: u64,
    pub callbacks: u64,
    pub run_state: u32,
    pub started: bool,
    pub stop_requested: bool,
    pub joined: bool,
}

/// An owning retained service and its callback context.
///
/// Dropping the public service stops and joins it. Native service destruction
/// and callback-context release are deferred until every realtime notifier has
/// first destroyed its native lease.
pub struct Service {
    inner: Arc<ServiceInner>,
}

impl Service {
    /// Publish the service continuation to the runtime.
    #[inline]
    pub fn start(&self) -> Result<(), i32> {
        status(unsafe { kc_service_start(self.inner.raw.as_ptr()) })
    }

    /// Send a control-plane notification. This may take the runtime mutex.
    #[inline]
    pub fn notify(&self) -> Result<(), i32> {
        status(unsafe { kc_service_notify(self.inner.raw.as_ptr()) })
    }

    /// Create a setup-time, retained realtime notification lease.
    pub fn realtime_notifier(&self) -> Result<RealtimeNotifier, i32> {
        let mut raw = std::ptr::null_mut();
        status(unsafe { kc_service_notifier_create(self.inner.raw.as_ptr(), &mut raw) })?;
        Ok(RealtimeNotifier {
            raw: NonNull::new(raw)
                .expect("kc_service_notifier_create returned success without an object"),
            service: Arc::clone(&self.inner),
            single: PhantomData,
        })
    }

    /// Close notification admission and request retirement.
    #[inline]
    pub fn stop(&self) {
        self.inner.stop()
    }

    /// Wait for all accepted notifications and the active callback to drain.
    #[inline]
    pub fn join(&self) -> Result<(), i32> {
        self.inner.join()
    }

    /// Stop and join this public handle. Native destruction is deferred until
    /// every retained notifier has been dropped.
    pub fn destroy(self) -> Result<(), i32> {
        self.stop();
        self.join()
    }

    /// Whether the callback has panicked. After the first panic the trampoline
    /// safely ignores subsequent invocations while the native service drains.
    #[inline]
    pub fn callback_panicked(&self) -> bool {
        self.inner
            .context
            .as_ref()
            .expect("live service lost its callback context")
            .panicked
            .load(Ordering::Acquire)
    }

    /// The native status from a failed local reschedule, if any. A stop racing
    /// a callback may legitimately close admission before `ReadyAgain` lands.
    #[inline]
    pub fn reschedule_error(&self) -> Option<i32> {
        let status = self
            .inner
            .context
            .as_ref()
            .expect("live service lost its callback context")
            .reschedule_error
            .load(Ordering::Acquire);
        (status != 0).then_some(status)
    }

    pub fn snapshot(&self) -> Result<ServiceSnapshot, i32> {
        let mut snapshot = NativeServiceSnapshot {
            size: std::mem::size_of::<NativeServiceSnapshot>() as u32,
            abi_version: ABI_VERSION,
            ..NativeServiceSnapshot::default()
        };
        status(unsafe { kc_service_snapshot_get(self.inner.raw.as_ptr(), &mut snapshot) })?;
        Ok(ServiceSnapshot {
            notifications: snapshot.notifications,
            handled_notifications: snapshot.handled_notifications,
            callbacks: snapshot.callbacks,
            run_state: snapshot.run_state,
            started: snapshot.started != 0,
            stop_requested: snapshot.stop_requested != 0,
            joined: snapshot.joined != 0,
        })
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop();
        let _ = self.join();
    }
}

/// A retained, single-producer lease for a service's realtime notify edge.
///
/// Creation and destruction are setup-time operations and may allocate or
/// lock. [`notify`](Self::notify) performs no Rust allocation and calls only
/// the native lock-free realtime path. Requiring `&mut self`, omitting `Sync`,
/// and omitting `Clone` make one lease structurally single-producer. The lease
/// may be moved to one producer thread.
pub struct RealtimeNotifier {
    raw: NonNull<c_void>,
    service: Arc<ServiceInner>,
    single: PhantomData<Cell<()>>,
}

unsafe impl Send for RealtimeNotifier {}

impl RealtimeNotifier {
    /// Publish the producer-owned predicate before calling this method.
    #[inline]
    pub fn notify(&mut self) -> Result<(), i32> {
        status(unsafe { kc_service_notifier_notify(self.raw.as_ptr()) })
    }
}

impl Drop for RealtimeNotifier {
    fn drop(&mut self) {
        if unsafe { kc_service_notifier_destroy(self.raw.as_ptr()) } != 0 {
            std::mem::forget(Arc::clone(&self.service));
        }
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
