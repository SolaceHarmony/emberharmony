//! Native kcoro coordination substrate.
//!
//! The production-facing Rust surface is intentionally narrow. Numerical
//! payloads never pass through this crate; callers publish retained notifier
//! edges that make predicate-driven continuations runnable. Only the native
//! runtime's resident worker may use its private idle-dormancy doorbell.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};
use std::sync::Arc;

const ABI_VERSION: u32 = 1;

#[repr(C)]
struct NativeRuntimeConfig {
    size: u32,
    abi_version: u32,
    worker_count: u32,
    reserved: u32,
}

type NativeCallback = unsafe extern "C" fn(*mut c_void);
type ServiceTask = dyn FnMut() -> ServiceOutcome + 'static;
type OwnerInitializer = dyn FnOnce() -> Box<ServiceTask> + Send + 'static;

#[repr(C)]
struct NativeServiceConfig {
    size: u32,
    abi_version: u32,
    callback: Option<NativeCallback>,
    context: *mut c_void,
    reserved: u64,
    owner_init: Option<NativeCallback>,
    owner_fini: Option<NativeCallback>,
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
    fn kc_service_complete_current(service: *mut c_void) -> i32;
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

/// Setup parameters for a [`Runtime`]. Zero workers selects one fixed owner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub workers: u32,
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
    /// The closure is owned by the returned service and always runs on its
    /// permanent runtime-worker owner. It must be `Send` and `'static`. Panics are caught by
    /// the FFI trampoline and recorded by [`Service::callback_panicked`].
    pub fn service<F>(&self, callback: F) -> Result<Service, i32>
    where
        F: FnMut() + Send + 'static,
    {
        let mut callback = callback;
        self.state_service(move || {
            callback();
            ServiceOutcome::Dormant
        })
    }

    /// Mount a bounded retained callback that can explicitly remain ready.
    ///
    /// Returning [`ServiceOutcome::Continue`] requeues this exact
    /// continuation after the callback yields. It does not wait for another
    /// producer, acquire the runtime mutex, ring a wait word, or create a timer.
    /// Use it when one invocation consumes a fixed quota from a still-ready
    /// predicate.
    pub fn state_service<F>(&self, callback: F) -> Result<Service, i32>
    where
        F: FnMut() -> ServiceOutcome + Send + 'static,
    {
        self.state_service_factory(|_| Ok::<_, i32>((callback, ())))
            .map(|(service, ())| service)
    }

    /// Build one retained service and its producer edges as one setup
    /// transaction.
    ///
    /// The native service exists while `build` runs, but it cannot be started
    /// and its callback is not installed yet. The restricted [`ServiceSetup`]
    /// handle may mint any number of distinct, single-producer realtime
    /// notifiers. `build` then returns the durable callback state and any
    /// producer-owned setup result. Only after that succeeds is the callback
    /// installed and the public [`Service`] returned. This closes the
    /// service-before-notifier construction cycle without an `AtomicPtr`, a
    /// mutex relay, or another runtime.
    pub fn state_service_factory<F, B, T>(&self, build: B) -> Result<(Service, T), i32>
    where
        F: FnMut() -> ServiceOutcome + Send + 'static,
        B: FnOnce(&ServiceSetup) -> Result<(F, T), i32>,
    {
        let context = Box::new(Callback {
            callback: UnsafeCell::new(None),
            initializer: UnsafeCell::new(None),
            panicked: AtomicBool::new(false),
            owner_local: false,
            owner_initialized: AtomicBool::new(false),
            owner_retired: AtomicBool::new(false),
            service: AtomicPtr::new(std::ptr::null_mut()),
            reschedule_error: AtomicI32::new(0),
        });
        let config = NativeServiceConfig {
            size: std::mem::size_of::<NativeServiceConfig>() as u32,
            abi_version: ABI_VERSION,
            callback: Some(invoke),
            context: (&*context as *const Callback).cast_mut().cast(),
            reserved: 0,
            owner_init: None,
            owner_fini: Some(retire),
        };
        let mut raw = std::ptr::null_mut();
        status(unsafe { kc_service_create(self.inner.raw.as_ptr(), &config, &mut raw) })?;
        context.service.store(raw, Ordering::Release);
        let lease = Arc::new(ServiceLease {
            raw: NonNull::new(raw).expect("kc_service_create returned success without an object"),
            runtime: Arc::clone(&self.inner),
        });
        let inner = Arc::new(ServiceInner {
            lease,
            context: Some(context),
        });
        let setup = ServiceSetup {
            inner: Arc::clone(&inner),
        };
        let (callback, value) = build(&setup)?;
        let context = inner
            .context
            .as_ref()
            .expect("new service lost its callback context");
        // SAFETY: the service remains in CREATED state and no public Service
        // exists until after this store. ServiceSetup cannot start or notify
        // it, so the native trampoline cannot concurrently read the slot.
        unsafe {
            *context.callback.get() = Some(Box::new(callback));
        }
        Ok((Service { inner }, value))
    }

    /// Construct and retain a non-`Send` state machine on one permanent owner.
    ///
    /// `build` runs on the caller during setup and may mint realtime producer
    /// edges. It returns a `Send` initializer, not the owner-local state itself.
    /// After [`Service::start`], kcoro invokes that initializer exactly once on
    /// the service's fixed worker. The returned callback may contain `!Send`
    /// resources such as platform audio streams: it is advanced only by that
    /// worker and is destroyed there before terminal completion is published.
    /// No task stack, TLS slot, waiter, or intermediary thread owns the state.
    pub fn owner_state_service_factory<I, F, B, T>(&self, build: B) -> Result<(Service, T), i32>
    where
        I: FnOnce() -> F + Send + 'static,
        F: FnMut() -> ServiceOutcome + 'static,
        B: FnOnce(&ServiceSetup) -> Result<(I, T), i32>,
    {
        let context = Box::new(Callback {
            callback: UnsafeCell::new(None),
            initializer: UnsafeCell::new(None),
            panicked: AtomicBool::new(false),
            owner_local: true,
            owner_initialized: AtomicBool::new(false),
            owner_retired: AtomicBool::new(false),
            service: AtomicPtr::new(std::ptr::null_mut()),
            reschedule_error: AtomicI32::new(0),
        });
        let config = NativeServiceConfig {
            size: std::mem::size_of::<NativeServiceConfig>() as u32,
            abi_version: ABI_VERSION,
            callback: Some(invoke),
            context: (&*context as *const Callback).cast_mut().cast(),
            reserved: 0,
            owner_init: Some(initialize),
            owner_fini: Some(retire),
        };
        let mut raw = std::ptr::null_mut();
        status(unsafe { kc_service_create(self.inner.raw.as_ptr(), &config, &mut raw) })?;
        context.service.store(raw, Ordering::Release);
        let lease = Arc::new(ServiceLease {
            raw: NonNull::new(raw).expect("kc_service_create returned success without an object"),
            runtime: Arc::clone(&self.inner),
        });
        let inner = Arc::new(ServiceInner {
            lease,
            context: Some(context),
        });
        let setup = ServiceSetup {
            inner: Arc::clone(&inner),
        };
        let (initializer, value) = build(&setup)?;
        let context = inner
            .context
            .as_ref()
            .expect("new service lost its callback context");
        // SAFETY: the native service remains CREATED. The initializer itself is
        // Send; the !Send task does not exist until initialize() runs on owner.
        unsafe {
            *context.initializer.get() = Some(Box::new(move || Box::new(initializer())));
        }
        Ok((Service { inner }, value))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Callback {
    callback: UnsafeCell<Option<Box<ServiceTask>>>,
    initializer: UnsafeCell<Option<Box<OwnerInitializer>>>,
    panicked: AtomicBool,
    owner_local: bool,
    owner_initialized: AtomicBool,
    owner_retired: AtomicBool,
    service: AtomicPtr<c_void>,
    reschedule_error: AtomicI32,
}

// Before owner initialization, every transferable field is Send. Afterwards,
// kc_service guarantees fixed-worker serialization and invokes retire() on that
// same owner before publishing DONE. Rust never exposes either UnsafeCell.
unsafe impl Send for Callback {}
unsafe impl Sync for Callback {}

unsafe extern "C" fn initialize(context: *mut c_void) {
    let context = unsafe { &*context.cast::<Callback>() };
    context.owner_initialized.store(true, Ordering::Release);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let initializer = unsafe { &mut *context.initializer.get() }
            .take()
            .expect("kcoro invoked an owner service without an initializer");
        unsafe { *context.callback.get() = Some(initializer()) };
    }));
    if let Err(payload) = result {
        context.panicked.store(true, Ordering::Release);
        std::mem::forget(payload);
        let status =
            unsafe { kc_service_complete_current(context.service.load(Ordering::Acquire)) };
        if status != 0 {
            context.reschedule_error.store(status, Ordering::Release);
        }
    }
}

unsafe extern "C" fn retire(context: *mut c_void) {
    let context = unsafe { &*context.cast::<Callback>() };
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop((*context.callback.get()).take());
        drop((*context.initializer.get()).take());
    }));
    if let Err(payload) = result {
        context.panicked.store(true, Ordering::Release);
        std::mem::forget(payload);
    }
    context.owner_retired.store(true, Ordering::Release);
}

unsafe extern "C" fn invoke(context: *mut c_void) {
    let context = unsafe { &*context.cast::<Callback>() };
    if context.panicked.load(Ordering::Acquire) {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let callback = unsafe { &mut *context.callback.get() }
            .as_mut()
            .expect("kcoro invoked an unsealed service");
        callback()
    }));
    match result {
        Ok(ServiceOutcome::Dormant) => {}
        Ok(ServiceOutcome::Continue) => {
            let status = unsafe { kc_service_ready_again(context.service.load(Ordering::Acquire)) };
            if status != 0 {
                context.reschedule_error.store(status, Ordering::Release);
            }
        }
        Ok(ServiceOutcome::Complete) => {
            let status =
                unsafe { kc_service_complete_current(context.service.load(Ordering::Acquire)) };
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
    /// The callback drained its predicate; only a producer edge resumes it.
    Dormant,
    /// Work remains after this invocation's fixed quota; yield and re-enter.
    Continue,
    /// The owned state machine reached a natural terminal edge. Close future
    /// notifications and retire after already-accepted edges drain, without
    /// stopping the shared runtime.
    Complete,
}

struct ServiceLease {
    raw: NonNull<c_void>,
    runtime: Arc<RuntimeInner>,
}

/* Producer edges retain only the native service lifetime, never the callback
 * that may itself own those edges. This separation prevents
 * callback -> notifier -> callback ownership cycles. */
unsafe impl Send for ServiceLease {}
unsafe impl Sync for ServiceLease {}

impl ServiceLease {
    #[inline]
    fn stop(&self) {
        unsafe { kc_service_request_stop(self.raw.as_ptr()) }
    }

    #[inline]
    fn join(&self) -> Result<(), i32> {
        status(unsafe { kc_service_join(self.raw.as_ptr()) })
    }
}

impl Drop for ServiceLease {
    fn drop(&mut self) {
        /* Every native notifier owns an Arc<ServiceLease> and destroys its
         * notifier before releasing that Arc. Therefore the final lease is the
         * proof that kc_service_destroy cannot race a producer edge. */
        let _ = unsafe { kc_service_destroy(self.raw.as_ptr()) };
    }
}

struct ServiceInner {
    lease: Arc<ServiceLease>,
    context: Option<Box<Callback>>,
}

fn realtime_notifier(inner: &Arc<ServiceInner>) -> Result<RealtimeNotifier, i32> {
    let mut raw = std::ptr::null_mut();
    status(unsafe { kc_service_notifier_create(inner.lease.raw.as_ptr(), &mut raw) })?;
    Ok(RealtimeNotifier {
        raw: NonNull::new(raw)
            .expect("kc_service_notifier_create returned success without an object"),
        service: Arc::clone(&inner.lease),
        single: PhantomData,
    })
}

/// Restricted setup-time view of a retained service.
///
/// It deliberately exposes only producer-edge creation: the service cannot be
/// started, notified, stopped, or observed until its callback state is sealed.
pub struct ServiceSetup {
    inner: Arc<ServiceInner>,
}

impl ServiceSetup {
    /// Mint one non-cloneable realtime edge for one producer.
    pub fn realtime_notifier(&self) -> Result<RealtimeNotifier, i32> {
        realtime_notifier(&self.inner)
    }

    /// Mint a cloneable atomics-only MPSC control edge. Each producer publishes
    /// its durable predicate before calling `notify`; notification itself takes
    /// no mutex, allocates nothing, and never invokes the service callback.
    pub fn shared_realtime_notifier(&self) -> SharedRealtimeNotifier {
        SharedRealtimeNotifier {
            service: Arc::clone(&self.inner.lease),
        }
    }
}

// Each native service has one permanent worker owner. Producer edges use only
// atomics; the owner invokes the Send closure serially.
unsafe impl Send for ServiceInner {}
unsafe impl Sync for ServiceInner {}

impl ServiceInner {
    #[inline]
    fn stop(&self) {
        self.lease.stop()
    }

    #[inline]
    fn join(&self) -> Result<(), i32> {
        self.lease.join()
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
            if self.lease.runtime.start().is_err() {
                self.leak();
                return;
            }
            self.stop();
            if self.join().is_err() {
                self.leak();
                return;
            }
        }
        /* The callback can own notifier edges minted during setup. Drop that
         * durable context after terminal acknowledgement. Owner-local task
         * contents were already destroyed by retire() on their fixed worker.
         * If that invariant is ever broken, leak rather than run a !Send
         * destructor on this administrative thread. */
        if self.context.as_ref().is_some_and(|context| {
            context.owner_local
                && context.owner_initialized.load(Ordering::Acquire)
                && !context.owner_retired.load(Ordering::Acquire)
        }) {
            self.leak();
            return;
        }
        drop(self.context.take());
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
        status(unsafe { kc_service_start(self.inner.lease.raw.as_ptr()) })
    }

    /// Send an atomics-only MPSC control notification.
    #[inline]
    pub fn notify(&self) -> Result<(), i32> {
        status(unsafe { kc_service_notify(self.inner.lease.raw.as_ptr()) })
    }

    /// Create a setup-time, retained realtime notification lease.
    pub fn realtime_notifier(&self) -> Result<RealtimeNotifier, i32> {
        realtime_notifier(&self.inner)
    }

    /// Create a cloneable atomics-only MPSC control edge.
    pub fn shared_realtime_notifier(&self) -> SharedRealtimeNotifier {
        SharedRealtimeNotifier {
            service: Arc::clone(&self.inner.lease),
        }
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
    /// a callback may legitimately close admission before `Continue` lands.
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
        status(unsafe { kc_service_snapshot_get(self.inner.lease.raw.as_ptr(), &mut snapshot) })?;
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
    service: Arc<ServiceLease>,
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

/// A cloneable realtime-safe MPSC edge for control and hardware callbacks.
///
/// The edge contains no mutex, waker, timer, or payload cell. Producers write
/// their own durable state first, then atomically make the service owner's
/// fixed inbound bit runnable. The callback always executes on that owner.
#[derive(Clone)]
pub struct SharedRealtimeNotifier {
    service: Arc<ServiceLease>,
}

impl SharedRealtimeNotifier {
    #[inline]
    pub fn notify(&self) -> Result<(), i32> {
        status(unsafe { kc_service_notify(self.service.raw.as_ptr()) })
    }
}

/// Preserve the explicit link anchor used by low-level ABI conformance tests.
#[inline(always)]
pub fn link_anchor() {}
