//! Callback-driven policy broker for the resident native engine.
//!
//! The broker is the sole native SQ producer. A dedicated ingress thread is the
//! sole native CQ consumer and wakes the broker continuation through one exact edge.
//! Payloads stay behind generation-protected native descriptors; only fixed 128-byte
//! control records cross this boundary.

use kcoro::{
    ring, Completion, Executor, ExecutorConfig, Receiver, Sender, Submission, TaskHandle,
    TaskResult, TicketId, TrySendError,
};
use std::ffi::c_void;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;

const FREE: u8 = 0;
const QUEUED: u8 = 1;
const PENDING: u8 = 2;
const COMPLETING: u8 = 3;
const READY: u8 = 4;

extern "C" {
    fn lfm_kernel_bridge_submit(bridge: *mut c_void, submission: *const Submission) -> i32;
    fn lfm_kernel_bridge_wait_completion(
        bridge: *mut c_void,
        completion: *mut Completion,
        deadline_ns: u64,
    ) -> i32;
    fn lfm_kernel_bridge_request_stop(bridge: *mut c_void);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Key {
    slot: u32,
    generation: u32,
}

#[derive(Clone, Copy)]
struct Pending {
    key: Key,
    ticket: TicketId,
    conversation_id: u64,
    epoch: u64,
}

struct Body {
    submission: Option<Submission>,
    result: Option<Result<Completion, i32>>,
}

struct Slot {
    generation: AtomicU32,
    phase: AtomicU8,
    body: Mutex<Body>,
    ready: Condvar,
}

struct Counters {
    admitted: AtomicU64,
    native_submissions: AtomicU64,
    native_completions: AtomicU64,
    resolved: AtomicU64,
    failed: AtomicU64,
    edge_signals: AtomicU64,
    live: AtomicUsize,
    max_generation: AtomicU32,
}

struct Edge {
    generation: AtomicU64,
    waker: Mutex<Option<Waker>>,
}

impl Edge {
    fn snapshot(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn signal(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        if let Some(waker) = lock(&self.waker).take() {
            waker.wake();
        }
    }

    fn wait(&self, expected: u64) -> EdgeWait<'_> {
        EdgeWait {
            edge: self,
            expected,
        }
    }
}

struct EdgeWait<'a> {
    edge: &'a Edge,
    expected: u64,
}

impl Future for EdgeWait<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.edge.snapshot() != self.expected {
            return Poll::Ready(());
        }

        let mut slot = lock(&self.edge.waker);
        if self.edge.snapshot() != self.expected {
            return Poll::Ready(());
        }
        if slot
            .as_ref()
            .is_none_or(|registered| !registered.will_wake(context.waker()))
        {
            *slot = Some(context.waker().clone());
        }
        if self.edge.snapshot() != self.expected {
            slot.take();
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

struct Inner {
    bridge: usize,
    slots: Box<[Slot]>,
    free: Mutex<Vec<usize>>,
    sender: Mutex<Option<Sender<Key>>>,
    pending: Mutex<Option<Pending>>,
    edge: Edge,
    stopping: AtomicBool,
    fault: AtomicBool,
    counters: Counters,
}

impl Inner {
    fn bridge(&self) -> *mut c_void {
        self.bridge as *mut c_void
    }

    fn request_stop(&self) {
        self.stopping.store(true, Ordering::Release);
        lock(&self.sender).take();
        // SAFETY: the native engine owns the bridge until Coordinator::shutdown joins
        // both endpoint owners. Stop is idempotent and wakes both native doorbells.
        unsafe { lfm_kernel_bridge_request_stop(self.bridge()) };
    }

    fn acquire(&self, submission: Submission) -> Result<Key, i32> {
        if self.stopping.load(Ordering::Acquire) {
            return Err(-libc::ECANCELED);
        }

        let (index, generation) = {
            let mut free = lock(&self.free);
            loop {
                let index = free.pop().ok_or(-libc::EAGAIN)?;
                let slot = &self.slots[index];
                if slot.phase.load(Ordering::Acquire) != FREE {
                    self.fault.store(true, Ordering::Release);
                    return Err(-libc::EFAULT);
                }
                let current = slot.generation.load(Ordering::Acquire);
                if current == u32::MAX {
                    continue;
                }
                let generation = current + 1;
                slot.generation.store(generation, Ordering::Release);
                break (index, generation);
            }
        };

        let slot = &self.slots[index];
        {
            let mut body = lock(&slot.body);
            body.submission = Some(submission);
            body.result = None;
        }
        slot.phase.store(QUEUED, Ordering::Release);
        self.counters.live.fetch_add(1, Ordering::Relaxed);
        self.counters
            .max_generation
            .fetch_max(generation, Ordering::Relaxed);
        Ok(Key {
            slot: index as u32,
            generation,
        })
    }

    fn abandon(&self, key: Key) {
        let Some(slot) = self.slots.get(key.slot as usize) else {
            return;
        };
        if slot.generation.load(Ordering::Acquire) != key.generation {
            return;
        }
        {
            let mut body = lock(&slot.body);
            body.submission = None;
            body.result = None;
        }
        slot.phase.store(FREE, Ordering::Release);
        self.recycle(key);
    }

    fn recycle(&self, key: Key) {
        self.counters.live.fetch_sub(1, Ordering::Relaxed);
        if key.generation != u32::MAX {
            lock(&self.free).push(key.slot as usize);
        }
    }

    fn begin(&self, key: Key) -> Option<Submission> {
        let slot = self.slots.get(key.slot as usize)?;
        if slot.generation.load(Ordering::Acquire) != key.generation {
            return None;
        }
        if slot
            .phase
            .compare_exchange(QUEUED, PENDING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        lock(&slot.body).submission
    }

    fn finish(&self, key: Key, result: Result<Completion, i32>) -> bool {
        let Some(slot) = self.slots.get(key.slot as usize) else {
            return false;
        };
        if slot.generation.load(Ordering::Acquire) != key.generation {
            return false;
        }

        let mut phase = slot.phase.load(Ordering::Acquire);
        loop {
            if phase != QUEUED && phase != PENDING {
                return false;
            }
            match slot.phase.compare_exchange_weak(
                phase,
                COMPLETING,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => phase = actual,
            }
        }

        let failed = result.is_err();
        lock(&slot.body).result = Some(result);
        slot.phase.store(READY, Ordering::Release);
        self.counters.resolved.fetch_add(1, Ordering::Relaxed);
        if failed {
            self.counters.failed.fetch_add(1, Ordering::Relaxed);
        }
        slot.ready.notify_one();
        true
    }

    fn finish_all(&self, error: i32) {
        for (index, slot) in self.slots.iter().enumerate() {
            let phase = slot.phase.load(Ordering::Acquire);
            if phase != QUEUED && phase != PENDING {
                continue;
            }
            let key = Key {
                slot: index as u32,
                generation: slot.generation.load(Ordering::Acquire),
            };
            self.finish(key, Err(error));
        }
    }

    fn install_pending(&self, pending: Pending) -> bool {
        let mut slot = lock(&self.pending);
        if slot.is_some() {
            return false;
        }
        *slot = Some(pending);
        true
    }

    fn take_pending(&self) -> Option<Pending> {
        lock(&self.pending).take()
    }

    fn take_pending_if(&self, key: Key) -> Option<Pending> {
        let mut slot = lock(&self.pending);
        if slot.as_ref().is_some_and(|pending| pending.key == key) {
            return slot.take();
        }
        None
    }

    fn submit(&self, submission: Submission) -> Result<Completion, i32> {
        if !submission.is_compatible() {
            return Err(-libc::EINVAL);
        }
        let key = self.acquire(submission)?;
        let send = {
            let mut sender = lock(&self.sender);
            match sender.as_mut() {
                Some(sender) => sender.try_send(key),
                None => Err(TrySendError::Closed(key)),
            }
        };
        if let Err(error) = send {
            let rc = match error {
                TrySendError::Full(_) => -libc::EAGAIN,
                TrySendError::Closed(_) => -libc::ECANCELED,
            };
            self.abandon(key);
            return Err(rc);
        }
        self.counters.admitted.fetch_add(1, Ordering::Relaxed);

        let slot = &self.slots[key.slot as usize];
        let mut body = lock(&slot.body);
        while body.result.is_none() {
            body = slot
                .ready
                .wait(body)
                .unwrap_or_else(|error| error.into_inner());
        }
        let result = body.result.take().expect("resolved coordinator slot");
        body.submission = None;
        drop(body);
        slot.phase.store(FREE, Ordering::Release);
        self.recycle(key);
        result
    }
}

async fn broker(inner: Arc<Inner>, mut receiver: Receiver<Key>) {
    while let Ok(key) = receiver.recv().await {
        if inner.stopping.load(Ordering::Acquire) {
            inner.finish(key, Err(-libc::ECANCELED));
            continue;
        }
        let Some(submission) = inner.begin(key) else {
            continue;
        };
        let expected = inner.edge.snapshot();
        let pending = Pending {
            key,
            ticket: submission.ticket,
            conversation_id: submission.conversation_id,
            epoch: submission.epoch,
        };
        if !inner.install_pending(pending) {
            inner.finish(key, Err(-libc::EBUSY));
            inner.fault.store(true, Ordering::Release);
            inner.request_stop();
            continue;
        }

        // SAFETY: the engine keeps the bridge alive until the broker and ingress
        // owners are joined. `submission` is an aligned, ABI-sized inline record.
        let rc = unsafe { lfm_kernel_bridge_submit(inner.bridge(), &submission) };
        if rc != 0 {
            if inner.take_pending_if(key).is_some() {
                inner.finish(key, Err(rc));
            }
            continue;
        }
        inner
            .counters
            .native_submissions
            .fetch_add(1, Ordering::Relaxed);
        inner.edge.wait(expected).await;
    }
}

fn ingress(inner: Arc<Inner>) {
    loop {
        let mut completion = MaybeUninit::<Completion>::zeroed();
        // SAFETY: the bridge remains alive until this endpoint owner is joined. The
        // destination has the C ABI's 64-byte alignment and full 128-byte extent.
        let rc = unsafe {
            lfm_kernel_bridge_wait_completion(inner.bridge(), completion.as_mut_ptr(), 0)
        };
        if rc == 0 {
            // SAFETY: rc == 0 means the native bridge initialized the full record.
            let completion = unsafe { completion.assume_init() };
            inner
                .counters
                .native_completions
                .fetch_add(1, Ordering::Relaxed);
            let Some(pending) = inner.take_pending() else {
                inner.fault.store(true, Ordering::Release);
                inner.request_stop();
                inner.edge.signal();
                inner.counters.edge_signals.fetch_add(1, Ordering::Relaxed);
                return;
            };
            let exact = completion.is_compatible()
                && completion.ticket == pending.ticket
                && completion.conversation_id == pending.conversation_id
                && completion.epoch == pending.epoch;
            let result = if exact {
                Ok(completion)
            } else {
                Err(-libc::ESTALE)
            };
            inner.finish(pending.key, result);
            inner.edge.signal();
            inner.counters.edge_signals.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let error = if rc < 0 { rc } else { -libc::EIO };
        if let Some(pending) = inner.take_pending() {
            inner.finish(pending.key, Err(error));
            inner.edge.signal();
            inner.counters.edge_signals.fetch_add(1, Ordering::Relaxed);
        }
        if rc != -libc::ECANCELED {
            inner.fault.store(true, Ordering::Release);
        }
        inner.request_stop();
        return;
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Snapshot {
    pub admitted: u64,
    pub native_submissions: u64,
    pub native_completions: u64,
    pub resolved: u64,
    pub failed: u64,
    pub edge_signals: u64,
    pub live: usize,
    pub max_generation: u32,
    pub executor_polls: u64,
    pub executor_wakes: u64,
    pub fault: bool,
}

pub(super) struct Coordinator {
    inner: Arc<Inner>,
    executor: Executor,
    broker: Option<TaskHandle>,
    ingress: Option<JoinHandle<()>>,
    closed: bool,
}

impl Coordinator {
    pub(super) fn new(bridge: *mut c_void, capacity: usize) -> Result<Self, String> {
        if bridge.is_null() {
            return Err("native bridge pointer is null".to_string());
        }
        let (sender, receiver) = ring(capacity).map_err(|_| "coordinator capacity is zero")?;
        let slots = (0..capacity)
            .map(|_| Slot {
                generation: AtomicU32::new(0),
                phase: AtomicU8::new(FREE),
                body: Mutex::new(Body {
                    submission: None,
                    result: None,
                }),
                ready: Condvar::new(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let inner = Arc::new(Inner {
            bridge: bridge as usize,
            slots,
            free: Mutex::new((0..capacity).rev().collect()),
            sender: Mutex::new(Some(sender)),
            pending: Mutex::new(None),
            edge: Edge {
                generation: AtomicU64::new(0),
                waker: Mutex::new(None),
            },
            stopping: AtomicBool::new(false),
            fault: AtomicBool::new(false),
            counters: Counters {
                admitted: AtomicU64::new(0),
                native_submissions: AtomicU64::new(0),
                native_completions: AtomicU64::new(0),
                resolved: AtomicU64::new(0),
                failed: AtomicU64::new(0),
                edge_signals: AtomicU64::new(0),
                live: AtomicUsize::new(0),
                max_generation: AtomicU32::new(0),
            },
        });
        let executor = Executor::new(ExecutorConfig {
            workers: 1,
            capacity: 2,
            drain_limit: 32,
            thread_name: "kcoro-kernel".to_string(),
        })
        .map_err(|error| error.to_string())?;
        let task = executor
            .spawn(broker(inner.clone(), receiver))
            .map_err(|error| format!("failed to start coordinator broker: {error:?}"))?;
        let ingress = match std::thread::Builder::new()
            .name("kcoro-cq".to_string())
            .spawn({
                let inner = inner.clone();
                move || ingress(inner)
            }) {
            Ok(handle) => handle,
            Err(error) => {
                inner.request_stop();
                let _ = task.wait();
                executor.request_stop();
                let _ = executor.join();
                return Err(format!("failed to start coordinator ingress: {error}"));
            }
        };

        Ok(Self {
            inner,
            executor,
            broker: Some(task),
            ingress: Some(ingress),
            closed: false,
        })
    }

    pub(super) fn context(&self) -> *mut c_void {
        Arc::as_ptr(&self.inner) as *mut c_void
    }

    pub(super) fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.inner.request_stop();
        if self
            .ingress
            .take()
            .is_some_and(|handle| handle.join().is_err())
        {
            self.inner.fault.store(true, Ordering::Release);
        }
        self.inner.finish_all(-libc::ECANCELED);
        self.inner.edge.signal();
        if self
            .broker
            .take()
            .is_some_and(|task| task.wait() != TaskResult::Completed)
        {
            self.inner.fault.store(true, Ordering::Release);
        }
        self.executor.request_stop();
        if self.executor.join().is_err() {
            self.inner.fault.store(true, Ordering::Release);
        }
        self.closed = true;
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> Snapshot {
        let executor = self.executor.stats();
        Snapshot {
            admitted: self.inner.counters.admitted.load(Ordering::Relaxed),
            native_submissions: self
                .inner
                .counters
                .native_submissions
                .load(Ordering::Relaxed),
            native_completions: self
                .inner
                .counters
                .native_completions
                .load(Ordering::Relaxed),
            resolved: self.inner.counters.resolved.load(Ordering::Relaxed),
            failed: self.inner.counters.failed.load(Ordering::Relaxed),
            edge_signals: self.inner.counters.edge_signals.load(Ordering::Relaxed),
            live: self.inner.counters.live.load(Ordering::Relaxed),
            max_generation: self.inner.counters.max_generation.load(Ordering::Relaxed),
            executor_polls: executor.polls,
            executor_wakes: executor.wakes,
            fault: self.inner.fault.load(Ordering::Acquire),
        }
    }
}

impl Drop for Coordinator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub(super) unsafe extern "C" fn submit(
    context: *mut c_void,
    submission: *const Submission,
    completion: *mut Completion,
) -> i32 {
    if context.is_null() || submission.is_null() || completion.is_null() {
        return -libc::EINVAL;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the registered context is Arc::as_ptr(&Coordinator::inner), and the
        // engine clears the callback before that Arc can be dropped.
        let inner = unsafe { &*context.cast::<Inner>() };
        // SAFETY: C++ passes an aligned, live KcSubmissionV1 for this blocking call.
        inner.submit(unsafe { *submission })
    }));
    match result {
        Ok(Ok(value)) => {
            // SAFETY: validated non-null output owned by the blocking C++ caller.
            unsafe { completion.write(value) };
            0
        }
        Ok(Err(error)) => error,
        Err(_) => -libc::EFAULT,
    }
}
