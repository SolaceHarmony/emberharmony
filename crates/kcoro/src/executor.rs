use crate::promise::{promise, Promise, Resolver};
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle};

const ACTIVE: u8 = 1 << 0;
const SCHEDULED: u8 = 1 << 1;
const RUNNING: u8 = 1 << 2;
const NOTIFIED: u8 = 1 << 3;

type Task = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

#[derive(Clone, Debug)]
pub struct Config {
    pub workers: usize,
    pub capacity: usize,
    pub drain_limit: usize,
    pub thread_name: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 1,
            capacity: 256,
            drain_limit: 64,
            thread_name: "kcoro".to_string(),
        }
    }
}

#[derive(Debug)]
pub enum CreateError {
    ZeroWorkers,
    ZeroCapacity,
    ZeroDrainLimit,
    CapacityTooLarge,
    Thread(std::io::Error),
}

impl fmt::Display for CreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWorkers => write!(f, "kcoro requires at least one worker"),
            Self::ZeroCapacity => write!(f, "kcoro requires at least one task slot"),
            Self::ZeroDrainLimit => write!(f, "kcoro drain limit must be nonzero"),
            Self::CapacityTooLarge => write!(f, "kcoro task capacity exceeds the task ID ABI"),
            Self::Thread(error) => write!(f, "failed to start kcoro worker: {error}"),
        }
    }
}

impl std::error::Error for CreateError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpawnError {
    Stopped,
    Full,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinError {
    Running,
    WorkerPanicked,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskResult {
    Completed,
    Stopped,
    Panicked,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskId {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    pub spawned: u64,
    pub polls: u64,
    pub wakes: u64,
    pub completions: u64,
    pub panics: u64,
}

struct Counters {
    spawned: AtomicU64,
    polls: AtomicU64,
    wakes: AtomicU64,
    completions: AtomicU64,
    panics: AtomicU64,
}

struct Body {
    future: Option<Task>,
    resolver: Option<Resolver<TaskResult>>,
    waker: Option<Waker>,
}

struct Slot {
    generation: AtomicU32,
    state: AtomicU8,
    body: Mutex<Body>,
}

#[derive(Clone, Copy)]
struct Key {
    slot: u32,
    generation: u32,
}

struct Inner {
    slots: Box<[Slot]>,
    lifecycle: Mutex<()>,
    free: Mutex<Vec<usize>>,
    ready: Mutex<VecDeque<Key>>,
    work: Condvar,
    stop: AtomicBool,
    fault: AtomicBool,
    drain_limit: usize,
    counters: Counters,
}

struct TaskWake {
    inner: Weak<Inner>,
    key: Key,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

impl TaskWake {
    fn schedule(&self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.schedule(self.key);
        }
    }
}

#[derive(Clone)]
pub struct Handle {
    inner: Arc<Inner>,
}

pub struct TaskHandle {
    id: TaskId,
    done: Promise<TaskResult>,
}

impl TaskHandle {
    pub fn id(&self) -> TaskId {
        self.id
    }

    pub fn wait(&self) -> TaskResult {
        self.done.wait()
    }
}

impl Future for TaskHandle {
    type Output = TaskResult;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.done).poll(cx)
    }
}

pub struct Executor {
    inner: Arc<Inner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

impl Executor {
    pub fn new(config: Config) -> Result<Self, CreateError> {
        if config.workers == 0 {
            return Err(CreateError::ZeroWorkers);
        }
        if config.capacity == 0 {
            return Err(CreateError::ZeroCapacity);
        }
        if config.drain_limit == 0 {
            return Err(CreateError::ZeroDrainLimit);
        }
        if config.capacity > u32::MAX as usize {
            return Err(CreateError::CapacityTooLarge);
        }

        let slots = (0..config.capacity)
            .map(|_| Slot {
                generation: AtomicU32::new(0),
                state: AtomicU8::new(0),
                body: Mutex::new(Body {
                    future: None,
                    resolver: None,
                    waker: None,
                }),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let free = (0..config.capacity).rev().collect::<Vec<_>>();
        let inner = Arc::new(Inner {
            slots,
            lifecycle: Mutex::new(()),
            free: Mutex::new(free),
            ready: Mutex::new(VecDeque::with_capacity(config.capacity)),
            work: Condvar::new(),
            stop: AtomicBool::new(false),
            fault: AtomicBool::new(false),
            drain_limit: config.drain_limit,
            counters: Counters {
                spawned: AtomicU64::new(0),
                polls: AtomicU64::new(0),
                wakes: AtomicU64::new(0),
                completions: AtomicU64::new(0),
                panics: AtomicU64::new(0),
            },
        });

        let mut workers = Vec::with_capacity(config.workers);
        for index in 0..config.workers {
            let worker = inner.clone();
            let name = format!("{}-{index}", config.thread_name);
            match thread::Builder::new()
                .name(name)
                .spawn(move || worker.run())
            {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    inner.stop.store(true, Ordering::Release);
                    inner.work.notify_all();
                    for handle in workers {
                        let _ = handle.join();
                    }
                    return Err(CreateError::Thread(error));
                }
            }
        }

        Ok(Self {
            inner,
            workers: Mutex::new(workers),
        })
    }

    pub fn handle(&self) -> Handle {
        Handle {
            inner: self.inner.clone(),
        }
    }

    pub fn spawn<F>(&self, future: F) -> Result<TaskHandle, SpawnError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.handle().spawn(future)
    }

    pub fn request_stop(&self) {
        let lifecycle = lock(&self.inner.lifecycle);
        self.inner.stop.store(true, Ordering::Release);
        self.inner.work.notify_all();
        drop(lifecycle);
    }

    pub fn join(&self) -> Result<(), JoinError> {
        if !self.inner.stop.load(Ordering::Acquire) {
            return Err(JoinError::Running);
        }
        // Keep the lifecycle lock through worker joins and the final slot sweep.
        // A second join caller must not drop futures while the first still has a
        // worker polling one of them.
        let mut workers = lock(&self.workers);
        let handles = std::mem::take(&mut *workers);
        let mut panicked = false;
        for handle in handles {
            panicked |= handle.join().is_err();
        }
        self.inner.stop_pending();
        if panicked {
            return Err(JoinError::WorkerPanicked);
        }
        if self.inner.fault.load(Ordering::Acquire) {
            return Err(JoinError::Faulted);
        }
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.request_stop();
        let _ = self.join();
    }
}

impl Handle {
    pub fn spawn<F>(&self, future: F) -> Result<TaskHandle, SpawnError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let lifecycle = lock(&self.inner.lifecycle);
        if self.inner.stop.load(Ordering::Acquire) {
            return Err(SpawnError::Stopped);
        }
        if self.inner.fault.load(Ordering::Acquire) {
            return Err(SpawnError::Faulted);
        }
        let slot_index = lock(&self.inner.free).pop().ok_or(SpawnError::Full)?;
        let slot = &self.inner.slots[slot_index];
        let generation = next_generation(&slot.generation);
        let (done, resolver) = promise();
        let key = Key {
            slot: slot_index as u32,
            generation,
        };
        let waker = Waker::from(Arc::new(TaskWake {
            inner: Arc::downgrade(&self.inner),
            key,
        }));
        {
            let mut body = lock(&slot.body);
            body.future = Some(Box::pin(future));
            body.resolver = Some(resolver);
            body.waker = Some(waker);
        }
        slot.state.store(ACTIVE | SCHEDULED, Ordering::Release);
        self.inner.counters.spawned.fetch_add(1, Ordering::Relaxed);
        if !self.inner.enqueue(key) {
            self.inner.stop.store(true, Ordering::Release);
            self.inner.work.notify_all();
            return Err(SpawnError::Faulted);
        }
        drop(lifecycle);
        Ok(TaskHandle {
            id: TaskId {
                slot: key.slot,
                generation,
            },
            done,
        })
    }
}

impl Inner {
    fn run(self: Arc<Self>) {
        loop {
            let Some(first) = self.wait_for_work() else {
                return;
            };
            self.poll(first);
            for _ in 1..self.drain_limit {
                if self.stop.load(Ordering::Acquire) {
                    return;
                }
                let Some(next) = lock(&self.ready).pop_front() else {
                    break;
                };
                self.poll(next);
            }
            if !lock(&self.ready).is_empty() {
                self.work.notify_one();
                thread::yield_now();
            }
        }
    }

    fn wait_for_work(&self) -> Option<Key> {
        let mut ready = lock(&self.ready);
        loop {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if let Some(key) = ready.pop_front() {
                return Some(key);
            }
            ready = self
                .work
                .wait(ready)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn enqueue(&self, key: Key) -> bool {
        let mut ready = lock(&self.ready);
        if ready.len() == self.slots.len() {
            self.fault.store(true, Ordering::Release);
            return false;
        }
        ready.push_back(key);
        drop(ready);
        self.work.notify_one();
        true
    }

    fn schedule(&self, key: Key) {
        if self.stop.load(Ordering::Acquire) {
            return;
        }
        let Some(slot) = self.slots.get(key.slot as usize) else {
            return;
        };
        if slot.generation.load(Ordering::Acquire) != key.generation {
            return;
        }
        let mut state = slot.state.load(Ordering::Acquire);
        loop {
            if state & ACTIVE == 0 {
                return;
            }
            if state & RUNNING != 0 {
                if state & NOTIFIED != 0 {
                    return;
                }
                match slot.state.compare_exchange_weak(
                    state,
                    state | NOTIFIED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.counters.wakes.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(actual) => state = actual,
                }
                continue;
            }
            if state & SCHEDULED != 0 {
                return;
            }
            match slot.state.compare_exchange_weak(
                state,
                state | SCHEDULED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.counters.wakes.fetch_add(1, Ordering::Relaxed);
                    if !self.enqueue(key) {
                        self.stop.store(true, Ordering::Release);
                        self.work.notify_all();
                    }
                    return;
                }
                Err(actual) => state = actual,
            }
        }
    }

    fn begin_poll(&self, key: Key) -> Option<&Slot> {
        let slot = self.slots.get(key.slot as usize)?;
        if slot.generation.load(Ordering::Acquire) != key.generation {
            return None;
        }
        let mut state = slot.state.load(Ordering::Acquire);
        loop {
            if state & (ACTIVE | SCHEDULED) != ACTIVE | SCHEDULED || state & RUNNING != 0 {
                return None;
            }
            let next = (state & !SCHEDULED) | RUNNING;
            match slot
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some(slot),
                Err(actual) => state = actual,
            }
        }
    }

    fn poll(self: &Arc<Self>, key: Key) {
        let Some(slot) = self.begin_poll(key) else {
            return;
        };
        self.counters.polls.fetch_add(1, Ordering::Relaxed);
        let waker = lock(&slot.body)
            .waker
            .as_ref()
            .expect("active kcoro slot without a waker")
            .clone();
        let mut context = Context::from_waker(&waker);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut body = lock(&slot.body);
            body.future
                .as_mut()
                .expect("active kcoro slot without a future")
                .as_mut()
                .poll(&mut context)
        }));
        match result {
            Ok(Poll::Ready(())) => self.finish(slot, key.slot as usize, TaskResult::Completed),
            Ok(Poll::Pending) => self.park(slot, key),
            Err(_) => {
                self.counters.panics.fetch_add(1, Ordering::Relaxed);
                self.finish(slot, key.slot as usize, TaskResult::Panicked);
            }
        }
    }

    fn park(&self, slot: &Slot, key: Key) {
        let mut state = slot.state.load(Ordering::Acquire);
        loop {
            if state & ACTIVE == 0 {
                return;
            }
            let notified = state & NOTIFIED != 0;
            let next = if notified { ACTIVE | SCHEDULED } else { ACTIVE };
            match slot
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    if notified && !self.enqueue(key) {
                        self.stop.store(true, Ordering::Release);
                        self.work.notify_all();
                    }
                    return;
                }
                Err(actual) => state = actual,
            }
        }
    }

    fn finish(&self, slot: &Slot, index: usize, result: TaskResult) {
        let resolver = {
            let mut body = lock(&slot.body);
            body.future.take();
            body.waker.take();
            body.resolver.take()
        };
        slot.state.store(0, Ordering::Release);
        if let Some(resolver) = resolver {
            let _ = resolver.try_resolve(result);
        }
        lock(&self.free).push(index);
        self.counters.completions.fetch_add(1, Ordering::Relaxed);
    }

    fn stop_pending(&self) {
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.state.load(Ordering::Acquire) & ACTIVE == 0 {
                continue;
            }
            let resolver = {
                let mut body = lock(&slot.body);
                body.future.take();
                body.waker.take();
                body.resolver.take()
            };
            slot.state.store(0, Ordering::Release);
            if let Some(resolver) = resolver {
                let _ = resolver.try_resolve(TaskResult::Stopped);
            }
            lock(&self.free).push(index);
        }
        lock(&self.ready).clear();
    }

    fn stats(&self) -> Stats {
        Stats {
            spawned: self.counters.spawned.load(Ordering::Relaxed),
            polls: self.counters.polls.load(Ordering::Relaxed),
            wakes: self.counters.wakes.load(Ordering::Relaxed),
            completions: self.counters.completions.load(Ordering::Relaxed),
            panics: self.counters.panics.load(Ordering::Relaxed),
        }
    }
}

fn next_generation(generation: &AtomicU32) -> u32 {
    let mut current = generation.load(Ordering::Acquire);
    loop {
        let next = match current.wrapping_add(1) {
            0 => 1,
            value => value,
        };
        match generation.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return next,
            Err(actual) => current = actual,
        }
    }
}
