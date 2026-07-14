use std::cell::UnsafeCell;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

const PENDING: u8 = 0;
const WRITING: u8 = 1;
const READY: u8 = 2;

struct Inner<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
    waker: Mutex<Option<Waker>>,
    wait: Mutex<()>,
    ready: Condvar,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send + Sync> Sync for Inner<T> {}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        if *self.state.get_mut() == READY {
            // SAFETY: READY is published only after the sole writer initialized value.
            unsafe { self.value.get_mut().assume_init_drop() };
        }
    }
}

pub struct Promise<T> {
    inner: Arc<Inner<T>>,
}

#[derive(Clone)]
pub struct Resolver<T> {
    inner: Arc<Inner<T>>,
}

pub fn promise<T>() -> (Promise<T>, Resolver<T>) {
    let inner = Arc::new(Inner {
        state: AtomicU8::new(PENDING),
        value: UnsafeCell::new(MaybeUninit::uninit()),
        waker: Mutex::new(None),
        wait: Mutex::new(()),
        ready: Condvar::new(),
    });
    (
        Promise {
            inner: inner.clone(),
        },
        Resolver { inner },
    )
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn take_waker(mutex: &Mutex<Option<Waker>>) -> Option<Waker> {
    lock(mutex).take()
}

impl<T> Resolver<T> {
    /// Claims the terminal edge. Exactly one competing resolver can succeed.
    pub fn try_resolve(&self, value: T) -> Result<(), T> {
        if self
            .inner
            .state
            .compare_exchange(PENDING, WRITING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(value);
        }

        // The wait mutex closes the check-before-sleep gap for blocking consumers.
        let wait = lock(&self.inner.wait);
        // SAFETY: the PENDING -> WRITING claim grants this resolver sole write access.
        unsafe { (*self.inner.value.get()).write(value) };
        self.inner.state.store(READY, Ordering::Release);
        let waker = take_waker(&self.inner.waker);
        self.inner.ready.notify_all();
        drop(wait);
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }
}

impl<T: Clone> Promise<T> {
    pub fn is_ready(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) == READY
    }

    pub fn wait(&self) -> T {
        let mut guard = lock(&self.inner.wait);
        loop {
            if let Some(value) = self.read() {
                return value;
            }
            guard = self
                .inner
                .ready
                .wait(guard)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    fn read(&self) -> Option<T> {
        if self.inner.state.load(Ordering::Acquire) != READY {
            return None;
        }
        // SAFETY: acquire observed READY after the sole writer initialized value;
        // value is immutable for the remainder of the shared inner's lifetime.
        Some(unsafe { (&*self.inner.value.get()).assume_init_ref().clone() })
    }
}

impl<T: Clone> Future for Promise<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(value) = self.read() {
            return Poll::Ready(value);
        }

        let mut slot = lock(&self.inner.waker);
        if let Some(value) = self.read() {
            return Poll::Ready(value);
        }
        if slot
            .as_ref()
            .is_none_or(|registered| !registered.will_wake(cx.waker()))
        {
            *slot = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}
