use std::cell::Cell;
use std::cell::UnsafeCell;
use std::future::Future;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

// Apple Silicon, including an x86_64 binary running under Rosetta, has a
// 128-byte destructive-interference line.  Keep the producer and consumer
// cursors in distinct lines on every supported target; over-aligning on a
// 64-byte x86 cache is harmless and avoids keying this layout off the ISA.
#[repr(align(128))]
struct CacheLine<T>(T);

// Keep independently produced/consumed cells on distinct Apple cache lines
// without changing the ABI alignment of the value stored inside the ring.
#[repr(align(128))]
struct Slot<T>(UnsafeCell<MaybeUninit<T>>);

struct Inner<T> {
    cells: Box<[Slot<T>]>,
    capacity: usize,
    head: CacheLine<AtomicUsize>,
    tail: CacheLine<AtomicUsize>,
    sender_open: AtomicBool,
    receiver_open: AtomicBool,
    sender_waker: Mutex<Option<Waker>>,
    receiver_waker: Mutex<Option<Waker>>,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let mut head = *self.head.0.get_mut();
        let tail = *self.tail.0.get_mut();
        while head != tail {
            let index = head % self.capacity;
            // SAFETY: cells in [head, tail) were initialized and not consumed.
            unsafe { (*self.cells[index].0.get()).assume_init_drop() };
            head = head.wrapping_add(1);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RingError {
    ZeroCapacity,
}

#[derive(Debug, Eq, PartialEq)]
pub enum TrySendError<T> {
    Full(T),
    Closed(T),
}

impl<T> TrySendError<T> {
    pub fn into_inner(self) -> T {
        match self {
            Self::Full(value) | Self::Closed(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryRecvError {
    Empty,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecvError;

pub struct Sender<T> {
    inner: Arc<Inner<T>>,
    single: PhantomData<Cell<()>>,
}

pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
    single: PhantomData<Cell<()>>,
}

pub fn ring<T>(capacity: usize) -> Result<(Sender<T>, Receiver<T>), RingError> {
    if capacity == 0 {
        return Err(RingError::ZeroCapacity);
    }
    let cells = (0..capacity)
        .map(|_| Slot(UnsafeCell::new(MaybeUninit::uninit())))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let inner = Arc::new(Inner {
        cells,
        capacity,
        head: CacheLine(AtomicUsize::new(0)),
        tail: CacheLine(AtomicUsize::new(0)),
        sender_open: AtomicBool::new(true),
        receiver_open: AtomicBool::new(true),
        sender_waker: Mutex::new(None),
        receiver_waker: Mutex::new(None),
    });
    Ok((
        Sender {
            inner: inner.clone(),
            single: PhantomData,
        },
        Receiver {
            inner,
            single: PhantomData,
        },
    ))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn take_waker(mutex: &Mutex<Option<Waker>>) -> Option<Waker> {
    lock(mutex).take()
}

fn register(slot: &mut Option<Waker>, waker: &Waker) {
    if slot
        .as_ref()
        .is_none_or(|registered| !registered.will_wake(waker))
    {
        *slot = Some(waker.clone());
    }
}

impl<T> Sender<T> {
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        let head = self.inner.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_send(&mut self, value: T) -> Result<(), TrySendError<T>> {
        if !self.inner.receiver_open.load(Ordering::Acquire) {
            return Err(TrySendError::Closed(value));
        }
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        let head = self.inner.head.0.load(Ordering::Acquire);
        if tail.wrapping_sub(head) == self.inner.capacity {
            return Err(TrySendError::Full(value));
        }
        let index = tail % self.inner.capacity;
        // SAFETY: SPSC ownership gives the sole sender this unoccupied cell.
        unsafe { (*self.inner.cells[index].0.get()).write(value) };
        self.inner
            .tail
            .0
            .store(tail.wrapping_add(1), Ordering::Release);
        if let Some(waker) = take_waker(&self.inner.receiver_waker) {
            waker.wake();
        }
        Ok(())
    }

    pub fn send(&mut self, value: T) -> SendFuture<'_, T> {
        SendFuture {
            sender: self,
            value: Some(value),
        }
    }

    fn has_space(&self) -> bool {
        let tail = self.inner.tail.0.load(Ordering::Relaxed);
        let head = self.inner.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head) < self.inner.capacity
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.inner.sender_open.store(false, Ordering::Release);
        if let Some(waker) = take_waker(&self.inner.receiver_waker) {
            waker.wake();
        }
    }
}

impl<T> Receiver<T> {
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        let head = self.inner.head.0.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        if head == tail {
            if self.inner.sender_open.load(Ordering::Acquire) {
                return Err(TryRecvError::Empty);
            }
            return Err(TryRecvError::Closed);
        }
        let index = head % self.inner.capacity;
        // SAFETY: SPSC ownership gives the sole receiver this initialized cell.
        let value = unsafe { (*self.inner.cells[index].0.get()).assume_init_read() };
        self.inner
            .head
            .0
            .store(head.wrapping_add(1), Ordering::Release);
        if let Some(waker) = take_waker(&self.inner.sender_waker) {
            waker.wake();
        }
        Ok(value)
    }

    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture { receiver: self }
    }

    fn has_value(&self) -> bool {
        let head = self.inner.head.0.load(Ordering::Relaxed);
        let tail = self.inner.tail.0.load(Ordering::Acquire);
        head != tail
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.receiver_open.store(false, Ordering::Release);
        if let Some(waker) = take_waker(&self.inner.sender_waker) {
            waker.wake();
        }
    }
}

pub struct SendFuture<'a, T> {
    sender: &'a mut Sender<T>,
    value: Option<T>,
}

impl<T> Unpin for SendFuture<'_, T> {}

impl<T> Future for SendFuture<'_, T> {
    type Output = Result<(), T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let value = self
                .value
                .take()
                .expect("send future polled after completion");
            match self.sender.try_send(value) {
                Ok(()) => return Poll::Ready(Ok(())),
                Err(TrySendError::Closed(value)) => return Poll::Ready(Err(value)),
                Err(TrySendError::Full(value)) => self.value = Some(value),
            }

            let mut slot = lock(&self.sender.inner.sender_waker);
            if !self.sender.inner.receiver_open.load(Ordering::Acquire) {
                drop(slot);
                return Poll::Ready(Err(self.value.take().unwrap()));
            }
            if self.sender.has_space() {
                drop(slot);
                continue;
            }
            register(&mut slot, cx.waker());
            return Poll::Pending;
        }
    }
}

pub struct RecvFuture<'a, T> {
    receiver: &'a mut Receiver<T>,
}

impl<T> Future for RecvFuture<'_, T> {
    type Output = Result<T, RecvError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match self.receiver.try_recv() {
                Ok(value) => return Poll::Ready(Ok(value)),
                Err(TryRecvError::Closed) => return Poll::Ready(Err(RecvError)),
                Err(TryRecvError::Empty) => {}
            }

            let mut slot = lock(&self.receiver.inner.receiver_waker);
            if self.receiver.has_value() {
                drop(slot);
                continue;
            }
            if !self.receiver.inner.sender_open.load(Ordering::Acquire) {
                return Poll::Ready(Err(RecvError));
            }
            register(&mut slot, cx.waker());
            return Poll::Pending;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheLine, Slot};
    use std::mem::{align_of, size_of};
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn ring_cursors_and_cells_occupy_distinct_apple_cache_lines() {
        assert_eq!(align_of::<CacheLine<AtomicUsize>>(), 128);
        assert_eq!(size_of::<CacheLine<AtomicUsize>>(), 128);
        assert_eq!(align_of::<Slot<u64>>(), 128);
        assert_eq!(size_of::<Slot<u64>>(), 128);
    }
}
