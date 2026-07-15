use kcoro::{ring, Executor, ExecutorConfig, JoinError, SpawnError, TaskResult};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::task::{Context, Poll};

fn runtime(workers: usize, capacity: usize) -> Executor {
    Executor::new(ExecutorConfig {
        workers,
        capacity,
        drain_limit: 8,
        thread_name: "kcoro-test".to_string(),
    })
    .unwrap()
}

#[test]
fn ring_edge_resumes_a_parked_continuation() {
    let runtime = runtime(2, 8);
    let (mut sender, mut receiver) = ring(2).unwrap();
    let observed = Arc::new(AtomicUsize::new(0));
    let output = observed.clone();
    let task = runtime
        .spawn(async move {
            let value = receiver.recv().await.unwrap();
            output.store(value, Ordering::Release);
        })
        .unwrap();

    sender.try_send(73).unwrap();
    assert_eq!(task.wait(), TaskResult::Completed);
    assert_eq!(observed.load(Ordering::Acquire), 73);
    runtime.request_stop();
    assert_eq!(runtime.join(), Ok(()));
}

#[test]
fn closing_a_ring_resolves_the_registered_receiver() {
    let runtime = runtime(1, 4);
    let (sender, mut receiver) = ring::<u32>(1).unwrap();
    let closed = Arc::new(AtomicBool::new(false));
    let output = closed.clone();
    let task = runtime
        .spawn(async move {
            output.store(receiver.recv().await.is_err(), Ordering::Release);
        })
        .unwrap();

    drop(sender);
    assert_eq!(task.wait(), TaskResult::Completed);
    assert!(closed.load(Ordering::Acquire));
    runtime.request_stop();
    assert_eq!(runtime.join(), Ok(()));
}

struct Recur {
    remaining: usize,
    polling: Arc<AtomicBool>,
    polls: Arc<AtomicUsize>,
}

impl Future for Recur {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(!self.polling.swap(true, Ordering::AcqRel));
        self.polls.fetch_add(1, Ordering::Relaxed);
        if self.remaining == 0 {
            self.polling.store(false, Ordering::Release);
            return Poll::Ready(());
        }
        self.remaining -= 1;
        cx.waker().wake_by_ref();
        self.polling.store(false, Ordering::Release);
        Poll::Pending
    }
}

#[test]
fn self_wakes_never_poll_one_continuation_concurrently() {
    let runtime = runtime(4, 8);
    let polling = Arc::new(AtomicBool::new(false));
    let polls = Arc::new(AtomicUsize::new(0));
    let task = runtime
        .spawn(Recur {
            remaining: 10_000,
            polling: polling.clone(),
            polls: polls.clone(),
        })
        .unwrap();

    assert_eq!(task.wait(), TaskResult::Completed);
    assert_eq!(polls.load(Ordering::Acquire), 10_001);
    assert!(!polling.load(Ordering::Acquire));
    runtime.request_stop();
    assert_eq!(runtime.join(), Ok(()));
}

#[test]
fn capacity_is_explicit_and_stop_resolves_pending_tasks() {
    let runtime = runtime(1, 2);
    let first = runtime.spawn(std::future::pending()).unwrap();
    let second = runtime.spawn(std::future::pending()).unwrap();
    assert!(matches!(
        runtime.spawn(std::future::pending()),
        Err(SpawnError::Full)
    ));
    assert_eq!(runtime.join(), Err(JoinError::Running));
    runtime.request_stop();
    assert_eq!(runtime.join(), Ok(()));
    assert_eq!(first.wait(), TaskResult::Stopped);
    assert_eq!(second.wait(), TaskResult::Stopped);
}

#[test]
fn zero_workers_is_rejected_instead_of_becoming_one() {
    let result = Executor::new(ExecutorConfig {
        workers: 0,
        ..ExecutorConfig::default()
    });
    assert!(result.is_err());
}

#[test]
fn a_panicking_continuation_does_not_kill_its_worker() {
    let runtime = runtime(1, 4);
    let panicked = runtime.spawn(async move { panic!("task fault") }).unwrap();
    assert_eq!(panicked.wait(), TaskResult::Panicked);

    let survived = runtime.spawn(async {}).unwrap();
    assert_eq!(survived.wait(), TaskResult::Completed);
    assert_eq!(runtime.stats().panics, 1);
    runtime.request_stop();
    assert_eq!(runtime.join(), Ok(()));
}

#[test]
fn stop_closes_admission_before_teardown_sweeps_slots() {
    let runtime = Arc::new(runtime(2, 512));
    let start = Arc::new(Barrier::new(9));
    let handles = Arc::new(Mutex::new(Vec::new()));
    let mut spawners = Vec::new();
    for _ in 0..8 {
        let runtime = runtime.clone();
        let start = start.clone();
        let handles = handles.clone();
        spawners.push(std::thread::spawn(move || {
            start.wait();
            for _ in 0..128 {
                match runtime.spawn(std::future::pending()) {
                    Ok(handle) => handles.lock().unwrap().push(handle),
                    Err(SpawnError::Stopped | SpawnError::Full) => return,
                    Err(SpawnError::Faulted) => panic!("executor faulted during admission race"),
                }
            }
        }));
    }

    start.wait();
    runtime.request_stop();
    for spawner in spawners {
        spawner.join().unwrap();
    }
    assert_eq!(runtime.join(), Ok(()));
    for handle in handles.lock().unwrap().drain(..) {
        assert_eq!(handle.wait(), TaskResult::Stopped);
    }
}
