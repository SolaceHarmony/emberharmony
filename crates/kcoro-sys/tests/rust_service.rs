use kcoro_sys::{Doorbell, Runtime, RuntimeConfig, Service, ServiceOutcome};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    calls: usize,
    inside: usize,
    maximum: usize,
    released: bool,
}

struct Gate {
    state: Mutex<State>,
    changed: Condvar,
}

impl Gate {
    fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        }
    }

    fn callback(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = self.state.lock().unwrap();
        state.calls += 1;
        state.inside += 1;
        state.maximum = state.maximum.max(state.inside);
        let call = state.calls;
        self.changed.notify_all();
        while call == 1 && !state.released {
            let wait = deadline.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                state.released = true;
                continue;
            }
            state = self.changed.wait_timeout(state, wait).unwrap().0;
        }
        state.inside -= 1;
        self.changed.notify_all();
    }

    fn wait(&self, calls: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = self.state.lock().unwrap();
        while state.calls < calls {
            let wait = deadline.saturating_duration_since(Instant::now());
            assert!(!wait.is_zero(), "service callback timed out");
            state = self.changed.wait_timeout(state, wait).unwrap().0;
        }
    }

    fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.changed.notify_all();
    }

    fn result(&self) -> (usize, usize) {
        let state = self.state.lock().unwrap();
        (state.calls, state.maximum)
    }
}

fn runtime(workers: u32) -> Runtime {
    Runtime::with_config(RuntimeConfig {
        workers,
        segment: 0,
        tickets: 4,
    })
    .unwrap()
}

fn finish(runtime: Runtime, service: Service) {
    service.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn doorbell_realtime_capability_is_queryable() {
    let doorbell = Doorbell::new().unwrap();
    let sequence = doorbell.observe();
    let _ = doorbell.realtime_safe();
    assert_eq!(doorbell.observe(), sequence);
}

#[test]
fn publication_precedes_the_safe_callback() {
    let runtime = runtime(1);
    let published = Arc::new(AtomicU64::new(0));
    let observed = Arc::new(AtomicU64::new(u64::MAX));
    let source = Arc::clone(&published);
    let target = Arc::clone(&observed);
    let service = runtime
        .service(move || {
            target.store(source.load(Ordering::Acquire), Ordering::Release);
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    runtime.run_until_idle().unwrap();
    assert_eq!(service.snapshot().unwrap().callbacks, 0);

    published.store(42, Ordering::Release);
    service.notify().unwrap();
    runtime.run_until_idle().unwrap();
    assert_eq!(observed.load(Ordering::Acquire), 42);

    finish(runtime, service);
}

#[test]
fn safe_callback_is_not_invoked_inline() {
    let runtime = runtime(1);
    let caller = std::thread::current().id();
    let called = Arc::new(AtomicBool::new(false));
    let inline = Arc::new(AtomicBool::new(true));
    let seen = Arc::clone(&called);
    let same = Arc::clone(&inline);
    let service = runtime
        .service(move || {
            same.store(std::thread::current().id() == caller, Ordering::Release);
            seen.store(true, Ordering::Release);
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    runtime.run_until_idle().unwrap();
    assert!(!called.load(Ordering::Acquire));

    service.notify().unwrap();
    runtime.run_until_idle().unwrap();
    assert!(called.load(Ordering::Acquire));
    assert!(!inline.load(Ordering::Acquire));

    finish(runtime, service);
}

#[test]
fn notify_during_callback_is_serial_and_coalesced() {
    let runtime = runtime(2);
    let gate = Arc::new(Gate::new());
    let callback = Arc::clone(&gate);
    let service = runtime.service(move || callback.callback()).unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    runtime.run_until_idle().unwrap();
    service.notify().unwrap();
    gate.wait(1);

    for _ in 0..64 {
        service.notify().unwrap();
    }
    gate.release();
    runtime.run_until_idle().unwrap();

    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 65);
    assert_eq!(snapshot.handled_notifications, 65);
    assert_eq!(snapshot.callbacks, 2);
    assert_eq!(gate.result(), (2, 1));

    finish(runtime, service);
}

#[test]
fn safe_bounded_callback_reenters_until_its_owned_predicate_is_drained() {
    const QUOTAS: usize = 19;
    let runtime = runtime(2);
    let remaining = Arc::new(AtomicUsize::new(QUOTAS));
    let calls = Arc::new(AtomicUsize::new(0));
    let work = Arc::clone(&remaining);
    let seen = Arc::clone(&calls);
    let service = runtime
        .polling_service(move || {
            let before = work.fetch_sub(1, Ordering::AcqRel);
            seen.fetch_add(1, Ordering::Release);
            if before > 1 {
                return ServiceOutcome::ReadyAgain;
            }
            ServiceOutcome::Park
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    service.notify().unwrap();
    runtime.run_until_idle().unwrap();

    assert_eq!(remaining.load(Ordering::Acquire), 0);
    assert_eq!(calls.load(Ordering::Acquire), QUOTAS);
    assert_eq!(service.reschedule_error(), None);
    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, QUOTAS as u64);
    assert_eq!(snapshot.handled_notifications, QUOTAS as u64);
    assert_eq!(snapshot.callbacks, QUOTAS as u64);

    finish(runtime, service);
}

#[test]
fn stop_race_drains_accepted_realtime_edges_before_releasing_lifetimes() {
    let runtime = runtime(2);
    let gate = Arc::new(Gate::new());
    let callback = Arc::clone(&gate);
    let service = runtime.service(move || callback.callback()).unwrap();
    let mut notifier = service.realtime_notifier().unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    runtime.run_until_idle().unwrap();
    notifier.notify().unwrap();
    gate.wait(1);

    let start = Arc::new(Barrier::new(2));
    let ready = Arc::new(Barrier::new(2));
    let accepted = std::thread::scope(|scope| {
        let producer = scope.spawn(|| {
            start.wait();
            let first = usize::from(notifier.notify().is_ok());
            ready.wait();
            first + (0..32_768).filter(|_| notifier.notify().is_ok()).count()
        });
        start.wait();
        ready.wait();
        service.stop();
        producer.join().unwrap()
    });
    let rejected = notifier.notify().is_err();
    gate.release();
    service.join().unwrap();

    let snapshot = service.snapshot().unwrap();
    assert!(accepted >= 1);
    assert!(rejected);
    assert_eq!(snapshot.notifications, accepted as u64 + 1);
    assert_eq!(snapshot.handled_notifications, snapshot.notifications);
    assert_eq!(gate.result().1, 1);

    service.destroy().unwrap();
    assert!(notifier.notify().is_err());
    drop(notifier);
    runtime.destroy().unwrap();
}

#[test]
fn callback_panic_is_contained_at_the_ffi_boundary() {
    let runtime = runtime(1);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let service = runtime
        .service(move || {
            seen.fetch_add(1, Ordering::Release);
            panic!("caught callback panic");
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    service.notify().unwrap();
    runtime.run_until_idle().unwrap();
    assert!(service.callback_panicked());

    service.notify().unwrap();
    runtime.run_until_idle().unwrap();
    assert_eq!(calls.load(Ordering::Acquire), 1);
    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 2);
    assert_eq!(snapshot.handled_notifications, 2);

    finish(runtime, service);
}
