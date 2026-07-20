use kcoro_sys::{
    Runtime, RuntimeConfig, Service, ServiceFaultCause, ServiceFaultEdge, ServiceOutcome,
    ServiceTerminal,
};
use std::rc::Rc;
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
    Runtime::with_config(RuntimeConfig { workers }).unwrap()
}

fn fault_supervisor(runtime: &Runtime) -> (Service, ServiceFaultEdge, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (service, edge) = runtime
        .state_service_factory(|setup| {
            let edge = setup.fault_edge();
            let record = edge.clone();
            let callback = move || {
                assert!(matches!(record.terminal(), ServiceTerminal::Fault(_)));
                seen.fetch_add(1, Ordering::Release);
                ServiceOutcome::Dormant
            };
            Ok::<_, i32>((callback, edge))
        })
        .unwrap();
    (service, edge, calls)
}

#[test]
fn safe_rust_surface_cannot_create_an_operation_waiter() {
    const WRAPPER: &str = include_str!("../src/lib.rs");
    for forbidden in [
        "pub struct Doorbell",
        "pub fn park",
        "kc_doorbell_park",
        "kc_port_wait_u32",
    ] {
        assert!(
            !WRAPPER.contains(forbidden),
            "safe Rust waiter surface survived: {forbidden}"
        );
    }
}

#[test]
fn fault_publication_is_one_atomic_record_then_one_prebound_wake() {
    const WRAPPER: &str = include_str!("../src/lib.rs");
    let start = WRAPPER
        .find("fn publish_fault(&self, cause: ServiceFaultCause")
        .unwrap();
    let end = WRAPPER[start..]
        .find("\n}\n\nstruct RuntimeInner")
        .map(|offset| start + offset)
        .unwrap();
    let body = &WRAPPER[start..end];
    assert!(body.contains("compare_exchange(0, terminal"));
    assert!(body.contains("self.notifier.notify()"));
    for forbidden in [
        "Box::new",
        "Mutex",
        "Condvar",
        ".lock(",
        ".wait(",
        "catch_unwind",
        "callback()",
        "panic!",
    ] {
        assert!(
            !body.contains(forbidden),
            "fault publication gained forbidden work: {forbidden}"
        );
    }
}

#[test]
fn concurrent_runtime_lifecycle_has_one_thread_set_owner() {
    let runtime = runtime(4);
    let gate = Arc::new(Barrier::new(17));
    let statuses = std::thread::scope(|scope| {
        let starts = (0..16)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let runtime = &runtime;
                scope.spawn(move || {
                    gate.wait();
                    runtime.start()
                })
            })
            .collect::<Vec<_>>();
        gate.wait();
        starts
            .into_iter()
            .map(|start| start.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(statuses.iter().any(Result::is_ok));
    assert!(statuses
        .iter()
        .all(|status| status.is_ok() || *status == Err(-16)));

    let gate = Arc::new(Barrier::new(17));
    let statuses = std::thread::scope(|scope| {
        let joins = (0..16)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let runtime = &runtime;
                scope.spawn(move || {
                    gate.wait();
                    runtime.join()
                })
            })
            .collect::<Vec<_>>();
        gate.wait();
        joins
            .into_iter()
            .map(|join| join.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(statuses.iter().any(Result::is_ok));
    assert!(statuses
        .iter()
        .all(|status| status.is_ok() || *status == Err(-16)));
    runtime.join().unwrap();
    runtime.destroy().unwrap();
}

fn finish(runtime: Runtime, service: Service) {
    service.destroy().unwrap();
    runtime.destroy().unwrap();
}

fn observe(service: &Service, handled: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if service.snapshot().unwrap().handled_notifications >= handled {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "service edge was not acknowledged"
        );
        std::thread::yield_now();
    }
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
    assert_eq!(service.snapshot().unwrap().callbacks, 0);

    published.store(42, Ordering::Release);
    service.notify().unwrap();
    observe(&service, 1);
    assert_eq!(observed.load(Ordering::Acquire), 42);

    finish(runtime, service);
}

#[test]
fn setup_factory_seals_one_callback_after_minting_distinct_producer_edges() {
    let runtime = runtime(2);
    let left = Arc::new(AtomicUsize::new(0));
    let right = Arc::new(AtomicUsize::new(0));
    let drained = Arc::new(AtomicUsize::new(0));
    let left_task = Arc::clone(&left);
    let right_task = Arc::clone(&right);
    let total = Arc::clone(&drained);
    let (service, (mut left_edge, mut right_edge)) = runtime
        .state_service_factory(|setup| {
            let left_edge = setup.realtime_notifier()?;
            let right_edge = setup.realtime_notifier()?;
            let callback = move || {
                let count =
                    left_task.swap(0, Ordering::AcqRel) + right_task.swap(0, Ordering::AcqRel);
                total.fetch_add(count, Ordering::Release);
                ServiceOutcome::Dormant
            };
            Ok::<_, i32>((callback, (left_edge, right_edge)))
        })
        .unwrap();

    assert_eq!(service.snapshot().unwrap().callbacks, 0);
    runtime.start().unwrap();
    service.start().unwrap();
    left.store(1, Ordering::Release);
    right.store(1, Ordering::Release);
    std::thread::scope(|scope| {
        scope.spawn(|| left_edge.notify().unwrap());
        scope.spawn(|| right_edge.notify().unwrap());
    });
    observe(&service, 2);

    assert_eq!(drained.load(Ordering::Acquire), 2);
    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 2);
    assert_eq!(snapshot.handled_notifications, 2);
    drop(left_edge);
    drop(right_edge);
    finish(runtime, service);
}

#[test]
fn callback_owned_notifier_does_not_retain_its_callback_context() {
    struct Dropped(Arc<AtomicBool>);

    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let runtime = runtime(1);
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(Arc::clone(&dropped));
    let (service, mut producer) = runtime
        .state_service_factory(|setup| {
            let callback_edge = setup.realtime_notifier()?;
            let producer = setup.realtime_notifier()?;
            let callback = move || {
                /* Keeping this producer in durable callback state used to form
                 * ServiceInner -> Callback -> RealtimeNotifier -> ServiceInner. */
                let _ = &callback_edge;
                let _ = &guard;
                ServiceOutcome::Complete
            };
            Ok::<_, i32>((callback, producer))
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    producer.notify().unwrap();
    observe(&service, 1);
    service.join().unwrap();
    service.destroy().unwrap();
    assert!(
        dropped.load(Ordering::Acquire),
        "terminal service retained a callback-owned notifier cycle"
    );
    assert!(producer.notify().is_err());
    drop(producer);
    runtime.destroy().unwrap();
}

#[test]
fn owner_local_state_is_created_advanced_and_destroyed_on_one_worker() {
    struct OwnerDrop {
        owner: std::thread::ThreadId,
        dropped: Arc<AtomicBool>,
        migrated: Arc<AtomicBool>,
    }

    impl Drop for OwnerDrop {
        fn drop(&mut self) {
            if std::thread::current().id() != self.owner {
                self.migrated.store(true, Ordering::Release);
            }
            self.dropped.store(true, Ordering::Release);
        }
    }

    let runtime = runtime(4);
    let initialized = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let migrated = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let init_seen = Arc::clone(&initialized);
    let drop_seen = Arc::clone(&dropped);
    let migration = Arc::clone(&migrated);
    let callback_calls = Arc::clone(&calls);
    let (service, mut edge) = runtime
        .owner_state_service_factory(|setup| {
            let edge = setup.realtime_notifier()?;
            let initializer = move || {
                let owner = std::thread::current().id();
                let local = Rc::new(());
                let guard = OwnerDrop {
                    owner,
                    dropped: drop_seen,
                    migrated: migration,
                };
                init_seen.store(true, Ordering::Release);
                move || {
                    let _ = (&local, &guard);
                    if std::thread::current().id() != owner {
                        guard.migrated.store(true, Ordering::Release);
                    }
                    callback_calls.fetch_add(1, Ordering::Release);
                    ServiceOutcome::Dormant
                }
            };
            Ok::<_, i32>((initializer, edge))
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    edge.notify().unwrap();
    observe(&service, 1);
    assert!(initialized.load(Ordering::Acquire));
    assert_eq!(calls.load(Ordering::Acquire), 1);

    drop(edge);
    service.stop();
    service.join().unwrap();
    assert!(dropped.load(Ordering::Acquire));
    assert!(!migrated.load(Ordering::Acquire));
    service.destroy().unwrap();
    runtime.destroy().unwrap();
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
    assert!(!called.load(Ordering::Acquire));

    service.notify().unwrap();
    observe(&service, 1);
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
    service.notify().unwrap();
    gate.wait(1);

    for _ in 0..64 {
        service.notify().unwrap();
    }
    gate.release();
    observe(&service, 65);

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
        .state_service(move || {
            let before = work.fetch_sub(1, Ordering::AcqRel);
            seen.fetch_add(1, Ordering::Release);
            if before > 1 {
                return ServiceOutcome::Continue;
            }
            ServiceOutcome::Dormant
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    service.notify().unwrap();
    observe(&service, QUOTAS as u64);

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
fn natural_completion_retires_only_the_current_service() {
    let runtime = runtime(2);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let service = runtime
        .state_service(move || {
            seen.fetch_add(1, Ordering::Release);
            ServiceOutcome::Complete
        })
        .unwrap();

    runtime.start().unwrap();
    service.start().unwrap();
    service.notify().unwrap();
    observe(&service, 1);
    service.join().unwrap();
    assert_eq!(calls.load(Ordering::Acquire), 1);
    let snapshot = service.snapshot().unwrap();
    assert!(snapshot.joined);
    assert!(!snapshot.stop_requested);
    service.destroy().unwrap();

    let probe = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&probe);
    let next = runtime
        .service(move || flag.store(true, Ordering::Release))
        .unwrap();
    next.start().unwrap();
    next.notify().unwrap();
    observe(&next, 1);
    assert!(probe.load(Ordering::Acquire));
    finish(runtime, next);
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
fn callback_panic_retires_the_service_without_stopping_the_runtime() {
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
    runtime.join_all().unwrap();
    service.join().unwrap();

    assert!(service.callback_panicked());
    assert_eq!(calls.load(Ordering::Acquire), 1);
    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 1);
    assert_eq!(snapshot.handled_notifications, 1);
    assert_eq!(snapshot.callbacks, 1);
    assert!(snapshot.joined);
    assert!(!snapshot.stop_requested);
    assert!(service.notify().is_err());
    service.destroy().unwrap();

    let continued = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&continued);
    let next = runtime
        .state_service(move || {
            observed.store(true, Ordering::Release);
            ServiceOutcome::Complete
        })
        .unwrap();
    next.start().unwrap();
    next.notify().unwrap();
    runtime.join_all().unwrap();
    next.join().unwrap();
    assert!(continued.load(Ordering::Acquire));

    finish(runtime, next);
}

#[test]
fn callback_panic_publishes_one_fault_edge_before_owner_finalization() {
    struct Finalized {
        edge: ServiceFaultEdge,
        ordered: Arc<AtomicBool>,
    }

    impl Drop for Finalized {
        fn drop(&mut self) {
            self.ordered.store(
                matches!(
                    self.edge.terminal(),
                    ServiceTerminal::Fault(fault)
                        if fault.cause == ServiceFaultCause::CallbackPanic
                            && fault.status == 0
                ),
                Ordering::Release,
            );
        }
    }

    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let ordered = Arc::new(AtomicBool::new(false));
    let guard = Finalized {
        edge: edge.clone(),
        ordered: Arc::clone(&ordered),
    };
    let (watched, ()) = runtime
        .state_service_factory_with_fault_edge(edge.clone(), |_| {
            let callback = move || {
                let _ = &guard;
                panic!("caught callback panic with a fault edge");
            };
            Ok::<_, i32>((callback, ()))
        })
        .unwrap();

    runtime.start().unwrap();
    supervisor.start().unwrap();
    watched.start().unwrap();
    watched.notify().unwrap();
    observe(&supervisor, 1);
    supervisor.stop();
    runtime.join_all().unwrap();
    watched.join().unwrap();
    supervisor.join().unwrap();

    assert!(watched.callback_panicked());
    assert!(ordered.load(Ordering::Acquire));
    assert_eq!(
        edge.terminal(),
        ServiceTerminal::Fault(kcoro_sys::ServiceFault {
            cause: ServiceFaultCause::CallbackPanic,
            status: 0,
        })
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(supervisor.snapshot().unwrap().notifications, 1);
    watched.destroy().unwrap();
    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn owner_initializer_panic_publishes_one_fault_and_runtime_survives() {
    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let (watched, ()) = runtime
        .owner_state_service_factory_with_fault_edge(edge.clone(), |_| {
            let initializer = || -> fn() -> ServiceOutcome {
                panic!("caught owner initializer panic");
            };
            Ok::<_, i32>((initializer, ()))
        })
        .unwrap();

    runtime.start().unwrap();
    supervisor.start().unwrap();
    watched.start().unwrap();
    observe(&supervisor, 1);
    supervisor.stop();
    runtime.join_all().unwrap();
    watched.join().unwrap();
    supervisor.join().unwrap();

    assert!(watched.callback_panicked());
    assert_eq!(
        edge.terminal(),
        ServiceTerminal::Fault(kcoro_sys::ServiceFault {
            cause: ServiceFaultCause::InitializerPanic,
            status: 0,
        })
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(watched.snapshot().unwrap().callbacks, 0);
    watched.destroy().unwrap();

    let alive = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&alive);
    let probe = runtime
        .state_service(move || {
            signal.store(true, Ordering::Release);
            ServiceOutcome::Complete
        })
        .unwrap();
    probe.start().unwrap();
    probe.notify().unwrap();
    runtime.join_all().unwrap();
    probe.join().unwrap();
    assert!(alive.load(Ordering::Acquire));
    probe.destroy().unwrap();

    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn owner_finalizer_fault_proves_the_hook_outlives_callback_state() {
    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("caught owner finalizer panic");
        }
    }

    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let guard = PanicOnDrop;
    let (watched, ()) = runtime
        .state_service_factory_with_fault_edge(edge.clone(), |_| {
            let callback = move || {
                let _ = &guard;
                ServiceOutcome::Complete
            };
            Ok::<_, i32>((callback, ()))
        })
        .unwrap();

    runtime.start().unwrap();
    supervisor.start().unwrap();
    watched.start().unwrap();
    watched.notify().unwrap();
    observe(&supervisor, 1);
    supervisor.stop();
    runtime.join_all().unwrap();
    watched.join().unwrap();
    supervisor.join().unwrap();

    assert!(watched.callback_panicked());
    assert_eq!(
        edge.terminal(),
        ServiceTerminal::Fault(kcoro_sys::ServiceFault {
            cause: ServiceFaultCause::OwnerFinalizerPanic,
            status: 0,
        })
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
    watched.destroy().unwrap();
    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn retirement_cancellation_is_normal_and_does_not_ring_the_fault_edge() {
    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let gate = Arc::new(Gate::new());
    let callback_gate = Arc::clone(&gate);
    let (watched, ()) = runtime
        .state_service_factory_with_fault_edge(edge.clone(), |_| {
            let callback = move || {
                callback_gate.callback();
                ServiceOutcome::Continue
            };
            Ok::<_, i32>((callback, ()))
        })
        .unwrap();

    runtime.start().unwrap();
    supervisor.start().unwrap();
    watched.start().unwrap();
    watched.notify().unwrap();
    gate.wait(1);
    watched.stop();
    gate.release();
    supervisor.stop();
    runtime.join_all().unwrap();
    watched.join().unwrap();
    supervisor.join().unwrap();

    assert_eq!(edge.terminal(), ServiceTerminal::Normal);
    assert_eq!(watched.reschedule_error(), None);
    assert_eq!(calls.load(Ordering::Acquire), 0);
    assert_eq!(supervisor.snapshot().unwrap().notifications, 0);
    watched.destroy().unwrap();
    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn one_fault_edge_cannot_be_installed_on_two_services() {
    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let (first, ()) = runtime
        .state_service_factory_with_fault_edge(edge.clone(), |_| {
            let callback = || {
                panic!("caught watched callback fault");
            };
            Ok::<_, i32>((callback, ()))
        })
        .unwrap();
    let second = runtime.state_service_factory_with_fault_edge(edge.clone(), |_| {
        let callback = || panic!("must never install");
        Ok::<_, i32>((callback, ()))
    });
    assert_eq!(second.err(), Some(-16));

    runtime.start().unwrap();
    supervisor.start().unwrap();
    first.start().unwrap();
    first.notify().unwrap();
    observe(&supervisor, 1);
    supervisor.stop();
    runtime.join_all().unwrap();
    first.join().unwrap();
    supervisor.join().unwrap();

    assert_eq!(
        edge.terminal(),
        ServiceTerminal::Fault(kcoro_sys::ServiceFault {
            cause: ServiceFaultCause::CallbackPanic,
            status: 0,
        })
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);
    let snapshot = supervisor.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 1);
    assert_eq!(snapshot.handled_notifications, 1);
    first.destroy().unwrap();
    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn failed_service_setup_releases_the_fault_edge_installation() {
    let runtime = runtime(2);
    let (supervisor, edge, calls) = fault_supervisor(&runtime);
    let failed = runtime.state_service_factory_with_fault_edge::<fn() -> ServiceOutcome, _, ()>(
        edge.clone(),
        |_| Err(-17),
    );
    assert_eq!(failed.err(), Some(-17));

    let (watched, ()) = runtime
        .state_service_factory_with_fault_edge(edge.clone(), |_| {
            Ok::<_, i32>((|| ServiceOutcome::Complete, ()))
        })
        .unwrap();
    runtime.start().unwrap();
    supervisor.start().unwrap();
    watched.start().unwrap();
    watched.notify().unwrap();
    watched.stop();
    supervisor.stop();
    runtime.join_all().unwrap();
    watched.join().unwrap();
    supervisor.join().unwrap();

    assert_eq!(edge.terminal(), ServiceTerminal::Normal);
    assert_eq!(calls.load(Ordering::Acquire), 0);
    watched.destroy().unwrap();
    drop(edge);
    supervisor.destroy().unwrap();
    runtime.destroy().unwrap();
}

#[test]
fn shared_realtime_control_edge_is_mpsc_and_callbacks_keep_one_owner() {
    let runtime = runtime(4);
    let owner = Arc::new(Mutex::new(None));
    let migrated = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_owner = Arc::clone(&owner);
    let callback_migrated = Arc::clone(&migrated);
    let callback_calls = Arc::clone(&calls);
    let service = runtime
        .service(move || {
            let current = std::thread::current().id();
            let mut owner = callback_owner.lock().unwrap();
            if let Some(first) = owner.as_ref() {
                if *first != current {
                    callback_migrated.store(true, Ordering::Release);
                }
            } else {
                *owner = Some(current);
            }
            callback_calls.fetch_add(1, Ordering::Release);
        })
        .unwrap();
    let edge = service.shared_realtime_notifier();

    runtime.start().unwrap();
    service.start().unwrap();
    for round in 0..16 {
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let edge = edge.clone();
                scope.spawn(move || edge.notify().unwrap());
            }
        });
        observe(&service, ((round + 1) * 8) as u64);
    }

    let snapshot = service.snapshot().unwrap();
    assert_eq!(snapshot.notifications, 128);
    assert_eq!(snapshot.handled_notifications, 128);
    assert!(calls.load(Ordering::Acquire) >= 16);
    assert!(!migrated.load(Ordering::Acquire));
    drop(edge);
    finish(runtime, service);
}

#[test]
fn shared_realtime_notify_surface_contains_no_callback_side_relay() {
    const WRAPPER: &str = include_str!("../src/lib.rs");
    let start = WRAPPER.find("impl SharedRealtimeNotifier").unwrap();
    let end = WRAPPER[start..]
        .find("/// Preserve the explicit link anchor")
        .map(|offset| start + offset)
        .unwrap();
    let body = &WRAPPER[start..end];
    for forbidden in ["Mutex", "UnsafeCell", "Box::", "Waker", "wait", "timer"] {
        assert!(
            !body.contains(forbidden),
            "shared realtime notify gained a callback-side relay: {forbidden}"
        );
    }
    assert!(body.contains("kc_service_notify"));
}

#[test]
fn retained_state_cursor_survives_callbacks_and_stop_retires_the_final_edge() {
    const STEPS: usize = 6;
    for _round in 0..32 {
        for stop in 0..STEPS {
            let runtime = runtime(3);
            let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
            let trail = Arc::new(Mutex::new(Vec::new()));
            let callback_gate = Arc::clone(&gate);
            let callback_trail = Arc::clone(&trail);
            let mut cursor = 0usize;
            let mut payload = 7u64;
            let service = runtime
                .state_service(move || {
                    payload = payload.wrapping_mul(17).wrapping_add(cursor as u64);
                    callback_trail.lock().unwrap().push((
                        cursor,
                        payload,
                        std::thread::current().id(),
                    ));
                    if cursor == stop {
                        let (lock, changed) = &*callback_gate;
                        let mut state = lock.lock().unwrap();
                        state.0 = true;
                        changed.notify_all();
                        while !state.1 {
                            state = changed.wait(state).unwrap();
                        }
                    }
                    cursor += 1;
                    if cursor == STEPS {
                        return ServiceOutcome::Complete;
                    }
                    ServiceOutcome::Continue
                })
                .unwrap();

            runtime.start().unwrap();
            service.start().unwrap();
            service.notify().unwrap();
            {
                let (lock, changed) = &*gate;
                let mut state = lock.lock().unwrap();
                while !state.0 {
                    state = changed.wait(state).unwrap();
                }
            }
            service.stop();
            {
                let (lock, changed) = &*gate;
                let mut state = lock.lock().unwrap();
                state.1 = true;
                changed.notify_all();
            }
            service.join().unwrap();

            let trail = trail.lock().unwrap();
            assert_eq!(trail.len(), stop + 1);
            assert!(trail.windows(2).all(|pair| pair[0].0 + 1 == pair[1].0));
            assert!(trail.iter().all(|entry| entry.2 == trail[0].2));
            let snapshot = service.snapshot().unwrap();
            assert_eq!(snapshot.notifications, snapshot.callbacks);
            assert_eq!(snapshot.handled_notifications, snapshot.notifications);
            drop(trail);
            service.destroy().unwrap();
            runtime.destroy().unwrap();
        }
    }

    const CONTINUATION: &str = include_str!("../vendor/kcoro_arena/core/src/koro_internal.h");
    assert!(!CONTINUATION.contains("cursor"));
    assert!(!CONTINUATION.contains("payload"));
}
