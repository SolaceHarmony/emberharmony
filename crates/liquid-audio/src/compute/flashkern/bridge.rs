use kcoro::{
    Cause, CommandKind, Completion, DescriptorId, Execution, Publication, ServiceClass, State,
    Submission, TerminalResultKind, TicketId, TicketKind,
};
use std::ffi::c_void;
use std::mem::MaybeUninit;

const ABI_VERSION: u32 = 1;

#[repr(C)]
struct Config {
    size: u32,
    abi_version: u32,
    capacity: u32,
    descriptor_capacity: u32,
}

type ReleaseFn = unsafe extern "C" fn(*mut c_void, *mut c_void);

#[repr(C)]
struct DescriptorSpec {
    size: u32,
    abi_version: u32,
    kind: u32,
    flags: u32,
    payload: *mut c_void,
    context: *mut c_void,
    release: Option<ReleaseFn>,
    reserved: [u64; 3],
}

#[repr(C)]
#[derive(Debug)]
struct DescriptorView {
    size: u32,
    abi_version: u32,
    kind: u32,
    flags: u32,
    payload: *mut c_void,
    reserved: u64,
}

#[repr(C)]
#[derive(Default)]
struct DescriptorSnapshot {
    size: u32,
    abi_version: u32,
    capacity: u32,
    live: u32,
    acquired: u64,
    retained: u64,
    released: u64,
    callbacks: u64,
    max_generation: u32,
    retired: u32,
}

#[repr(C)]
#[derive(Default)]
struct Snapshot {
    size: u32,
    abi_version: u32,
    capacity: u32,
    stopping: u32,
    submissions_accepted: u64,
    submissions_consumed: u64,
    completions_published: u64,
    completions_consumed: u64,
    active_waits: u32,
    reserved: u32,
}

extern "C" {
    fn lfm_kernel_bridge_create(config: *const Config, out: *mut *mut c_void) -> i32;
    fn lfm_kernel_bridge_descriptor_create(
        bridge: *mut c_void,
        spec: *const DescriptorSpec,
        out: *mut DescriptorId,
    ) -> i32;
    fn lfm_kernel_bridge_descriptor_retain(bridge: *mut c_void, descriptor: DescriptorId) -> i32;
    fn lfm_kernel_bridge_descriptor_get(
        bridge: *mut c_void,
        descriptor: DescriptorId,
        out: *mut DescriptorView,
    ) -> i32;
    fn lfm_kernel_bridge_descriptor_release(bridge: *mut c_void, descriptor: DescriptorId) -> i32;
    fn lfm_kernel_bridge_descriptor_snapshot(
        bridge: *mut c_void,
        out: *mut DescriptorSnapshot,
    ) -> i32;
    fn lfm_kernel_bridge_submit(bridge: *mut c_void, submission: *const Submission) -> i32;
    fn lfm_kernel_bridge_wait_submission(
        bridge: *mut c_void,
        out: *mut Submission,
        deadline_ns: u64,
    ) -> i32;
    fn lfm_kernel_bridge_publish_completion(
        bridge: *mut c_void,
        completion: *const Completion,
    ) -> i32;
    fn lfm_kernel_bridge_wait_completion(
        bridge: *mut c_void,
        out: *mut Completion,
        deadline_ns: u64,
    ) -> i32;
    fn lfm_kernel_bridge_request_stop(bridge: *mut c_void);
    fn lfm_kernel_bridge_snapshot(bridge: *mut c_void, out: *mut Snapshot) -> i32;
    fn lfm_kernel_bridge_destroy(bridge: *mut c_void) -> i32;
}

struct Bridge(*mut c_void);

impl Bridge {
    fn new(capacity: u32) -> Self {
        Self::new_with(capacity, capacity.saturating_add(2))
    }

    fn new_with(capacity: u32, descriptor_capacity: u32) -> Self {
        let config = Config {
            size: std::mem::size_of::<Config>() as u32,
            abi_version: ABI_VERSION,
            capacity,
            descriptor_capacity,
        };
        let mut bridge = std::ptr::null_mut();
        assert_eq!(unsafe { lfm_kernel_bridge_create(&config, &mut bridge) }, 0);
        assert!(!bridge.is_null());
        Self(bridge)
    }

    fn address(&self) -> usize {
        self.0 as usize
    }

    fn submit(&self, submission: &Submission) -> i32 {
        unsafe { lfm_kernel_bridge_submit(self.0, submission) }
    }

    fn descriptor(&self, kind: u32) -> DescriptorLease {
        self.descriptor_with(kind, None, std::ptr::null_mut())
    }

    fn descriptor_with(
        &self,
        kind: u32,
        release: Option<ReleaseFn>,
        context: *mut c_void,
    ) -> DescriptorLease {
        let spec = DescriptorSpec {
            size: std::mem::size_of::<DescriptorSpec>() as u32,
            abi_version: ABI_VERSION,
            kind,
            flags: 0,
            payload: self.0,
            context,
            release,
            reserved: [0; 3],
        };
        let mut id = DescriptorId::NONE;
        assert_eq!(
            unsafe { lfm_kernel_bridge_descriptor_create(self.0, &spec, &mut id) },
            0
        );
        DescriptorLease {
            bridge: self.0,
            id,
            held: true,
        }
    }

    fn descriptor_snapshot(&self) -> DescriptorSnapshot {
        let mut snapshot = DescriptorSnapshot {
            size: std::mem::size_of::<DescriptorSnapshot>() as u32,
            abi_version: ABI_VERSION,
            ..DescriptorSnapshot::default()
        };
        assert_eq!(
            unsafe { lfm_kernel_bridge_descriptor_snapshot(self.0, &mut snapshot) },
            0
        );
        snapshot
    }

    fn descriptor_view(&self, id: DescriptorId) -> Result<DescriptorView, i32> {
        let mut view = DescriptorView {
            size: std::mem::size_of::<DescriptorView>() as u32,
            abi_version: ABI_VERSION,
            kind: 0,
            flags: 0,
            payload: std::ptr::null_mut(),
            reserved: 0,
        };
        let rc = unsafe { lfm_kernel_bridge_descriptor_get(self.0, id, &mut view) };
        if rc != 0 {
            return Err(rc);
        }
        Ok(view)
    }

    fn wait_submission(&self) -> Result<Submission, i32> {
        wait_submission(self.address())
    }

    fn publish(&self, completion: &Completion) -> i32 {
        unsafe { lfm_kernel_bridge_publish_completion(self.0, completion) }
    }

    fn wait_completion(&self) -> Result<Completion, i32> {
        wait_completion(self.address())
    }

    fn request_stop(&self) {
        unsafe { lfm_kernel_bridge_request_stop(self.0) }
    }

    fn snapshot(&self) -> Snapshot {
        let mut snapshot = Snapshot {
            size: std::mem::size_of::<Snapshot>() as u32,
            abi_version: ABI_VERSION,
            ..Snapshot::default()
        };
        assert_eq!(
            unsafe { lfm_kernel_bridge_snapshot(self.0, &mut snapshot) },
            0
        );
        snapshot
    }
}

struct DescriptorLease {
    bridge: *mut c_void,
    id: DescriptorId,
    held: bool,
}

// SAFETY: descriptor retain/release is serialized by the native bridge; the
// lease carries no dereferenceable Rust pointer.
unsafe impl Send for DescriptorLease {}

impl DescriptorLease {
    fn release(&mut self) {
        if !self.held {
            return;
        }
        assert_eq!(
            unsafe { lfm_kernel_bridge_descriptor_release(self.bridge, self.id) },
            0
        );
        self.held = false;
    }
}

impl Drop for DescriptorLease {
    fn drop(&mut self) {
        self.release();
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        unsafe { lfm_kernel_bridge_request_stop(self.0) };
        assert_eq!(unsafe { lfm_kernel_bridge_destroy(self.0) }, 0);
    }
}

fn wait_submission(address: usize) -> Result<Submission, i32> {
    let mut out = MaybeUninit::<Submission>::uninit();
    let rc =
        unsafe { lfm_kernel_bridge_wait_submission(address as *mut c_void, out.as_mut_ptr(), 0) };
    if rc != 0 {
        return Err(rc);
    }
    Ok(unsafe { out.assume_init() })
}

fn wait_completion(address: usize) -> Result<Completion, i32> {
    let mut out = MaybeUninit::<Completion>::uninit();
    let rc =
        unsafe { lfm_kernel_bridge_wait_completion(address as *mut c_void, out.as_mut_ptr(), 0) };
    if rc != 0 {
        return Err(rc);
    }
    Ok(unsafe { out.assume_init() })
}

fn submission(sequence: u64, descriptor: DescriptorId) -> Submission {
    Submission::new(
        TicketId::new(7, sequence, sequence as u32, TicketKind::Pass),
        TicketId::new(7, 1, 1, TicketKind::Turn),
        41,
        3,
        descriptor,
        CommandKind::RunPass,
        ServiceClass::Interactive,
    )
}

fn completion(submission: &Submission, pass_id: u64) -> Completion {
    let mut completion = Completion::new(
        submission.ticket,
        submission.conversation_id,
        submission.epoch,
        pass_id,
        Execution::Completed,
        State::Committed,
        Publication::Committed,
        Cause::Success,
    );
    completion
        .set_results(TerminalResultKind::TextToken, &[11, 13, 17])
        .unwrap();
    completion
}

unsafe extern "C" fn count_release(_payload: *mut c_void, context: *mut c_void) {
    let count = unsafe { &*(context as *const std::sync::atomic::AtomicUsize) };
    count.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

struct ReleaseGate {
    state: std::sync::Mutex<(bool, bool)>,
    changed: std::sync::Condvar,
}

unsafe extern "C" fn gate_release(_payload: *mut c_void, context: *mut c_void) {
    let gate = unsafe { &*(context as *const ReleaseGate) };
    let mut state = gate.state.lock().unwrap();
    state.0 = true;
    gate.changed.notify_all();
    while !state.1 {
        state = gate.changed.wait(state).unwrap();
    }
}

#[test]
fn descriptor_generations_reject_stale_leases() {
    let bridge = Bridge::new_with(1, 1);
    let mut first = bridge.descriptor(17);
    let first_id = first.id;
    let view = bridge.descriptor_view(first_id).unwrap();
    assert_eq!(view.kind, 17);
    assert_eq!(view.payload, bridge.0);

    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_retain(bridge.0, first_id) },
        0
    );
    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_release(bridge.0, first_id) },
        0
    );
    first.release();
    assert_eq!(bridge.descriptor_view(first_id).unwrap_err(), -libc::ESTALE);
    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_retain(bridge.0, first_id) },
        -libc::ESTALE
    );
    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_release(bridge.0, first_id) },
        -libc::ESTALE
    );

    let second = bridge.descriptor(19);
    assert_eq!(second.id.slot, first_id.slot);
    assert_eq!(second.id.generation, first_id.generation + 1);
    let snapshot = bridge.descriptor_snapshot();
    assert_eq!(snapshot.capacity, 1);
    assert_eq!(snapshot.live, 1);
    assert_eq!(snapshot.acquired, 2);
    assert_eq!(snapshot.retained, 1);
    assert_eq!(snapshot.released, 2);
}

#[test]
fn accepted_submission_holds_descriptor_until_cq_consumption() {
    let bridge = Bridge::new_with(1, 1);
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let mut lease =
        bridge.descriptor_with(23, Some(count_release), &calls as *const _ as *mut c_void);
    let input = submission(1, lease.id);
    assert_eq!(bridge.submit(&input), 0);
    lease.release();
    assert_eq!(calls.load(std::sync::atomic::Ordering::Acquire), 0);

    assert_eq!(bridge.wait_submission().unwrap(), input);
    let output = completion(&input, 1);
    assert_eq!(bridge.publish(&output), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Acquire), 0);
    assert_eq!(bridge.wait_completion().unwrap(), output);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Acquire), 1);

    let snapshot = bridge.descriptor_snapshot();
    assert_eq!(snapshot.live, 0);
    assert_eq!(snapshot.retained, 1);
    assert_eq!(snapshot.released, 2);
    assert_eq!(snapshot.callbacks, 1);
}

#[test]
fn descriptor_slot_recycles_only_after_release_callback_returns() {
    let bridge = Bridge::new_with(1, 1);
    let gate = std::sync::Arc::new(ReleaseGate {
        state: std::sync::Mutex::new((false, false)),
        changed: std::sync::Condvar::new(),
    });
    let first = bridge.descriptor_with(
        29,
        Some(gate_release),
        std::sync::Arc::as_ptr(&gate) as *mut c_void,
    );
    let first_id = first.id;
    let releaser = std::thread::spawn(move || drop(first));

    {
        let mut state = gate.state.lock().unwrap();
        while !state.0 {
            state = gate.changed.wait(state).unwrap();
        }
    }
    assert_eq!(bridge.descriptor_snapshot().live, 1);
    let spec = DescriptorSpec {
        size: std::mem::size_of::<DescriptorSpec>() as u32,
        abi_version: ABI_VERSION,
        kind: 31,
        flags: 0,
        payload: bridge.0,
        context: std::ptr::null_mut(),
        release: None,
        reserved: [0; 3],
    };
    let mut blocked = DescriptorId::NONE;
    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_create(bridge.0, &spec, &mut blocked) },
        -libc::ENOSPC
    );

    {
        let mut state = gate.state.lock().unwrap();
        state.1 = true;
        gate.changed.notify_all();
    }
    releaser.join().unwrap();
    assert_eq!(bridge.descriptor_snapshot().live, 0);
    assert_eq!(bridge.descriptor_snapshot().callbacks, 1);

    let second = bridge.descriptor(31);
    assert_eq!(second.id.slot, first_id.slot);
    assert_eq!(second.id.generation, first_id.generation + 1);
}

#[test]
fn bridge_destroy_rejects_a_live_descriptor() {
    let bridge = Bridge::new_with(1, 1);
    let mut lease = bridge.descriptor(37);
    bridge.request_stop();
    assert_eq!(
        unsafe { lfm_kernel_bridge_descriptor_retain(bridge.0, lease.id) },
        -libc::ECANCELED
    );
    assert_eq!(unsafe { lfm_kernel_bridge_destroy(bridge.0) }, -libc::EBUSY);
    lease.release();
}

#[test]
fn native_bridge_round_trip_preserves_protocol_cells() {
    let bridge = Bridge::new(4);
    let address = bridge.address();
    let _lease = bridge.descriptor(1);
    let input = submission(9, _lease.id);
    let expected = input;
    let executor = std::thread::spawn(move || {
        let received = wait_submission(address).unwrap();
        assert_eq!(received, expected);
        let completion = completion(&received, 101);
        assert_eq!(
            unsafe { lfm_kernel_bridge_publish_completion(address as *mut c_void, &completion) },
            0
        );
    });
    let address = bridge.address();
    let ingress = std::thread::spawn(move || wait_completion(address).unwrap());

    assert_eq!(bridge.submit(&input), 0);
    executor.join().unwrap();
    let output = ingress.join().unwrap();
    assert_eq!(output, completion(&input, 101));

    let snapshot = bridge.snapshot();
    assert_eq!(snapshot.capacity, 4);
    assert_eq!(snapshot.submissions_accepted, 1);
    assert_eq!(snapshot.submissions_consumed, 1);
    assert_eq!(snapshot.completions_published, 1);
    assert_eq!(snapshot.completions_consumed, 1);
    assert_eq!(snapshot.active_waits, 0);
}

#[test]
fn completion_reservation_controls_submission_backpressure() {
    let bridge = Bridge::new(2);
    let _first_lease = bridge.descriptor(1);
    let _second_lease = bridge.descriptor(1);
    let _third_lease = bridge.descriptor(1);
    let first = submission(1, _first_lease.id);
    let second = submission(2, _second_lease.id);
    let third = submission(3, _third_lease.id);
    assert_eq!(bridge.submit(&first), 0);
    assert_eq!(bridge.submit(&second), 0);
    assert_eq!(bridge.submit(&third), -libc::EAGAIN);

    assert_eq!(bridge.wait_submission().unwrap(), first);
    assert_eq!(bridge.publish(&completion(&first, 1)), 0);
    assert_eq!(
        bridge.submit(&third),
        -libc::EAGAIN,
        "taking SQ work must not release its reserved CQ capacity"
    );
    assert_eq!(bridge.wait_completion().unwrap(), completion(&first, 1));
    assert_eq!(bridge.submit(&third), 0);

    assert_eq!(bridge.wait_submission().unwrap(), second);
    assert_eq!(bridge.publish(&completion(&second, 2)), 0);
    assert_eq!(bridge.wait_completion().unwrap(), completion(&second, 2));
    assert_eq!(bridge.wait_submission().unwrap(), third);
    assert_eq!(bridge.publish(&completion(&third, 3)), 0);
    assert_eq!(bridge.wait_completion().unwrap(), completion(&third, 3));
}

#[test]
fn out_of_order_completion_cannot_steal_another_ticket_reservation() {
    let bridge = Bridge::new(2);
    let _first_lease = bridge.descriptor(1);
    let _second_lease = bridge.descriptor(1);
    let first = submission(1, _first_lease.id);
    let second = submission(2, _second_lease.id);
    assert_eq!(bridge.submit(&first), 0);
    assert_eq!(bridge.submit(&second), 0);
    assert_eq!(bridge.wait_submission().unwrap(), first);

    assert_eq!(bridge.publish(&completion(&second, 2)), -libc::ESTALE);
    assert_eq!(bridge.publish(&completion(&first, 1)), 0);
    assert_eq!(bridge.wait_completion().unwrap(), completion(&first, 1));
    assert_eq!(bridge.wait_submission().unwrap(), second);
    assert_eq!(bridge.publish(&completion(&second, 2)), 0);
    assert_eq!(bridge.wait_completion().unwrap(), completion(&second, 2));
}

#[test]
fn stop_wakes_empty_submission_and_completion_waits() {
    let bridge = Bridge::new(2);
    let _lease = bridge.descriptor(1);
    let input = submission(1, _lease.id);
    let submit_address = bridge.address();
    let completion_address = bridge.address();
    let submission_waiter = std::thread::spawn(move || wait_submission(submit_address));
    let completion_waiter = std::thread::spawn(move || wait_completion(completion_address));

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if bridge.snapshot().active_waits == 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "bridge waiters did not arm"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(bridge.snapshot().active_waits, 2);
    bridge.request_stop();
    assert_eq!(bridge.submit(&input), -libc::ECANCELED);
    assert_eq!(submission_waiter.join().unwrap(), Err(-libc::ECANCELED));
    assert_eq!(completion_waiter.join().unwrap(), Err(-libc::ECANCELED));
    assert_eq!(bridge.snapshot().active_waits, 0);
}

#[test]
fn accepted_submission_remains_drainable_after_stop() {
    let bridge = Bridge::new(1);
    let _lease = bridge.descriptor(1);
    let input = submission(1, _lease.id);
    assert_eq!(bridge.submit(&input), 0);
    bridge.request_stop();
    assert_eq!(bridge.wait_submission().unwrap(), input);
    let output = completion(&input, 1);
    assert_eq!(bridge.publish(&output), 0);
    assert_eq!(bridge.wait_completion().unwrap(), output);
}

#[test]
fn stop_and_submit_arbitrate_without_orphaning_accepted_work() {
    for sequence in 1..=1_000 {
        let bridge = Bridge::new(1);
        let address = bridge.address();
        let _lease = bridge.descriptor(1);
        let input = submission(sequence, _lease.id);
        let start = std::sync::Arc::new(std::sync::Barrier::new(3));
        let submit_start = start.clone();
        let submitter = std::thread::spawn(move || {
            submit_start.wait();
            unsafe { lfm_kernel_bridge_submit(address as *mut c_void, &input) }
        });
        let address = bridge.address();
        let stop_start = start.clone();
        let stopper = std::thread::spawn(move || {
            stop_start.wait();
            unsafe { lfm_kernel_bridge_request_stop(address as *mut c_void) };
        });
        start.wait();
        let rc = submitter.join().unwrap();
        stopper.join().unwrap();
        if rc == -libc::ECANCELED {
            assert_eq!(bridge.snapshot().submissions_accepted, 0);
            continue;
        }
        assert_eq!(rc, 0);
        let accepted = bridge.wait_submission().unwrap();
        assert_eq!(accepted, input);
        let output = completion(&accepted, sequence);
        assert_eq!(bridge.publish(&output), 0);
        assert_eq!(bridge.wait_completion().unwrap(), output);
    }
}

#[test]
fn bridge_wraps_without_losing_ticket_identity() {
    let bridge = Bridge::new(7);
    for sequence in 1..=10_000 {
        let _lease = bridge.descriptor(1);
        let input = submission(sequence, _lease.id);
        assert_eq!(bridge.submit(&input), 0);
        assert_eq!(bridge.wait_submission().unwrap(), input);
        let output = completion(&input, sequence);
        assert_eq!(bridge.publish(&output), 0);
        assert_eq!(bridge.wait_completion().unwrap(), output);
    }
    let snapshot = bridge.snapshot();
    assert_eq!(snapshot.submissions_accepted, 10_000);
    assert_eq!(snapshot.submissions_consumed, 10_000);
    assert_eq!(snapshot.completions_published, 10_000);
    assert_eq!(snapshot.completions_consumed, 10_000);
}

#[test]
fn incompatible_submission_is_rejected_before_reservation() {
    let bridge = Bridge::new(2);
    let _lease = bridge.descriptor(1);
    let mut input = submission(1, _lease.id);
    input.abi_version += 1;
    assert_eq!(bridge.submit(&input), -libc::EINVAL);
    let snapshot = bridge.snapshot();
    assert_eq!(snapshot.submissions_accepted, 0);
    assert_eq!(snapshot.completions_consumed, 0);
}
