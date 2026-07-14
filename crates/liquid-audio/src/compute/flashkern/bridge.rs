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
    reserved: u32,
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
        let config = Config {
            size: std::mem::size_of::<Config>() as u32,
            abi_version: ABI_VERSION,
            capacity,
            reserved: 0,
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

fn submission(sequence: u64) -> Submission {
    Submission::new(
        TicketId::new(7, sequence, sequence as u32, TicketKind::Pass),
        TicketId::new(7, 1, 1, TicketKind::Turn),
        41,
        3,
        DescriptorId::new((sequence % 4) as u32, sequence as u32),
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

#[test]
fn native_bridge_round_trip_preserves_protocol_cells() {
    let bridge = Bridge::new(4);
    let address = bridge.address();
    let input = submission(9);
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
    let first = submission(1);
    let second = submission(2);
    let third = submission(3);
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
    let first = submission(1);
    let second = submission(2);
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
    assert_eq!(bridge.submit(&submission(1)), -libc::ECANCELED);
    assert_eq!(submission_waiter.join().unwrap(), Err(-libc::ECANCELED));
    assert_eq!(completion_waiter.join().unwrap(), Err(-libc::ECANCELED));
    assert_eq!(bridge.snapshot().active_waits, 0);
}

#[test]
fn accepted_submission_remains_drainable_after_stop() {
    let bridge = Bridge::new(1);
    let input = submission(1);
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
        let input = submission(sequence);
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
        let input = submission(sequence);
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
    let mut input = submission(1);
    input.abi_version += 1;
    assert_eq!(bridge.submit(&input), -libc::EINVAL);
    let snapshot = bridge.snapshot();
    assert_eq!(snapshot.submissions_accepted, 0);
    assert_eq!(snapshot.completions_consumed, 0);
}
