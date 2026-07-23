use std::ffi::{c_char, c_void, CString};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
#[cfg(unix)]
use std::process::{Command, Stdio};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ticket {
    runtime_epoch: u64,
    sequence: u64,
    generation: u32,
    kind: u32,
}

#[repr(C, align(128))]
#[derive(Clone, Copy, Debug)]
struct FatalCapsule {
    size: u32,
    abi_version: u32,
    request: u32,
    stage: u32,
    program_kind: u32,
    program_phase: u32,
    program_flags: u32,
    quorum_status: i32,
    workflow: Ticket,
    pass: Ticket,
    deadline: Ticket,
    conversation_id: u64,
    epoch: u64,
    scope_generation: u64,
    team_generation: u64,
    program_outer: u64,
    program_inner: u64,
    shape0: u64,
    shape1: u64,
    shape2: u64,
    expected_mask: u64,
    entered_mask: u64,
    returned_mask: u64,
    never_entered_mask: u64,
    entered_not_returned_mask: u64,
    armed_ns: u64,
    hard_budget_ns: u64,
    elapsed_ns: u64,
    deadline_event_sequence: u64,
    scheduled_arm_generation: u64,
    current_arm_generation: u64,
}

impl Default for FatalCapsule {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: 2,
            request: 0,
            stage: 0,
            program_kind: 0,
            program_phase: 0,
            program_flags: 0,
            quorum_status: 0,
            workflow: Ticket::default(),
            pass: Ticket::default(),
            deadline: Ticket::default(),
            conversation_id: 0,
            epoch: 0,
            scope_generation: 0,
            team_generation: 0,
            program_outer: 0,
            program_inner: 0,
            shape0: 0,
            shape1: 0,
            shape2: 0,
            expected_mask: 0,
            entered_mask: 0,
            returned_mask: 0,
            never_entered_mask: 0,
            entered_not_returned_mask: 0,
            armed_ns: 0,
            hard_budget_ns: 0,
            elapsed_ns: 0,
            deadline_event_sequence: 0,
            scheduled_arm_generation: 0,
            current_arm_generation: 0,
        }
    }
}

#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct FatalSinkHeader {
    magic: u64,
    abi_version: u32,
    header_size: u32,
    capsule_size: u32,
    committed: u32,
    runtime_epoch: u64,
    checksum: u64,
    reserved: [u8; 88],
}

unsafe extern "C" {
    fn lfm_internal_engine_team_terminal_race_for_test(
        generation: u64,
        first: u32,
        second: u32,
        winner_bits: *mut u32,
        terminal_state: *mut u32,
    ) -> i32;
    fn lfm_internal_engine_completion_decision_for_test(
        generation: u64,
        retire_result: i32,
        decision: *mut i32,
        terminal_state: *mut u32,
    ) -> i32;
    fn lfm_internal_engine_fatal_capsule_for_test(
        generation: u64,
        expected_mask: u64,
        entered_mask: u64,
        returned_mask: u64,
        out: *mut FatalCapsule,
    ) -> i32;
    fn lfm_internal_engine_new_hard_timeout_probe_for_test(
        workers: i32,
        member: u32,
        point: u32,
        fatal_path: *const c_char,
    ) -> *mut c_void;
    fn lfm_internal_engine_trigger_hard_timeout_probe_for_test(engine: *mut c_void) -> i32;
    fn lfm_internal_engine_hard_budget_for_test(probe: u32, request: u32) -> u64;
}

const ACTIVE: u32 = 1;
const COMPLETED: u32 = 2;
const TIMED_OUT: u32 = 3;
const RETIRE_RETIRED: i32 = 0;
const RETIRE_EXPIRY_WON: i32 = 1;
const PROCESS: i32 = 1;
const DEFER_TO_EXPIRY: i32 = 2;
const NEVER_ENTERED: u32 = 1;
const ENTERED_NEVER_RETURNED: u32 = 2;
const FATAL_MAGIC: u64 = 0x314c415441464b46;
const FATAL_ABI: u32 = 1;
const FATAL_COMMITTED: u32 = 1;
const FATAL_HEADER_BYTES: usize = 128;

#[cfg(unix)]
fn fatal_child(point: u32, path: &str) -> ! {
    /* This one-shot is only the test watchdog. It can terminate a broken child,
     * but it can never advance the numerical state machine. */
    unsafe { libc::alarm(5) };
    let path = CString::new(path).expect("fatal sink path contains NUL");
    let engine =
        unsafe { lfm_internal_engine_new_hard_timeout_probe_for_test(4, 1, point, path.as_ptr()) };
    assert!(
        !engine.is_null(),
        "could not create hard-timeout probe engine"
    );
    assert_eq!(
        unsafe { lfm_internal_engine_trigger_hard_timeout_probe_for_test(engine) },
        0
    );
    unsafe { libc::pause() };
    unsafe { libc::_exit(90) }
}

#[cfg(unix)]
fn run_fatal_child(point: u32) -> FatalCapsule {
    let path = std::env::temp_dir().join(format!(
        "emberharmony-hard-supervision-{}-{point}.bin",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let executable = std::env::current_exe().expect("hard supervision test executable");
    let mut child = Command::new(executable)
        .arg("--exact")
        .arg("hard_timeout_fault_injection_is_process_fatal")
        .arg("--nocapture")
        .env("LFM_HARD_TIMEOUT_CHILD", point.to_string())
        .env("LFM_HARD_TIMEOUT_PATH", &path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn hard-timeout child");
    let status = child.wait().expect("reap hard-timeout child");
    assert_eq!(
        status.signal(),
        Some(libc::SIGABRT),
        "only the fatal supervisor may terminate the child: {status:?}"
    );
    let bytes = std::fs::read(&path).expect("read persistent fatal capsule image");
    std::fs::remove_file(&path).expect("remove consumed fatal capsule image");
    assert_eq!(
        bytes.len(),
        FATAL_HEADER_BYTES + std::mem::size_of::<FatalCapsule>()
    );
    let header = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<FatalSinkHeader>()) };
    assert_eq!(header.magic, FATAL_MAGIC);
    assert_eq!(header.abi_version, FATAL_ABI);
    assert_eq!(header.header_size as usize, FATAL_HEADER_BYTES);
    assert_eq!(
        header.capsule_size as usize,
        std::mem::size_of::<FatalCapsule>()
    );
    assert_eq!(header.committed, FATAL_COMMITTED);
    assert_ne!(header.runtime_epoch, 0);
    let capsule_bytes = &bytes[FATAL_HEADER_BYTES..];
    let checksum = capsule_bytes
        .iter()
        .fold(14_695_981_039_346_656_037_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211)
        });
    assert_eq!(header.checksum, checksum);
    let capsule =
        unsafe { std::ptr::read_unaligned(capsule_bytes.as_ptr().cast::<FatalCapsule>()) };
    capsule
}

#[test]
fn mounted_production_passes_receive_the_closed_hard_budget() {
    /* Every currently mounted request family is an explicit member of the
     * closed production table. Unknown selectors still receive no fallback
     * and are rejected before a team generation can be dispatched. */
    for request in [2, 3, 4, 8, 13, 14, 15] {
        assert_eq!(
            unsafe { lfm_internal_engine_hard_budget_for_test(0, request) },
            1_000_000_000,
            "request {request} lost its production hard budget"
        );
    }
    assert_eq!(
        unsafe { lfm_internal_engine_hard_budget_for_test(0, 0xffff) },
        0,
        "an unknown request acquired a fallback budget"
    );
    assert_eq!(
        unsafe { lfm_internal_engine_hard_budget_for_test(1, 15) },
        1_000_000_000,
        "the deterministic poisoned-lane probe must remain supervised"
    );
}

#[test]
fn only_authoritative_deadline_retirement_permits_completion() {
    /* Keep the owning crate in this integration-test link so its private
     * native archive (where these probes live) is retained by the linker. */
    let _ = liquid_audio::NativeVoiceSampling::default();
    for (retire, expected_decision, expected_terminal) in [
        (RETIRE_RETIRED, PROCESS, COMPLETED),
        (RETIRE_EXPIRY_WON, DEFER_TO_EXPIRY, ACTIVE),
    ] {
        let mut decision = 0;
        let mut terminal = 0;
        let status = unsafe {
            lfm_internal_engine_completion_decision_for_test(
                0x5678,
                retire,
                &mut decision,
                &mut terminal,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(decision, expected_decision);
        assert_eq!(terminal, expected_terminal);
    }
}

#[test]
fn completion_and_timeout_share_one_generation_terminal_cas() {
    for (first, second) in [(COMPLETED, TIMED_OUT), (TIMED_OUT, COMPLETED)] {
        let mut winners = 0;
        let mut terminal = 0;
        let status = unsafe {
            lfm_internal_engine_team_terminal_race_for_test(
                0x1234,
                first,
                second,
                &mut winners,
                &mut terminal,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(winners, 1, "only the first terminal publisher may win");
        assert_eq!(terminal, first);
    }
}

#[test]
fn fatal_capsule_keeps_exact_lineage_and_quorum_evidence() {
    let mut capsule = FatalCapsule::default();
    let status =
        unsafe { lfm_internal_engine_fatal_capsule_for_test(123, 0xff, 0x7f, 0x3f, &mut capsule) };
    assert_eq!(status, 0);
    assert_eq!(capsule.abi_version, 2);
    assert_eq!(capsule.request, 4);
    assert_eq!(capsule.stage, 2);
    assert_eq!(capsule.program_kind, 1);
    assert_eq!(capsule.program_phase, 1);
    assert_eq!(capsule.program_flags, 0x5a);
    assert_eq!(capsule.quorum_status, 0);
    assert_eq!(
        capsule.workflow,
        Ticket {
            runtime_epoch: 11,
            sequence: 21,
            generation: 31,
            kind: 7,
        }
    );
    assert_eq!(
        capsule.pass,
        Ticket {
            runtime_epoch: 11,
            sequence: 22,
            generation: 32,
            kind: 4,
        }
    );
    assert_eq!(
        capsule.deadline,
        Ticket {
            runtime_epoch: 11,
            sequence: 23,
            generation: 33,
            kind: 9,
        }
    );
    assert_eq!(capsule.conversation_id, 44);
    assert_eq!(capsule.epoch, 55);
    assert_eq!(capsule.scope_generation, 66);
    assert_eq!(capsule.team_generation, 123);
    assert_eq!(capsule.program_outer, 12);
    assert_eq!(capsule.program_inner, 13);
    assert_eq!(
        (capsule.shape0, capsule.shape1, capsule.shape2),
        (2048, 8192, 7)
    );
    assert_eq!(capsule.expected_mask, 0xff);
    assert_eq!(capsule.entered_mask, 0x7f);
    assert_eq!(capsule.returned_mask, 0x3f);
    assert_eq!(capsule.never_entered_mask, 0x80);
    assert_eq!(capsule.entered_not_returned_mask, 0x40);
    assert_eq!(capsule.armed_ns, 1000);
    assert_eq!(capsule.hard_budget_ns, 1_000_000_000);
    assert_eq!(capsule.elapsed_ns, capsule.hard_budget_ns + 123);
    assert_eq!(capsule.deadline_event_sequence, 77);
    assert_eq!(capsule.scheduled_arm_generation, 88);
    assert_eq!(capsule.current_arm_generation, 88);

    let invalid =
        unsafe { lfm_internal_engine_fatal_capsule_for_test(123, 0b0011, 0b0100, 0, &mut capsule) };
    assert_eq!(invalid, -22);
}

#[test]
#[cfg(unix)]
fn hard_timeout_fault_injection_is_process_fatal() {
    if let Ok(point) = std::env::var("LFM_HARD_TIMEOUT_CHILD") {
        let point = point.parse().expect("fault point");
        let path = std::env::var("LFM_HARD_TIMEOUT_PATH").expect("fatal capsule path");
        fatal_child(point, &path);
    }

    let cases = [
        (NEVER_ENTERED, 0b1101, 0b1101, 0b0010, 0),
        (ENTERED_NEVER_RETURNED, 0b1111, 0b1101, 0, 0b0010),
    ];
    for (point, entered, returned, never, hung) in cases {
        let capsule = run_fatal_child(point);
        assert_eq!(capsule.abi_version, 2);
        assert_eq!(capsule.request, 15);
        assert_eq!(capsule.quorum_status, 0);
        assert_eq!(capsule.team_generation, 1);
        assert_eq!(capsule.expected_mask, 0b1111);
        assert_eq!(capsule.entered_mask, entered);
        assert_eq!(capsule.returned_mask, returned);
        assert_eq!(capsule.never_entered_mask, never);
        assert_eq!(capsule.entered_not_returned_mask, hung);
        assert_eq!(capsule.hard_budget_ns, 1_000_000_000);
        assert!(capsule.elapsed_ns >= capsule.hard_budget_ns);
        assert_eq!(capsule.deadline.kind, 9);
        assert_eq!(capsule.pass.kind, 4);
        assert_eq!(capsule.workflow.kind, 7);
        assert_eq!(
            capsule.scheduled_arm_generation,
            capsule.current_arm_generation
        );
    }
}
