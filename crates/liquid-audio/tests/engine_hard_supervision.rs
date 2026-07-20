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
    workflow: Ticket,
    pass: Ticket,
    deadline: Ticket,
    conversation_id: u64,
    epoch: u64,
    scope_generation: u64,
    team_generation: u64,
    expected_mask: u64,
    entered_mask: u64,
    returned_mask: u64,
    hard_budget_ns: u64,
    elapsed_floor_ns: u64,
    deadline_event_sequence: u64,
    scheduled_arm_generation: u64,
    current_arm_generation: u64,
}

impl Default for FatalCapsule {
    fn default() -> Self {
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: 1,
            request: 0,
            stage: 0,
            workflow: Ticket::default(),
            pass: Ticket::default(),
            deadline: Ticket::default(),
            conversation_id: 0,
            epoch: 0,
            scope_generation: 0,
            team_generation: 0,
            expected_mask: 0,
            entered_mask: 0,
            returned_mask: 0,
            hard_budget_ns: 0,
            elapsed_floor_ns: 0,
            deadline_event_sequence: 0,
            scheduled_arm_generation: 0,
            current_arm_generation: 0,
        }
    }
}

unsafe extern "C" {
    fn lfm_internal_engine_team_terminal_race_for_test(
        generation: u64,
        first: u32,
        second: u32,
        winner_bits: *mut u32,
        terminal_state: *mut u32,
    ) -> i32;
    fn lfm_internal_engine_deferred_handshake_for_test(
        generation: u64,
        ack_first: u32,
        after_first: *mut u64,
        after_second: *mut u64,
        after_duplicate: *mut u64,
    ) -> i32;
    fn lfm_internal_engine_fatal_capsule_for_test(
        generation: u64,
        expected_mask: u64,
        entered_mask: u64,
        returned_mask: u64,
        out: *mut FatalCapsule,
    ) -> i32;
}

#[test]
fn completion_deferral_resumes_only_after_both_edges_and_only_once() {
    /* Keep the owning crate in this integration-test link so its private
     * native archive (where these probes live) is retained by the linker. */
    let _ = liquid_audio::NativeVoiceSampling::default();
    for ack_first in [0, 1] {
        let mut first = u64::MAX;
        let mut second = u64::MAX;
        let mut duplicate = u64::MAX;
        let status = unsafe {
            lfm_internal_engine_deferred_handshake_for_test(
                0x5678,
                ack_first,
                &mut first,
                &mut second,
                &mut duplicate,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(first, 0, "one edge alone must not advance the generation");
        assert_eq!(
            second, 0x5678,
            "the second edge resumes the exact generation"
        );
        assert_eq!(duplicate, 0, "the generation may resume only once");
    }
}

const COMPLETED: u32 = 2;
const TIMED_OUT: u32 = 3;

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
    assert_eq!(capsule.request, 4);
    assert_eq!(capsule.stage, 2);
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
    assert_eq!(capsule.expected_mask, 0xff);
    assert_eq!(capsule.entered_mask, 0x7f);
    assert_eq!(capsule.returned_mask, 0x3f);
    assert_eq!(capsule.hard_budget_ns, 1_000_000_000);
    assert_eq!(capsule.elapsed_floor_ns, capsule.hard_budget_ns);
    assert_eq!(capsule.deadline_event_sequence, 77);
    assert_eq!(capsule.scheduled_arm_generation, 88);
    assert_eq!(capsule.current_arm_generation, 88);

    let invalid =
        unsafe { lfm_internal_engine_fatal_capsule_for_test(123, 0b0011, 0b0100, 0, &mut capsule) };
    assert_eq!(invalid, -22);
}
