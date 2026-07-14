use kcoro::{
    ring, Cause, CommandKind, Completion, DescriptorId, Execution, Publication, ServiceClass,
    State, Submission, TerminalResultKind, TicketId, TicketKind, TryRecvError, TrySendError,
};
use std::mem::{align_of, offset_of, size_of};

fn submission(sequence: u64) -> Submission {
    Submission::new(
        TicketId::new(7, sequence, 3, TicketKind::Pass),
        TicketId::new(7, 1, 1, TicketKind::Turn),
        99,
        11,
        DescriptorId::new(sequence as u32, 5),
        CommandKind::RunPass,
        ServiceClass::Interactive,
    )
}

#[test]
fn submission_ring_is_bounded_ordered_and_wraps() {
    let (mut sender, mut receiver) = ring(4).unwrap();
    for sequence in 1..=4 {
        sender.try_send(submission(sequence)).unwrap();
    }
    assert!(matches!(
        sender.try_send(submission(5)),
        Err(TrySendError::Full(_))
    ));
    assert_eq!(receiver.try_recv().unwrap().ticket.sequence, 1);
    assert_eq!(receiver.try_recv().unwrap().ticket.sequence, 2);
    sender.try_send(submission(5)).unwrap();
    sender.try_send(submission(6)).unwrap();
    for sequence in 3..=6 {
        let value = receiver.try_recv().unwrap();
        assert_eq!(value.ticket.sequence, sequence);
        assert!(value.is_compatible());
    }
    assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
}

#[test]
fn completion_cell_preserves_all_terminal_facts() {
    let ticket = TicketId::new(7, 8, 9, TicketKind::Pass);
    let mut completion = Completion::new(
        ticket,
        99,
        12,
        44,
        Execution::Completed,
        State::Committed,
        Publication::Stale,
        Cause::Canceled,
    );
    completion
        .set_results(
            TerminalResultKind::AudioCodes,
            &[4, 8, 15, 16, 23, 42, 9, 7],
        )
        .unwrap();
    assert!(completion.is_compatible());
    assert_eq!(completion.execution, Execution::Completed as u32);
    assert_eq!(completion.state, State::Committed as u32);
    assert_eq!(completion.publication, Publication::Stale as u32);
    assert_eq!(completion.cause, Cause::Canceled as u32);
    assert_eq!(
        completion.result_kind,
        TerminalResultKind::AudioCodes as u32
    );
    assert_eq!(completion.result_count, 8);
    assert_eq!(completion.results, [4, 8, 15, 16, 23, 42, 9, 7]);
    assert_eq!(size_of::<Submission>(), 128);
    assert_eq!(size_of::<Completion>(), 128);
    assert_eq!(align_of::<Submission>(), 64);
    assert_eq!(align_of::<Completion>(), 64);
    assert_eq!(offset_of!(Submission, ticket), 8);
    assert_eq!(offset_of!(Submission, parent), 32);
    assert_eq!(offset_of!(Submission, conversation_id), 56);
    assert_eq!(offset_of!(Submission, descriptor), 72);
    assert_eq!(offset_of!(Submission, deadline_ns), 96);
    assert_eq!(offset_of!(Completion, ticket), 8);
    assert_eq!(offset_of!(Completion, conversation_id), 32);
    assert_eq!(offset_of!(Completion, execution), 56);
    assert_eq!(offset_of!(Completion, results), 88);
    assert_eq!(offset_of!(Completion, reserved), 120);
}

#[test]
fn completion_rejects_more_than_eight_inline_results() {
    let mut completion = Completion::new(
        TicketId::new(1, 1, 1, TicketKind::Pass),
        2,
        3,
        4,
        Execution::Completed,
        State::Committed,
        Publication::Committed,
        Cause::Success,
    );
    assert!(completion
        .set_results(TerminalResultKind::AudioCodes, &[0; 9])
        .is_err());
    assert_eq!(completion.result_count, 0);
}

#[test]
fn descriptor_identity_is_inline_and_generation_protected() {
    let first = DescriptorId::new(4, 8);
    let recycled = DescriptorId::new(4, 9);
    assert_ne!(first, recycled);
    assert_eq!(size_of::<DescriptorId>(), 8);
}
