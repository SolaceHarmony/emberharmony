use std::mem::size_of;

pub const ABI_VERSION: u32 = 1;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketKind {
    Session = 1,
    Turn = 2,
    Frame = 3,
    Pass = 4,
    ContextSwitch = 5,
    Checkpoint = 6,
    Workflow = 7,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandKind {
    RunPass = 1,
    RunStandingOrder = 2,
    SetAttention = 3,
    Pause = 4,
    Resume = 5,
    Cancel = 6,
    Stop = 7,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceClass {
    Deadline = 1,
    Interactive = 2,
    Background = 3,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Execution {
    NotDispatched = 0,
    Completed = 1,
    Failed = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    None = 0,
    Committed = 1,
    RolledBack = 2,
    Poisoned = 3,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Publication {
    None = 0,
    Committed = 1,
    Stale = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cause {
    Success = 0,
    Rejected = 1,
    Canceled = 2,
    TimedOut = 3,
    StaleEpoch = 4,
    Stop = 5,
    Fault = 6,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalResultKind {
    None = 0,
    TextToken = 1,
    AudioCodes = 2,
    Frame = 3,
    Control = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalResultError;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TicketId {
    pub runtime_epoch: u64,
    pub sequence: u64,
    pub generation: u32,
    pub kind: u32,
}

impl TicketId {
    pub const NONE: Self = Self {
        runtime_epoch: 0,
        sequence: 0,
        generation: 0,
        kind: 0,
    };

    pub const fn new(runtime_epoch: u64, sequence: u64, generation: u32, kind: TicketKind) -> Self {
        Self {
            runtime_epoch,
            sequence,
            generation,
            kind: kind as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DescriptorId {
    pub slot: u32,
    pub generation: u32,
}

impl DescriptorId {
    pub const NONE: Self = Self {
        slot: u32::MAX,
        generation: 0,
    };

    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

/// One inline command cell. Tensor, PCM, KV, and weight bytes never appear here.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Submission {
    pub size: u32,
    pub abi_version: u32,
    pub ticket: TicketId,
    pub parent: TicketId,
    pub conversation_id: u64,
    pub epoch: u64,
    pub descriptor: DescriptorId,
    pub command: u32,
    pub service_class: u32,
    pub flags: u32,
    pub pass_budget: u32,
    pub deadline_ns: u64,
    pub reserved: [u64; 3],
}

impl Submission {
    pub fn new(
        ticket: TicketId,
        parent: TicketId,
        conversation_id: u64,
        epoch: u64,
        descriptor: DescriptorId,
        command: CommandKind,
        service_class: ServiceClass,
    ) -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION,
            ticket,
            parent,
            conversation_id,
            epoch,
            descriptor,
            command: command as u32,
            service_class: service_class as u32,
            flags: 0,
            pass_budget: 1,
            deadline_ns: 0,
            reserved: [0; 3],
        }
    }

    pub fn is_compatible(&self) -> bool {
        self.abi_version == ABI_VERSION && self.size as usize == size_of::<Self>()
    }
}

/// One completion cell preserving execution, state, publication, and cause.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Completion {
    pub size: u32,
    pub abi_version: u32,
    pub ticket: TicketId,
    pub conversation_id: u64,
    pub epoch: u64,
    pub pass_id: u64,
    pub execution: u32,
    pub state: u32,
    pub publication: u32,
    pub cause: u32,
    pub status: i32,
    pub flags: u32,
    pub result_kind: u32,
    pub result_count: u32,
    pub results: [u32; 8],
    pub reserved: u64,
}

impl Completion {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ticket: TicketId,
        conversation_id: u64,
        epoch: u64,
        pass_id: u64,
        execution: Execution,
        state: State,
        publication: Publication,
        cause: Cause,
    ) -> Self {
        Self {
            size: size_of::<Self>() as u32,
            abi_version: ABI_VERSION,
            ticket,
            conversation_id,
            epoch,
            pass_id,
            execution: execution as u32,
            state: state as u32,
            publication: publication as u32,
            cause: cause as u32,
            status: 0,
            flags: 0,
            result_kind: TerminalResultKind::None as u32,
            result_count: 0,
            results: [0; 8],
            reserved: 0,
        }
    }

    pub fn set_results(
        &mut self,
        kind: TerminalResultKind,
        values: &[u32],
    ) -> Result<(), TerminalResultError> {
        if values.len() > self.results.len() {
            return Err(TerminalResultError);
        }
        self.result_kind = kind as u32;
        self.result_count = values.len() as u32;
        self.results.fill(0);
        self.results[..values.len()].copy_from_slice(values);
        Ok(())
    }

    pub fn is_compatible(&self) -> bool {
        self.abi_version == ABI_VERSION && self.size as usize == size_of::<Self>()
    }
}

const _: [(); 24] = [(); size_of::<TicketId>()];
const _: [(); 8] = [(); size_of::<DescriptorId>()];
const _: [(); 128] = [(); size_of::<Submission>()];
const _: [(); 128] = [(); size_of::<Completion>()];
