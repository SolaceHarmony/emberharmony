//! Callback-driven coordination primitives for EmberHarmony's resident runtime.
//!
//! This crate owns policy scheduling only. Numerical payloads remain in native
//! memory and cross the docking boundary as generation-protected descriptors.

mod executor;
mod promise;
mod protocol;
mod ring;
mod scope;

pub use executor::{
    Config as ExecutorConfig, CreateError, Executor, Handle, JoinError, SpawnError, Stats,
    TaskHandle, TaskId, TaskResult,
};
pub use promise::{promise, Promise, Resolver};
pub use protocol::{
    Cause, CommandKind, Completion, DescriptorId, Execution, Publication, ServiceClass, State,
    Submission, TerminalResultError, TerminalResultKind, TicketId, TicketKind, ABI_VERSION,
};
pub use ring::{
    ring, Receiver, RecvError, RecvFuture, RingError, SendFuture, Sender, TryRecvError,
    TrySendError,
};
pub use scope::{Control, ControlSnapshot, Scope};
