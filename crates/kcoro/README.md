# kcoro

`kcoro` is EmberHarmony's Rust coordination kernel. It is deliberately smaller
than a general async runtime:

- fixed task capacity and a preallocated intrusive-ready equivalent;
- dedicated resident workers with bounded draining;
- exact-once promises that wake registered continuations;
- bounded SPSC rings for the Flashkern SQ/CQ and Tauri docking boundaries;
- inherited pause/cancel scope words with generation fencing;
- versioned, fixed-size control records containing descriptors, never payloads.

Completion cells carry terminal facts and at most eight inline token/codebook
IDs. Timing and queue telemetry belong to the sampled observer plane, not the
progress-bearing completion ring.

The crate owns policy and lifecycle. It does not own model arithmetic, PCM,
weights, activations, KV, mel, sampling, or codec buffers. Those stay in the
native engine and are named by generation-protected descriptor IDs.

`kcoro-sys` remains the C conformance oracle and supplies the native wait-word
substrate during migration. Production mounting into Flashkern is a separate
gate: the Rust coordinator must use the ring ABI and must never execute on a
fixed compute lane or audio callback.
