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

The native substrate now also owns fixed compute teams. `kc_team` creates,
parks, dispatches, stops, and joins stable members without work stealing;
`kc_collective` supplies generation reconvergence and exactly-once
last-arrival transitions; `kc_doorbell` is the cache-isolated expected-value
edge shared by those primitives. Flashkern supplies numerical member programs,
not OS-thread lifecycle or a private barrier implementation.

Wall-clock time has two non-progress roles: latency telemetry and device/storage
liveness faults. Speech durations are capture sample counts. No interval timer,
timeout receive, or periodic probe is an inference-progress source.

The crate owns policy and lifecycle. It does not own model arithmetic, PCM,
weights, activations, KV, mel, sampling, or codec buffers. Those stay in the
native engine and are named by generation-protected descriptor IDs.

`kcoro-sys` supplies the C native substrate and its conformance tests. Flashkern
has adopted its fixed-team and collective ownership in the working tree. The
bridge dispatcher and route broker remain transitional native pthread loops and
must become kcoro-owned continuations before the ownership cutover is complete.
The Rust coordinator must use the ring ABI and must never execute on a fixed
compute lane or audio callback.
