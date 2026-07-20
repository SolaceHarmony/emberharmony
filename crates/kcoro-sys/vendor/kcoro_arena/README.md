# kcoro_arena

`kcoro_arena` is EmberHarmony's callback-driven coordination kernel. It owns a
resident runtime, retained services, and fixed numerical teams. Computational
progress comes only from publication: a producer edge, a completed fixed-team
generation, released capacity, cancellation, or control. Exact product tickets
and borrowed byte views live in Flashkern; kcoro does not duplicate them.

There are no channels, actors, work stealing, generic coroutine spawning,
process-global runtimes, timers, sleeps, deadline completions, WALs, workflows,
or transport pumps in this production snapshot. Git is the archive; deleted
mechanisms are not retained as compatibility surfaces.

The canonical public include is:

```c
#include "kcoro_arena.h"
```

## Runtime contract

- `kc_runtime` owns a fixed worker set. Every retained service is permanently
  bound to one worker, and every worker owns a private ready bitmap and idle
  doorbell. A resident worker may park only when its private bitmap is empty.
  Administrative teardown uses `join_all`, stop, and join; it never advances
  work.
- `kc_service` is one retained stackless state machine. Producer-specific
  realtime notifier leases publish lock-free edges. The callback drains its
  durable predicate and returns dormant, locally ready again, or terminal.
- `kc_team` owns fixed, non-stealing numerical workers. One generation is one
  phase. Every member returns; the final return invokes the completion callback,
  which advances the durable program or publishes its product ticket. No member
  waits on another member inside a production phase.
- `kc_doorbell` is a cache-isolated generation edge. It is private kernel idle
  capacity, not an operation-level wait API.

Wall-clock timestamps are telemetry. A separately named device-liveness
watchdog may publish a fault edge, but time never makes a continuation runnable
and never advances speech, inference, or route state.

```text
producer callback
    -> realtime notifier edge
       -> retained kc_service state machine
          -> product ticket / fixed-team generation
             -> final member return
                -> completion callback
                   -> next retained phase or terminal publication
```

## Source map

- `include/kcoro_arena.h`: canonical narrow C surface
- `core/src/kc_runtime.c`: resident worker lifecycle and callback delivery
- `core/src/kc_service.c`: retained stackless services and realtime edges
- `core/src/kc_continuation.c`: private service continuation record
- `core/src/kc_team.c`: fixed-member non-stealing execution
- `core/src/kc_doorbell.c`: cache-isolated idle generation edge
- `port/posix.c`: host thread and expected-value dormancy adapter

The core calls the compile-time `kc_port_*` contract; the host adapter is linked
as a separate archive.

## Verification

From the EmberHarmony repository root:

```sh
cargo test -p kcoro-sys
```

The tests execute the real native runtime and source-gate the absence of the
retired scheduler, channel, timer, and persistence APIs.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
