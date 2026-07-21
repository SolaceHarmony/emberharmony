# kcoro_arena

`kcoro_arena` is EmberHarmony's callback-driven coordination kernel. It owns a
resident runtime, retained services, and fixed numerical teams. Computational
progress comes only from publication: a producer edge, a completed fixed-team
generation, released capacity, cancellation, or control. Exact product tickets
and borrowed byte views live in Flashkern; kcoro does not duplicate them.

There are no channels, actors, work stealing, process-global runtimes, sleeps,
interval-timer progress, WALs, workflows, or transport pumps in this production
snapshot. Correlated one-shot deadlines are external callback sources, never a
scheduler. Git is the archive; deleted mechanisms are not retained as
compatibility surfaces.

The canonical public include is:

```c
#include "kcoro_arena.h"
```

## Runtime contract

- `kc_runtime` owns one fixed worker set and one bounded ready board. A logical
  continuation stores its program counter and fixed heap frame independently
  of workers; any free eligible worker may resume it. A resident worker may
  enter infrastructure dormancy only when the shared runnable board is empty.
  Administrative teardown never advances work.
- `kc_service` is one retained stackless state machine. Producer-specific
  realtime notifier leases publish lock-free edges. The callback drains its
  durable predicate and returns dormant, locally ready again, or terminal.
- `kc_team` is a fixed set of logical lane continuations on that same pool. One
  generation is one phase. Every member returns; the final return publishes a
  callback that resumes the suspended orchestration frame. No member owns a
  pthread or waits on another member inside a production phase.
- `kc_doorbell` is a cache-isolated generation edge. It is private kernel idle
  capacity, not an operation-level wait API.

Wall-clock timestamps are telemetry. A separately named device-liveness
watchdog may publish a fault edge, but time never makes a continuation runnable
and never advances speech, inference, or route state.

```text
producer callback
    -> realtime notifier edge
       -> exact saved continuation frame
          -> product ticket / fixed-team generation
             -> final member return
                -> correlated callback resumes the frame
                   -> next retained phase or terminal publication
```

## Source map

- `include/kcoro_arena.h`: canonical narrow C surface
- `core/src/kc_runtime.c`: bounded ready board, worker lifecycle, and callback delivery
- `core/src/kcoro_stackless.c`: saved program counter/frame and exact-ticket resume
- `core/src/kc_service.c`: retained stackless services and realtime edges
- `core/src/kc_team.c`: fixed logical-member execution on the runtime pool
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
