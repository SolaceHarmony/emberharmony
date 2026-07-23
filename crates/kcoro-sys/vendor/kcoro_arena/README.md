# kcoro_arena

`kcoro_arena` is EmberHarmony's callback-driven coordination kernel. It owns a
resident runtime, retained services, and fixed numerical teams. Computational
progress comes only from publication: a producer edge, a completed fixed-team
generation, released capacity, cancellation, or control. Exact product tickets
and fixed coordination storage live in kcoro; numerical buffer views live in
Flashkern and are never copied into the coordination layer.

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
- `include/kc_coordination.hpp`: unversioned C++23 ticket, admission, and
  generation-lease primitives
- `include/kc_mailbox.hpp`: fixed typed request/completion exchange with
  sequence-stamped cells, endpoint leases, and correlated callback edges
- `include/kc_team_executor.hpp`: retained mailbox-to-team continuation with
  exact generation callbacks and asynchronous retirement
- `include/kc_permit_broker.hpp`: fixed fair permit pool with one retained
  continuation per admitted operation, exact completion resumption, and
  callback-published capacity
- `include/kc_team_supervisor.hpp`: correlated per-generation hard supervision,
  exact completion/expiry arbitration, and quorum-failure capture
- `include/kc_fatal_store.hpp`: prefaulted, locked, durable fatal-record
  publication with no failure-path allocation or storage syscall
- `core/src/kc_runtime.c`: bounded ready board, worker lifecycle, and callback delivery
- `core/src/kcoro_stackless.c`: saved program counter/frame and exact-ticket resume
- `core/src/kc_service.c`: retained stackless services and realtime edges
- `core/src/kc_team.c`: fixed logical-member execution on the runtime pool
- `core/src/kc_doorbell.c`: cache-isolated idle generation edge
- `port/posix.c`: host thread and expected-value dormancy adapter

The core calls the compile-time `kc_port_*` contract; the host adapter is linked
as a separate archive.

## Verification

The authoritative kernel tests are native C++23 programs. Build outside the
source tree:

```sh
cmake -S crates/kcoro-sys/vendor/kcoro_arena \
      -B /tmp/emberharmony-kcoro-build
cmake --build /tmp/emberharmony-kcoro-build --parallel
ctest --test-dir /tmp/emberharmony-kcoro-build --output-on-failure
```

The native gates prove the complete substrate contract: bounded registration
and generation-safe slot reuse; exact callback correlation; saved-frame
migration between eligible workers; one fixed OS-worker population with no
per-operation threads; notify-during-execution handoff without a lost edge;
sub-percent idle-worker CPU before and after work; and four logical team
members completing one exact quorum generation over two physical workers.
They also prove bounded admission publication, permanent-stop precedence over
temporary sealing, unique canonical ticket minting, and stale fixed-slot lease
rejection. The team-executor contract saturates its mailbox, advances one
request across two quorum generations, rejects another request through its
correlated completion, drains coalesced edges, and retires without a waiter.
The permit-broker contract proves service-class/FIFO/age ordering, age zero for
records newer than the selector snapshot, exact stale-completion rejection,
two concurrently dehydrated operations, completion during execution without a
lost edge, retained multi-hop frames, and callback-driven retirement.
The team-supervisor contract proves an unbudgeted generation cannot dispatch,
normal completion retires the exact deadline, completion and expiry publish
one terminal state, never-entered and entered-never-returned lanes produce
exact quorum masks, and fatal evidence survives the required process abort in
the durable store.
Rust bindings are not the authority for the native kernel.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
