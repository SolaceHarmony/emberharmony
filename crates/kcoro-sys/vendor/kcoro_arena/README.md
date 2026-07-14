# kcoro_arena

`kcoro_arena` is a C11 stackless coroutine runtime built around explicit
runtime ownership, retained operation records, and event-driven continuation
wakes. Its in-memory runtime, portable WAL, at-least-once delivery, serializable
workflows, and host adapter boundaries are active. It remains `0.x` while the
release-hardening and ABI-stability gates continue. Its first GPU-like substrate
is active: precise work signaling, zero-spin expected-value waits, a preallocated
ticket slab, and exact completion callbacks. EmberHarmony provides the first
fixed-lane integration; generic broker submission and native recurrence remain.

The ticket, precise-wake, prepared wait-word, and build-identity claims in this
tree are anchored by implementation commit
`bcdc03d1a0731ee3116c850f3f9bd7cb27b55101`. Later documentation commits may
describe that code, but an uncommitted working tree is never cited as evidence.

The canonical public include is:

```c
#include "kcoro_arena.h"
```

## Current Runtime

- `kc_runtime` owns workers, the ready queue, operation identities, shutdown,
  and lifecycle accounting. There is no production process-global scheduler
  except the compatibility wrapper used by the small `koro_*` API.
- Every channel wait is a retained `kc_op`. Match, close, and cancellation
  arbitrate one terminal result while holding the channel mutex, then publish
  the wake after releasing it.
- Continuations use `NEW`, `QUEUED`, `RUNNING`, `WAITING`, and `DONE` states.
  Park and wake transitions are serialized so one continuation cannot execute
  concurrently or lose a completion doorbell.
- Rendezvous, bounded FIFO, growable unlimited, and conflated channels use the
  same operation queues.
- Cancellation trees, deadlines, timers, select clauses, joins, and scopes all
  terminate through the same retained operation arbitration. Actors are scope
  children that park on mailbox receives instead of polling.
- Bounded administration calls expose runtime, memory, operation, channel,
  timer, descriptor, and scope snapshots. Public configuration and snapshot
  records are size/version checked. The baseline capability function still
  reports compiled optional services unconditionally; configured-service bits
  are a required next-ABI repair.
- The core calls the compile-time `kc_port_*` contract. The BSD POSIX adapter
  under `port/` is linked separately and no OS threading symbol is admitted to
  the core archive.
- Work arrivals signal one `work_cv` waiter. Idle/join/stop predicates use a
  separate `lifecycle_cv`, so lifecycle transitions do not wake worker herds.
- Tickets are generation-protected slab entries with retained descriptor leases,
  checked ID completion/cancellation, and one reserved terminal-delivery
  reference. Completion queues the ticket itself and invokes its callback on an
  arena worker exactly once.
- The POSIX adapter prepares direct raw-`uint32_t` expected-value wait handles
  once, then provides one/all wake, timeout, and entered-waiter teardown without
  registry lookup or polling.
- Copy-mode payloads use generation-protected descriptors over reclaimable,
  runtime-owned aligned segments. Borrowed and host-owned regions hand off
  pointers directly; host release callbacks run exactly once after the final
  descriptor lease.
- The portable WAL uses explicit little-endian records, CRC32C, transaction
  begin/commit markers, and host-confirmed sync. Torn tail transactions are
  removed on open; complete corrupt records fail recovery.
- Durable messages persist publication, delivery attempt, acknowledgement,
  retry, and dead-letter transitions. Unacknowledged work is redelivered after
  restart; external consumers must be idempotent.
- Workflow definitions are registered before recovery. Only stable
  type/version IDs, instance/state IDs, encoded state, correlations, and
  commands are persisted. Input acknowledgement, workflow state, and emitted
  messages share one WAL transaction.
- Transport, storage, and shared-region providers are direct link-time host
  contracts. The core archive contains no filesystem, socket, or IPC backend.

```text
host application
    -> kc_runtime
       -> worker-ready queue -> stackless continuation
       -> operation registry -> kc_op
          -> channel/scope/timer waiter or buffered descriptor
          -> one terminal claim
          -> continuation wake
       -> ticket slab -> kc_ticket
          -> descriptor lease + terminal disposition
          -> intrusive completion queue -> exact callback
    -> host-linked kc_port_* adapter
    -> kc_wal -> host-linked kc_store_* storage
       -> kc_durable -> kc_workflows
       -> kc_delivery -> host-linked kc_transport_*
    -> kc_shared_payload -> host-linked kc_region_provider_*
```

Terminal callbacks run on arena workers; there is no separate callback
token-kernel thread or pointer-token table. Stable correlation lives in
runtime-epoch/sequence operation and ticket IDs, and queues retain their
operation or ticket records directly.

## Build And Verify

```sh
make clean
make test
make -C tests test-full
make test-race
make test-soak
make check-symbols
make check-licenses
make check-build-identity
```

On Clang platforms with AddressSanitizer and UndefinedBehaviorSanitizer:

```sh
make clean
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make DEBUG=1 \
  EXTRA_CFLAGS='-fsanitize=address,undefined' \
  EXTRA_LDFLAGS='-fsanitize=address,undefined' test
```

`scripts/check_core_symbols.sh` rejects direct OS dependencies in
`libkcoro_arena.a`. TSan runs on macOS and Linux where the selected toolchain
supports it; Linux remains the gate for leak detection and the libFuzzer smoke
runs (`make -C fuzz run`).

## Source Map

- `include/kcoro_arena.h`: canonical C API entry point
- `core/src/kc_runtime.c`: explicit scheduler lifecycle and workers
- `core/src/kc_op.c`: retained operation identity and terminal publication
- `core/src/kc_ticket.c`: pooled action receipts and exact callback delivery
- `core/src/kc_chan_stackless.c`: all active channel policies
- `core/src/kc_desc.c`: descriptor leases, regions, and segment reclamation
- `core/src/kcoro_stackless.c`: continuation and begin/finish operations
- `core/src/kc_scope.c`: structured child ownership and joins
- `core/src/kc_actor.c`: parked mailbox consumers
- `core/src/kc_admin.c`: bounded runtime administration
- `core/src/kc_wal.c`: portable transactional WAL and snapshot ordering
- `core/src/kc_durable.c`: durable message state and checkpoint section
- `core/src/kc_workflow.c`: serializable workflow state machines
- `core/src/kc_transport.c`: acknowledgement-driven delivery pump
- `core/src/kc_shared.c`: portable shared-region references and leases
- `port/`: optional host adapter, excluded from the core archive
- `include/kc_atomic.h`: shared raw-word atomic operations for C/C++ doorbells
- `tests/`: production-backed tests only

See [Architecture](docs/ARCHITECTURE.md), [Durability](docs/DURABILITY.md),
[Host Adapters](docs/HOST_ADAPTERS.md), [Runtime Backport](docs/KCORO_RUNTIME_BACKPORT.md),
[GPU-Like Execution Kernel](docs/GPU_KERNEL_CONTRACT.md),
[Tickets And Completion Callbacks](docs/TICKETS_AND_CALLBACKS.md), and
[Roadmap](docs/ROADMAP.md).

The process-global compatibility surface is not canonical design and is removed
after explicit-runtime callers migrate. Git commits are the archive; retired
source and documentation are not copied into the release tree.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
