# Vendored kcoro — provenance and local patches

Vendored 2026-07-08 from Sydney's kcoro tree (`/Volumes/stuff/Projects/kcoro`,
BSD-3-Clause): `include/*.h`, `core/src/*.{c,h}`, `arch/{aarch64,x86_64}/kc_ctx_switch.S`.
Built by this crate's `build.rs` with the upstream Makefile's flags
(`-std=c11 -O2 -pthread -D_GNU_SOURCE -DKC_SCHED=1`). The prototype channel chassis
(`crates/liquid-audio/src/compute/flashkern/engine.rs`) and resident native stage machine
(`crates/liquid-audio/native/src/engine/flashkern_engine.cpp`)
are the consumers.

## Patch 0001 — park/unpark lost-wakeup race (candidate for upstream)

**Symptom.** Rendezvous-channel producer/consumer deadlocks: both sides parked forever.
Reproducer: 8 consumer coroutines + 1 feeder over one `KC_RENDEZVOUS` channel hung
within the first 32 sends, every scheduler worker asleep on `park_cv`.

**Root cause.** A channel waiter enqueues itself and unlocks the channel mutex *before*
calling `kcoro_park()`. A wake landing inside that window was dropped on both legs:
`kcoro_unpark` ignored non-parked coroutines, and `kc_sched_enqueue_ready` skips
`KCORO_RUNNING` ones. `co->state` is a plain int and cannot order the wake against the
park switch.

**Fix — a three-state park gate** (`atomic_int park_notify` on `kcoro_t`; the same
protocol as Rust's thread parker, all transitions seq_cst on the one atomic):

- `kcoro_park`: exchange the gate to `PARKED` *first*; if it held `NOTIFIED`, consume
  it and return without parking (a spurious return — every park caller loops and
  re-checks). On resume, retire the cycle by storing `EMPTY`.
- `kcoro_unpark`: exchange the gate to `NOTIFIED`; only a previous value of `PARKED`
  readies + enqueues the coroutine. Any other value means the target is running: the
  token is left for its next `kcoro_park` to consume. If the target is still mid-switch
  when enqueued, the scheduler's `running_flag` CAS keeps re-queueing it until the
  switch completes, so the wake cannot be lost.
- Scheduler worker loop (defense in depth): after `kcoro_resume` returns — the one
  point strictly after the coroutine has switched out — a coroutine left `PARKED` with
  a `NOTIFIED` gate is promoted back to `READY` (CAS consumes only `NOTIFIED`, never
  the coroutine's own `PARKED` marker).
- `kc_chan_schedule_wake` routes through `kcoro_unpark` unconditionally so the
  not-yet-parked case reaches the gate; the trailing `kc_sched_enqueue_ready` stays for
  suspended/yielded waiters (`ready_enqueued` dedups).

Note: rendezvous *send* treats return-from-park as "committed". A stale coalesced token
can now cause a send to return one commit early; the payload was already heap-staged in
the waiter at enqueue, so delivery still happens — semantics shade toward buffered for
that one message. Acceptable for descriptor traffic; flagged for upstream review.

Also out of scope here: the `kc_select` wake paths gate on `kcoro_is_parked()` in
several places and likely have the same class of race for select users.

## Patch 0002 — fiber-unsafe TLS: post-switch writes poison the old thread

**Symptom.** Patch 0001 alone did not fix the reproducer: wakes that were provably
generated (`schedule_wake … parked=1` in `KCORO_TRACE`) targeted coroutines that never
ran again; hang points moved around nondeterministically (7/32, 10/32, 29/32).
ThreadSanitizer pointed at it: `kcoro_park` on one thread writing the same address
`kcoro_set_thread_main` writes on another.

**Root cause.** `current_kcoro` / `main_kcoro` are `__thread`, and a compiler may
legally cache a TLS variable's address for the lifetime of a stack frame — C assumes a
frame never changes threads. A coroutine frame DOES change threads across
`kcoro_switch` under the M:N scheduler. The post-switch tails of `kcoro_yield`,
`kcoro_yield_to`, and `kcoro_park` did `current_kcoro = current` with the *old*
thread's cached TLS address, overwriting that thread's current-coroutine pointer.
The old thread then registered the wrong coroutine as a channel waiter, its wakes went
to the wrong target, and the real waiter slept forever. `kcoro_trampoline`'s exit had
the same defect for `main_kcoro` — after migration it could switch a finished coroutine
into a *different thread's* main context.

**Fix.**

- Deleted the post-switch `current_kcoro = current` writes in `kcoro_yield`,
  `kcoro_yield_to`, and `kcoro_park`. They were redundant by construction — the
  resuming thread's `kcoro_resume` sets its own `current_kcoro = co` *before* switching
  in — so their only possible effect was cross-thread corruption.
- `kcoro_trampoline`'s exit re-reads `main_kcoro` through a `noinline` helper
  (`kc_tls_main_fresh`) whose fresh frame recomputes the TLS address on the thread
  actually executing it, and no longer writes `current_kcoro`.
- Rule for upstream: after a `kcoro_switch` returns, the same frame must not touch a
  `__thread` variable directly; go through a non-inlined call.

**Verification.** The C reproducer (8 workers + feeder, 200 passes × 32 rendezvous
handoffs, pthread-condvar pass boundary — mirrors the liquid-audio engine wiring) went
from hanging in pass 0 to 30/30 clean runs. The Rust engine smoke test
(`cargo test --lib engine_gemv`) passes with bit-exact GEMV parity; full crate suite
161/161 green.

## Patch 0003 — context switch drops AAPCS64 FP state (d8-d15, FPCR)

**Risk, not yet a reproduced failure.** `arch/aarch64/kc_ctx_switch.S` saved only
x19-x28/x29/x30/sp, with a header note that "libkcoro avoids FP/SIMD on ARM64". That
invariant cannot hold once coroutine bodies run arbitrary C/Rust: AAPCS64 makes d8-d15
callee-saved (and FPCR callee-preserved), so a compiler may keep live FP values in
d8-d15 across ANY call — including one that parks the coroutine (`kc_chan_recv`).
Another coroutine scheduled on the same thread then clobbers them, and the resumed
coroutine continues with corrupted FP state — silent numerics corruption, the worst
failure class for this engine.

**Fix.** The switch now saves/restores d8-d15 at `reg[16..23]` and FPCR at `reg[24]`
(`reg[32]` had free slots). A fresh coroutine's calloc'd frame restores zeros: d8-d15=0
is harmless (callee-saved, initialized before use) and FPCR=0 is the AArch64 default FP
state (RNE, no traps, no FTZ) — exactly the regime the kernels assume.

Out of scope, flagged for upstream: the x86-64 switch does not save MXCSR/x87 control
words (SysV makes their control bits callee-preserved; all xmm registers are
caller-saved, so data registers are fine there).

**Verification.** Rendezvous stress 5×200 passes clean; crate suite 161/161; engine
GEMV bit-parity unchanged.

## Patch 0004 — unpark queues to the coroutine's owning scheduler

**Symptom.** The resident native stage machine needs plain Rust/OS threads to ring a coroutine
doorbell: write the request slot, then `kcoro_unpark(coord)`. The old unpark path could only
enqueue on the caller's current scheduler. From an external thread that scheduler is `NULL`; with
multiple dispatchers it can also be the wrong scheduler.

**Root cause.** A coroutine already records its owning scheduler on spawn/resume, but unpark did
not prefer that owner. That made external-thread doorbells depend on ambient scheduler context
rather than the target coroutine's actual queue.

**Fix.** `kcoro_unpark` now enqueues to `co->scheduler` first, falling back to
`kc_sched_current()` only for manually-driven coroutines without an owner. No default scheduler is
created implicitly. This is what makes the native engine's request doorbell and the last-worker
stage-done doorbell legal from non-coroutine contexts.

**Verification.** The native engine's `native_engine_mlp_bit_parity` test runs through the
external-thread request doorbell and the worker-to-coordinator stage doorbell; the bit-parity test
passes against the threadgroup port.

## Patch 0005 — precise parking: wake tokens + untimed idle waits

**Symptom.** Bit-identical decode builds swung 24k → 244k audio underrun samples
run-to-run (the "wake lottery"). Every doorbell (`kcoro_unpark` →
`kc_sched_enqueue_ready`) rode a lossy handoff into the worker idle path, and each
lost wake stalled ready work for up to 5 ms.

**Root cause.** Two windows in `worker_main`'s idle path, both papered over by a 5 ms
`pthread_cond_timedwait` recovery poll:
1. A producer that enqueued BEFORE the worker's `idle_workers++` read 0 idle and
   skipped the signal entirely; the worker then slept on live work.
2. A signal issued between `idle_workers++` and `pthread_cond_wait` found no waiter
   and evaporated (condvars carry no memory).
A third, structural hazard: `kc_spawn`'s `last_task` fast slot is owner-only, but
`pthread_cond_signal` wakes an arbitrary waiter — the wrong worker wakes, finds
nothing, re-parks, and the owner's slot work waits for the poll.

**Fix.** Exact signal accounting, three parts:
- `park_tokens` (guarded by `park_mu`): producers that see an idle worker mint a
  token with their signal; a worker consumes one just before waiting (covering
  window 2) or on wake. Tokens persist until consumed — a wake cannot be lost.
- Dekker re-check: after `idle_workers++`, the worker re-checks every source it can
  acquire (own slot, ready queue, inject queue, all deques) before touching park_mu
  — the producer either observes the idle declaration or the worker observes the
  enqueue (window 1).
- `sched_wake_slot` (broadcast) for `last_task` pushes, plus an owner's own-slot
  check under `park_mu` right before waiting — the owner is guaranteed to wake and
  cannot sleep on a slot filled after its re-check.
With those in place the idle wait is UNTIMED: the 5 ms poll is gone, wakes are
µs-bounded, and an idle scheduler consumes zero CPU.

Drive-by: the steal scan divided by `workers - 1` — with a single-worker scheduler
that is a modulo by zero (UB). The scan is now skipped when `workers == 1`.

**Verification.** Full crate suite green (168); engine bit-parity tests green through
the doorbell path; audible dual-path e2e green with CPU underruns/latency holding the
lane-uniform band across repeated runs (see DECODE_ENGINE.md state of play).
