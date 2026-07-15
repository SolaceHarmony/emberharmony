# G3 Shared-Doorbell Evidence

Status: implementation and gates committed.

## Identity

- Flashkern implementation: `d2c43abdd6cc64e7f3452c145f130fcc545a8196`
- Percentile harness: `3625df4e5616c1af6af853115c7badceaa338e9e`
- Ember kcoro vendor commit: `8d510f83`
- Upstream kcoro snapshot: `bd530f4c9196d948472067c5bc379e7117c645b2`
- G0 comparison: `321538f1`
- Machine: Apple M2 Max, 12 logical CPUs, 32 GiB RAM
- OS: macOS 26.6 (`25G5028f`)
- Shape: BF16 fused MLP, `H=1024`, `I=4096`, 8 fixed lanes

## Implemented Contract

Flashkern uses two cache-line-isolated expected-value words:

```text
submission
  -> publish request slot and pass generation
  -> increment shared dispatch word
  -> one wake-all call for the fixed lane team

stage boundary
  -> non-last lanes declare bits in park_mask
  -> recheck generation and wait on shared fence word
  -> last lane runs serial section and publishes generation
  -> if park_mask != 0, increment fence word
  -> one wake-all call for threads actually waiting on that address
```

The pass and fence paths allocate nothing, search no registry, and perform no
bounded spin. Changed-before-wait and wake-before-wait are closed by the
expected-value word. `park_mask` suppresses empty wakes and records the exact
logical waiter set; the host address-wait primitive fans that set out with one
syscall instead of one syscall per lane.

The raw C ABI also owns the single-slot claim. Concurrent callers receive
`-EBUSY` before either can write request, context, or scratch state. One ticket
represents one full pass, and only the ticket callback releases the blocking
rim for the matching pass epoch.

## Wake And Lifecycle Gate

The production-backed 10,000-pass soak executes two fence generations per pass
through the real `REQ_CALL` path. A representative committed run reported:

```text
10,000 submissions
10,000 numerical completions
10,000 ticket callbacks
10,000 shared dispatch wake calls
20,000 fence generations
20,000 fence wake calls
139,967 logical fence waiters
0 passes still claimed at teardown
```

The former per-lane wake shape required up to 140,000 fence wake calls for the
same waiter population. Shared address fan-out reduces that upper bound to
20,000 without waking unrelated coordination workers.

Other executed gates:

- raw concurrent request rejection before payload mutation: pass;
- `cargo test -p kcoro-sys -- --nocapture`: pass;
- `cargo test -p liquid-audio --lib -- --nocapture`: 170 pass, 3 ignored model gates;
- hermetic `liquid-audio` integration tests: pass;
- cold and post-pass idle CPU: `0.005-0.006%` with 8 parked lanes;
- bit parity across 1, 3, and 8 lanes: pass.

## Raw Latency

The committed harness warms each path for 20 passes, alternates path order, and
retains 1,000 individual pass durations. Five runs reported native
`p50 / p95 / p99` milliseconds:

```text
0.439 / 0.515 / 0.538
0.435 / 0.521 / 0.574
0.439 / 0.527 / 0.616
0.437 / 0.530 / 0.611
0.439 / 0.524 / 0.561
```

The median across those run-level percentiles is:

```text
p50 0.439 ms   p95 0.524 ms   p99 0.574 ms
```

Against the raw G0 median (`0.330 / 0.576 / 0.732 ms`):

| Percentile | G0 FENCE_SPIN | G3 shared doorbell | Change |
|---|---:|---:|---:|
| p50 | 0.330 ms | 0.439 ms | +33.0% |
| p95 | 0.576 ms | 0.524 ms | -9.0% |
| p99 | 0.732 ms | 0.574 ms | -21.6% |

G0 wins the median because short stages often finish inside its 8,192-relax
spin window. G3 deliberately pays the address-wait transition instead. G3 has
the better tail: it stops burning scheduler time while peers finish and removes
the old timing cliff when a lane crossed from spin into stackful kcoro park.

## Disposition

The original `0.406/0.280 ms` one-shot comparison is superseded by raw
percentiles, but the `+33.0%` p50 keeps the product-default G3 latency gate open.
The p95/p99 improvement is real positive evidence, not permission to ignore the
median and not permission to restore spinning. Next performance work belongs at
the plan level: remove redundant stage boundaries where dependency analysis
permits, then benchmark a real full token/Moshi frame and checkpoint rather than
extrapolating from one MLP block.
