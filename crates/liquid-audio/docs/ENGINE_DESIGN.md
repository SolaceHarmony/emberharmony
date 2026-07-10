# flashkern engine — the design (CPU as GPU)

This is the chassis design the piece-by-piece kernel work mounts onto. It is the concrete
layer below DECODE_ENGINE.md's contract: the memory map, the kernel ABI, the tile library,
and the adherence rules. Status: living target plus as-built notes. A phase is only live when
its code path and parity oracle have landed.

The one-sentence spec: **one buffer, computed on through pointers and shared memory, by one
C++ kernel program run by a persistent lane team, handing back to Rust once per pass.**

## 0. What "kernel" means here (the adherence rule that was broken)

A kernel is ONE native program executed by the resident lane team, owning all control flow
between published stages — layer loop included. Rust builds the context, rings the doorbell,
and reads rings; it does not run between stages. AS-BUILT (2026-07-09, supersedes the line
below): the WHOLE backbone token is one resident lane program (REQ_TOKEN_PASS: embed →
every conv/attention layer → final norm), and DepthDecode rides the same team as a
lane-uniform Rust program via the generic REQ_CALL + exported lane fence. The stage board
described elsewhere in this file was replaced by generation fences (lane-uniform kernel);
rayon executes nothing per-token. Current diagrams + numbers: DECODE_ENGINE.md §0.
Historical (pre-arc): the backbone FFN MLP was the first resident mount, with ShortConv and
DepthDecode orchestrated from Rust rayon lane closures.

## 1. The arena — one region, fixed capacities, stable pointers

Weights are NOT copied into the arena: the safetensors mmap IS the weight buffer (design
rule: reads are the floor, movement is theft). The engine owns a `WeightTable` parsed once
from the safetensors header: `name → { offset, rows, cols, dtype }` over the mmap base.
candle no longer stands between the engine and the bytes.

Everything mutable lives in ONE 64-byte-aligned allocation with fixed capacities — growth
is a config decision at engine build, never a runtime realloc (pointer stability is what
makes "computation through pointers" legal). Layout for LFM2.5-Audio-1.5B, B=1,
`max_ctx = 4096`, offsets rounded to 64:

| region | shape | bytes | notes |
|---|---|---|---|
| doorbell + pass ctl | epoch u64, reason u32, pass_seq u64, lane park words | 256 | the ONLY cross-thread control words |
| kv region | attn_layers × 2 × [8][4096][64] bf16 + len cursor/layer | ~8 MB × attn_layers | fixed cap: no growth realloc, ever; cursor rollback stays O(1) |
| conv states | conv_layers × [2048][K−1] bf16 | 8 KB × conv_layers | kernel shifts in place; snapshot copies OUT (tiny) |
| depth kv planes | 6 × 2 × [8][8][32] bf16 | 24 KB | cursor reset per frame |
| activation planes | x, xn, h, qkv[6144], gu[2·8192], t[8192], attn[2048], y[2048] — ×2 (double-buffered for stage pipelining) | ~350 KB | all bf16 bits except in-register f32 |
| logits plane | [65536] f32 → bf16 bits | 384 KB | largest head wins |
| rope tables | backbone [4096][32] f32; depth [4096][16] f32 | 1.3 MB | copied ONCE at build for locality (ends the 6× per-Mha duplication) |
| token ring | 1024 × u32 + rd/wr seq | 4 KB | descriptors, not Vecs |
| pcm ring | 10 s × 24 kHz f32 + rd/wr seq | 960 KB | CPAL callback reads; reserve/commit API |
| sampler state | rng words + top-k scratch | 4 KB | v2 (see §3) |

Total mutable arena ≈ 60–90 MB, dominated by fixed-cap KV. Every kernel argument is
`arena_base + offset` or `mmap_base + offset`. Nothing else crosses the ABI.

## 2. The kernel program — CURRENT: resident stage machine

**Correction (2026-07-09, after the Fable 5/kcoro-hop audit):** the Rust `TileEngine`
channel chassis proved the parked descriptor model and exposed real kcoro bugs, but it is
not the final hot-loop shape. A rendezvous channel hop still costs roughly 15-20 us
(waiter allocation, mutex, ready queue, context switch). The live direction is the
resident native stage machine in `native/src/engine/flashkern_engine.cpp`: no channels, no descriptor
staging, no malloc inside the published stages (and none once scratch is warm for the fixed
shape), and no Rust between stages.

- **Persistent native team**: `kc_dispatcher_new(P_cores + coordinator)` starts one
  coordinator coroutine and a parked worker team. Workers are sized from the same P-core
  policy as the rest of the CPU runtime, not logical-core `available_parallelism`.
- **One request doorbell**: the Rust rim writes one request slot, unparks the coordinator,
  then waits on a pthread condvar for the pass boundary. Stop/shutdown is a request observed
  at the pass boundary, never per op.
- **Stage board, not channels**: the coordinator publishes `{kind, count, chunk}`, resets
  `next`, sets `remaining = workers`, then bumps an epoch. Workers wake, race
  `next.fetch_add()` dry, and the last worker unparks the coordinator. That is the whole
  stage-completion doorbell.
- **Descriptors stay at the boundary**: rings still carry `(offset, len, epoch)` between
  subsystems. Inside the engine hot loop, work is represented by shared stage state and raw
  pointers into the mmap/arena, not per-tile messages.
- **Determinism remains explicit**: reductions that affect bits fold in fixed order. Tile
  over-decomposition is allowed only where rows are independent or the oracle pins the
  exact reduction order.
- **As-built/live mount**: the backbone FFN MLP routes through
  `native_engine::process_engine()` when the native engine is built, with a bit-identical
  threadgroup fallback. Attention, ShortConv, and DepthDecode are still outside the full
  token-pass program.

Linkage: kcoro is vendored in the sibling `kcoro-sys` crate and built by that crate's
build script with the upstream Makefile's flags — no machine-local path. The vendored copy
carries local runtime fixes documented in `crates/kcoro-sys/vendor/kcoro/PATCHES.md` and
source comments: 0001 a three-state park gate for lost wakeups, 0002 fiber-safe TLS after
M:N migration, 0003 AAPCS64 FP-state save, and 0004 enqueue-to-owning-scheduler so an
external-thread `kcoro_unpark` is a legal doorbell.
On supported `aarch64`/`x86_64` GCC/Clang targets, kcoro, the architecture kernel,
and the native engine are built unconditionally. Unsupported targets fail the build;
there is no `has_*` cfg or degraded engine branch.

The older Rust `src/compute/flashkern/engine.rs` `TileEngine` remains as a prototype/reference rung:
it verifies channel-dispatch parity and documents kcoro channel rules. Do not mount new
production passes there.

## 2-old. (superseded) The kernel program

```c
// THE kernel. Uniform control flow; every lane runs this same program.
void lfm_token_pass(const EngineCtx* ctx, uint32_t lane) {
    for (l = 0; l < ctx->n_layers; l++) {
        if (ctx->layer_kind[l] == ATTN) attn_block(ctx, l, lane);   // norm→qkv→rope→append→attend→out+res
        else                            conv_block(ctx, l, lane);   // norm→in_proj→conv update→out+res
        mlp_block(ctx, l, lane);                                    // norm→gate/up→swiglu→down+res
    }
    final_norm(ctx, lane);
    logits_head(ctx, lane);            // rb'd bf16 logits → logits plane
}   // barriers INSIDE; Rust re-entered only after return
```

- **Team**: P-core-count pthreads created at engine init, pinned, parked on a
  spin-then-futex hybrid. `pass_seq` bump wakes the team; the team runs one pass; lane 0
  publishes; all repark. One Rust handback per pass. Doorbell checked at the boundary only.
- **Stage fences**: the existing SpinBarrier, now a C++ generation barrier in the arena.
- **Audio frame pass**: `lfm_frame_pass` = the DepthDecode program (8 codebook steps × 6
  blocks) as the same shape — already proven; it moves from Rust closures into the program.
- **v1 sampling compromise (parity-driven)**: the pass ends at the logits plane; Rust
  samples at the boundary (µs, once per pass) because the sampler must reproduce candle's
  LogitsProcessor RNG stream bit-for-bit for the parity oracles. v2 ports the RNG into the
  kernel and sampling moves inside — the frame pass needs this to be fully Rust-free.

  **RNG decisions (deep-research verified, 2026-07-08; 105-agent adversarial pass):**
  * *Deterministic stream (v2 port)*: ChaCha12 (6 double-rounds) + rand_core's PCG32
    seed-expansion (MUL 6364136223846793005, INC 11634580027462260723, advance-then-output,
    LE bytes) + one u32 per f32 uniform draw ([1,2) mantissa trick, 9 bits discarded) +
    WeightedIndex partition_point over sequential f32 cumulative sums — all read directly
    from the pinned crates. Fine details are locked by GOLDEN VECTORS generated from the
    Rust crate (10k draws per seed), not by documentation: the research pass confirmed web
    sources are unreliable at this level of detail.
  * *Seed minting (production)*: `getentropy(2)` — FEAT_RNG/RNDR does NOT exist on any
    Apple core (M1 confirmed by privileged ID-register dump; XNU contains zero FEAT_RNG
    plumbing; the missing sysctl key on this M2 means "undefined", and executing RNDR would
    SIGILL). Apple DTS explicitly endorses getentropy for exactly this seeding use case.
    Never probe RNDR on Apple Silicon.
  * *Future per-lane streams*: Philox4x32-10 — stateless (counter,key)→output, ≥2^64
    independent streams, BigCrush-clean with a 3-round margin, published KAT vectors for
    bit-parity of any NEON port, C++26 `std::philox_engine`. Adopt only when a kernel
    genuinely needs lane-addressable streams (batch sampling/dropout); the decode sampler
    stays a single sequential ChaCha12 stream for candle parity. Threefry/xoshiro noted as
    faster alternatives where counter semantics aren't required — measurement decides, as
    always.

## 3. The tile library — simdgroup_matrix on NEON (not yet built; this specifies it)

```cpp
// fk_sg8x8: Metal simdgroup_float8x8. 16 f32 accum registers in BFMMLA 2×2 layout.
struct fk_sg8x8 { float32x4_t t[4][4]; };
void fk_sg_fill(fk_sg8x8&, float);
void fk_sg_load_a(fk_bf16_panel&, const uint16_t* a, int lda);   // 8×8 bf16 → MMLA order
void fk_sg_load_b(fk_bf16_panel&, const uint16_t* b, int ldb);
void fk_sg_mma(fk_sg8x8& acc, const fk_bf16_panel& a, const fk_bf16_panel& b); // 16× BFMMLA
void fk_sg_store(const fk_sg8x8&, float* c, int ldc);            // masked ragged edge
void fk_sg_store_rb(const fk_sg8x8&, uint16_t* c, int ldc);      // fused RNE epilogue
```

### 3b. Tile backends by target (portability matrix)

| target | decode tiles (bandwidth-bound) | prefill tiles (compute-bound) | detection |
|---|---|---|---|
| Apple M1–M3 (macOS) | NEON BFMMLA / widening FMA | Accelerate sgemm → AMX (measured: §E4) | cfg + sysctl FEAT_* |
| Apple M4+ (macOS) | same | Accelerate → SME (same call, architectural unit) | FEAT_SME sysctl |
| Graviton 3/4, Neoverse V1/V2, Cortex-A78+ (Linux) | NEON BFMMLA / widening FMA | fk_sg8x8 BFMMLA; option: OpenBLAS `sbgemm` (bf16-in/f32-out — no widening tax, the non-Apple Accelerate analog) — adopt only by on-target measurement | HWCAP2_BF16 / HWCAP2_I8MM (built) |
| SME/SME2 cores (Cortex-X4+, Dimensity 9300+) | NEON | FMOPA outer-product tiles as a first-class fk_sg backend (architectural, compiler-supported — unlike AMX) | HWCAP2_SME |
| pre-bf16 ARMv8 (Pi 5 / Cortex-A76) | f32 FMLA broadcast microtile (4×4 baseline) | same | absence of BF16 |
| x86-64 | AVX2 / VDPBF16PS (built) | same + AVX-512 tiles | CPUID (built) |

**Honest constraint (pre-bf16 row):** this model is bf16; on cores without FEAT_BF16 the
loader already falls back to f32 — 2× weight bytes (~6 GB) against Pi-class ~17 GB/s
bandwidth ⇒ ~350+ ms/token floor. Pre-bf16 boards are functional targets, not real-time
ones, for this model. Real-time non-Apple targets are the bf16 rows (Graviton 3+,
Neoverse, recent Cortex-A/X).

x86 twin over VDPBF16PS/AVX2. Consumers: the GEMM (refactored to compose from it — the
existing 4×4 BFMMLA loop becomes `fk_sg_mma` calls), prefill M>4 tiles, prefill attention
(q·Kᵀ tiles), the monarch/fft fanout ports when they move from Rust to the program. One
tile type, every matrix kernel composes from it — that is what "simdgroup_matrix
equivalent" means, and it is the unit the rb-epilogue lands in.

## 4. Adherence rules (hard constraints, reviewable per diff)

1. No heap allocation inside a pass. Planes are arena offsets; violation = review reject.
2. No candle type crosses the engine ABI. Ptr/len/offset only. candle remains: loader
   (until the WeightTable lands), Metal path, training, and the parity reference chain.
3. Pointer stability: fixed capacities; changing a capacity is an engine rebuild.
4. Weight movement is theft: any transpose/pack/copy of a weight in a hot path must cite
   this document's exception list (currently empty) in a comment, or it does not merge.
5. Every phase lands with its oracle: byte tier (wav-hash flag-off) or ulp tier (flagged +
   bound test) — stated in the PR, no silent tier changes (the fused_conv_decode A/B
   regression is the cautionary case).

## 5. Migration phases (each = one reviewable piece with an oracle)

- **E1a native stage-machine mount**: ✅ **As-built for the FFN MLP only.** The model hot
  path calls `native_engine::process_engine()` and falls back to the threadgroup port on
  native-engine failure. Oracle: native MLP bit-parity vs the threadgroup port.
- **E1b full token-pass chassis**: arena + WeightTable + persistent team +
  `lfm_token_pass` for the backbone decode step (existing block math demoted to device
  functions). Oracle: flag-off wav-hash byte parity; flag-on A/B vs current blocks.
- **E2 frame pass**: DepthDecode into the program (v1 boundary sampling per codebook step
  batch; v2 in-pass RNG). Oracle: token-sequence A/B vs current DepthDecode.
- **E3 rings**: token/PCM rings live; per-token tensor construction deleted from the loop;
  sampler v2. Oracle: e2e gates + allocation counter == 0 in-pass.
- **E4 prefill pass**: chunked `lfm_prefill_pass`, streams during capture (the
  doorbell-legal chunk boundary), kills the M>4 transpose exception and the conformer's
  candle chain. **Tile backend: DECIDED BY MEASUREMENT (2026-07-08, on the target M2)** —
  Accelerate `cblas_sgemm` (the sanctioned dispatcher to the AMX matrix units; SME on
  M4+ via the same call) at ~1.0–1.5 TFLOP/s f32 vs our BFMMLA GEMM's ~55–61 GF/s at
  prefill shapes: 19–28× including the bf16→f32 widening tax. Widening is tile/layer-
  transient per turn (never a resident f32 weight copy — cites the movement rule's
  exception list: this is the one entry, bounded and per-turn). Raw AMX via encodings
  (corsix) is DEMOTED to "only if a measured gap vs Accelerate ever justifies
  unsupported ISA" — currently it does not. fk_sg8x8/BFMMLA remains the decode-side and
  non-macOS tile backend. Oracle: prefill-output parity vs candle at f32 tier
  (measured rel ≈ 1e-5), behind an object-graph backend selector, reference = current
  path. Decode stays NEON: sgemm at M=1 would mean widening the full weight per token —
  theft; bandwidth floor unchanged.
- **E5 codec**: Mimi decode path. Oracle: wav parity per frame.

## 6. What stays candle, permanently

Loading/verification tooling, the Metal device path, training (autograd needs the graph),
and the composed reference chain that every oracle compares against. The engine replaces
candle as the RUNTIME for the CPU voice path — not as the model definition.
