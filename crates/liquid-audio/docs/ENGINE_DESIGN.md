# flashkern engine — the design (CPU as GPU)

This is the chassis design the piece-by-piece kernel work mounts onto. It is the concrete
layer below DECODE_ENGINE.md's contract: the memory map, the kernel ABI, the tile library,
and the adherence rules. Status: living target plus as-built notes. A phase is only live when
its code path and parity oracle have landed.

The one-sentence target: **one resident weight image plus fixed mutable arenas,
computed through pointers and shared memory by one C++ kernel program on a
persistent lane team, handing compact completion facts to Rust once per pass.**

That sentence describes the CPU backend. Flashkern never owns Metal dispatch.
Matrix coprocessing is a required peer path: CPU matrix opcodes remain first-class,
and Apple GPU execution moves to a separate MLX C++/Metal engine selected by the
model/device layer. The current Candle Metal path is temporary migration code.

## 0. What "kernel" means here (the adherence rule that was broken)

A kernel is ONE native program executed by the resident lane team, owning all control flow
between published stages — layer loop included. Rust builds the context, rings the doorbell,
and reads rings; it does not run between stages. AS-BUILT (2026-07-14): the WHOLE
backbone token is one resident lane program (REQ_TOKEN_PASS: embed →
every conv/attention layer → final norm → optional sample), and the complete
Depthformer frame is a typed C++ `REQ_DEPTH_FRAME` program: projection, every
codebook/layer, KV recurrence, logits, sampling, and sampled-embedding feedback.
Rust installs pointer descriptors and rings one pass; it runs no numerical frame.
The stage board described elsewhere in this file was replaced by generation fences
(lane-uniform kernel); rayon executes nothing per-token. Current diagrams + numbers:
DECODE_ENGINE.md §0.

## 1. Target arena — fixed capacities and stable pointers

The weight side is partly built. `native/src/io/safetensors.cpp` reads all selected
shards once into one 64-byte-aligned immutable image and builds a tensor table over
that image. It does not require mmap. Remaining Candle modules obtain explicit,
counted payload copies through `ResidentWeights::candle_builder`; current native
contexts therefore still capture some pointers into Candle compatibility storage.

At cutover, weights are not copied into a mutable arena. Native model plans bind
`name → { offset, shape, dtype }` directly over the resident image, and Candle no
longer stands between the engine and those bytes.

The following mutable arena is target design, not an as-built inventory. Everything
mutable ultimately lives in fixed-capacity 64-byte-aligned storage; growth is a
runtime-construction decision, never a warmed-pass realloc. Layout for
LFM2.5-Audio-1.5B, B=1,
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
| pcm ring | 10 s × 24 kHz f32 + rd/wr seq | 960 KB | native platform playback callback reads; reserve/commit API |
| sampler state | one native ChaCha20 stream image per conversation/generation | 192 B each | one draw order crosses text and every audio codebook; the current Rust rim owns the opaque image until native conversation ownership lands |
| sampler scratch | [largest vocab] f32 weights + [largest vocab] f32 top-k heap + lane partials | ~512 KB at vocab 65,536 | engine-owned and reserved when heads are installed; no logit payload copy or warmed-pass allocation |

Target mutable arena size is approximately 60–90 MB, dominated by fixed-cap KV.
Every target kernel argument is `arena_base + offset` or `image_base + offset`.

## 2. The kernel program — CURRENT: resident stage machine

The discarded Rust `TileEngine` prototype proved the descriptor model and exposed the
cost of a channel operation per tile. It has been deleted; git history is the archive.
The live engine is `native/src/engine/flashkern_engine.cpp`: no numerical channels, no
descriptor staging, no allocation inside a warmed pass, and no Rust between native stages.

- **Persistent native team**: one stable pthread per logical lane, sized from the same
  P-core policy as the rest of the CPU runtime. Numerical call stacks never migrate.
- **Mounted command doorbell**: the C++ rim writes one borrowed request slot,
  creates one generation-protected descriptor with `Engine*` as payload, and
  invokes the registered Rust submitter. One safe Rust kcoro broker publishes the
  128-byte SQ cell; dedicated Rust CQ ingress resolves the preallocated result
  slot after lane 0 publishes the matching completion.
- **Stage board, not channels**: every lane enters the same `run_stage`; the opening
  fence's last arriver publishes `{kind, count, chunk}` and resets `next`. Workers race
  `next.fetch_add()` dry, and the next generation fence proves all claimed tiles landed.
- **Descriptors stay at the boundary**: the mounted ring carries fixed IDs and
  scalars, while numerical pointers stay in the single engine request slot. The
  target promotes that slot to an owned region-retaining descriptor. Inside the
  engine hot loop, work is shared stage state and raw pointers, not per-tile
  messages.
- **Determinism remains explicit**: reductions that affect bits fold in fixed order. Tile
  over-decomposition is allowed only where rows are independent or the oracle pins the
  exact reduction order.
- **As-built/live mount**: `REQ_TOKEN_PASS` executes embed, every native ShortConv or
  attention block, each MLP, final norm, and optional logits over one team entry.
  `REQ_DEPTH_FRAME` executes the complete Depthformer frame over the same fixed team.
  Both backbone and Depthformer plans coexist by stable identity; a ticket selects
  one immutable plan while the executor and scratch arena remain shared.
- **As-built CPU streaming convolution**: `REQ_DEPTHWISE_STREAM` partitions full
  `(batch, channel)` rows across the same fixed team. The C ABI borrows split
  state/input/weight planes and separate output/state destinations, then publishes
  one completion after the program-final fence. No Metal dispatch exists in this
  request or anywhere else in Flashkern.
- **As-built native sampler (2026-07-14)**: `run_sampler` at
  `native/src/engine/flashkern_engine.cpp:831` is a fixed-lane collective over
  pointer-borrowed F32/BF16 logits. Greedy selection uses one fence and no RNG;
  stochastic selection uses three fences, engine-owned probability/top-k
  scratch, and exactly one mutation of the shared ChaCha stream. The text head
  calls it inside `REQ_TOKEN_PASS`; `run_depth_frame` calls it directly for each
  codebook inside one `REQ_DEPTH_FRAME`, so neither path creates a per-draw or
  per-codebook ticket. `REQ_SAMPLE` remains the standalone prefill/fallback and
  conformance entry. `REQ_PRNG` independently pins stream/assembly behavior.
  AArch64 and x86_64 assembly expand ChaCha20 blocks, and Apple
  `SecRandomCopyBytes` supplies key/nonce material only at seed time.
  Per-block request entries remain as parity fixtures; there is no alternate engine.

Linkage has two distinct kcoro roles. `crates/kcoro` is the safe Rust product
coordinator and owns the current broker future. The sibling `kcoro-sys` crate
builds the vendored C conformance runtime and POSIX expected-value adapter; its C
ticket scheduler is not on the production pass path. Flashkern uses the wait
adapter but keeps its fixed numerical workers outside either coordination ready
queue. On supported `aarch64`/`x86_64` GCC/Clang targets, the coordinator, wait
substrate, architecture kernel, and native engine build unconditionally.
Unsupported targets fail; there is no degraded engine branch.

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

**Honest constraint (pre-bf16 row):** this model is bf16. The current CPU loader
rejects a machine without the required bf16 kernel; it does not silently widen the
checkpoint. A future explicit f32 portability backend would read about 2x weight
bytes (~6 GB) against Pi-class ~17 GB/s bandwidth, implying a ~350+ ms/token
floor. Pre-bf16 boards are therefore functional targets, not real-time ones, for
this model. Real-time non-Apple targets are the bf16 rows (Graviton 3+,
Neoverse, recent Cortex-A/X).

x86 twin over VDPBF16PS/AVX2. Consumers: the GEMM (refactored to compose from it — the
existing 4×4 BFMMLA loop becomes `fk_sg_mma` calls), prefill M>4 tiles, prefill attention
(q·Kᵀ tiles), the monarch/fft fanout ports when they move from Rust to the program. One
tile type, every matrix kernel composes from it — that is what "simdgroup_matrix
equivalent" means, and it is the unit the rb-epilogue lands in.

## 4. Adherence rules (hard constraints, reviewable per diff)

1. No heap allocation inside a pass. Planes are arena offsets; violation = review reject.
2. No Candle type crosses the engine ABI. Ptr/len/offset only. Candle remains a
   migration/reference owner today; no production inference owner remains at
   target cutover.
3. Pointer stability: fixed capacities; changing a capacity is an engine rebuild.
4. Weight movement is theft: any transpose/pack/copy of a weight in a hot path must cite
   this document's exception list (currently empty) in a comment, or it does not merge.
5. Every phase lands with its oracle: byte tier (wav-hash flag-off) or ulp tier (flagged +
   bound test) — stated in the PR, no silent tier changes (the fused_conv_decode A/B
   regression is the cautionary case).

## 5. Migration phases (each = one reviewable piece with an oracle)

- **E1a native stage-machine mount**: ✅ **As-built.** The mandatory process
  engine owns the fixed team and the MLP/layer parity entries. Construction
  failure is fatal rather than a second production scheduler. Oracle: native MLP
  bit parity vs the test-only composed reference.
- **E1b full token-pass chassis**: ✅ **As-built.** `REQ_TOKEN_PASS` executes the
  backbone decode step on the persistent team. Weight pointers and mutable state
  still arrive through the borrowed compatibility request slot; direct resident
  image binding and the complete target arena remain open.
- **E2 frame pass**: ✅ **As-built typed frame.** `REQ_DEPTH_FRAME` owns the
  projection, all Depthformer layers/codebooks, native zero-spin fences, logits,
  sampler, and sampled-embedding recurrence. The Rust numerical callback,
  `SpinBarrier`, logits Tensor rebuild, nested sampler ABI, and BF16 hidden copy
  are deleted. The remaining outer migration is to bind the input/output slots and
  shared RNG image directly to native conversation state instead of returning a
  small Rust token `Vec`. Oracle: seeded token sequence plus the one-ticket typed
  plan lifecycle test.
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
- **E5 codec**: native serial Mimi decode is built and production-swapped; mounting
  it as a typed pass on the fixed lane team remains open. Oracle: chain parity and
  wav parity per frame.

## 6. Candle disposition

Candle currently supplies compatibility tensor owners, unfinished numerical paths,
temporary Metal execution, training tools, and parity references. That is migration
state, not a permanent production boundary. CPU inference binds the native resident
image and executes Flashkern C++/SIMD/assembly without Candle or Rust numerical
callbacks. Apple GPU inference remains mandatory, but its replacement is a separate
MLX C++/Metal device engine with its own command/memory boundary. It is not compiled
into Flashkern. References and training tooling may remain outside shipped inference.
