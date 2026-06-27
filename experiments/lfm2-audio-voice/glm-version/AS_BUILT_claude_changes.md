# Claude's threading, bf16, mask memoization, and `to_vec4` changes — as-built

> This documents the changes Claude made to `liquid-audio-rs` across two sessions,
> **as built on disk** (not as proposed). It covers what was changed, how it works,
> what's verified, and what remains. All changes are currently **uncommitted** except
> the `d87a52e` commit (deps + `lib.rs`/`loader.rs` wiring).

## State at a glance

| File | Status | What changed |
|---|---|---|
| `src/threads.rs` | **untracked** (new) | torch-parity intra-op thread pool |
| `src/bf16_gemm.rs` | **untracked** (new) | NEON BFMMLA bf16 GEMM: FFI + `CustomOp2` + runtime gate |
| `csrc/bf16_gemm.c` | **untracked** (new) | the C BFMMLA micro-kernel |
| `build.rs` | **untracked** (new) | compiles the C kernel on aarch64 via `cc` |
| `THREADING_PARITY.md` | **untracked** (new) | Claude's writeup (partially outdated — see below) |
| `.github/workflows/rust-voice.yml` | **untracked** (new) | Rust CI: build + test on Linux + macOS arm64 |
| `src/lib.rs` | **committed** (`d87a52e`) | `pub mod threads; pub mod bf16_gemm;` + re-exports |
| `src/loader.rs` | **committed** (`d87a52e`) | calls `configure_intraop_threads()` at top of `from_pretrained` |
| `Cargo.toml` | **committed** (`d87a52e`) | `rayon`/`num_cpus`/`libc`/`half` deps, `cc` build-dep, `accelerate` feature |
| `Cargo.lock` | **committed** (`d87a52e`) | resolved |
| `src/audio_out.rs` | **unstaged** (modified) | `AudioDetokenizer: Send` bound |
| `src/model/mlp.rs` | **unstaged** (modified) | `Sequential` → `Vec<Box<dyn Module + Send>>` + 3 tests |
| `src/model/lfm2_hf.rs` | **unstaged** (modified) | vendored `build_causal_mask`/`repeat_kv` from candle-transformers; mask memoization (`Cache::mask`); reverted the `KvCache` swap (faithful `Tensor::cat` restored) |
| `src/candle_ext/transformers_utils.rs` | **untracked** (new) | the vendored `build_causal_mask` + `repeat_kv` (candle 0.10→0.9.2 backport) |
| `src/candle_ext/tensor_ext.rs` | **untracked** (new) | `TensorExt::to_vec4` (candle's `to_vecN` ladder stops at 3; this extends it by one rank) |
| `src/candle_ext/mod.rs` | **unstaged** (modified) | `pub mod tensor_ext; pub mod transformers_utils;` |

**Build state:** `cargo test --lib` → **57 passed; 0 failed**. `cargo build --all-targets` → clean. The MLP `Send` fix (which unblocked a prior compile error) is in place.

---

## 1. Intra-op thread pool — torch parity (`src/threads.rs`)

### What it does
Replicates torch's `at::intraop_default_num_threads()` (`aten/src/ATen/ParallelCommon.cpp`):
1. Honours `OMP_NUM_THREADS` → `MKL_NUM_THREADS` → `RAYON_NUM_THREADS` (in that order; the first valid `>0` parse wins).
2. Else, on macOS, queries `hw.perflevel0.physicalcpu` via `libc::sysctlbyname` — the **performance cores only**, deliberately excluding the efficiency (E) cores. Falls back to `hw.physicalcpu` (Intel Macs), then `num_cpus::get_physical()`.
3. Installs that count as rayon's **global** pool (`ThreadPoolBuilder::new().num_threads(n).build_global()`) so candle's matmul (`gemm`) + conv/sort inherit it. `build_global` is idempotent — a second call (or a pool already built by a prior candle op) is a harmless no-op (`let _ =`).

### Why
candle/`gemm` default to `num_cpus::get()` (**all** logical cores), which schedules compute-bound matmul onto the slow E-cores — hurting throughput *and* tail latency via work-steal imbalance. On this M2 Max: rayon's default would be 12 (all logical); the fix picks 8 (P-cores), matching torch.

### Wiring
Called once at the top of `from_pretrained` (`loader.rs`), before the first tensor op.

### Tests
- `intraop_threads_is_sane_and_not_all_logical` — asserts `1 ≤ n ≤ num_cpus::get()`; prints `intraop threads = 8 (physical 12, logical 12)` on M2 Max.
- `realtime_pipeline_types_are_send` — asserts `LFM2AudioModel: Send` and `LFM2AudioProcessor: Send` (feasibility probe for the worker-thread pipeline; requires the MLP `Send` fix).

### Verification
`intraop threads = 8 (physical 12, logical 12)` on M2 Max — matches `sysctl hw.perflevel0.physicalcpu`.

---

## 2. NEON BFMMLA bf16 CPU GEMM (`src/bf16_gemm.rs` + `csrc/bf16_gemm.c` + `build.rs`)

### What it does
A hardware bf16 GEMM for candle's CPU path, closing the gap where candle 0.9.2's CPU matmul allowlist is `F16 | F32 | F64` only (bf16 → `UnsupportedDTypeForOp`, so the loader forced f32 on CPU). The Arm BFloat16 extension (FEAT_BF16) provides `BFMMLA`, which does a 2×4·4×2 bf16 matmul with **f32 accumulate** — the same numerics torch's CPU bf16 matmul uses.

### The C kernel (`csrc/bf16_gemm.c`)
- `lfm_bf16_gemm_f32(A, B, C, M, N, K)` — `C(M×N, f32) = A(M×K, bf16) · B(K×N, bf16)`, all row-major.
- Packs A/B into BFMMLA tile order: `vbfmmlaq_f32(acc, av, bv)` treats `av`/`bv` as 2×4 bf16 matrices, computes `a · bᵀ` (2×4·4×2 → 2×2), accumulates into a 2×2 f32 `acc` laid out `[c00,c01,c10,c11]`. Packing B's lane-row `r` = column `(jt+r)` of B over a 4-deep K block makes `(a · bᵀ)[i][j] = Σ_k A[it+i][k]·B[k][jt+j]` — an ordinary `C = A·B`.
- Zero-pads M→Mp (mult of 2), N→Np (mult of 2), K→Kp (mult of 4) via `calloc` (bf16 +0.0 padding contributes nothing to the dot products). Handles odd dims correctly (the test exercises 5×13×7).
- Compiled by `build.rs` via `cc` with `-march=armv8.2-a+bf16`, `opt_level(3)`, gated to aarch64 via `CARGO_CFG_TARGET_ARCH == "aarch64"`. Sets `cargo::rustc-cfg=has_bf16_kernel` so the Rust FFI is only wired where the kernel was built.

### The Rust FFI + op (`src/bf16_gemm.rs`)
- `extern "C" { fn lfm_bf16_gemm_f32(...) }` — declared under `cfg(all(target_arch = "aarch64", has_bf16_kernel))`.
- `has_feat_bf16() -> bool` — **runtime** FEAT_BF16 detection via `libc::sysctlbyname(c"hw.optional.arm.FEAT_BF16")`, cached in a `OnceLock<bool>`. On non-macOS aarch64, returns `false` (Linux `HWCAP2_BF16` via `getauxval` not wired yet). On non-aarch64, returns `false`.
- `bf16_gemm_available() -> bool` — `cfg!(all(target_arch = "aarch64", has_bf16_kernel)) && has_feat_bf16()`. The kernel must be both **built in** and **supported** by the running CPU.
- `Bf16Gemm` — a `candle_core::CustomOp2` (`cpu_fwd` only; backward and GPU paths intentionally bail). The single FFI call site. Validates 2-D shapes, contiguity, bf16 storage; extracts the `half::bf16` slices from `CpuStorage::BF16`; allocates `vec![0f32; m*n]`; calls the kernel; returns `(CpuStorage::F32(c), Shape::from((m, n)))`.
- `bf16_matmul(a, b) -> Result<Option<Tensor>>` — the safe wrapper: casts inputs to bf16 + contiguous, calls `a16.apply_op2_no_bwd(&b16, &Bf16Gemm)`. Returns `Ok(None)` when unavailable so callers fall back to candle's f32 path.

### Portability
The binary stays portable: `build.rs` only compiles the kernel on aarch64 (`cfg(has_bf16_kernel)`), and even on aarch64 the runtime `has_feat_bf16()` gate prevents calling `BFMMLA` on a CPU without FEAT_BF16 (it would `SIGILL`). Non-aarch64 targets compile with no kernel and `bf16_gemm_available()` is always `false`.

### Test
`bf16_gemm_matches_f32_reference` — 5×13×7 (odd dims, exercises the zero-padded edges). Reference: round inputs to bf16, then f32 matmul (BFMMLA's exact-product f32-accumulate numerics, modulo accumulation order). **Result: max 0.000e0 (rel 0.000e0)** — bit-exact on M2 Max. Self-skips on targets without FEAT_BF16.

### `accelerate` Cargo feature (opt-in)
Added `accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]` — Apple vecLib (Accelerate) BLAS for the CPU f32 matmul path (torch's CPU backend on Apple Silicon). Not in `default` features; compile-checked by the CI workflow on macOS.

### What's NOT done (task #25)
The backbone `Linear` matmuls do **not** call `bf16_matmul` yet, and `loader.rs` still rejects `bf16` on CPU. The kernel + `CustomOp2` + wrapper are ready; the routing is the remaining wiring.

---

## 3. MLP `Send` fix (`src/model/mlp.rs`)

### What it does
Replaces `MLP`'s `candle_nn::Sequential` (which holds `Vec<Box<dyn Module>>` — **not** `Send` because `dyn Module` has no `Send` bound) with `Vec<Box<dyn Module + Send>>` + a manual left-fold `forward`.

### Why
`candle_nn::Sequential` is unfixably non-`Send` (it's in the upstream crate). The `realtime_pipeline_types_are_send` test revealed the chain: `LFM2AudioModel` → `MLP` → `Sequential` → `Vec<Box<dyn Module>>` (non-`Send`). Without this fix, the crate doesn't compile (the `is_send::<LFM2AudioModel>()` test fails to compile). With it, `LFM2AudioModel: Send` and `LFM2AudioProcessor: Send` are both true — the worker-thread realtime pipeline is unblocked.

### The rewrite
- `model: Sequential` → `model: Vec<Box<dyn Module + Send>>`.
- `seq().add(x)` → `model.push(Box::new(x))` for `LayerNorm`, `Linear`, `Activation::Gelu`.
- `forward`: `let mut h = self.model[0].forward(x)?; for layer in &self.model[1..] { h = layer.forward(&h)?; } Ok(h)`. The comment notes candle tensors are Arc-backed handles, so rebinding `h` is a refcount bump, not a data copy — same semantics as `Sequential::forward`.
- The `model.{idx}` weight-path bookkeeping is unchanged (still `idx += 1` for every slot including no-weight GELU/Dropout), so checkpoint loading is unaffected.

### Tests
- `forward_maps_in_channels_to_out_channels` — all 4 bias/layernorm combos; shape + finiteness.
- `single_linear_no_hidden` — the no-activation edge (one Linear, no GELU).
- `mlp_is_send` — `is_send::<MLP>()` (the point of the rewrite).

---

## 4. `AudioDetokenizer: Send` (`src/audio_out.rs`)

### What it does
Adds `Send` to the `AudioDetokenizer` trait: `pub trait AudioDetokenizer: Send`.

### Why
The processor holds `Option<Box<dyn AudioDetokenizer>>` for `audio_out` and `mimi`. Without `Send` on the trait, `Box<dyn AudioDetokenizer>` is not `Send`, so `LFM2AudioProcessor` is not `Send`, so it can't move to a worker thread. Both backends (`LFM2AudioDetokenizer`, `MimiDetokenizer`) are already `Send` by construction — the bound just makes the trait object `Send`.

---

## 5. KV cache + mask memoization (`src/model/lfm2_hf.rs` + `candle_ext/transformers_utils.rs`)

> **CORRECTED** — the original §5 documented a zero-copy `KvCache` swap that was
> **reverted** by Claude as a deviation from the reference. This section now
> reflects what's actually on disk.

### What it does
Two changes, both faithful to candle-transformers' `models/lfm2.rs` (the file this port was copied from):

1. **Vendored `build_causal_mask` + `repeat_kv`** (`candle_ext/transformers_utils.rs`, new) — the exact two `crate::utils::*` helpers that `lfm2.rs` imports, backported from candle 0.10.x onto the 0.9.2 pin (adapted only `candle`→`candle_core`). The port now uses the **same** helpers as the reference rather than the hand-rolled `causal_mask`/`repeat_kv` that were previously in `lfm2_hf.rs`.

2. **Mask memoization** (`Cache::mask`) — a `HashMap<(usize, usize), Tensor>` on `Cache` that builds each boolean causal mask once per `(seq_len, kv_len)` shape via the vendored `build_causal_mask`, then reuses it across all 6 attention layers × every decode step, instead of rebuilding the mask on every call. The mask only depends on the `(seq_len, index_pos)` geometry, so caching by `(seq_len, kv_len)` is exact. `masks` survive `clear()` (a fresh turn reuses the same geometry) — matching the reference, which never drops them.

### What was reverted (and why)
Claude had previously swapped the `Tensor::cat`-based KV cache for candle-nn's preallocated `KvCache` (in-place `slice_set` + `narrow` view, no re-alloc/re-copy). That was **reverted** because it was a deviation: HF's `Lfm2HybridConvCache` and candle-transformers `lfm2.rs` both use `Tensor::cat` on the time axis. The original `cat`-based code was already the faithful port. The `KvCache` swap was the "random utility" deviation.

### What's there now
- `kvs: Vec<Option<(Tensor, Tensor)>>` — the original `cat`-and-clone KV cache (faithful to the reference).
- `cache.mask(seq_len, index_pos)?` — the memoized boolean mask (faithful to `lfm2.rs`'s `Cache::mask`).
- `masked_fill(&att, &mask, f32::NEG_INFINITY)` — the reference's `masked_fill` (boolean mask → `-inf` via `where_cond`), replacing the old hand-rolled additive `causal_mask`.
- `repeat_kv` — the vendored cat-based form (huggingface/candle#2043 — faster than the expand form, avoids strided copies).
- The detokenizer's sliding-window `add_mask` path is unchanged (still the additive f32 mask supplied by the caller — the documented deviation the reference has no custom-mask path for).

### What this fixes
The per-call mask-construction cost: the old hand-rolled `causal_mask` built a `(seq_len, kv_len)` f32 tensor via a host-side scalar double-loop + `Tensor::from_vec` on **every attention layer, every decode step** (6 layers × O(L) steps × O(L²) per mask). Now it's built once per shape and memoized — the faithful answer to the per-call mask-construction cost (the backbone sibling of the detokenizer sliding-mask issue, PR comment #1).

---

## 6. Rust CI workflow (`.github/workflows/rust-voice.yml`)

### What it does
Formalizes the `liquid-audio-rs` test suite in the build: every change to the crate (or the workflow) builds + runs `cargo test` on both x86_64 Linux and arm64 macOS.

### Triggers
`push` / `pull_request` on path filter `experiments/lfm2-audio-voice/liquid-audio-rs/**` + `.github/workflows/rust-voice.yml`, plus `workflow_dispatch`.

### Matrix
- `ubuntu-latest` — proves the cfg fallbacks compile and portable tests pass (BFMMLA self-skips, thread policy falls back to physical cores).
- `macos-latest` (arm64) — where the hardware-specific paths actually execute: the NEON BFMMLA kernel (FEAT_BF16) and the `hw.perflevel0.physicalcpu` thread policy.

### Steps
1. Checkout (`actions/checkout@v6`).
2. Install Rust stable.
3. Cache cargo (`Swatinem/rust-cache@v2` with the workspace path).
4. Linux: `apt-get install libasound2-dev` (cpal/ALSA dev-dep).
5. `cargo build --all-targets`.
6. `cargo test --lib -- --nocapture`.
7. macOS only: `cargo build --lib --features accelerate` (compile-check the Accelerate feature).

### Concurrency
`cancel-in-progress: true` (matches `test.yml`).

---

## `THREADING_PARITY.md` — Claude's writeup (partially outdated)

Claude's `THREADING_PARITY.md` documents items 1–4 above but was written **before** the mask memoization + vendored helpers (item 5) and the `to_vec4` extension. It lists "Realtime pipeline threading — DESIGN (remaining, task #24)" as the only remaining item. As-built, the mask memoization and `to_vec4` are also done. The writeup's "Remaining (task #25)" note (route bf16 through the model) is still accurate — that wiring is not done.

---

## What remains (as-built)

| Task | Status |
|---|---|
| Intra-op thread pool | ✅ done, verified |
| `accelerate` feature | ✅ done (compile-checked; not benchmarked) |
| bf16 BFMMLA kernel + `CustomOp2` | ✅ done, bit-exact verified |
| MLP `Send` fix | ✅ done, tested |
| `AudioDetokenizer: Send` | ✅ done |
| Zero-copy KV cache | ❌ **reverted** — the `KvCache` swap was a deviation; faithful `Tensor::cat` restored + mask memoization added instead |
| Mask memoization | ✅ done (faithful to `lfm2.rs`'s `Cache::mask`; eliminates per-call mask construction) |
| Vendored `build_causal_mask`/`repeat_kv` | ✅ done (candle-transformers 0.10→0.9.2 backport) |
| `to_vec4` | ✅ done (`TensorExt` trait; tested contiguous + strided) |
| Rust CI workflow | ✅ done (untracked) |
| **Route bf16 through the model** (task #25) | ❌ not done — kernel + op ready; backbone `Linear` matmuls don't call `bf16_matmul` yet; `loader.rs` still rejects bf16 on CPU |
| **Realtime worker thread** (task #24) | ❌ not done — `Send` bounds are in place (the prerequisite); the worker thread + channels + barge-in not built |
| **Commit the work** | ❌ nothing committed except `d87a52e` (deps/wiring); the 5 modified files + 6 untracked files are all uncommitted |

---

## Build + test verification (as-built)

```
$ cargo test --lib
test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo build --all-targets
Finished `dev` profile [unoptimized + debuginfo] target(s)

$ cargo test --lib -- --nocapture | grep key
BFMMLA bf16 GEMM vs f32(bf16-inputs) ref: max 0.000e0 (rel 0.000e0)
intraop threads = 8 (physical 12, logical 12)
test model::mlp::tests::mlp_is_send ... ok
test threads::tests::realtime_pipeline_types_are_send ... ok
test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

57 tests pass (was 50 before this work; +3 MLP tests, +1 bf16 test, +1 threads test, +1 Send probe, +1 `to_vec4` test).