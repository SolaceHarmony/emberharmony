# candle-flashfftconv — architecture

The Metal/candle port of FlashFFTConv that runs LFM2-Audio's long convolutions natively
in the Tauri desktop binary (no CUDA, no torch). The conv operators that ML stacks
normally gate behind custom CUDA kernels, reimplemented as candle `CustomOp`s that run on
**CPU** (a faithful reference) **and Metal** (a real fused kernel) from one call site.

---

## 1. The reference chain

Every kernel here is the third link in a verified chain — nothing is ported by
resemblance:

```
   CUDA (truth)                MLX oracle                     candle (this crate)
   ────────────                ──────────                     ───────────────────
   flash-fft-conv/csrc/    →   mlxports/.../monarch_metal/ →  src/*.rs + inline MSL
   monarch_cuda/*.h,*.cu       butterfly_*_fused.py,           CustomOp{2,3}: CPU ref
   butterfly/*.cu              the fused "zoo"                  + Metal kernel, gated
   conv1d/*.cu                 (machine-precision vs naive)     by a differential test
```

The MLX layer is a fast-to-write **oracle**: each kernel is gated against a naive scalar
reference to machine precision, so the design is proven before it is translated here. This
crate is the **deliverable** — what ships. Each op carries a CPU reference and a Metal
kernel and a `metal == cpu` test, so correctness is differential, not asserted.

---

## 2. Operators

| Op | What | Status |
|---|---|---|
| `depthwise_conv1d` / `depthwise3_causal` | LFM2 short-conv / FlashFFTConv short filter | metal == cpu, 5.96e-8 |
| `butterfly_fft_forward` / `_inverse` | Monarch butterfly FFT (row-DFT→twiddle→col-DFT), un-fused (3 dispatches) | verified |
| `monarch_conv` | long conv `IFFT(FFT(u) ⊙ k_f)`, the **differential oracle** for the fused path | == circular conv, 9.7e-8 |
| `fused_fft_conv` | single-dispatch **radix-2** conv (`rfft→⊙→irfft→+u·D`), pow2 ≤ 1024 | == direct linear conv, 1.2e-7 |
| `butterfly_fft_forward_fused` | fused **tensor-core** forward butterfly (1 dispatch) | metal == cpu == un-fused, 3.8e-6 |
| **`monarch_conv_fused`** | the **full** fused tensor-core conv (1 dispatch) | == `monarch_conv` 1.5e-8, any N,L |
| `warmup` | pre-compile the fused kernels at init | — |

---

## 3. The fused tensor-core Monarch conv (`monarch_conv_fused`)

The headline kernel. Where `monarch_conv` is ~7 dispatches (forward FFT ×3, native
`complex_mul`, inverse FFT ×3) with device round-trips between each, this is **one**
dispatch. Every sub-DFT is an 8×8 `simdgroup_float8x8` matmul on Apple's matrix units (the
CUDA `wmma` analog) with **fp32 accumulate** — better than CUDA's pure-fp16 wmma. The
`[N,L]` intermediate never leaves threadgroup memory.

### Dataflow (one threadgroup per `(b,h)`, one simdgroup = 32 lanes)

```
  u[N,L] (device, real)
     │  preamble: zero-fill stage → ux[Np,Lp]  (threadgroup)
     ▼
 ┌─ stage 1  row DFT / L     A = ux @ dL        2 GEMMs ──► A (axr,axi)
 │  stage 2  twiddle         A ·= tw            in-kernel elementwise
 │  stage 3  col DFT / N     B = dN @ A         4 GEMMs ──► B (bxr,bxi)
 │  stage 4  × k_f           B ·= k_f[b,h]      in-kernel elementwise   ◄── the fused multiply
 │  stage 5  col IDFT / N    A = idN @ B        4 GEMMs ──► A
 │  stage 6  conj-twiddle    A ·= itw           in-kernel elementwise
 │  stage 7  row IDFT / L    out = Re(A @ idL)·1/(N·L)   2 GEMMs ──► out[N,L] (device, real)
 └─ simdgroup_barrier between every stage; nothing returns to device until stage 7
```

**Ping-pong (A↔B):** the col-DFT stages (3, 5) read *every* row of their input, so writing
the result back in place would corrupt later tiles. Stage 3 reads A → writes B; stage 5
reads B → writes A. Two complex threadgroup buffers, no aliasing.

### Edge tiles — any N,L, no caller padding

The `simdgroup` GEMM works on 8×8 tiles, so dims must be multiples of 8 *inside* the
kernel. We get that without forcing the caller to pad, by applying the fork's zero-fill
principle where it's cheapest:

- **Constant matrices padded at pack time.** `pack_full` zero-pads each matrix to the next
  multiple of 8 (`Np=ceil8(N)`, `Lp=ceil8(L)`). A zero row/col contributes nothing to a
  DFT sum, so the padded GEMM is bit-correct.
- **The intermediate is padded threadgroup space** `[Np,Lp]`; every 8×8 `simdgroup_load`
  is in-bounds and padding carries real zeros.
- **The ragged `[N,L]` boundary touches only 3 elementwise spots:** the `u` staging
  (zero-fill), the `×k_f` read (valid indices), the output write (valid indices). The
  `1/(N·L)` scale uses the *true* N·L.

Verified on non-multiples-of-8 (6×10, 12×20, 8×24, 10×6) vs the oracle at ~1e-8.

### Dispatch contract

```
grid        = (B·H, 1, 1) threadgroups      // one tile of work per (b,h)
threadgroup = (32, 1, 1)                     // one simdgroup
threadgroup memory = 5·Np·Lp + 256 floats   // ux, axr, axi, bxr, bxi, scratch  (<32 KB)
buffers: 0 params · 1 u · 2 packed · 3 k_f · 4 out
```

`packed` block order (real/imag matrices separated, twiddles interleaved, all padded):
`dLr | dLi | dNr | dNi | tw | idNr | idNi | idLr | idLi | itw`.

This is faithful to the verified MLX `butterfly_forward_fused.py` `_fwd_src`/`_inv_src` and
to the candle oracle `butterfly_fft_inverse` (ColDft(idN)→Twiddle(itw)→RowIDftReal). The
fork's standalone `simdgroup_gemm.metal` does the same DFT-as-matmul with per-tile `As/Bs`
staging; our pack-time padding is the simpler equivalent for fused stages.

---

## 4. Compiled kernel vs. instances (the pipeline cache)

`metal_util::pipeline` compiles MSL → `MTLComputePipelineState` once and caches it. The
cache is **global and thread-safe** (`OnceLock<Mutex<HashMap>>`), not thread-local:

- The **compiled kernel** (`ComputePipeline`, verified `Send + Sync`) is immutable compiled
  code — compiled **once process-wide** and shared across every thread.
- An **instance of the kernel** is one dispatch: a command encoder + buffers, built cheaply
  per call, per thread.

So the realtime worker thread and the main thread share a single compile (verified: a
worker thread does **0 recompiles** and returns a bit-identical result). `warmup()` moves
that one compile to engine-init, off the first audio frame.

---

## 5. Precision regimes

The FlashFFTConv CUDA kernels run in bf16/f16; the trained weights were fit *around* that
rounding. So there are correct ports at three precisions, not one:

| Regime | Where | vs f64 truth (256-pt circular conv) |
|---|---|---|
| **bf16-faithful** (bug-for-bug) | `monarch_conv_bf16` | 2.69e-1 |
| **f32** (clean) | `monarch_conv`, `monarch_conv_fused` | 7.6e-6 |
| **double-double** (~f64) | `fused_fft_conv_dd`, `complex_mul_dd` | ≤ 1.2e-7 |

The fused tensor-core path is f32 today (fp32 accumulate). bf16/fp16 fused variants — load
8×8 tiles as `simdgroup_bfloat8x8`/`half8x8`, keep the fp32 accumulator — are the next
step (the MLX zoo already proves them).

---

## 6. Verification discipline

Every op is gated `metal == cpu`, and the fused path additionally `== monarch_conv` (the
un-fused oracle) and `== a direct circular/linear convolution`. Run from this dir:

```
cargo test --features metal -- --nocapture
```

30/30 green, zero warnings. The fused conv tests live in `src/fused_monarch.rs`:
`fused_conv_cpu_matches_monarch_conv` (0.0), `fused_conv_metal_matches_cpu_and_oracle`
(1.5e-8), `fused_conv_edge_dims` (the edge-tile gate, ~1e-8), `fused_conv_matches_circular`
(9.7e-8), and the cross-thread cache proof `fused_forward_compiled_once_shared_across_threads`.

---

## 7. Follow-ons

- bf16 / fp16 fused variants (storage low-precision, fp32 accumulate).
- gated / padded fused variants (the rest of the MLX zoo).
- Wiring `monarch_conv_fused` into the LFM2 backbone call site + a `warmup()` at engine init.
