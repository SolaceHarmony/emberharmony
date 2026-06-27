# moshi_util_compile
**Code:** `MU02` · **Source:** `moshi/utils/compile.py` · **Rust:** `- (candle eager)` · **On the LFM2-Audio inference path:** yes

## Role
This is the **CUDA-graph capture/replay + torch.compile gating layer** for Kyutai's vendored Mimi codec (and the off-path Moshi 7B LM). It does **no tensor math** of its own — it is pure execution-orchestration infrastructure that wraps an existing `nn.Module.forward` so that, on CUDA, the per-frame streaming call becomes a captured-and-replayed CUDA graph (eliminating per-step kernel-launch overhead) and hot pointwise functions get JIT-compiled lazily. Everything in this file is a **no-op / identity passthrough off CUDA**, which is exactly how the Rust candle port treats it: there is no Rust counterpart class — candle runs the same modules eagerly. It exists on the inference path only because `CUDAGraphed` wraps Mimi's encoder/decoder/transformers (`compression.py:225-229`) used to decode the model's 8-codebook audio frames into 24 kHz waveform.

## How it works
This file is **control-flow infrastructure**, not a forward pass; the "mechanism" is the graph state machine and the gating predicates.

**`torch_compile_lazy(fun)` (`compile.py:37-54`)** — a decorator that defers `torch.compile(fun)` until first *call* (so importing the module never spawns torch's compile worker pool). On each call: if env `NO_TORCH_COMPILE` is set it returns the raw `fun` (line 41); if the module-global `_compile_disabled` flag is set it calls `fun` eagerly (line 48-49); otherwise it lazily compiles once and caches `fun_compiled` (line 50-52). The `no_compile()` context manager (`compile.py:24-34`) flips `_compile_disabled` true/false around a `yield`, restoring the previous value (re-entrant). This decorator is applied to the codec's pointwise kernels elsewhere — `apply_rope` (`rope.py:11`), `_rms_norm` (`transformer.py:36`), `gating_forward_kernel` (`gating.py:13`) — so it indirectly governs RoPE, the codec-transformer RMSNorm, and the gated-FFN activation.

**`CUDAGraphed` (`compile.py:190-280`)** — the core. Wraps a callable whose tensor args must be **top-level positional** (no nested structures, no kwargs — `__call__` raises if `kwargs` present, line 219-220). State: `_graph` (a `cuda.CUDAGraph`), `_output` (the captured output tuple), `_args` (the captured *input* tensor buffers), and a `warmup_steps` counter.

The `__call__` decision tree:
1. **Bypass** (line 221-222): if `self.disable`, or `_is_cuda_graph_enabled()` is false, or we are already nested inside a graph (`in_cuda_graph()`), just call `self.func(*args)` — full passthrough.
2. **Warmup** (line 273-275): while `warmup_steps > 0`, decrement and run eagerly. This lets any `torch.compile`'d sub-functions finish compiling *before* capture (a graph can't be captured around an in-progress compile).
3. **Capture** (line 263-272): when warmup is exhausted and `_graph is None`, allocate a `cuda.CUDAGraph`, **clone the input tensors into persistent buffers** (`_clone_tensors`, line 224-230 — so the captured graph owns stable input addresses), capture `self._output = self.func(*self._args)` under `with cuda.graph(self._graph)`, then `replay()` once (line 271) because capture itself runs nothing real, and return `_output`.
4. **Replay** (line 276-280): on every subsequent call, `_match_values_copy_tensors` (line 232-259) validates each arg against the captured one — tensor args must keep **identical shape** (raises with a `NO_CUDA_GRAPH=1` hint otherwise, line 243-249) and are `copy_`'d into the persistent input buffers; non-tensor args must be unchanged by value (line 256-259). Then `_graph.replay()` re-runs the captured kernels in place and returns the *same* `_output` tensor objects (their storage was overwritten by the replay). `reset()` (line 210-216) nulls `_graph/_output/_args` so the next call re-captures — used when KV-cache or shapes change.

**Enable/disable predicates.** `_is_cuda_graph_enabled()` (`compile.py:169-175`): false if module-global `_disable_cuda_graph` is set, or if env `NO_CUDA_GRAPH` is a non-empty non-`{0,no,n}` string; else true. `no_cuda_graph()` (line 178-187) is the scoped disable. `in_cuda_graph()`/`_set_in_cuda_graph()` (line 153-166) maintain a reentrancy guard so nested `CUDAGraphed` calls degrade to eager (only the outermost wrapper captures).

**`cuda_graph(func, warmup_steps=1)` (`compile.py:283-287)`** — convenience factory: returns `func` unchanged when graphing is globally disabled, else `CUDAGraphed(func, warmup_steps)`.

**`Checkpoint` / `simple_checkpoint` (`compile.py:57-146`)** — a hand-rolled `torch.autograd.Function` activation-checkpoint (forward under `no_grad`, recompute in backward) that is FSDP- and compile-safe. This is **training-only** and **off the inference path**; it never runs during LFM2-Audio generation.

**The single binding fact for this component's relevance:** in Mimi's `_init_streaming_state` (`compression.py:218-229`), `disable = device.type != 'cuda'` (line 220), and that `disable` is passed into every `CUDAGraphed(...)` wrapping `encoder_transformer`, `decoder_transformer`, `encoder`, `decoder`. So on CPU/MPS the codec's per-frame decode runs eager; only on a CUDA box does the streaming decode become a replayed graph.

## Dtypes & shapes
This component is **dtype- and shape-transparent**: it neither casts nor reshapes. It only *asserts* that replay-time tensor args match capture-time shape/dtype exactly (`compile.py:243-250`), and `copy_`'s values in place. The dtypes/shapes below are those of the *wrapped* Mimi callables it passes through unchanged.

| Wrapped callable (via `CUDAGraphed`) | Input dtype+shape (passthrough) | Output dtype+shape (passthrough) |
|---|---|---|
| `encoder` (`compression.py:228`) | waveform f32/bf16 `(B,1,1920·k)` | latent model-dtype `(B,512,k·…)` @25Hz |
| `decoder` (`compression.py:229`) | latent model-dtype `(B,512,·)` @25Hz | waveform **f32** `(B,1,1920·k)` @24kHz |
| `encoder_transformer` / `decoder_transformer` (`compression.py:225,227`) | model-dtype `(B,512,T')` | model-dtype `(B,512,T')` |
| `torch_compile_lazy` kernels (`apply_rope`/`_rms_norm`/`gating_forward_kernel`) | model-dtype (bf16 cuda) tensors | same dtype, same shape |

No internal promotions occur here — the f32-upcast-for-RMSNorm, the bf16 codec weights, the int Mimi codes (u32 in Rust) all live in the wrapped modules, not in this file.

## Wiring
**Upstream (who constructs/feeds this):**
- [moshi_compression](../models/compression.md) builds the four `CUDAGraphed` wrappers in `_MimiState` and drives them per streaming frame (edge: latent model-dtype `(B,512,·)` @25Hz / codes int into the wrapped encoder/decoder; nothing flows *into* `compile.py` except the modules + tensor args).
- [moshi_streaming](../modules/streaming.md) wraps `_set_exec_mask` in `CUDAGraphed` (`streaming.py:36,209`).
- [moshi_transformer](../modules/transformer.md), [moshi_gating](../modules/gating.md), [moshi_rope](../modules/rope.md), [moshi_lm](../models/lm.md) (off-path) apply `torch_compile_lazy` / `CUDAGraphed` to their hot functions.

**Downstream (who consumes this component's output):** the *replayed* output is just the wrapped module's output, so the real consumer is whoever consumes Mimi's decode — i.e. the demo/processor audio-out path:
- [moshi_compression](../models/compression.md) consumes the graphed encoder/decoder output (latent @25Hz, waveform f32 `(B,1,1920)`).
- [core_processor](../../processor.md) / [demo_chat](../../../demo/chat.md) consume the resulting waveform **f32 `(1,1,1920)` @24kHz** per frame to play audio.

## Python ↔ Rust
There is **no Rust file** for this component — by design (`PYTHON_VS_RUST.md §2.3` "upstream reuse", `ARCH_1_MIMI_CODEC.md §4,§7.3`). The mapping is *deletion-by-equivalence*:

| Python symbol | Rust | Divergence (deliberate) |
|---|---|---|
| `CUDAGraphed`, `cuda_graph` | — (none) | candle runs Mimi (the `moshi` crate) **eager**; no CUDA-graph capture layer exists. `disable = device.type != 'cuda'` (`compression.py:220`) means Python *itself* runs this eager off-CUDA, so on CPU/Metal the two are identical execution (`PYTHON_VS_RUST.md §2.1`). |
| `torch_compile_lazy`, `no_compile` | — (none) | no JIT-compile concept in candle; ops are dispatched directly. The decorated kernels (RoPE, codec RMSNorm, gated FFN) are ported as plain candle ops in the `moshi` crate / `transformer.rs`. |
| `Checkpoint`, `simple_checkpoint` | `// PORT:` stub (`wrap_activation_checkpoint`, `PORT_STATUS.md §"// PORT: markers"`) | no autograd/backward in the inference port → no activation checkpointing. |
| `in_cuda_graph`/`no_cuda_graph`/`_is_cuda_graph_enabled` | — | no graph state machine to gate. |

This is the same class of divergence as `moshi_util_autocast` (`TorchAutocast` → candle no-op) — torch execution-mode plumbing with no semantic effect on the numbers.

## Precision / gotchas
- **Numerically inert.** This file changes *no* values; it only affects *latency* on CUDA. The 1e-6 cross-library floor (`PYTHON_VS_RUST.md §1.4`) is unrelated to this component. A graph-replayed Mimi and an eager Mimi are bit-identical on the same device — `CUDAGraphed` literally re-runs the captured kernels.
- **Shape-frozen replay.** The load-bearing gotcha: once captured, **every call must use the exact same tensor shapes** (`compile.py:243-249`). This is why Mimi streaming requires input lengths that are exact multiples of `frame_size=1920` (`ARCH_1_MIMI_CODEC.md §5`) — the per-frame call shape must be constant for replay to be valid; a ragged frame would trip the shape assertion. `reset()` (line 210) is the escape hatch when KV-cache/shape changes.
- **Stable input buffers.** `_clone_tensors` at capture (line 224-230) plus `copy_` at replay (line 250) means the graph owns persistent input addresses; callers must not assume the returned `_output` is a fresh allocation each call — it is the **same tensor object** overwritten by replay (aliasing hazard if held across frames).
- **Off-CUDA = pure passthrough.** `disable=True` (CPU/Metal) makes `__call__` return `self.func(*args)` verbatim (line 221-222), so the Rust port's absence of this layer is faithful, not a gap.
- **Training-only `Checkpoint`.** Do not conflate the activation-checkpoint `autograd.Function` with the CUDA-graph path; it has no role in generation and is a `// PORT:` stub in Rust.
