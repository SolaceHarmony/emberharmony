# moshi_streaming
**Code:** `MO06` · **Source:** `moshi/modules/streaming.py` · **Rust:** `moshi crate streaming` · **On the LFM2-Audio inference path:** yes

## Role
`streaming.py` defines the **base streaming state-machine API** that every Mimi codec submodule inherits — `State` (`streaming.py:25`), `StreamingModule[StateT]` (`streaming.py:54`), and the trivial concrete `StreamingContainer` (`streaming.py:214`). It carries no neural weights and runs no math itself; it is the plumbing that lets stateful causal modules (causal `StreamingConv1d`/`ConvTranspose1d`, the SEANet enc/dec resnet blocks, the codec `StreamingTransformer`/`StreamingMultiheadAttention`, and the higher-level `MimiModel`/`LMModel`/`LMGen`) hold per-stream buffers (conv ring-buffers, KV caches, partial-frame remainders) across successive `forward()` calls so that feeding audio frame-by-frame produces the same result as feeding the whole sequence at once. It exists on the LFM2-Audio path purely because the demo/server decode loop drives the Mimi detokenizer in `mimi.streaming(1)` context (`demo/chat.py`, `moshi/server.py:59`).

## How it works
The contract is a **context-manager-scoped state tree**, not a forward pass. Mechanism:

- **`State` dataclass** (`streaming.py:25-48`): a base per-module record holding `batch_size`, `device`, and an `exec_mask` (a `(B,)` bool tensor, `streaming.py:35`, default all-`True`). Subclasses (e.g. `_StreamingConv1dState` in `conv.py:162`, `_MimiState` in `compression.py:98`, `_MHAState`/`_LayerState`/`_TransformerState` in `transformer.py`) extend it with the actual buffers. `State` supports the context protocol (`__enter__`/`__exit__`, no-ops by default at `streaming.py:44-48`) and a `reset(reset_mask)` that, per-batch-row, ORs the row's exec_mask back to `True` (`streaming.py:42`); buffer-carrying subclasses override `reset` to **zero `previous` and re-arm `first`** where `reset_mask` is `True` (`conv.py:166-169`, using `torch.where(reset_mask.view(-1,1,1), zeros, previous)` so resets are per-stream within a batch).

- **`streaming(batch_size)` context** (`streaming.py:131-137`) is the entry point. It builds an `ExitStack`, calls `_start_streaming` (`streaming.py:110`), then registers `_stop_streaming` as the exit callback. `_start_streaming` walks the module tree once via `_apply_named_streaming` and, for each `StreamingModule`, asserts no state is live, calls the subclass `_init_streaming_state(batch_size)` (abstract, `streaming.py:126`) to allocate that module's buffers, enters the state as a context (`exit_stack.enter_context(state)`), and stores it in `module._streaming_state` (`streaming.py:113-115`). On exit the stack nulls every `_streaming_state` (`streaming.py:119-123`), so leaving the `with` block **voids** all streaming state. `streaming_forever(batch_size)` (`streaming.py:128`) just calls `__enter__` and never exits — used by the persistent server/inference loops (`server.py:59`, `run_inference.py:89`).

- **Tree traversal + detach** (`_apply_named_streaming`, `streaming.py:88-108`): a DFS over `named_children()` that collects every descendant `StreamingModule` into a memoized `_cached_children` list, then applies `fn(name, child)` to each. The `_streaming_detached` flag (`streaming.py:78-86`, `93`) lets a submodule **opt out** of inheriting a parent's streaming request unless it is the direct receiver (prefix `""`) — this is how the RQ-Transformer's inner depth transformer streams over the codebook axis independently of the outer time axis. `is_streaming` is just `_streaming_state is not None` (`streaming.py:75`).

- **`reset_streaming(reset_mask=None)`** (`streaming.py:139-156`): broadcasts a `reset` to every live sub-state. `reset_mask=None` ⇒ reset all `B` rows (`torch.ones((B,), bool)`, `streaming.py:153`); otherwise per-row, enabling **desynchronized batched streams** (reset one conversation while others continue). Called between turns (`server.py:134,167`).

- **`exec_mask` machinery** (`streaming.py:183-211`, `State.set_exec_mask` at `38`): a `(B,)` bool gate saying "for these rows, advance internal state as if real data arrived; for the others, leave state untouched." It lets a single batched step service streams that are temporally misaligned. There is "no magic" (`streaming.py:193`): each subclass must itself honor the mask. The setter wraps the broadcast in a `CUDAGraphed` callable (`streaming.py:207-211`), **disabled whenever `device.type != 'cuda'`** (`streaming.py:208`) — i.e. it's a plain function call off-GPU.

- **State get/set** (`streaming.py:158-181`) snapshot and restore the whole `{name: state}` dict, used for KV-cache save/restore and the `LMGen` plumbing that registers a `set_exec_mask_callback` so the LM and Mimi exec masks stay coupled (`lm.py:529,538-541,653-659`).

`StreamingContainer` (`streaming.py:214-217`) is the no-op concrete subclass: `_init_streaming_state` just returns a bare `State(batch_size, device)` taken from the first parameter's device. It's the base for modules that hold no buffers of their own but must propagate streaming to children (SEANet enc/dec, `ProjectedTransformer`).

## Dtypes & shapes
This module allocates **bookkeeping tensors only**; the audio/latent tensors flow through the subclass `forward()`s, not here.

| Object | dtype | shape | notes |
|---|---|---|---|
| `State.exec_mask` | bool | `(B,)` | `streaming.py:35`; all-True at init |
| `reset_mask` arg | bool | `(B,)` | `streaming.py:153` |
| `_StreamingConv1dState.previous` (subclass) | **model dtype** (bf16 Metal / f32 CPU; Python bf16/cuda) | `(B, C_in, k_eff − stride)` | `conv.py:240`; the causal ring-buffer |
| `_StreamingConv1dState.first` (subclass) | bool | `(B,)` | `conv.py:242` |
| `_MimiState` graphs (subclass) | — | — | `CUDAGraphed | None`, all `None` off-cuda (`compression.py:222-227`) |
| (Mimi step audio in/out — carried by subclass forward) | int codes u32 / f32 wav | `(B,8,T)` codes ↔ `(B,1,1920·T)` @24kHz | not allocated here |

No dtype promotion happens in this file; the f32-upcast-for-norm/softmax, f64 mel, etc. all live in the leaf modules. The only "dtype" decisions here are: exec_mask/first = `bool`, and the conv ring-buffer inherits `next(parameters()).dtype` (`conv.py:238`).

## Wiring
**Upstream (who enters streaming on these):** the LFM2-Audio decode loop opens `mimi.streaming(1)` / `streaming_forever(1)` around the Mimi codec — driven by [core_processor](../../model/lfm2_audio.md)'s `decode()` dispatch and the demo loop ([demo_chat](../../demo/chat.md), `moshi/server.py`). Edge carried: the per-frame audio codes `int/u32 (B,8,T)` that the Mimi `forward`/`decode` consumes inside the `with` block.

**This base class is mixed into (subclassed by):**
- [moshi_compression](../models/compression.md) — `CompressionModel`/`MimiModel` (`compression.py:40,105`); `_MimiState` carries the CUDA-graphed enc/dec transformers.
- [moshi_seanet](seanet.md) — `SEANetEncoder`/`Decoder`/`ResnetBlock` are `StreamingContainer`s (`seanet.py:20,96,242`).
- [moshi_conv](conv.md) — `StreamingConv1d`/`ConvTranspose1d` (`conv.py:172`) hold the causal ring-buffer + partial-frame state.
- [moshi_transformer](transformer.md) — `StreamingTransformer`/`Layer`/`MultiheadAttention` (`transformer.py:328,586,789`) hold the KV cache + RoPE offset.
- [moshi_lm](../models/lm.md) — `LMModel`/`LMGen` (`lm.py:49,550`) are off the LFM2-Audio compute path (reference Moshi LM) but reuse the identical state API.

**Downstream (consumes this component's output / API):** the modules above; their decoded waveform `f32 @24kHz (B,1,L)` exits the streaming context back to [core_processor](../../model/lfm2_audio.md) → audio sink.

## Python ↔ Rust
The Rust port (`moshi-0.6.4/src/streaming.rs`) keeps the **concept** but inverts the **shape of the abstraction** — this is a deliberate, idiomatic divergence (PYTHON_VS_RUST.md §2.3 "upstream reuse instead of re-implementation": the Mimi codec is reused as Kyutai's own `moshi` crate, so its streaming primitives come along verbatim rather than being re-ported from the Python).

| Python (`streaming.py`) | Rust (`moshi crate streaming.rs`) | Note |
|---|---|---|
| `StreamingModule[StateT]` (nn.Module, context-manager state-tree) | `trait StreamingModule { fn step(&mut self, xs:&StreamTensor, mask:&StreamMask)->Result<StreamTensor>; fn reset_state(&mut self); }` (`streaming.rs:188`) | Python hangs state on the module via `with`; Rust threads it through an explicit `&mut self` per-step `step()` + `reset_state()`. No `ExitStack`/context manager — ownership/borrow does the scoping. |
| `State.exec_mask` `(B,)` bool tensor | `StreamMask(Option<MaskInner{cpu:Vec<bool>, mask:Tensor}>)` (`streaming.rs:20,42`) | Rust keeps **both** a CPU `Vec<bool>` (for cheap `is_active(i)` row checks, `streaming.rs:48`) and the device tensor; empty mask ≡ all-active. |
| implicit per-module buffer (`previous`, KV cache) inside each `State` subclass | `StreamTensor(Option<Tensor>)` (`streaming.rs:11`) with `cat2`/`split`/`narrow`/`reset` (`streaming.rs:113-185`) | First-class "maybe-empty time-buffer" type; `cat2` concatenates prev+new on the time dim, `split` peels off the consumable prefix — the explicit reification of Python's `previous[:]` ring-buffer bookkeeping. |
| `streaming(bs)` / `streaming_forever(bs)` context | none — caller holds the `&mut` module and loops `step()` | The Rust port runs a **synchronous streaming generator** (PORT_STATUS.md): no async, no context manager. |
| `reset_streaming(reset_mask)` (per-row) | `reset_state()` (whole-module) + `StreamingBinOp::reset_batch_idx(i,B)` (per-row zero of row `i`, `streaming.rs:260`) | Per-stream reset is done by zeroing one batch slice rather than a `where(reset_mask,…)`. |
| `set_exec_mask` wrapped in `CUDAGraphed(disable = device≠cuda)` (`streaming.py:207-209`) | no CUDA-graph layer; mask is a plain `Tensor`/`Vec<bool>` | Matches PYTHON_VS_RUST.md §2.2 (custom CUDA kernels → portable candle ops) and the CUDAGraphed-disabled-off-cuda fact. |
| `StreamingBinOp` (Python: ad-hoc inside conv/seanet) | explicit `StreamingBinOp{prev_lhs, prev_rhs, op, dim}` with buffered `step` (`streaming.rs:202-273`) | Rust factors the "buffer two streams, operate on the common-length prefix, retain the remainder" pattern into a reusable struct; refuses a non-empty mask with leftover buffer (`streaming.rs:250`). |
| `StreamingContainer` | `Map<T: Module>` (`streaming.rs:276`, no buffering — `reset_state` no-op) | The trivial pass-through wrapper. |

## Precision / gotchas
- **No numerics here** — this file allocates only bool masks and (in subclasses) zero-initialized buffers; the cross-library f32 floor, RMSNorm bf16 multiply order, and f64 mel all live downstream. The conv ring-buffer (`conv.py:240`) inherits the model dtype, so on CPU it's f32 and on Metal bf16, consistent with the rest of the codec.
- **Streaming ≡ non-streaming is a *contract*, not a guarantee here**: correctness depends on each leaf `forward` honoring `previous`/`exec_mask`. `streaming.py:193` is explicit that the base class does no enforcement. A subclass that ignores `exec_mask` will silently corrupt desynchronized batched streams.
- **`first` flag (`conv.py:164,242,169`)** distinguishes the very first chunk (no left-context to prepend) from subsequent chunks; on `reset` it is re-armed to `True`. Off-by-one in the prepend logic would shift causal alignment by one frame — handled in `conv.py`, not here.
- **`streaming_forever` never voids state** (`streaming.py:128-129`): the server/inference loops rely on `reset_streaming()` at turn boundaries (`server.py:134,167`) instead of leaving the context. Forgetting that reset leaks one stream's KV/conv history into the next turn.
- **exec_mask CUDAGraphed is disabled off-cuda** (`streaming.py:208`): on CPU/Metal it is a plain call — semantically identical, just uncaptured. The Rust port has no analog at all (no CUDA-graph layer), which matches the "candle-ops-not-CUDA-kernels" divergence.
- **`StreamMask` empty-vs-all-True**: in Rust an empty `StreamMask` means "everyone active" (`is_active` returns `true` when `cpu()` is `None`, `streaming.rs:48-50`); the equivalent Python default is an all-`True` `exec_mask`. Treating "empty" as "none active" would be the inverted-mask bug to watch for when reading the Rust.
- **`StreamingBinOp` rejects mask + leftover buffer** (`streaming.rs:250-254`): combining a stream mask with a buffered binary op is an explicit `bail!`, a Rust-side invariant with no Python equivalent — relevant if a future codec change tries to mask a streaming add/mul.
