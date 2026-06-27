<!-- topic: Moshi Utilities -->
# MU03 · TorchAutocast
**Code:** `MU03` · **Source:** `moshi/utils/autocast.py` · **Rust:** `- (explicit dtype)` · **On the LFM2-Audio inference path:** no

## Role
`TorchAutocast` is a thin wrapper around `torch.autocast` (PyTorch AMP / mixed-precision context manager) that adds an *enable/disable toggle* and a friendlier error message. It exists so cluster code can flip mixed-precision on or off per-architecture/per-machine (different GPUs have different bf16/f16 support) without sprinkling `if enabled:` branches at every call site. It is **vendored Kyutai/Moshi machinery** carried in wholesale with the rest of `moshi/`; in `liquid_audio` this exact class is **dead code** — nothing imports or constructs it (grep for `TorchAutocast` matches only its own definition). The autocast that the LFM2-Audio code actually uses is `accelerator.autocast()` in the trainer and the conformer's `avoid_float16_autocast_context` / `torch.amp.autocast(enabled=False)` guards in the mel front-end, none of which route through this file.

## How it works
This is a context-manager utility, not a tensor op — there is no forward pass, no norm, no attention, no RoPE, no convolution. The mechanism is entirely Python context-manager plumbing over `torch.autocast`:

- **Construction** (`autocast.py:26-27`): `__init__(self, enabled, *args, **kwargs)` eagerly builds `self.autocast = torch.autocast(*args, **kwargs)` when `enabled` is truthy, else stores `None`. The `*args, **kwargs` are forwarded verbatim to `torch.autocast` (e.g. `device_type="cuda"`, `dtype=torch.bfloat16`/`torch.float16`, `cache_enabled=...`). So "enabled=False" is a true no-op: the wrapped object is `None` and never touches PyTorch's autocast state machine.
- **Enter** (`autocast.py:29-40`): `__enter__` returns immediately if `self.autocast is None` (the disabled path). Otherwise it delegates to `torch.autocast.__enter__()`, which pushes a dtype-casting policy onto PyTorch's dispatcher: while the context is live, eligible ops (matmul/conv/linear/SDPA) auto-cast their *inputs* to the autocast `dtype` (typically bf16/f16), while precision-sensitive reductions (softmax, layernorm/rmsnorm internals, loss) are kept in f32 by PyTorch's autocast op allow/deny lists. If `torch.autocast.__enter__()` raises `RuntimeError` (e.g. the GPU/driver can't honor the requested fast dtype), it re-raises with a hint to use `autocast_dtype=float16`, reading back `self.autocast.device` and `self.autocast.fast_dtype` for the message (`autocast.py:34-40`).
- **Exit** (`autocast.py:42-45`): symmetric — no-op when `None`, otherwise pops the autocast policy via `torch.autocast.__exit__(*args, **kwargs)`, restoring the prior dispatcher state.

Key behavior to note as "mechanism": **autocast never changes stored weight dtypes**; it only changes the *compute* dtype of dispatched ops inside the `with` block by casting operands. That distinction is exactly why the inference port can drop it — in the port, compute dtype is set explicitly per tensor/op rather than inferred from an ambient thread-local context.

Where the *concept* actually bites in this codebase (none of it through `TorchAutocast`):
- Training wraps the model call in `with self.accelerator.autocast():` (`trainer.py:176`, `trainer.py:194`) so the bf16 forward runs under HuggingFace Accelerate's autocast, with the loss accumulated in f32.
- The mel front-end deliberately *disables* autocast to keep full f32 range through the STFT/log: `with torch.amp.autocast(x.device.type, enabled=False):` (`conformer/processor.py:444`, `conformer/processor.py:468`).
- The conformer attention guards against an active **f16** autocast by forcing bf16-or-f32 compute via `avoid_float16_autocast_context()` (`conformer/utils.py:25-38`, used at `conformer/mha.py:266,270,391,395`).

## Dtypes & shapes
This component manipulates an ambient *compute-dtype policy*, not a tensor; it has no input/output tensor shape of its own. The table describes the dtype effect of the context on tensors that flow through ops dispatched inside it.

| Aspect | Disabled (`enabled=False`) | Enabled (`enabled=True`, typical) |
|---|---|---|
| Input to `__init__` | `enabled: bool`, `*args/**kwargs` for `torch.autocast` | same |
| Wrapped object | `None` | `torch.autocast(device_type=..., dtype=bf16/f16)` |
| Effect on op operands inside `with` | unchanged (caller's dtype, e.g. bf16 weights / f32 mel) | matmul/conv/linear/SDPA operands cast to autocast dtype (bf16/f16) |
| Kept in f32 by PyTorch policy | n/a | softmax, norm internals, loss reduction (autocast deny-list) |
| Weight storage dtype | never modified | never modified (bf16 on disk stays bf16) |
| Return of `__enter__` | `None` | `None` (used as plain `with`, no `as` target) |

Reference global dtypes for context (set elsewhere, merely *observed* by autocast): backbone/codec weights bf16 on disk; Python default compute bf16 on cuda; mel computed in f32/f64 with autocast explicitly off; token ids int64.

## Wiring
**Upstream (who would construct it):** in `liquid_audio`, **nobody** — `TorchAutocast` has zero call sites. In upstream Moshi it is constructed by the Moshi LM / solver code (out of this repo's path). The conceptually-adjacent, actually-used autocast in this codebase comes from the trainer ([core_trainer](CO04-Trainer), via `accelerator.autocast()`) wrapping the model forward, and from the mel front-end ([conformer_processor](CF04-Mel-Frontend)) and conformer attention ([conformer_mha](CF02-RelPos-MHA) + [conformer_utils](CF06-Conformer-Utils)) toggling autocast off / away from f16.

**Downstream (who consumes its output):** none. It produces no tensor and no value (`__enter__` returns `None`). On the LFM2-Audio inference path there is no consumer. The only "consumer" of the *concept* is the trainer's bf16 forward + f32 loss, i.e. [core_trainer](CO04-Trainer) — and that path uses `accelerator.autocast()`, not this class.

## Python ↔ Rust
**Symbol mapping:** `TorchAutocast` → **no Rust symbol** (`- (explicit dtype)`). This is a deliberate, documented divergence, not a missing port:

- **PORT_STATUS.md** records "torch autocast (`avoid_float16_autocast_context`) — candle has no autocast; compute dtype is explicit." candle/Rust has no ambient thread-local mixed-precision dispatcher, so a context manager that mutates such state has nothing to wrap.
- **PYTHON_VS_RUST.md §2.1** (device & dtype defaults are explicit, device-agnostic) and **§2.4** (precision order is explicit per-op) are the umbrella rationale: every loader takes `device: &Device` + `dtype: DType`, and each op's compute dtype is chosen at the call site. The autocast "policy" is therefore inlined as concrete casts where they matter (e.g. RMSNorm composed to upcast to f32 for the norm + weight-multiply then cast back; mel run in f64→f32). **PYTHON_VS_RUST.md §2.6** maps `accelerator.autocast()` itself to "candle equivalents" — the model already runs at the load dtype and the loss is computed in f32 inside the model, so there is **no separate cast context** (`trainer.rs:21`, `trainer.rs:429-431`).
- The one piece of autocast logic that *was* ported is the conformer guard, as a **pure function**: `avoid_float16_autocast_context(autocast_dtype, bf16_supported) -> Option<DType>` (`conformer/utils.rs:74`) reproduces the decision "f16 active → bf16 if supported else f32; otherwise no override" without any global state. `TorchAutocast` itself has no analog because it is unused.

## Precision / gotchas
- **Dead vendored code.** Do not treat the absence of a Rust `TorchAutocast` as a port gap — the Python class is never instantiated in `liquid_audio`. Mis-reading this as "missing" would be a false positive.
- **Autocast ≠ weight cast.** It changes *op compute* dtype by casting operands inside the block, never the stored bf16 weights. The port's "explicit dtype" model is faithful precisely because it controls those op-level dtypes directly.
- **f16-vs-bf16 hazard the wrapper exists for.** The friendlier `RuntimeError` (`autocast.py:34-40`) is about hardware that rejects the requested fast dtype; the real codebase handles the analogous hazard at the conformer (`avoid_float16_autocast_context`), forcing bf16/f32 so f16's narrow range never corrupts attention scores.
- **Front-end must stay un-autocast.** The mel/STFT is precision-sensitive and is wrapped in `autocast(enabled=False)` (`processor.py:444,468`); the Rust port honors this by running the mel in f64→f32 on CPU (PYTHON_VS_RUST.md §1.4) rather than relying on any ambient context. If autocast policy ever leaked into the front-end it would silently degrade the STFT — which is exactly the failure mode `enabled=False` guards against.
- **No off-by-one / special-token concerns** apply here — this component touches no codes, no EOAudio, no sampling.
