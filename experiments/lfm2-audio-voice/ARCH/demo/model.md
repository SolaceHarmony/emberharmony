# demo_model
**Code:** `DM02` · **Source:** `demo/model.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
`demo/model.py` is the demo's **singleton-loader / warmup module**: 26 lines whose entire job is to construct, at import time, the three shared objects the gradio chat needs — `proc` (the `LFM2AudioProcessor`), `lfm2_audio` (the `LFM2AudioModel`), and `mimi` (the Mimi codec, aliased from `proc.mimi`) — and to run a 5-iteration CUDA warmup of the Mimi streaming decoder. It exists so that the expensive `from_pretrained` loads and the codec's lazy CUDA-graph / kernel JIT happen **once, eagerly, before the first user turn**, not on the latency-critical path inside `chat_response`. It carries no model math of its own; it is pure orchestration and is **not** ported to Rust (the Rust tree builds its own singletons in `loader.rs`/`mic_chat.rs`).

## How it works
The module is a flat top-level script — there are no classes or functions; the side effects fire on `import liquid_audio.demo.model`.

1. **Processor load (`model.py:16`).** `proc = LFM2AudioProcessor.from_pretrained(HF_DIR).eval()` with `HF_DIR = "LiquidAI/LFM2.5-Audio-1.5B"` (`model.py:13`). `from_pretrained` resolves the snapshot dir (`utils.get_model_dir` → `huggingface_hub.snapshot_download`), builds the HF BPE text tokenizer (`AutoTokenizer`/`tokenizer.json`), the conformer mel front-end, the **Mimi** codec (built empty via `moshi.models.loaders.get_mimi(None, device="cuda")` then `load_state_dict(strict=True)` from `tokenizer-e351c8d8-checkpoint125.safetensors`), and — only if `audio_detokenizer/` is present in the snapshot — the LFM2 ISTFT detokenizer. `.eval()` puts every sub-module in inference mode (disables dropout, freezes BatchNorm running stats — load-bearing for the conformer conv module's BatchNorm). Processor default device is `"cuda"`, dtype `bfloat16`.

2. **Model load (`model.py:18`).** `lfm2_audio = LFM2AudioModel.from_pretrained(HF_DIR).eval()`. Internally: `json.load(config.json)` → `Lfm2Config(**cfg.lfm)` (the HF backbone config) → `accelerate.init_on_device("cuda")` meta-init → `set_attn_implementation("flash_attention_2" if module_exists("flash_attn") else "sdpa")` → `accelerate.load_checkpoint_in_model(model, dir)` streaming the bf16 `model.safetensors` into the meta tensors. So the **attention backend is chosen here at load time**: flash-attn-2 when the package is importable, else stock torch SDPA. Default `dtype=torch.bfloat16`, `device="cuda"`.

3. **Mimi alias (`model.py:20`).** `mimi = proc.mimi.eval()` — no second load; `mimi` is the *same* `MimiModel` object the processor already holds, re-exported as a top-level name so `chat.py` can drive streaming decode directly without going through `proc`. (Note: `proc.mimi` and `proc.audio_out` are two independent fields — Mimi is the demo/v1 streaming decode path; the LFM2 detok is the high-quality vocoder. The demo uses `mimi`.)

4. **CUDA warmup (`model.py:23-26`).** Under `with mimi.streaming(1), torch.no_grad():` it loops 5×, each iteration generating `x = torch.randint(2048, (1, 8, 1), device="cuda")` — a single Mimi frame of 8-codebook codes, each in `[0, 2048)` — and calling `mimi.decode(x)`. The `streaming(1)` context (batch=1) installs the codec's per-layer streaming state (SEANet causal-conv partial-frame buffers, transformer KV cache, resample partial-frame buffers); decoding 5 throwaway frames forces lazy init: the `CUDAGraphed` decode wrapper captures/replays its CUDA graph, cuDNN/cuBLAS pick conv+matmul algorithms, and any `torch.compile` paths in the codec transformer trace — all of which would otherwise add hundreds of ms of stall to the *first real* `mimi.decode` inside `chat_producer`. The random codes are discarded; only the side effect (warmed kernels/graphs) is kept. `device="cuda"` is hard-coded here, so this module **cannot run on a CPU-only host** — the `torch.randint(..., device="cuda")` raises immediately without a GPU.

There is no forward pass, normalization, attention, RoPE, conv, quantization, or sampling logic in this file — all of that lives in the components it instantiates. The only "mechanism" is load-ordering and the warmup loop's exercise of the Mimi streaming-decode state machine.

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| `from_pretrained` (proc) | `HF_DIR` str + snapshot weights (bf16 backbone, fp32 Mimi module) | `LFM2AudioProcessor` (tokenizer + mel + Mimi + optional detok) |
| `from_pretrained` (model) | `HF_DIR` str + `model.safetensors` bf16 | `LFM2AudioModel` (bf16 weights, attn=flash/sdpa, device cuda) |
| `proc.mimi` alias | — | `MimiModel` (same object) |
| warmup `torch.randint` | scalar `2048`, shape `(1, 8, 1)` | int64 codes in `[0, 2048)`, device cuda |
| `mimi.decode(x)` (warmup) | int64/int codes `(1, 8, 1)` | f32 waveform `(1, 1, 1920)` @ 24 kHz (discarded) |

No dtype promotions occur in this file. Promotions happen *inside* the loaded modules (f32-upcast RMSNorm/softmax in the backbone, f64 mel in the conformer front-end, bf16 backbone weights, int64 token ids, int/u32 codes into Mimi). The warmup codes are int64 from `torch.randint` (Mimi internally treats them as codebook indices, `< 2048`; note the real EOAudio sentinel `2048` is deliberately excluded by `randint(2048, …)` since EOAudio frames are skipped, not decoded — `chat.py:31`).

## Wiring
**Upstream (what this loads, weights flow in):**
- HF snapshot `LiquidAI/LFM2.5-Audio-1.5B` (`model.safetensors` bf16, `tokenizer-…-checkpoint125.safetensors` fp32, `tokenizer.json`, `config.json`) → resolved by [core_utils](../core/utils.md) `get_model_dir` (snapshot_download).
- Constructs [core_processor](../core/processor.md) (tokenizer + mel + Mimi dispatch + `ChatState`) as `proc`, bf16/cuda.
- Constructs [model_lfm2_audio](../model/lfm2_audio.md) (top model: prefill + `generate_interleaved` + text head + depthformer) as `lfm2_audio`, bf16 weights `(…,2048)`, cuda.
- Aliases [moshi_compression](../moshi/models/compression.md) `MimiModel` (held by the processor) as `mimi`.
- Warmup feeds int64 codes `(1,8,1)` into `mimi.decode` → f32 `(1,1,1920)` @24 kHz, discarded.

**Downstream (who imports these singletons):**
- [demo_chat](chat.md) — the **only** consumer: `from .model import lfm2_audio, mimi, proc` (`chat.py:11`). It drives `lfm2_audio.generate_interleaved(**chat, …)` (yielding text tokens `(1,)` int64 and audio frames `(8,)` int), `mimi.decode(t[None,:,None])` for `(8,)`→`(1,1,1920)` f32 streaming audio under `mimi.streaming(1)`, and `proc.text.decode(...)` to detokenize text. `proc` also seeds `ChatState(proc)`.

## Python ↔ Rust
No Rust counterpart — this demo glue is **not ported** (PYTHON_VS_RUST.md §4: "`liquid_audio/demo/**` … is not ported"). The Rust tree achieves the same singleton-construction differently and device-agnostically:

| Python (`demo/model.py`) | Rust equivalent |
|---|---|
| `LFM2AudioProcessor.from_pretrained(HF_DIR)` | `loader.rs::from_pretrained(dir, dtype, device)` → processor (`processor.rs`) |
| `LFM2AudioModel.from_pretrained(HF_DIR)` | `loader.rs::from_pretrained` → `LFM2AudioModel` (`model/lfm2_audio.rs`); `from_pretrained_hub` for repo-id |
| `proc.mimi` alias | `loader.rs::load_mimi` → `moshi::mimi::load(...)` wrapped `MimiDetokenizer` (`audio_out.rs`) |
| `mimi.streaming(1)` + 5× `mimi.decode` CUDA warmup | **no warmup loop** — candle has no CUDA-graph capture / `torch.compile`, so the JIT-warm rationale evaporates; cpal callbacks call `decode_step` directly (`mic_chat.rs`) |
| `device="cuda"`, `dtype=bf16` hard-coded | nothing hardcodes a device; loaders take `device: &Device`+`dtype: DType`, default `(Cpu, F32)`, Metal opt-in via `LFM_DEVICE=metal`→bf16 (PYTHON_VS_RUST.md §2.1) |
| attn = `flash_attention_2`/`sdpa` chosen at load | always eager `matmul + additive causal mask + softmax` (the sdpa/no-flash math; PYTHON_VS_RUST.md §2.2) |

The single deliberate divergence relevant here is **device-coupling**: as written, `demo/model.py` requires CUDA (the warmup's `device="cuda"` and the detok's `.cuda()` in `processor.py:151`); the Rust port boots the full 1.5B end-to-end on `Device::Cpu` with no warmup stage.

## Precision / gotchas
- **CUDA-only by construction.** `torch.randint(2048, (1,8,1), device="cuda")` (`model.py:25`) hard-pins CUDA; this module raises on a CPU/Metal-only host. The warmup is the *reason* the demo feels instant on the first turn (CUDA-graph/`torch.compile`/cuDNN-algo selection is paid here) and the reason it is non-portable.
- **`randint(2048, …)` excludes EOAudio.** Codes are sampled in `[0, 2048)`, so the warmup never feeds the `2048` EOAudio sentinel into `mimi.decode` — matching the real loop, which `continue`s past any frame containing `2048` rather than decoding it (`chat.py:31`). Decoding an EOAudio code would be an out-of-vocab index into the codec (audio vocab is 2049 = 2048 codes + EOAudio, but only the 2048 real codes are decodable).
- **`.eval()` is load-bearing, not cosmetic.** It freezes the conformer conv module's BatchNorm running stats and disables dropout; skipping it would corrupt understanding-path features. `mimi.eval()` (`model.py:20`) is redundant in effect (same object already eval-able) but mirrors upstream's belt-and-suspenders.
- **Shared-object aliasing.** `mimi` is `proc.mimi` — not an independent codec. Streaming state mutated via `mimi.streaming(1)` in `chat.py` is therefore *the processor's* Mimi state; there is exactly one codec instance, so warmup state and real-turn state share the same buffers (each `chat_producer` re-enters `mimi.streaming(1)`, which resets them).
- **No dtype gotchas in this file** — it introduces no casts. The bf16-weight / f32-CPU-floor / f64-mel / RMSNorm-bf16-order subtleties all live in the loaded components ([model_lfm2_backbone](../model/lfm2_backbone.md), [conformer_processor](../model/conformer/processor.md), [model_transformer](../model/transformer.md)), not here.
