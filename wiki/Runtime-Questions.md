<!-- topic: Overview -->
# LFM2-Audio — Archaeological Map (Python source → Rust port)

Direct, code-verified mapping of LFM2.5-Audio's runtime: the model token flow, the
codec, weight loading, "turn detection," and the concurrency model. Every claim
below is traced to `file:line` in the **vendored** Python
(`upstream-liquid-audio/src/liquid_audio/`) and the Rust port
(`liquid-audio-rs/src/`, `examples/`, plus the sibling `lfm-voice` crate `src/`).
Constants are quoted from `model/config.json`.

> Reading note: this model is **not** plain text LFM2. It is a mixed audio+text
> token model — a text **LFM2 backbone** (imported from `transformers`) wrapped with
> a FastConformer audio encoder, an audio **depthformer** head, and an audio
> detokenizer/codec. The pure-text `Lfm2Model` is only the inner backbone.

---

## 0. Token I/O flow (mic-in → wav-out)

```
                         ┌─────────────────────────── INPUT ASSEMBLY (prefill) ───────────────────────────┐
 mic wav (1,N) f32
   │ resample → 16 kHz                                              processor.py:233  (torchaudio.functional.resample)
   ▼
 (1,N') ── AudioToMel (FP32) ─► mel (1,128,F) ─► [0] ─► (128,F) bf16   conformer/processor.py:60 ; processor.py:236-238
                                  │   128 = mel bins (config.features)
                                  ▼
        Conformer encoder  ── 8× subsample ──►  (B, 512, T'),  T' = ceil(F/8)   conformer/encoder.py:491,688
                                  │   d_model = 512
                                  ▼
        audio_adapter  MLP 512→2048  ─────────►  audio_in_emb (ΣT', 2048)        lfm2_audio.py:87,353
                                                                                  modality slot = AUDIO_IN (=2)

 text ids (N) ── lfm.embed_tokens ───────────►  text_emb (N, 2048)               lfm2_audio.py:334   (TEXT =1)
 audio_out codes (8,L) + offsets ── SharedEmbedding.sum(0) ─► (L, 2048)          lfm2_audio.py:358   (AUDIO_OUT =3)
                                                  │   offsets = arange(8)*2049
   scatter rows by modality_flag  ──────────────►  in_emb (1, L, 2048)           lfm2_audio.py:366-370
                         └──────────────────────────────────┬──────────────────────────────────────────────┘
                                                            ▼
                            ┌──────────  LFM2 backbone (Lfm2Model) ──────────┐
                            │  16 hybrid layers (conv ×11 / full_attn ×5),   │   lfm2_audio.py:83,199-203
                            │  hidden 2048, bf16 weights                     │   config.lfm.*
                            └───────────────────┬───────────────┬───────────┘
                                                │ h = hidden[0,-1] (2048,)
                          ┌─────────────────────┘               └─────────────────────────┐
                          ▼ TEXT head (tied)                                   ▼ AUDIO head (depthformer)
        linear(h, embed_tokens.W) → (65536,)                 depth_linear 2048 → 8*1024  → (8,1024)   lfm2_audio.py:509
        argmax / temp+top-k  → 1 text token                  for i in 0..8:                            lfm2_audio.py:501-534
        lfm2_audio.py:208,273,483                              cur = depthformer_in[i] + prev_emb
                                                               depthformer (6× MHA-1024).forward_cached
                                                               depth_embeddings[i].get_logits → (2049,)
                                                               sample → code_i ∈ [0,2048]
                                                             → audio frame (8,)   (8 codebooks, EOAudio=2048)
                          └───────────────┬────────────────────────────┬──────────────────┘
                                          ▼  INTERLEAVE  (generate_interleaved, lfm2_audio.py:233-305)
                          ≤ 6 text tokens  ⇄  12 audio frames (×8 codebooks each), repeating
                          early-out: text token 130 (<|text_end|>) ends text; code 2048 (EOAudio) ends audio
                          stop: text token 7 (<|im_end|>/EOS)
                                          ▼  OUTPUT DECODE
        ┌─────────────────────────────────────────────┬───────────────────────────────────────────┐
        │ Mimi (v1 / demo, streaming)                  │ LFM2 detokenizer (LFM2.5, if weights present)│
        │ mimi.decode((1,8,1)) → wav (1,1920) @24 kHz  │ processor.decode((1,8,T)) → ISTFT → wav @24k │
        │ chat.py:34 ; audio_out.rs (moshi crate)      │ detokenizer.py:120-136 ; detokenizer.rs      │
        └─────────────────────────────────────────────┴───────────────────────────────────────────┘
```

**Load-bearing constants:** mel bins **128**; conformer subsample **8×**
(`mel2emb_len = ceil(F/8)`, `utils.py:15`); backbone width **2048**, **16** layers;
depthformer **6** layers × width **1024**; codebooks **8**; audio vocab **2049**
(2048 codes + EOAudio), fused audio-embed vocab `2049*8 = 16392`; interleave **6
text / 12 audio**; special tokens EOS=**7**, `<|audio_start|>`=**128**,
`<|text_end|>`=**130**, EOAudio code=**2048**; Mimi frame = **1920** samples @ 24 kHz.
`LFMModality`: TEXT=1, AUDIO_IN=2, AUDIO_OUT=3 (`utils.py:9`).

---

## Q1 — The Mimi codec (and what audio path actually uses it)

**Key fact:** for LFM2.5-Audio, Mimi is **not** the primary audio path. The
processor holds **two** independent audio-out fields, and **audio-in never uses
Mimi** — mic audio enters through the **conformer mel front-end**, not Mimi.

| Direction | What's used | Python | Rust |
|---|---|---|---|
| **audio-IN** (mic → model) | conformer **mel** (128-bin), *not* Mimi | `processor.py:226-250` `ChatState.add_audio` → `self.proc.audio` | `processor.rs` `add_audio`; mel in the same canonical `processor.rs` |
| **audio-OUT** (codes → wav), LFM2.5 | **LFM2 detokenizer** (ISTFT vocoder) | `detokenizer.py:120-136`; dispatched `processor.py:165` | `detokenizer.rs`; dispatch `processor.rs:158` |
| **audio-OUT**, v1/demo | **Mimi** streaming decode | `demo/chat.py:34` `mimi.decode(...)` | `audio_out.rs` `MimiDetokenizer` (moshi crate) |
| **audio-OUT encode** (data prep) | **Mimi** encode | `data/mapper.py:229` `processor.mimi.encode` | `audio_out.rs::encode` |

**Where the Mimi model lives / how it loads (Python):** built empty by
`moshi.models.loaders.get_mimi(None, device=...)` (`processor.py:113`;
`moshi/models/loaders.py:296-333` — SEANet enc/dec + 2× `ProjectedTransformer` +
`SplitResidualVectorQuantizer`), then weights loaded from the **local safetensors**
`tokenizer-e351c8d8-checkpoint125.safetensors` via
`safetensors.torch.load_file` + `load_state_dict(strict=True)`
(`processor.py:111-115`, filename hard-coded `processor.py:67`). `SAMPLE_RATE=24000`,
`FRAME_RATE=12.5`, `num_codebooks=8`.

**Rust:** `loader.rs:296-303` `load_mimi` → `moshi::mimi::load("tokenizer-…-checkpoint125.safetensors", Some(codebooks), device)`
(same filename), wrapped as `MimiDetokenizer { inner: RefCell<moshi::mimi::Mimi> }`
(`audio_out.rs:74-119`) with real streaming `decode_step`/`encode`/`reset_state`.
The two-field split (`mimi` vs `audio_out`) is preserved in `loader.rs:151-159`;
`processor.rs:158` dispatches `audio_out.or(mimi)`.

**CPU vs GPU / CUDA kernels:**
- **Python is GPU-coupled.** `from_pretrained(device="cuda")` default
  (`processor.py:61`); the LFM2 detok hard-codes `.cuda()` (`processor.py:151`);
  demo hard-codes `device="cuda"` (`demo/model.py:25`). The codec leans on
  `F.scaled_dot_product_attention` (`moshi/modules/transformer.py:562`) and
  `torch.compile`, which only engages on CUDA (`transformer.py:765` —
  `if x.device.type != 'cuda': no_compile()`). No `causal_conv1d`/`flash_attn`/`triton`
  import in the vendored codec — stock torch SDPA + `torch.compile`. **As shipped,
  the Python won't boot on a CPU-only host** (the detok `.cuda()` crashes).
- **Rust is device-agnostic.** Every loader takes `device: &Device` + `dtype: DType`
  (`loader.rs:296`), nothing hardcoded; SDPA is eager `matmul + mask + softmax` (the
  `sdpa`/no-flash math). Defaults `(Cpu, F32)`; Metal opt-in (`LFM_DEVICE=metal`,
  `Cargo.toml:77`). On CPU it uses F32 (candle has no CPU bf16 matmul).
- **Off-path note:** the `candle-flashfftconv` crate (bf16×2 `__nv_bfloat162` FFT
  kernels) is **not** wired into this model — `liquid-audio-rs` has zero references to
  it and no dep in `Cargo.lock`. The only FFTs on-path are the f32 mel STFT and the
  ISTFT, done as candle `Conv1d`/`matmul`/`conv_transpose1d`.

---

## Q2 — LFM2 weights: filename, loading, inference code, audio libs

**Weight files (`model/`):**

| File | Holds | Loaded by (Python) | dtype |
|---|---|---|---|
| `model.safetensors` (~2.94 GB) | the whole `LFM2AudioModel`: `lfm` (HF `Lfm2Model` backbone), `conformer`, `audio_adapter`, `audio_embedding`, `depthformer`, `depth_linear`, `depth_embeddings` | `accelerate.load_checkpoint_in_model(model, dir)` — `lfm2_audio.py:167` | bf16 (`config.lfm.torch_dtype`) |
| `tokenizer-e351c8d8-checkpoint125.safetensors` (~384 MB) | Kyutai **Mimi** codec | `safetensors.torch.load_file` + `load_state_dict` — `processor.py:111-115` | fp32 module |
| `tokenizer.json` | HF BPE text tokenizer | `AutoTokenizer.from_pretrained` — `processor.py:45` | n/a |
| `config.json` | all hyperparameters | `json.load` — `lfm2_audio.py:146`, `processor.py:64` | n/a |
| `audio_detokenizer/` (model.safetensors+config) | LFM2 ISTFT detok | `processor.py:153-157` (**only if present** — ABSENT in the local `model/`, so the local tree uses the Mimi path) | bf16 |

**Load trace (Python):** `LFM2AudioModel.from_pretrained` (`lfm2_audio.py:135-169`):
`get_model_dir` (`utils.py:40` → `huggingface_hub.snapshot_download`) → `json.load`
config → `Lfm2Config(**cfg.lfm)` (from `transformers`) →
`accelerate.init_on_device(device)` meta-init → `set_attn_implementation("flash_attention_2"
if module_exists("flash_attn") else "sdpa")` (`:162`) →
`accelerate.load_checkpoint_in_model(model, dir)` (`:167`). Default
`dtype=torch.bfloat16`, `device="cuda"`.

**Inference functions (`model/lfm2_audio.py`):** `_prefill` (:307), `generate_interleaved`
(:233, the demo path), `generate_sequential` (:171), text head inline `F.linear(h,
embed_tokens.weight)` (:208,273), `_sample_text_token` (:483), `_sample_audio_frame`
(:501, the depthformer loop), depthformer `RawLMBackbone` (:121).

**Audio libraries — definitive:**

| Library | Present? | Where / for what |
|---|---|---|
| **torchaudio** | yes — **resample only** | `processor.py:233`, `data/mapper.py:193,227` (`functional.resample`). The conformer's `use_torchaudio` flag defaults **False** → manual STFT. |
| **FFMPEG / PyAV / torchaudio.io** | **NO** | zero hits anywhere. |
| **soundfile** | yes | `data/mapper.py:237` `soundfile.read(dtype="float32")` — the only file-decode site. |
| **librosa** | yes — **mel filterbank only** | `conformer/processor.py:338` `librosa.filters.mel(norm="slaney")`. |
| **sentencepiece** | present but **off-path** | only the unused `moshi/` TTS stack; LFM2-Audio text = `AutoTokenizer`/`tokenizer.json`. |

**Rust mapping:** `loader.rs::from_pretrained` (:102) reads `config.json` (serde) →
`VarBuilder::from_mmaped_safetensors(dir, dtype, device)` (:136) over every
`.safetensors`. **No `accelerate`** — candle mmaps the tensors directly (CPU+BF16 is
rejected up front, :103). Backbone = `lfm2_hf.rs` (adapted from candle-transformers
`lfm2.rs` onto plain `candle_nn`, because candle 0.9 has only `quantized_lfm2`).
`utils.rs::get_model_dir` (:56) mirrors `snapshot_download` via the **`hf-hub`** crate.
`torchaudio.resample → resample.rs` (windowed-sinc, hand-port). `soundfile →
symphonia` (`data/mapper.rs:404`). `AutoTokenizer → tokenizers` crate
(`processor.rs:114`). **No ffmpeg, no PyAV, no sentencepiece** in either language on
the active path.

---

## Q3 — Turn detection

**There is NO turn-detection / VAD / endpointing model anywhere** — no weights, no
dtype, no device, nothing to load. Verified by grep (`silero|semantic_vad|webrtcvad|
onnxruntime|vad_model` → zero hits in the vendored Python and the Rust). The model
itself does **not** decide "the user stopped"; that is always an external endpointer.

| Path | How turn-taking actually works | Source |
|---|---|---|
| **Python real-time demo** | `fastrtc.ReplyOnPause` — a **third-party library VAD**; on pause it fires `chat_response` with the captured utterance. `can_interrupt=False` ⇒ no barge-in. Its VAD weights live in the (uninstalled) `fastrtc` package, not this repo. | `demo/chat.py:7,122-128` |
| **Python moshi server** | Fixed Mimi frame cadence + streaming inner-monologue; turn ends when the model emits the text tokenizer **EOS** (`text_tokenizer.eos_id()`). No VAD. | `moshi/run_inference.py:138-182` |
| **Model's only contribution** | chat-template tokens `<|im_start|>{role}` / `<|im_end|>` — plain text, not detection. | `processor.py:252-256` |
| **Rust** (`mic_chat.rs`, `lfm-voice` `src/audio.rs`) | hand-rolled **RMS energy VAD**: start when a 200 ms window crosses `LFM_VAD_THRESHOLD` (**default 0.012**), end after **800 ms** of silence (`audio.rs`: 1.0 s). That `last_voice.elapsed() >= silence` break **is** the entire endpointer. | `mic_chat.rs:103-134`; `src/audio.rs:43-124` |

So the "turn detector" in project memory = the few-line RMS+silence break, not a
model. Python's `fastrtc.ReplyOnPause` and the moshi server are **not ported**.

---

## Q4 — Concurrency / full-duplex

**The model is a synchronous streaming generator; async exists only at the transport
— verified.** `generate_interleaved` returns `Generator[Tensor]` and `yield`s
(`lfm2_audio.py:247,279,304`); no `async`/`await` in the model. Three different
concurrency shells wrap it:

| Path | Mechanism | Full-duplex? | Source |
|---|---|---|---|
| **Python demo** | `threading.Thread` runs `chat_producer` → `queue.Queue` → main-thread generator consumes; `mimi.streaming(1)` decodes audio frames. | **No** — turn-based (`ReplyOnPause`, `can_interrupt=False`). | `demo/chat.py:1-2,64-66,72-89` |
| **Python moshi server/client** | `asyncio` + `aiohttp` websockets; server runs 3 coroutines `recv_loop`/`opus_loop`/`send_loop` via `asyncio.gather` under an `asyncio.Lock`; client adds PortAudio (`sounddevice`) callback threads + `queue.Queue` for playback. Uses `lm_gen.step()`, not `generate_interleaved`. | **Yes** — true simultaneous mic+speaker. | `moshi/server.py:78-171`, `moshi/client.py:35-141` |
| **Rust** (`mic_chat.rs`) | **No async, no tokio, no spawned threads, no channels.** Only `cpal`'s own callback threads + `Arc<Mutex<…>>` buffers. Mic capture fills `Arc<Mutex<Vec<f32>>>`; `generate_interleaved(&chat, &params, \|tok\| …)` runs **synchronously on main**; the audio callback decodes (`mimi.decode_step`) and pushes to an `Arc<Mutex<VecDeque<f32>>>` ring that the cpal **output** callback drains. | **No** — `drop(stream)` stops the mic *before* generating (`:135`), then spin-waits for the ring to drain (`:283`). | `mic_chat.rs:77-180,245-289` |

**Cross-thread carriers:** Python demo `queue.Queue` (model→main); moshi
`OpusStreamWriter`/`Reader` + websocket frames + `queue.Queue`. Rust: `Arc<Mutex<Vec>>`
(mic→main) and `Arc<Mutex<VecDeque>>` ring (model→speaker) — cpal callbacks only.

**Divergence:** the Rust ports only the **turn-based demo shape** (half-duplex). The
genuinely simultaneous **moshi async/websocket full-duplex path is unported**, so no
async runtime is even a dependency. `PORT_STATUS.md`'s "demo thread+queue → std::thread
+ channel" overstates it — the real Rust mechanism is "main-thread synchronous
callback + `Arc<Mutex>` ring + cpal's native callback threads," with no worker thread
or channel.

---

## Consolidated archaeological mapping

| Concern | Python (file:line / symbol / lib) | Rust (file:line / symbol / crate) |
|---|---|---|
| Backbone | `transformers.Lfm2Model` (`lfm2_audio.py:14,83`) | `model/lfm2_hf.rs` (candle_nn) |
| Depthformer | `RawLMBackbone` (`lfm2_audio.py:121`) | `model/transformer.rs` `RawLmBackbone` |
| Conformer enc | `model/conformer/encoder.py` | `model/conformer/encoder.rs` |
| Mel front-end | NeMo `FilterbankFeatures` + `librosa.filters.mel` (`conformer/processor.py`) | `src/processor.rs` (native slaney mel, Conv1d DFT) |
| Weight load | `accelerate.load_checkpoint_in_model` (`lfm2_audio.py:167`) | `VarBuilder::from_mmaped_safetensors` (`loader.rs:136`) |
| HF download | `huggingface_hub.snapshot_download` (`utils.py:48`) | `hf-hub` crate (`utils.rs:72`) |
| Text tokenizer | `transformers.AutoTokenizer` (`processor.py:45`) | `tokenizers` crate (`processor.rs:114`) |
| Resample | `torchaudio.functional.resample` (`processor.py:233`) | `resample.rs` (windowed-sinc) |
| Audio decode | `soundfile.read` (`data/mapper.py:237`) | `symphonia` (`data/mapper.rs:404`) |
| Mimi codec | vendored `liquid_audio/moshi` `MimiModel` (`processor.py:113`) | `moshi` crate `mimi::Mimi` (`audio_out.rs`, `loader.rs:296`) |
| LFM2 detok | `detokenizer.py` (`torch.fft.irfft`+`F.fold`) | `detokenizer.rs` (inverse-DFT matmul + `conv_transpose1d`) |
| Sampling | `_sample_*` + `torch.multinomial` (`lfm2_audio.py:483,501`) | `candle_transformers::generation::LogitsProcessor` |
| Turn detect | `fastrtc.ReplyOnPause` (`demo/chat.py:123`) / EOS (moshi) | RMS energy VAD (`mic_chat.rs:103`, `src/audio.rs:45`) |
| Concurrency | `Thread`+`queue.Queue` (demo) / `asyncio`+`aiohttp` (moshi) | `cpal` callbacks + `Arc<Mutex>`; **no async** |
| Generate IO | sync generator `yield` (`lfm2_audio.py:247`) | sync callback `FnMut(GenToken)` (`lfm2_audio.rs:853`) |

## Honest gaps & divergences
1. **Half-duplex vs full-duplex.** Rust ports the turn-based demo only; moshi's async
   websocket full-duplex (simultaneous listen+speak) is unported.
2. **Device coupling.** Python is CUDA-pinned (won't boot CPU-only); Rust is
   device-agnostic and CPU-default.
3. **Audio-out backend in the local tree.** `model/` ships no `audio_detokenizer/`, so
   both languages fall back to the **Mimi** decode path there; the LFM2 ISTFT detok runs
   only against the full HF snapshot.
4. **No external endpointer ported.** `fastrtc.ReplyOnPause` (the Python demo's actual
   VAD) is replaced by a hand-rolled RMS+silence break.
5. **Off-path artifacts.** `candle-flashfftconv` (bf16×2 FFT kernels) and the moshi
   server/client are present in the tree but not wired into LFM2-Audio inference.
