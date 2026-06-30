# EmberHarmony Voice — Stack Architecture

> Status legend used throughout: ✅ built & proven this stack · 🔧 in progress / partially
> wired · ◻️ designed, not yet built · ⚠️ known gap / open question.
>
> This document describes the **native Rust voice stack**: the faithful in‑tree port of
> Liquid AI's **LFM2.5‑Audio** model (the `liquid-audio` crate), the generation and
> context machinery built on top of it, the **orchestration layer** that lets a 1.5B speech
> model actually *do work* by delegating to a capable model, the **audio I/O**, and the
> **Tauri integration** that fuses all of it into the desktop app on real OS threads.
>
> It is the companion to `FRONTEND_DESIGN.md` (the webview/UX side) and supersedes the
> scattered notes in `crates/liquid-audio/{ARCHAEOLOGY,PORT_STATUS,PYTHON_VS_RUST,THREADING_PARITY}.md`.

---

## Table of contents

```
 0.  How to read this document
 1.  Philosophy — the inversion of the brain
 2.  The 10,000‑foot view (master diagram)
 3.  Layer 0 — Why native Rust (the migration)
 4.  Layer 1 — The model engine (the liquid-audio crate)
       4.1  What LFM2.5‑Audio actually is
       4.2  Continuous audio‑in:  mel → ConvSubsampling → Conformer → adapter
       4.3  The LFM2 backbone (hybrid short‑conv + GQA)
       4.4  The two output heads:  text head + Depthformer
       4.5  Mimi — the discrete audio codec
       4.6  Full forward diagram
       4.7  The faithful‑port procedure
       4.8  Numerics: candle 0.9.2, bf16 Metal, f32 CPU parity
       4.9  The conv kernels — candle-flashfftconv
 5.  Layer 2 — Generation (the interleaved modality machine)
 6.  Layer 3 — Context is prior thoughts (the heart of the design)
       6.1  The principle
       6.2  ChatState — the five fields
       6.3  Continuous‑in vs discrete‑out (image‑embedding vs RVQ tokens)
       6.4  audio_embedding (context) vs Mimi (sound) — two sinks
       6.5  The prefill scatter
       6.6  Multi‑turn: append / from_parts / persistent conv
       6.7  I/O‑independence (the barge‑in fix)
 7.  Layer 4 — The KV cache and the memory problem
 8.  Layer 5 — The orchestration layer (voice‑as‑agent + delegation)
 9.  Layer 6 — Audio I/O (cpal)
10.  Layer 7 — The Tauri integration (pipeline + bridge)
11.  Threading model
12.  Phasing — turn parity now, full‑duplex (Moshi) later
13.  File & module map
14.  Verification & proofs
15.  Open questions / next moves
16.  Glossary
```

---

## 0. How to read this document

The stack is seven layers. Each layer below is independently comprehensible, but they stack:

```
   Layer 7   Tauri integration   (desktop commands, Channel<VoiceEvent>, State)
   Layer 6   Audio I/O           (cpal mic capture, speaker playback, VAD)
   Layer 5   Orchestration       (LFM front + DELEGATE marker + capable‑model subagent)
   Layer 4   KV cache / memory   (preallocated KvCache, inter‑turn persistence)
   Layer 3   Context             (ChatState; prior thoughts; multi‑turn)
   Layer 2   Generation          (interleaved text+audio; the modality machine)
   Layer 1   Model engine        (mel→Conformer→LFM2→text head+Depthformer→Mimi)
   Layer 0   Native runtime       (candle + Metal, no LiveKit, no Node, no Python)
```

Read top‑down for "what does the app call," bottom‑up for "how does the model work." The
heart of the whole thing is **Layer 3** — everything else exists to feed, generate, or speak
the model's *prior thoughts*.

---

## 1. Philosophy — the inversion of the brain

The earlier voice architecture (on the LiveKit branch) was **dumb I/O bridged to a big brain**:

```
   mic ─► WebRTC ─► LiveKit Cloud SFU ─► agent worker (Node) ─► SessionLLM "brain"
                                                                      │
   speaker ◄─ WebRTC ◄─ LiveKit Cloud SFU ◄─ TTS ◄────────────────────┘
```

Every utterance from a *local* user made **two** round‑trips to a cloud SFU, plus a Node
worker, plus token/room/dispatch plumbing, plus a bundled voice runtime that had to be code‑
signed. The microphone was a dumb pipe; all intelligence lived elsewhere.

**We inverted it.** The new design runs a **small, local, conversational model as the agent
itself**, and treats "doing hard work" as a *delegated tool*, not the default path:

```
   mic ─► LFM2.5‑Audio (LOCAL, native Rust, candle+Metal)
            │  it IS the agent: ears, voice, small talk, and judgement about when to hand off
            ├─ ordinary turn        ─► speak its own reply            (small talk, quick answers)
            └─ "DELEGATE: <task>"   ─► capable model / EmberHarmony agent does the work
                                          └─► LFM speaks the result back
```

Two facts force this shape:

1. **LFM2.5‑Audio is ~1.5B and cannot code, reason deeply, or touch the system.** It is a
   *speech* model. It converses; it does not engineer. Pretending otherwise produces confident
   nonsense. So real work **must** be delegated to a capable model.
2. **LFM2.5‑Audio has no native function calling.** (Liquid ships tool calling only in separate
   text models, e.g. `LFM2‑1.2B‑Tool`.) So delegation cannot use a tool‑call API on the audio
   model. Instead it rides a **one‑line text convention** the small model *can* reliably follow:
   it emits `DELEGATE: <task>` on its text channel, and the orchestrator routes that to the
   engineer.

The consequence is a **two‑tier agent**: a fast local *interface tier* (the voice) and a
capable *worker tier* (the delegate). The interface tier is always‑on and cheap; the worker
tier is invoked only when there is genuine work. This is the whole thesis of the stack.

> The standalone `lfm-voice` binary that prototyped this (in `experiments/lfm2-audio-voice/`)
> is *scaffolding*. The **layer** — voice‑front + delegation routing + capable‑model subagent —
> is the architecture, and it migrates into the desktop build (Layer 5 / Layer 7).

---

## 2. The 10,000‑foot view (master diagram)

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │  DESKTOP APP (Tauri)                                                                        │
 │                                                                                            │
 │   webview (SolidJS)  ── voice_* commands ──►  src-tauri/src/voice/   (Rust, real threads)  │
 │       ▲   │                                        │                                       │
 │       │   │ Channel<VoiceEvent>                    │                                       │
 │       │   ▼                                        ▼                                       │
 │   transcript / level / audio‑clip          ┌──────────────────────────────────────────┐   │
 │                                            │  Orchestration (Layer 5)                  │   │
 │                                            │   converse → watch text channel →         │   │
 │                                            │   route { speak | DELEGATE → engineer }    │   │
 │                                            └───────────┬───────────────┬──────────────┘   │
 │                                                        │ speech        │ DELEGATE: task    │
 │                                                        ▼               ▼                   │
 │   ┌──────────────────────────────────────────────┐    ┌────────────────────────────────┐  │
 │   │  liquid-audio engine (Layers 1‑4)            │    │  Engineer (capable model)       │  │
 │   │   RealtimePipeline → Lfm2VoiceEngine          │    │   GLM‑5.1 / user model /         │  │
 │   │     ChatState (prior thoughts)                │    │   EmberHarmony's own agent       │  │
 │   │     generate_interleaved (text + audio)       │    │   tool loop (bash, files, …)     │  │
 │   │     KvCache · Depthformer · Mimi              │    └────────────────────────────────┘  │
 │   └───────────────▲───────────────┬──────────────┘                                         │
 │                   │ mic PCM       │ reply PCM                                               │
 │            ┌──────┴───────────────▼──────┐                                                  │
 │            │  Audio I/O (Layer 6, cpal)  │  mic capture (VAD) · speaker playback           │
 │            └─────────────────────────────┘                                                  │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
        no LiveKit · no Node worker · no cloud SFU round‑trip · no Python
```

Everything inside the dashed box is **in‑process Rust on real OS threads**. The only network
hop is the *optional* delegate call to a capable model (and even that can be a local model).

---

## 3. Layer 0 — Why native Rust (the migration)

**Decision:** rebuild voice as native Rust inside `packages/desktop/src-tauri`, deleting the
LiveKit + `@livekit/agents` Node worker. Rationale:

- Removes the cloud‑SFU **double round‑trip** for a *local* assistant (mic→SFU→agent→SFU→spk).
- Removes the Node worker (zombies/IPC/orphans), the bundled+codesigned voice runtime, and the
  token/room/dispatch surface.
- Real OS threads are the right home for realtime audio; Tauri's Rust side has them.

**Two migrations happened, and I initially only did the first:**

| # | From | To | Status |
|---|------|----|--------|
| 1 | `experiments/lfm2-audio-voice/liquid-audio-rs/` (the model crate) | `packages/desktop/src-tauri/crates/liquid-audio/` | ✅ done — built into the desktop app (`Cargo.toml`: `liquid-audio = { path = "crates/liquid-audio", features = ["metal"] }`) |
| 2 | `experiments/lfm2-audio-voice/src/` (the **orchestration layer**: `main.rs` routing, `glm.rs` subagent) | `packages/desktop/src-tauri/src/voice/` | 🔧 **not yet** — this is the layer wrongly left behind; Layer 5 below is its plan |

Desktop‑only is acceptable for now: desktop is the sole test target and we are not supporting
multi‑backend / cross‑platform voice yet. The pure‑web deployment can't run Rust threads; if
voice is ever needed there it's a separate path.

A third, *internal* migration is implied by the native engine: the prototype's `lfm.rs` shelled
out to an external `llama-liquid-audio-cli` (llama.cpp GGUF). The native `liquid-audio` engine
**obsoletes that** — ASR/TTS/interleaved all run in‑process on candle+Metal, no subprocess, no
GGUF. The orchestration around it stays; only the runtime swaps.

---

## 4. Layer 1 — The model engine (the `liquid-audio` crate)

### 4.1 What LFM2.5‑Audio actually is

It is **not** a text LLM with a bolted‑on codec. It is a multimodal *interleaved‑generation*
model with three distinct modalities flowing through one shared backbone:

- **continuous audio‑in** — raw waveform → mel → Conformer → adapter → backbone embeddings
  (an *analog* representation, like a base‑64 image embedded in context);
- **text** — ordinary token embeddings;
- **discrete audio‑out** — 8 residual‑vector‑quantized (RVQ) codes per 80 ms frame, produced by
  a *second* autoregressive transformer (the Depthformer) and embedded by the model's own
  `audio_embedding` table.

These three share **one LFM2 backbone**. Two heads read the backbone's hidden state: a **text
head** and the **Depthformer**. The model weaves text and audio into a single interleaved
stream. A separate codec (**Mimi**, from the `moshi` crate) turns audio‑out codes into a
playable waveform — but Mimi is a *playback* device, not part of the model's reasoning (see
§6.4).

```
   modality        representation            produced by            consumed by
   ───────────     ───────────────────       ───────────────        ──────────────
   audio‑in        mel → Conformer embeds    user's microphone      backbone (read)
   text            token embeddings          text head (output)     backbone (read+write)
   audio‑out       8 RVQ codes / frame       Depthformer (output)   backbone (read+write) + Mimi (playback)
```

### 4.2 Continuous audio‑in:  mel → ConvSubsampling → Conformer → adapter

The audio‑in front end is a faithful port of NeMo's FastConformer preprocessing + a Conformer
encoder + a small MLP "audio adapter" that projects encoder features into the backbone's
embedding space.

```
   waveform (1, L) @ 16 kHz
        │  FilterbankFeatures  (processor.rs)  — torch.stft as a strided DFT‑basis conv1d,
        │     Hann window folded into the kernel, slaney mel filterbank, log, 128 mel bins
        ▼
   mel  (128, T)                         T = get_seq_len(L)   (valid centered frames)
        │  ConvSubsampling 8×            (model/conformer/subsampling.rs)
        ▼
   sub  (d, T/8)
        │  ConformerEncoder             (model/conformer/encoder.rs)  — relative‑pos MHA,
        │     conv module, FFN×2, streaming‑capable att‑context sizing
        ▼
   enc  (T', d)
        │  audio_adapter  (MLP)         (model/mlp.rs)
        ▼
   audio‑in embeddings  (T', hidden)    ← scattered into the backbone sequence at AUDIO_IN positions
```

Key invariant (PR thread T20, ⚠️ to fix): the number of audio‑in tokens must be
`mel2emb_len(get_seq_len(L))`, the count of **valid** centered frames — not the raw returned mel
width, which the centered STFT pads with zero columns. The data mapper narrows to `get_seq_len`
before storing; `ChatState::add_audio_16k` currently uses the raw width and must do the same.

### 4.3 The LFM2 backbone (hybrid short‑conv + GQA)

`model/lfm2_hf.rs` is a faithful adaptation of candle‑transformers' `lfm2.rs` (itself a port of
HF `modeling_lfm2.py`) onto plain `candle_nn` 0.9.2. The backbone is **hybrid**: most layers are
GQA attention; some are a **short causal convolution** ("ShortConv") instead of attention. Each
layer is one of:

```
   Attention layer                         ShortConv layer
   ───────────────                         ───────────────
   q/k/v proj → q_norm/k_norm (RMS)        in_proj → (B·gate · x)  (gated input)
   RoPE (NeoX half‑split, differentiable)  causal depthwise conv1d, l_cache = 3
   KvCache.append  (preallocated)          conv‑state cache (small, cat‑based)
   GQA repeat_kv → scaled‑dot attn (f32)   out_proj
   o_proj
```

Per‑layer state lives in a single `Cache` struct (`lfm2_hf.rs`):

```
   struct Cache {
     use_kv_cache: bool,
     kvs:          Vec<KvCache>,           // ✅ candle_nn::kv_cache::KvCache (preallocated; §7)
     conv_states:  Vec<Option<Tensor>>,    // ShortConv state (l_cache=3, tiny)
     masks:        HashMap<(usize,usize), Tensor>,  // memoized causal masks (shape‑keyed)
     cos, sin:     Tensor,                 // RoPE tables
   }
```

The **text head** is weight‑tied to `embed_tokens` (the text logits are `hidden · embedᵀ`).

### 4.4 The two output heads:  text head + Depthformer

At each generated position the backbone produces a hidden vector `h_last`. **Both** heads read
the *same* `h_last`:

```
                          ┌────────────► text head:  logits = linear(h_last, embed_tokensᵀ)
   backbone ─► h_last ────┤                          → sample text token (greedy)
                          └────────────► Depthformer:  a 2nd autoregressive transformer that,
                                          conditioned on h_last, emits the 8 RVQ codebook codes
                                          for this 80 ms audio frame, one codebook at a time
                                          (its own tiny KV cache = ConcatKvCache, §7)
```

This is why "the model reasons over both at once" is literally true: the heads diverge only at
the last step; the *reasoning* (the backbone state) is shared and has attended over the entire
interleaved history of text **and** audio.

The Depthformer's audio vocabulary is `AUDIO_VOCAB_SIZE = 2048 + 1` (the `+1` is the **EOAudio**
terminator, code 2048) per codebook, with `codebooks = 8`. Codes from codebook *c* are shifted
into a shared embedding table by `codebook_offsets[c] = c · AUDIO_VOCAB_SIZE`.

### 4.5 Mimi — the discrete audio codec

Mimi (Kyutai's codec, reused from the `moshi` crate) turns the 8‑code frames into a 24 kHz
waveform. It ships as `tokenizer-…checkpoint125.safetensors` in the model dir and is loaded
**independently** of the model weights. Two crucial properties:

- It has a **true streaming `decode_step`** that keeps codec state *across* frames, for gapless
  realtime playback — exactly the Python demo's `with mimi.streaming(1): mimi.decode(frame)`
  loop (`chat.py`).
- It is **playback only** (§6.4). The model never reads Mimi's waveform; it reads the *codes* via
  its own `audio_embedding`. Mimi is a sink, never a source of context.

> **Hard rule:** Mimi always ships and is *required* for streaming audio‑out. The streaming
> path uses `proc.mimi()` directly and hard‑errors if absent — **no fallback** to the LFM2
> detokenizer's degenerate one‑shot decode. A fallback would mask a broken build behind choppy
> audio. (See `no-fallbacks-mimi-required`.)

### 4.6 Full forward diagram

```
   ┌─────────────────────────── INPUTS (one interleaved sequence) ───────────────────────────┐
   │   text tokens          audio‑in (mel→Conformer→adapter)        audio‑out (RVQ codes)     │
   │       │                          │                                     │                 │
   │   embed_tokens            audio‑in embeddings              audio_embedding(codes+offsets) │
   │       │                          │                                .sum(over 8 codebooks)  │
   │       └──────────────┬───────────┴──────────────┬──────────────────────┘                 │
   │                      ▼  scatter by modality_flag ▼                                        │
   │              in_emb  (1, L, hidden)   ← the woven multimodal sequence                     │
   └──────────────────────────────────────┬───────────────────────────────────────────────────┘
                                           ▼
                          ┌────────────────────────────────────┐
                          │  LFM2 backbone (hybrid conv + GQA)  │   per‑layer KvCache / conv state
                          └───────────────┬────────────────────┘
                                          ▼  h_last (1, hidden)
                        ┌─────────────────┴──────────────────┐
                        ▼                                     ▼
                  text head                             Depthformer (8 RVQ codes)
                        │                                     │
                  GenToken::Text(u32)                  GenToken::Audio(Vec<u32>)
                        │                                     │
                        │                                     ├─► append to context (audio_embedding)
                        │                                     └─► Mimi.decode_step → PCM → speaker
                        └──────────── interleaved stream ─────┘
```

### 4.7 The faithful‑port procedure

The port follows a strict procedure (memory `lfm2-rust-port-faithfulness-guardrails`), *no*
resemblance‑guessing:

```
   1. Read, line for line, what the Python does AND what its libraries do.
   2. Does candle (0.9.2, our pin) already have the EXACT thing?  → use it.
   3. Does a newer candle (0.10.2) have it and we need it?        → vendor that code in,
                                                                     rewire to 0.9.2.
   4. Is it only *similar* but not exact?                         → write our own; never
                                                                     bring code that merely
                                                                     resembles it.
```

Concrete applications of the rule in this stack:

- `candle_ext/transformers_utils.rs` — vendored `build_causal_mask` + `repeat_kv` from
  candle‑transformers (exact), rewired to 0.9.2.
- `candle_ext/kv_cache.rs` — vendored `ConcatKvCache` from candle‑nn 0.10.2 (the cat‑based cache,
  exact match to Python's `LayerKVCache.update`), used by the Depthformer.
- `candle_ext/tensor_ext.rs` — `to_vec4` written ourselves (candle 0.10.2 has no exact match;
  it's `narrow + to_vec3` per slice, the same math as `torch/numpy .tolist()`).
- Backbone KV cache — uses candle‑nn 0.9.2's **own** preallocated `KvCache` (it was there all
  along; §7).

Faithfulness is gated **behaviorally**, not by structural similarity: ported tests + golden
differential dumps (`parity/dump_*.py` → `parity/golden/*.safetensors`, gitignored, regenerable)
+ real end‑to‑end runs. Structural/AST distance is a shadow and is cheatable; we don't use it.

### 4.8 Numerics: safetensor dtype, bf16 Metal/CPU, f32 local math

- **candle 0.9.2** is the pin (transitively via `moshi`). Everything is built against it.
- **Persistent model weights keep the floating dtype stored in the safetensors headers.** The
  Rust loader does not accept a caller-selected model dtype and does not use config metadata to
  upcast BF16 weights.
- **bf16 on Metal** is the deployed path; the model ships bf16 and Metal runs it in real time.
- **bf16 on CPU** uses the in-tree NEON `BFMMLA` bridge for 2-D linear/logit matmuls when the
  CPU exposes FEAT_BF16. If that CPU feature is missing, CPU LFM2 inference fails clearly rather
  than loading a second F32 model copy.
- **F32 remains intentional local math**, not persistent weight storage: audio PCM/front-end
  buffers, logits/sampling/loss calculations, attention-score/value matmuls, and BF16 matmul
  accumulation use F32 where the canonical path requires it.
- `LFM_DEVICE=metal` selects Apple GPU execution; default/`cpu` uses CPU BF16 through the
  NEON bridge. Audio sampling uses `temperature=1.0, top_k=4` (greedy audio is degenerate);
  text is greedy.

The NEON BF16 GEMM is `src/bf16_gemm.rs` plus `csrc/bf16_gemm.c`, compiled by `build.rs` with
`-march=armv8.2-a+bf16`. `model::linear` routes BF16 CPU linears/logits through that bridge and
keeps the 4-D attention matmuls on the explicit F32 accumulation path.

### 4.9 The conv kernels — `candle-flashfftconv`

The convolution operators that ML stacks normally gate behind custom CUDA live in their own
crate, `experiments/lfm2-audio-voice/candle-flashfftconv/` — candle `CustomOp`s that run on
**CPU** (faithful reference) **and Metal** (real fused kernels), no CUDA, no torch. Two
families:

- **Short conv** — `depthwise_conv1d` (the LFM2 short‑filter / `conv_L_cache` path). metal == cpu, 5.96e‑8.
- **Long conv** — the FlashFFTConv Monarch FFT path, ported CUDA → MLX‑oracle → candle. The
  headline is **`monarch_conv_fused`**: the full `IFFT(FFT(u) ⊙ k_f)` collapsed into **one**
  tiled `simdgroup_matrix` (Apple tensor‑core) dispatch — every sub‑DFT an 8×8 fp32‑accumulate
  GEMM, the `[N,L]` intermediate resident in threadgroup memory, the `×k_f` multiply fused
  in‑kernel, edge tiles zero‑filled so any `N,L` works with no caller padding. Drop‑in for the
  un‑fused `monarch_conv`. Verified `metal == monarch_conv` (1.5e‑8, incl. non‑mult‑of‑8 dims)
  and `== circular convolution` (9.7e‑8); 30/30 tests green.

Pipelines are compiled **once process‑wide** and shared across threads (the compiled kernel vs.
per‑dispatch instances; `warmup()` moves the one compile to engine init). Like the bf16‑CPU
kernel above, the fused tensor‑core conv is **built and verified**; wiring it into the LFM2
backbone call site is the remaining step (§15).

→ Full kernel design, dataflow diagram, dispatch contract, edge‑tile handling, the global
pipeline cache, and the precision regimes (fp32 / bf16‑faithful / double‑double): see
[`candle-flashfftconv/ARCHITECTURE.md`](../../../../experiments/lfm2-audio-voice/candle-flashfftconv/ARCHITECTURE.md).

---

## 5. Layer 2 — Generation (the interleaved modality machine)

`generate_interleaved` weaves text and audio into one stream by **time‑multiplexing** the two
heads. The model config sets the cadence: `interleaved_n_text = 6`, `interleaved_n_audio = 12`.

```
   current = TEXT, modality_left = 6
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ loop (≤ max_new_tokens):                                                       │
   │   h_last = backbone(in_emb, cache)         # one forward, KV cache appended    │
   │                                                                                │
   │   if current == TEXT:                                                          │
   │       tok = sample_text(text_head(h_last)) # greedy                            │
   │       if tok == 7  (<|im_end|>):  break     # end of turn                      │
   │       emit Text(tok)                                                            │
   │       if tok == 130 (<|text_end|>): text_done = true                           │
   │       if --modality_left == 0 or text_done: current = AUDIO; modality_left = 12 │
   │       in_emb = embed_tokens(tok)                                                │
   │                                                                                │
   │   elif current == AUDIO:                                                        │
   │       frame = sample_audio(Depthformer(h_last))   # 8 codes, temp 1.0 / topk 4  │
   │       if --modality_left == 0 and not text_done: current = TEXT; modality_left=6│
   │       if frame[0] == 2048:  frame = [2048;8]; current = TEXT   # EOAudio        │
   │       emit Audio(frame)                                                          │
   │       in_emb = audio_embedding(frame + offsets).sum(0)   # feed own audio back   │
   └──────────────────────────────────────────────────────────────────────────────┘
```

```
   modality timeline (one turn):
     TEXT TEXT TEXT TEXT TEXT TEXT │ AUD AUD AUD … (×12) │ TEXT … │ AUD …  │ … │ <|im_end|>
     └──── 6 text tokens ────────┘ └─── 12 audio frames ─┘
     (text may finish early via <|text_end|>, then audio‑only to EOAudio / im_end)
```

The stream is delivered to callers as a `GenToken`:

```
   enum GenToken { Text(u32), Audio(Vec<u32>) }    // Audio = the 8 RVQ codes of one frame
```

Two variants exist: `generate_interleaved` (text‑first conversational) and `generate_sequential`
(used by the demo's ASR/TTS modes — text fully, then audio). Each has a `*_cancellable` form
that polls an `AtomicBool` every step and breaks promptly on **barge‑in**, instead of running to
`max_new_tokens` and tying up the P‑cores.

The **EOAudio** frame (all‑2048, code 2048 in codebook 0) is *yielded* to the caller before the
modality flips back to text — both Python (`lfm2_audio.py`) and our Rust (`lfm2_audio.rs`) emit
it. It matters for context (§6.6): the full audio‑out *including* the EOAudio terminator goes
into history; only the Mimi *decode* drops it (`audio_out[:-1]`).

---

## 6. Layer 3 — Context is prior thoughts (the heart of the design)

### 6.1 The principle

> **Context = the model's prior thoughts = its own generated outputs (text AND emitted audio).
> What enters context is decided by what the model GENERATED — never by I/O state (mic open?
> speaker muted? turn interrupted?).**

A chat model's context is user‑inputs *and* its own prior responses. For this model the response
is text *and* the audio it can't help but emit — co‑generated off one backbone (the "hebbian"
point: dropping either modality is brain damage). And the audio it emits is a **thought**, not
disposable output‑to‑play: it lives in the prefix, as audio, as part of the reasoning. Muting a
speaker or interrupting a turn must not change what the model remembers of itself.

(Memory: `context-is-thoughts-not-io`.)

### 6.2 ChatState — the five fields

`ChatState` (processor.rs) accumulates the model inputs across a conversation. It mirrors the
Python fields exactly; `**chat` unpacking becomes direct field access in `generate_*`:

```
   struct ChatState<'a> {
     proc:           &'a LFM2AudioProcessor,   // borrowed; transient per call (no Arc)
     codebooks:      usize,                    // 8
     text:           Tensor,   // (1, n)        i64 token ids               (torch.long)
     audio_in:       Tensor,   // (128, frames) f32 mel (continuous in)
     audio_in_lens:  Tensor,   // (k,)          i64 frame lengths per segment
     audio_out:      Tensor,   // (8, m)        i64 RVQ codes (discrete out)
     modality_flag:  Tensor,   // (1, n+…)      i64 LFMModality per position (the interleave order)
   }
```

Builder methods mirror the Python usage: `new_turn(role)` / `add_text` / `add_audio` /
`end_turn` / `append`. `LFMModality` = `{ Text=1, AudioIn=2, AudioOut=3 }`.

### 6.3 Continuous‑in vs discrete‑out (image‑embedding vs RVQ tokens)

This distinction is the single most important thing to understand about the model:

```
   USER's voice (audio‑IN)                     MODEL's voice (audio‑OUT)
   ───────────────────────                     ─────────────────────────
   continuous embeddings                        discrete RVQ codes (8 per frame)
   mel → Conformer → adapter                    Depthformer output
   "like a base‑64 image in context"            "tokens, embedded by audio_embedding"
   modality AUDIO_IN                            modality AUDIO_OUT
   re‑encoded through the Conformer on prefill  embedded via audio_embedding+offsets on prefill
```

The user's voice is **not transcribed** to text — that would be lossy and slow. It stays as
analog embeddings the backbone attends to directly, exactly like an embedded image. The model's
own voice is **not** kept as a waveform — it is kept as the *codes* it generated, embedded by
the model's own table.

### 6.4 `audio_embedding` (context) vs Mimi (sound) — two sinks of the same codes

The same 8‑code frame fans out to **two disjoint consumers**:

```
                       frame = 8 RVQ codes  (GenToken::Audio)
                               │
              ┌────────────────┴───────────────────┐
              ▼                                     ▼
   CONTEXT path (reasoning)               PLAYBACK path (sound)
   audio_embedding.embed(codes+offsets)   Mimi.decode_step(codes) → PCM
        .sum(over 8 codebooks) → (D,)      → resample → speaker ring
        = the model's OWN audio token      = disposable; never written to ChatState
   ─────────────────────────────────────  ─────────────────────────────────────
   model weights (vb.pp("audio_embedding"))   moshi codec (tokenizer-…checkpoint125)
   in the prefix forever                       gone after it's heard
```

`audio_embedding` is a `SharedEmbedding(hidden, AUDIO_VOCAB_SIZE × codebooks)` — part of the
**model**, not Mimi. So: *codes → audio_embedding → prefix* (thought); *codes → Mimi → sound*
(output). Muting the speaker skips only the right‑hand branch; the prefix is byte‑identical.

### 6.5 The prefill scatter

`prefill_inputs` (lfm2_audio.rs) reconstructs the woven input sequence from the three separated
streams using `modality_flag` as the order:

```
   text_emb   = embed_tokens(text)                              # (n_text, D)
   ai_emb     = [Conformer(adapter) per audio_in segment]       # (n_ai,   D)
   ao_emb     = audio_embedding(audio_out + offsets).sum(0)     # (n_ao,   D)

   combined   = cat[ text_emb ; ai_emb ; ao_emb ]               # (n_total, D)
   index[pos] = for each modality_flag position, the next index into the matching block
   in_emb     = combined.index_select(index)                    # (1, L, D)  ← the woven sequence
```

So the backbone sees text, the user's continuous audio, and the model's prior discrete audio,
**all interleaved in the exact order they occurred** — one multimodal context, nothing flattened.

### 6.6 Multi‑turn: append / from_parts / persistent conv

Single‑turn never exercises the discrete‑audio‑context path. The model's real use is multi‑turn,
feeding its own generated audio back as context. Two mechanisms make this work in the engine:

```
   ChatState::append(text, audio_out, modality_flag)     ✅ processor.rs
     ── cats the generated text + discrete audio_out (FULL, incl. EOAudio) + interleaved flags
        onto the persistent state.  (Python: chat.append(...) ; chat.end_turn())

   ChatState::from_parts(proc, codebooks, 5 tensors)     ✅ processor.rs
     ── rebuild a transient ChatState from a persisted conversation (no fresh <|startoftext|>).
        ChatState borrows the processor, so it can't be stored beside it; the engine holds the
        five tensors (ConversationState) and rebuilds a ChatState each turn via from_parts.
```

The engine (`Lfm2VoiceEngine`, realtime.rs) holds `conv: Option<ConversationState>`:

```
   TURN 1 (cold start)                         TURN n (warm)
   ───────────────────                         ─────────────
   ChatState::new + system turn (once)         ChatState::from_parts(conv.clone())
   add_audio(user)                              new_turn(user) + add_audio(user)
   generate_interleaved → collect              generate_interleaved → collect
      text_ids, audio_frames(incl EOAudio),        (same)
      modality_out (interleaved order)
   append(text, audio_out, modality)            append(...)
   save → self.conv                             save → self.conv
```

```
  ┌── the discrete audio_out → context loop across turns ───────────────────────────────────┐
  │                                                                                          │
  │  TURN 1   question.wav ─►add_audio─► [Conformer] ──┐ CONTINUOUS in                        │
  │           system/user text ─►tokenizer ───────────┤                                      │
  │                                                    ▼                                      │
  │                                            [LFM2 backbone] ─► hidden                       │
  │                                        ┌───────────┴───────────┐                          │
  │                                        ▼                       ▼                          │
  │                                   [text head]          [Depthformer] 8 codes/frame        │
  │                                        │                       │  DISCRETE out             │
  │                                        └──── interleaved ──────┘                          │
  │                          collect: text(1,n) · audio_out(8,m) · modality(1,n+m)            │
  │                                   ┌────────────┴─────────────┐                            │
  │                                   ▼                          ▼                            │
  │                       Mimi.decode(audio_out[:-1])      chat.append(...) ─► persistent conv │
  │                          ─► answer1 (sound)                  │                            │
  │  TURN 2  add_text("…chairs…") ──────────────────────┐       │                            │
  │                                                     ▼       ▼                            │
  │             prefill scatters [ text | audio_in | audio_out(→audio_embedding) ]            │
  │                                          ◄── turn 1's audio = CONTEXT                     │
  │                                                     ▼                                      │
  │                          [LFM2 backbone]  (conditioned on its own prior audio)            │
  │                                                     ▼                                      │
  │                              interleaved text + audio ─► answer2 (chairs‑conditioned)      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

✅ Proven: `examples/chat_multiturn` on Metal bf16 — turn 1 "Handcrafted Excellence, Every Day"
(woodworking), turn 2 a *chairs/seating* slogan, conditioned on turn 1's appended audio. Plus
`realtime::tests::engine_multiturn_grows_conv` (real model, `#[ignore]`, ~41 s) asserting `conv`
text/audio_out/modality all grow turn‑1→turn‑2.

### 6.7 I/O‑independence (the barge‑in fix)

Persistence is keyed on what the model **generated**, not on I/O:

- Audio frames are collected for `append` at *generation* time, **before** the EOAudio/playback
  branch — so a muted speaker changes nothing about what enters context. ✅
- Persistence is **not** gated on clean completion. Previously `if completed { append }` *discarded*
  the model's partial response on barge‑in — making context depend on an I/O event (the
  interrupt). Fixed: append whenever the model generated anything (`!text_ids.is_empty() ||
  !audio_frames.is_empty()`), complete or cut short. A thought the model started is still a prior
  thought. ✅

---

## 7. Layer 4 — The KV cache and the memory problem

There are **two** distinct quadratic costs; they were conflated and must not be.

### 7.1 Intra‑generation: the cat cache (a *speed* waste)

The backbone originally hand‑rolled the KV cache as `Tensor::cat(whole cache, new) +
(k.clone(), v.clone())` every attention step — 2× O(L) copies per step ⇒ O(L²) memory *traffic*
over a generation. This is GLM's "death of prefill." It is bandwidth, not gigabytes.

✅ **Fixed:** the backbone now uses **candle‑nn 0.9.2's own preallocated `KvCache`** — it was in
our pin all along; the backbone simply wasn't using it (the "vendored but never rewired" gap).

```
   cat cache (old)                          preallocated KvCache (now)
   ──────────────                           ──────────────────────────
   every step:                              every step:
     cat([cache, new]) → O(L) alloc+copy      slice_set(new) into a fixed buffer → O(new)
     (k.clone(), v.clone()) → O(L) copy        return narrow(0..len) view (no copy)
   = O(L²) traffic / generation             = O(L) traffic / generation
```

`Cache.kvs: Vec<KvCache>`, `KvCache::new(dim=2, KV_CACHE_INITIAL_CAP=512)` per layer, grown
geometrically. Parity verified: identical greedy text ("Handcrafted Excellence, Every Day"); 62
lib tests green. The **Depthformer** keeps the cat‑based `ConcatKvCache` (faithful to Python's
`LayerKVCache`, and its sequences are 8 codes long — no O(N²) there). `ConcatKvCache` stays.

### 7.2 Inter‑turn: re‑prefill (the *14 GB* monster)

The live multi‑turn test ballooned `mic_chat` to **14 GB resident + 4.3 GB compressed**. That is
**not** the cat cache (the KV tensors are a few MB/layer). It is the attention **score matrix**:
`q·kᵀ` materializes `(heads, L, L)` and is upcast to **f32**; at L≈5–6 k tokens × heads × 4 B
that is multiple GB *per layer*, transiently stacked. And the engine **re‑prefills the entire
conversation every turn** — rebuilds `ChatState` from `conv`, re‑encodes *all* prior audio, runs
the *whole* accumulated context through the backbone with a fresh cache. So turn 15 pays the full
O(L²) score matrix over 15 turns. That is the 14 GB and the 9 s→23 s slowdown.

◻️ **The real fix (next):** **inter‑turn KV persistence** — keep the backbone `KvCache` (and conv
state) across turns; each turn encodes + prefills only the *new* turn and appends. `L` per turn
stays small → the score matrix stays small → memory flat, speed flat. The preallocated‑KvCache
rewire (§7.1) is the *prerequisite* that makes this a small, surgical change.

```
   re‑prefill (now)                         persisted KV (next)
   ────────────────                         ───────────────────
   turn n: prefill( all of turns 1..n )      turn n: prefill( only turn n's new tokens )
           score matrix (Lₜₒₜ, Lₜₒₜ)                 attend new queries against cached K
           = O(Lₜₒₜ²)  → 14 GB                       = O(Lₜₒₜ) per new token  → flat
```

A context **bound** (cap/trim oldest turns) is the cheap stopgap; KV persistence is the answer.
candle‑nn 0.9.2 also ships `RotatingKvCache` (a bounded ring buffer) if we later want a hard cap.

---

## 8. Layer 5 — The orchestration layer (voice‑as‑agent + delegation)

This is the layer wrongly left in `experiments/`; it migrates into `src-tauri/src/voice/`.

### 8.1 Why delegation exists

LFM2.5‑Audio is the *interface*, not the *worker*. It cannot code, do research, or touch the
system, and it has no native function calling. So it gets one primitive — a **text‑marker tool**.
The system prompt instructs it: chat naturally and answer simple questions itself, but for *real
engineering, coding, research, or file/system work*, **do not attempt it**; say "I'll get my
engineer on it" and emit exactly one line:

```
   DELEGATE: <a clear, self‑contained description of the task>
```

The orchestrator watches the **text channel** (the model's own text stream — the same prior‑
thoughts stream from §6) for that marker and routes accordingly.

### 8.2 The routing loop

```
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │  mic ─► record_utterance ─► engine.respond(utt)  (interleaved: speech + text channel)  │
   │                                  │                                                     │
   │                            text channel                                                │
   │                                  ▼                                                      │
   │                        extract_delegation(text)                                        │
   │                       ┌──────────┴───────────┐                                         │
   │                  no marker               "DELEGATE: task"                              │
   │                       │                       │                                        │
   │              play LFM's own reply    (1) play LFM's ack in its OWN voice (if produced) │
   │              (small talk)            (2) engineer.run(task)   ◄── the capable model     │
   │                                      (3) LFM speaks the engineer's result (TTS)         │
   └──────────────────────────────────────────────────────────────────────────────────────┘

   routes:  marker (default) · chat (LFM only, never delegate) · delegate (everything → engineer)
```

The "ack first, then delegate" step is deliberate barge‑in feel: the user hears LFM say "on it"
immediately, *then* the round‑trip happens, instead of dead air.

### 8.3 The subagent (the engineer)

`glm.rs` is a real **agentic tool loop**, not a one‑shot call:

```
   run_subagent(task, allow_exec, cwd, max_steps):
     messages = [ system(engineer), user(task) ]
     tools    = bash_tool  iff  LFM_ALLOW_EXEC=1
     loop (≤ max_steps):
        msg = chat.completions(messages, tools)        # OpenAI‑compatible HTTP
        if msg has no tool_calls:  return msg.content   # short, speakable summary
        for call in msg.tool_calls:                     # e.g. bash
            result = run(call); append tool result to messages
```

- **Pluggable backend by design:** `GLM_BASE_URL` (default `https://ollama.com/v1`), `GLM_MODEL`
  (default `glm-5.1`), key from `$OLLAMA_API_KEY` or the EmberHarmony auth store. It is "GLM **or
  any other user's model**" over the OpenAI‑compatible API.
- The engineer returns a **short, plain‑spoken** summary (no markdown/code fences) so LFM can read
  it aloud.

### 8.4 ⚠️ The open question — who is the engineer?

Three candidate delegation targets, in increasing integration:

```
   (a) GLM/ollama‑cloud subagent as written      → a voice toy that can run bash
   (b) any user‑configured OpenAI‑compatible model→ generic, BYO‑model
   (c) EmberHarmony's OWN agent loop             → talk to EmberHarmony; it does the real work
        (file tools, session, context, the same brain the text UI drives)
```

Recommendation: **(c)/(b)** — the `DELEGATE:` marker should hand to the real EmberHarmony agent
(or the user's configured model), with the GLM HTTP subagent as one concrete backend. That is the
difference between "a voice front that shells out" and "*talk to EmberHarmony* and it does the
work." This decision drives how much of `glm.rs` stays as‑is vs becomes a thin adapter onto the
real brain. **This is the next decision to make before wiring Layer 5 into Layer 7.**

### 8.5 Hardening (the PR threads on this layer)

Real defects to fix as this layer comes into the build (not won't‑fix):
- UTF‑8 char‑boundary panics in byte‑indexed truncations (`glm.rs` tool output / error body,
  `lfm.rs` stderr) — cut on `char_indices` boundaries. ⚠️
- Unbounded delegated `bash` (`.output()` with no timeout/cap) — bound with timeout + output
  cap/kill. ⚠️
- Auth path ignores `XDG_DATA_HOME` — resolve the same data dir as the main auth code. ⚠️
- `lfm.rs` (external `llama-liquid-audio-cli`) and `setup.sh`/GGUFs — **retired**, replaced by the
  native engine.

---

## 9. Layer 6 — Audio I/O (cpal)

Native mic/speaker via `cpal`, the same approach across the prototype's `audio.rs` and the
crate's examples. To be **unified** into one module (not duplicated) as Layer 5 lands.

```
   CAPTURE                                          PLAYBACK
   ───────                                          ────────
   default input device                             default output device
   downmix to mono, normalize per sample format      ring buffer (VecDeque<f32>)
   RMS gate: start on first speech (rms ≥ thr),       generate loop pushes decoded PCM chunks
     end after ~0.8–1.0 s silence (or max cap)        cpal callback drains ring → all channels
   resample device‑rate → 16 kHz (Conformer input)   resample 24 kHz (Mimi) → device rate
   → utterance (Vec<f32>, 16 kHz) or WAV              barge‑in: clear ring + interrupt
```

- **Turn mode (Phase 1):** audio‑in is a *webview‑recorded clip* sent over IPC; audio‑out is an
  `AudioClip{wav}` the webview plays in an `<audio>` element. **No cpal in turn mode.**
- **Live mode (Phase 2):** cpal owns capture+playback in Rust; only `Level{rms}` (a scalar)
  crosses the IPC boundary for the visualizer — no PCM, no `MediaStreamTrack` in the webview.

VAD today is a simple energy/RMS gate. Open question: keep it, or add a real VAD (Silero via
`ort`) for live mode endpointing.

---

## 10. Layer 7 — The Tauri integration (pipeline + bridge)

### 10.1 The realtime pipeline

`RealtimePipeline` (realtime.rs) is a faithful restructuring of `chat.py`'s producer/consumer
threading: a **persistent inference worker thread** owns the model and loops
`recv utterance → respond (emit text + decode audio → emit PCM) → TurnComplete`. Because the model
lives on its own thread, capture and playback are never blocked by generation (full‑duplex), and
a new utterance can request **barge‑in** via an `AtomicBool` the generate loop polls.

```
   ┌──────────────────────────── RealtimePipeline ────────────────────────────────────────┐
   │   submit(Utterance) ──► crossbeam unbounded ──►  worker thread (owns Lfm2VoiceEngine)  │
   │                                                     for utt in rx.iter():              │
   │                                                        cancel.store(false)             │
   │                                                        engine.respond(utt, &cancel,    │
   │                                                           |ev| event_tx.send(ev))       │
   │                                                        send terminal (TurnComplete |    │
   │                                                                       Interrupted |     │
   │                                                                       Error)            │
   │   interrupt() ──► cancel.store(true)  (barge‑in: respond breaks, emits Interrupted)    │
   │   events() ──► crossbeam Receiver<VoiceEvent>  (drained by the consumer/bridge)         │
   │   Drop: cancel + close utt channel + join worker  (no detached thread, no leak)         │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

`VoiceEngine` is a trait so the threading is unit‑tested with fakes (no model needed):
`ScriptEngine`, `LoopEngine`, `ErrEngine`, `GuardedEngine` cover ordering, persistence across
turns, barge‑in, error survival, and Drop‑joins‑worker. `Lfm2VoiceEngine` is the real
implementation (owns model + processor + Mimi + the persistent `conv`).

### 10.2 The crossbeam → Tauri `Channel` bridge

```
   pipeline events (crossbeam Receiver<realtime::VoiceEvent>)
        │   a plain std::thread:  for ev in rx.iter() { channel.send(map(ev)) }
        ▼
   tauri::ipc::Channel<control::VoiceEvent>   (ordered, high‑throughput, webview‑facing)
```

No tokio, no `try_recv` timer. `Channel::send` is `Send + Sync` (tauri
`OnMessageFn = Box<dyn Fn(..) + Send + Sync>`), callable from any OS thread, so blocking
`rx.iter()` is lowest‑latency. One bridge per session (there is only ever one active session — the
demo's `isGenerating` guard — so no competing consumers steal events).

### 10.3 Event mapping (the four gaps)

`realtime::VoiceEvent {Text, Audio, TurnComplete, Interrupted, Error}` → `control::VoiceEvent
{State, Transcript, Level, AudioClip, Ended, Error}` is **not** 1:1:

```
   realtime                control                            note
   ────────                ───────                            ────
   Text(frag)          →   Transcript{Assistant, CUMULATIVE}  bridge accumulates a per‑turn string
   Audio(pcm)          →   live:  cpal ring + Level{rms}      PLAY it; only a scalar crosses IPC
                           turn:  accumulate → AudioClip{wav} encode at TurnComplete
   TurnComplete        →   turn: AudioClip then State{Idle}; live: State{Listening}
   Interrupted         →   turn: State{Idle};                live: State{Listening}
   (synthesized)       →   State{Thinking} on submit, State{Speaking} on first Audio
```

### 10.4 The resolved design decisions (Phase‑1 wiring)

```
   1. Model lifecycle   → EAGER‑ASYNC on provider==lfm2 (not app‑start, not first‑press).
                          3 GB + ~21 s load; voice_status gains loading|ready; mic enables on ready.
   2. Pipeline storage  → ONE persistent RealtimePipeline in State<Mutex<Option<…>>>; the warm
                          model lives INSIDE it; reused every turn; dropped on provider‑change/exit.
   3. Bridge            → plain std::thread, rx.iter() → channel.send; no tokio/timer.
   4. Event mapping     → §10.3 (cumulative transcript; PLAY the PCM; synthesized states).
   5. Phase 1 surface   → voice_generate_turn (one‑shot) ON the persistent pipeline; live mode
                          (voice_start_live/stop_live, cpal VAD) is Phase‑2 and additive.
```

### 10.5 Command surface

```
   voice_status() -> VoicePlan { provider, ready, detail }     # branch by provider; runtime readiness
   voice_generate_turn(req, channel)     # turn: one turn, streams VoiceEvents, ends State::Idle
   voice_start_live(ctx, channel)        # Phase 2: continuous; cpal capture + VAD
   voice_stop_live()                     # Phase 2
   voice_abort_turn()                    # cancel in‑flight generation (the AtomicBool)
   voice_set_mic_enabled(on)             # live only: pause/resume capture
   voice_settings_get / voice_settings_set
```

`TurnMode {Asr, Tts, Interleaved}` (control.rs) carries the demo's three system prompts
(`"Perform ASR."` / `"Perform TTS. Use the UK …​ voice."` / `"Respond with interleaved text and
audio."`) and per‑mode `max_new_tokens` (100 / 1024 / 1024). The engine holds a `system_prompt`
field (default interleaved) settable via `with_system_prompt`; the desktop layer maps `TurnMode →
(prompt, budget)` because the crate can't depend on the desktop's `TurnMode` (dependency points
the other way).

---

## 11. Threading model

```
   webview (JS event loop)
        │  invoke(voice_*)                      ▲  Channel<VoiceEvent>
        ▼                                       │
   Tauri command (async, tokio)                 │
        │  submit(Utterance) / interrupt()      │  bridge std::thread: rx.iter() → channel.send
        ▼                                       │
   RealtimePipeline                             │
        │  crossbeam unbounded                  │  crossbeam Receiver
        ▼                                       │
   inference worker (std::thread, owns model) ──┘   ← P‑cores; generation never blocks I/O
        │
        ├─ candle intra‑op threads (parity with torch's at::intraop_default_num_threads)
        └─ cpal callback threads (capture push / playback drain)  — live mode
```

- The model lives on **one** dedicated OS thread; the webview never blocks on generation.
- `tauri::ipc::Channel::send` is `Send + Sync` — verified from tauri source — so the bridge is a
  plain thread; no tokio dance needed to cross from the worker to the webview.
- Intra‑op thread count mirrors torch's policy (`threads.rs`) so CPU numerics/perf track the
  Python reference. A NEON `BFMMLA` bf16 GEMM (`bf16_gemm.rs` + `csrc/bf16_gemm.c`) *would* close
  candle's bf16‑on‑CPU gap — but it is ⚠️ **built and tested, not wired into any matmul** (§4.8).

---

## 12. Phasing — turn parity now, full‑duplex (Moshi) later

```
   Phase 1  (NOW)  — absolute parity with Liquid AI's WebGPU demo
       turn‑based · ASR / TTS / Interleaved · clip‑based
       audio‑in = webview clip ; audio‑out = AudioClip player ; no cpal
       prove we do exactly what Liquid did, with the native engine instead of transformers.js

   Phase 2  (LATER) — natural, full‑duplex, interruptible conversation
       the end state is NOT turn‑based; speech is continuous and barge‑in‑able
       the right model is MOSHI (the LM, not just Mimi the codec): architecturally full‑duplex,
       processes input AND output streams every frame → true simultaneous text+audio EMISSION
       cpal owns capture+playback in Rust; webview gets Level + Transcript only
```

LFM2 vs Moshi, precisely:

```
   LFM2 (now)                                Moshi (Phase 2)
   ──────────                                ───────────────
   reasons over both per step (shared        same shared reasoning, PLUS
     backbone) but EMITS one modality        emits text + all audio codebooks EVERY frame
     per step (time‑multiplex 6:12)            (parallel streams, inner monologue aligned)
   turn‑based + a VAD loop (stopgap)         architecturally full‑duplex (no VAD hack)
```

The event‑driven core (`VoiceEvent` session) is **identical** across phases; Phase 2 is additive
(a new trigger + a different model behind the same session), not a rewrite. Build Phase 1 to
parity; do not paint Phase 2 into a corner.

---

## 13. File & module map

```
   packages/desktop/src-tauri/
   ├─ Cargo.toml                         liquid-audio dep (features=["metal"] on macOS)
   ├─ src/voice/
   │   ├─ VOICE_ARCHITECTURE.md          ← this document
   │   ├─ FRONTEND_DESIGN.md             webview/UX design + the resolved Phase‑1 decisions
   │   ├─ control.rs                     voice_status, VoiceEvent, TurnMode, voice_* (stubs → wiring)
   │   ├─ session.rs                     (legacy) LiveKit SSE reducer — retired in the cutover
   │   ├─ orchestration.rs   ◻️           Layer 5: DELEGATE routing  (migrate from experiments main.rs)
   │   └─ engineer.rs        ◻️           Layer 5: capable‑model subagent (migrate from experiments glm.rs)
   └─ crates/liquid-audio/               the native model engine (Layers 1‑4)
       ├─ src/
       │   ├─ loader.rs                  config.json + safetensors → model + processor
       │   ├─ processor.rs               LFM2AudioProcessor + ChatState (new/append/from_parts/add_*)
       │   ├─ realtime.rs                RealtimePipeline + Lfm2VoiceEngine + VoiceEvent  (Layer 7 seam)
       │   ├─ detokenizer.rs             LFM2 audio detokenizer (one‑shot decode backend)
       │   ├─ audio_out.rs               AudioDetokenizer trait + MimiDetokenizer (moshi)
       │   ├─ resample.rs                torchaudio.functional.resample (windowed‑sinc) port
       │   ├─ threads.rs                 intra‑op thread parity with torch
       │   ├─ bf16_gemm.rs (+ csrc/)     NEON BFMMLA bf16 CPU GEMM  ⚠️ built+tested, NOT wired (§4.8)
       │   ├─ candle_ext/                vendored candle 0.10 backports on the 0.9.2 pin
       │   │   ├─ kv_cache.rs            ConcatKvCache (Depthformer)
       │   │   ├─ transformers_utils.rs  build_causal_mask + repeat_kv
       │   │   └─ tensor_ext.rs          to_vec4
       │   ├─ model/
       │   │   ├─ lfm2_audio.rs          generate_* · prefill_inputs · Depthformer · GenParams
       │   │   ├─ lfm2_hf.rs             LFM2 backbone (hybrid conv+GQA) + Cache (KvCache)
       │   │   ├─ transformer.rs         shared blocks · LayerKvCache (wraps ConcatKvCache)
       │   │   ├─ mlp.rs                 audio_adapter
       │   │   └─ conformer/             subsampling · encoder · mha · processor (mel)
       │   └─ data/                      mapper · dataloader · arrow_io  (training preprocessing)
       ├─ examples/
       │   ├─ generate.rs                ✅ single‑turn end‑to‑end (the canonical proof)
       │   ├─ chat_multiturn.rs          ✅ canonical 2‑turn discrete‑audio‑context proof
       │   ├─ mic_chat.rs                ✅ live mic loop (now engine‑backed, persistent memory)
       │   ├─ duplex_chat.rs             full‑duplex barge‑in loop (RealtimePipeline)
       │   └─ text_chat.rs               text→text proof
       └─ parity/                        dump_*.py + golden/*.safetensors (gitignored, regenerable)

   experiments/lfm2-audio-voice/         the prototype this stack supersedes
   ├─ src/{main,glm,lfm,audio}.rs        ← Layer 5 source‑of‑truth to migrate (main+glm); lfm/audio retired
   ├─ setup.sh                           builds llama.cpp PR #18641 + GGUFs — retired (native engine)
   └─ upstream-liquid-audio/             the Python reference we port from (kept, not shipped)
```

---

## 14. Verification & proofs

```
   ✅ text→text            examples/text_chat        tokenizer→backbone→text head coherent
   ✅ single‑turn E2E      examples/generate         "Handcrafted Excellence, Every Day" + 7.7 s Mimi WAV,
                                                     Metal bf16, ~15 tok/s
   ✅ multi‑turn discrete  examples/chat_multiturn    turn 2 chairs‑conditioned on turn 1's appended audio
   ✅ engine persistence   realtime::tests::engine_multiturn_grows_conv  (real model, #[ignore], ~41 s):
                                                     conv text/audio_out/modality all grow t1→t2
   ✅ live mic             examples/mic_chat (engine) real conversation on hardware ("It works SO good"):
                                                     cpal mic → 48k→16k resample → mel → Conformer → LFM2 →
                                                     Depthformer → Mimi streaming → speaker, real‑time
   ✅ KvCache rewire       cargo build --features metal + 62 lib tests + identical greedy text (parity)
   ✅ threading            realtime::tests           ordering · persistence · barge‑in · error survival · Drop
```

Faithfulness is gated on **behavior** (ported tests, golden differential dumps, real runs), never
on structural/AST similarity (a cheatable shadow). Heavy/real tests are **run**, not stubbed and
ignored.

---

## 15. Open questions / next moves

```
   ◻️ #1  The engineer (§8.4): GLM subagent / any user model / EmberHarmony's own agent?
            → drives where Layer 5 plugs into Layer 7 and how much of glm.rs survives.
   ◻️ #2  Inter‑turn KV persistence (§7.2): the real 14 GB fix; prerequisite (KvCache) is done.
   ◻️ #3  Migrate Layer 5 (main.rs routing + glm.rs subagent) into src-tauri/src/voice/ and
            wire DELEGATE → the chosen engineer; retire lfm.rs/setup.sh (native engine replaces).
   ◻️ #4  Wire Phase‑1 voice_generate_turn on the persistent pipeline; rewrite voice.tsx; route
            the main button through the selected provider (PR threads T6/T9/T10/T17).
   ⚠️ #5  Crate hardening (PR threads): rate/sample_rate>0 guards (T4/T5), zero‑length Metal
            tensors (T8/T22), mel get_seq_len narrow (T20), encoder setup_streaming_params (T24),
            loader tokenizer‑* skip (T3), detok mask cache (T1).
   ⚠️ #6  Orchestration hardening (§8.5): UTF‑8 boundaries, bounded bash, XDG auth path.
   ◻️ #7  CI: macOS cargo build --features metal (T23); wiki scripts or remove workflow (T7).
   ◻️ #8  Phase 2: bring in the Moshi LM for true full‑duplex parallel streams.
   ⚠️ #9  Wire the NEON BFMMLA bf16 kernel into the matmul path (§4.8): add a bf16‑CPU path —
            device selection loads bf16 on CPU; a custom Linear routes 2‑D projections through
            bf16_matmul (f32 fallback); 4‑D attention scores stay f32. Built+tested today, but
            called by NOTHING — pure dead weight until wired. (= task "Thread parity #25".)
```

---

## 16. Glossary

```
   audio‑in            the user's voice; CONTINUOUS mel→Conformer embeddings (modality AUDIO_IN)
   audio‑out           the model's voice; DISCRETE 8 RVQ codes/frame (modality AUDIO_OUT)
   audio_embedding     the MODEL's learned audio token table (codes → backbone input) — CONTEXT
   ChatState           the five accumulating model‑input tensors (text/audio_in/lens/audio_out/flag)
   Conformer           the audio‑in encoder (ConvSubsampling + relative‑pos MHA + conv + FFN)
   ConcatKvCache       cat‑based KV cache (Depthformer; faithful to Python LayerKVCache)
   conv (engine)       ConversationState: the persisted five tensors carried across turns
   DELEGATE:           the one‑line text‑marker primitive the voice model uses to hand off work
   Depthformer         the 2nd autoregressive transformer; emits the 8 RVQ codes per frame
   EOAudio             code 2048 / all‑2048 frame; the end‑of‑audio terminator
   GenToken            Text(u32) | Audio(Vec<u32>) — the generation stream item
   interleaved_n_*     6 text : 12 audio — the modality time‑multiplex cadence
   KvCache             candle‑nn 0.9.2 PREALLOCATED KV cache (slice_set; backbone)
   LFM2 backbone       the hybrid short‑conv + GQA transformer shared by both heads
   Lfm2VoiceEngine     the real VoiceEngine: owns model+proc+Mimi+persistent conv (realtime.rs)
   Mimi                the codec (codes → 24 kHz waveform) for PLAYBACK only (moshi crate)
   modality_flag       per‑position LFMModality; the order the prefill scatter rebuilds
   RealtimePipeline    the worker‑thread pipeline (submit/interrupt/events)
   VoiceEvent          the streamed reply item (realtime: Text/Audio/…; control: State/Transcript/…)
```

---

*End of stack architecture. The heart is Layer 3: everything feeds, generates, or speaks the
model's prior thoughts; the orchestration layer is what lets those thoughts turn into work.*
