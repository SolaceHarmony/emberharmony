<!-- topic: Transport (off-path) -->
# TR01 · Moshi websocket server (asyncio)
**Code:** `TR01` · **Source:** `moshi/server.py` · **Rust:** `NOT ported (no async runtime)` · **On the LFM2-Audio inference path:** no

## Role
A standalone `asyncio` + `aiohttp` WebSocket server that exposes the **Moshi 7B** full-duplex speech LM (a *different* model from LFM2-Audio) over a binary websocket protocol with Opus-compressed audio. It is the vendored Kyutai reference transport: it streams Opus mic audio in, runs the Mimi codec + Moshi `LMGen` at a fixed 12.5 Hz frame cadence, and streams Opus speech + inner-monologue text tokens back. It is **off the LFM2-Audio inference path** — LFM2-Audio uses its own backbone + depthformer (`model_lfm2_audio`) and a synchronous streaming generator, never this server, never `lm_gen.step`. It exists here only because the whole `moshi/` subtree was vendored wholesale, and it is the conceptual reference for what a "true full-duplex" shell would look like (the Rust port stops at the half-duplex demo shape and ships no async runtime).

## How it works
The whole thing is a `ServerState` dataclass plus a `main()` CLI bootstrap. There is **no neural code here** — every tensor op is delegated to `MimiModel` (`moshi_compression`) and `LMGen` (`moshi_lm`). The mechanism is entirely the streaming/transport state machine.

**Construction (`server.py:47-60`).** `ServerState.__init__` stores the Mimi codec, a `sentencepiece` text tokenizer, and builds `self.lm_gen = LMGen(lm, cfg_coef, condition_tensors=…, **lm_gen_config)`. Two load-bearing scalars:
- `self.frame_size = int(mimi.sample_rate / mimi.frame_rate)` (`:56`). With `SAMPLE_RATE=24000`, `FRAME_RATE=12.5` this is exactly **1920 samples** — one Mimi frame = 80 ms of 24 kHz audio. This is the quantum the entire loop is chunked on.
- Both stateful modules are put into permanent streaming mode: `mimi.streaming_forever(1)` and `lm_gen.streaming_forever(1)` (`:59-60`), batch size 1. This installs the persistent KV-cache / conv-ring-buffer streaming state on the codec transformers and the LM so each `.step`/`.encode`/`.decode` is incremental, not a full re-encode.

**Warmup (`server.py:62-72`).** Pushes 4 chunks of `torch.zeros(1,1,frame_size)` f32 through the real path — `mimi.encode(chunk)` → per-codebook-column `lm_gen.step(codes[:,:,c:c+1])` → `mimi.decode(tokens[:,1:])` — purely to trigger CUDA-graph capture / `torch.compile` lazy compilation (`moshi_util_compile.CUDAGraphed`) before the first client, then `torch.cuda.synchronize()`. This is CUDA-only; the graphed wrappers no-op off-cuda.

**The per-connection coroutine triad (`handle_chat`, `server.py:74-173`).** One websocket = one `web.WebSocketResponse`. The connection is serialized by `async with self.lock` (an `asyncio.Lock` — **one active session at a time** for the whole server, because the codec/LM streaming state is a single shared mutable object). On entry it fresh-allocates `sphn.OpusStreamWriter`/`OpusStreamReader` at `mimi.sample_rate`, calls `mimi.reset_streaming()` + `lm_gen.reset_streaming()` to wipe per-session state, sends a 1-byte handshake `b"\x00"`, then runs three coroutines concurrently under `asyncio.gather(opus_loop(), recv_loop(), send_loop())` (`:171`). Shared `close` flag (set in `recv_loop`'s `finally`) is the cooperative shutdown signal; the loops poll it and `return`.

- **`recv_loop` (`:78-105`)** — `async for message in ws`. Only `WSMsgType.BINARY` is processed; the first byte is a `kind` tag. `kind==1` ⇒ audio: the remaining bytes are appended to `opus_reader.append_bytes(payload)` (Opus depacketization happens inside `sphn`). Anything else is logged and dropped. On exit it sets `close=True`.

- **`opus_loop` (`:107-151`)** — the actual inference pump. Each iteration `await asyncio.sleep(0.001)` (cooperative yield), then `pcm = opus_reader.read_pcm()` pulls decoded f32 PCM out of the Opus reader. PCM is accumulated into `all_pcm_data` via `np.concatenate`. While `>= frame_size` samples are buffered, it slices exactly `frame_size` (1920) samples, `torch.from_numpy(chunk).to(device)[None,None]` → shape `(1,1,1920)` f32, and `codes = mimi.encode(chunk)` → `(1, n_q, T)` int codes.
  - **`skip_frames` (`:129-135`)**: the *first* encoded frame is discarded — from the model's POV the first mic frame is "in the past," and Mimi's left-padding gives that first frame an anomalous structure, so it `mimi.reset_streaming()` to re-apply the left pad on the next call. Only the encode is thrown away, not re-done.
  - **The LM step (`:136-150`)**: for each codebook-time column `c`, `tokens = lm_gen.step(codes[:,:,c:c+1])`. `LMGen` internally applies the acoustic-delay pattern, runs the Moshi backbone + depformer, and **returns `None` until the delay warmup is satisfied**, then returns `(1, dep_q+1, 1)` — index 0 is the *text* (inner-monologue) stream, indices `1..dep_q+1` are the `dep_q=8` audio codebooks. `assert tokens.shape[1] == lm_gen.lm_model.dep_q + 1` (`:140`). Audio codes (drop the text row: `tokens[:,1:]`) go to `mimi.decode(...)` → `(1,1,1920)` f32 waveform, moved to CPU and pushed as raw PCM into `opus_writer.append_pcm(main_pcm[0,0].numpy())`. The text token `tokens[0,0,0].item()` is decoded *only* if it is not `0` (pad) or `3` (epad/special) (`:144-145`); `id_to_piece` → replace SentencePiece `▁` with a space → framed as `b"\x02" + utf8text` → `ws.send_bytes`.

- **`send_loop` (`:153-160`)** — drains `opus_writer.read_bytes()` (Opus-encoded output frames) and ships each as `b"\x01" + opus_bytes`. Same 1 ms cooperative-sleep poll.

**Protocol summary (byte-0 tag).** In: `\x01`=audio. Out: `\x00`=handshake, `\x01`=Opus audio, `\x02`=UTF-8 text token. The `\x02` text and `\x01` audio are the two duplex streams the browser client renders.

**`main()` (`:176-291`).** Pure CLI/IO plumbing: argparse, `seed_all(42424242)`, `CheckpointInfo.from_hf_repo` (→ `moshi_loaders`) to fetch Mimi + Moshi LM + tokenizer, `--cfg-coef` classifier-free-guidance coefficient, `--half` toggles bf16→fp16, `--device` defaults `cuda`. Optionally extracts a `dist.tgz` static bundle from HF and serves it, optionally a gradio tunnel, optionally SSL. `web.run_app` under a top-level `with torch.no_grad(): main()`.

There is **no normalization / attention / RoPE / conv math in this file** — all of that lives in `moshi_compression` (Mimi), `moshi_transformer`, `moshi_lm`, `moshi_vq`. This component is the I/O and frame-cadence state machine only.

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| Opus packet (ws `\x01`) | `bytes` | — |
| `opus_reader.read_pcm` | — | f32 PCM `(N,)` @ 24 kHz |
| frame slice → tensor | f32 `(1,1,1920)` | — |
| `mimi.encode` | f32 `(1,1,1920)` | int codes `(1, n_q, T)` (u32 in Rust) |
| `lm_gen.step` (per code column) | int `(1, n_q, 1)` | `None` during delay warmup, else int `(1, dep_q+1, 1)` = `(1,9,1)` |
| text stream row | `tokens[0,0,0]` int64 scalar | UTF-8 piece (skip ids 0,3) |
| `mimi.decode` (audio rows) | int `(1, 8, 1)` = `tokens[:,1:]` | f32 waveform `(1,1,1920)` |
| `opus_writer.read_bytes` | f32 PCM | Opus `bytes` (ws `\x02`/`\x01`) |

Model weights bf16 on disk (Python default cuda/bf16; `--half`⇒fp16). Mimi `encode`/`decode` run in the codec's model dtype; the mic PCM tensor is constructed f32 and Mimi's SEANet front-end upcasts internally. No norm/softmax dtype subtleties are visible at this layer — they are inside the delegated modules.

## Wiring
Off the LFM2-Audio tensor path; this is a self-contained Moshi-7B transport. Its internal neighbors are all Moshi-stack components:
- **Upstream (into the loop):** Opus bytes over the websocket → decoded to f32 PCM `(N,)` @ 24 kHz by `sphn`. The model-side upstream is the Mimi codec encode side: f32 `(1,1,1920)` → [moshi_compression](MM01-Mimi-Codec) `MimiModel.encode` → int codes `(1,n_q,1)`.
- **Core compute:** int codes `(1,n_q,1)` → [moshi_lm](MM03-Moshi-LM) `LMGen.step` → int `(1,9,1)` token frame (text + 8 audio codebooks). Conditioning tensors are built by `moshi_run_inference.get_condition_tensors` and passed into `LMGen` at construct time.
- **Downstream (out of the loop):** the 8 audio rows `(1,8,1)` → [moshi_compression](MM01-Mimi-Codec) `MimiModel.decode` → f32 `(1,1,1920)` @ 24 kHz → Opus-encoded → websocket `\x01`. The text row scalar → SentencePiece `id_to_piece` → websocket `\x02`. The natural client consumer is [moshi_client](TR02-WS-Client).

Note: none of [core_processor](CO01-Processor-ChatState), [model_lfm2_audio](MD01-LFM2AudioModel), or [core_detokenizer](CO02-Detokenizer) feed or consume this file — the LFM2-Audio path is entirely separate.

## Python ↔ Rust
**Not ported.** `liquid-audio-rs` ships no async runtime (no tokio), so `server.py`, `client.py`, and the whole websocket full-duplex shell have no Rust counterpart — this is a deliberate scope cut, documented in PYTHON_VS_RUST.md §4 (vendored `moshi/**` is reused as the `moshi` crate, not re-ported) and ARCHAEOLOGY.md Q4. The Rust analog of "drive Mimi + an LM frame-by-frame" exists only as the **half-duplex** `mic_chat.rs` demo: `cpal` callback threads + `Arc<Mutex<…>>` ring buffers, `generate_interleaved` run synchronously on the main thread, no coroutines, no `asyncio.Lock`, no `asyncio.gather`. Symbol-level there is nothing to map: `ServerState`, `recv_loop`/`opus_loop`/`send_loop`, `handle_chat`, `warmup`, `main` have **no Rust referent**. The Mimi `streaming_forever`/`reset_streaming`/`encode`/`decode` it calls *do* have a Rust analog in the `moshi` crate (`moshi::mimi::Mimi`, used via `audio_out.rs::MimiDetokenizer`), but driven by `mic_chat.rs`, not by a websocket server. Deliberate divergences inherited from the reused Mimi crate (eager SDPA vs flash, candle ops vs CUDA kernels, CUDAGraph disabled off-cuda) are catalogued in PYTHON_VS_RUST.md §2.2 / §2.3.

## Precision / gotchas
- **Single global session.** `self.lock` is one `asyncio.Lock` on the shared `ServerState`; the codec/LM carry one mutable streaming state, so the server handles exactly one client at a time. Concurrent connects serialize.
- **First-frame skip is mandatory (`:129-135`).** Forgetting the `skip_frames`/`reset_streaming` dance leaves the anomalous left-padded first frame in the stream and corrupts alignment — the comment in source is explicit that the encode is still run (for code simplicity) but discarded.
- **`lm_gen.step` returns `None` during delay warmup.** The acoustic delay pattern means the first several `step` calls produce no output token frame; the `if tokens is None: continue` (`:138-139`) is not an error path, it is the steady-state warmup of the multi-stream delay.
- **Text-stream special tokens.** Ids `0` and `3` are suppressed (`:145`) — `0` is pad, `3` is the inner-monologue epad/marker; only "real" pieces are emitted, with SentencePiece `▁`→space.
- **`tokens[:,1:]` drops the text row before Mimi decode (`:141`).** Row 0 is text; only rows `1..9` are the 8 audio codebooks Mimi understands. `dep_q+1 == 9`.
- **`frame_size` must stay `int(sample_rate/frame_rate)=1920`.** The whole accumulate-and-slice loop assumes integer frame samples; a non-integer ratio would desync encode/decode.
- **CUDA-coupled by construction.** `warmup()` ends in `torch.cuda.synchronize()` and the CUDA-graph/compile speedups only engage on cuda; off-cuda it runs but the graphing no-ops (see `moshi_util_compile`). This file is not part of the device-agnostic LFM2-Audio surface.
- **This is Moshi 7B, not LFM2-Audio.** Anyone tracing the LFM2-Audio pipeline should ignore this file: different model, different head (`dep_q` depformer vs LFM2 depthformer), different sampler. It is reference-only.
