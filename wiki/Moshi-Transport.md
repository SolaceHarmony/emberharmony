<!-- topic: Transport (off-path) -->
# Moshi Transport (off path)

The websocket full-duplex transport + offline runners (asyncio) — not on the LFM2-Audio path. Each `##` keeps its architecture code.

---

## TR01 · Moshi websocket server (asyncio)
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
- **Downstream (out of the loop):** the 8 audio rows `(1,8,1)` → [moshi_compression](MM01-Mimi-Codec) `MimiModel.decode` → f32 `(1,1,1920)` @ 24 kHz → Opus-encoded → websocket `\x01`. The text row scalar → SentencePiece `id_to_piece` → websocket `\x02`. The natural client consumer is [moshi_client](Moshi-Transport).

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

---

## TR02 · Moshi websocket client (asyncio)
**Code:** `TR02` · **Source:** `moshi/client.py` · **Rust:** `NOT ported` · **On the LFM2-Audio inference path:** no

## Role
A standalone asyncio CLI client for the **Moshi 7B full-duplex websocket server** (`moshi/server.py`). It captures microphone PCM via PortAudio (sounddevice), Opus-encodes it, streams it to the server over an aiohttp websocket, receives Opus-encoded model speech + text tokens back, decodes and plays the audio, and renders the text in the terminal. It contains **no model code** — it is pure transport + audio I/O + terminal UX, and it talks to the *Moshi* multi-stream LM, a different model from LFM2-Audio. It exists as the reference real-time chat front-end shipped with the vendored kyutai `moshi` package; LFM2-Audio's own realtime path is `demo/chat.py` (fastrtc), not this.

## How it works
The unit of streaming is one **Mimi frame = 1920 samples @ 24 kHz** (`Connection.__init__`, `client.py:26`), i.e. 80 ms, which is `sample_rate / frame_rate = 24000 / 12.5` — the codec framerate the server steps the LM at. All four streams (mic block, speaker block, Opus packetization, server frames) are sized around this 1920-sample frame.

**Audio device callbacks (PortAudio threads, not the event loop).** Two `sounddevice` streams are opened with `blocksize=frame_size=1920`, `channels=1`, `samplerate=24000` (`client.py:35-47`):
- `_on_audio_input` (`client.py:122`): PortAudio hands it an `(1920, 1)` f32 buffer; it asserts the shape, takes channel 0 `in_data[:, 0]` → `(1920,)` f32, and pushes it into `sphn.OpusStreamWriter.append_pcm`. No resampling — capture rate is already 24 kHz.
- `_on_audio_output` (`client.py:126`): PortAudio asks for `(1920, 1)`; it does a **non-blocking** `queue.Queue.get(block=False)` of one decoded `(1920,)` f32 frame and writes it to `out_data[:, 0]`. On `queue.Empty` it fills the buffer with zeros (silence) and calls `printer.print_lag()` to show a `[LAG]` marker. There is no ring buffer / FIFO smoothing — exactly one frame per callback, a deliberate "TODO" simplification (`client.py:130`).

These callbacks run on PortAudio's own threads; the thread-safe bridge to the asyncio world is the stdlib `queue.Queue` (`self._output_queue`, `client.py:50`) for playback, and the internal locking inside the `sphn` Opus reader/writer for capture.

**Three asyncio coroutines, run concurrently under `asyncio.gather`** inside `run()` (`client.py:137-141`), with the two device streams held open via a `with self._in_stream, self._out_stream:` context:

1. `_queue_loop` (`client.py:52`): the **uplink**. Polls every 1 ms (`asyncio.sleep(0.001)`), pulls finished Opus packets with `self._opus_writer.read_bytes()`, and if non-empty sends `b"\x01" + msg` over the websocket as a **binary** frame. The `\x01` byte is the audio-kind tag in the wire protocol. On send exception it logs and calls `_lost_connection`.
2. `_decoder_loop` (`client.py:66`): the **downlink audio assembler**. Polls every 1 ms, calls `self._opus_reader.read_pcm()` to drain whatever PCM the Opus decoder has produced from bytes fed by `_recv_loop`, `np.concatenate`s it onto a running `all_pcm_data` accumulator, then while `>= frame_size` slices off exactly 1920 samples and `put`s each into `_output_queue` for the speaker callback. The leftover tail is re-wrapped with `np.array(...)` to keep it contiguous (`client.py:79`).
3. `_recv_loop` (`client.py:81`): the **downlink demux**. `async for message in self.websocket` — handles `WSMsgType.CLOSED`/`ERROR`, ignores non-`BINARY`, then reads `kind = message[0]` (`client.py:102`): **kind==1 → audio**, append `message[1:]` to the Opus reader (`append_bytes`) and tick the spinner (`print_pending`); **kind==2 → text**, decode `message[1:]` as utf-8 and `print_token`. Unknown kinds warn. This is the mirror image of the server's `b"\x01"+opus` / `b"\x02"+utf8` framing (`server.py:148,160`).

**No model, no sampling, no tensors.** All of generation — backbone, depformer, Mimi encode/decode, RVQ, temperature/top-k/top-p sampling, KV cache, turn-taking — happens **server-side**. The client never sees codes or logits, only Opus byte streams and utf-8 text tokens. Streaming "state" here is purely the `all_pcm_data` numpy accumulator + the two Opus stream objects + the playback queue; there is no neural streaming state.

**Connection bring-up** (`run()`, `client.py:144`): builds a `ws://`/`wss://` URI for path `/api/chat`, either from `--host/--port/--https` or a raw `--url` (with protocol normalization of `ws/http`→`ws`, `wss/https`→`wss`). Opens one `aiohttp.ClientSession` → `ws_connect`, prints a header, constructs `Connection`, and awaits `connection.run()`.

**Terminal UX** (`client_utils.py`): `Printer` (TTY) maintains a single live `Line` with manual `\r` cursor erase/redraw, an 80-col box, ANSI colorization (`colorize`, `client_utils.py:11`), word-wrap that prefers breaking on spaces, a `[LAG]` token, and a rotating spinner (`print_pending` cycles `| / - \` through green/yellow/red, `client_utils.py:205`). `RawPrinter` (non-TTY) just writes tokens straight through. Selection is by `sys.stdout.isatty()` in `main()` (`client.py:184`).

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| Mic callback `_on_audio_input` | `(1920, 1)` f32 (PortAudio) | `(1920,)` f32 → Opus writer |
| Uplink `_queue_loop` | Opus packet `bytes` | ws binary `b"\x01" + bytes` |
| Downlink demux `_recv_loop` (audio) | ws binary `b"\x01" + bytes` | `bytes` → Opus reader |
| Downlink demux `_recv_loop` (text) | ws binary `b"\x02" + bytes` | `str` (utf-8) → terminal |
| Decode `_decoder_loop` | Opus reader → PCM `np.float32` `(N,)` | per-frame `(1920,)` f32 → `queue.Queue` |
| Speaker callback `_on_audio_output` | `(1920,)` f32 from queue | `(1920, 1)` f32 (PortAudio) or zeros |

No bf16, no int64 token ids, no codes, no mel anywhere in this file — those all live server-side. The only numeric dtype the client handles is **f32 PCM @ 24 kHz** (mono) and opaque Opus byte strings. The `1920` frame size derives from `sample_rate(24000) / frame_rate(12.5)`.

## Wiring
**Upstream (over the network, not an in-process tensor edge):**
- Microphone / PortAudio → `(1920,1)` f32 mono @ 24 kHz → this client.
- **Moshi server** [moshi_server](Moshi-Transport) → ws binary frames: `b"\x01"+opus` (model speech) and `b"\x02"+utf8` (text tokens) → this client's `_recv_loop`. The server's own audio comes out of the Mimi codec [moshi_compression](MM01-Mimi-Codec) at 24 kHz, re-encoded to Opus by `sphn`.

**Downstream:**
- This client → ws binary `b"\x01"+opus` (mic) → **Moshi server** [moshi_server](Moshi-Transport), which Opus-decodes to 24 kHz PCM and feeds the Mimi encoder + Moshi LM [moshi_lm](MM03-Moshi-LM).
- This client → speaker / PortAudio (`(1920,1)` f32) and terminal (utf-8 text). These are device sinks, not modeled components.

Note: every edge here is a **websocket / PortAudio boundary**, not an in-process tensor flow. This component does not feed any LFM2-Audio module — it is wired only to the Moshi server and the OS audio devices.

## Python ↔ Rust
**Not ported.** There is no `client.rs` (or `moshi/` subtree) under `liquid-audio-rs/src/` — the Rust port covers the model/codec/processor/detokenizer/trainer, not the kyutai websocket transport. `PORT_STATUS.md:126` states the policy explicitly: the demo's thread+queue maps to `std::thread` + channel, and "`moshi` websocket server/client → async (tokio) **only if the transport is ported**" — and it is not. So there is no symbol-level mapping for this file.

If it were ported, the natural shape would be: `sounddevice` callbacks → `cpal` input/output streams; `sphn` Opus → an `opus`/`ogg` crate; `aiohttp` ws → `tokio-tungstenite`; the three `asyncio` coroutines → `tokio::select!`/spawned tasks; `queue.Queue` → an `std::sync::mpsc` / `crossbeam` channel. None of this exists today. This is a **transport/UX shim with no numerical content**, so there are no candle-ops-vs-CUDA, eager-vs-flash, or dtype divergences to record — it falls entirely outside the scope captured in `PYTHON_VS_RUST.md`.

## Precision / gotchas
- **Not the LFM2-Audio path.** This talks to the Moshi 7B LM [moshi_lm](MM03-Moshi-LM), a *different* model. Reading it tells you the Moshi server wire protocol, not anything about LFM2-Audio inference. The on-path realtime client is `demo/chat.py` ([demo_chat](DM01-Realtime-Chat)).
- **Wire framing is a 1-byte tag, not a length-prefixed protocol.** `kind = message[0]`: `1`=audio (Opus), `2`=text (utf-8). Empty messages are dropped. Must stay byte-identical to the server (`server.py:148/160`); any third kind is silently warned and ignored.
- **One frame per output callback, no jitter buffer.** `_on_audio_output` pulls exactly one 1920-sample frame non-blocking; an empty queue produces a full 80 ms of silence + a `[LAG]` marker rather than stretching/concealing. Under-buffering is audible. The `TODO` at `client.py:130` flags the missing ring buffer.
- **Hard shape asserts in the audio callbacks** (`client.py:123,127,131`): the code assumes PortAudio always delivers exactly `(frame_size, channels)` and that every queued PCM chunk is exactly `(frame_size,)`. A device that negotiates a different blocksize will crash the callback thread.
- **1 ms busy-poll loops.** `_queue_loop` and `_decoder_loop` spin on `asyncio.sleep(0.001)` rather than awaiting readiness — fine for a CLI demo, but it is polling, not back-pressured streaming.
- **Capture/playback are fixed at 24 kHz mono** to match Mimi's sample rate; there is no resampling, so the host audio device must support 24 kHz or PortAudio will resample under the hood.
- **No EOAudio / special-token handling here.** Turn boundaries, EOS, and the audio `2048`=EOAudio code are all server-side concepts; the client only ever sees finished Opus audio and decoded text and never reasons about codes or end-of-turn.
- `_lost_connection` is idempotent via the `_done` flag (`client.py:117`); both the uplink and the recv loop funnel terminal errors through it so all three coroutines wind down together.

---

## TR03 · Moshi client utils
**Code:** `TR03` · **Source:** `moshi/client_utils.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
A terminal-presentation helper for the **Moshi** command-line client/server (the vendored Kyutai reference, not LFM2-Audio's own pipeline). It defines the `AnyPrinter = Printer | RawPrinter` abstraction that streamed text tokens and status (LAG / pending-spinner / log lines) are rendered through, plus ANSI `colorize`/`make_log` helpers. It carries **zero tensors and zero model logic** — it is pure stdout/stderr terminal I/O with cursor-control (carriage-return rewrites) for in-place word-wrapping of a streaming token feed. It exists only so the Moshi `client.py`, `server.py`, and `run_inference.py` loops have a uniform sink for incremental decode output.

## How it works
This file is presentation/transport glue; the "forward pass" here is a terminal write loop, not a network. Mechanism in detail:

- **`colorize(text, color)` (`:11`)** wraps a string in a raw SGR escape `\033[{color}m … \033[0m`. `color` is the bare SGR parameter string, e.g. `"31"` (red), `"1;31"` (bold-red), `"1;34"` (bold-blue), `"32"`/`"33"` (green/yellow for the spinner). No 256-color/truecolor; just classic SGR codes.
- **`make_log(level,msg)` (`:17`)** maps `"warning"→[Warn]` bold-red, `"info"→[Info]` bold-blue, `"error"→[Err ]` bold-red, else raises `ValueError`; prefixes the colorized tag + space to `msg`. **`log(level,msg)` (`:29`)** is the module-level convenience that `print`s `make_log(...)` to stdout — used directly by `run_inference.py` (imported as `log` at `run_inference.py:18`).

- **`RawPrinter` (`:34`)** — the dumb sink for non-TTY / piped output. `print_token(token)` (`:42`) writes the raw token to `self.stream` (default `sys.stdout`) and `flush()`es immediately so streaming decode is visible token-by-token with no buffering. `log()` (`:46`) writes `"{Level}: {msg}"` to `err_stream` (default `sys.stderr`) — keeping logs off the token stream so a redirect of stdout captures clean text. `print_header`/`print_lag`/`print_pending` are intentionally near/fully no-ops (`print_lag` emits a red ` [LAG]` to stderr; `print_pending` does nothing). This is the printer chosen when `--no-fancy`/non-interactive.

- **`Printer` (`:127`)** — the fancy TTY sink with in-place line rewriting. Core state is a **`Line`** object (`:72`) holding an ordered `list[LineEntry]` plus `_max_line_length` (the widest the line ever got) and a `_has_padding` flag. Key mechanics:
  - **In-place erase via carriage return.** `Line.erase(count)` (`:97`) clears the buffer, writes `"\r"` (cursor to column 0), then re-renders the entries it wants to keep (all but the last `count`). This is how the spinner char and partial words get overwritten without a real terminal-control library — every rewrite is `\r` + re-emit.
  - **Padding to clear stale glyphs.** `Line.flush` (`:119`) and `Line.newline` (`:110`) compute `missing = _max_line_length - len(self)` and pad with spaces so that when the new line is *shorter* than a previously rewritten longer line, the leftover characters are blanked. `flush` sets `_has_padding=True` so the next `_add` knows to `erase(count=0)` (re-render clean) first (`:90`).
  - **`len(Line)` (`:82`)** sums `len(entry.msg)` over entries — it counts **visible characters only**, because `LineEntry.__len__` (`:68`) returns `len(self.msg)` (the un-colorized text), while `render()` (`:62`) emits the ANSI-wrapped form. This split is what keeps the `max_cols` width math correct despite invisible escape bytes.
  - **Word-wrap at `max_cols` (default 80).** `Printer.print_token` (`:149`) first calls `_remove_pending()` to erase any spinner glyph, then `remaining = max_cols - len(self.line)`. If the token fits, just `line.add`. If not, it wraps: (a) if the token starts with a space, lstrip it, pad+`" |"` close the current boxed line, `newline`, open `"| "`, add token; (b) otherwise it walks the existing entries **backwards** looking for the last word boundary (an entry whose `msg` starts with a space) or a colored entry (assumed a `[LAG]` marker) — `erase`s back to it, closes the box, opens a new line, and re-emits the carried-over prefix + token (`:163-190`). This reflows a mid-word break to the previous whitespace, terminal-style.
  - **`print_header` (`:136`)** draws the `-`-rule top border and opens the `"| "` gutter — the `| … |` box the streamed transcript lives in.
  - **`print_pending` (`:205`)** animates a spinner from `["|","/","-","\\"]` cycling color `["32","33","31"]`, advancing `_pending_count` and dividing by 5 to slow it; it sets `_pending_printed=True` so the next `print_token`/`log` erases it via `_remove_pending` (`:142`). This is the "model is thinking / awaiting next frame" indicator.
  - **`Printer.log` (`:193`)** closes the current in-box line (`newline` if non-empty), flushes, then prints the `make_log` line to **stderr** — again segregating logs from the boxed token stream on stdout.

- **`AnyPrinter = Printer | RawPrinter` (`:216`)** is the union type the consumers annotate against; selection is `Printer()` for TTY, `RawPrinter()` otherwise (decided in `client.py:185` / `run_inference.py:93`).

No normalization, attention, RoPE, convolution, quantization, sampling, or streaming-tensor state lives here — those concepts do not apply to this component. The only "streaming state" is the terminal-line buffer (`_line`, `_max_line_length`, `_has_padding`, `_pending_count`, `_pending_printed`).

## Dtypes & shapes
No tensors. All I/O is Python `str` over text streams; numeric state is small Python `int`/`bool` line bookkeeping.

| Input | Output |
|---|---|
| `token: str` (decoded text fragment from the LM stream) | bytes written to `stdout` (with ANSI SGR escapes in `Printer`) |
| `level: str ∈ {warning,info,error}`, `msg: str` | colorized log line to `stderr` |
| spinner / LAG triggers (no payload) | transient glyphs to `stdout`/`stderr`, overwritten via `\r` |
| internal: `_max_line_length:int`, `_pending_count:int`, `_has_padding:bool`, `_pending_printed:bool` | — (line-layout bookkeeping) |

No dtype promotions, no bf16/f32/f64, no int64/u32 — this component never touches model dtypes.

## Wiring
**Upstream (who feeds it):** the Moshi client/server decode loops, which produce **`str` text tokens** (one per LM step) and status events:
- [moshi_client](Moshi-Transport) — imports `AnyPrinter, Printer, RawPrinter` (`client.py:16`); its `recv_loop` calls `printer.print_token(payload.decode())`, `printer.print_pending()`, `printer.print_lag()`, `printer.print_header()`, and `printer.log(...)`. Edge: decoded text `str`.
- [moshi_server](Moshi-Transport) — imports the same printers for server-side logging. Edge: log `str`.
- [moshi_run_inference](Moshi-Transport) — imports `AnyPrinter, Printer, RawPrinter, log` (`run_inference.py:18`); the offline streaming loop emits text via `printer.print_token(text)` and warns `"EOS sampled too early."` / logs timing. Edge: decoded text `str`.

**Downstream (who consumes its output):** the **terminal** (`sys.stdout`/`sys.stderr`) and, transitively, the human operator. There is **no downstream model component** — output leaves the program as terminal bytes. This is a leaf on the transport/presentation side, not part of the tensor graph that flows through [core_processor](CO01-Processor-ChatState) → [model_lfm2_audio](MD01-LFM2AudioModel) → [core_detokenizer](CO02-Detokenizer).

## Python ↔ Rust
**No Rust counterpart exists, by design.** Per `PYTHON_VS_RUST.md` §4 ("Out of scope / reused, not ported"), the vendored `liquid_audio/moshi/**` CLI/demo surface is **reused as Kyutai's `moshi` crate, not re-ported**, and the `compare_symbols.py --scope core` audit (170/170) **excludes** `moshi/` exactly so these terminal helpers don't count against parity. `liquid-audio-rs` is a library + parity examples; it never ships the interactive Moshi CLI, so there is nothing to map `colorize`/`Printer`/`RawPrinter` onto. The `Rust:` field for this component is `-`.

This is **not a divergence/bug** — it is the deliberate "use what exists; extend, don't fork" stance in §2.3. If a Rust CLI ever needed equivalent in-place streaming output, it would lean on a crate like `crossterm`/`indicatif` rather than re-implementing the `\r`-rewrite logic; none of that is on the LFM2-Audio inference path.

## Precision / gotchas
- **No numerical concerns at all** — there is no float reduction, no RMSNorm order, no FFT, no EOAudio/special-token handling here. The global dtype facts (bf16 weights, f64 mel, int64 ids, u32 codes, the cross-library f32 floor) are **irrelevant** to this file; do not look for them here.
- The one correctness subtlety is the **visible-length vs rendered-length split**: `LineEntry.__len__`/`Line.__len__` count un-colorized characters while `render()` emits ANSI bytes (`:62-68`, `:82-83`). All `max_cols` wrap math depends on this; if a future edit made `__len__` count the escape bytes, every wrap/erase column would be wrong.
- **stdout vs stderr segregation is intentional**: tokens and the box go to `stdout`; all `log()` output and `[LAG]` go to `stderr`, so piping stdout yields a clean transcript. `RawPrinter` exists precisely to give that clean, escape-free stream for non-TTY consumers.
- `make_log` raises `ValueError` on an unknown level (`:25`) — the only hard failure path in the module.
- The wrap backtrack in `Printer.print_token` treats any **colored** trailing entry as a `[LAG]` marker (`:168` comment) and breaks the scan there — an implicit coupling between the LAG-coloring convention and the wrap heuristic; a differently-colored token mid-line could change wrap behavior. Cosmetic only.

---

## TR04 · Moshi run_inference (offline loop)
**Code:** `TR04` · **Source:** `moshi/run_inference.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
The vendored Kyutai **offline** streaming-inference CLI for the **Moshi / Hibiki / STT** family — a *different* model from LFM2-Audio. It reads one audio file, chops it into fixed Mimi frames, feeds the codes through `LMGen.step` at a constant 12.5 Hz cadence, and emits per-item `(text_tokens, audio_tokens)`, optionally vocoding the audio codes back to a wav. It is **off the LFM2-Audio path**: LFM2-Audio uses its own backbone + depthformer (`model_lfm2_audio`) driven by a synchronous streaming generator and `mimi.streaming(1)` decode in `demo/chat.py`, never `LMGen`, never `lm_gen.step`. It lives here only because the whole `moshi/` subtree was vendored wholesale, and it is the conceptual reference for the fixed-frame, EOS-on-frame offline loop.

## How it works
There is **no neural code in this file** — every tensor op is delegated to `MimiModel` (`moshi_compression`) and `LMGen` (`moshi_lm`). The mechanism is a fixed-cadence frame loop plus a per-batch-item turn-end state machine. State lives in the `InferenceState` dataclass; the loop is `InferenceState.run`.

**Construction (`run_inference.py:66-95`).** Stores the Mimi codec, a `sentencepiece` tokenizer, and builds `self.lm_gen = LMGen(lm, cfg_coef, condition_tensors=…, **kwargs)`. Two load-bearing scalars and one mode switch:
- `self.frame_size = int(mimi.sample_rate / mimi.frame_rate)` (`:87`). With `SAMPLE_RATE=24000`, `FRAME_RATE=12.5` → exactly **1920 samples** = one Mimi frame = 80 ms. This is the quantum everything is chunked on.
- `condition_tensors = get_condition_tensors(...)` (`:82`, `:34-57`): only `hibiki` builds real conditions (`text={"description":"very_good"}`, plus a `"very_bad"` negative branch appended when `cfg_coef != 1.0` for classifier-free guidance); any other model with a conditioner `raise`s. Moshi/STT pass through with `{}`.
- Both stateful modules go into permanent streaming mode: `mimi.streaming_forever(batch_size)` + `lm_gen.streaming_forever(batch_size)` (`:89-90`). This installs the persistent KV-cache / conv-ring-buffer streaming state on the codec transformers and the LM, so each `.encode`/`.step`/`.decode` is incremental, not a full re-encode.

**STT-only input padding (`run_inference.py:121-127`).** For `model_type == "stt"` it pads the raw PCM with `audio_silence_prefix_seconds` of left silence and `(audio_delay_seconds + 1.0)` of right silence (both × 24000 samples, `mode="constant"`). This bakes the STT alignment delay into the input stream. Moshi/Hibiki skip this.

**Frame deque (`run_inference.py:128-135`).** `in_pcms.split(frame_size, dim=2)` then keeps **only fully-sized frames** (`chunk.shape[-1] == frame_size`) — any trailing partial frame is dropped (no zero-pad of the last frame). The kept frames become a `collections.deque`, popped left-to-right.

**The main loop (`run_inference.py:138-202`), `while not all(eos_reached)`.** Per iteration:
1. **Source a frame of codes.**
   - If the deque is non-empty: `chunk = chunks.popleft()`, `codes = mimi.encode(chunk)` → int codes `(B, n_q, T)` (`:140-141`).
   - Else, end-of-file behavior is model-specific (`:142-163`):
     - **hibiki**: the *first* post-EOF frame feeds an explicit end-of-stream marker — a code tensor filled with `mimi.cardinality` (= **2048**) on *all* codebooks, shape `(B, num_codebooks, 1)`, `dtype=long` (`:144-154`); subsequent post-EOF frames encode `frame_size` of silence (`:155-160`). This lets the model keep generating its translation tail after the input ends, until it emits text-EOS.
     - **other models (moshi/stt)**: `break` immediately at EOF (`:161-163`).
2. **First-frame priming (`run_inference.py:164-170`).** On the very first frame it calls `lm_gen.step(codes)` an *extra* time and discards the result; if `max(delays) > 0` that priming step must return `None` (asserted). Rationale (comment `:165-166`): without it the first real slice of codes would be overwritten by `LMGen`'s initial-token bootstrap, so the model never "sees" frame 0.
3. **The LM step (`run_inference.py:171-174`).** `tokens = lm_gen.step(codes)`. `LMGen` internally applies the per-codebook **acoustic-delay** pattern (`_delay_sequence` / `_undelay_sequence`, `lm.py:344-369`), runs the Moshi backbone + depformer, and **returns `None` until the delay warmup is satisfied** (`continue` on `None`). Once warm it returns `(B, dep_q+1, 1)` — index 0 is the **text** (inner-monologue) stream, indices `1..dep_q+1` are the `dep_q` audio codebooks. `assert tokens.shape[1] == dep_q + 1` (`:174`).
4. **Decode + per-item turn-end (`run_inference.py:175-201`), when `dep_q > 0`.**
   - `out_pcm = mimi.decode(tokens[:, 1:]).cpu()` — drop the text row, vocode the audio codes → `(B,1,frame_size)` f32 @ 24 kHz.
   - Per batch item `b`: if already `eos_reached[b]`, skip. Else if the text token equals `text_tokenizer.eos_id()` → mark `eos_reached[b]=True` (but warn "EOS sampled too early" if `need_eos_input` is still set, i.e. the model emitted EOS before the input file even ended — `:182-187`). Append the text token and the decoded pcm to the per-item accumulators.
   - For `b == 0` only, live-print: skip ids `0` (pad) and `3` (epad/special), else `id_to_piece` → replace SentencePiece `▁` with space → `printer.print_token` (`:191-195`).
   - **Text-only models (`dep_q == 0`, `run_inference.py:196-201`)**: no Mimi decode; just print `tokens[0,0]` with the same `0`/`3` skip.

**Sampling.** This file does not sample — `use_sampling`, `temp` (audio temperature), `temp_text` (text temperature) are owned by `LMGen` and only *logged* here (`:113-115`). The actual top-k/top-p multinomial lives in `moshi_util_sampling`.

**Output assembly (`run_inference.py:208-217`).** Per item, `torch.cat(one_texts, dim=0)` (text ids along time) and `torch.cat(one_pcms, dim=1)` (waveform along time) → `list[(text_tokens, audio_tokens)]`. `main()` then writes each item's pcm to a wav via `sphn.write_wav` at `mimi.sample_rate` (`:304-315`).

**`main()` plumbing (`run_inference.py:220-315`).** argparse, `seed_all(4242)` (`:23-31`: sets torch/cuda/python/numpy seeds, `cudnn.deterministic=False`), `CheckpointInfo.from_hf_repo` (→ `moshi_loaders`), `get_mimi`/`get_text_tokenizer`/`get_moshi`, `--device` defaults **cuda**, `--half` toggles bf16→fp16 (`dtype` default `torch.bfloat16`, `:245-252`), `--cfg-coef` CFG coefficient, `--batch-size` default 8 (forced to 1 when `dep_q == 0`). Input read with `sphn.read(infile, sample_rate=mimi.sample_rate)` → f32, then `in_pcms[None, 0:1].expand(batch_size, -1, -1)` broadcasts the **same** mono clip across the batch. Whole program runs under `with torch.no_grad()` (`:318-320`).

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| `sphn.read(infile)` | wav file | f32 PCM `(C, N)` @ 24 kHz |
| batch expand (`:290`) | f32 `(1,1,N)` | f32 `(B,1,N)` (same clip broadcast) |
| frame slice (deque) | f32 `(B,1,N)` | f32 frames `(B,1,1920)` each (partial dropped) |
| `mimi.encode` (`:141`) | f32 `(B,1,1920)` | int codes `(B, n_q, T)` (u32 in Rust); EOF marker = `(B,n_q,1)` filled `2048`, int64 |
| `lm_gen.step` (`:171`) | int codes `(B, n_q, 1)` | `None` during delay warmup, else int `(B, dep_q+1, 1)` |
| text row | `tokens[:,0]` int64 | SentencePiece piece (skip ids 0, 3; stop on `eos_id()`) |
| `mimi.decode(tokens[:,1:])` (`:176`) | int codes `(B, dep_q, 1)` | f32 waveform `(B,1,1920)` @ 24 kHz |
| `run` return | — | `list[(text_tokens int64 (Ttok,), audio_tokens f32 (1, Nsamp))]` |

Notes: no f32-upcast norm/softmax/mel happens *in this file* (all inside `moshi_compression`/`moshi_lm`); EOF marker code value `2048` = `mimi.cardinality` = the per-codebook EOAudio/end-of-stream sentinel on this codec.

## Wiring
**Upstream (feeds this):**
- [moshi_loaders](MM02-Mimi-Loaders) — `CheckpointInfo.from_hf_repo` → Mimi codec, Moshi LM, `sentencepiece` tokenizer, `lm_gen_config`.
- [MimiModel](MM01-Mimi-Codec) — `mimi.encode` produces the int code frames `(B, n_q, 1920→T)` that drive each step; `mimi.decode` vocodes the audio codes back to f32 @ 24 kHz. Edge in: f32 `(B,1,1920)`; edge out (decode): f32 `(B,1,1920)`.
- [moshi_lm / LMGen](MM03-Moshi-LM) — `LMGen.step` consumes int codes `(B, n_q, 1)` and returns the `(B, dep_q+1, 1)` text+audio token frame. The delay/undelay + depformer math lives there.

**Downstream (consumes this output):**
- Terminal printer ([moshi_client_utils](Moshi-Transport)) — `Printer`/`RawPrinter.print_token` renders the text stream live (decoded pieces, skipping ids 0/3).
- `sphn.write_wav` (external) — the accumulated per-item f32 audio `(1, Nsamp)` @ 24 kHz is written to disk in `main()`.
- No LFM2-Audio component consumes this — it is a leaf CLI, not part of the LFM2 graph ([model_lfm2_audio](MD01-LFM2AudioModel), [demo_chat](DM01-Realtime-Chat) are the on-path analogues).

## Python ↔ Rust
**Not ported.** `Rust: -`. The `liquid-audio-rs` `core` scope excludes the vendored `moshi/` subtree by design (PYTHON_VS_RUST.md §4: "vendored `liquid_audio/moshi/**` is reused as the `moshi` crate … not re-ported"). The Rust port reuses Kyutai's **`moshi` crate** for the Mimi codec only (PYTHON_VS_RUST.md §2.3) and never reconstructs `LMGen`, this CLI, or the Moshi 7B LM — LFM2-Audio's Rust path is `model_lfm2_audio` + its own depthformer + `demo/chat.py`-shaped synchronous streaming, not this fixed-cadence `lm_gen.step` loop. The closest Rust analogue in spirit (offline frame loop driving the model) is `examples/generate.rs`, which drives the **LFM2** model, not Moshi.

## Precision / gotchas
- **Off-path.** Nothing in the LFM2-Audio inference graph calls this; do not treat `dep_q+1` text+audio interleaving or `LMGen` delays as the LFM2 contract. LFM2-Audio's depthformer emits an `(8,)` audio frame with `2048 = EOAudio` per codebook and a separate text head — a different head/cadence than Moshi's `dep_q+1` row.
- **Cardinality sentinel `2048`.** `mimi.cardinality` is reused both as the per-codebook EOAudio/end-of-stream marker (hibiki EOF, `:148-154`) and as the count of valid code values — code ids run `0..2048` with `2048` reserved. Same numeric sentinel as LFM2-Audio's EOAudio, different model.
- **Partial-frame truncation (`:131-134`).** The trailing sub-1920-sample remainder of the input is silently dropped, not zero-padded — output is quantized to whole 80 ms frames.
- **Same clip across the batch (`:290`).** `expand(batch_size,…)` broadcasts one mono channel; the batch is *not* independent clips — it is the same audio replicated, so per-item `eos_reached` divergence is purely from stochastic sampling, not different inputs.
- **First-frame double-step (`:164-170`).** The discarded priming `lm_gen.step` is required; skipping it makes the model miss frame 0. With non-zero delays the primer must return `None` (asserted).
- **"EOS sampled too early" (`:182-186`).** A warning, not a stop: if text-EOS arrives while `need_eos_input` is still true (input not yet exhausted), the item is *not* marked done — it logs and keeps going, because EOS before EOF is considered impossible/anomalous for hibiki.
- **CUDA-coupled.** `--device` defaults `cuda`; `seed_all` touches `torch.cuda` guarded by `is_available()`. No device-agnostic path here (contrast PYTHON_VS_RUST.md §2.1 for the LFM2 Rust port's CPU-first design).

---

## TR05 · Moshi run_tts
**Code:** `TR05` · **Source:** `moshi/run_tts.py` · **Rust:** `-` (not ported) · **On the LFM2-Audio inference path:** no

## Role
A standalone offline CLI batch driver for **Kyutai's Moshi/DSM TTS** model (`kyutai/tts-1.6b-en_fr`) — a *different* model from LFM2-Audio. It reads a JSONL file of `{turns, voices, id}` requests, runs Delayed-Streams-Modeling (DSM) text-to-speech in fixed-size batches, decodes the generated Mimi audio codes to a waveform, and writes `.wav` (plus optional `.safetensors` raw frames and `.json` debug info) per request. It exists in this tree only because the whole `moshi/` subtree was vendored wholesale; nothing in the LFM2-Audio pipeline imports or calls it.

## How it works
There is **no neural code in this file** — every tensor op is delegated to `TTSModel` (`moshi_tts`), which wraps a Moshi `LMModel` + `LMGen` + the `MimiModel` codec (`moshi_compression`). This file is the argument-parsing, batching, decode-loop, and file-IO shell around that.

**CLI surface (`run_tts.py:39-79`).** `argparse` exposes: `--hf-repo`/`--voice-repo` (default `DEFAULT_DSM_TTS_REPO`/`DEFAULT_DSM_TTS_VOICE_REPO`), local-checkpoint overrides (`--config`, `--tokenizer`, `--mimi-weight`, `--moshi-weight`), and the generation knobs `--batch-size 32`, `--nq 32` (codebooks to generate), `--temp 0.6`, `--cfg-coef 2.0`, plus the DSM padding/alignment controls `--max-padding 8`, `--initial-padding 2`, `--final-padding 4`, `--padding-bonus 0.`, `--padding-between 1`. `--device` defaults `"cuda"`; `--half` flips the generation dtype from `torch.bfloat16` to `torch.float16` (the `dest="dtype"` `store_const` at `:71-72`). `--only-wav` suppresses the debug artifacts.

**Model construction (`run_tts.py:84-101`).** `CheckpointInfo.from_hf_repo(...)` (→ `moshi_loaders`) resolves the Moshi LM + Mimi + tokenizer; `TTSModel.from_checkpoint_info(...)` builds the TTS state machine with all the padding/CFG knobs. **CFG-distillation branch (`:92-100`):** if `tts_model.valid_cfg_conditionings` (the model was trained with classifier-free-guidance *distillation*), the requested `--cfg-coef` is moved into a *conditioning* attribute `cfg_coef_conditioning` and the runtime `tts_model.cfg_coef` is forced to `1.` (no second forward pass), with `cfg_is_no_text=cfg_is_no_prefix=False`. Otherwise classic CFG is used: `cfg_coef_conditioning=None` and `cfg_is_no_text=cfg_is_no_prefix=True` (the unconditional branch drops text + prefix). This selects between *distilled single-pass* CFG and *classic two-pass* CFG.

**Batching driver (`run_tts.py:197-209`).** Reads the JSONL line-by-line into `batch: list[TTSRequest]`; whenever `len(batch) >= --batch-size` it calls `_flush()`, and a final `_flush()` drains the remainder. `TTSRequest` is `{turns: list[str], voices: list[str], id: str}` (`:21-37`); `turns[0]`/`voices[0]` is the MAIN speaker, later entries are additional turns/speakers.

**`_flush()` — the actual work (`run_tts.py:103-195`).**
1. **Script + conditioning prep (`:111-122`).** Per request: `entries = tts_model.prepare_script(request.turns, padding_between=…)` tokenizes the text into DSM `Entry` words (→ `moshi_tts`). `make_condition_attributes(voices, cfg_coef_conditioning)` builds the speaker/voice + CFG conditioning. The model is either **multi-speaker** (voices passed as conditioning, no prefix) or **single-speaker with a voice prefix**: when `not tts_model.multi_speaker`, exactly one voice is allowed and `get_prefix(get_voice_path(voice))` loads a pre-computed audio prefix that primes the LM (`:119-122`).
2. **Generation (`:125-133`).** `tts_model.generate(all_entries, all_attributes, prefixes=…, cfg_is_no_prefix=…, cfg_is_no_text=…)` runs the DSM state machine: it co-generates the *time-aligned padded text stream* and the *audio codebook stream*, the LM signalling when the next step begins a new word so a word can be "popped" and fed over the following steps (the DSM alignment trick — see the `moshi_tts` module docstring). `result.frames` is a list of per-step token frames; `result.end_steps[idx]` marks where each item's speech ended; `result.all_transcripts`, `all_consumption_times`, `logged_text_tokens` are debug telemetry. `frames = torch.cat(result.frames, dim=-1).cpu()` is the full `(B, 1+nq, T_steps)` token tensor (row 0 = text stream, rows `1..nq` = audio codebooks). Throughput is reported as `total_duration / time_taken` (the "x realtime" speed).
3. **Streaming Mimi decode (`:135-140`).** Under `torch.no_grad()` and `tts_model.mimi.streaming(len(all_entries))` (batched streaming codec state): for each frame *after* the `tts_model.delay_steps` warmup, `wav_frames.append(tts_model.mimi.decode(frame[:, 1:]))`. `frame[:, 1:]` **drops the text row**, passing only the audio codebooks to the codec. Frames are decoded one at a time (the comment notes they could be grouped for speed). `wavs = torch.cat(wav_frames, dim=-1)` is `(B, 1, total_samples)` f32 @ `mimi.sample_rate` (24 kHz).
4. **Per-item trim + write (`:141-184`).** For each request, the usable length is `wav_length = int(mimi.sample_rate * (end_step + final_padding) / mimi.frame_rate)` (`:148`) — converting *frame steps* to *samples* via the 12.5 Hz↔24 kHz ratio (1920 samples/frame). If `end_step is None`, generation failed and the whole buffer is kept with a warning (`:144-146`). For single-speaker prefix models, the prefix region is sliced off the front: `start = int(mimi.sample_rate * prefixes[idx].shape[-1] / mimi.frame_rate)` (`:152-155`). The waveform is `.clamp(-1, 1)` and written with `sphn.write_wav(filename, wav.numpy(), mimi.sample_rate)`. Unless `--only-wav`, `frames[idx].short()` (int16) is saved as `.safetensors` and a `.json` of all generation knobs + transcript + timing is written (`:161-183`).
5. **Telemetry (`:185-195`).** Reports `total speed` (assuming all batch items equal length) vs `effective speed` (only counting usable audio, so early-finishing items that wasted compute don't inflate the number), then `batch.clear()`.

**No model math here.** Normalization, attention, RoPE, RVQ, convolutions, sampling — all of it lives in the delegated modules (`moshi_tts`, `moshi_lm`, `moshi_compression`, `moshi_vq`). This component is the offline batch/decode/IO state machine only; its one numerically meaningful operation is the **step→sample arithmetic** (`sample_rate * step / frame_rate`) and the final `clamp(-1,1)`.

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| JSONL line | `str` | `TTSRequest{turns:list[str], voices:list[str], id:str}` |
| `prepare_script(turns)` | `list[str]` | DSM `list[Entry]` (text token ids, int64) |
| `make_condition_attributes(voices)` | voice paths / cfg coef | `ConditionAttributes` |
| `get_prefix(path)` (single-speaker) | voice file | audio-prefix token tensor `(…, P_steps)` int |
| `tts_model.generate(...)` | entries + attrs + prefixes | `result.frames`: list of int frames, each `(B, 1+nq, 1)`; `end_steps: list[int|None]` |
| `torch.cat(result.frames, -1)` | per-step int frames | `frames` int `(B, 1+nq, T_steps)` (row 0 = text, rows 1..nq = audio codes) |
| `mimi.decode(frame[:, 1:])` | int audio codes `(B, nq, 1)` (u32 in Rust analog) | f32 waveform `(B, 1, 1920)` @ 24 kHz |
| `torch.cat(wav_frames, -1)` | per-frame waveforms | f32 `(B, 1, total_samples)` |
| trim → `clamp(-1,1)` → write | f32 `(1, wav_length)` | `.wav` (24 kHz) |
| debug | `frames[idx].short()` | int16 `.safetensors` + `.json` |

Generation dtype is **bf16** by default (`--half`⇒fp16); model weights are bf16 on disk. Mimi `decode` runs in the codec's model dtype and produces **f32** PCM. Token frames are integer code ids throughout (text token ids are int64; audio codebook indices are small ints). The `step→sample` conversions use Python `int()` truncation, not rounding.

## Wiring
**Off the LFM2-Audio tensor path** — this is a self-contained Moshi/DSM-TTS batch tool. Its neighbors are all Moshi-stack components:
- **Upstream:** a JSONL file of `TTSRequest`s (text + voice names), plus the checkpoint resolved by [moshi_loaders](MM02-Mimi-Loaders) (`CheckpointInfo.from_hf_repo` → Mimi + Moshi LM + tokenizer). Voice/prefix tensors come from the `--voice-repo`.
- **Core compute:** the request `turns`/`voices` → [moshi_tts](MM05-Moshi-TTS) `TTSModel.prepare_script` / `make_condition_attributes` / `generate`, which internally drives [moshi_lm](MM03-Moshi-LM) `LMModel`/`LMGen` to co-generate the int token frames `(B, 1+nq, T)`.
- **Downstream (consumer of this component's output):** the audio rows of each frame `frame[:, 1:]` int `(B, nq, 1)` → [moshi_compression](MM01-Mimi-Codec) `MimiModel.decode` → f32 waveform `(B,1,1920)` @ 24 kHz, then `sphn.write_wav` to disk. There is **no in-process downstream component** — the terminal sink is the `.wav`/`.safetensors`/`.json` files; nothing in this codebase reads them back.

None of [core_processor](CO01-Processor-ChatState), [model_lfm2_audio](MD01-LFM2AudioModel), or [core_detokenizer](CO02-Detokenizer) feed or consume this file — the LFM2-Audio path is entirely separate.

## Python ↔ Rust
**Not ported.** `liquid-audio-rs` ships no Moshi/DSM-TTS driver. Per PYTHON_VS_RUST.md §4, the vendored `liquid_audio/moshi/**` is *reused as the `moshi` crate* (Kyutai's own Rust port) rather than re-ported, and `compare_symbols.py`'s `core` scope excludes the `moshi/` subtree by design — so `run_tts.py`, `TTSRequest`, `main`, and `_flush` have **no Rust referent**. The only piece with a Rust analog is the codec it calls: `tts_model.mimi.streaming(...)` / `mimi.decode(...)` map to `moshi::mimi::Mimi` (used by `audio_out.rs::MimiDetokenizer`), but that path is driven by LFM2-Audio's own loops (`mic_chat.rs` / `generate_interleaved`), never by a TTS batch script. Deliberate divergences inherited from the reused Mimi crate — eager SDPA vs flash-attention, candle ops vs custom CUDA kernels, CUDAGraph/compile disabled off-cuda — are catalogued in PYTHON_VS_RUST.md §2.2 / §2.3. The `safetensors`/`sphn` IO and `argparse` CLI have no port at all.

## Precision / gotchas
- **This is Moshi/DSM-TTS, not LFM2-Audio.** Different model, different head (Moshi depformer vs the LFM2 depthformer), different sampler, and a *delayed-streams* alignment machine that LFM2-Audio does not use. Anyone tracing the LFM2-Audio audio-out should ignore this file and follow [core_detokenizer](CO02-Detokenizer) (the LFM2 ISTFT vocoder) instead.
- **`frame[:, 1:]` drops the text row before decode (`:139`).** Row 0 of every frame is the inner-monologue/text stream; only rows `1..nq` are the audio codebooks Mimi understands. Passing the full frame to `mimi.decode` would misinterpret the text id as a codebook index.
- **`delay_steps` warmup is skipped, not an error (`:137`).** The DSM acoustic-delay pattern means the first `tts_model.delay_steps` frames carry no usable audio; the decode loop starts at `result.frames[tts_model.delay_steps:]`. Decoding from frame 0 would emit garbage prefix audio.
- **`end_step is None` ⇒ generation failed (`:144-146`).** That item's `wav_length` falls back to the full buffer with a printed warning; the `.wav` will contain the untrimmed (possibly runaway) generation. The expected path uses `int(sample_rate*(end_step+final_padding)/frame_rate)` to trim to real speech length.
- **Single-speaker prefix trim (`:151-155`).** For non-multi-speaker models the leading prefix audio (used only to prime the voice) is sliced off via `prefixes[idx].shape[-1]` steps → samples; forgetting this would leak the conditioning prefix into the output. Multi-speaker models pass voices as conditioning and have no prefix region.
- **CFG mode is checkpoint-dependent (`:92-100`).** A CFG-*distilled* model runs single-pass with the coefficient folded into conditioning and `cfg_coef` forced to `1.`; a non-distilled model runs classic two-pass CFG (`cfg_is_no_text`/`cfg_is_no_prefix=True`). Hard-coding either breaks the other.
- **`int()` truncation on step→sample conversion (`:148,:154`).** Both the end trim and the prefix start use truncating `int()`, not rounding — a sub-sample bias by design, consistent with Mimi's integer 1920 samples/frame (`sample_rate/frame_rate = 24000/12.5`).
- **CUDA-coupled defaults.** `--device` defaults `cuda` and the speedups assume CUDA-graph/compile; off-cuda it runs but the graphing no-ops (`moshi_util_compile`). This file is not part of the device-agnostic LFM2-Audio surface.
- **`--nq 32` vs Mimi's 8 codebooks.** The TTS LM can generate up to 32 codebooks; `frame[:, 1:]` forwards however many were generated to `mimi.decode`, whose split-RVQ (`moshi_vq`) consumes the codebook stack. The terminal waveform is the Mimi reconstruction, `clamp(-1,1)`-ed before write to guard against codec overshoot.

---

## TR06 · Moshi gradio client
**Code:** `TR06` · **Source:** `moshi/client_gradio.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
A browser-based, WebRTC voice client for the **Moshi** websocket server (`moshi/server.py`). It is a thin transport-and-UI shell — no model, no codec, no tensors of its own — that bridges a `gradio-webrtc` audio component to the server's `/api/chat` websocket using Opus-compressed PCM in both directions. It exists purely as a deployable web demo (Heroku/Spaces-friendly via `rtc_configuration`). It is **vendored Moshi code, off the LFM2-Audio path**: LFM2-Audio's own realtime UI is `demo/chat.py` (fastrtc `ReplyOnPause`), and this gradio client is not ported to Rust.

## How it works
The whole client is one `gradio-webrtc` `StreamHandler` subclass, `MoshiHandler` (`client_gradio.py:21`), plus a `gr.Blocks` wiring `main()` (`:113`). There is no neural network here — the mechanism is **codec framing + websocket multiplexing + an output-rate reblocking buffer**, driven by gradio-webrtc's callback contract (`receive` for mic-in, `emit` for speaker-out, `copy` to clone per-connection, `shutdown` to tear down).

**URL → websocket scheme normalization** (`:29-39`). The constructor splits `url` on `"://"`, maps `ws`/`http`→`ws` and `wss`/`https`→`wss`, and targets the path `/api/chat` (the Moshi server route). The websocket itself is **lazily** opened on the first `receive` (`:51-52`), using the *synchronous* `websockets.sync.client` — gradio-webrtc runs the handler callbacks on worker threads, so a blocking sync socket is correct here (contrast the server's asyncio coroutines, [server.md](Moshi-Transport)).

**Opus framing (both directions) via `sphn`.** `sphn.OpusStreamWriter`/`OpusStreamReader` are constructed at `output_sample_rate` (24000) (`:40-41`). These are streaming Opus (re)encoders: you push raw PCM in and pull encoded byte-frames out incrementally; the reader is the inverse. This is the same `sphn` codec layer used by `moshi/client.py` ([client.md](Moshi-Transport)) and the server ([server.md](Moshi-Transport)).

**Mic-in path — `receive(frame)` (`:50-57`).** gradio-webrtc hands a `(sample_rate, NDArray)` tuple. The handler:
1. `array.squeeze().astype(np.float32) / 32768.0` (`:54`) — collapses channel dims and converts **int16-scaled** samples to **float32 in [-1,1)** by dividing by 32768. (Input is declared mono, `input_sample_rate=24000` at `:47`.)
2. `stream_writer.append_pcm(array)` (`:55`) feeds the f32 PCM into the Opus encoder.
3. `b"\x01" + self.stream_writer.read_bytes()` (`:56`) — drains whatever encoded Opus bytes are ready and **prepends a one-byte kind tag `0x01` (audio)**. This 1-byte-tag framing is the Moshi websocket wire protocol: `0x01`=audio, `0x02`=text. The tagged blob is sent over the websocket (`:57`). Note the writer can return an empty byte-string when no full Opus frame is ready yet; the message is still sent with just the tag.

**Speaker-out path — `generator()` + `emit()` (`:59-94`).** `emit()` is gradio-webrtc's pull callback (called repeatedly to get the next output chunk). It lazily instantiates a Python generator over the websocket (`:89-90`) and returns `next(...)` each call; on `StopIteration` it `reset()`s (`:93-94`) so the generator is rebuilt on reconnect. The generator (`:59`) iterates inbound websocket messages and dispatches on the **first byte (kind tag)**:
- **`kind == 1` (audio, `:66-81`)**: strip the tag (`payload = message[1:]`), `stream_reader.append_bytes(payload)` then `stream_reader.read_pcm()` → decoded f32 PCM. This decoded PCM is appended to a running `self.all_output_data` numpy buffer (`:70-73`). Then a **reblocking loop** (`:74-81`) emits fixed-size chunks: while the buffer holds ≥ `output_chunk_size` (**1920** samples, `:37`) it `yield`s `(output_sample_rate, chunk.reshape(1,-1))` and slices that prefix off the buffer. This decouples the Opus/Mimi frame size from gradio-webrtc's `output_frame_size` (480) — i.e. it accumulates and re-chops into 1920-sample mono frames at 24 kHz. (1920 samples @ 24 kHz = the Mimi frame = 80 ms.)
- **`kind == 2` (text, `:82-84`)**: the payload is UTF-8 text (the model's inner-monologue tokens streamed as bytes by the server); it is `yield`ed as `AdditionalOutputs(payload.decode())`, gradio-webrtc's side-channel for non-audio outputs.
- **empty message (`:63-64`)**: `yield None` (idle/keepalive — no audio this tick).

**UI wiring — `main()` (`:113-159`).** Builds a `gr.Blocks` with a `gr.Chatbot` and a `WebRTC` component in `mode="send-receive"`, `modality="audio"`. `webrtc.stream(MoshiHandler(args.url), …, time_limit=90)` (`:138-143`) binds the handler with a **90-second per-conversation cap** (also stated in the page HTML). `webrtc.on_additional_outputs(add_text, …)` (`:151-157`) routes the `kind==2` text deltas into the chatbot: `add_text` (`:145-149`) appends to the last assistant message's `content`, i.e. **token-streaming concatenation** of the transcript. `copy()` (`:100-106`) clones the handler per WebRTC peer (fresh sockets/buffers/Opus streams per connection — required because each peer needs isolated streaming state). `shutdown()` (`:108-110`) closes the websocket.

No sampling, normalization, attention, RoPE, convolution, or quantization happens in this file — all of that lives server-side in the Moshi LM ([../moshi/models/lm.md] equivalents) and the Mimi codec. This component only moves Opus-framed audio and text tags across the wire and reblocks the decoded PCM.

## Dtypes & shapes
| Stage | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `receive` mic frame (`:50-54`) | `(int sr, NDArray int16-scaled)` → squeezed | **f32** PCM `(N,)` in [-1,1) after `/32768` |
| `append_pcm` → `read_bytes` (`:55-56`) | f32 PCM `(N,)` | Opus bytes, tagged `b"\x01"+bytes` |
| websocket inbound audio (`:67-69`) | tagged bytes `0x01‖opus` | **f32** PCM `(M,)` from `read_pcm()` |
| reblock buffer `all_output_data` (`:70-81`) | f32 PCM `(M,)` accumulated | f32 chunk `(1, 1920)` @ 24 kHz, yielded |
| websocket inbound text (`:82-84`) | tagged bytes `0x02‖utf8` | Python `str` (in `AdditionalOutputs`) |
| `add_text` (`:145-149`) | `str` delta | chat history `list[{role,content}]` |

No model dtypes (bf16/f32 weights, int64 ids, u32 codes) appear here — this transport sees only **f32 audio PCM @ 24 kHz** and **UTF-8 text bytes**. The single numeric promotion is the **int16→f32 `/32768.0`** mic conversion (`:54`); decoded output PCM is already f32 from `sphn`.

## Wiring
**Upstream (what feeds this):**
- **Mic / WebRTC peer** → `receive()` as `(sr, int16-scaled NDArray)`; converted to f32 PCM. This is browser-side audio, no in-repo md neighbor.
- **Moshi websocket server** ([server.md](Moshi-Transport)) → inbound messages on `/api/chat`: tagged **Opus audio (`0x01`)** decoded to f32 PCM `(M,)` @ 24 kHz, and **text (`0x02`)** as UTF-8 bytes. The server's `send_loop` produces exactly these tagged frames; this client is the symmetric peer of `moshi/client.py` ([client.md](Moshi-Transport)) but with a WebRTC/gradio front-end instead of `sounddevice`.

**Downstream (what consumes this output):**
- **Moshi websocket server** ([server.md](Moshi-Transport)) ← outbound tagged Opus audio (`b"\x01"+opus`) from `receive()`; the server's `recv_loop` decodes it and feeds Mimi-encoded codes into `lm_gen.step`.
- **Speaker / WebRTC peer** ← f32 `(1,1920)` @ 24 kHz audio chunks from `emit()`.
- **`gr.Chatbot` UI** ← streamed text via `AdditionalOutputs` → `add_text` (`:145`).

This component does **not** touch the LFM2-Audio core ([../model/lfm2_audio.md], [../processor.md], [../detokenizer.md]) at all — it is a Moshi-server transport. The Mimi codec ([models/compression.md](MM01-Mimi-Codec)) is relevant only on the *server* side of the wire.

## Python ↔ Rust
**No Rust port.** `client_gradio.py` is in the vendored `liquid_audio/moshi/**`, which `liquid-audio-rs` **reuses as the `moshi` crate (Kyutai's own port) rather than re-porting** (PYTHON_VS_RUST.md §4 "Out of scope / reused"; PORT_STATUS.md: the `moshi/*` row → "♻ reuse the `moshi` crate"). `compare_symbols.py`'s `core` scope **excludes** `moshi/` by design, so there is no symbol-level mapping for this file.

Closest Rust-side analog is **not** a port of this gradio client but the LFM2-Audio Rust transport choice: per ARCHAEOLOGY.md Q4, the Rust ports **only the turn-based demo shape** (`mic_chat.rs`: synchronous `generate_interleaved` on main + `cpal` callback threads + `Arc<Mutex>` rings, **no async, no websocket, no tokio**). The genuinely full-duplex Moshi **asyncio/websocket** stack — `server.py`, `client.py`, **and this `client_gradio.py`** — is **deliberately unported** (PYTHON_VS_RUST.md §2.1 device-agnostic + ARCHAEOLOGY.md "Honest gaps" #1). So the divergence here is *category*, not *op-level*: a WebRTC browser transport has no candle/Rust referent in this project.

## Precision / gotchas
- **int16 scale, not int16 dtype.** `array / 32768.0` (`:54`) assumes the incoming frame is **int16-valued** (range ±32768). gradio-webrtc delivers int16-scaled audio; dividing by `2^15` maps to f32 [-1,1). Feeding already-normalized f32 here would silence the signal by ~32768×.
- **Wire protocol = 1-byte kind tag.** `0x01`=audio, `0x02`=text, prepended on send (`:56`) and consumed as `message[0]` on receive (`:65`). This must stay in lockstep with the server's framing ([server.md](Moshi-Transport)); a tag mismatch routes audio into the text path or vice-versa.
- **Empty-frame handling is asymmetric.** On receive, a zero-length message yields `None` (`:63-64`) — but the code then **still reads `message[0]`** (`:65`) right after, which would `IndexError` on a truly empty `message`; in practice gradio-webrtc/the server never deliver a 0-length frame to this branch, so the `len==0` guard is effectively a keepalive hint. On send, an empty `read_bytes()` is sent with just the tag (harmless to the decoder).
- **Reblocking buffer is unbounded in principle.** `all_output_data` (`:70-81`) accumulates decoded PCM and only drains in 1920-sample steps; if the consumer (`emit`) is pulled slower than audio arrives, the buffer grows. The 90 s `time_limit` (`:142`) bounds total session length, capping the risk.
- **Per-peer isolation via `copy()`.** Each WebRTC connection must get its own `OpusStreamReader/Writer`, websocket, and `all_output_data`; `copy()` (`:100`) re-constructs a fresh `MoshiHandler`. Sharing streaming codec state across peers would corrupt both Opus streams.
- **24 kHz everywhere.** `output_sample_rate` and `input_sample_rate` are both 24000; `output_chunk_size=1920` = exactly one Mimi 80 ms frame. No resampling is done in this client — rate matching is the server's/codec's job.
- **No model special tokens here.** EOAudio (code 2048), EOS (text 7), `<|text_end|>` (130) etc. are interpreted **server-side**; this client only sees decoded audio and already-detokenized text bytes, so none of the LFM2-Audio token semantics apply.
