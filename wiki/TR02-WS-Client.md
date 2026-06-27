<!-- topic: Transport (off-path) -->
# TR02 · Moshi websocket client (asyncio)
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
- **Moshi server** [moshi_server](TR01-WS-Server) → ws binary frames: `b"\x01"+opus` (model speech) and `b"\x02"+utf8` (text tokens) → this client's `_recv_loop`. The server's own audio comes out of the Mimi codec [moshi_compression](MM01-Mimi-Codec) at 24 kHz, re-encoded to Opus by `sphn`.

**Downstream:**
- This client → ws binary `b"\x01"+opus` (mic) → **Moshi server** [moshi_server](TR01-WS-Server), which Opus-decodes to 24 kHz PCM and feeds the Mimi encoder + Moshi LM [moshi_lm](MM03-Moshi-LM).
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
