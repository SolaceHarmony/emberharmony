<!-- topic: Transport (off-path) -->
# TR06 · Moshi gradio client
**Code:** `TR06` · **Source:** `moshi/client_gradio.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
A browser-based, WebRTC voice client for the **Moshi** websocket server (`moshi/server.py`). It is a thin transport-and-UI shell — no model, no codec, no tensors of its own — that bridges a `gradio-webrtc` audio component to the server's `/api/chat` websocket using Opus-compressed PCM in both directions. It exists purely as a deployable web demo (Heroku/Spaces-friendly via `rtc_configuration`). It is **vendored Moshi code, off the LFM2-Audio path**: LFM2-Audio's own realtime UI is `demo/chat.py` (fastrtc `ReplyOnPause`), and this gradio client is not ported to Rust.

## How it works
The whole client is one `gradio-webrtc` `StreamHandler` subclass, `MoshiHandler` (`client_gradio.py:21`), plus a `gr.Blocks` wiring `main()` (`:113`). There is no neural network here — the mechanism is **codec framing + websocket multiplexing + an output-rate reblocking buffer**, driven by gradio-webrtc's callback contract (`receive` for mic-in, `emit` for speaker-out, `copy` to clone per-connection, `shutdown` to tear down).

**URL → websocket scheme normalization** (`:29-39`). The constructor splits `url` on `"://"`, maps `ws`/`http`→`ws` and `wss`/`https`→`wss`, and targets the path `/api/chat` (the Moshi server route). The websocket itself is **lazily** opened on the first `receive` (`:51-52`), using the *synchronous* `websockets.sync.client` — gradio-webrtc runs the handler callbacks on worker threads, so a blocking sync socket is correct here (contrast the server's asyncio coroutines, [server.md](TR01-WS-Server)).

**Opus framing (both directions) via `sphn`.** `sphn.OpusStreamWriter`/`OpusStreamReader` are constructed at `output_sample_rate` (24000) (`:40-41`). These are streaming Opus (re)encoders: you push raw PCM in and pull encoded byte-frames out incrementally; the reader is the inverse. This is the same `sphn` codec layer used by `moshi/client.py` ([client.md](TR02-WS-Client)) and the server ([server.md](TR01-WS-Server)).

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
- **Moshi websocket server** ([server.md](TR01-WS-Server)) → inbound messages on `/api/chat`: tagged **Opus audio (`0x01`)** decoded to f32 PCM `(M,)` @ 24 kHz, and **text (`0x02`)** as UTF-8 bytes. The server's `send_loop` produces exactly these tagged frames; this client is the symmetric peer of `moshi/client.py` ([client.md](TR02-WS-Client)) but with a WebRTC/gradio front-end instead of `sounddevice`.

**Downstream (what consumes this output):**
- **Moshi websocket server** ([server.md](TR01-WS-Server)) ← outbound tagged Opus audio (`b"\x01"+opus`) from `receive()`; the server's `recv_loop` decodes it and feeds Mimi-encoded codes into `lm_gen.step`.
- **Speaker / WebRTC peer** ← f32 `(1,1920)` @ 24 kHz audio chunks from `emit()`.
- **`gr.Chatbot` UI** ← streamed text via `AdditionalOutputs` → `add_text` (`:145`).

This component does **not** touch the LFM2-Audio core ([../model/lfm2_audio.md], [../processor.md], [../detokenizer.md]) at all — it is a Moshi-server transport. The Mimi codec ([models/compression.md](MM01-Mimi-Codec)) is relevant only on the *server* side of the wire.

## Python ↔ Rust
**No Rust port.** `client_gradio.py` is in the vendored `liquid_audio/moshi/**`, which `liquid-audio-rs` **reuses as the `moshi` crate (Kyutai's own port) rather than re-porting** (PYTHON_VS_RUST.md §4 "Out of scope / reused"; PORT_STATUS.md: the `moshi/*` row → "♻ reuse the `moshi` crate"). `compare_symbols.py`'s `core` scope **excludes** `moshi/` by design, so there is no symbol-level mapping for this file.

Closest Rust-side analog is **not** a port of this gradio client but the LFM2-Audio Rust transport choice: per ARCHAEOLOGY.md Q4, the Rust ports **only the turn-based demo shape** (`mic_chat.rs`: synchronous `generate_interleaved` on main + `cpal` callback threads + `Arc<Mutex>` rings, **no async, no websocket, no tokio**). The genuinely full-duplex Moshi **asyncio/websocket** stack — `server.py`, `client.py`, **and this `client_gradio.py`** — is **deliberately unported** (PYTHON_VS_RUST.md §2.1 device-agnostic + ARCHAEOLOGY.md "Honest gaps" #1). So the divergence here is *category*, not *op-level*: a WebRTC browser transport has no candle/Rust referent in this project.

## Precision / gotchas
- **int16 scale, not int16 dtype.** `array / 32768.0` (`:54`) assumes the incoming frame is **int16-valued** (range ±32768). gradio-webrtc delivers int16-scaled audio; dividing by `2^15` maps to f32 [-1,1). Feeding already-normalized f32 here would silence the signal by ~32768×.
- **Wire protocol = 1-byte kind tag.** `0x01`=audio, `0x02`=text, prepended on send (`:56`) and consumed as `message[0]` on receive (`:65`). This must stay in lockstep with the server's framing ([server.md](TR01-WS-Server)); a tag mismatch routes audio into the text path or vice-versa.
- **Empty-frame handling is asymmetric.** On receive, a zero-length message yields `None` (`:63-64`) — but the code then **still reads `message[0]`** (`:65`) right after, which would `IndexError` on a truly empty `message`; in practice gradio-webrtc/the server never deliver a 0-length frame to this branch, so the `len==0` guard is effectively a keepalive hint. On send, an empty `read_bytes()` is sent with just the tag (harmless to the decoder).
- **Reblocking buffer is unbounded in principle.** `all_output_data` (`:70-81`) accumulates decoded PCM and only drains in 1920-sample steps; if the consumer (`emit`) is pulled slower than audio arrives, the buffer grows. The 90 s `time_limit` (`:142`) bounds total session length, capping the risk.
- **Per-peer isolation via `copy()`.** Each WebRTC connection must get its own `OpusStreamReader/Writer`, websocket, and `all_output_data`; `copy()` (`:100`) re-constructs a fresh `MoshiHandler`. Sharing streaming codec state across peers would corrupt both Opus streams.
- **24 kHz everywhere.** `output_sample_rate` and `input_sample_rate` are both 24000; `output_chunk_size=1920` = exactly one Mimi 80 ms frame. No resampling is done in this client — rate matching is the server's/codec's job.
- **No model special tokens here.** EOAudio (code 2048), EOS (text 7), `<|text_end|>` (130) etc. are interpreted **server-side**; this client only sees decoded audio and already-detokenized text bytes, so none of the LFM2-Audio token semantics apply.
