<!-- topic: Transport (off-path) -->
# TR03 · Moshi client utils
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
- [moshi_client](TR02-WS-Client) — imports `AnyPrinter, Printer, RawPrinter` (`client.py:16`); its `recv_loop` calls `printer.print_token(payload.decode())`, `printer.print_pending()`, `printer.print_lag()`, `printer.print_header()`, and `printer.log(...)`. Edge: decoded text `str`.
- [moshi_server](TR01-WS-Server) — imports the same printers for server-side logging. Edge: log `str`.
- [moshi_run_inference](TR04-Run-Inference) — imports `AnyPrinter, Printer, RawPrinter, log` (`run_inference.py:18`); the offline streaming loop emits text via `printer.print_token(text)` and warns `"EOS sampled too early."` / logs timing. Edge: decoded text `str`.

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
