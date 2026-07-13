# Mimi streaming flush — drain the ConvTr tails at turn end

Found 2026-07-12 troubleshooting the self-chat e2e (Sydney's call: "lack of
waiting until it finishes draining out remaining audio").

## The bug

Every streaming ConvTranspose banks `invalid = ksize − stride` COMPLETED
output samples (bias included) in its `prev` carry (`mimi_conv.cpp`,
MimiConvTrState). `mimi_convtr_reset` drops them. Nothing ever emits them:
`mimi_decoder_step` returns exactly `n_tr × 960` PCM per frame, and the turn
boundary calls `reset_stream()`. Trailing loss per turn:

| layer | k, s | rate | tail |
|---|---|---|---|
| upsample ConvTr | s=2 (k per checkpoint) | 25 Hz latent | ≈ 2 latent frames ≈ 80 ms PCM |
| seanet convtr r8 | 16, 8 | 200 Hz | 40 ms |
| seanet convtr r6 | 12, 6 | 1.2 kHz | 5 ms |
| seanet convtr r5 | 10, 5 | 6 kHz | 0.8 ms |
| seanet convtr r4 | 8, 4 | 24 kHz | 0.17 ms |

**≈ 120–125 ms of speech dropped at EVERY turn end — production path
(`respond` → MimiDetokenizer → NativeMimi), not just the harness.** The
parity gate never saw it because streaming-vs-offline comparisons were
truncated to the emitted region.

## Why the tail is emittable

A ConvTr carry row is FINAL once no more input arrives: future input only
adds contributions at later output positions. So at end-of-turn the carry IS
the layer's last `invalid` output steps, bias already included. Flushing is
pure emission + downstream propagation — no fabricated codes, no model call.

## The flush cascade (design)

`mimi_decoder_flush(MimiDecoder *d, float *pcm_out) -> n_pcm`:

1. **upsample**: emit its carry (≤ invalid latent frames) into up_buf.
2. **transformer**: normal `mimi_transformer_step` on those frames (causal —
   consumes input, emits same count, no held output of its own).
3. **seanet flush walk** (`mimi_seanet_flush`): maintain a tail-activation
   buffer, initially the transformer output. For each layer in order:
   - conv/ELU/resnet stages: normal step on the tail buffer (0-in → 0-out).
   - convtr stages: `out = convtr_step(tail_in)` (emits n·stride, re-banks)
     then APPEND the layer's own carry emission (`convtr_flush_carry`: copy
     prev[oc, 0..invalid], mark prev_valid=0). Downstream stages see
     `n·stride + invalid` frames of ordinary input.
4. Final conv processes the tail; return total samples (≈ 3k ≈ 126 ms).

New primitive: `mimi_convtr_flush(MimiConvTrState*, float *y) -> n` — emit
carry rows, clear prev_valid. Same for MimiUpsampleState.

**Buffer audit required**: inter-stage latent buffers are sized for ~4
frames; the flush pushes up to `2→24→150→755→3024` samples through the layer
boundaries. Check `MIMI_MAX_LATENT`, `MIMI_CONV_MAX_NIN`,
`MIMI_CONV_GEMM_MAX_N` and per-layer scratch against the flush widths — size
them for the flush case at init (arena, fixed capacity; growth is theft).

## Rim + runtime wiring

- FFI: `mimi_decoder_flush` exported; `NativeMimi::flush() -> Vec<f32>`.
- Trait: `AudioDetokenizer::flush_stream() -> Result<Option<Tensor>>`
  (default None). MimiDetokenizer implements via native flush.
- `respond` (realtime.rs): after the generation loop, BEFORE `TurnComplete`:
  flush, emit tail as a final `VoiceEvent::Audio { pcm, rate }` and include
  in the turn accounting. `reset_stream` at next turn start is then a no-op
  on the carries (already drained).
- self_chat needs no change — the tail arrives via the normal callback.

## The oracle (self-oracle, no torch needed)

`streaming decode of N frames + flush ≡ offline one-shot decode of the same
N frames`, full length, within the existing parity band (≤ ~4.2e-6 worst).
No truncation allowed — the tail is exactly what truncation used to hide.
Add to the parity suite next to the existing streaming gate. Gate the wav-hash
e2e after: turn WAVs grow by the tail; endings audibly complete.

## Related harness fixes (landed with this doc)

- self_chat budgets 160/112 → 512 (interleaved steps: audio frames cost 1
  each; 160 ≈ 8.5 s speech ceiling — the observed truncations). Production
  interleaved budget is 8192 (control.rs) — unaffected.
- EXHAUSTED eprintln now newline-prefixed (was gluing to the transcript line).
- Separate, pending decision: doubled <|im_end|> on naturally-completed turns
  (respond collects the generated im_end AND end_turn appends the footer) —
  see conversation notes; fix is to drop the stop token from text_ids.
