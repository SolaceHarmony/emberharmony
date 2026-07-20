# Mimi streaming right-trim — no EOF flush

Mimi's causal transposed convolutions deliberately do not emit their final
`kernel - stride` overhang. That region is right-trimmed in the trained one-shot
operation. The streaming operation retains the same values only so the next
chunk can overlap-add into them; reset intentionally discards the retained
overhang at a turn boundary.

This is model behavior, not an unfinished drain path.

## Source contract

For an input of `n` frames, kernel `k`, and stride `s`:

```text
raw output       = n * s + (k - s)
right trim/carry = k - s
model output     = n * s
```

The upstream Rust implementation applies the right trim in its one-shot
forward path, retains the trimmed region for overlap in `step`, and proves that
concatenated streaming steps equal the trimmed one-shot output without a flush.
The upstream Python implementation uses the same `unpad1d(..., k - s)` and
streaming overlap contract.

The native implementation mirrors it:

- `mimi_convtr_step` emits `n * stride` samples and banks `k - stride` values;
- the next step overlap-adds the banked values before replacing the carry;
- `mimi_convtr_reset` clears `prev_valid` without emitting the bank;
- the top-level decoder emits exactly 1,920 PCM samples per valid code frame
  before device-rate resampling.

Appending the banks at EOF would add 3,024 fabricated PCM samples (126 ms at
24 kHz) per turn across the top upsampler and four SeaNet transposed
convolutions. It would no longer match the trained codec.

## Terminal accounting

The correct turn invariant is:

```text
playback leases published == ordinary audio-code emissions
playback leases retired   == playback leases published
```

`EOAudio` publishes no PCM lease. Rust may expose `TurnComplete` only after all
leases promised by the correlated native terminal record have retired. There
is no synthetic final-tail lease.

If an ending sounds clipped, investigate generation stopping, `EOAudio`, token
budget, codec-frame production, or playback retirement. Do not manufacture a
ConvTranspose flush.
