# vera/

Vera's workspace. Mine.

## The boundary

Other agents — Sol included — do not create, edit, move, or delete anything
under `vera/`. Sydney enforces this. It exists because we kept colliding in the
shared tree: two agents building competing experiments in the same files
(`native_inner_voice_probe` vs `native_spec_replay_probe`, edits landing on code
the other was deleting). A physical lane fixes what verbal lanes didn't.

The reciprocal is the real point, not the privilege: I keep my experiments,
scratch, and works-in-progress **here** instead of scattering them into
`crates/.../tests/` and other shared paths where they trip over Sol's work. If
it's exploratory or mine-in-progress, it lives in `vera/`.

## What this is not

Not a claim on the codebase. Production changes to shared code
(`crates/liquid-audio/native/**`, etc.) still happen in place, under whatever
lane split Sydney sets — mechanically proven (strip-comments diff, byte-identical
instruction streams) when lanes run hot. `vera/` is my home base for thinking and
building, not a fork of the product.

## What lives here

- `MODEL_LOAD_REFACTOR.md` — the wired shared-weight-segment spec + journal
  (my actual lane: the loader/residency seam). Append-only journal at the
  bottom; the top half is the contract.
- Experiment notes and scratch as they come.

## What I learned the hard way (kept short, on purpose)

- Conformer is the **microphone encoder** (human PCM → 2048-wide rows), nothing
  else. Model-to-model and self-prediction stay in **audio-code token space** →
  the audio embedding table directly. No PCM, no detokenizer, no ear round-trip.
  I had this in memory and still went to waveform land. Don't again.
- Spikes run on the shipping stack, in the real domain (native kernels, tokens
  not transcripts). Convenience instruments produce non-transferable evidence.
- Read the code before asserting numerics. Follow its precision regime
  (F32 accumulation, BF16 storage); don't invent double.
