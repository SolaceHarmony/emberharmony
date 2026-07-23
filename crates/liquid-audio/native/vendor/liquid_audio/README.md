# Liquid Audio reference contract

This directory pins the released LFM2.5-Audio output contract used by the
native implementation. It does **not** vendor checkpoint payload bytes into
Git. The production loader reads the user's Hugging Face snapshot directly
into one sealed resident image.

Pinned authorities:

- model repository: `LiquidAI/LFM2.5-Audio-1.5B`
- model revision: `c362a0625dfe45aa588dce5f0ada28a7e5707628`
- reference repository: `Liquid4All/liquid-audio`
- reference revision: `19e65845923a7f136442c95137884ec61eb386aa`
- detokenizer reference SHA-256: `1076458d10e91f2c5b4c133298436dc46269ed05253de0af8158c9566f5d3c94`
- processor reference SHA-256: `1dd745ec825582b6d1d9c7f3637a8a76e51ff5920506164ca46323874f1b5029`
- main-model reference SHA-256: `5a61b5e198ac419f036f91e62d973a40a3a48730a11ed05ec8fa65d5a360093a`

`reference/{detokenizer,processor,lfm2_audio}.py` and `LICENSE` are byte-exact
copies from that reference revision. The root and nested checkpoint configs and
special-token map are byte-exact checkpoint copies beside the full hash
manifest. The checked-in chat template has one normalized terminal LF because
text patches require it; the manifest records both the authoritative upstream
hash/length and the normalized vendored hash/length. Production reads and
validates the checkpoint copy, not the normalized documentation copy.
Our C++/assembly implementation is a transliteration of the observable
contract, not a Python runtime dependency.

The processor source is load-bearing: it maps the checkpoint spelling
`sliding_attention` to Transformers' executable `full_attention` layer type,
then `decode()` unconditionally selects `audio_detokenizer`. Native performs
that mapping at plan construction and has no Mimi fallback.

The load-bearing output path is:

```text
8 codebooks × 2048 values
  -> offset embedding mean (512)
  -> nearest-exact repeat ×6
  -> eight-layer F32 LFM2, causal sliding window 30
  -> F32 linear 512→1282
  -> 641-bin exp(log-magnitude) + angle
  -> inverse real DFT 1280 / Hann / hop 320 / same-trim overlap-add
  -> mono F32 PCM at 24 kHz (1920 samples per code frame)
```

The checkpoint's `dtype` config says `bfloat16`, but every tensor in
`audio_detokenizer/model.safetensors` is actually `F32`. The safetensors
metadata is authoritative. Native binding must therefore select direct F32
view kernels at plan construction; conversion or BF16 staging is forbidden.

The root `tokenizer-e351c8d8-checkpoint125.safetensors` is the legacy Mimi
codec. It is retained in the upstream distribution for encode/training and
older clients, but it is not the released LFM2.5 `processor.decode` path and
must not be opened by production audio output.
