#!/usr/bin/env python
"""Dump the mel-featurizer reference WITHOUT the model weights.

The mel featurizer (`AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures` in
`liquid_audio/model/conformer/processor.py`) computes its window + slaney mel
filterbank at init — it needs *no* checkpoint tensors, only the `preprocessor`
block of the model `config.json`. So this dumps a faithful reference using the
real upstream NeMo code with just a tiny config fetch (no ~3 GB download).

We load `conformer/processor.py` directly by file path (importlib) so the
`liquid_audio` package __init__ — which imports torchaudio (no 2.12 wheel) — is
never triggered.

Usage:
    python parity/dump_mel_reference.py            # uses parity/cfg/config.json
    python parity/dump_mel_reference.py /path/to/config.json parity/golden
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent.parent / "upstream-liquid-audio" / "src" / "liquid_audio"
PROC_PY = UPSTREAM / "model" / "conformer" / "processor.py"


def load_nemo_processor():
    spec = importlib.util.spec_from_file_location("nemo_processor", PROC_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # self-contained: imports only logging/abc/random/librosa/torch
    return mod


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "cfg" / "config.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads(cfg_path.read_text())
    prep = config["preprocessor"]

    nemo = load_nemo_processor()
    # Same construction as LFM2AudioProcessor: AudioToMelSpectrogramPreprocessor(**preprocessor).
    preproc = nemo.AudioToMelSpectrogramPreprocessor(**prep).eval()  # .eval() ⇒ no dither

    torch.manual_seed(0)
    wav = (torch.randn(1, 16000) * 0.1).to(torch.float32)  # fixed, deterministic
    length = torch.tensor([wav.shape[1]])

    mel, mel_len = preproc(wav, length)

    refs = {
        "wav": wav.contiguous(),
        "mel": mel.to(torch.float32).contiguous(),         # (1, nfilt, T)
        "mel_len": mel_len.to(torch.int64).contiguous(),   # (1,)
    }
    save_file(refs, str(out / "mel_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    print("mel_len:", int(mel_len[0]))

    # exact_pad=True variant (center=False + explicit (n_fft - hop)//2 signal pad).
    # The LFM2.5-Audio config uses center=True; this is an off-path branch we port
    # for completeness, so dump a separate golden on the SAME deterministic wav.
    prep_ep = dict(prep)
    prep_ep["exact_pad"] = True
    preproc_ep = nemo.AudioToMelSpectrogramPreprocessor(**prep_ep).eval()
    mel_ep, mel_len_ep = preproc_ep(wav, length)
    refs_ep = {
        "wav": wav.contiguous(),
        "mel": mel_ep.to(torch.float32).contiguous(),
        "mel_len": mel_len_ep.to(torch.int64).contiguous(),
    }
    save_file(refs_ep, str(out / "mel_refs_exactpad.safetensors"))
    print("dumped exact_pad:", {k: tuple(v.shape) for k, v in refs_ep.items()})
    print("mel_len exact_pad:", int(mel_len_ep[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
