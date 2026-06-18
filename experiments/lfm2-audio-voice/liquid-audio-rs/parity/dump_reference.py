#!/usr/bin/env python
"""Dump reference tensors from the Python `liquid_audio` for Rust parity checks.

Runs a fixed, seeded input through the deterministic front-end (mel featurizer +
FastConformer encoder) and saves the intermediate tensors. The Rust side
(`tests/parity.rs`) loads the SAME model weights + the SAME input and asserts its
ported modules match within tolerance.

Usage:
    python parity/dump_reference.py /path/to/LFM2-Audio-1.5B parity/refs

Extend `refs` with prefill / lfm-hidden / first-token tensors to widen coverage.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2-Audio-1.5B"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "refs"
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    proc = LFM2AudioProcessor.from_pretrained(repo, device=device)
    model = LFM2AudioModel.from_pretrained(repo, device=device, dtype=torch.float32)

    torch.manual_seed(0)
    # fixed 1s of low-level noise at 16 kHz (deterministic across runs)
    wav = (torch.randn(1, 16000) * 0.1).to(torch.float32)

    refs: dict[str, torch.Tensor] = {"wav": wav.contiguous()}

    # mel featurizer
    mel, mel_len = proc.audio(wav, torch.tensor([wav.shape[1]]))
    refs["mel"] = mel.to(torch.float32).contiguous()  # (1, nfilt, T)

    # FastConformer encoder
    enc, _ = model.conformer(mel.to(torch.float32), mel_len)
    refs["conformer"] = enc.to(torch.float32).contiguous()  # (1, d, T')

    save_file(refs, str(out / "refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
