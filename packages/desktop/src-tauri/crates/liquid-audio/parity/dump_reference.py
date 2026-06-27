#!/usr/bin/env python
"""Dump Tier-2 reference tensors (mel + FastConformer encoder) from the real
upstream `liquid_audio`, with shared weights + a fixed seeded input. The Rust
side (`tests/parity.rs::front_end_parity`) loads the SAME weights + input and
asserts its ported modules match within tolerance.

Env notes (Python 3.14):
  - torchaudio has no 2.12 wheel, but `liquid_audio` only *imports* it on this
    path (never calls it), so we register a spec'd stub before importing.
  - We compute the mel via the standalone NeMo preprocessor (config-only) instead
    of LFM2AudioProcessor, to skip loading the 384 MB Mimi/audio tokenizer.

Usage:
    python parity/dump_reference.py /path/to/LFM2-Audio-1.5B parity/golden
"""
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

# --- torchaudio stub (import-only dependency on this path) -------------------
_ta = types.ModuleType("torchaudio")
_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
_ta.__version__ = "0.0.0-stub"
sys.modules.setdefault("torchaudio", _ta)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("_upstream").SRC))

from liquid_audio import LFM2AudioModel  # noqa: E402
from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2-Audio-1.5B"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    config = json.loads((Path(model_dir) / "config.json").read_text())
    preproc = AudioToMelSpectrogramPreprocessor(**config["preprocessor"]).eval()

    # f32 so the Rust f32 load matches (the checkpoint is bf16 on disk).
    # Pass a Path (not str) so get_model_dir uses the local-dir branch.
    model = LFM2AudioModel.from_pretrained(Path(model_dir), device=device, dtype=torch.float32)
    model.eval()

    torch.manual_seed(0)
    wav = (torch.randn(1, 16000) * 0.1).to(torch.float32)  # fixed, deterministic
    length = torch.tensor([wav.shape[1]])

    mel, mel_len = preproc(wav, length)
    # The model feeds the conformer the FULL mel width as the length (see
    # ChatState.add_audio: audio_in_lens = mel.shape[1]), not the preprocessor's
    # valid mel_len — so a single clip gets no intermediate conv masking.
    conformer_len = torch.tensor([mel.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        enc, enc_len = model.conformer(mel.to(torch.float32), conformer_len)

    refs = {
        "wav": wav.contiguous(),
        "mel": mel.to(torch.float32).contiguous(),          # (1, nfilt, T)
        "mel_len": mel_len.to(torch.int64).contiguous(),    # (1,)
        "conformer": enc.to(torch.float32).contiguous(),    # (1, d, T')
        "conformer_len": enc_len.to(torch.int64).contiguous(),
    }
    save_file(refs, str(out / "refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    print("mel_len:", int(mel_len[0]), "conformer_len:", int(enc_len[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
