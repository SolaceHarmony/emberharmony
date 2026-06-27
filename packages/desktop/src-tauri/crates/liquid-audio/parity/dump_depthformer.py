#!/usr/bin/env python
"""Depthformer parity reference: the greedy 8-codebook audio frame produced by
`_sample_audio_frame` for a fixed lfm-hidden embedding. Token-exact comparison
(greedy is deterministic), so no tolerance needed.

Usage:
    python parity/dump_depthformer.py /path/to/model parity/golden
"""
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

_ta = types.ModuleType("torchaudio")
_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
_ta.__version__ = "0.0.0-stub"
sys.modules.setdefault("torchaudio", _ta)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("_upstream").SRC))

from liquid_audio import LFM2AudioModel  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "../model"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)

    model = LFM2AudioModel.from_pretrained(Path(model_dir), device="cpu", dtype=torch.float32).eval()
    hidden = model.lfm.config.hidden_size

    torch.manual_seed(0)
    embedding = (torch.randn(hidden) * 0.1).to(torch.float32)  # fixed lfm-hidden vector

    with torch.no_grad():
        tokens = model._sample_audio_frame(embedding, temperature=None, top_k=None)  # greedy

    refs = {
        "embedding": embedding.contiguous(),               # (H,)
        "tokens": tokens.to(torch.int64).contiguous(),     # (codebooks,)
    }
    save_file(refs, str(out / "depthformer_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()}, "tokens:", tokens.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
