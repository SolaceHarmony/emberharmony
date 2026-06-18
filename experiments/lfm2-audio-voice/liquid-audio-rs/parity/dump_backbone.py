#!/usr/bin/env python
"""Backbone (lfm) parity reference: a full causal forward over a fixed embedding
sequence. Compares the HF Lfm2Model (hybrid short-conv + GQA attention) against
the Rust port's `forward_embeds`.

Usage:
    python parity/dump_backbone.py /path/to/model parity/golden
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
sys.path.insert(0, str(HERE.parent.parent / "upstream-liquid-audio" / "src"))

from liquid_audio import LFM2AudioModel  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "../model"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)

    model = LFM2AudioModel.from_pretrained(Path(model_dir), device="cpu", dtype=torch.float32).eval()
    hidden = model.lfm.config.hidden_size

    torch.manual_seed(0)
    # Realistic-magnitude embeddings via the model's own token embedding table,
    # so the backbone sees an in-distribution input. L spans several layers.
    ids = torch.randint(0, 256, (1, 24))
    embeds = model.lfm.embed_tokens(ids).to(torch.float32)

    with torch.no_grad():
        out_hidden = model.lfm(inputs_embeds=embeds, use_cache=False).last_hidden_state
        # text head: tied-embedding logits for the last position (as in generate)
        text_logits = torch.nn.functional.linear(out_hidden[0, -1], model.lfm.embed_tokens.weight)

    refs = {
        "embeds": embeds.contiguous(),       # (1, L, H)
        "backbone": out_hidden.to(torch.float32).contiguous(),  # (1, L, H)
        "text_logits": text_logits.to(torch.float32).contiguous(),  # (vocab,)
    }
    save_file(refs, str(out / "backbone_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
