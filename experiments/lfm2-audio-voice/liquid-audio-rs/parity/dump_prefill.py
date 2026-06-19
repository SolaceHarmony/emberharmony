#!/usr/bin/env python
"""Prefill parity reference: the assembled input embeddings for a real ChatState
(system text + a user audio turn + assistant turn start). Exercises the
modality-scatter assembly — text embeddings, conformer+adapter audio-in
embeddings, and audio-out embeddings interleaved by modality_flag.

Dumps the raw ChatState fields (so the Rust side loads identical inputs, avoiding
any chat-template tokenization mismatch) plus the _prefill output.

Usage:
    python parity/dump_prefill.py /path/to/model parity/golden
"""
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

# torchaudio stub incl. an identity resample (input is already 16 kHz).
_ta = types.ModuleType("torchaudio")
_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
_ta.__version__ = "0.0.0-stub"
_ta.functional = types.SimpleNamespace(resample=lambda w, orig, new: w)
sys.modules.setdefault("torchaudio", _ta)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "upstream-liquid-audio" / "src"))

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "../model"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)

    proc = LFM2AudioProcessor.from_pretrained(Path(model_dir), device="cpu")
    model = LFM2AudioModel.from_pretrained(Path(model_dir), device="cpu", dtype=torch.float32).eval()

    chat = ChatState(proc, dtype=torch.float32)
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    # Two user audio turns of DIFFERENT lengths, so Python pads them into a batch
    # and length-masks — the case where per-segment Rust encoding could diverge.
    torch.manual_seed(0)
    chat.new_turn("user")
    chat.add_audio((torch.randn(1, 16000) * 0.1).to(torch.float32), 16000)  # 1.0 s
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio((torch.randn(1, 9000) * 0.1).to(torch.float32), 16000)   # ~0.56 s
    chat.end_turn()
    chat.new_turn("assistant")

    with torch.no_grad():
        in_emb = model._prefill(**chat)

    refs = {
        "text": chat.text.to(torch.int64).contiguous(),
        "audio_in": chat.audio_in.to(torch.float32).contiguous(),
        "audio_in_lens": chat.audio_in_lens.to(torch.int64).contiguous(),
        "audio_out": chat.audio_out.to(torch.int64).contiguous(),
        "modality_flag": chat.modality_flag.to(torch.int64).contiguous(),
        "in_emb": in_emb.to(torch.float32).contiguous(),
    }
    save_file(refs, str(out / "prefill_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
