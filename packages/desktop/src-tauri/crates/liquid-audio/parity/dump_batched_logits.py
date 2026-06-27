#!/usr/bin/env python
"""Python golden for the BATCHED (B=2) training path: model.logits + model.forward.

This is the detector the B>1 row-0 bug needed: it compares the Rust logits/forward
on a real 2-sample batch to Python. The old code read only batch row 0, so its
text_logits were (n_text, V) — half of Python's (2*n_text, V); this golden makes
that a hard failure, not a passing self-consistency check.

The 2-sample batch is a duplicate of the prefill_refs sample, collated the way
`lfm2_collator` does (text/audio_in/audio_out cat dim=1; modality/supervision/lens
cat dim=0), with all-ones supervision.

Usage: <python-with-Lfm2> parity/dump_batched_logits.py /path/to/snapshot
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("_upstream").SRC))

from liquid_audio import LFM2AudioModel  # noqa: E402
from liquid_audio.data.types import LFM2AudioModelInput  # noqa: E402


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B"
        "/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    model = LFM2AudioModel.from_pretrained(Path(snapshot), device="cpu", dtype=torch.float32).eval()

    r = load_file(HERE / "golden" / "prefill_refs.safetensors")
    text = r["text"].long()                  # (1, n)
    audio_in = r["audio_in"].float()         # (128, f)
    audio_in_lens = r["audio_in_lens"].long()  # (s,)
    audio_out = r["audio_out"].long()        # (8, a)
    modality = r["modality_flag"].long()     # (1, L)
    sup = torch.ones((1, modality.shape[1]), dtype=torch.bool)

    # B=2 by duplication, matching lfm2_collator's cat dims.
    batch = LFM2AudioModelInput(
        text=torch.cat([text, text], 1),
        audio_in=torch.cat([audio_in, audio_in], 1),
        audio_in_lens=torch.cat([audio_in_lens, audio_in_lens], 0),
        audio_out=torch.cat([audio_out, audio_out], 1),
        modality_flag=torch.cat([modality, modality], 0),
        supervision_mask=torch.cat([sup, sup], 0),
    )

    with torch.no_grad():
        text_logits, audio_logits, text_labels, audio_labels = model.logits(batch)
        out = model.forward(batch)

    refs = {
        "text_logits": text_logits.float().contiguous(),
        "audio_logits": audio_logits.float().contiguous(),
        "text_labels": text_labels.long().contiguous(),
        "audio_labels": (audio_labels.long() if audio_labels.numel() else torch.zeros(0, dtype=torch.long)).contiguous(),
        "loss": out.loss.float().reshape(1).contiguous(),
        "text_loss": out.text_loss.float().reshape(1).contiguous(),
        "audio_loss": out.audio_loss.float().reshape(1).contiguous(),
    }
    save_file(refs, str(HERE / "golden" / "batched_logits_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    print("loss:", out.loss.item(), "text_loss:", out.text_loss.item())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
