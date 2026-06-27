#!/usr/bin/env python
"""End-to-end greedy generation golden: run the upstream `generate_interleaved` on the
prefill_refs inputs and dump the exact token sequence (text ids + audio frames, in
order). This is the ONLY test that exercises the full autoregressive loop — multi-step
KV cache, sampling, the depthformer per audio frame, and the interleaved text/audio
modality switching — against Python. Greedy (temperature=None) ⇒ deterministic.

Usage: <python-with-torch+liquid_audio> parity/dump_generate.py /path/to/snapshot [N]
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("_upstream").SRC))
from liquid_audio import LFM2AudioModel  # noqa: E402


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B"
        "/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    model = LFM2AudioModel.from_pretrained(Path(snapshot), device="cpu", dtype=torch.float32).eval()

    r = load_file(HERE / "golden" / "prefill_refs.safetensors")
    kw = dict(
        text=r["text"].to(torch.int64),
        audio_in=r["audio_in"].to(torch.float32),
        audio_in_lens=r["audio_in_lens"].to(torch.int64),
        audio_out=r["audio_out"].to(torch.int64),
        modality_flag=r["modality_flag"].to(torch.int64),
    )

    seq_mod, text_vals, audio_vals = [], [], []
    with torch.no_grad():
        for tok in model.generate_interleaved(
            **kw, max_new_tokens=n,
            text_temperature=None, text_top_k=None, audio_temperature=None, audio_top_k=None,
        ):
            t = tok.detach().cpu().flatten()
            if t.numel() == 1:  # text scalar
                seq_mod.append(0)
                text_vals.append(int(t.item()))
            else:  # audio frame (C,)
                seq_mod.append(1)
                audio_vals.append(t.to(torch.int64).tolist())

    refs = {
        "seq_mod": torch.tensor(seq_mod, dtype=torch.int64),
        "text_vals": torch.tensor(text_vals, dtype=torch.int64) if text_vals else torch.zeros(0, dtype=torch.int64),
        "audio_vals": torch.tensor(audio_vals, dtype=torch.int64) if audio_vals else torch.zeros((0, 0), dtype=torch.int64),
    }
    save_file(refs, str(HERE / "golden" / "generate_refs.safetensors"))
    print(f"generated {len(seq_mod)} tokens: {sum(1 for m in seq_mod if m==0)} text, {sum(1 for m in seq_mod if m==1)} audio")
    print("seq_mod:", seq_mod)
    print("text_vals:", text_vals)
    if audio_vals:
        print("audio_vals[0]:", audio_vals[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
