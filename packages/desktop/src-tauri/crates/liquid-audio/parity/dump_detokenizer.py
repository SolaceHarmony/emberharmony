#!/usr/bin/env python
"""Golden for the LFM2 audio detokenizer (codes -> 24 kHz waveform) — the model's audio
OUTPUT, which had NO numeric test (only the ISTFT sub-component was locked vs a Rust
reference). Runs the upstream `processor.decode` (= LFM2AudioDetokenizer) on a fixed
deterministic code tensor and dumps the waveform.

Usage: <python-with-torch+liquid_audio> parity/dump_detokenizer.py /path/to/snapshot
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("_upstream").SRC))
from liquid_audio.processor import LFM2AudioProcessor  # noqa: E402


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B"
        "/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    proc = LFM2AudioProcessor.from_pretrained(Path(snapshot))

    # Fixed deterministic codes (1, 8, T), values in [0, 2047] (the same pattern the
    # Rust test feeds), so the golden is reproducible without a generation run.
    k, t = 8, 16
    codes = torch.tensor([(i * 37) % 2048 for i in range(k * t)], dtype=torch.int64).reshape(1, k, t)

    with torch.no_grad():
        wav = proc.decode(codes)  # (1, T') f32 waveform

    refs = {"codes": codes.contiguous(), "wav": wav.to(torch.float32).contiguous()}
    save_file(refs, str(HERE / "golden" / "detokenizer_refs.safetensors"))
    print("dumped:", {k_: tuple(v.shape) for k_, v in refs.items()})
    print(f"wav: {wav.shape}, max|amp| {wav.abs().max().item():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
