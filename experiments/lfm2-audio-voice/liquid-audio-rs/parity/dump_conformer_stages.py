#!/usr/bin/env python
"""Stage-by-stage conformer intermediates, to localize a parity mismatch.

Dumps: post-subsampling (pre_encode) output, pos-encoded x, relative pos-emb,
after-layer-0 output, and the final encoder output. Run after dump_reference.py.

Usage:
    python parity/dump_conformer_stages.py /path/to/model parity/golden
"""
import importlib.machinery
import json
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
from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "../model"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((Path(model_dir) / "config.json").read_text())
    preproc = AudioToMelSpectrogramPreprocessor(**config["preprocessor"]).eval()
    model = LFM2AudioModel.from_pretrained(Path(model_dir), device="cpu", dtype=torch.float32).eval()

    torch.manual_seed(0)
    wav = (torch.randn(1, 16000) * 0.1).to(torch.float32)
    mel, mel_len = preproc(wav, torch.tensor([wav.shape[1]]))

    enc = model.conformer
    acts = {}

    def hook_conv(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        acts["conv_out"] = t.detach().clone()  # (B, C, T', F') pre flatten+linear

    def hook_sub(m, i, o):
        acts["sub"] = o[0].detach().clone()  # (B, T', d)

    def hook_pos(m, i, o):
        acts["posx"] = o[0].detach().clone()
        acts["posemb"] = o[1].detach().clone()

    def hook_l0(m, i, o):
        acts["layer0"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    hs = [
        enc.pre_encode.conv.register_forward_hook(hook_conv),
        enc.pre_encode.register_forward_hook(hook_sub),
        enc.pos_enc.register_forward_hook(hook_pos),
        enc.layers[0].register_forward_hook(hook_l0),
    ]
    # full mel width as conformer length (matches ChatState.add_audio), so the
    # single-clip path applies no intermediate conv masking.
    conformer_len = torch.tensor([mel.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        final, _ = enc(mel.to(torch.float32), conformer_len)
    for h in hs:
        h.remove()

    acts["final"] = final.detach().clone()
    refs = {k: v.to(torch.float32).contiguous() for k, v in acts.items()}
    save_file(refs, str(out / "conformer_stages.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
