#!/usr/bin/env python
"""Dump Python conformer stages for the SAME mel the Rust test feeds (refs[mel]).

The existing `dump_conformer_stages.py` generates its own seed-0 mel, but
`conformer_stages_parity` feeds `refs.safetensors[mel]` — so the golden never
matched the test input. This dumps the Python conformer stages for refs[mel],
loading ONLY the (pure-torch) conformer modules so it runs without the full
`liquid_audio` package (no transformers/librosa/accelerate needed).

Usage: <python-with-torch> parity/dump_conformer_on_refmel.py /path/to/snapshot
"""
import glob
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
CONF = HERE.parent.parent / "upstream-liquid-audio" / "src" / "liquid_audio" / "model" / "conformer"


def load_conformer_module():
    """Load encoder.py + its siblings as a synthetic `lac` package (bypasses the
    heavy `liquid_audio/__init__.py` import chain)."""
    pkg = types.ModuleType("lac")
    pkg.__path__ = [str(CONF)]
    sys.modules["lac"] = pkg
    for name in ["utils", "mha", "modules", "subsampling", "encoder"]:
        spec = importlib.util.spec_from_file_location(f"lac.{name}", CONF / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "lac"
        sys.modules[f"lac.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["lac.encoder"]


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B"
        "/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    enc_mod = load_conformer_module()
    ConformerEncoder = enc_mod.ConformerEncoder

    cfg = json.loads((Path(snapshot) / "config.json").read_text())["encoder"]
    # ConformerEncoder.__init__ accepts a superset; pass the config keys it knows.
    import inspect
    valid = set(inspect.signature(ConformerEncoder.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    enc = ConformerEncoder(**kwargs).eval()

    # Load the conformer weights from the snapshot shards (keys `conformer.*`).
    state = {}
    for f in sorted(glob.glob(str(Path(snapshot) / "*.safetensors"))):
        try:
            sd = load_file(f)
        except Exception:
            continue
        for k, v in sd.items():
            if k.startswith("conformer."):
                state[k[len("conformer."):]] = v.to(torch.float32)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    print(f"loaded conformer weights: {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first missing:", missing[:4])

    # The exact mel the Rust test feeds.
    mel = load_file(HERE / "golden" / "refs.safetensors")["mel"].to(torch.float32)
    print("mel shape:", tuple(mel.shape))

    acts = {}

    # NB: a forward-hook that RETURNS non-None replaces the module output, so each
    # hook must store and return nothing.
    def hook_conv(m, i, o):
        acts["conv_out"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    def hook_sub(m, i, o):
        acts["sub"] = o[0].detach().clone()

    def hook_pos(m, i, o):
        acts["posx"] = o[0].detach().clone()
        acts["posemb"] = o[1].detach().clone()

    def hook_l0(m, i, o):
        acts["layer0"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    enc.pre_encode.conv.register_forward_hook(hook_conv)
    enc.pre_encode.register_forward_hook(hook_sub)
    enc.pos_enc.register_forward_hook(hook_pos)
    enc.layers[0].register_forward_hook(hook_l0)

    conformer_len = torch.tensor([mel.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        final, _ = enc(mel, conformer_len)
    acts["final"] = final.detach().clone()

    refs = {k: v.to(torch.float32).contiguous() for k, v in acts.items()}
    save_file(refs, str(HERE / "golden" / "conformer_stages.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
