#!/usr/bin/env python
"""Dump the upstream ConformerEncoder cache-aware STREAMING forward, to verify the Rust
`forward_streaming` (output AND next caches) directly against Python.

Loads the pure-torch conformer modules (no full liquid_audio package), builds the
encoder from the snapshot weights, configures a bounded left context, and runs
`forward_internal` on one chunk with the initial (zero) cache.

Usage: <python-with-torch> parity/dump_conformer_streaming.py /path/to/snapshot
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
    import logging
    enc_mod.logging = logging  # the synthetic module load skips encoder.py's logging import
    ConformerEncoder = enc_mod.ConformerEncoder
    cfg = json.loads((Path(snapshot) / "config.json").read_text())["encoder"]
    import inspect
    valid = set(inspect.signature(ConformerEncoder.__init__).parameters)
    enc = ConformerEncoder(**{k: v for k, v in cfg.items() if k in valid}).eval()

    state = {}
    for f in sorted(glob.glob(str(Path(snapshot) / "*.safetensors"))):
        try:
            sd = load_file(f)
        except Exception:
            continue
        for k, v in sd.items():
            if k.startswith("conformer."):
                state[k[len("conformer."):]] = v.to(torch.float32)
    enc.load_state_dict(state, strict=False)

    mel = load_file(HERE / "golden" / "refs.safetensors")["mel"].to(torch.float32)  # (1, 128, 101)
    length = torch.tensor([mel.shape[-1]], dtype=torch.long)

    # Bound the left context wider than the clip; first-chunk streaming with the initial
    # cache ⇒ output equals the offline forward (same invariant the Rust test uses).
    enc.set_default_att_context_size([29, -1])
    enc.setup_streaming_params()
    cch, ctime, clen = enc.get_initial_cache_state(batch_size=1, dtype=torch.float32, device=mel.device, max_dim=0)

    with torch.no_grad():
        out, out_len, next_ch, next_time, next_len = enc.forward_internal(mel, length, cch, ctime, clen)

    refs = {
        "out": out.to(torch.float32).contiguous(),
        "out_len": out_len.to(torch.int64).contiguous(),
        "next_channel": next_ch.to(torch.float32).contiguous(),
        "next_time": next_time.to(torch.float32).contiguous(),
        "next_len": next_len.to(torch.int64).contiguous(),
    }
    save_file(refs, str(HERE / "golden" / "conformer_streaming_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items()})
    print("out_len:", out_len.tolist(), "next_len:", next_len.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
