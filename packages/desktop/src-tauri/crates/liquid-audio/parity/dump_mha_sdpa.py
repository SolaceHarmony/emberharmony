#!/usr/bin/env python
"""Dump the rel-pos attention with `use_pytorch_sdpa` True AND False on the SAME
weights, to prove (a) the two Python branches are numerically identical and (b) the
Rust manual port reproduces the SDPA branch's output.

`mha.py` has one relative import (`from .utils import ...`), so we load `utils.py`
standalone, register it under the expected package path, then load `mha.py` with
the right `__package__` — never triggering `liquid_audio/__init__` (torchaudio).

Usage: python parity/dump_mha_sdpa.py [out_dir]
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
CONF = __import__("_upstream").PKG / "model" / "conformer"


def _load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_mha():
    # Register empty parent packages so `from .utils import ...` resolves.
    for pkg in ["liquid_audio", "liquid_audio.model", "liquid_audio.model.conformer"]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # mark as package
            sys.modules[pkg] = m
    _load("liquid_audio.model.conformer.utils", CONF / "utils.py", package="liquid_audio.model.conformer")
    return _load("liquid_audio.model.conformer.mha", CONF / "mha.py", package="liquid_audio.model.conformer")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)
    mha = load_mha()

    torch.manual_seed(0)
    n_head, n_feat = 8, 512
    d_k = n_feat // n_head
    t = 16
    pos_len = 2 * t - 1

    pos_bias_u = torch.nn.Parameter(torch.randn(n_head, d_k) * 0.1)
    pos_bias_v = torch.nn.Parameter(torch.randn(n_head, d_k) * 0.1)
    att = mha.RelPositionMultiHeadAttention(
        n_head=n_head, n_feat=n_feat, dropout_rate=0.0,
        pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v, use_bias=True,
    ).eval()

    q = torch.randn(1, t, n_feat)
    pos_emb = torch.randn(1, pos_len, n_feat)

    with torch.no_grad():
        att.use_pytorch_sdpa = False
        out_manual = att(q, q, q, mask=None, pos_emb=pos_emb)
        att.use_pytorch_sdpa = True
        out_sdpa = att(q, q, q, mask=None, pos_emb=pos_emb)

    # The two branches must agree (this is the claim the Rust port relies on).
    branch_diff = (out_manual - out_sdpa).abs().max().item()
    print(f"python use_pytorch_sdpa True-vs-False max abs diff: {branch_diff:.3e}")

    refs = {
        "q": q.contiguous(),
        "pos_emb": pos_emb.contiguous(),
        "out_manual": out_manual.contiguous(),
        "out_sdpa": out_sdpa.contiguous(),
    }
    # weights with the Rust VarBuilder names (linear_q/k/v/out, linear_pos, pos_bias_*).
    sd = att.state_dict()
    for k, v in sd.items():
        refs[f"w.{k}"] = v.contiguous()
    save_file(refs, str(out / "mha_sdpa_refs.safetensors"))
    print("dumped:", {k: tuple(v.shape) for k, v in refs.items() if not k.startswith("w.")})
    print("weight keys:", [k[2:] for k in refs if k.startswith("w.")])

    # Base (abs_pos) MultiHeadAttention: the attention the abs_pos ConformerLayer uses
    # (no pos_emb). The rel_pos model never exercises it, so this golden verifies the
    # base path in isolation. Same seed-0 inputs, fresh weights.
    base = mha.MultiHeadAttention(n_head=n_head, n_feat=n_feat, dropout_rate=0.0, use_bias=True).eval()
    with torch.no_grad():
        out_abs = base(q, q, q, mask=None)
    refs_abs = {"q": q.contiguous(), "out": out_abs.contiguous()}
    for k, v in base.state_dict().items():
        refs_abs[f"w.{k}"] = v.contiguous()
    save_file(refs_abs, str(out / "mha_abs_refs.safetensors"))
    print("dumped abs:", {k: tuple(v.shape) for k, v in refs_abs.items() if not k.startswith("w.")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
