#!/usr/bin/env python
"""Dump ConvSubsampling outputs for each NON-causal scheme, to verify the Rust
`ConvSubsampling::new_scheme` port.

subsampling.py imports only logging/math/torch/nn (no relative imports), so it loads
standalone — we construct the real upstream ConvSubsampling per scheme, run forward on
a fixed input, and dump input / output / state_dict (weights named conv.{i}.weight /
out.weight, matching the Rust VarBuilder).

Usage: python parity/dump_subsampling_schemes.py [out_dir]
"""
import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
CONF = __import__("_upstream").PKG / "model" / "conformer"
SUB_PY = CONF / "subsampling.py"
MODULES_PY = CONF / "modules.py"


def load_sub():
    spec = importlib.util.spec_from_file_location("nemo_subsampling", SUB_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # modules.py has a relative import, so ast-extract the self-contained CausalConv1D
    # class (extends nn.Conv1d, uses F.pad) and inject it for the causal conv1d scheme.
    src = MODULES_PY.read_text()
    tree = ast.parse(src)
    cls_src = next(
        ast.get_source_segment(src, n) for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "CausalConv1D"
    )
    ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(textwrap.dedent(cls_src), ns)
    mod.CausalConv1D = ns["CausalConv1D"]
    return mod


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)
    sub = load_sub()

    torch.manual_seed(0)
    b, t, feat_in = 1, 20, 128
    feat_out, conv_channels, factor = 512, 256, 8
    x0 = torch.randn(b, t, feat_in)  # (B, T, F) — the Rust ConvSubsampling::forward input
    lengths = torch.tensor([t] * b, dtype=torch.int64)  # full ⇒ masking is a no-op

    conv2d_schemes = {"vggnet", "dw_striding", "striding"}
    # (golden_key, scheme, is_causal)
    schemes = [
        ("vggnet", "vggnet", False),
        ("striding", "striding", False),
        ("striding_conv1d", "striding_conv1d", False),
        ("striding_conv1d_causal", "striding_conv1d", True),
        ("dw_striding_conv1d", "dw_striding_conv1d", False),
        ("dw_striding", "dw_striding", False),
    ]

    refs = {"x": x0.contiguous()}
    for key, name, is_causal in schemes:
        m = sub.ConvSubsampling(
            subsampling=name,
            subsampling_factor=factor,
            feat_in=feat_in,
            feat_out=feat_out,
            conv_channels=conv_channels,
            is_causal=is_causal,
        ).eval()
        # The upstream MaskedConvSequential masking wrapper only works for plain Conv2d
        # (it unsqueezes to 4-D and assumes tuple kernel/stride), so it breaks on the
        # conv1d schemes AND on vggnet's MaxPool2d. Apply the REAL conv layers
        # (m.conv's torch modules + m.out) directly in the documented ConvSubsampling.forward
        # order — the masking it skips is a no-op at full length, so this is the faithful
        # torch reference (real Conv2d/Conv1d/MaxPool2d/Linear ops on the real weights).
        with torch.no_grad():
            if name in conv2d_schemes:
                xt = x0.unsqueeze(1)  # (B, 1, T, F)
                for layer in m.conv:
                    xt = layer(xt)  # (B, C, T', F')
                bb, cc, tt, ff = xt.shape
                y = m.out(xt.transpose(1, 2).reshape(bb, tt, cc * ff))  # (B, T', feat_out)
            else:
                xt = x0.transpose(1, 2)  # (B, feat_in, T)
                for layer in m.conv:
                    xt = layer(xt)  # (B, C, T')
                y = xt.transpose(1, 2)  # (B, T', C)
        refs[f"{key}.out"] = y.contiguous()
        for k, v in m.state_dict().items():
            refs[f"{key}.w.{k}"] = v.contiguous()
        print(f"{key:24} out {tuple(y.shape)}  weights={sum(1 for k in m.state_dict() if k.endswith('weight'))}")

    save_file(refs, str(out / "subsampling_schemes_refs.safetensors"))
    print("dumped", len(refs), "tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
