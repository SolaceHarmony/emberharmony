#!/usr/bin/env python
"""Dump `_create_masks` outputs for several att_context configs, to verify the Rust
`ConformerEncoder::build_masks` port.

`encoder.py` has a deep relative-import tree, so instead of importing it we extract
the EXACT `_create_masks` method source via `ast` and exec just that function (it only
touches `self.self_attention_model` / `self.att_context_style` + torch ops), bound to
a dummy self. So the golden is the real upstream code, not a copy.

Usage: python parity/dump_create_masks.py [out_dir]
"""
import ast
import sys
import textwrap
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ENCODER_PY = HERE.parent.parent / "upstream-liquid-audio" / "src" / "liquid_audio" / "model" / "conformer" / "encoder.py"


def load_create_masks():
    src = ENCODER_PY.read_text()
    tree = ast.parse(src)
    fn_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_create_masks":
            fn_src = ast.get_source_segment(src, node)
            break
    if fn_src is None:
        raise RuntimeError("_create_masks not found in encoder.py")
    ns = {"torch": torch}
    exec(textwrap.dedent(fn_src), ns)
    return ns["_create_masks"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "golden"
    out.mkdir(parents=True, exist_ok=True)
    create_masks = load_create_masks()
    dev = torch.device("cpu")

    m = 8
    plen = torch.tensor([8, 5], dtype=torch.int64)  # B=2: full clip + clip padded to 5
    off = torch.tensor([0, 2], dtype=torch.int64)

    # (name, self_attention_model, att_context_style, att_context_size, offset)
    cases = [
        ("regular_unlimited", "rel_pos", "regular", [-1, -1], None),
        ("regular_band11", "rel_pos", "regular", [1, 1], None),
        ("regular_left2", "rel_pos", "regular", [2, -1], None),
        ("chunked_c4", "rel_pos", "chunked_limited", [4, 3], None),  # chunk_size=4, left_chunks=1
        ("chunked_rightunlim", "rel_pos", "chunked_limited", [2, -1], None),
        ("regular_band11_offset", "rel_pos", "regular", [1, 1], off),
    ]

    refs = {"padding_length": plen.contiguous(), "offset": off.contiguous()}
    for name, sam, style, acs, offset in cases:
        dummy = types.SimpleNamespace(self_attention_model=sam, att_context_style=style)
        pad_mask, att_mask = create_masks(dummy, acs, plen, m, offset, dev)
        refs[f"{name}.pad_mask"] = pad_mask.to(torch.uint8).contiguous()
        if att_mask is not None:
            refs[f"{name}.att_mask"] = att_mask.to(torch.uint8).contiguous()
        nz = 0 if att_mask is None else int(att_mask.sum())
        print(f"{name:24} pad_mask{tuple(pad_mask.shape)} att_mask ignore-count={nz}")

    save_file(refs, str(out / "create_masks_refs.safetensors"))
    print("dumped", len(refs), "tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
