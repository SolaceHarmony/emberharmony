#!/usr/bin/env python
"""Regenerate depthformer_refs (greedy tokens) from the HF SNAPSHOT weights.

The committed golden was dumped from `../model` (a different checkpoint), so the
greedy tokens never matched the snapshot the Rust loads. The depthformer stack
(transformer.py) is pure-torch, so this rebuilds it from the snapshot and re-runs
the greedy `_sample_audio_frame` loop on the SAME input embedding, producing a
golden consistent with what the Rust runs.

Usage: <python-with-torch> parity/dump_depthformer_from_snapshot.py /path/to/snapshot
"""
import glob
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = __import__("_upstream").PKG / "model"


def load_transformer_module():
    """Load model/transformer.py as a synthetic `lt` module (pure torch, no deps)."""
    pkg = types.ModuleType("lt")
    pkg.__path__ = [str(SRC)]
    sys.modules["lt"] = pkg
    spec = importlib.util.spec_from_file_location("lt.transformer", SRC / "transformer.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "lt"
    sys.modules["lt.transformer"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    snapshot = sys.argv[1] if len(sys.argv) > 1 else str(
        Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-Audio-1.5B"
        "/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    T = load_transformer_module()
    cfg = json.loads((Path(snapshot) / "config.json").read_text())
    codebooks = cfg["codebooks"]
    hidden = cfg["lfm"]["hidden_size"]
    depth_dim = cfg["depthformer"]["dim"]
    depth_layers = cfg["depthformer"]["layers"]
    depth_tie = cfg["depthformer"]["tie"]
    audio_vocab = 2048 + 1

    # Mirror LFM2AudioModel.__init__ (depthformer parts).
    scale = 1 / math.sqrt(2 * depth_layers)
    layers = [T.StandardBlock(T.MHA(depth_dim, out_init_scale=scale), out_init_scale=scale) for _ in range(depth_layers)]
    depthformer = T.RawLMBackbone(layers, has_embedding=False).eval()
    depth_linear = nn.Linear(hidden, depth_dim * codebooks)
    depth_embeddings = nn.ModuleList(
        [T.SharedEmbedding(dim=depth_dim, vocab_size=audio_vocab, tie_embedding=depth_tie) for _ in range(codebooks)]
    ).eval()

    # Load the snapshot weights for these submodules.
    full = {}
    for f in sorted(glob.glob(str(Path(snapshot) / "*.safetensors"))):
        try:
            full.update(load_file(f))
        except Exception:
            pass

    def load_into(mod, prefix):
        sub = {k[len(prefix):]: v.to(torch.float32) for k, v in full.items() if k.startswith(prefix)}
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        print(f"{prefix:20s} loaded={len(sub)} missing={len(missing)} unexpected={len(unexpected)}")
        return missing

    load_into(depthformer, "depthformer.")
    load_into(depth_linear, "depth_linear.")
    load_into(depth_embeddings, "depth_embeddings.")

    # Reuse the existing input embedding from the committed golden (provenance-neutral:
    # it is just an input vector the depthformer processes).
    embedding = load_file(HERE / "golden" / "depthformer_refs.safetensors")["embedding"].to(torch.float32)

    # Greedy _sample_audio_frame (lfm2_audio.py L501-534) with the snapshot weights.
    with torch.no_grad():
        depthformer_in = depth_linear(embedding).reshape(codebooks, depth_dim)
        df_token = torch.zeros_like(depthformer_in[0])
        cache = None
        tokens = []
        for i in range(codebooks):
            cur = depthformer_in[i] + df_token
            out, cache = depthformer.forward_cached(cur[None, None, :], cache)
            logits = depth_embeddings[i].get_logits(out.squeeze())
            tok = logits.argmax()
            tokens.append(tok)
            df_token = depth_embeddings[i](tok).squeeze()
        tokens = torch.stack(tokens).to(torch.int64)

    save_file({"embedding": embedding.contiguous(), "tokens": tokens.contiguous()},
              str(HERE / "golden" / "depthformer_refs.safetensors"))
    print("dumped tokens:", tokens.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
