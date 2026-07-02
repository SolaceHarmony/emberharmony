#!/usr/bin/env python
"""Dump a trace from the vendored upstream `liquid_audio/moshi/server.py` loop.

This is the Python half of the realtime Moshi parity harness. It intentionally
keeps the same step order as `server.py`: fixed PCM frame -> Mimi encode ->
LMGen.step -> Mimi decode. Compare its JSON output with the Rust example:

  conda activate py312
  MOSHI_GREEDY=1 MOSHI_TRACE_FRAMES=16 \
    python parity/dump_moshi_realtime.py kyutai/moshiko-pytorch-bf16 \
      assets/question-24khz.wav /tmp/python-moshi.json

The Rust runtime consumes the Candle layout (`kyutai/moshiko-candle-bf16`). When
the Python side sees that layout, it remaps the Candle depformer keys into the
vendored Python module names before loading, so same-checkpoint parity remains
the default contract.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch

from _upstream import SRC

sys.path.insert(0, str(SRC))

from liquid_audio.moshi.models import LMGen, loaders  # noqa: E402
from liquid_audio.moshi.models.loaders import CheckpointInfo  # noqa: E402
from liquid_audio.moshi.run_inference import get_condition_tensors  # noqa: E402


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def file_fingerprint(path: Path) -> dict[str, int | str]:
    value = FNV_OFFSET
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            for byte in chunk:
                value ^= byte
                value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return {
        "path": str(path),
        "bytes": size,
        "fnv1a64": f"{value:016x}",
    }


def quick_file_info(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def safetensors_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def safetensors_keys(path: Path) -> list[str]:
    return list(safetensors_header(path))


def moshi_weight_layout(path: Path) -> str:
    if path.suffix not in (".safetensors", ".sft", ".sfts"):
        return "torch-pickle"
    keys = safetensors_keys(path)
    if any(key.startswith("depformer.") and key.endswith(".linear_in.weight") for key in keys):
        return "candle"
    if any(
        key.startswith(("depformer_in.", "linears.", "depformer_emb."))
        for key in keys
    ):
        return "python"
    return "unknown-safetensors"


def resolve_checkpoint(model: str) -> CheckpointInfo:
    path = Path(model)
    if not path.is_dir():
        return CheckpointInfo.from_hf_repo(model)

    config = path / "config.json"
    if not config.is_file():
        return CheckpointInfo(
            path / loaders.MOSHI_NAME,
            path / loaders.MIMI_NAME,
            path / loaders.TEXT_TOKENIZER_NAME,
        )

    raw = json.loads(config.read_text())
    nested = raw.get("lm_config")
    nested_lm = nested if isinstance(nested, dict) else {}
    lm_config = dict(nested_lm) if nested_lm else dict(raw)

    def pop_string(key: str, default: str | None):
        root = raw.get(key)
        nested = nested_lm.get(key)
        value = root if isinstance(root, str) else nested
        lm_config.pop(key, None)
        return value if isinstance(value, str) else default

    def pop_object(key: str):
        value = raw[key] if key in raw else nested_lm.get(key, {})
        lm_config.pop(key, None)
        return value if isinstance(value, dict) else {}

    moshi_name = pop_string("moshi_name", loaders.MOSHI_NAME)
    mimi_name = pop_string("mimi_name", loaders.MIMI_NAME)
    tokenizer_name = pop_string("tokenizer_name", loaders.TEXT_TOKENIZER_NAME)
    lora_name = pop_string("lora_name", None)
    model_type = pop_string("model_type", "moshi")
    lm_gen_config = pop_object("lm_gen_config")
    tts_config = pop_object("tts_config")
    stt_config = pop_object("stt_config")
    model_id = pop_object("model_id")
    lm_config.pop("lm_config", None)
    return CheckpointInfo(
        path / moshi_name,
        path / mimi_name,
        path / tokenizer_name,
        lm_config,
        raw,
        model_type,
        path / lora_name if lora_name else None,
        lm_gen_config=lm_gen_config,
        tts_config=tts_config,
        stt_config=stt_config,
        model_id=model_id,
    )


def depformer_slice_index(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "depformer" or not parts[1].isdigit():
        return None
    return int(parts[1])


def concat_metadata(values: list[dict], dim: int) -> dict:
    first = dict(values[0])
    shape = list(first["shape"])
    dtype = first["dtype"]
    if any(value["dtype"] != dtype for value in values):
        raise RuntimeError("cannot concatenate safetensors metadata with mixed dtypes")
    if any(len(value["shape"]) != len(shape) for value in values):
        raise RuntimeError("cannot concatenate safetensors metadata with mixed ranks")
    for value in values[1:]:
        for idx, size in enumerate(value["shape"]):
            if idx != dim and size != shape[idx]:
                raise RuntimeError("cannot concatenate safetensors metadata with incompatible shapes")
        shape[dim] += value["shape"][dim]
    first["shape"] = shape
    return first


def remap_candle_moshi_header(header: dict[str, dict]) -> dict[str, dict]:
    """Metadata-only mirror of `remap_candle_moshi_state` for cheap coverage checks."""

    slices = sorted({idx for key in header if (idx := depformer_slice_index(key)) is not None})
    if not slices:
        raise RuntimeError("Candle Moshi state has no depformer slices to remap")

    out: dict[str, dict] = {}
    attention: dict[tuple[int, str], dict[int, dict]] = {}
    for key, value in header.items():
        idx = depformer_slice_index(key)
        if idx is None:
            out[key] = value
            continue

        rest = key.split(".", 2)[2]
        if rest == "emb.weight":
            target = "depformer_text_emb.weight" if idx == 0 else f"depformer_emb.{idx - 1}.weight"
            out[target] = value
            continue
        if rest == "linear_in.weight":
            out[f"depformer_in.{idx}.weight"] = value
            continue
        if rest == "linear_out.weight":
            out[f"linears.{idx}.weight"] = value
            continue

        parts = rest.split(".")
        if (
            len(parts) >= 5
            and parts[0] == "transformer"
            and parts[1] == "layers"
            and parts[2].isdigit()
        ):
            layer = int(parts[2])
            tail = ".".join(parts[3:])
            if tail in ("self_attn.in_proj_weight", "self_attn.out_proj.weight"):
                attention.setdefault((layer, tail), {})[idx] = value
                continue
            if tail.startswith("gating."):
                out[f"depformer.layers.{layer}.gating.{idx}.{tail.removeprefix('gating.')}"] = value
                continue
            if tail in ("norm1.alpha", "norm2.alpha"):
                if idx == 0:
                    out[f"depformer.layers.{layer}.{tail}"] = value
                continue

        raise RuntimeError(f"unmapped Candle Moshi key: {key}")

    for (layer, tail), values in attention.items():
        missing = [idx for idx in slices if idx not in values]
        if missing:
            raise RuntimeError(f"missing depformer slices for layer {layer} {tail}: {missing}")
        out[f"depformer.layers.{layer}.{tail}"] = concat_metadata(
            [values[idx] for idx in slices],
            dim=0,
        )
    return out


def remap_candle_moshi_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Kyutai's Candle Moshi key layout into the vendored Python LM layout."""

    slices = sorted({idx for key in state if (idx := depformer_slice_index(key)) is not None})
    if not slices:
        raise RuntimeError("Candle Moshi state has no depformer slices to remap")

    out: dict[str, torch.Tensor] = {}
    attention: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    for key, value in state.items():
        idx = depformer_slice_index(key)
        if idx is None:
            out[key] = value
            continue

        rest = key.split(".", 2)[2]
        if rest == "emb.weight":
            target = "depformer_text_emb.weight" if idx == 0 else f"depformer_emb.{idx - 1}.weight"
            out[target] = value
            continue
        if rest == "linear_in.weight":
            out[f"depformer_in.{idx}.weight"] = value
            continue
        if rest == "linear_out.weight":
            out[f"linears.{idx}.weight"] = value
            continue

        parts = rest.split(".")
        if (
            len(parts) >= 5
            and parts[0] == "transformer"
            and parts[1] == "layers"
            and parts[2].isdigit()
        ):
            layer = int(parts[2])
            tail = ".".join(parts[3:])
            if tail in ("self_attn.in_proj_weight", "self_attn.out_proj.weight"):
                attention.setdefault((layer, tail), {})[idx] = value
                continue
            if tail.startswith("gating."):
                out[f"depformer.layers.{layer}.gating.{idx}.{tail.removeprefix('gating.')}"] = value
                continue
            if tail in ("norm1.alpha", "norm2.alpha"):
                if idx == 0:
                    out[f"depformer.layers.{layer}.{tail}"] = value
                continue

        raise RuntimeError(f"unmapped Candle Moshi key: {key}")

    for (layer, tail), values in attention.items():
        missing = [idx for idx in slices if idx not in values]
        if missing:
            raise RuntimeError(f"missing depformer slices for layer {layer} {tail}: {missing}")
        out[f"depformer.layers.{layer}.{tail}"] = torch.cat(
            [values[idx] for idx in slices],
            dim=0,
        )
    return out


def load_moshi_for_trace(info: CheckpointInfo, device: str, dtype: torch.dtype, layout: str):
    if layout != "candle":
        return info.get_moshi(device, dtype=dtype).eval(), layout

    from safetensors.torch import load_file

    lm = info.get_moshi(device, dtype=dtype, load_weight=False).eval()
    state = load_file(info.moshi_weights, device=str(device))
    for key, value in state.items():
        if value.dtype.is_floating_point:
            state[key] = value.to(dtype)
    remapped = remap_candle_moshi_state(state)
    del state
    lm.load_state_dict(remapped, assign=True)
    del remapped
    return lm.eval(), "candle-remapped-to-python"


@torch.no_grad()
def warmup(mimi, lm_gen, frame_size: int, device: torch.device | str, frames: int) -> None:
    for _ in range(frames):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def write_metadata_trace(args, info: CheckpointInfo, layout: str) -> None:
    header = safetensors_header(Path(info.moshi_weights)) if layout != "torch-pickle" else {}
    remapped = remap_candle_moshi_header(header) if layout == "candle" else header
    trace = {
        "source": "python",
        "mode": "verify-remap-only",
        "model": args.model,
        "model_type": info.model_type,
        "checkpoint": {
            "moshi": quick_file_info(Path(info.moshi_weights)) | {"layout": layout},
            "mimi": quick_file_info(Path(info.mimi_weights)),
            "tokenizer": quick_file_info(Path(info.tokenizer)),
        },
        "moshi_header_keys": len(header),
        "remapped_moshi_keys": len(remapped),
        "depformer_slices": sorted(
            {idx for key in header if (idx := depformer_slice_index(key)) is not None}
        ),
    }
    args.out.write_text(json.dumps(trace, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF repo id or upstream-compatible model reference")
    parser.add_argument("wav", type=Path, help="input WAV; resampled to Mimi sample rate")
    parser.add_argument("out", type=Path, help="output JSON trace")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=4,
        help="number of empty Mimi/LM frames to run before the trace; server.py parity uses 4",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="load/remap the checkpoint and write metadata without running the realtime step loop",
    )
    parser.add_argument(
        "--verify-remap-only",
        action="store_true",
        help="validate the Candle-to-Python key remap from safetensors metadata only",
    )
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    info = resolve_checkpoint(args.model)
    layout = moshi_weight_layout(Path(info.moshi_weights))
    if layout == "unknown-safetensors":
        raise SystemExit(
            "selected Moshi checkpoint is safetensors but does not look like the "
            "upstream Python Moshi layout; refusing to dump a misleading trace."
        )
    if args.verify_remap_only:
        write_metadata_trace(args, info, layout)
        return

    mimi = info.get_mimi(args.device)
    text = info.get_text_tokenizer()
    lm, layout = load_moshi_for_trace(info, args.device, dtype, layout)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    trace = {
        "source": "python",
        "model": args.model,
        "model_type": info.model_type,
        "checkpoint": {
            "moshi": file_fingerprint(Path(info.moshi_weights)) | {"layout": layout},
            "mimi": file_fingerprint(Path(info.mimi_weights)),
            "tokenizer": file_fingerprint(Path(info.tokenizer)),
        },
        "input": str(args.wav),
        "greedy": bool(args.greedy),
        "sample_rate": int(mimi.sample_rate),
        "frame_size": int(frame_size),
        "warmup_frames": int(args.warmup_frames),
        "input_frames": 0,
        "elapsed_s": 0.0,
        "input_audio_tokens": [],
        "text_tokens": [],
        "text": "",
        "audio_tokens": [],
        "audio_chunks": [],
    }
    if args.load_only:
        trace["mode"] = "load-only"
        args.out.write_text(json.dumps(trace, indent=2))
        return

    condition_tensors = get_condition_tensors(info.model_type, lm, batch_size=1, cfg_coef=args.cfg_coef)
    lm_config = dict(info.lm_gen_config)
    if args.greedy:
        lm_config["use_sampling"] = False
    lm_gen = LMGen(lm, cfg_coef=args.cfg_coef, condition_tensors=condition_tensors, **lm_config)

    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, lm_gen, frame_size, args.device, args.warmup_frames)
    mimi.reset_streaming()
    lm_gen.reset_streaming()

    in_pcms, _ = sphn.read(args.wav, sample_rate=mimi.sample_rate)
    if in_pcms.ndim == 2:
        in_pcms = in_pcms.mean(axis=0)
    all_pcm = np.asarray(in_pcms, dtype=np.float32)

    text_tokens: list[int] = []
    input_audio_tokens: list[list[int]] = []
    audio_tokens: list[list[int]] = []
    audio_chunks: list[dict[str, float | int]] = []
    skip_frames = 1
    frames = 0
    start = time.time()
    with torch.no_grad():
        for offset in range(0, all_pcm.shape[-1] - frame_size + 1, frame_size):
            if args.frames is not None and frames >= args.frames:
                break
            frames += 1
            chunk_np = all_pcm[offset : offset + frame_size]
            chunk = torch.from_numpy(chunk_np).to(device=args.device)[None, None]
            codes = mimi.encode(chunk)
            if skip_frames:
                mimi.reset_streaming()
                skip_frames -= 1
            for c in range(codes.shape[-1]):
                input_audio_tokens.append([int(v) for v in codes[0, :, c].detach().cpu().tolist()])
                tokens = lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                audio_tokens.append([int(v) for v in tokens[0, 1:, 0].detach().cpu().tolist()])
                main_pcm = mimi.decode(tokens[:, 1:]).detach().cpu()[0, 0].numpy()
                token = int(tokens[0, 0, 0].item())
                if token not in (0, 3):
                    text_tokens.append(token)
                audio_chunks.append(
                    {
                        "samples": int(main_pcm.shape[-1]),
                        "rate": int(mimi.sample_rate),
                        "rms": rms(main_pcm),
                        "first": float(main_pcm[0]) if main_pcm.shape[-1] else 0.0,
                    }
                )

    trace["mode"] = "step"
    trace["input_frames"] = int(frames)
    trace["elapsed_s"] = time.time() - start
    trace["input_audio_tokens"] = input_audio_tokens
    trace["text_tokens"] = text_tokens
    trace["text"] = text.decode(text_tokens) if text_tokens else ""
    trace["audio_tokens"] = audio_tokens
    trace["audio_chunks"] = audio_chunks
    args.out.write_text(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
