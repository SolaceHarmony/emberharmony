#!/usr/bin/env python
"""Compare Rust and Python realtime Moshi traces.

Generate inputs with:

  conda activate py312
  python parity/dump_moshi_realtime.py <python-model> input.wav /tmp/py.json --greedy --frames 16 --warmup-frames 4
  MOSHI_GREEDY=1 MOSHI_TRACE_FRAMES=16 MOSHI_WARMUP_FRAMES=4 MOSHI_SEED=42424242 \
    cargo run --release --example moshi_realtime_trace -- <candle-model-dir> input.wav /tmp/rs.json
  python parity/compare_moshi_realtime.py /tmp/py.json /tmp/rs.json

The checkpoint names may differ when one side is a converted Candle snapshot, but
that must be acknowledged with `--allow-converted-checkpoints`. By default, this
comparator requires matching checkpoint byte fingerprints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-6)


def assert_same_checkpoints(py: dict, rs: dict) -> None:
    py_checkpoint = py.get("checkpoint")
    rs_checkpoint = rs.get("checkpoint")
    assert py_checkpoint and rs_checkpoint, "both traces must include checkpoint fingerprints"
    for key in ("moshi", "mimi", "tokenizer"):
        a = py_checkpoint[key]
        b = rs_checkpoint[key]
        assert a["bytes"] == b["bytes"], {
            "checkpoint": key,
            "python_bytes": a["bytes"],
            "rust_bytes": b["bytes"],
        }
        assert a["fnv1a64"] == b["fnv1a64"], {
            "checkpoint": key,
            "python_hash": a["fnv1a64"],
            "rust_hash": b["fnv1a64"],
        }


def assert_step_trace(name: str, trace: dict) -> None:
    mode = trace.get("mode")
    assert mode == "step", {
        "trace": name,
        "mode": mode,
        "message": "Moshi parity requires realtime stepping traces, not load/remap metadata",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_trace", type=Path)
    parser.add_argument("rust_trace", type=Path)
    parser.add_argument("--rms-rtol", type=float, default=5e-2)
    parser.add_argument(
        "--allow-converted-checkpoints",
        action="store_true",
        help="Skip byte-fingerprint equality when comparing an explicitly converted PyTorch/Candle pair.",
    )
    args = parser.parse_args()

    py = load(args.python_trace)
    rs = load(args.rust_trace)

    assert_step_trace("python", py)
    assert_step_trace("rust", rs)
    if not args.allow_converted_checkpoints:
        assert_same_checkpoints(py, rs)
    assert py.get("greedy") == rs.get("greedy"), {
        "python": py.get("greedy"),
        "rust": rs.get("greedy"),
        "message": "Moshi traces must use the same sampling mode",
    }
    assert py.get("seed") == rs.get("seed"), {
        "python": py.get("seed"),
        "rust": rs.get("seed"),
        "message": "Moshi traces must use the same sampling seed",
    }
    assert float(py.get("cfg_coef", 1.0)) == float(rs.get("cfg_coef", 1.0)), {
        "python": py.get("cfg_coef"),
        "rust": rs.get("cfg_coef"),
        "message": "Rust Moshi parity only covers the unconditioned cfg_coef=1 path",
    }
    assert py["sample_rate"] == rs["sample_rate"], (py["sample_rate"], rs["sample_rate"])
    assert py["frame_size"] == rs["frame_size"], (py["frame_size"], rs["frame_size"])
    assert py.get("warmup_frames") == rs.get("warmup_frames"), (
        py.get("warmup_frames"),
        rs.get("warmup_frames"),
    )
    assert py["input_frames"] == rs["input_frames"], (py["input_frames"], rs["input_frames"])
    assert py["input_audio_tokens"] == rs["input_audio_tokens"], {
        "python": py["input_audio_tokens"],
        "rust": rs["input_audio_tokens"],
    }
    assert py["text_tokens"] == rs["text_tokens"], {
        "python": py["text_tokens"],
        "rust": rs["text_tokens"],
    }
    assert py["audio_tokens"] == rs["audio_tokens"], {
        "python": py["audio_tokens"],
        "rust": rs["audio_tokens"],
    }

    py_audio = py["audio_chunks"]
    rs_audio = rs["audio_chunks"]
    assert len(py_audio) == len(rs_audio), (len(py_audio), len(rs_audio))
    for i, (a, b) in enumerate(zip(py_audio, rs_audio)):
        assert a["samples"] == b["samples"], (i, a["samples"], b["samples"])
        assert a["rate"] == b["rate"], (i, a["rate"], b["rate"])
        err = rel(float(b["rms"]), float(a["rms"]))
        assert err <= args.rms_rtol, {
            "chunk": i,
            "python_rms": a["rms"],
            "rust_rms": b["rms"],
            "relative_error": err,
            "tolerance": args.rms_rtol,
        }

    print(
        "moshi realtime parity ok: "
        f"{py['input_frames']} input frames, "
        f"{len(py['text_tokens'])} text tokens, "
        f"{len(py_audio)} audio chunks"
    )


if __name__ == "__main__":
    main()
