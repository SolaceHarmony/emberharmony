#!/usr/bin/env python
"""Dump a trace from the vendored upstream `liquid_audio/moshi/server.py` loop.

This is the Python half of the realtime Moshi parity harness. It intentionally
keeps the same step order as `server.py`: fixed PCM frame -> Mimi encode ->
LMGen.step -> Mimi decode. Compare its JSON output with the Rust example:

  conda activate py312
  MOSHI_GREEDY=1 MOSHI_TRACE_FRAMES=16 \
    python parity/dump_moshi_realtime.py kyutai/moshiko-pytorch-bf16 \
      assets/question-24khz.wav /tmp/python-moshi.json

The Rust runtime currently consumes the Candle layout (`kyutai/moshiko-candle-bf16`).
Exact parity requires equivalent converted weights; this script is the reference
trace source for that converted-pair check.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch

from _upstream import SRC

sys.path.insert(0, str(SRC))

from liquid_audio.moshi.models import LMGen  # noqa: E402
from liquid_audio.moshi.models.loaders import CheckpointInfo  # noqa: E402
from liquid_audio.moshi.run_inference import get_condition_tensors  # noqa: E402


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def warmup(mimi, lm_gen, frame_size: int, device: torch.device | str) -> None:
    for _ in range(4):
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
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    info = CheckpointInfo.from_hf_repo(args.model)
    mimi = info.get_mimi(args.device)
    text = info.get_text_tokenizer()
    lm = info.get_moshi(args.device, dtype=dtype).eval()
    condition_tensors = get_condition_tensors(info.model_type, lm, batch_size=1, cfg_coef=args.cfg_coef)
    lm_config = dict(info.lm_gen_config)
    if args.greedy:
        lm_config["use_sampling"] = False
    lm_gen = LMGen(lm, cfg_coef=args.cfg_coef, condition_tensors=condition_tensors, **lm_config)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    warmup(mimi, lm_gen, frame_size, args.device)
    mimi.reset_streaming()
    lm_gen.reset_streaming()

    in_pcms, _ = sphn.read(args.wav, sample_rate=mimi.sample_rate)
    if in_pcms.ndim == 2:
        in_pcms = in_pcms.mean(axis=0)
    all_pcm = np.asarray(in_pcms, dtype=np.float32)

    text_tokens: list[int] = []
    audio_chunks: list[dict[str, float | int]] = []
    skip_frames = 1
    frames = 0
    start = time.time()
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
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
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

    trace = {
        "source": "python",
        "model": args.model,
        "input": str(args.wav),
        "greedy": bool(args.greedy),
        "sample_rate": int(mimi.sample_rate),
        "frame_size": int(frame_size),
        "input_frames": int(frames),
        "elapsed_s": time.time() - start,
        "text_tokens": text_tokens,
        "text": text.decode(text_tokens) if text_tokens else "",
        "audio_chunks": audio_chunks,
    }
    args.out.write_text(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
