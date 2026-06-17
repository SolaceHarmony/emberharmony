#!/usr/bin/env python
"""
LFM2.5-Audio runtime — thin wrapper over llama.cpp's `llama-liquid-audio-cli`.

This treats LFM2.5-Audio as a plain local model: speech in, speech/text out, run
on CPU/Metal via llama.cpp. No LiveKit, no network. The exact CLI is taken
verbatim from the official GGUF model card
(https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF):

  ASR:         -sys "Perform ASR."  --audio IN.wav            (text -> stdout)
  TTS:         -sys "Perform TTS."  -p "text" --output OUT.wav
  Interleaved: -sys "Respond with interleaved text and audio." --audio IN.wav --output OUT.wav

The binaries come from an in-progress llama.cpp PR (ggml-org/llama.cpp#18641);
see setup.sh. If the binary or model files are missing, every call raises a
clear, actionable error rather than pretending to work.

Env:
  LFM_BIN        path to llama-liquid-audio-cli (default: found on PATH)
  LFM_MODEL_DIR  dir holding the four GGUF files (default: ./models)
  LFM_QUANT      quantization tag (default: Q4_0)
  LFM_VOICE      TTS system-prompt voice hint, e.g. "Use the US female voice."
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

QUANT = os.environ.get("LFM_QUANT", "Q4_0")
MODEL_DIR = Path(os.environ.get("LFM_MODEL_DIR", Path(__file__).parent / "models"))
VOICE = os.environ.get("LFM_VOICE", "")  # e.g. "Use the UK male voice."


class LfmNotInstalled(RuntimeError):
    """The llama-liquid-audio binary or GGUF files are not available yet."""


def _bin() -> str:
    explicit = os.environ.get("LFM_BIN")
    if explicit:
        if not Path(explicit).exists():
            raise LfmNotInstalled(f"LFM_BIN={explicit} does not exist")
        return explicit
    found = shutil.which("llama-liquid-audio-cli")
    if not found:
        raise LfmNotInstalled(
            "llama-liquid-audio-cli not found. Build it from llama.cpp PR #18641 "
            "and put it on PATH or set LFM_BIN (see setup.sh)."
        )
    return found


@dataclass
class ModelFiles:
    model: Path
    mmproj: Path
    vocoder: Path
    tokenizer: Path

    @classmethod
    def resolve(cls) -> "ModelFiles":
        f = cls(
            model=MODEL_DIR / f"LFM2.5-Audio-1.5B-{QUANT}.gguf",
            mmproj=MODEL_DIR / f"mmproj-LFM2.5-Audio-1.5B-{QUANT}.gguf",
            vocoder=MODEL_DIR / f"vocoder-LFM2.5-Audio-1.5B-{QUANT}.gguf",
            tokenizer=MODEL_DIR / f"tokenizer-LFM2.5-Audio-1.5B-{QUANT}.gguf",
        )
        missing = [p.name for p in (f.model, f.mmproj, f.vocoder, f.tokenizer) if not p.exists()]
        if missing:
            raise LfmNotInstalled(
                f"missing GGUF files in {MODEL_DIR}: {', '.join(missing)} — run setup.sh to download them."
            )
        return f


def _base_cmd(files: ModelFiles) -> list[str]:
    return [
        _bin(),
        "-m", str(files.model),
        "-mm", str(files.mmproj),
        "-mv", str(files.vocoder),
        "--tts-speaker-file", str(files.tokenizer),
    ]


def _run(cmd: list[str], timeout: float = 120.0) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-liquid-audio-cli failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return proc.stdout


def asr(in_wav: str | Path, timeout: float = 120.0) -> str:
    """Speech -> text. Returns the transcript (raw stdout; callers may post-trim)."""
    files = ModelFiles.resolve()
    cmd = _base_cmd(files) + ["-sys", "Perform ASR.", "--audio", str(in_wav)]
    return _run(cmd, timeout=timeout).strip()


def tts(text: str, out_wav: str | Path, voice: str | None = None, timeout: float = 120.0) -> Path:
    """Text -> speech WAV. `voice` overrides LFM_VOICE (e.g. 'Use the US female voice.')."""
    files = ModelFiles.resolve()
    sys_prompt = "Perform TTS."
    v = voice if voice is not None else VOICE
    if v:
        sys_prompt = f"{sys_prompt} {v}"
    cmd = _base_cmd(files) + ["-sys", sys_prompt, "-p", text, "--output", str(out_wav)]
    _run(cmd, timeout=timeout)
    return Path(out_wav)


def interleaved(
    in_wav: str | Path,
    out_wav: str | Path,
    system: str = "Respond with interleaved text and audio.",
    timeout: float = 180.0,
) -> tuple[str, Path]:
    """
    Speech in -> (text reply, speech WAV). This is the conversational mode: LFM2.5
    answers in its own voice and on the text channel at once. The returned text is
    what the orchestrator inspects for the delegate marker.
    """
    files = ModelFiles.resolve()
    cmd = _base_cmd(files) + ["-sys", system, "--audio", str(in_wav), "--output", str(out_wav)]
    text = _run(cmd, timeout=timeout).strip()
    return text, Path(out_wav)


def available() -> bool:
    """True if the binary and all GGUF files are present."""
    try:
        ModelFiles.resolve()
        _bin()
        return True
    except LfmNotInstalled:
        return False


if __name__ == "__main__":
    import sys

    if available():
        print("LFM2.5-Audio runtime ready:")
        print("  bin:", _bin())
        print("  models:", MODEL_DIR)
    else:
        print("LFM2.5-Audio runtime NOT ready — run setup.sh.", file=sys.stderr)
        raise SystemExit(1)
