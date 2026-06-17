#!/usr/bin/env python
"""
Local voice loop — LFM2.5-Audio up front, GLM subagent for the hard work.

The reset architecture (see README.md):

    mic ─► LFM2.5-Audio (interleaved: it converses in its own voice + emits text)
                │
                ├─ no marker ─► play LFM's own spoken reply        (small talk, quick answers)
                │
                └─ "DELEGATE: <task>" on the text channel ─► GLM-5.1 subagent does the work
                                                              └─► LFM2.5-Audio TTS speaks the result

LFM2.5-Audio is the intelligence at the front. Its single "primitive tool" is a
text-marker convention (`DELEGATE: ...`) the orchestrator watches for — because
the audio model has no native function calling. GLM (ollama-cloud/glm-5.1) is the
heavy lifter, reached as a plain API subagent. No LiveKit, no brain bridge.

Env:
  LFM_ROUTE        marker (default) | chat | delegate
  LFM_ALLOW_EXEC   1 to let the GLM subagent run shell commands (default off)
  LFM_SILENCE_SEC  trailing silence that ends an utterance (default 1.0)
  LFM_MAX_SEC      max utterance length (default 20)
  (plus the LFM_* / GLM_* vars documented in lfm_runtime.py and glm_subagent.py)
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import lfm_runtime as lfm
from glm_subagent import run_subagent

SAMPLE_RATE = 16000  # LFM2.5-Audio / FastConformer encoder expects 16 kHz mono input
SILENCE_SEC = float(os.environ.get("LFM_SILENCE_SEC", "1.0"))
MAX_SEC = float(os.environ.get("LFM_MAX_SEC", "20"))
ROUTE = os.environ.get("LFM_ROUTE", "marker")
ALLOW_EXEC = os.environ.get("LFM_ALLOW_EXEC") == "1"

# System prompt for interleaved mode: converse naturally, but hand real work to
# the subagent by emitting a single DELEGATE line on the text channel.
CONVERSE_SYSTEM = (
    "Respond with interleaved text and audio. You are a warm, brief voice assistant. "
    "Chat naturally and answer simple questions yourself in one or two short spoken sentences. "
    "But when the user asks for real engineering, coding, research, or file/system work, do NOT attempt it "
    "yourself. Instead, briefly say you'll get your engineer on it, and on the TEXT channel output exactly one "
    "line of the form: DELEGATE: <a clear, self-contained description of the task>. "
    "Only emit DELEGATE for genuine work, never for small talk."
)

DELEGATE_RE = re.compile(r"DELEGATE:\s*(.+)", re.IGNORECASE)


def record_utterance(path: Path) -> bool:
    """Energy-gated recorder: capture from first speech until SILENCE_SEC of quiet. False if nothing heard."""
    block = int(SAMPLE_RATE * 0.05)  # 50 ms blocks
    silence_blocks = int(SILENCE_SEC / 0.05)
    max_blocks = int(MAX_SEC / 0.05)
    # crude energy threshold; tune per mic. Calibrated against typical ambient.
    threshold = float(os.environ.get("LFM_RMS_THRESHOLD", "0.012"))

    captured: list[np.ndarray] = []
    started = False
    quiet = 0
    print("  …listening (speak)…", end="", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block) as stream:
        for _ in range(max_blocks):
            data, _overflow = stream.read(block)
            mono = data[:, 0]
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-9)
            if rms >= threshold:
                started = True
                quiet = 0
            elif started:
                quiet += 1
            if started:
                captured.append(mono.copy())
                if quiet >= silence_blocks:
                    break
    print(" done.")
    if not started or not captured:
        return False
    audio = np.concatenate(captured)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return True


def play_wav(path: Path) -> None:
    audio, sr = sf.read(str(path), dtype="float32")
    sd.play(audio, sr)
    sd.wait()


def extract_delegation(text: str) -> str | None:
    m = DELEGATE_RE.search(text or "")
    return m.group(1).strip() if m else None


def handle_turn(workdir: Path) -> None:
    in_wav = workdir / "in.wav"
    out_wav = workdir / "out.wav"
    reply_wav = workdir / "reply.wav"

    if not record_utterance(in_wav):
        return

    if ROUTE == "delegate":
        # Pure delegation: transcribe, send everything to GLM, speak the result.
        user_text = lfm.asr(in_wav)
        print(f"  you: {user_text}")
        result = run_subagent(user_text, allow_exec=ALLOW_EXEC, verbose=True)
        print(f"  glm: {result}")
        lfm.tts(result, reply_wav)
        play_wav(reply_wav)
        return

    # marker / chat: LFM converses (interleaved) and may flag work on the text channel.
    text, spoken = lfm.interleaved(in_wav, out_wav, system=CONVERSE_SYSTEM)
    print(f"  lfm(text): {text}")

    task = None if ROUTE == "chat" else extract_delegation(text)
    if task:
        # Acknowledge in LFM's own voice first (if it produced audio), then delegate.
        if spoken.exists() and spoken.stat().st_size > 0:
            play_wav(spoken)
        print(f"  → delegating to GLM: {task}")
        result = run_subagent(task, allow_exec=ALLOW_EXEC, verbose=True)
        print(f"  glm: {result}")
        lfm.tts(result, reply_wav)
        play_wav(reply_wav)
    else:
        # LFM handled it itself — speak its reply.
        play_wav(spoken)


def main() -> int:
    if not lfm.available():
        print(
            "LFM2.5-Audio is not installed yet. Run ./setup.sh to build llama-liquid-audio-cli "
            "(llama.cpp PR #18641) and download the GGUF files, then retry.",
            file=sys.stderr,
        )
        return 1
    print(f"[lfm2-audio voice loop] route={ROUTE} allow_exec={ALLOW_EXEC}. Ctrl-C to quit.")
    with tempfile.TemporaryDirectory(prefix="lfm-voice-") as td:
        workdir = Path(td)
        try:
            while True:
                handle_turn(workdir)
        except KeyboardInterrupt:
            print("\n[lfm2-audio voice loop] bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
