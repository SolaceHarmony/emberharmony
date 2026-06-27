"""Locate the vendored upstream Python reference (`upstream-liquid-audio`).

The Rust crate was moved out of `experiments/lfm2-audio-voice/` into the desktop build
(`packages/desktop/src-tauri/crates/liquid-audio`), but the Python reference it is a port of
**stays** in `experiments/lfm2-audio-voice/upstream-liquid-audio` — it is reference material,
not shippable code. So the old `HERE.parent.parent / "upstream-liquid-audio"` /
`CRATE.parent / "upstream-liquid-audio"` sibling assumption is broken; every parity/dump/audit
script resolves the reference through here instead.

Resolution order:
1. `$LFM2_UPSTREAM_SRC` if set and present (explicit override, e.g. a checkout elsewhere).
2. Walk up from this file to the repo root (the dir containing `.git`) and take the fixed
   `experiments/lfm2-audio-voice/upstream-liquid-audio/src`.

Fails loudly with a clear message rather than silently reporting 0/0 coverage.
"""

import os
from pathlib import Path


def _resolve_src() -> Path:
    env = os.environ.get("LFM2_UPSTREAM_SRC")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / ".git").exists():  # repo root (.git is a dir for a worktree's main, a file for a linked worktree)
            cand = d / "experiments" / "lfm2-audio-voice" / "upstream-liquid-audio" / "src"
            if cand.exists():
                return cand
    raise FileNotFoundError(
        "upstream-liquid-audio not found. The crate moved into the desktop build but the "
        "Python reference stays in experiments/lfm2-audio-voice/upstream-liquid-audio. "
        "Set $LFM2_UPSTREAM_SRC to <checkout>/upstream-liquid-audio/src to override."
    )


# Computed once on import. `SRC` is `.../upstream-liquid-audio/src` (for sys.path); `PKG` is
# the `liquid_audio` package dir (for direct file references).
SRC = _resolve_src()
PKG = SRC / "liquid_audio"
