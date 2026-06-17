#!/usr/bin/env bash
#
# Build llama.cpp's liquid-audio runners and download the LFM2.5-Audio GGUFs.
#
# The audio runners (llama-liquid-audio-cli / -server) are from an in-progress
# llama.cpp PR (ggml-org/llama.cpp#18641), so we build that PR head. The GGUF
# weights come from LiquidAI/LFM2.5-Audio-1.5B-GGUF.
#
# Result: models/ holds the four GGUF files; the cli/server binaries land in
# $LLAMA_CPP_DIR/build/bin. Export LFM_BIN to that cli (or put it on PATH).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="$HERE/models"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HERE/llama.cpp}"
PR=18641
QUANT="${LFM_QUANT:-Q4_0}"
REPO="LiquidAI/LFM2.5-Audio-1.5B-GGUF"

echo "==> 1/2  llama.cpp liquid-audio runners (PR #$PR)"
CLI_BIN="$LLAMA_DIR/build/bin/llama-liquid-audio-cli"
if command -v llama-liquid-audio-cli >/dev/null 2>&1; then
  echo "    already on PATH: $(command -v llama-liquid-audio-cli)"
elif [ -x "$CLI_BIN" ]; then
  echo "    already built: $CLI_BIN"
else
  if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  cd "$LLAMA_DIR"
  git fetch origin "pull/$PR/head:lfm-audio-$PR"
  git checkout "lfm-audio-$PR"
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  # Build everything: the WIP PR's exact target names aren't guaranteed, and the
  # liquid-audio binaries land in build/bin alongside the rest.
  cmake --build build --config Release -j
  echo "    built into $LLAMA_DIR/build/bin"
  if [ ! -x "$CLI_BIN" ]; then
    echo "    WARNING: llama-liquid-audio-cli not found in build/bin — check the PR's target name." >&2
    ls "$LLAMA_DIR/build/bin" | grep -i liquid || true
  fi
fi
echo "    -> export LFM_BIN=\"$CLI_BIN\"   (or add $LLAMA_DIR/build/bin to PATH)"

echo "==> 2/2  GGUF weights ($QUANT) into $MODELS"
mkdir -p "$MODELS"
files=(
  "LFM2.5-Audio-1.5B-$QUANT.gguf"
  "mmproj-LFM2.5-Audio-1.5B-$QUANT.gguf"
  "vocoder-LFM2.5-Audio-1.5B-$QUANT.gguf"
  "tokenizer-LFM2.5-Audio-1.5B-$QUANT.gguf"
)
for f in "${files[@]}"; do
  if [ -f "$MODELS/$f" ]; then
    echo "    have $f"
  else
    echo "    downloading $f"
    curl -fL "https://huggingface.co/$REPO/resolve/main/$f" -o "$MODELS/$f"
  fi
done

echo
echo "Done. Quick check:"
echo "  export LFM_BIN=\"$CLI_BIN\""
echo "  python lfm_runtime.py        # confirms binary + models resolve"
echo "  python voice_loop.py         # start talking"
