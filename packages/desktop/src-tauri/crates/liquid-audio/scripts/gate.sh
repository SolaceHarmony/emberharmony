#!/bin/bash
# The liquid-audio gate stack — run from the crate directory. Every rung of engine
# work passes ALL of these before it lands (ENGINE_DESIGN.md verification contract):
#
#   1. full release suite
#   2. byte oracle, reference chain  (must print 2f9c907aad76919839993d9d92a53304b72f7608)
#   3. byte oracle, perf chain       (must print 45125c9e9206d27f2222f4dfd69bcb4a3b0741e4)
#   4. audible two-turn e2e, CPU     (speaker required; drains before teardown)
#   5. audible two-turn e2e, Metal   (same clip, same stack — the dual-path contract:
#                                     both device choices ship, both get the gate)
#
# Perf-chain hash note: it is machine-tied (lane counts) — re-arm it in this file and
# DECODE_ENGINE.md together, never alone.
set -euo pipefail
cd "$(dirname "$0")/.."

REF_HASH="2f9c907aad76919839993d9d92a53304b72f7608"
PERF_HASH="45125c9e9206d27f2222f4dfd69bcb4a3b0741e4"

# Model dir: env override, else the HF cache snapshot the examples resolve to.
if [ -z "${LFM_MODEL_DIR:-}" ]; then
    LFM_MODEL_DIR=$(find ~/.cache/huggingface/hub -maxdepth 4 -type d \
        -path "*LFM2.5-Audio-1.5B*/snapshots/*" 2>/dev/null | head -1)
fi
[ -n "$LFM_MODEL_DIR" ] || { echo "gate: no model dir (set LFM_MODEL_DIR)"; exit 1; }
export LFM_MODEL_DIR

echo "== [1/5] release suite =="
cargo test --release --lib

echo "== [2/5] byte oracle: reference chain =="
LFM_DEVICE=cpu cargo run --release --example generate -- --reference >/dev/null 2>&1
got=$(shasum out.wav | awk '{print $1}')
[ "$got" = "$REF_HASH" ] || { echo "REF ORACLE MISMATCH: $got"; exit 1; }
echo "ref oracle exact: $got"

echo "== [3/5] byte oracle: perf chain =="
LFM_DEVICE=cpu cargo run --release --example generate >/dev/null 2>&1
got=$(shasum out.wav | awk '{print $1}')
[ "$got" = "$PERF_HASH" ] || { echo "PERF ORACLE MISMATCH: $got"; exit 1; }
echo "perf oracle exact: $got"

echo "== [4/5] audible e2e: CPU (two turns through the real speaker) =="
LFM_DEVICE=cpu cargo test --release --features audio-io --test e2e_voice_runtime -- --nocapture

echo "== [5/5] audible e2e: Metal =="
LFM_DEVICE=metal cargo test --release --features metal,audio-io --test e2e_voice_runtime -- --nocapture

echo "== gate stack: ALL GREEN =="
