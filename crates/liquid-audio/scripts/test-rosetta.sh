#!/bin/bash
# Local Apple-Silicon x86 lane. This deliberately stays out of GitHub Actions:
# it cross-builds Darwin x86_64 artifacts, then makes Cargo launch every test
# binary through Rosetta instead of relying on implicit translation.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"

case "${1:-}" in
    "") REQUIRE_AVX2=0 ;;
    --require-avx2) REQUIRE_AVX2=1 ;;
    *)
        echo "usage: $0 [--require-avx2]" >&2
        exit 2
        ;;
esac

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "rosetta gate: requires Apple Silicon macOS" >&2
    exit 1
fi

if ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    echo "rosetta gate: Rosetta is not installed" >&2
    echo "install with: softwareupdate --install-rosetta --agree-to-license" >&2
    exit 1
fi

if ! rustup target list --installed | grep -qx x86_64-apple-darwin; then
    echo "rosetta gate: Rust x86_64 target is not installed" >&2
    echo "install with: rustup target add x86_64-apple-darwin" >&2
    exit 1
fi

FEATURES=$(/usr/bin/arch -x86_64 /usr/sbin/sysctl -n machdep.cpu.features 2>/dev/null || true)
case " $FEATURES " in
    *" AVX2 "*) echo "rosetta gate: AVX2 is exposed; SIMD correctness tests will execute" ;;
    *)
        if [ "$REQUIRE_AVX2" -eq 1 ]; then
            echo "rosetta gate: AVX2 is not exposed; refusing a false-green SIMD gate" >&2
            exit 1
        fi
        echo "rosetta gate: AVX2 is not exposed; SIMD tests will skip, but x86 build/link/ABI tests still execute"
        ;;
esac

RUNNER='target.x86_64-apple-darwin.runner = ["/usr/bin/arch", "-x86_64"]'
TARGET=x86_64-apple-darwin

echo "== [1/3] kcoro x86 ticket and wait-word tests =="
cargo --config "$RUNNER" test -p kcoro-sys --target "$TARGET" --tests -- --nocapture

echo "== [2/3] candle-flashfftconv x86 tests =="
cargo --config "$RUNNER" test -p candle-flashfftconv --target "$TARGET" -- --nocapture

echo "== [3/3] liquid-audio x86 library tests =="
cargo --config "$RUNNER" test -p liquid-audio --target "$TARGET" --lib -- --nocapture

echo "== Rosetta x86 gate: ALL GREEN =="
