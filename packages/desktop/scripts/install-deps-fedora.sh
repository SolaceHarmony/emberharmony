#!/usr/bin/env bash
#
# Install the system libraries required to BUILD the EmberHarmony desktop app
# (Tauri 2 + GTK3 / WebKitGTK) from source on Fedora.
#
# These are build-time dependencies: the headers and pkg-config (.pc) files the
# Rust GTK/WebKit crates link against (glib-sys, webkit2gtk, soup3-sys, …). End
# users who install the .rpm do NOT need this — the package declares its runtime
# deps and the package manager pulls them in. Only people building from source
# (e.g. Fedora users, who have no prebuilt .rpm-to-binary path) need it.
#
# CI installs the Debian equivalents on its Ubuntu runners
# (libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf); this is
# the Fedora/dnf mirror of that step.
#
# Usage (from the repo root):
#   ./packages/desktop/scripts/install-deps-fedora.sh
#
set -euo pipefail

# --- 1. Confirm this is a Fedora / RHEL-like host -------------------------
if [[ ! -r /etc/os-release ]]; then
  echo "error: /etc/os-release not found — cannot identify the distribution." >&2
  exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
if [[ "${ID:-}" != "fedora" && "${ID_LIKE:-}" != *fedora* && "${ID_LIKE:-}" != *rhel* ]]; then
  echo "error: this script targets Fedora / RHEL-like distros (dnf); detected '${PRETTY_NAME:-${ID:-unknown}}'." >&2
  echo "       Debian/Ubuntu equivalent:" >&2
  echo "         sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf" >&2
  exit 1
fi
if [[ "${ID:-}" != "fedora" ]]; then
  echo "warning: detected RHEL-like '${PRETTY_NAME:-${ID:-}}' (not Fedora) — package names may differ and" >&2
  echo "         some (e.g. webkit2gtk4.1-devel) may require EPEL/CRB. Proceeding anyway." >&2
fi

# --- 2. Choose privilege escalation ---------------------------------------
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "error: must run as root, or have sudo installed, to install packages." >&2
    exit 1
  fi
fi

# --- 3. Build-time dependencies -------------------------------------------
# webkit2gtk4.1-devel pulls most of the GTK stack transitively; the rest are
# named explicitly so a missing one fails loudly here instead of 5 minutes into
# a cargo compile (which is how `glib-sys` surfaces it).
packages=(
  gcc gcc-c++ make            # C/C++ toolchain for native (-sys) crate build scripts
  webkit2gtk4.1-devel         # Tauri 2 webview — webkit2gtk-4.1.pc (pulls gtk3/glib/gdk/pango/cairo)
  gtk3-devel                  # gtk+-3.0.pc
  glib2-devel                 # glib-2.0 / gobject-2.0 / gio-2.0 — the crate that fails without it
  libsoup3-devel              # soup-3.0.pc
  librsvg2-devel              # SVG/icon rendering (mirrors librsvg2-dev)
  openssl-devel               # Rust openssl-sys
  patchelf                    # Tauri Linux bundling (mirrors CI)
  rpm-build                   # native rpmbuild — scripts/build-rpm.ts streams the .rpm (not Tauri's in-RAM crate)
  pkgconf-pkg-config          # pkg-config itself
  curl wget file              # download/inspect helpers used during bundling
)

echo "[deps] installing Tauri build dependencies via dnf…"
$SUDO dnf install -y "${packages[@]}"

# System-tray indicator. The package was renamed across Fedora releases
# (libayatana-* is current; libappindicator-* is the legacy name), so try both
# and don't abort the whole setup if neither resolves — the tray is optional.
echo "[deps] installing system-tray indicator (best effort)…"
$SUDO dnf install -y libayatana-appindicator-gtk3-devel \
  || $SUDO dnf install -y libappindicator-gtk3-devel \
  || echo "[deps] warning: no appindicator -devel package found; tray icon support may be limited." >&2

# --- 4. Verify the core pkg-config entries resolve ------------------------
echo "[deps] verifying pkg-config can see the core libraries…"
ok=1
for pc in glib-2.0 gobject-2.0 gio-2.0 gtk+-3.0 webkit2gtk-4.1 libsoup-3.0; do
  if pkg-config --exists "$pc" 2>/dev/null; then
    printf '  ok    %-16s %s\n' "$pc" "$(pkg-config --modversion "$pc")"
  else
    printf '  MISS  %s\n' "$pc"
    ok=0
  fi
done

if [[ "$ok" -ne 1 ]]; then
  echo "error: some libraries are still missing — the build will fail. See MISS lines above." >&2
  exit 1
fi

echo "[deps] all dependencies present. You can now build:  bun run desktop:build"
