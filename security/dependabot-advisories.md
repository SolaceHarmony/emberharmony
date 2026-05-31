# Dependabot Advisories — Status & Remediation

Tracking for every open Dependabot advisory. We don't ignore advisories; the ones
that can't be patched yet are recorded here with the verified blocker, real-world
exposure, and the remediation path so they're tracked rather than forgotten.

Last reviewed: 2026-05-30.

## Fixed (on `dev`)

| Package | Sev | Fix |
|---|---|---|
| turbo | low+med | `2.5.6 → 2.9.16` |
| nitro | med (×2) | `3.0.1-alpha.1 → 3.0.260522-beta` (console/app + enterprise) |
| tar (rust) | med | `0.4.45 → 0.4.46` |
| rand (rust) | low | runtime copies `0.8.5 → 0.8.6`, `0.9.2 → 0.9.3` |

## Open — no shippable fix yet (do not silently drop)

### 1. glib 0.18.5 — medium — `packages/desktop/src-tauri` (Rust)
- **Blocker:** pinned by `webkit2gtk = "=2.0.2"`, the Linux GTK webview binding. The
  latest published `webkit2gtk` crate *is* 2.0.2 — it's effectively abandoned and
  ships no glib-0.20 binding. Tauri 2.11.2 (latest 2.x) does not move it.
- **Exposure:** Linux desktop builds only. macOS (WKWebView) and Windows (WebView2)
  pull no glib at all — see the `cfg(target_os = "linux")` deps in `Cargo.toml`.
- **Remediation:** upgrade when the Tauri/wry Linux webview stack adopts a
  glib-0.20 binding (likely a Tauri major), or stop shipping the GTK webview. There
  is no in-tree workaround — our own glib usage in `window_customizer.rs` is
  re-exported *through* webkit2gtk.

### 2. rand 0.7.3 — low — `packages/desktop/src-tauri` (Rust)
- **Blocker:** build-time transitive — `phf_generator 0.8` (pins `rand 0.7`) ←
  `kuchikiki` (Tauri's HTML-parser fork) ← `tauri-utils` ← `tauri`.
- **Exposure:** compile-time codegen only; not present in the shipped binary.
- **Remediation:** clears when Tauri updates the `kuchikiki`/`phf` chain. (The
  runtime `rand` copies were already bumped to 0.8.6 / 0.9.3.)

### 3. @ai-sdk/provider-utils ≤ 3.0.97 — low — `packages/emberharmony` (npm)
- **Blocker:** only fixed in v4, which is **AI-SDK v6**. The repo is pinned to
  `ai@5.0.119`; the SDK is used across ~27 files plus ~10 provider packages
  (`@ai-sdk/openai`, `@ai-sdk/xai`, `@ai-sdk/mistral`, …).
- **Exposure:** the agent's core model layer.
- **Remediation:** a coordinated AI-SDK v5 → v6 migration, developed and
  real-model-tested on its own branch before merging — not a drop-in bump.
