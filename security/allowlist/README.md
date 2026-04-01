# Allowlist (Non-Secrets That Trigger Scanners)

This folder documents files that routinely trigger secret scanners but are **not credentials**.

Use this as a reference when:

- triaging GitHub Secret Scanning alerts
- configuring third-party scanners (gitleaks, trufflehog, etc.)
- deciding which values should be injected via `.env` or GitHub Actions secrets instead of committed

## Files

### `flake.lock`

- Contains Nix `narHash` values in SRI format (`sha256-...`).
- These are integrity hashes used for reproducible builds. Not secrets.

### `nix/hashes.json`

- Contains Nix fixed-output hashes for platform-specific `node_modules` derivations (`sha256-...`).
- These are integrity hashes used for reproducible builds. Not secrets.

### `packages/desktop/src-tauri/tauri.prod.conf.json`

- The Tauri updater `pubkey` can be a long base64 string and can be flagged as a "password".
- The **public key is not a secret**. The private key (used to sign updates) must never be committed.
- Preferred approach in this repo: set the pubkey at build time via environment (`TAURI_UPDATER_PUBKEY`),
  so the committed config does not contain base64 blobs.

### `github/README.md`

- Avoid putting real-looking API keys in docs, even as examples.
- Use placeholders like `sk-ant-...` / `github_pat_...` / `pk_test_...`.

## Suggested Environment Variables

These should come from GitHub Actions secrets (CI) or a local `.env` file (dev):

- `ANTHROPIC_API_KEY` (secret)
- `GITHUB_TOKEN` / `MOCK_TOKEN` (secret)
- `VITE_STRIPE_PUBLISHABLE_KEY` (public-ish, but still avoid hardcoding)
- `TAURI_UPDATER_PUBKEY` (public key; keep out of repo to avoid false positives)
- `TAURI_SIGNING_PRIVATE_KEY` (secret)
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (secret, if used)
