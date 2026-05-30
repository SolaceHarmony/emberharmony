# Building EmberHarmony

This document covers the build pipeline: how artifacts are produced locally, in CI, and at release time. For day-to-day development (`bun dev`, running the API/web/desktop apps) see [CONTRIBUTING.md](./CONTRIBUTING.md). For macOS code signing and notarization specifics, see [APPLE.md](./APPLE.md).

## Artifacts

EmberHarmony ships two artifacts:

1. **CLI binary** — `emberharmony` (`.exe` on Windows). Cross-compiled from TypeScript using `Bun.build({ compile: ... })`. Produced for 11 platform/variant combinations.
2. **Tauri desktop app** — native installer/bundle that wraps the web UI and embeds the CLI binary as a Tauri sidecar. Five platform targets.

## Build paths

There are three ways the build runs, all driven by the same two underlying scripts.

| Path | Trigger | Entry | Output |
|---|---|---|---|
| **Local desktop** | `bun desktop:build` from repo root | `packages/desktop/scripts/build-local.ts` | Signed `.app` + `.dmg` (macOS), `.msi`/`.nsis` (Windows), `.deb`/`.rpm`/`.AppImage` (Linux) under `packages/desktop/src-tauri/target/release/bundle/` |
| **Local CLI only** | `./packages/emberharmony/script/build.ts --single` | Same script, single-platform mode | `packages/emberharmony/dist/emberharmony-<platform>/bin/emberharmony` |
| **CI verification** | Push to `main` or `dev` | `.github/workflows/ci.yml` → `_build.yml` | Workflow artifacts (CLI dist + desktop bundles per platform) |
| **Release** | `gh workflow run publish.yml` or GitHub release event | `.github/workflows/publish.yml` → `_build.yml` | npm publish + GitHub release with attached desktop installers |

## Local builds

### CLI binary

For your current platform only:

```bash
./packages/emberharmony/script/build.ts --single
```

Output: `packages/emberharmony/dist/emberharmony-<platform>/bin/emberharmony` (e.g. `emberharmony-darwin-arm64`).

To cross-compile for all 11 targets (slow, used in CI):

```bash
./packages/emberharmony/script/build.ts
```

### Desktop app

From repo root:

```bash
bun desktop:build              # Full build with bundling (DMG on macOS, etc.)
bun desktop:build:nodmg        # Skip macOS DMG packaging
```

`build-local.ts` orchestrates the pipeline:

1. Load `.env` from repo root (for Apple signing/notarization creds).
2. Build the CLI binary (`script/build.ts --single`), unless `EMBERHARMONY_SKIP_CLI=1`.
3. Copy the CLI binary into `packages/desktop/src-tauri/sidecars/` for Tauri to embed.
4. Run `cargo tauri build` with platform-appropriate bundle list. macOS DMG is skipped here because Tauri's upstream `bundle_dmg.sh` is broken.
5. On macOS, manually create the installer DMG via `hdiutil`, then `codesign` it if `APPLE_SIGNING_IDENTITY` is set.

Prerequisites: Bun 1.3+, Rust stable, Tauri prerequisites for your OS (see https://v2.tauri.app/start/prerequisites/).

## CI build pipeline

### Reusable workflow: `.github/workflows/_build.yml`

The build is defined once and called by two workflows. `_build.yml` is a `workflow_call` workflow that takes:

- `version` (required) — semver string the artifacts carry
- `release` (default `""`) — GitHub release ID to attach artifacts to
- `tag` (default `""`) — git tag name for the release

When `release` and `tag` are empty, `tauri-action` produces workflow artifacts only — no release attachment. When both are set, it uploads to the specified GitHub release draft.

Two jobs:

- **`build-cli`** (ubuntu-latest) — runs `./packages/emberharmony/script/build.ts` to cross-compile the CLI for all 11 targets in one Linux job. Uploads `emberharmony-cli` artifact.
- **`build-tauri`** (matrix) — depends on `build-cli`. Downloads the CLI artifact, runs `packages/desktop/scripts/prepare.ts` to stage the sidecar and inject updater keys, then `tauri-apps/tauri-action@v0.6` builds the platform bundle.

### Matrix

| Host | Target | Output |
|---|---|---|
| `macos-latest` | `x86_64-apple-darwin` | `.app`/`.dmg` Intel |
| `macos-latest` | `aarch64-apple-darwin` | `.app`/`.dmg` Apple Silicon |
| `windows-latest` | `x86_64-pc-windows-msvc` | `.nsis`/`.msi` |
| `ubuntu-24.04` | `x86_64-unknown-linux-gnu` | `.deb`/`.rpm`/`.AppImage` |
| `ubuntu-24.04-arm` | `aarch64-unknown-linux-gnu` | `.deb`/`.rpm`/`.AppImage` |

### Verification builds: `ci.yml`

Triggers:

- `push` to `main` or `dev` (i.e. after a PR merges into an integration branch)
- `workflow_dispatch` (manual re-run from the Actions UI)

Notably, **not** `pull_request`. That keeps the expensive matrix off PR review, prevents fork code from being executed with secrets, and bounds Actions-minute spend to merge rate rather than push rate.

If a merge-triggered run fails, re-run it from the Actions UI via `workflow_dispatch` without making another commit.

### Release publish: `publish.yml`

Triggered **solely by a published GitHub release** (`release` event, `types: [published]`). There is no `workflow_dispatch` path, and nothing in CI commits, tags, or pushes — the version travels with the code and the tag is created by the release you cut.

Job flow:

1. **`version`** — extracts the version from the release tag (`refs/tags/v<x>`) and fails fast unless it matches `packages/emberharmony/package.json`.
2. **`build`** — calls `_build.yml` with the `version`, `release` ID, and `tag`. Same 11-CLI + 5-Tauri matrix as CI, with artifacts attached to the release.
3. **`publish`** — runs `./script/publish.ts` to publish `@thesolaceproject/emberharmony` (and the 11 platform npm packages) to npmjs.org.

To cut a release:

```bash
# 1. Bump "version" in packages/emberharmony/package.json and commit it.
# 2. Create a GitHub release whose tag matches that version:
gh release create v1.3.0 --title v1.3.0 --generate-notes
```

## Signing and notarization

Signing is enabled per-platform when the relevant secrets are set on the repo. All signing steps gracefully no-op when secrets are missing — useful for fork PRs and local builds.

### macOS

| Secret | Purpose |
|---|---|
| `APPLE_CERTIFICATE` | base64-encoded Developer ID Application `.p12` |
| `APPLE_CERTIFICATE_PASSWORD` | password for the `.p12` |
| `APPLE_SIGNING_IDENTITY` | (optional) explicit signing identity string, e.g. `Developer ID Application: Org Name (TEAMID)` |
| `APPLE_API_ISSUER` + `APPLE_API_KEY` + `APPLE_API_KEY_PATH` | App Store Connect API key (preferred notarization auth) |
| `APPLE_ID` + `APPLE_PASSWORD` + `APPLE_TEAM_ID` | Apple ID + app-specific password (fallback notarization auth) |

`APPLE_TEAM_ID` is auto-derived from the signing identity by parsing the `(TEAMID)` suffix when not explicitly set. See [APPLE.md](./APPLE.md) for the full notarization flow.

### Tauri updater

Signs the update manifests so the auto-updater accepts them.

| Secret | Purpose |
|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | Tauri updater private key |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | password for the private key |

### npm publish

| Secret | Purpose |
|---|---|
| `NPM_TOKEN` | npmjs.org automation token with publish rights on `@thesolaceproject` |

### Listing what's set

```bash
gh secret list
```

Returns names only — secret values can never be read back.

## CodeQL

`.github/workflows/codeql.yml` runs CodeQL on push to `main`/`dev`, on PRs to those branches, and weekly on Monday at 04:23 UTC. Two language groups are scanned:

- **`javascript-typescript`** — the entire TS/JS codebase.
- **`actions`** — scans `.github/workflows/**` for secret-handling and injection bugs. Worth keeping after the prior fork-PR-secret incident (commit `5569e76`).

This replaces GitHub's "default setup" CodeQL. The workflow file gives explicit control over languages, paths, and schedule.

> [!IMPORTANT]
> If you re-enable "Default setup" in repo Settings → Code security, it will conflict with `codeql.yml`. Pick one.

## Workflow files reference

| File | Purpose |
|---|---|
| `.github/workflows/_build.yml` | Reusable build workflow. Builds CLI + Tauri matrix. |
| `.github/workflows/ci.yml` | Post-merge verification on `main`/`dev`. Calls `_build.yml`. |
| `.github/workflows/publish.yml` | Release pipeline. Calls `_build.yml`, then publishes to npm + GitHub release. |
| `.github/workflows/codeql.yml` | Security analysis on TS/JS and workflow files. |
| `.github/workflows/test.yml` | Test suite (Bun) on PRs. |
| `.github/workflows/typecheck.yml` | `bun turbo typecheck` on push + PRs. |

## Useful commands

```bash
# What is actually being run on GitHub?
gh workflow list
gh workflow view ci.yml
gh run list --workflow=ci.yml

# Validate workflow syntax locally
actionlint .github/workflows/*.yml

# Trigger a CI run on the current branch from your laptop
gh workflow run ci.yml --ref <branch>

# Cut a release
gh workflow run publish.yml
```
