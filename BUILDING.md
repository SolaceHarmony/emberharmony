# Building EmberHarmony

This document covers the build pipeline: how artifacts are produced locally, in CI, and at release time. For day-to-day development (`bun dev`, running the API/web/desktop apps) see [CONTRIBUTING.md](./CONTRIBUTING.md). For macOS code signing and notarization specifics, see [APPLE.md](./APPLE.md).

## Artifacts

EmberHarmony ships two artifacts:

1. **CLI binary** — `emberharmony` (`.exe` on Windows). Cross-compiled from TypeScript using `Bun.build({ compile: ... })`. Produced for 11 platform/variant combinations.
2. **Tauri desktop app** — native installer/bundle that wraps the web UI and embeds the CLI binary as a Tauri sidecar. Five platform targets.

## Build paths

There are four ways the build runs:

| Path                        | Trigger                                               | Entry                                                         | Output                                                                                               |
| --------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Local desktop (default)** | `bun desktop:build`                                   | `packages/desktop/scripts/build-local.ts`                     | Signed + notarized `.app` + `.dmg` (macOS), `.exe` NSIS installer (Windows), `.deb` + `.rpm` (Linux) |
| **Local desktop (fast)**    | `bun desktop:build:fast`                              | `packages/desktop/scripts/build-local.ts --dev --no-notarize` | Signed `.dmg` (macOS), no notarization wait                                                          |
| **Local desktop (quick)**   | `bun desktop:build:quick`                             | `packages/desktop/scripts/build-local.ts --quick`             | Ad-hoc `.app` only (macOS), fastest iteration                                                        |
| **Local CLI only**          | `./packages/emberharmony/script/build.ts --single`    | Same script, single-platform mode                             | `packages/emberharmony/dist/emberharmony-<platform>/bin/emberharmony`                                |
| **CI verification**         | Push to `main` or `dev`                               | `.github/workflows/ci.yml` → `_build.yml`                     | Workflow artifacts (CLI dist + desktop bundles per platform)                                         |
| **Release**                 | `gh workflow run publish.yml` or GitHub release event | `.github/workflows/publish.yml` → `_build.yml`                | npm publish + GitHub release with attached desktop installers                                        |

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

All local builds read Apple signing credentials from the repo-root `.env` file. Signing and notarization are **on by default** — the default build produces a distributable app that macOS won't quarantine.

| Command                   | Config | Signing      | Notarized | Bundle      | Time     |
| ------------------------- | ------ | ------------ | --------- | ----------- | -------- |
| `bun desktop:build`       | prod   | Developer ID | yes       | DMG/deb/rpm | 8-12 min |
| `bun desktop:build:dev`   | dev    | Developer ID | yes       | DMG/deb/rpm | 8-12 min |
| `bun desktop:build:fast`  | dev    | Developer ID | no        | DMG/deb/rpm | 3-5 min  |
| `bun desktop:build:quick` | dev    | ad-hoc       | no        | `.app` only | 2-3 min  |
| `bun desktop:build:nodmg` | prod   | Developer ID | yes       | `.app` only | 8-12 min |

**What each command produces by platform:**

| Command               | macOS                  | Windows                   | Linux           |
| --------------------- | ---------------------- | ------------------------- | --------------- |
| `desktop:build`       | `EmberHarmony.dmg`     | `EmberHarmony.exe` (NSIS) | `.deb` + `.rpm` |
| `desktop:build:dev`   | `EmberHarmony Dev.dmg` | `EmberHarmony Dev.exe`    | `.deb` + `.rpm` |
| `desktop:build:fast`  | `EmberHarmony Dev.dmg` | `EmberHarmony Dev.exe`    | `.deb` + `.rpm` |
| `desktop:build:quick` | `EmberHarmony Dev.app` | `EmberHarmony Dev.exe`    | `.deb`          |
| `desktop:build:nodmg` | `EmberHarmony.app`     | `EmberHarmony.exe`        | `.deb` + `.rpm` |

Notarization is the slow step — Apple's notary service takes 2-8 minutes. `desktop:build:fast` skips it for faster iteration while still producing a signed, unquarantined app. `desktop:build:quick` skips everything for rapid iteration.

### Flags

`build-local.ts` supports these flags:

| Flag            | Effect                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `--dev`         | Use dev config (`EmberHarmony Dev`, `ai.ofharmony.code.dev`). Default is prod config.                  |
| `--quick`       | Ad-hoc signing, skip notarization, skip DMG, dev config. Fastest build. macOS will quarantine the app. |
| `--no-notarize` | Sign with Developer ID but skip notarization. Faster build, macOS may warn on first launch.            |
| `--no-dmg`      | Build `.app`/bundle but skip DMG creation.                                                             |
| `--no-bundle`   | Skip bundling entirely (binary only).                                                                  |
| `--no-voice`    | Skip voice runtime assembly (voice will be disabled in the build).                                     |

Flags can be combined: `--dev --no-notarize --no-dmg` is equivalent to `desktop:build:fast` minus DMG.

### Prerequisites

| Requirement             | Declared by                                                      | Notes                                                                                        |
| ----------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Bun 1.3+                | `packageManager` in root `package.json`                          | The only supported package manager — npm cannot resolve this workspace's `catalog:` versions |
| Tauri CLI               | `@tauri-apps/cli` devDependency                                  | Installed by `bun install`; build scripts invoke it via `bun run tauri`                      |
| Rust toolchain          | `src-tauri/rust-toolchain.toml`                                  | rustup picks the pinned version automatically; install via https://rustup.rs                 |
| Platform libraries      | [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) | OS packages (e.g. webkit2gtk on Linux)                                                       |
| Apple Developer ID      | `.env` `APPLE_SIGNING_IDENTITY`                                  | Required for signing. Pass `--quick` for ad-hoc builds without a cert.                       |
| Apple notarization keys | `.env` `APPLE_API_KEY`, `APPLE_API_ISSUER`, `APPLE_API_KEY_PATH` | Required for notarization. Pass `--quick` or `--no-notarize` to skip.                        |

### macOS signing for local builds

`build-local.ts` reads `APPLE_SIGNING_IDENTITY` from `.env` and verifies the certificate is in your keychain before starting the build. If the identity isn't found, it fails fast with a list of valid identities.

With `--quick`, the identity is set to `-` (ad-hoc signing) and notarization is skipped. The app will work locally but macOS Gatekeeper will quarantine it on first launch.

With `--no-notarize`, the app is signed with your Developer ID but not notarized. macOS may show a warning on first launch — click Open to proceed.

For the full signing + notarization flow, see [APPLE.md](./APPLE.md).

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

| Host               | Target                      | Output                      |
| ------------------ | --------------------------- | --------------------------- |
| `macos-latest`     | `aarch64-apple-darwin`      | `.app`/`.dmg` Apple Silicon |
| `windows-latest`   | `x86_64-pc-windows-msvc`    | `.nsis`/`.msi`              |
| `ubuntu-24.04`     | `x86_64-unknown-linux-gnu`  | `.deb`/`.rpm`/`.AppImage`   |
| `ubuntu-24.04-arm` | `aarch64-unknown-linux-gnu` | `.deb`/`.rpm`               |

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
2. **`build`** — calls `_build.yml` with the `version`, `release` ID, and `tag`. Same 11-CLI + 4-Tauri matrix as CI, with artifacts attached to the release.
3. **`publish`** — runs `./script/publish.ts` to publish `@thesolaceproject/emberharmony` (and the 11 platform npm packages) to npmjs.org.

To cut a release:

```bash
# 1. Bump "version" in packages/emberharmony/package.json and commit it.
# 2. Create a GitHub release whose tag matches that version:
gh release create v1.3.0 --title v1.3.0 --generate-notes
```

## Signing and notarization

Local builds are **signed and notarized by default**. The `.env` file at the repo root should contain:

```
APPLE_SIGNING_IDENTITY=Developer ID Application: Your Name (TEAMID)
APPLE_TEAM_ID=TEAMID
APPLE_API_KEY=ABC123DEFG
APPLE_API_ISSUER=your-issuer-id
APPLE_API_KEY_PATH=/path/to/AuthKey_ABC123DEFG.p8
```

### macOS

| Variable                                                    | Purpose                                                           |
| ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `APPLE_SIGNING_IDENTITY`                                    | Developer ID Application certificate common name                  |
| `APPLE_TEAM_ID`                                             | Apple team ID (auto-derived from signing identity if not set)     |
| `APPLE_API_KEY` + `APPLE_API_ISSUER` + `APPLE_API_KEY_PATH` | App Store Connect API key for notarization                        |
| `APPLE_CERTIFICATE` + `APPLE_CERTIFICATE_PASSWORD`          | `.p12` certificate for CI (not needed locally — keychain is used) |

For the full notarization flow, see [APPLE.md](./APPLE.md).

### Tauri updater

Signs the update manifests so the auto-updater accepts them.

| Variable                             | Purpose                      |
| ------------------------------------ | ---------------------------- |
| `TAURI_SIGNING_PRIVATE_KEY`          | Tauri updater private key    |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | password for the private key |

### npm publish

| Variable    | Purpose                                                               |
| ----------- | --------------------------------------------------------------------- |
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

| File                              | Purpose                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `.github/workflows/_build.yml`    | Reusable build workflow. Builds CLI + Tauri matrix.                           |
| `.github/workflows/ci.yml`        | Post-merge verification on `main`/`dev`. Calls `_build.yml`.                  |
| `.github/workflows/publish.yml`   | Release pipeline. Calls `_build.yml`, then publishes to npm + GitHub release. |
| `.github/workflows/codeql.yml`    | Security analysis on TS/JS and workflow files.                                |
| `.github/workflows/test.yml`      | Test suite (Bun) on PRs.                                                      |
| `.github/workflows/typecheck.yml` | `bun turbo typecheck` on push + PRs.                                          |

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

# Build the desktop app locally
bun desktop:build              # Full: signed + notarized + prod + DMG (matches CI)
bun desktop:build:dev          # Full: signed + notarized + dev config + DMG
bun desktop:build:fast        # Fast: signed (no notarize) + dev + DMG
bun desktop:build:quick       # Quick: ad-hoc + dev + .app only
bun desktop:build:nodmg        # Full: signed + notarized + prod + .app only
```
