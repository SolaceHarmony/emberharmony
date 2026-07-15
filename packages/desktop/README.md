# EmberHarmony Desktop

Native EmberHarmony desktop app, built with Tauri v2.

## Development

From the repo root:

```bash
bun install
bun run --cwd packages/desktop tauri dev
```

This starts the Vite dev server on http://localhost:1420 and opens the native window.

If you only want the web dev server (no native shell):

```bash
bun run --cwd packages/desktop dev
```

## Build

To create a production `dist/` and build the native app bundle:

```bash
bun run --cwd packages/desktop tauri build
```

## Prerequisites

Every requirement is declared in the repo; nothing relies on ad-hoc global installs:

| Requirement | Declared by | Notes |
|---|---|---|
| Bun | `packageManager` in the root `package.json` | The only supported package manager — npm cannot resolve this workspace's `catalog:` versions |
| Tauri CLI | `@tauri-apps/cli` devDependency | Installed by `bun install`; build scripts invoke it via `bun run tauri`, resolving strictly from `node_modules/.bin` (never `bunx`, which auto-installs from the registry, and not the cargo-installed `cargo tauri`) |
| Rust toolchain | root `rust-toolchain.toml` | rustup picks the pinned version up automatically; install rustup via <https://rustup.rs> |
| Platform libraries | [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) | OS packages (e.g. webkit2gtk on Linux) |

`scripts/build-local.ts` verifies all of this up front (preflight) and fails with
install guidance before any compilation starts.

### macOS signing for local builds

`build-local.ts` checks that `APPLE_SIGNING_IDENTITY` (usually from the repo-root
`.env`) resolves to a certificate actually present in the keychain, and fails
fast listing the valid identities if not. With no identity configured, local
builds are ad-hoc signed (`-`): runnable on this machine, not distributable,
never notarized. Notarization is exclusively a release-pipeline concern.

```bash
# explicit ad-hoc local build
APPLE_SIGNING_IDENTITY="-" bun run --cwd packages/desktop build:local
```

## Platform-Specific Features

### macOS

The desktop app includes native macOS window vibrancy effects for better contrast and visual integration with the operating system. This feature provides:

- **Native vibrancy effects**: The window background uses macOS's built-in vibrancy/blur effects for a modern, native appearance
- **Improved contrast**: Semi-transparent backgrounds with vibrancy ensure better text readability and reduced eye strain
- **Adaptive theming**: Automatically adjusts to light/dark mode preferences with appropriate transparency levels

The vibrancy effect is implemented using the `window-vibrancy` crate and uses the `HudWindow` material, which provides a suitable backdrop for application interfaces. The theme system automatically applies semi-transparent backgrounds when running on macOS to work seamlessly with the vibrancy effects.

**Note**: The window transparency setting is enabled globally in the configuration, but the semi-transparent CSS backgrounds are only applied on macOS. On other platforms, the window remains opaque due to the standard background styling.
