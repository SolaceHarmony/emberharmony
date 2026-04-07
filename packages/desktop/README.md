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

Running the desktop app requires additional Tauri dependencies (Rust toolchain, platform-specific libraries). See the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for setup instructions.

## Platform-Specific Features

### macOS

The desktop app includes native macOS window vibrancy effects for better contrast and visual integration with the operating system. This feature provides:

- **Native vibrancy effects**: The window background uses macOS's built-in vibrancy/blur effects for a modern, native appearance
- **Improved contrast**: Semi-transparent backgrounds with vibrancy ensure better text readability and reduced eye strain
- **Adaptive theming**: Automatically adjusts to light/dark mode preferences with appropriate transparency levels

The vibrancy effect is implemented using the `window-vibrancy` crate and uses the `HudWindow` material, which provides a suitable backdrop for application interfaces. The theme system automatically applies semi-transparent backgrounds when running on macOS to work seamlessly with the vibrancy effects.

**Note**: The window transparency setting is enabled globally in the configuration, but the semi-transparent CSS backgrounds are only applied on macOS. On other platforms, the window remains opaque due to the standard background styling.
