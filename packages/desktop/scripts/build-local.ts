#!/usr/bin/env bun
/**
 * First-class local build for the EmberHarmony desktop app.
 *
 * Handles the full pipeline:
 *   1. Builds the emberharmony CLI binary for the current platform
 *   2. Copies it into src-tauri/sidecars/ as the Tauri sidecar
 *   3. Runs `cargo tauri build` (no DMG, to avoid the upstream Tauri bundle_dmg.sh bug)
 *   4. Creates a DMG manually via `hdiutil` on macOS
 *
 * Flags:
 *   --no-dmg    Skip DMG creation (macOS only)
 *   --no-bundle Skip all bundling, just build the binary
 *   --release   Signed release build (requires Apple signing env vars)
 *
 * Environment:
 *   EMBERHARMONY_SKIP_CLI=1   Reuse an existing CLI binary in ../emberharmony/dist
 *
 * Exit codes: 0 success, non-zero on any failure.
 */

import { $ } from "bun"
import { existsSync } from "fs"
import path from "path"
import { fileURLToPath } from "url"

import { copyBinaryToSidecarFolder, getCurrentSidecar, windowsify } from "./utils"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const desktopDir = path.resolve(__dirname, "..")
const emberharmonyDir = path.resolve(desktopDir, "../emberharmony")
const repoRoot = path.resolve(desktopDir, "../..")

// Load .env from repo root so Apple signing/notarization credentials are available
// to `cargo tauri build`. Bun only auto-loads .env from cwd.
const rootEnv = path.join(repoRoot, ".env")
if (existsSync(rootEnv)) {
  const envText = await Bun.file(rootEnv).text()
  for (const line of envText.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const eq = trimmed.indexOf("=")
    if (eq === -1) continue
    const key = trimmed.slice(0, eq).trim()
    let value = trimmed.slice(eq + 1).trim()
    // Strip surrounding quotes if present
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    // Only set if key has a value and isn't already in env — empty values would
    // trigger Tauri's cert-import path with garbage.
    if (key && value && !(key in process.env)) {
      process.env[key] = value
    }
  }
}

process.chdir(desktopDir)

const noDmg = process.argv.includes("--no-dmg")
const noBundle = process.argv.includes("--no-bundle")
const isRelease = process.argv.includes("--release")
const skipCli = process.env.EMBERHARMONY_SKIP_CLI === "1"

// Resolve the Rust target triple for the current host.
// Tauri normally exports TAURI_ENV_TARGET_TRIPLE; we mirror its logic for standalone runs.
function currentRustTarget(): string {
  const envTarget = Bun.env.TAURI_ENV_TARGET_TRIPLE ?? Bun.env.RUST_TARGET
  if (envTarget) return envTarget

  const { platform, arch } = process
  if (platform === "darwin") return arch === "arm64" ? "aarch64-apple-darwin" : "x86_64-apple-darwin"
  if (platform === "linux") return arch === "arm64" ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu"
  if (platform === "win32") return "x86_64-pc-windows-msvc"
  throw new Error(`Unsupported platform: ${platform}/${arch}`)
}

const rustTarget = currentRustTarget()
const sidecar = getCurrentSidecar(rustTarget)

console.log(`[build-local] target: ${rustTarget} (${sidecar.ocBinary})`)

// --- Step 1: Build CLI binary ---------------------------------------------
const cliBinaryPath = path.join(emberharmonyDir, "dist", sidecar.ocBinary, "bin", windowsify("emberharmony"))

if (skipCli && existsSync(cliBinaryPath)) {
  console.log(`[build-local] reusing existing CLI binary: ${cliBinaryPath}`)
} else {
  console.log(`[build-local] building CLI binary...`)
  await $`bun run script/build.ts --single`.cwd(emberharmonyDir)
  if (!existsSync(cliBinaryPath)) {
    throw new Error(`CLI binary not found after build: ${cliBinaryPath}`)
  }
}

// --- Step 2: Copy sidecar --------------------------------------------------
console.log(`[build-local] copying sidecar to src-tauri/sidecars/...`)
await copyBinaryToSidecarFolder(cliBinaryPath, rustTarget)

// --- Step 3: Run tauri build (no-bundle to skip the broken DMG path) ------
if (noBundle) {
  console.log(`[build-local] running: cargo tauri build --no-bundle`)
  await $`cargo tauri build --no-bundle`
  console.log(`[build-local] done (no bundle)`)
  process.exit(0)
}

// Always skip DMG in the Tauri invocation because the upstream bundle_dmg.sh
// fails. We create the DMG manually afterwards if the host is macOS.
const tauriArgs = ["build"]
// Note: `cargo tauri build` is release mode by default; `--debug` is the only toggle.
// Apple signing env vars (if set) are picked up automatically by Tauri.
void isRelease

// Build only the .app on macOS (skip dmg in Tauri's own bundler)
if (process.platform === "darwin") {
  tauriArgs.push("--bundles", "app")
} else if (process.platform === "linux") {
  tauriArgs.push("--bundles", "deb,rpm,appimage")
} else if (process.platform === "win32") {
  tauriArgs.push("--bundles", "nsis,msi")
}

console.log(`[build-local] running: cargo tauri ${tauriArgs.join(" ")}`)
await $`cargo tauri ${tauriArgs}`

// --- Step 4: Create DMG manually (macOS only) -----------------------------
if (process.platform === "darwin" && !noDmg) {
  const targetRelease = path.join(desktopDir, "src-tauri/target/release")
  const appBundle = path.join(targetRelease, "bundle/macos/EmberHarmony Dev.app")

  if (!existsSync(appBundle)) {
    throw new Error(`.app bundle not found: ${appBundle}`)
  }

  const pkg = await Bun.file(path.join(desktopDir, "package.json")).json()
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64"
  const dmgDir = path.join(targetRelease, "bundle/dmg")
  const dmgPath = path.join(dmgDir, `EmberHarmony Dev_${pkg.version}_${arch}.dmg`)

  await $`mkdir -p ${dmgDir}`
  await $`rm -f ${dmgPath}`

  console.log(`[build-local] creating installer DMG: ${dmgPath}`)

  // Build a proper installer DMG by staging .app + /Applications symlink in a
  // temp directory, then packaging it. This gives the drag-to-install UX.
  const stagingDir = path.join(targetRelease, "bundle/dmg/.staging")
  await $`rm -rf ${stagingDir}`
  await $`mkdir -p ${stagingDir}`
  await $`cp -R ${appBundle} ${stagingDir}/`
  await $`ln -s /Applications ${stagingDir}/Applications`

  await $`hdiutil create -volname "EmberHarmony Dev" -srcfolder ${stagingDir} -ov -format UDZO ${dmgPath}`
  await $`rm -rf ${stagingDir}`

  // Sign the DMG itself so Gatekeeper trusts the container too.
  const signingIdentity = process.env.APPLE_SIGNING_IDENTITY
  if (signingIdentity) {
    console.log(`[build-local] signing DMG with: ${signingIdentity}`)
    await $`codesign --force --sign ${signingIdentity} ${dmgPath}`
  }

  console.log(`[build-local] DMG created: ${dmgPath}`)
}

console.log(`[build-local] done`)
