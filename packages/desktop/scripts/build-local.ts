#!/usr/bin/env bun
/**
 * First-class local build for the EmberHarmony desktop app.
 *
 * Handles the full pipeline:
 *   1. Builds the emberharmony CLI binary for the current platform
 *   2. Copies it into src-tauri/sidecars/ as the Tauri sidecar
 *   3. Assembles and signs the voice runtime
 *   4. Runs `tauri build` with appropriate bundle flags
 *   5. Creates a DMG manually on macOS (workaround for upstream bundle_dmg.sh bug)
 *
 * Defaults match CI: signed, notarized, prod config, full bundle.
 * Use flags to opt out for faster iteration.
 *
 * Flags:
 *   --dev          Use dev config (EmberHarmony Dev, ai.ofharmony.code.dev)
 *   --quick        Ad-hoc signing, skip notarization, skip DMG, dev config.
 *                  Fastest iteration. App will be quarantined on first launch.
 *   --no-notarize  Sign but skip notarization. Faster, macOS may warn on first launch.
 *   --no-dmg       Skip DMG creation (macOS only).
 *   --no-bundle    Skip all bundling, just build the binary.
 *   --no-voice     Skip voice runtime assembly (voice will be disabled in build).
 *
 * Environment:
 *   EMBERHARMONY_SKIP_CLI=1   Reuse an existing CLI binary in ../emberharmony/dist
 *
 * Signing credentials are read from the repo-root .env:
 *   APPLE_SIGNING_IDENTITY  Developer ID certificate common name
 *   APPLE_API_KEY           App Store Connect API key ID
 *   APPLE_API_ISSUER        App Store Connect API issuer ID
 *   APPLE_API_KEY_PATH      Path to AuthKey_*.p8
 *   APPLE_TEAM_ID           Apple team ID
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

// Load .env from repo root. Bun only auto-loads .env from cwd.
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
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    if (key && value && !(key in process.env)) {
      process.env[key] = value
    }
  }
}

process.chdir(desktopDir)

const quick = process.argv.includes("--quick")
const dev = process.argv.includes("--dev")
const noNotarize = process.argv.includes("--no-notarize")
const noDmg = process.argv.includes("--no-dmg")
const noBundle = process.argv.includes("--no-bundle")
const skipCli = process.env.EMBERHARMONY_SKIP_CLI === "1"

// --quick implies --dev --no-notarize --no-dmg and ad-hoc signing
const useDevConfig = quick || dev
const skipNotarization = quick || noNotarize
const skipDmg = quick || noDmg

// Resolve the Rust target triple for the current host.
function currentRustTarget(): string {
  const envTarget = Bun.env.TAURI_ENV_TARGET_TRIPLE ?? Bun.env.RUST_TARGET
  if (envTarget) return envTarget

  const { platform, arch } = process
  if (platform === "darwin") return "aarch64-apple-darwin"
  if (platform === "linux") return arch === "arm64" ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu"
  if (platform === "win32") return "x86_64-pc-windows-msvc"
  throw new Error(`Unsupported platform: ${platform}/${arch}`)
}

const rustTarget = currentRustTarget()
const sidecar = getCurrentSidecar(rustTarget)

const configLabel = useDevConfig ? "dev" : "prod"
const signLabel = quick ? "ad-hoc" : "Developer ID"
const notarizeLabel = skipNotarization ? "no" : "yes"
console.log(
  `[build-local] target: ${rustTarget} (${sidecar.ocBinary}), config: ${configLabel}, signing: ${signLabel}, notarize: ${notarizeLabel}`,
)

// --- Step 0: Preflight --------------------------------------------------------
{
  const missing: string[] = []

  const cargo = await $`cargo --version`.quiet().nothrow()
  if (cargo.exitCode !== 0) {
    missing.push(
      "Rust toolchain (cargo) — install via https://rustup.rs; src-tauri/rust-toolchain.toml pins the version",
    )
  }

  const tauri = await $`bun run tauri --version`.quiet().nothrow()
  if (tauri.exitCode !== 0) {
    missing.push("Tauri CLI (@tauri-apps/cli devDependency) — run `bun install` at the repo root")
  }

  if (process.platform === "darwin" && !skipDmg && !noBundle) {
    const hdiutil = await $`hdiutil info`.quiet().nothrow()
    if (hdiutil.exitCode !== 0) {
      missing.push("hdiutil (macOS DMG creation) — or pass --no-dmg to skip the DMG step")
    }
  }

  // Notarization: on by default, opt out with --quick or --no-notarize.
  // When notarizing, verify the API key file exists.
  if (process.platform === "darwin" && !noBundle && !skipNotarization) {
    const keyPath = process.env.APPLE_API_KEY_PATH
    if (keyPath && !existsSync(keyPath)) {
      missing.push(`APPLE_API_KEY_PATH points to "${keyPath}" which does not exist on this machine`)
    }
  }

  // When skipping notarization, strip the Apple API env vars so Tauri doesn't
  // attempt notarization either.
  if (skipNotarization && process.platform === "darwin") {
    const notarizeVars = ["APPLE_API_KEY", "APPLE_API_ISSUER", "APPLE_API_KEY_PATH", "APPLE_ID", "APPLE_PASSWORD"]
    const present = notarizeVars.filter((name) => process.env[name])
    if (present.length > 0) {
      for (const name of present) delete process.env[name]
      console.log(`[build-local] notarization disabled — stripped ${present.join(", ")}`)
    }
  }

  // macOS signing: verify the configured identity is in the keychain.
  if (process.platform === "darwin" && !noBundle) {
    if (quick) {
      process.env.APPLE_SIGNING_IDENTITY = "-"
      console.log(`[build-local] --quick: using ad-hoc signing ("-")`)
    } else {
      const identity = process.env.APPLE_SIGNING_IDENTITY
      if (identity && identity !== "-") {
        const identities = (await $`security find-identity -v -p codesigning`.quiet().nothrow()).stdout.toString()
        if (!identities.includes(identity)) {
          missing.push(
            `signing identity "${identity}" (from APPLE_SIGNING_IDENTITY / .env) is not in the keychain.\n` +
              `    Valid identities:\n${identities
                .split("\n")
                .filter((line) => line.includes('"'))
                .map((line) => `      ${line.trim()}`)
                .join("\n")}\n` +
              `    Install that certificate, fix .env, or pass --quick for an ad-hoc build`,
          )
        }
      } else if (!identity) {
        // No identity configured and not --quick — fail rather than silently
        // fall back to ad-hoc. The default should be real signing.
        missing.push(
          `No APPLE_SIGNING_IDENTITY found in .env or environment.\n` +
            `    Add your Developer ID identity to .env (e.g. APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)")\n` +
            `    or pass --quick for an ad-hoc build (not distributable, macOS will quarantine)`,
        )
      }
    }
  }

  if (missing.length > 0) {
    console.error(`[build-local] missing requirements:`)
    for (const item of missing) console.error(`  - ${item}`)
    process.exit(1)
  }
  console.log(
    `[build-local] preflight ok: ${cargo.stdout.toString().trim()}, tauri-cli ${tauri.stdout.toString().trim()}`,
  )
}

// --- Step 1: Build CLI binary -------------------------------------------------
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

// --- Step 2: Copy sidecar -----------------------------------------------------
console.log(`[build-local] copying sidecar to src-tauri/sidecars/...`)
await copyBinaryToSidecarFolder(cliBinaryPath, rustTarget)

// --- Step 2b: Assemble the bundled voice runtime -----------------------------
if (!process.argv.includes("--no-voice")) {
  console.log(`[build-local] assembling voice runtime resource...`)
  await $`bun ./scripts/build-voice-runtime.ts`.cwd(desktopDir)
  await $`bun ./scripts/sign-voice-runtime.ts`.cwd(desktopDir)
} else {
  console.log(`[build-local] --no-voice: skipping voice runtime (voice will be disabled in this build)`)
  await $`rm -rf ${path.join(desktopDir, "src-tauri/resources/voice")}`
}

// --- Step 3: Run tauri build -------------------------------------------------
if (noBundle) {
  const configArg = useDevConfig ? [] : ["--config", "./src-tauri/tauri.prod.conf.json"]
  console.log(`[build-local] running: tauri build --no-bundle ${configArg.join(" ")}`)
  await $`bun run tauri build --no-bundle ${configArg}`
  console.log(`[build-local] done (no bundle)`)
  process.exit(0)
}

const tauriArgs = ["build"]

// Use prod config by default, dev config with --dev or --quick
if (!useDevConfig) {
  tauriArgs.push("--config", "./src-tauri/tauri.prod.conf.json")
}

if (process.platform === "darwin") {
  tauriArgs.push("--bundles", "app")
} else if (process.platform === "linux") {
  tauriArgs.push("--bundles", "deb,rpm,appimage")
} else if (process.platform === "win32") {
  tauriArgs.push("--bundles", "nsis,msi")
}

console.log(`[build-local] running: tauri ${tauriArgs.join(" ")}`)
await $`bun run tauri ${tauriArgs}`

// --- Step 4: Create DMG manually (macOS only) --------------------------------
if (process.platform === "darwin" && !skipDmg) {
  const targetRelease = path.join(desktopDir, "src-tauri/target/release")
  const appName = useDevConfig ? "EmberHarmony Dev" : "EmberHarmony"
  const appBundle = path.join(targetRelease, `bundle/macos/${appName}.app`)

  if (!existsSync(appBundle)) {
    throw new Error(`.app bundle not found: ${appBundle}`)
  }

  const pkg = await Bun.file(path.join(desktopDir, "package.json")).json()
  const arch = process.arch === "arm64" ? "aarch64" : "x86_64"
  const dmgDir = path.join(targetRelease, "bundle/dmg")
  const dmgPath = path.join(dmgDir, `${appName}_${pkg.version}_${arch}.dmg`)

  await $`mkdir -p ${dmgDir}`
  await $`rm -f ${dmgPath}`

  console.log(`[build-local] creating installer DMG: ${dmgPath}`)

  const stagingDir = path.join(targetRelease, "bundle/dmg/.staging")
  await $`rm -rf ${stagingDir}`
  await $`mkdir -p ${stagingDir}`
  await $`cp -R ${appBundle} ${stagingDir}/`
  await $`ln -s /Applications ${stagingDir}/Applications`

  await $`hdiutil create -volname "${appName}" -srcfolder ${stagingDir} -ov -format UDZO ${dmgPath}`
  await $`rm -rf ${stagingDir}`

  console.log(`[build-local] DMG created: ${dmgPath}`)
}

console.log(`[build-local] done`)
