#!/usr/bin/env bun
/**
 * Codesign the native binaries inside the bundled voice runtime (macOS only).
 *
 * Apple notarization rejects an app if any nested Mach-O is not signed with a
 * Developer ID certificate and a secure timestamp. Tauri signs the app bundle,
 * the main binary, and the externalBin sidecar — but not the `.node`/`.dylib`
 * files that ship inside `resources/voice/node_modules` (rtc-ffi, onnxruntime,
 * sharp/libvips). They arrive pre-built and unsigned from npm, so notarization
 * fails with "not signed with a valid Developer ID certificate" / "signature
 * does not include a secure timestamp".
 *
 * Sign each native library here, BEFORE `tauri build` assembles and signs the
 * app: Tauri copies the resource into the bundle and seals it without re-signing
 * nested resource code (no `--deep`), so these signatures survive into the
 * notarized app. The bundled `bun` is intentionally left alone — oven-sh already
 * ships it Developer-ID signed, timestamped, and with `disable-library-
 * validation`, which is also what lets it load these differently-teamed libs.
 *
 * Libraries inherit the host process's entitlements, so they are signed with
 * `--options runtime --timestamp` and no entitlements of their own.
 *
 * Usage:
 *   bun run scripts/sign-voice-runtime.ts --identity "Developer ID Application: …"
 *   bun run scripts/sign-voice-runtime.ts --identity "$APPLE_SIGNING_IDENTITY" --keychain /path/to.keychain-db
 *
 * No-op (exit 0) on non-macOS hosts and when the identity is absent or "-"
 * (ad-hoc / local non-notarized builds don't need timestamps).
 */
import { $ } from "bun"
import { readdir } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

function parseArg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : undefined
}

if (process.platform !== "darwin") {
  console.log("[sign-voice-runtime] not macOS — nothing to sign")
  process.exit(0)
}

const identity = parseArg("identity") ?? process.env["APPLE_SIGNING_IDENTITY"] ?? ""
if (!identity || identity === "-") {
  console.log("[sign-voice-runtime] no Developer ID identity (ad-hoc/local build) — skipping nested signing")
  process.exit(0)
}

const keychain = parseArg("keychain")
const desktopDir = path.resolve(fileURLToPath(import.meta.url), "../..")
const dir = parseArg("dir") ?? path.join(desktopDir, "src-tauri/resources/voice")

// Collect every nested Mach-O library shipped in the runtime. `.node` native
// addons and `.dylib`/`.so` shared libraries all need a valid signature.
async function findLibs(root: string): Promise<string[]> {
  const out: string[] = []
  async function walk(d: string) {
    for (const entry of await readdir(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name)
      if (entry.isDirectory()) await walk(full)
      else if (/\.(node|dylib|so)$/.test(entry.name)) out.push(full)
    }
  }
  await walk(root)
  return out
}

const libs = await findLibs(dir)
if (libs.length === 0) {
  throw new Error(`[sign-voice-runtime] no .node/.dylib/.so found under ${dir} — is the voice runtime assembled?`)
}

console.log(`[sign-voice-runtime] signing ${libs.length} native libs with "${identity}"`)
const kcArgs = keychain ? ["--keychain", keychain] : []
for (const lib of libs) {
  await $`codesign --force --timestamp --options runtime ${kcArgs} --sign ${identity} ${lib}`.quiet()
  // --strict catches a signature that codesign wrote but that won't pass Gatekeeper.
  await $`codesign --verify --strict --verbose=2 ${lib}`.quiet()
  console.log(`  ✓ ${path.relative(dir, lib)}`)
}
console.log("[sign-voice-runtime] done")
