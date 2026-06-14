#!/usr/bin/env bun
/**
 * Codesign every Mach-O binary inside the bundled voice runtime (macOS only).
 *
 * Apple notarization rejects an app if any nested Mach-O is not signed with a
 * Developer ID certificate and a secure timestamp. Tauri signs the app bundle,
 * the main binary, and the externalBin sidecar — but not the binaries that ship
 * inside `resources/voice` (`bun`, plus the `.node`/`.dylib` native libs from
 * rtc-ffi, onnxruntime, sharp/libvips). The native libs arrive pre-built and
 * unsigned from npm, so notarization fails with "not signed with a valid
 * Developer ID certificate" / "signature does not include a secure timestamp".
 *
 * We sign everything we ship — no exceptions — BEFORE `tauri build` assembles
 * and signs the app: Tauri copies the resource into the bundle and seals it
 * without re-signing nested resource code (no `--deep`), so these signatures
 * survive into the notarized app. Binaries are found by Mach-O magic bytes
 * rather than by name/extension, so `bun` and any future native binary are all
 * covered.
 *
 * `--preserve-metadata=entitlements` keeps each binary's existing entitlements:
 * the prebuilt `bun` needs `allow-jit` / `allow-unsigned-executable-memory` /
 * `disable-library-validation` to run and to load the (now same-team) libs, and
 * losing them would crash it; the native libs carry none, so it is a no-op for
 * them. `--options runtime` sets the hardened runtime flag notarization expects.
 *
 * Usage:
 *   bun run scripts/sign-voice-runtime.ts --identity "Developer ID Application: …"
 *   bun run scripts/sign-voice-runtime.ts --identity "$APPLE_SIGNING_IDENTITY" --keychain /path/to.keychain-db
 *
 * No-op (exit 0) on non-macOS hosts and when the identity is absent or "-"
 * (ad-hoc / local non-notarized builds don't need timestamps).
 */
import { $ } from "bun"
import { open, readdir } from "node:fs/promises"
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

// Mach-O magic numbers (thin 32/64-bit, both byte orders, and fat/universal).
const MACHO_MAGIC = new Set([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe, 0xbebafeca])

async function isMachO(file: string): Promise<boolean> {
  const fh = await open(file, "r")
  try {
    const buf = Buffer.alloc(4)
    const { bytesRead } = await fh.read(buf, 0, 4, 0)
    if (bytesRead < 4) return false
    return MACHO_MAGIC.has(buf.readUInt32BE(0)) || MACHO_MAGIC.has(buf.readUInt32LE(0))
  } finally {
    await fh.close()
  }
}

// Collect every Mach-O shipped in the runtime — `bun`, native addons, dylibs —
// by inspecting magic bytes, so nothing is skipped by name or extension.
async function findMachO(root: string): Promise<string[]> {
  const out: string[] = []
  async function walk(d: string) {
    for (const entry of await readdir(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name)
      if (entry.isSymbolicLink()) continue
      if (entry.isDirectory()) await walk(full)
      else if (entry.isFile() && (await isMachO(full))) out.push(full)
    }
  }
  await walk(root)
  return out
}

const binaries = await findMachO(dir)
if (binaries.length === 0) {
  throw new Error(`[sign-voice-runtime] no Mach-O binaries found under ${dir} — is the voice runtime assembled?`)
}

console.log(`[sign-voice-runtime] signing ${binaries.length} Mach-O binaries with "${identity}"`)
const kcArgs = keychain ? ["--keychain", keychain] : []
for (const bin of binaries) {
  await $`codesign --force --timestamp --options runtime --preserve-metadata=entitlements ${kcArgs} --sign ${identity} ${bin}`.quiet()
  // --strict catches a signature that codesign wrote but that won't pass Gatekeeper.
  await $`codesign --verify --strict --verbose=2 ${bin}`.quiet()
  console.log(`  ✓ ${path.relative(dir, bin)}`)
}
console.log("[sign-voice-runtime] done")
