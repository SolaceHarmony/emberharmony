#!/usr/bin/env bun
/**
 * Remove EmberHarmony desktop build artifacts so a rebuild starts truly clean —
 * no stale bundles (old .dmg/.deb/.rpm from ANY platform), no leftover voice
 * runtime, no cached frontend or Rust output. `src-tauri/target` is shared
 * across macOS and Linux builds here, so old artifacts (e.g. loose .dmg files
 * from a previous macOS build) linger there unless explicitly cleared — this is
 * what removes them.
 *
 * Always removes:
 *   packages/desktop/src-tauri/target           Rust build + ALL bundles (dmg/deb/rpm/macos/appimage/nsis)
 *   packages/desktop/dist                       vite frontend output
 *   packages/desktop/src-tauri/resources/voice  bundled voice runtime (~750MB, regenerated every build)
 *   packages/desktop/src-tauri/sidecars         copied CLI sidecar binaries
 *   packages/desktop/node_modules/.vite         vite dep-optimizer cache
 *   packages/emberharmony/dist                  compiled CLI binary
 *   <repo>/.voice-runtime-stage                 voice-runtime npm staging dir
 *   packages/** /*.tsbuildinfo                   tsgo incremental build info
 *
 * With --deep (alias --node-modules): ALSO removes node_modules everywhere
 * (repo root + every package). You MUST run `bun install` before the next build.
 *
 * Usage:
 *   bun ./scripts/clean.ts            # build artifacts only
 *   bun ./scripts/clean.ts --deep     # + node_modules (then `bun install`)
 */
import { $ } from "bun"
import { existsSync } from "node:fs"
import { readdir, rm } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const desktopDir = path.resolve(scriptDir, "..")
const repoRoot = path.resolve(desktopDir, "../..")
const deep = process.argv.includes("--deep") || process.argv.includes("--node-modules")

const targets = [
  path.join(desktopDir, "src-tauri/target"),
  path.join(desktopDir, "dist"),
  path.join(desktopDir, "src-tauri/resources/voice"),
  path.join(desktopDir, "src-tauri/sidecars"),
  path.join(desktopDir, "node_modules/.vite"),
  path.join(repoRoot, "packages/emberharmony/dist"),
  path.join(repoRoot, ".voice-runtime-stage"),
]

if (deep) {
  targets.push(path.join(repoRoot, "node_modules"))
  for (const e of await readdir(path.join(repoRoot, "packages"), { withFileTypes: true }).catch(() => []))
    if (e.isDirectory()) targets.push(path.join(repoRoot, "packages", e.name, "node_modules"))
}

console.log(`[clean] ${deep ? "deep clean (build artifacts + node_modules)" : "build artifacts"}`)
let removed = 0
for (const t of targets) {
  const rel = path.relative(repoRoot, t)
  if (!existsSync(t)) {
    console.log(`  ·  ${rel}`)
    continue
  }
  await rm(t, { recursive: true, force: true })
  console.log(`  ✓  ${rel}`)
  removed++
}

// stray tsgo incremental files (outside node_modules)
const tsbuild = (
  await $`find ${path.join(repoRoot, "packages")} -name '*.tsbuildinfo' -not -path '*/node_modules/*'`.quiet().nothrow().text()
)
  .trim()
  .split("\n")
  .filter(Boolean)
for (const f of tsbuild) {
  await rm(f, { force: true })
  console.log(`  ✓  ${path.relative(repoRoot, f)}`)
  removed++
}

console.log(`[clean] removed ${removed} item(s)${deep ? "  —  run \`bun install\` before building" : ""}`)
