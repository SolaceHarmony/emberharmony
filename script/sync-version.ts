#!/usr/bin/env bun
/**
 * Single source of truth: ../version.json.
 * Propagates its `version` into every workspace package.json so npm, Tauri, and
 * the publish flow all agree. Run after editing version.json:
 *
 *   bun run version:sync
 *
 * Independently-versioned packages are skipped (the JS SDK and the VS Code
 * extension carry their own version line).
 */
import { Glob } from "bun"
import path from "path"

const ROOT = path.resolve(import.meta.dir, "..")
const meta = (await Bun.file(path.join(ROOT, "version.json")).json()) as { version?: string }
const version = meta.version
if (typeof version !== "string" || !version) {
  throw new Error("version.json is missing a valid `version` field")
}

// Packages that intentionally track their own version, not the app version.
const SKIP = ["packages/sdk/js/package.json", "sdks/vscode/package.json"]

const updated: string[] = []
const skipped: string[] = []
const noVersion: string[] = []

for await (const rel of new Glob("**/package.json").scan({ cwd: ROOT })) {
  if (rel.includes("node_modules") || rel.includes("/dist") || rel.includes("/.output/")) continue
  if (SKIP.includes(rel)) {
    skipped.push(rel)
    continue
  }
  const file = path.join(ROOT, rel)
  const text = await Bun.file(file).text()

  // Authoritatively decide whether this package carries its own version by
  // parsing it: a nested "version" (inside scripts/dependencies/etc.) is never
  // a top-level key, so it can't be mistaken for the package's own version.
  // A malformed package.json throws here rather than being silently skipped.
  const parsed = JSON.parse(text) as { version?: unknown }
  if (!("version" in parsed)) {
    noVersion.push(rel)
    continue
  }

  // The package has a top-level version. Rewrite it in place with a
  // format-preserving replace targeting a two-space-indented line. If that
  // pattern doesn't match, the file uses unexpected formatting — fail loudly
  // instead of silently leaving its version out of sync.
  const TOP_LEVEL_VERSION = /^( {2}"version":\s*)"[^"]*"/m
  if (!TOP_LEVEL_VERSION.test(text)) {
    throw new Error(
      `${rel} has a top-level "version" key that is not a two-space-indented line; ` +
        `reformat it or update sync-version.ts rather than silently skipping it`,
    )
  }
  const next = text.replace(TOP_LEVEL_VERSION, `$1"${version}"`)
  if (next !== text) {
    await Bun.write(file, next)
    updated.push(rel)
  }
}

console.log(`version.json → ${version}`)
console.log(`updated ${updated.length} package.json file(s):`)
for (const r of updated.sort()) console.log(`  ${r}`)
if (skipped.length) console.log(`skipped (independently versioned): ${skipped.join(", ")}`)
if (noVersion.length) console.log(`no top-level version (left untouched): ${noVersion.sort().join(", ")}`)
