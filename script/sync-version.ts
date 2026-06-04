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

for await (const rel of new Glob("**/package.json").scan({ cwd: ROOT })) {
  if (rel.includes("node_modules") || rel.includes("/dist") || rel.includes("/.output/")) continue
  if (SKIP.includes(rel)) {
    skipped.push(rel)
    continue
  }
  const file = path.join(ROOT, rel)
  const text = await Bun.file(file).text()
  // Only touch a package's own top-level "version" field. Anchoring to a
  // two-space indent at line start ensures we never match a nested "version"
  // key (deeper indentation) inside scripts/dependencies/etc.
  const TOP_LEVEL_VERSION = /^( {2}"version":\s*)"[^"]*"/m
  if (!TOP_LEVEL_VERSION.test(text)) continue
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
