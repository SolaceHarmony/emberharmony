import { $, semver } from "bun"
import path from "path"

const rootPkgPath = path.resolve(import.meta.dir, "../../../package.json")
const rootPkg = await Bun.file(rootPkgPath).json()
const expectedBunVersion = rootPkg.packageManager?.split("@")[1]

if (!expectedBunVersion) {
  throw new Error("packageManager field not found in root package.json")
}

const expectedBunVersionRange = `^${expectedBunVersion}`

if (!semver.satisfies(process.versions.bun, expectedBunVersionRange)) {
  throw new Error(`This script requires bun@${expectedBunVersionRange}, but you are using bun@${process.versions.bun}`)
}

const env = {
  EMBERHARMONY_CHANNEL: process.env["EMBERHARMONY_CHANNEL"],
  EMBERHARMONY_RELEASE: process.env["EMBERHARMONY_RELEASE"],
  EMBERHARMONY_TARGET: process.env["EMBERHARMONY_TARGET"],
}

const CHANNEL = await (async () => {
  if (env.EMBERHARMONY_CHANNEL) return env.EMBERHARMONY_CHANNEL
  // Release builds run on a detached tag checkout where `git branch` can't see
  // dev/main, so the channel comes from the branch the human targeted in the
  // GitHub release UI (EMBERHARMONY_TARGET = release.target_commitish):
  // main → stable "latest", dev → "dev". Anything else is a misconfigured
  // release and fails the build. Local/CI branch builds keep preview logic.
  if (env.EMBERHARMONY_RELEASE) {
    if (env.EMBERHARMONY_TARGET === "main") return "latest"
    if (env.EMBERHARMONY_TARGET === "dev") return "dev"
    throw new Error(
      `release builds require EMBERHARMONY_TARGET of "main" or "dev", got "${env.EMBERHARMONY_TARGET ?? "(unset)"}"`,
    )
  }
  const branch = await $`git branch --show-current`
    .text()
    .then((x) => x.trim())
    .catch(() => "")
  return branch === "dev" || branch === "main" ? "latest" : branch || "preview"
})()
const IS_PREVIEW = CHANNEL !== "latest"

// Source of truth: the root version.json. Version travels with the code;
// to release a new version, edit version.json and run `bun run version:sync`.
const STATIC_VERSION = await (async () => {
  const pkgPath = path.resolve(import.meta.dir, "../../../version.json")
  const data = (await Bun.file(pkgPath).json()) as { version?: unknown }
  if (typeof data.version !== "string" || data.version.length === 0) {
    throw new Error(`version field missing or invalid in ${pkgPath}`)
  }
  return data.version
})()

// The embedded version is hard-coded from version.json for every release
// build regardless of tag or channel — the tag names the release, it never
// defines the version. Timestamped preview versions exist only for local /
// non-integration-branch builds.
const VERSION =
  env.EMBERHARMONY_RELEASE || !IS_PREVIEW
    ? STATIC_VERSION
    : `0.0.0-${CHANNEL}-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}`

export const Script = {
  get channel() {
    return CHANNEL
  },
  get version() {
    return VERSION
  },
  get preview() {
    return IS_PREVIEW
  },
  get release() {
    return env.EMBERHARMONY_RELEASE
  },
}
console.log(`emberharmony script`, JSON.stringify(Script, null, 2))
