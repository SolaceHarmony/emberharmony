import { $, semver } from "bun"
import path from "path"

const rootPkgPath = path.resolve(import.meta.dir, "../../../package.json")
const rootPkg = await Bun.file(rootPkgPath).json()
const expectedBunVersion = rootPkg.packageManager?.split("@")[1]

if (!expectedBunVersion) {
  throw new Error("packageManager field not found in root package.json")
}

// relax version requirement
const expectedBunVersionRange = `^${expectedBunVersion}`

if (!semver.satisfies(process.versions.bun, expectedBunVersionRange)) {
  throw new Error(`This script requires bun@${expectedBunVersionRange}, but you are using bun@${process.versions.bun}`)
}

const env = {
  OPENCODE_CHANNEL: process.env["OPENCODE_CHANNEL"],
  OPENCODE_BUMP: process.env["OPENCODE_BUMP"],
  OPENCODE_VERSION: process.env["OPENCODE_VERSION"],
  OPENCODE_RELEASE: process.env["OPENCODE_RELEASE"],
}
const CHANNEL = await (async () => {
  if (env.OPENCODE_CHANNEL) return env.OPENCODE_CHANNEL
  if (env.OPENCODE_BUMP) return "latest"
  if (env.OPENCODE_VERSION && !env.OPENCODE_VERSION.startsWith("0.0.0-")) return "latest"
  return await $`git branch --show-current`.text().then((x) => x.trim())
})()
const IS_PREVIEW = CHANNEL !== "latest"

const bump = (current: string, kind: string | undefined) => {
  const [major, minor, patch] = current.split(".").map((x) => Number(x) || 0)
  const t = kind?.toLowerCase()
  if (t === "major") return `${major + 1}.0.0`
  if (t === "minor") return `${major}.${minor + 1}.0`
  return `${major}.${minor}.${patch + 1}`
}

const VERSION = await (async () => {
  if (env.OPENCODE_VERSION) return env.OPENCODE_VERSION
  if (IS_PREVIEW) return `0.0.0-${CHANNEL}-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}`

  const npm = await fetch("https://registry.npmjs.org/code-harmony/latest")
    .then(async (res) => {
      if (!res.ok) return
      const data = (await res.json()) as unknown
      if (typeof data !== "object" || !data) return
      if (!("version" in data)) return
      const value = (data as { version?: unknown }).version
      if (typeof value !== "string") return
      return value
    })
    .catch(() => undefined)

  const local = await (async () => {
    const paths = [
      path.resolve(import.meta.dir, "../../app/package.json"),
      path.resolve(import.meta.dir, "../../opencode/package.json"),
    ]
    for (const item of paths) {
      const file = Bun.file(item)
      const exists = await file.exists()
      if (!exists) continue
      const data = (await file.json()) as unknown
      if (typeof data !== "object" || !data) continue
      if (!("version" in data)) continue
      const value = (data as { version?: unknown }).version
      if (typeof value !== "string") continue
      return value
    }
  })()

  const tag = await $`git tag --list "v*" --sort=-version:refname | head -n 1`
    .text()
    .then((x) => x.trim())
    .then((x) => (x.startsWith("v") ? x.slice(1) : x))

  const current = npm ?? local ?? (tag.length > 0 ? tag : "0.0.0")
  return bump(current, env.OPENCODE_BUMP)
})()

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
    return env.OPENCODE_RELEASE
  },
}
console.log(`code-harmony script`, JSON.stringify(Script, null, 2))
