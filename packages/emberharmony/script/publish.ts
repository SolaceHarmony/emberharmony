#!/usr/bin/env bun
import { $ } from "bun"
import path from "path"
import pkg from "../package.json"
import { Script } from "@thesolaceproject/emberharmony-script"
import { fileURLToPath } from "url"

const dir = fileURLToPath(new URL("..", import.meta.url))
process.chdir(dir)
const root = path.resolve(dir, "..", "..")
const publishName = process.env.EMBERHARMONY_PUBLISH_NAME ?? "@thesolaceproject/emberharmony"
const cliName = publishName.includes("/") ? publishName.split("/").pop() || publishName : publishName

const binaries: Record<string, string> = {}
const entries: { dir: string; name: string; version: string }[] = []
for (const filepath of new Bun.Glob("*/package.json").scanSync({ cwd: "./dist" })) {
  const data = await Bun.file(`./dist/${filepath}`).json()
  const dir = filepath.split("/")[0]
  entries.push({ dir, name: data.name, version: data.version })
  if (dir === pkg.name) {
    continue
  }
  binaries[data.name] = data.version
}
console.log("binaries", binaries)
const version = Object.values(binaries)[0] || Script.version

await $`mkdir -p ./dist/${pkg.name}`
await $`cp -r ./bin ./dist/${pkg.name}/bin`
await $`cp ./script/postinstall.mjs ./dist/${pkg.name}/postinstall.mjs`

const meta = await Bun.file(path.join(root, "package.json"))
  .json()
  .catch(() => ({}) as unknown)

const description =
  typeof meta === "object" && meta && "description" in meta && typeof meta.description === "string"
    ? meta.description
    : undefined

const homepage =
  typeof meta === "object" && meta && "homepage" in meta && typeof meta.homepage === "string"
    ? meta.homepage
    : undefined

const license =
  typeof meta === "object" && meta && "license" in meta && typeof meta.license === "string" ? meta.license : "MIT"

const repository =
  typeof meta === "object" &&
  meta &&
  "repository" in meta &&
  (typeof meta.repository === "object" || typeof meta.repository === "string")
    ? meta.repository
    : undefined

const bugs =
  typeof meta === "object" && meta && "bugs" in meta && (typeof meta.bugs === "object" || typeof meta.bugs === "string")
    ? meta.bugs
    : undefined

await Bun.file(`./dist/${pkg.name}/package.json`).write(
  JSON.stringify(
    {
      name: publishName,
      description,
      homepage,
      repository,
      bugs,
      license,
      bin: {
        [cliName]: `bin/${cliName}`,
      },
      scripts: {
        postinstall: "bun ./postinstall.mjs || node ./postinstall.mjs",
      },
      version: version,
      optionalDependencies: binaries,
      files: ["bin/**", "postinstall.mjs", "README.md", "LICENSE"],
    },
    null,
    2,
  ),
)

const copyMeta = async (target: string) => {
  const readme = path.join(root, "README.md")
  const license = path.join(root, "LICENSE")
  if (await Bun.file(readme).exists()) {
    await $`cp ${readme} ${path.join(target, "README.md")}`
  }
  if (await Bun.file(license).exists()) {
    await $`cp ${license} ${path.join(target, "LICENSE")}`
  }
}

await copyMeta(`./dist/${pkg.name}`)

const publish = async (target: string) => {
  await copyMeta(target)
  await $`bun pm pack`.cwd(target)
  const tarballs = await Array.fromAsync(new Bun.Glob(`*${Script.version}*.tgz`).scan({ cwd: target }))
  const fallback = await Array.fromAsync(new Bun.Glob("*.tgz").scan({ cwd: target }))
  const files = tarballs.length > 0 ? tarballs : fallback
  const tarball = files[0]
  if (!tarball) {
    throw new Error(`No tarball found to publish in ${target}`)
  }
  const errors: string[] = []
  const delays = [0, 10_000, 30_000, 60_000, 120_000]
  for (const delay of delays) {
    if (delay > 0) {
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
    const result = await $`npm publish ${tarball} --access public --tag ${Script.channel}`.cwd(target).nothrow()
    if (result.exitCode === 0) {
      return
    }
    const stderr = result.stderr.toString()
    if (stderr.includes("cannot be republished") || stderr.includes("previously published")) {
      console.log(`skip publish ${tarball}`)
      return
    }
    const retryable =
      stderr.includes("Failed to save packument") ||
      stderr.includes("npm error code E409") ||
      stderr.includes("409 Conflict") ||
      stderr.includes("Too Many Requests") ||
      stderr.includes("rate limited")
    if (retryable) {
      errors.push(stderr)
      continue
    }
    throw new Error(`npm publish failed: ${stderr}`)
  }
  const last = errors[errors.length - 1]
  if (!last) {
    throw new Error("npm publish failed")
  }
  throw new Error(`npm publish failed after retries: ${last}`)
}

const publishPlatforms = process.env.EMBERHARMONY_PUBLISH_PLATFORMS === "1"
if (!publishPlatforms) {
  console.log("Skipping platform package publish (set EMBERHARMONY_PUBLISH_PLATFORMS=1 to enable)")
}
if (publishPlatforms) {
  const items = entries.filter((entry) => entry.dir !== pkg.name)
  for (const entry of items) {
    if (process.platform !== "win32") {
      await $`chmod -R 755 .`.cwd(`./dist/${entry.dir}`)
    }
    await publish(`./dist/${entry.dir}`)
  }
}
await publish(`./dist/${pkg.name}`)
