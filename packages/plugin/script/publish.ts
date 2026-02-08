#!/usr/bin/env bun
import { Script } from "@opencode-harmony/script"
import { $ } from "bun"

const dir = new URL("..", import.meta.url).pathname
process.chdir(dir)

await $`bun tsc`
const pkg = await import("../package.json").then((m) => m.default)
const original = JSON.parse(JSON.stringify(pkg))
for (const [key, value] of Object.entries(pkg.exports)) {
  const file = value.replace("./src/", "./dist/").replace(".ts", "")
  // @ts-ignore
  pkg.exports[key] = {
    import: file + ".js",
    types: file + ".d.ts",
  }
}
await Bun.write("package.json", JSON.stringify(pkg, null, 2))
await $`bun pm pack`
const tarballs = await Array.fromAsync(new Bun.Glob(`*${Script.version}*.tgz`).scan())
const fallback = await Array.fromAsync(new Bun.Glob("*.tgz").scan())
const files = tarballs.length > 0 ? tarballs : fallback
const tarball = files[0]
if (!tarball) {
  throw new Error("No tarball found to publish")
}
const delays = [0, 5000, 10000]
const errors: string[] = []
for (const delay of delays) {
  if (delay > 0) {
    await new Promise((resolve) => setTimeout(resolve, delay))
  }
  const result = await $`npm publish ${tarball} --tag ${Script.channel} --access public`.nothrow()
  if (result.exitCode === 0) {
    await Bun.write("package.json", JSON.stringify(original, null, 2))
    return
  }
  const stderr = result.stderr.toString()
  if (stderr.includes("cannot be republished") || stderr.includes("previously published")) {
    console.log(`skip publish ${tarball}`)
    await Bun.write("package.json", JSON.stringify(original, null, 2))
    return
  }
  if (stderr.includes("Too Many Requests") || stderr.includes("rate limited")) {
    errors.push(stderr)
    continue
  }
  await Bun.write("package.json", JSON.stringify(original, null, 2))
  throw new Error(`npm publish failed: ${stderr}`)
}
await Bun.write("package.json", JSON.stringify(original, null, 2))
const last = errors[errors.length - 1]
throw new Error(`npm publish failed: ${last}`)
