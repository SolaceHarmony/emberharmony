#!/usr/bin/env bun

import { Script } from "@thesolaceproject/code-harmony-script"
import { $ } from "bun"

const main = async () => {
  const dir = new URL("..", import.meta.url).pathname
  process.chdir(dir)

  const pkg = await import("../package.json").then((m) => m.default)
  const original = JSON.parse(JSON.stringify(pkg))
  for (const [key, value] of Object.entries(pkg.exports)) {
    if (typeof value !== "string") {
      continue
    }
    const file = value.replace("./src/", "./dist/").replace(".ts", "")
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
  await Bun.write("package.json", JSON.stringify(original, null, 2))
  throw new Error(`npm publish failed: ${stderr}`)
}

await main()
