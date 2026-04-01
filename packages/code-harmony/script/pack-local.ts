#!/usr/bin/env bun
import { $ } from "bun"
import path from "path"
import { fileURLToPath } from "url"

const main = async () => {
  const dir = fileURLToPath(new URL("..", import.meta.url))
  process.chdir(dir)

  const pkg = await Bun.file("./package.json").json()
  if (typeof pkg !== "object" || !pkg) {
    throw new Error("invalid package.json")
  }
  const name = "name" in pkg && typeof pkg.name === "string" ? pkg.name : "code-harmony"
  const ver = "version" in pkg && typeof pkg.version === "string" ? pkg.version : "0.0.0"
  const publish = process.env.CODE_HARMONY_PUBLISH_NAME ?? name
  const cli = publish.includes("/") ? publish.split("/").pop() || publish : publish

  const install = process.argv.includes("--install")
  const nobuild = process.argv.includes("--no-build")

  const platform = process.platform === "win32" ? "windows" : process.platform
  const arch = process.arch

  const bin = platform === "windows" ? "code-harmony.exe" : "code-harmony"
  const ext = platform === "windows" ? ".exe" : ""

  if (!nobuild) {
    await $`CODE_HARMONY_CHANNEL=latest CODE_HARMONY_VERSION=${ver} bun run script/build.ts --single`
  }

  const hits = await Array.fromAsync(new Bun.Glob(`${name}-${platform}-${arch}*/bin/${cli}${ext}`).scan({ cwd: "./dist" }))
  if (hits.length === 0) {
    throw new Error(`no build output found for ${platform}-${arch} (expected dist/${name}-${platform}-${arch}*/bin/${cli}${ext})`)
  }

  const hit = hits.sort()[0]
  const outdir = hit.split("/")[0]
  const variant = outdir.replace(`${name}-`, "")

  const stage = path.join(dir, "dist-local", name)
  await $`rm -rf ${stage}`
  await $`mkdir -p ${path.join(stage, "bin", variant)}`

  await $`cp ${path.join(dir, "bin", cli)} ${path.join(stage, "bin", cli)}`
  await $`cp ${path.join(dir, "dist", hit)} ${path.join(stage, "bin", variant, bin)}`

  const root = path.resolve(dir, "..", "..")
  const readme = path.join(root, "README.md")
  const license = path.join(root, "LICENSE")
  if (await Bun.file(readme).exists()) {
    await $`cp ${readme} ${path.join(stage, "README.md")}`
  }
  if (await Bun.file(license).exists()) {
    await $`cp ${license} ${path.join(stage, "LICENSE")}`
  }

  await Bun.write(
    path.join(stage, "package.json"),
    JSON.stringify(
      {
        name: publish,
        version: ver,
        license: "MIT",
        type: "commonjs",
        bin: {
          [cli]: `bin/${cli}`,
        },
        files: ["bin/**", "README.md", "LICENSE", "package.json"],
      },
      null,
      2,
    ) + "\n",
  )

  if (process.platform !== "win32") {
    await $`chmod -R 755 ${path.join(stage, "bin")}`
  }

  const packed = await $`npm pack --json`.cwd(stage).text()
  const json = JSON.parse(packed) as unknown
  if (!Array.isArray(json) || typeof json[0] !== "object" || !json[0]) {
    throw new Error(`unexpected npm pack output: ${packed}`)
  }
  const filename = "filename" in json[0] && typeof json[0].filename === "string" ? json[0].filename : ""
  if (!filename) {
    throw new Error(`unable to determine tarball from npm pack output: ${packed}`)
  }

  const tgz = path.join(stage, filename)
  console.log(tgz)
  console.log(`npm i -g ${tgz}`)

  if (!install) return

  const prefix = path.join(dir, "dist-local", "npm")
  await $`rm -rf ${prefix}`
  await $`mkdir -p ${prefix}`
  await $`npm i -g ${tgz} --prefix ${prefix}`
  const exe = path.join(prefix, "bin", cli)
  await $`${exe} --version`
  console.log(exe)
}

await main()
