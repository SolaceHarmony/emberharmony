#!/usr/bin/env bun
import { $ } from "bun"
import path from "path"
import pkg from "../package.json"
import { Script } from "@opencode-harmony/script"
import { fileURLToPath } from "url"

const dir = fileURLToPath(new URL("..", import.meta.url))
process.chdir(dir)
const root = path.resolve(dir, "..", "..")
const publishName = process.env.OPENCODE_PUBLISH_NAME ?? pkg.name
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

await Bun.file(`./dist/${pkg.name}/package.json`).write(
  JSON.stringify(
    {
      name: publishName,
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

const publishPlatforms = process.env.OPENCODE_PUBLISH_PLATFORMS === "1"
if (!publishPlatforms) {
  console.log("Skipping platform package publish (set OPENCODE_PUBLISH_PLATFORMS=1 to enable)")
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

// registries
if (!Script.preview) {
  // Calculate SHA values
  const arm64Sha = await $`sha256sum ./dist/${cliName}-linux-arm64.tar.gz | cut -d' ' -f1`.text().then((x) => x.trim())
  const x64Sha = await $`sha256sum ./dist/${cliName}-linux-x64.tar.gz | cut -d' ' -f1`.text().then((x) => x.trim())
  const macX64Sha = await $`sha256sum ./dist/${cliName}-darwin-x64.zip | cut -d' ' -f1`.text().then((x) => x.trim())
  const macArm64Sha = await $`sha256sum ./dist/${cliName}-darwin-arm64.zip | cut -d' ' -f1`.text().then((x) => x.trim())

  const [pkgver, _subver = ""] = Script.version.split(/(-.*)/, 2)

  /*
  // arch
  const binaryPkgbuild = [
    "# Maintainer: dax",
    "# Maintainer: adam",
    "",
    "pkgname='code-harmony-bin'",
    `pkgver=${pkgver}`,
    `_subver=${_subver}`,
    "options=('!debug' '!strip')",
    "pkgrel=1",
    "pkgdesc='The AI coding agent built for the terminal.'",
    "url='https://github.com/SolaceHarmony/code-harmony'",
    "arch=('aarch64' 'x86_64')",
    "license=('MIT')",
    "provides=('code-harmony')",
    "conflicts=('code-harmony')",
    "depends=('ripgrep')",
    "",
    `source_aarch64=("\${pkgname}_\${pkgver}_aarch64.tar.gz::https://github.com/SolaceHarmony/code-harmony/releases/download/v\${pkgver}\${_subver}/${cliName}-linux-arm64.tar.gz")`,
    `sha256sums_aarch64=('${arm64Sha}')`,

    `source_x86_64=("\${pkgname}_\${pkgver}_x86_64.tar.gz::https://github.com/SolaceHarmony/code-harmony/releases/download/v\${pkgver}\${_subver}/${cliName}-linux-x64.tar.gz")`,
    `sha256sums_x86_64=('${x64Sha}')`,
    "",
    "package() {",
    `  install -Dm755 ./${cliName} "${pkgdir}/usr/bin/${cliName}"`,
    "}",
    "",
  ].join("\n")

  // Source-based PKGBUILD for opencode
  const sourcePkgbuild = [
    "# Maintainer: dax",
    "# Maintainer: adam",
    "",
    "pkgname='code-harmony'",
    `pkgver=${pkgver}`,
    `_subver=${_subver}`,
    "options=('!debug' '!strip')",
    "pkgrel=1",
    "pkgdesc='The AI coding agent built for the terminal.'",
    "url='https://github.com/SolaceHarmony/code-harmony'",
    "arch=('aarch64' 'x86_64')",
    "license=('MIT')",
    "provides=('code-harmony')",
    "conflicts=('code-harmony-bin')",
    "depends=('ripgrep')",
    "makedepends=('git' 'bun' 'go')",
    "",
    `source=("opencode-\${pkgver}.tar.gz::https://github.com/SolaceHarmony/code-harmony/archive/v\${pkgver}\${_subver}.tar.gz")`,
    `sha256sums=('SKIP')`,
    "",
    "build() {",
    `  cd "opencode-\${pkgver}"`,
    `  bun install`,
    "  cd ./packages/opencode",
    `  OPENCODE_CHANNEL=latest OPENCODE_VERSION=${pkgver} bun run ./script/build.ts --single`,
    "}",
    "",
    "package() {",
    `  cd "opencode-\${pkgver}/packages/opencode"`,
    '  mkdir -p "${pkgdir}/usr/bin"',
    '  target_arch="x64"',
    '  case "$CARCH" in',
    '    x86_64) target_arch="x64" ;;',
    '    aarch64) target_arch="arm64" ;;',
    '    *) printf "unsupported architecture: %s\\n" "$CARCH" >&2 ; return 1 ;;',
    "  esac",
    '  libc=""',
    "  if command -v ldd >/dev/null 2>&1; then",
    "    if ldd --version 2>&1 | grep -qi musl; then",
    '      libc="-musl"',
    "    fi",
    "  fi",
    '  if [ -z "$libc" ] && ls /lib/ld-musl-* >/dev/null 2>&1; then',
    '    libc="-musl"',
    "  fi",
    '  base=""',
    '  if [ "$target_arch" = "x64" ]; then',
    "    if ! grep -qi avx2 /proc/cpuinfo 2>/dev/null; then",
    '      base="-baseline"',
    "    fi",
    "  fi",
    `  bin="dist/opencode-linux-\${target_arch}\${base}\${libc}/bin/${cliName}"`,
    '  if [ ! -f "$bin" ]; then',
    '    printf "unable to find binary for %s%s%s\\n" "$target_arch" "$base" "$libc" >&2',
    "    return 1",
    "  fi",
    `  install -Dm755 "$bin" "${pkgdir}/usr/bin/${cliName}"`,
    "}",
    "",
  ].join("\n")

  for (const [pkg, pkgbuild] of [
    ["code-harmony-bin", binaryPkgbuild],
    ["code-harmony", sourcePkgbuild],
  ]) {
    for (let i = 0; i < 30; i++) {
      try {
        await $`rm -rf ./dist/aur-${pkg}`
        await $`git clone ssh://aur@aur.archlinux.org/${pkg}.git ./dist/aur-${pkg}`
        await $`cd ./dist/aur-${pkg} && git checkout master`
        await Bun.file(`./dist/aur-${pkg}/PKGBUILD`).write(pkgbuild)
        await $`cd ./dist/aur-${pkg} && makepkg --printsrcinfo > .SRCINFO`
        await $`cd ./dist/aur-${pkg} && git add PKGBUILD .SRCINFO`
        await $`cd ./dist/aur-${pkg} && git commit -m "Update to v${Script.version}"`
        await $`cd ./dist/aur-${pkg} && git push`
        break
      } catch (e) {
        continue
      }
    }
  }
  */

  // Tap updates disabled for now.
}
