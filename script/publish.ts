#!/usr/bin/env bun

import { $ } from "bun"
import { Script } from "@thesolaceproject/code-harmony-script"

const highlightsTemplate = `
<!--
Add highlights before publishing. Delete this section if no highlights.

- For multiple highlights, use multiple <highlight> tags
- Highlights with the same source attribute get grouped together
-->

<!--
<highlight source="SourceName (TUI/Desktop/Web/Core)">
  <h2>Feature title goes here</h2>
  <p short="Short description used for Desktop Recap">
    Full description of the feature or change
  </p>

  https://github.com/user-attachments/assets/uuid-for-video (you will want to drag & drop the video or picture)

  <img
    width="1912"
    height="1164"
    alt="image"
    src="https://github.com/user-attachments/assets/uuid-for-image"
  />
</highlight>
-->

`

console.log("=== publishing ===\n")

const pkgjsons = await Array.fromAsync(
  new Bun.Glob("**/package.json").scan({
    absolute: true,
  }),
).then((arr) => arr.filter((x) => !x.includes("node_modules") && !x.includes("dist")))

for (const file of pkgjsons) {
  let pkg = await Bun.file(file).text()
  pkg = pkg.replaceAll(/"version": "[^"]+"/g, `"version": "${Script.version}"`)
  console.log("updated:", file)
  await Bun.file(file).write(pkg)
}

const extensionToml = new URL("../packages/extensions/zed/extension.toml", import.meta.url).pathname
let toml = await Bun.file(extensionToml).text()
toml = toml.replace(/^version = "[^"]+"/m, `version = "${Script.version}"`)
toml = toml.replaceAll(/releases\/download\/v[^/]+\//g, `releases/download/v${Script.version}/`)
console.log("updated:", extensionToml)
await Bun.file(extensionToml).write(toml)

await $`BUN_SECURITY_SCAN=0 bun install --config=packages/app/bunfig-ci.toml`
await import(`../packages/sdk/js/script/build.ts`)

if (Script.release) {
  const skipGit = process.env.OPENCODE_SKIP_GIT === "1"
  if (!skipGit) {
    const changed = await $`git status --porcelain=v1`.text().then((x) => x.trim().length > 0)
    if (changed) {
      await $`git commit -am "release: v${Script.version}"`
    } else {
      console.log(`No changes to commit for v${Script.version}`)
    }

    const tag = `v${Script.version}`
    const exists = await $`git rev-parse -q --verify refs/tags/${tag}`.nothrow()
    if (exists.exitCode !== 0) {
      await $`git tag ${tag}`
    } else {
      console.log(`Tag ${tag} already exists`)
    }

    await $`git fetch origin`
    await $`git cherry-pick HEAD..origin/main`.nothrow()
    await $`git push origin HEAD --tags --no-verify --force-with-lease`
  } else {
    console.log("Skipping git commit/tag/push (OPENCODE_SKIP_GIT=1)")
  }

  await new Promise((resolve) => setTimeout(resolve, 5_000))
  await $`gh release edit v${Script.version} --draft=false`
}

console.log("\n=== cli ===\n")
await import(`../packages/code-harmony/script/publish.ts`)

const publishAll = process.env.OPENCODE_PUBLISH_ALL === "1"

console.log("\n=== sdk ===\n")
if (!publishAll) {
  console.log("Skipping SDK publish (set OPENCODE_PUBLISH_ALL=1 to enable)")
}
if (publishAll) {
  await import(`../packages/sdk/js/script/publish.ts`)
}

console.log("\n=== plugin ===\n")
if (!publishAll) {
  console.log("Skipping plugin publish (set OPENCODE_PUBLISH_ALL=1 to enable)")
}
if (publishAll) {
  await import(`../packages/plugin/script/publish.ts`)
}

const dir = new URL("..", import.meta.url).pathname
process.chdir(dir)
