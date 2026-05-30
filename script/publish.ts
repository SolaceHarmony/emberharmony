#!/usr/bin/env bun

import { $ } from "bun"
import { Script } from "@thesolaceproject/emberharmony-script"

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

await $`BUN_SECURITY_SCAN=0 bun install --config=packages/app/bunfig-ci.toml`
await import(`../packages/sdk/js/script/build.ts`)

// Publishing is triggered by a published GitHub release, so the tag already exists
// and the release is already public. This script never commits, tags, or pushes —
// it only updates in-tree version strings (above) and publishes the built artifacts.

console.log("\n=== cli ===\n")
await import(`../packages/emberharmony/script/publish.ts`)

const publishAll = process.env.EMBERHARMONY_PUBLISH_ALL === "1"

console.log("\n=== sdk ===\n")
if (!publishAll) {
  console.log("Skipping SDK publish (set EMBERHARMONY_PUBLISH_ALL=1 to enable)")
}
if (publishAll) {
  await import(`../packages/sdk/js/script/publish.ts`)
}

console.log("\n=== plugin ===\n")
if (!publishAll) {
  console.log("Skipping plugin publish (set EMBERHARMONY_PUBLISH_ALL=1 to enable)")
}
if (publishAll) {
  await import(`../packages/plugin/script/publish.ts`)
}

const dir = new URL("..", import.meta.url).pathname
process.chdir(dir)
