#!/usr/bin/env bun
import { $ } from "bun"

import { copyBinaryToSidecarFolder, getCurrentSidecar, windowsify } from "./utils"
import { Script } from "@thesolaceproject/emberharmony-script"

const pkg = await Bun.file("./package.json").json()
pkg.version = Script.version
await Bun.write("./package.json", JSON.stringify(pkg, null, 2) + "\n")
console.log(`Updated package.json version to ${Script.version}`)

const sidecar = getCurrentSidecar()

const dir = "src-tauri/target/emberharmony-binaries"

await $`mkdir -p ${dir}`
await $`gh run download ${Bun.env.GITHUB_RUN_ID} -n emberharmony-cli`.cwd(dir)

await copyBinaryToSidecarFolder(windowsify(`${dir}/${sidecar.ocBinary}/bin/emberharmony`))
