#!/usr/bin/env bun
import { $ } from "bun"

import { Script } from "@thesolaceproject/code-harmony-script"
import { copyBinaryToSidecarFolder, getCurrentSidecar, windowsify } from "./utils"

async function tauri() {
  const path = "./src-tauri/tauri.prod.conf.json"
  const data = await Bun.file(path).json()
  const pub = (Bun.env.TAURI_UPDATER_PUBKEY ?? "").trim()
  const key = (Bun.env.TAURI_SIGNING_PRIVATE_KEY ?? "").trim()
  const on = pub.length > 0 && key.length > 0

  if (!data.bundle) data.bundle = {}
  data.bundle.createUpdaterArtifacts = on

  if (!on) {
    if (data.plugins && data.plugins.updater) delete data.plugins.updater
    await Bun.write(path, JSON.stringify(data, null, 2) + "\n")
    console.log("Updater disabled (missing TAURI_SIGNING_PRIVATE_KEY or TAURI_UPDATER_PUBKEY)")
    return
  }

  if (!data.plugins) data.plugins = {}
  if (!data.plugins.updater) data.plugins.updater = {}
  data.plugins.updater.pubkey = pub
  await Bun.write(path, JSON.stringify(data, null, 2) + "\n")
  console.log("Updater enabled (pubkey injected from TAURI_UPDATER_PUBKEY)")
}

const pkg = await Bun.file("./package.json").json()
pkg.version = Script.version
await Bun.write("./package.json", JSON.stringify(pkg, null, 2) + "\n")
console.log(`Updated package.json version to ${Script.version}`)

await tauri()

const sidecar = getCurrentSidecar()

const dir = "src-tauri/target/code-harmony-binaries"

await $`mkdir -p ${dir}`
await $`gh run download ${Bun.env.GITHUB_RUN_ID} -n code-harmony-cli`.cwd(dir)

await copyBinaryToSidecarFolder(windowsify(`${dir}/${sidecar.ocBinary}/bin/code-harmony`))
