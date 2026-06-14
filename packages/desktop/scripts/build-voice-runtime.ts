#!/usr/bin/env bun
/**
 * Assemble the self-contained voice runtime shipped inside the desktop app.
 *
 * The LiveKit agents framework forks `node_modules` scripts and dynamically
 * imports the agent file, so it cannot run inside the compiled single-file CLI
 * sidecar (no on-disk node_modules). Instead we ship a small Bun runtime + the
 * bundled worker + the pruned native deps + the ONNX models as a Tauri
 * resource, and the CLI sidecar spawns `bun agent.js start` against it.
 *
 * Output: packages/desktop/src-tauri/resources/voice/
 *   bun(.exe)        platform Bun runtime
 *   agent.js         bundled worker (emberharmony code inlined; native deps external)
 *   node_modules/    voice deps, pruned to the target platform's native binaries
 *   models/          pre-downloaded Silero VAD + turn-detector ONNX models
 *
 * Usage:
 *   bun run scripts/build-voice-runtime.ts                 # current platform
 *   bun run scripts/build-voice-runtime.ts --os darwin --arch arm64
 */
import { $ } from "bun"
import { existsSync } from "node:fs"
import { cp, mkdir, rm, readdir, chmod, stat } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

const BUN_VERSION = "1.3.8"

function parseArg(name: string, fallback: string) {
  const i = process.argv.indexOf(`--${name}`)
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1]! : fallback
}

const hostOs = process.platform === "win32" ? "windows" : process.platform // darwin | linux | windows
const hostArch = process.arch === "x64" ? "x64" : process.arch // x64 | arm64
const targetOs = parseArg("os", hostOs) as "darwin" | "linux" | "windows"
const targetArch = parseArg("arch", hostArch) as "x64" | "arm64"
const isCross = targetOs !== hostOs || targetArch !== hostArch

const desktopDir = path.resolve(fileURLToPath(import.meta.url), "../..")
const repoRoot = path.resolve(desktopDir, "../..")
const emberharmonyDir = path.join(repoRoot, "packages/emberharmony")
const outDir = path.join(desktopDir, "src-tauri/resources/voice")

console.log(`[voice-runtime] target: ${targetOs}/${targetArch} (host ${hostOs}/${hostArch}${isCross ? ", CROSS" : ""})`)

await rm(outDir, { recursive: true, force: true })
await mkdir(path.join(outDir, "node_modules"), { recursive: true })

// --- 1. Bundle the worker -------------------------------------------------
// Inline the emberharmony code (flag/bridge/registry/constants/workflow) and
// leave the native LiveKit/onnx deps external so they resolve from node_modules.
console.log("[voice-runtime] bundling agent.js")
const externalArgs = ["@livekit/*", "livekit-*", "onnxruntime-*", "@msgpack/*", "pino", "ws", "sharp"].flatMap(
  (e) => ["--external", e],
)
await $`bun build ${path.join(emberharmonyDir, "src/voice/agent.ts")} --target=bun --outfile=${path.join(outDir, "agent.js")} ${externalArgs}`.quiet()

// --- 2. Install the voice deps, pruned to the target platform -------------
console.log("[voice-runtime] installing voice deps")
const stage = path.join(repoRoot, ".voice-runtime-stage")
await rm(stage, { recursive: true, force: true })
await mkdir(stage, { recursive: true })
await Bun.write(
  path.join(stage, "package.json"),
  JSON.stringify(
    {
      name: "emberharmony-voice-runtime",
      version: "0.0.0",
      dependencies: {
        "@livekit/agents": "1.4.5",
        "@livekit/agents-plugin-livekit": "1.4.5",
        "@livekit/agents-plugin-silero": "1.4.5",
        "@livekit/rtc-node": "0.13.29",
      },
    },
    null,
    2,
  ),
)
// --os/--cpu prune the optional native deps (rtc-ffi-bindings-*) to the target.
const bunOs = targetOs === "windows" ? "win32" : targetOs
await $`bun install --no-save --production --os=${bunOs} --cpu=${targetArch}`
  .cwd(stage)
  .env({ ...process.env, BUN_SECURITY_SCAN: "0" })
  .quiet()
await cp(path.join(stage, "node_modules"), path.join(outDir, "node_modules"), { recursive: true, dereference: true })

// onnxruntime-node bundles every platform in one package; keep only the target.
const ort = path.join(outDir, "node_modules/onnxruntime-node/bin")
if (existsSync(ort)) {
  for (const napi of await readdir(ort)) {
    const napiDir = path.join(ort, napi)
    if (!(await stat(napiDir)).isDirectory()) continue
    for (const os of await readdir(napiDir)) {
      if (os !== bunOs) {
        await rm(path.join(napiDir, os), { recursive: true, force: true })
        continue
      }
      for (const arch of await readdir(path.join(napiDir, os))) {
        if (arch !== targetArch) await rm(path.join(napiDir, os, arch), { recursive: true, force: true })
      }
    }
  }
}

// --- 3. Pre-download the ONNX models (Silero VAD + turn detector) ----------
// `download-files` populates the HF/agents caches; point them at models/ so the
// bundled worker finds them offline. Only possible for a same-platform host
// (the JS download path is platform-independent, so host == any works).
console.log("[voice-runtime] downloading ONNX models")
const modelsDir = path.join(outDir, "models")
await mkdir(modelsDir, { recursive: true })
const modelEnv = { ...process.env, HF_HOME: modelsDir, XDG_CACHE_HOME: modelsDir, BUN_SECURITY_SCAN: "0" }
await $`bun run ${path.join(outDir, "agent.js")} download-files`
  .cwd(outDir)
  .env(modelEnv)
  .quiet()

// --- 3b. Prune the staged node_modules to runtime-only -------------------
// The install pulls heavy transitive deps the worker never loads. Drop the
// ones with zero references in the worker's import graph, then strip
// non-runtime files (source maps, type decls, docs, test fixtures). This runs
// AFTER the model download, which needs @huggingface/hub and onnxruntime-node.
console.log("[voice-runtime] pruning node_modules")
const nm = path.join(outDir, "node_modules")
// onnxruntime-web is the browser build; the worker uses onnxruntime-node only.
// typescript is a dev/types dependency never required at runtime.
for (const dead of ["onnxruntime-web", "typescript"]) {
  await rm(path.join(nm, dead), { recursive: true, force: true })
}
// Strip files that are pure build/debug/docs weight (the deps run as plain JS).
const stripExt = new Set([".map", ".d.ts", ".d.cts", ".d.mts", ".md", ".markdown"])
const stripDir = new Set(["test", "tests", "__tests__", "docs", "example", "examples"])
async function strip(dir: string) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (stripDir.has(entry.name)) await rm(full, { recursive: true, force: true })
      else await strip(full)
    } else if ([...stripExt].some((e) => entry.name.endsWith(e))) {
      await rm(full, { force: true })
    }
  }
}
await strip(nm)

// --- 4. Stage the Bun runtime for the target ------------------------------
console.log("[voice-runtime] staging bun runtime")
const bunBin = path.join(outDir, targetOs === "windows" ? "bun.exe" : "bun")
if (!isCross) {
  const which = (await $`which bun`.text()).trim()
  const real = await Bun.file(which).exists() ? which : (await $`readlink -f ${which}`.text()).trim()
  await cp(real, bunBin, { dereference: true })
} else {
  // Cross-platform (CI): download the matching Bun release.
  const bunTarget =
    targetOs === "darwin"
      ? `bun-darwin-${targetArch === "arm64" ? "aarch64" : "x64"}`
      : targetOs === "linux"
        ? `bun-linux-${targetArch === "arm64" ? "aarch64" : "x64"}`
        : `bun-windows-x64`
  const url = `https://github.com/oven-sh/bun/releases/download/bun-v${BUN_VERSION}/${bunTarget}.zip`
  const tmpZip = path.join(stage, "bun.zip")
  await $`curl -fsSL ${url} -o ${tmpZip}`.quiet()
  await $`unzip -q -o ${tmpZip} -d ${stage}/bun-extract`.quiet()
  const extracted = path.join(stage, "bun-extract", bunTarget, targetOs === "windows" ? "bun.exe" : "bun")
  await cp(extracted, bunBin)
}
if (targetOs !== "windows") await chmod(bunBin, 0o755)

await rm(stage, { recursive: true, force: true })

const total = (await $`du -sh ${outDir}`.text()).trim().split("\t")[0]
console.log(`[voice-runtime] done -> ${path.relative(repoRoot, outDir)} (${total})`)
