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
import { cp, mkdir, rm, readdir, chmod, stat, realpath } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"

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

// Single source of truth. The bundled runtime must install exactly the
// @livekit versions agent.ts is compiled against (the workspace catalog), and
// the staged bun must match the repo's bun (packageManager) — hardcoding either
// drifts silently from the rest of the repo.
const rootPkg = JSON.parse(await Bun.file(path.join(repoRoot, "package.json")).text())
const BUN_VERSION = String(rootPkg.packageManager ?? "").replace(/^bun@/, "")
// Allow pre-release/canary tags (e.g. 1.3.8-canary.1), not just exact semver.
if (!/^\d+\.\d+\.\d+(?:-.+)?$/.test(BUN_VERSION)) {
  throw new Error(`[voice-runtime] no bun version in root package.json "packageManager": ${rootPkg.packageManager}`)
}
// Catalogs may live at the top level or under workspaces (bun supports both).
const catalog = rootPkg.catalog ?? rootPkg.workspaces?.catalog ?? {}
const voiceDeps: Record<string, string> = {}
for (const name of ["@livekit/agents", "@livekit/agents-plugin-livekit", "@livekit/agents-plugin-silero", "@livekit/rtc-node"]) {
  const version = catalog[name]
  if (!version) throw new Error(`[voice-runtime] ${name} missing from root package.json workspaces.catalog`)
  voiceDeps[name] = version
}

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
      dependencies: voiceDeps,
    },
    null,
    2,
  ),
)
// Install with npm, not bun. Bun's standalone install deep-nests
// @livekit/agents' @opentelemetry version conflicts (OTel 1.x AND 2.x are both
// genuinely required) 4-5 node_modules levels deep, blowing past Windows'
// 260-char MAX_PATH in the NSIS bundler. npm's hoisting collapses the same
// unavoidable conflicts into a shallow, shippable tree. npm resolves optional
// native deps (rtc-ffi-bindings-*) for the host, so the runner must be the
// target platform — CI builds each platform on its own native runner.
if (isCross) {
  throw new Error(
    `[voice-runtime] cannot cross-assemble for ${targetOs}/${targetArch} on ${hostOs}/${hostArch}: ` +
      "npm installs the host's native deps — run the assembly on a native runner for the target.",
  )
}
const bunOs = targetOs === "windows" ? "win32" : targetOs // onnxruntime-node bin dir os name
await $`npm install --omit=dev --no-audit --no-fund --loglevel=error`.cwd(stage).quiet()
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

// --- 2c. Patch the agents job-proc loader for paths with special chars ----
// @livekit/agents forks job_proc_lazy_main.js and loads the agent with
// `import(pathToFileURL(moduleFile).pathname)`. `.pathname` keeps percent-
// encoding (a space becomes %20) but drops the file:// scheme, so import()
// looks for a literal "%20" on disk and fails — which happens whenever the
// install path contains a space ("EmberHarmony Dev.app", Windows "Program
// Files", a user home with a space). The framework's own download.js uses
// `.href` for the same call; mirror that. Fail loudly if the expected source
// is absent (upstream changed it or the file moved) rather than shipping a
// silently-broken worker.
{
  const loader = path.join(outDir, "node_modules/@livekit/agents/dist/ipc/job_proc_lazy_main.js")
  const before = "import(pathToFileURL(moduleFile).pathname)"
  const after = "import(pathToFileURL(moduleFile).href)"
  const src = await Bun.file(loader).text()
  if (!src.includes(before)) {
    throw new Error(
      `[voice-runtime] cannot patch job_proc_lazy_main.js: expected \`${before}\` not found in ${loader}. ` +
        "The @livekit/agents loader changed; re-verify the agent-path fix before shipping.",
    )
  }
  await Bun.write(loader, src.replace(before, after))
  console.log("[voice-runtime] patched agents job-proc loader (pathname -> href)")
}

// --- 3. Pre-download the ONNX models (Silero VAD + turn detector) ----------
// `download-files` populates the HF/agents caches; point them at models/ so the
// bundled worker finds them offline. Only possible for a same-platform host
// (the JS download path is platform-independent, so host == any works).
console.log("[voice-runtime] downloading ONNX models")
const modelsDir = path.join(outDir, "models")
await mkdir(modelsDir, { recursive: true })
// The turn-detector plugin's hf_utils caches under os.homedir()/.cache/
// huggingface/hub and ignores HF_HOME entirely; os.homedir() honors $HOME, so
// HOME is what actually lands the model inside the bundle. The worker sets the
// SAME HOME at runtime to resolve it offline. HF_HOME/XDG_CACHE_HOME stay for
// libs that do read them; cwd is already outDir for any cwd-relative cache.
const modelEnv = {
  ...process.env,
  HOME: modelsDir,
  HF_HOME: modelsDir,
  XDG_CACHE_HOME: modelsDir,
  BUN_SECURITY_SCAN: "0",
}
// HuggingFace downloads flake — rate limits and transient errors surface as
// e.g. "tokenizerConfig.tokenizer_class undefined" (a non-JSON response parsed
// as the tokenizer config). A single flake otherwise aborts the whole build,
// so retry with backoff and only fail loudly once attempts are exhausted.
// Use the dedicated `livekit-agents download-files` bin, NOT `agent.js
// download-files`. The bin discovers @livekit/agents-plugin-* packages in
// node_modules and runs their downloadFiles() directly — without loading the
// agent, so it needs no LiveKit creds, no ffmpeg, and doesn't depend on the
// agent having registered its plugins. The deprecated agent.js/cli.runApp path
// silently downloaded nothing (exit 0, empty cache).
const downloadBin = path.join(outDir, "node_modules/@livekit/agents/dist/bin/livekit-agents.js")
const MODEL_DL_ATTEMPTS = 4
let lastDownloadOutput = "(no output captured)"
for (let attempt = 1; ; attempt++) {
  const res = await $`bun ${downloadBin} download-files`.cwd(outDir).env(modelEnv).quiet().nothrow()
  lastDownloadOutput =
    [res.stdout.toString().trim(), res.stderr.toString().trim()].filter(Boolean).join("\n\n") || "(no output captured)"
  if (res.exitCode === 0) break
  if (attempt >= MODEL_DL_ATTEMPTS) {
    throw new Error(`[voice-runtime] model download failed after ${MODEL_DL_ATTEMPTS} attempts:\n${lastDownloadOutput}`)
  }
  const backoffSec = attempt * 5
  console.log(`[voice-runtime] model download attempt ${attempt}/${MODEL_DL_ATTEMPTS} failed; retrying in ${backoffSec}s`)
  await Bun.sleep(backoffSec * 1000)
}

// Replace symlinks with real files. The HF hub cache stores model files as
// symlinks (snapshots/*/onnx/model_q8.onnx -> ../../blobs/<etag>); symlinks
// don't survive Windows installers and are fragile across bundlers, so we
// dereference every symlink in the runtime into a real copy. (node_modules was
// already copied with dereference:true; the HF cache is the remaining source.)
console.log("[voice-runtime] dereferencing symlinks -> real files")
async function dereferenceSymlinks(dir: string): Promise<void> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isSymbolicLink()) {
      const target = await realpath(full) // resolves the link to the real file/dir
      await rm(full)
      await cp(target, full, { recursive: true, dereference: true })
    } else if (entry.isDirectory()) {
      await dereferenceSymlinks(full)
    }
  }
}
await dereferenceSymlinks(modelsDir)

// Drop the now-orphaned HF blob store. After dereferencing, each model lives as
// a real file under snapshots/; the runtime's local_files_only path reads only
// refs/ + snapshots/ (verified in hf_utils.downloadFileToCacheDir — blobs/ is
// never read), so keeping blobs/ would ship every model twice (the multilingual
// turn detector alone is ~378MB).
for (const blobs of await Array.fromAsync(
  new Bun.Glob("**/blobs").scan({ cwd: modelsDir, dot: true, onlyFiles: false }),
)) {
  await rm(path.join(modelsDir, blobs), { recursive: true, force: true })
}

// Fail loud if the turn-detector assets didn't actually land in the bundle.
// download-files can exit 0 while caching nothing reachable, which silently
// ships a voice runtime that can't do turn detection — the runtime loads BOTH
// with local_files_only and throws if either is missing:
//   - model:     @huggingface/hub caches under HOME/.cache/huggingface/hub/...
//   - tokenizer: @huggingface/transformers caches under its OWN package dir
//                (node_modules/@huggingface/transformers/.cache/...)
// Globs need dot:true to descend into those .cache directories. After
// dereferencing, model_q8.onnx is a real file, so no followSymlinks needed.
const bundledModel = await Array.fromAsync(new Bun.Glob("**/model_q8.onnx").scan({ cwd: modelsDir, dot: true }))
const bundledTokenizer = await Array.fromAsync(
  new Bun.Glob("node_modules/@huggingface/transformers/.cache/**/tokenizer.json").scan({ cwd: outDir, dot: true }),
)
if (bundledModel.length === 0 || bundledTokenizer.length === 0) {
  throw new Error(
    `[voice-runtime] turn-detector assets missing after download — ` +
      `model_q8.onnx: ${bundledModel.length}, tokenizer.json: ${bundledTokenizer.length}. ` +
      `The HuggingFace cache is not landing in the bundle.\n\ndownload-files output:\n${lastDownloadOutput}`,
  )
}

// Guard: no symlinks may ship — Windows installers (NSIS/MSI) can't represent
// them, and they break when the bundle is relocated. Catch any that slip in.
async function findSymlinks(dir: string, found: string[] = []): Promise<string[]> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isSymbolicLink()) found.push(path.relative(outDir, full))
    else if (entry.isDirectory()) await findSymlinks(full, found)
  }
  return found
}
const symlinks = await findSymlinks(outDir)
if (symlinks.length > 0) {
  throw new Error(
    `[voice-runtime] ${symlinks.length} symlink(s) remain in the bundle (Windows installers can't ship them):\n` +
      symlinks.slice(0, 20).join("\n"),
  )
}
console.log(
  `[voice-runtime] turn-detector bundled: ${bundledModel.length} model + ${bundledTokenizer.length} tokenizer file(s), 0 symlinks`,
)

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
