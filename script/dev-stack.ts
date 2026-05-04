#!/usr/bin/env bun
/**
 * Full dev stack: launches the emberharmony backend server on :4096 and the
 * Vite web UI on :3000 together, wires them with VITE_EMBERHARMONY_SERVER_* env
 * vars, and tears both down on Ctrl-C so Claude Preview / manual testing works
 * out of the box.
 *
 * Usage:
 *   bun run script/dev-stack.ts
 *   (or `bun run dev:stack` from the repo root)
 */

import { spawn } from "bun"
import path from "path"
import { fileURLToPath } from "url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, "..")

const BACKEND_PORT = Number(process.env.EMBERHARMONY_BACKEND_PORT ?? 4096)
const UI_PORT = Number(process.env.EMBERHARMONY_UI_PORT ?? 3000)

// Print the ember splash before booting services
const { UI } = await import(path.join(repoRoot, "packages/emberharmony/src/cli/ui.ts"))
console.log("")
console.log(UI.logo("  "))
console.log("")

console.log(`[dev-stack] starting emberharmony backend on :${BACKEND_PORT}`)
const backend = spawn({
  cmd: [
    "bun",
    "run",
    "--conditions=browser",
    "src/index.ts",
    "serve",
    "--hostname",
    "127.0.0.1",
    "--port",
    String(BACKEND_PORT),
  ],
  cwd: path.join(repoRoot, "packages/emberharmony"),
  stdio: ["ignore", "inherit", "inherit"],
  env: { ...process.env },
})

console.log(`[dev-stack] starting Vite UI on :${UI_PORT}`)
const ui = spawn({
  cmd: ["bun", "run", "dev", "--", "--host", "0.0.0.0", "--port", String(UI_PORT), "--strictPort"],
  cwd: path.join(repoRoot, "packages/app"),
  stdio: ["ignore", "inherit", "inherit"],
  env: {
    ...process.env,
    VITE_EMBERHARMONY_SERVER_HOST: "127.0.0.1",
    VITE_EMBERHARMONY_SERVER_PORT: String(BACKEND_PORT),
  },
})

let stopping = false
async function stop(code = 0) {
  if (stopping) return
  stopping = true
  console.log("\n[dev-stack] shutting down…")
  try {
    backend.kill()
  } catch {}
  try {
    ui.kill()
  } catch {}
  await Promise.allSettled([backend.exited, ui.exited])
  process.exit(code)
}

process.on("SIGINT", () => void stop(0))
process.on("SIGTERM", () => void stop(0))

// If either process exits on its own, bring down the stack.
void backend.exited.then((code) => {
  console.error(`[dev-stack] backend exited with code ${code}`)
  void stop(code ?? 1)
})
void ui.exited.then((code) => {
  console.error(`[dev-stack] UI exited with code ${code}`)
  void stop(code ?? 1)
})
