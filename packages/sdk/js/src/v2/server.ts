import { spawn } from "node:child_process"
import { type Config } from "./gen/types.gen.js"

export type ServerOptions = {
  hostname?: string
  port?: number
  signal?: AbortSignal
  timeout?: number
  config?: Config
}

export type TuiOptions = {
  project?: string
  model?: string
  session?: string
  agent?: string
  signal?: AbortSignal
  config?: Config
}

export async function createEmberHarmonyServer(options?: ServerOptions) {
  options = Object.assign(
    {
      hostname: "127.0.0.1",
      port: 4096,
      timeout: 30_000,
    },
    options ?? {},
  )

  const bin = process.env.EMBERHARMONY_BIN ?? "emberharmony"
  const args = [`serve`, `--hostname=${options.hostname}`, `--port=${options.port}`]
  if (options.config?.logLevel) args.push(`--log-level=${options.config.logLevel}`)

  const proc = spawn(bin, args, {
    signal: options.signal,
    env: {
      ...process.env,
      EMBERHARMONY_CONFIG_CONTENT: JSON.stringify(options.config ?? {}),
    },
  })

  const url = await new Promise<string>((resolve, reject) => {
    const out = { value: "" }
    const id = setTimeout(() => {
      proc.kill()
      const text = out.value.trim()
      const tail = text.length > 4000 ? text.slice(-4000) : text
      const info = tail ? `\nServer output:\n${tail}` : ""
      reject(new Error(`Timeout waiting for server to start after ${options.timeout}ms${info}`))
    }, options.timeout)
    const scan = (chunk: Buffer) => {
      out.value += chunk.toString()
      const lines = out.value.split("\n")
      for (const line of lines) {
        const low = line.toLowerCase()
        if (!low.includes("listening")) continue
        if (!low.includes("server")) continue
        const match = line.match(/https?:\/\/[^\s]+/)
        if (!match) continue
        clearTimeout(id)
        resolve(match[0]!)
        return
      }
    }
    proc.stdout?.on("data", (chunk) => {
      scan(chunk)
    })
    proc.stderr?.on("data", (chunk) => {
      scan(chunk)
    })
    proc.on("exit", (code) => {
      clearTimeout(id)
      let msg = `Server exited with code ${code}`
      if (out.value.trim()) {
        msg += `\nServer output: ${out.value}`
      }
      reject(new Error(msg))
    })
    proc.on("error", (error) => {
      clearTimeout(id)
      reject(error)
    })
    if (options.signal) {
      options.signal.addEventListener("abort", () => {
        clearTimeout(id)
        reject(new Error("Aborted"))
      })
    }
  })

  return {
    url,
    close() {
      proc.kill()
    },
  }
}

export function createEmberHarmonyTui(options?: TuiOptions) {
  const bin = process.env.EMBERHARMONY_BIN ?? "emberharmony"
  const args = []

  if (options?.project) {
    args.push(`--project=${options.project}`)
  }
  if (options?.model) {
    args.push(`--model=${options.model}`)
  }
  if (options?.session) {
    args.push(`--session=${options.session}`)
  }
  if (options?.agent) {
    args.push(`--agent=${options.agent}`)
  }

  const proc = spawn(bin, args, {
    signal: options?.signal,
    stdio: "inherit",
    env: {
      ...process.env,
      EMBERHARMONY_CONFIG_CONTENT: JSON.stringify(options?.config ?? {}),
    },
  })

  return {
    close() {
      proc.kill()
    },
  }
}
