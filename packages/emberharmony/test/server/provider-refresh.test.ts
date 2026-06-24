import { describe, expect, test, mock, beforeEach } from "bun:test"
import path from "path"
import fs from "fs/promises"
import { Instance } from "../../src/project/instance"
import { Server } from "../../src/server/server"
import { Log } from "../../src/util/log"
import { Global } from "../../src/global"

mock.module("../../src/bun/index", () => ({
  BunProc: {
    install: async (pkg: string, _version?: string) => {
      const lastAtIndex = pkg.lastIndexOf("@")
      return lastAtIndex > 0 ? pkg.substring(0, lastAtIndex) : pkg
    },
    run: async () => {
      throw new Error("BunProc.run should not be called in tests")
    },
    which: () => process.execPath,
    InstallFailedError: class extends Error {},
  },
}))

import { tmpdir } from "../fixture/fixture"
import { Provider } from "../../src/provider/provider"

Log.init({ print: false })

const ollamaURL = "http://localhost:11434"
const lmstudioURL = "http://127.0.0.1:1234"
const offlineURL = "http://127.0.0.1:65534"

const authHeader: Record<string, string> = (() => {
  const password = process.env.EMBERHARMONY_SERVER_PASSWORD
  const username = process.env.EMBERHARMONY_SERVER_USERNAME ?? "emberharmony"
  if (!password) return {} as Record<string, string>
  return { Authorization: `Basic ${btoa(`${username}:${password}`)}` }
})()

async function clearProviderCache() {
  const storageDir = path.join(Global.Path.data, "storage", "provider")
  await fs.rm(storageDir, { recursive: true, force: true }).catch(() => {})
}

async function fetchJSON(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url, { signal: AbortSignal.timeout(3000) })
  if (!res.ok) throw new Error(`${url} returned ${res.status}`)
  return (await res.json()) as Record<string, unknown>
}

async function isReachable(url: string): Promise<boolean> {
  try {
    await fetch(url, { signal: AbortSignal.timeout(2000) })
    return true
  } catch {
    return false
  }
}

const ollamaRunning = await isReachable(`${ollamaURL}/api/tags`)
const lmstudioRunning = await isReachable(`${lmstudioURL}/v1/models`)

describe("provider routes — offline (no daemon required)", () => {
  beforeEach(async () => {
    await clearProviderCache()
  })

  test("GET /provider includes ollama and lmstudio even when offline", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              ollama: { options: { baseURL: offlineURL } },
              lmstudio: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const app = Server.App()
        const response = await app.request(`/provider?directory=${encodeURIComponent(tmp.path)}`, {
          headers: authHeader,
        })
        expect(response.status).toBe(200)
        const body = (await response.json()) as {
          all: Array<{ id: string; name: string }>
          connected: string[]
        }
        const ids = body.all.map((p) => p.id)
        expect(ids).toContain("ollama")
        expect(ids).toContain("lmstudio")
        expect(body.connected).toContain("ollama")
        expect(body.connected).toContain("lmstudio")
        await Instance.dispose()
      },
    })
  })
})

describe.skipIf(!ollamaRunning)("provider routes — Ollama (requires daemon)", () => {
  beforeEach(async () => {
    await clearProviderCache()
  })

  test("POST /provider/ollama/refresh returns live models from real Ollama", async () => {
    const tags = await fetchJSON(`${ollamaURL}/api/tags`)
    const modelNames = (tags.models as Array<{ name: string }>).map((m) => m.name)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const app = Server.App()
        const response = await app.request(`/provider/ollama/refresh?directory=${encodeURIComponent(tmp.path)}`, {
          method: "POST",
          headers: authHeader,
        })
        expect(response.status).toBe(200)
        const body = (await response.json()) as { models: Record<string, unknown> }
        const keys = Object.keys(body.models)
        expect(keys.length).toBeGreaterThan(0)
        for (const name of modelNames) {
          expect(keys).toContain(name)
        }

        const providers = await Provider.list()
        expect(Object.keys(providers["ollama"].models)).toEqual(keys)
        await Instance.dispose()
      },
    })
  })

  test("POST /provider/ollama/refresh after offline shows cached models", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const initial = await Provider.list()
        const modelNames = Object.keys(initial["ollama"].models)
        expect(modelNames.length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })

    await Instance.provide({
      directory: tmp.path,
      init: async () => {
        const configPath = path.join(tmp.path, "emberharmony.json")
        await Bun.write(
          configPath,
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              ollama: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
      fn: async () => {
        const app = Server.App()
        const response = await app.request(`/provider/ollama/refresh?directory=${encodeURIComponent(tmp.path)}`, {
          method: "POST",
          headers: authHeader,
        })
        expect(response.status).toBe(200)
        const body = (await response.json()) as { models: Record<string, unknown> }
        expect(Object.keys(body.models).length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })
  })
})

describe.skipIf(!lmstudioRunning)("provider routes — LM Studio (requires daemon)", () => {
  beforeEach(async () => {
    await clearProviderCache()
  })

  test("POST /provider/lmstudio/refresh returns live models from real LM Studio", async () => {
    const res = await fetchJSON(`${lmstudioURL}/v1/models`)
    const modelIds = (res.data as Array<{ id: string }>).map((m) => m.id)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const app = Server.App()
        const response = await app.request(`/provider/lmstudio/refresh?directory=${encodeURIComponent(tmp.path)}`, {
          method: "POST",
          headers: authHeader,
        })
        expect(response.status).toBe(200)
        const body = (await response.json()) as { models: Record<string, unknown> }
        const keys = Object.keys(body.models)
        expect(keys.length).toBeGreaterThan(0)
        for (const id of modelIds) {
          expect(keys).toContain(id)
        }

        const providers = await Provider.list()
        expect(Object.keys(providers["lmstudio"].models)).toEqual(keys)
        await Instance.dispose()
      },
    })
  })
})
