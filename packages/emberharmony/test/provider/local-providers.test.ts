import { describe, expect, test, mock, beforeEach } from "bun:test"
import path from "path"
import fs from "fs/promises"
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
import { Instance } from "../../src/project/instance"
import { Provider } from "../../src/provider/provider"

const ollamaURL = "http://localhost:11434"
const lmstudioURL = "http://127.0.0.1:1234"
const offlineURL = "http://127.0.0.1:65534"

async function clearProviderCache() {
  const storageDir = path.join(Global.Path.data, "storage", "provider")
  await fs.rm(storageDir, { recursive: true, force: true }).catch(() => {})
}

async function fetchJSON(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url, { signal: AbortSignal.timeout(3000) })
  if (!res.ok) throw new Error(`${url} returned ${res.status}`)
  return (await res.json()) as Record<string, unknown>
}

describe("local provider detection (Ollama + LM Studio)", () => {
  beforeEach(async () => {
    await clearProviderCache()
  })

  test("ollama provider always present even when offline with no cache", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              ollama: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["ollama"]).toBeDefined()
        expect(Object.keys(providers["ollama"].models)).toHaveLength(0)
        await Instance.dispose()
      },
    })
  })

  test("ollama models discovered via /api/tags", async () => {
    const tags = await fetchJSON(`${ollamaURL}/api/tags`)
    const modelNames = (tags.models as Array<{ name: string }>).map((m) => m.name)
    expect(modelNames.length).toBeGreaterThan(0)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["ollama"]).toBeDefined()
        const models = Object.keys(providers["ollama"].models)
        expect(models.length).toBeGreaterThan(0)
        for (const name of modelNames) {
          expect(models).toContain(name)
        }
        const first = providers["ollama"].models[modelNames[0]]
        expect(first.providerID).toBe("ollama")
        expect(first.capabilities.toolcall).toBe(true)
        expect(first.cost.input).toBe(0)
        expect(first.api.url).toContain("/v1")
        await Instance.dispose()
      },
    })
  })

  test("ollama models persist across sessions (cache fallback when offline)", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        const modelNames = Object.keys(providers["ollama"].models)
        expect(modelNames.length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })

    await using tmp2 = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              ollama: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp2.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["ollama"]).toBeDefined()
        const models = Object.keys(providers["ollama"].models)
        expect(models.length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })
  })

  test("Provider.refresh re-probes and updates ollama model list", async () => {
    const tags = await fetchJSON(`${ollamaURL}/api/tags`)
    const modelNames = (tags.models as Array<{ name: string }>).map((m) => m.name)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const initial = await Provider.list()
        const initialModels = Object.keys(initial["ollama"].models)
        expect(initialModels.length).toBeGreaterThan(0)

        const refreshed = await Provider.refresh("ollama")
        expect(Object.keys(refreshed).length).toBeGreaterThan(0)
        const refreshedNames = Object.keys(refreshed)
        for (const name of modelNames) {
          expect(refreshedNames).toContain(name)
        }

        const after = await Provider.list()
        const afterModels = Object.keys(after["ollama"].models)
        expect(afterModels).toEqual(Object.keys(refreshed))
        await Instance.dispose()
      },
    })
  })

  test("lmstudio provider always present even when offline with no cache", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              lmstudio: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["lmstudio"]).toBeDefined()
        expect(Object.keys(providers["lmstudio"].models)).toHaveLength(0)
        await Instance.dispose()
      },
    })
  })

  test("lmstudio models discovered via /v1/models", async () => {
    const res = await fetchJSON(`${lmstudioURL}/v1/models`)
    const modelIds = (res.data as Array<{ id: string }>).map((m) => m.id)
    expect(modelIds.length).toBeGreaterThan(0)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["lmstudio"]).toBeDefined()
        const models = Object.keys(providers["lmstudio"].models)
        expect(models.length).toBeGreaterThan(0)
        for (const id of modelIds) {
          expect(models).toContain(id)
        }
        const first = providers["lmstudio"].models[modelIds[0]]
        expect(first.providerID).toBe("lmstudio")
        expect(first.api.url).toContain("/v1")
        await Instance.dispose()
      },
    })
  })

  test("lmstudio models persist across sessions (cache fallback when offline)", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        const modelNames = Object.keys(providers["lmstudio"].models)
        expect(modelNames.length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })

    await using tmp2 = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              lmstudio: { options: { baseURL: offlineURL } },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp2.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["lmstudio"]).toBeDefined()
        const models = Object.keys(providers["lmstudio"].models)
        expect(models.length).toBeGreaterThan(0)
        await Instance.dispose()
      },
    })
  })

  test("Provider.refresh re-probes and updates lmstudio model list", async () => {
    const res = await fetchJSON(`${lmstudioURL}/v1/models`)
    const modelIds = (res.data as Array<{ id: string }>).map((m) => m.id)

    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const initial = await Provider.list()
        const initialModels = Object.keys(initial["lmstudio"].models)
        expect(initialModels.length).toBeGreaterThan(0)

        const refreshed = await Provider.refresh("lmstudio")
        expect(Object.keys(refreshed).length).toBeGreaterThan(0)
        const refreshedIds = Object.keys(refreshed)
        for (const id of modelIds) {
          expect(refreshedIds).toContain(id)
        }

        const after = await Provider.list()
        const afterModels = Object.keys(after["lmstudio"].models)
        expect(afterModels).toEqual(Object.keys(refreshed))
        await Instance.dispose()
      },
    })
  })

  test("disabled_providers excludes local providers", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            disabled_providers: ["ollama", "lmstudio"],
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["ollama"]).toBeUndefined()
        expect(providers["lmstudio"]).toBeUndefined()
        await Instance.dispose()
      },
    })
  })

  test("ollama baseURL configurable via emberharmony.json", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "emberharmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              ollama: {
                options: {
                  baseURL: ollamaURL,
                },
              },
            },
          }),
        )
      },
    })
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const providers = await Provider.list()
        expect(providers["ollama"]).toBeDefined()
        const models = Object.keys(providers["ollama"].models)
        expect(models.length).toBeGreaterThan(0)
        const firstModel = providers["ollama"].models[models[0]]
        expect(firstModel.api.url).toContain(ollamaURL)
        await Instance.dispose()
      },
    })
  })
})
