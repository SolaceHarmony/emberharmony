import { describe, expect, test, mock, beforeEach, afterEach } from "bun:test"
import path from "node:path"
import fs from "node:fs/promises"
import os from "node:os"

mock.module("../../src/bun/index", () => ({
  BunProc: {
    install: async () => "",
    run: async () => ({ exitCode: 0 }),
    which: () => process.execPath,
    InstallFailedError: class extends Error {},
  },
}))

mock.module("../../src/voice/token", () => ({
  Voice: {
    AGENT_NAME: "emberharmony-voice",
    AUTH_PROVIDER_ID: "livekit",
    NotConfiguredError: class extends Error {},
    settings: async () => ({
      disabled: false,
      url: "wss://fake.livekit.cloud",
      apiKey: "fake-key",
      apiSecret: "fake-secret",
      stt: "deepgram/nova-3",
      tts: "cartesia/sonic-3",
      intent: "gpt-4o-mini",
      available: true,
    }),
  },
}))

import { VoiceWorker } from "../../src/voice/worker"
import { Instance } from "../../src/project/instance"
import { tmpdir } from "../fixture/fixture"

const FAKE_RUNTIME_DIR = path.join(os.tmpdir(), "emberharmony-voice-test-runtime")

async function ensureFakeRuntime() {
  await fs.mkdir(FAKE_RUNTIME_DIR, { recursive: true })
  const agentSrc = path.join(import.meta.dir, "../fixture/voice/fake-agent.js")
  const agentDest = path.join(FAKE_RUNTIME_DIR, "agent.js")
  await fs.copyFile(agentSrc, agentDest)
}

async function cleanupSocket() {
  const sockets = await Array.fromAsync(
    new Bun.Glob("emberharmony-voice-*.sock").scan({ cwd: os.tmpdir(), absolute: true }),
  )
  for (const s of sockets) await fs.unlink(s).catch(() => {})
}

describe("VoiceWorker lifecycle (IPC shutdown, no LiveKit)", () => {
  beforeEach(async () => {
    await ensureFakeRuntime()
    await cleanupSocket()
  })

  afterEach(async () => {
    await VoiceWorker.stop()
    await cleanupSocket()
  })

  test("start spawns the worker process", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        const ok = await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)
        expect(ok).toBe(true)
        expect(VoiceWorker.running()).toBe(true)
        await VoiceWorker.stop()
        await Instance.dispose()
      },
    })
  })

  test("start is idempotent — duplicate calls don't spawn twice", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        const ok1 = await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)
        expect(ok1).toBe(true)

        const ok2 = await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)
        expect(ok2).toBe(true)
        expect(VoiceWorker.running()).toBe(true)

        await VoiceWorker.stop()
        expect(VoiceWorker.running()).toBe(false)
        await Instance.dispose()
      },
    })
  })

  test("stop via IPC gracefully shuts down the worker", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)
        expect(VoiceWorker.running()).toBe(true)

        const start = Date.now()
        await VoiceWorker.stop()
        const elapsed = Date.now() - start

        expect(VoiceWorker.running()).toBe(false)
        expect(elapsed).toBeLessThan(5000)
        await Instance.dispose()
      },
    })
  })

  test("stop when not running is a no-op", async () => {
    await VoiceWorker.stop()
    expect(VoiceWorker.running()).toBe(false)
  })

  test("restart stops the old worker and starts a new one", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)
        expect(VoiceWorker.running()).toBe(true)

        const ok = await VoiceWorker.restart({
          disabled: false,
        } as any)
        expect(ok).toBe(true)
        expect(VoiceWorker.running()).toBe(true)

        await VoiceWorker.stop()
        await Instance.dispose()
      },
    })
  })

  test("no zombie processes after stop", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        await VoiceWorker.start("http://localhost:0", {
          disabled: false,
        } as any)

        await VoiceWorker.stop()

        await new Promise((r) => setTimeout(r, 500))

        const zombies = getChildPids().filter((p) => p.state === "Z")
        expect(zombies.length).toBe(0)
        await Instance.dispose()
      },
    })
  })
})

function getChildPids() {
  try {
    const output = Bun.spawnSync(["ps", "-eo", "pid,ppid,state,comm"]).stdout.toString()
    return output
      .split("\n")
      .slice(1)
      .filter(Boolean)
      .map((line) => {
        const [pid, ppid, state, ...commParts] = line.trim().split(/\s+/)
        return { pid: Number(pid), ppid: Number(ppid), state, comm: commParts.join(" ") }
      })
      .filter((p) => p.comm.includes("bun") || p.comm.includes("fake-agent") || p.comm.includes("agent.js"))
  } catch {
    return []
  }
}
