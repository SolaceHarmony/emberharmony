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

import { existsSync } from "node:fs"
import { VoiceWorker } from "../../src/voice/worker"
import { Instance } from "../../src/project/instance"
import { Global } from "../../src/global"
import { tmpdir } from "../fixture/fixture"

const FAKE_RUNTIME_DIR = path.join(os.tmpdir(), "emberharmony-voice-test-runtime")
// the worker keys its pidfile by the owning (serve) pid — here that is the test process
const PIDFILE = path.join(Global.Path.state, `voice-worker.${process.pid}.json`)

async function cleanupPidfiles() {
  const files = await fs.readdir(Global.Path.state).catch(() => [] as string[])
  for (const f of files) {
    if (/^voice-worker\.\d+\.json$/.test(f)) await fs.unlink(path.join(Global.Path.state, f)).catch(() => {})
  }
}

async function ensureFakeRuntime() {
  await fs.mkdir(FAKE_RUNTIME_DIR, { recursive: true })
  const agentSrc = path.join(import.meta.dir, "../fixture/voice/fake-agent.js")
  const agentDest = path.join(FAKE_RUNTIME_DIR, "agent.js")
  await fs.copyFile(agentSrc, agentDest)
  // resolveLaunch() only selects bundled mode when BOTH agent.js and the bun
  // binary exist in the runtime dir. Without a bun here the worker falls back
  // to spawning the real src/voice/agent.ts — i.e. a real LiveKit worker —
  // making these lifecycle tests hang or fail. Stage the current runtime as bun.
  const bunDest = path.join(FAKE_RUNTIME_DIR, process.platform === "win32" ? "bun.exe" : "bun")
  await fs.copyFile(process.execPath, bunDest)
  await fs.chmod(bunDest, 0o755)
}

async function listSocketNames() {
  // readdir, not Bun.Glob: glob.scan() skips non-regular files, so it never
  // matches (or cleans up) the Unix-domain socket files the worker creates.
  return (await fs.readdir(os.tmpdir()).catch(() => [])).filter((n) => /^emberharmony-voice-.*\.sock$/.test(n))
}

async function cleanupSocket() {
  for (const name of await listSocketNames()) await fs.unlink(path.join(os.tmpdir(), name)).catch(() => {})
}

describe("VoiceWorker lifecycle (IPC shutdown, no LiveKit)", () => {
  beforeEach(async () => {
    await ensureFakeRuntime()
    await cleanupSocket()
  })

  afterEach(async () => {
    await VoiceWorker.stop()
    await cleanupSocket()
    await cleanupPidfiles()
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

  test("concurrent starts spawn only one worker (no leak)", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR

        // count sockets created by THIS test (tmpdir may hold unrelated ones).
        // Each spawned worker creates its own IPC socket; exactly one new socket
        // means a single live worker.
        const before = new Set(await listSocketNames())
        const created = async () => (await listSocketNames()).filter((n) => !before.has(n))

        // fire several starts in the same tick: the old code passed the
        // running() guard before `proc` was assigned and Bun.spawn'd one worker
        // per call, leaking every spawn but the last (its IPC socket survives a
        // tracked stop()). Serialized lifecycle ops must spawn exactly one.
        const results = await Promise.all(
          Array.from({ length: 4 }, () => VoiceWorker.start("http://localhost:0", { disabled: false } as any)),
        )
        expect(results.every((r) => r === true)).toBe(true)
        expect(VoiceWorker.running()).toBe(true)

        // the spawned worker's IPC socket appears once its bun process boots
        // (cold start ~1s); poll for it, then settle so any extra (regression)
        // workers would have surfaced their sockets too.
        const deadline = Date.now() + 8000
        while (Date.now() < deadline && (await created()).length < 1) {
          await new Promise((r) => setTimeout(r, 50))
        }
        await new Promise((r) => setTimeout(r, 800))
        expect((await created()).length).toBe(1)

        // a tracked stop() only shuts down the one worker it knows about; a
        // leaked worker's socket would still be here afterwards
        await VoiceWorker.stop()
        expect(VoiceWorker.running()).toBe(false)
        expect((await created()).length).toBe(0)
        await Instance.dispose()
      },
    })
  })

  test("writes a pidfile on start and clears it + the socket on stop", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        await VoiceWorker.start("http://localhost:0", { disabled: false } as any)
        expect(existsSync(PIDFILE)).toBe(true)
        const rec = JSON.parse(await fs.readFile(PIDFILE, "utf8"))
        expect(typeof rec.pid).toBe("number")
        expect(rec.pid).toBeGreaterThan(0)
        // socket appears once the worker boots
        for (let i = 0; i < 40 && !existsSync(rec.socket); i++) await new Promise((r) => setTimeout(r, 50))
        expect(existsSync(rec.socket)).toBe(true)

        await VoiceWorker.stop()
        expect(existsSync(PIDFILE)).toBe(false)
        expect(existsSync(rec.socket)).toBe(false)
        await Instance.dispose()
      },
    })
  })

  test("reaps a recorded orphan worker on next start", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        // a long-lived process whose argv looks like a voice worker (filename
        // ends in agent.js) so looksLikeVoiceWorker() will verify + reap it
        const orphanScript = path.join(os.tmpdir(), `emberharmony-orphan-${process.pid}-agent.js`)
        await fs.writeFile(orphanScript, "setInterval(() => {}, 1000)")
        const orphan = Bun.spawn({ cmd: [process.execPath, orphanScript], stdout: "ignore", stderr: "ignore" })
        await new Promise((r) => setTimeout(r, 150))

        // record it as a worker owned by a DEAD serve (ppid 999999) so it counts
        // as a reapable orphan; keyed by that dead serve pid, not ours
        await fs.mkdir(Global.Path.state, { recursive: true })
        const orphanSocket = path.join(os.tmpdir(), `emberharmony-voice-orphan-${process.pid}.sock`)
        const orphanPidfile = path.join(Global.Path.state, "voice-worker.999999.json")
        await fs.writeFile(orphanPidfile, JSON.stringify({ pid: orphan.pid, socket: orphanSocket, ppid: 999_999 }))

        // starting a fresh worker must reap the recorded orphan first
        await VoiceWorker.start("http://localhost:0", { disabled: false } as any)
        for (let i = 0; i < 40; i++) {
          let alive = true
          try {
            process.kill(orphan.pid, 0)
          } catch {
            alive = false
          }
          if (!alive) break
          await new Promise((r) => setTimeout(r, 50))
        }
        let stillAlive = true
        try {
          process.kill(orphan.pid, 0)
        } catch {
          stillAlive = false
        }
        expect(stillAlive).toBe(false)

        await VoiceWorker.stop()
        await fs.unlink(orphanScript).catch(() => {})
        await Instance.dispose()
      },
    })
  })

  test("does NOT reap a worker whose owning serve is still alive (no fratricide)", async () => {
    await using tmp = await tmpdir({})
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"] = FAKE_RUNTIME_DIR
        // a live "owner serve" and its "worker" (argv looks like a voice worker)
        const ownerScript = path.join(os.tmpdir(), `eh-owner-${process.pid}.js`)
        const workerScript = path.join(os.tmpdir(), `eh-peer-${process.pid}-agent.js`)
        await fs.writeFile(ownerScript, "setInterval(() => {}, 1000)")
        await fs.writeFile(workerScript, "setInterval(() => {}, 1000)")
        const owner = Bun.spawn({ cmd: [process.execPath, ownerScript], stdout: "ignore", stderr: "ignore" })
        const worker = Bun.spawn({ cmd: [process.execPath, workerScript], stdout: "ignore", stderr: "ignore" })
        await new Promise((r) => setTimeout(r, 150))

        // record the worker as owned by the LIVE owner serve
        await fs.mkdir(Global.Path.state, { recursive: true })
        const peerPidfile = path.join(Global.Path.state, `voice-worker.${owner.pid}.json`)
        await fs.writeFile(peerPidfile, JSON.stringify({ pid: worker.pid, socket: "/tmp/nope.sock", ppid: owner.pid }))

        await VoiceWorker.start("http://localhost:0", { disabled: false } as any)
        await new Promise((r) => setTimeout(r, 300))

        // a live owner means the worker is NOT an orphan — it must survive, and
        // the peer's pidfile must be left intact
        let workerAlive = true
        try {
          process.kill(worker.pid, 0)
        } catch {
          workerAlive = false
        }
        expect(workerAlive).toBe(true)
        expect(existsSync(peerPidfile)).toBe(true)

        await VoiceWorker.stop()
        owner.kill()
        worker.kill()
        await fs.unlink(ownerScript).catch(() => {})
        await fs.unlink(workerScript).catch(() => {})
        await fs.unlink(peerPidfile).catch(() => {})
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
