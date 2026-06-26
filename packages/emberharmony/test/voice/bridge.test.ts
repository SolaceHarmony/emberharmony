import { describe, expect, test, mock, afterEach } from "bun:test"

// The session bridge (SessionLLM/SessionLLMStream) is the most regression-prone
// voice component — three shipped voice bugs lived here — yet had zero tests.
// We exercise the REAL bridge.ts unmodified by (a) shimming @livekit/agents with
// a thin base that reimplements the queue->async-iterator contract the bridge
// relies on, and (b) stubbing globalThis.fetch to route /event (a fake SSE
// stream), /prompt_async, /message, and /abort. No LiveKit runtime is loaded.

// Shared @livekit/agents shim — see _livekit-agents.shim.ts. mock.module() is
// process-global, so every voice test must register the SAME superset shim; a
// per-file shim that omits an export another file needs (bridge omitted `log`)
// clobbers it when the suites run together.
import { agentsShim, ChatContextShim } from "./_livekit-agents.shim"

mock.module("@livekit/agents", () => agentsShim)

// `import type` is erased at runtime, so it does NOT trigger loading the real
// @livekit/agents. The value import must be a dynamic import AFTER mock.module
// (static imports are hoisted above it, which would load the real module first).
import type { SessionBridgeOptions, SessionLLM as SessionLLMType } from "../../src/voice/bridge"

const { SessionLLM } = await import("../../src/voice/bridge")

// ---- SSE + fetch fixtures ----------------------------------------------------

const SID = "ses_test"
const enc = new TextEncoder()
const REAL_NOW = Date.now
const REAL_FETCH = globalThis.fetch

const sse = (obj: any) => `data: ${JSON.stringify(obj)}\n\n`
const connected = () => ({ type: "server.connected" })
const heartbeat = () => ({ type: "server.heartbeat" })
const idle = (sessionID = SID) => ({ type: "session.idle", properties: { sessionID } })
const part = (messageID: string, delta: string, sessionID = SID) => ({
  type: "message.part.updated",
  properties: { part: { sessionID, type: "text", messageID }, delta },
})
const msgUpdated = (sessionID = SID) => ({
  type: "message.updated",
  properties: { info: { sessionID, role: "assistant", time: { completed: 1 } } },
})
const sessionError = (error: any, sessionID = SID) => ({ type: "session.error", properties: { sessionID, error } })

/** Enqueue each frame as one read, then close. Raw strings pass through verbatim. */
function staticBody(frames: Array<any | string>) {
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const f of frames) c.enqueue(enc.encode(typeof f === "string" ? f : sse(f)))
      c.close()
    },
  })
}

/** Enqueue frames but never close; error the stream on abort (models a stalled feed). */
function stallBody(frames: Array<any | string>, signal?: AbortSignal) {
  return new ReadableStream<Uint8Array>({
    start(c) {
      for (const f of frames) c.enqueue(enc.encode(typeof f === "string" ? f : sse(f)))
      signal?.addEventListener("abort", () => {
        try {
          c.error(Object.assign(new Error("aborted"), { name: "AbortError" }))
        } catch {}
      })
    },
  })
}

/** Manually-driven stream so a test can advance a mocked clock between frames. */
function manualBody() {
  let ctrl!: ReadableStreamDefaultController<Uint8Array>
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      ctrl = c
    },
  })
  return {
    stream,
    push: (f: any | string) => ctrl.enqueue(enc.encode(typeof f === "string" ? f : sse(f))),
    close: () => ctrl.close(),
  }
}

interface FetchHandlers {
  event?: (init: any) => Response
  message?: (init: any) => Response
  prompt?: (init: any) => Response
}

function installFetch(h: FetchHandlers = {}) {
  const calls = { event: 0, message: 0, abort: 0, prompt: [] as any[] }
  globalThis.fetch = (async (url: any, init: any = {}) => {
    const u = String(url)
    if (u.endsWith("/event")) {
      calls.event++
      return (h.event ?? (() => new Response(staticBody([connected(), idle()]))))(init)
    }
    if (u.includes("/prompt_async")) {
      calls.prompt.push(init.body ? JSON.parse(init.body) : null)
      return (h.prompt ?? (() => new Response("", { status: 200 })))(init)
    }
    if (u.includes("/abort")) {
      calls.abort++
      return new Response("", { status: 200 })
    }
    if (u.includes("/message")) {
      calls.message++
      return (h.message ?? (() => new Response("[]", { status: 200 })))(init)
    }
    throw new Error("unexpected fetch: " + u)
  }) as typeof fetch
  return calls
}

const tick = () => new Promise((r) => setTimeout(r, 0))
async function ticks(n = 3) {
  for (let i = 0; i < n; i++) await tick()
}

function bridge(extra: Partial<SessionBridgeOptions> = {}) {
  return new SessionLLM({ serverUrl: "http://x", directory: "/d", sessionID: SID, ...extra })
}
function chatOf(llmInst: SessionLLMType, userText: string | null = "hello") {
  const ctx = ChatContextShim.empty()
  if (userText !== null) ctx.addMessage({ role: "user", content: userText })
  return llmInst.chat({ chatCtx: ctx as any })
}
async function drain(stream: AsyncIterable<any>) {
  const chunks: any[] = []
  for await (const c of stream) chunks.push(c)
  return chunks
}
const textOf = (chunks: any[]) => chunks.map((c) => c.delta.content).join("")
const idsOf = (chunks: any[]) => [...new Set(chunks.map((c) => c.id))]

afterEach(() => {
  // Restore BOTH globals we stub — bun runs test files in one process, so a
  // leaked globalThis.fetch / Date.now pollutes unrelated later suites.
  Date.now = REAL_NOW
  globalThis.fetch = REAL_FETCH
})

// ---- tests -------------------------------------------------------------------

describe("SessionLLMStream (voice session bridge)", () => {
  test("THE SHIPPED REGRESSION: tool-call step finalizes mid-turn, continuation still streams, one replyId, no /abort", async () => {
    const calls = installFetch({
      event: () =>
        new Response(
          staticBody([
            connected(),
            part("m1", "Let me check. "),
            msgUpdated(), // tool-call step finalizes its assistant message (time.completed) — must NOT end the turn
            part("m2", "The answer is 42."),
            idle(),
          ]),
        ),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("Let me check. The answer is 42.") // continuation streamed, not cut at m1 completion
    expect(idsOf(chunks)).toEqual(["m1"]) // coalesced under the first message's id
    expect(chunks.length).toBe(2)
    expect(calls.abort).toBe(0) // a natural turn end must NOT abort the server-side generation
  })

  test("multi-message turn coalesces all assistant messages under one stable replyId", async () => {
    installFetch({
      event: () =>
        new Response(staticBody([connected(), part("m1", "a"), part("m2", "b"), part("m3", "c"), idle()])),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("abc")
    expect(idsOf(chunks)).toEqual(["m1"]) // size 1, equals the FIRST text part's id
  })

  test("genuine user interrupt aborts the stream and fires exactly one POST /abort", async () => {
    const calls = installFetch({
      event: (init) => new Response(stallBody([connected(), part("m1", "speaking…")], init.signal)),
    })
    const stream = chatOf(bridge())
    const chunks: any[] = []
    const p = (async () => {
      for await (const c of stream) chunks.push(c)
    })()
    await ticks() // let the first delta stream, then the feed stalls (no idle)
    ;(stream as any).close() // voice barge-in
    await p
    await tick()
    expect(calls.abort).toBe(1)
    expect(chunks.length).toBe(1)
  })

  test("ignores parts/idle for a different sessionID", async () => {
    installFetch({
      event: () =>
        new Response(
          staticBody([
            connected(),
            part("x1", "FOREIGN", "other-session"),
            idle("other-session"), // must NOT end our turn
            part("m1", "ours"),
            idle(),
          ]),
        ),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("ours")
    expect(idsOf(chunks)).toEqual(["m1"])
  })

  test("scoped session.error throws; a foreign-session error is ignored", async () => {
    installFetch({
      event: () =>
        new Response(
          staticBody([
            connected(),
            sessionError({ message: "not ours" }, "other-session"), // ignored
            part("m1", "x"),
            sessionError({ message: "boom" }), // ours — throws
          ]),
        ),
    })
    await expect(drain(chatOf(bridge()))).rejects.toThrow("session error")
  })

  test("empty / no-user chatCtx is a no-op: nothing is fetched", async () => {
    const calls = installFetch()
    const chunks = await drain(chatOf(bridge(), null))
    expect(chunks).toEqual([])
    expect(calls.event).toBe(0)
    expect(calls.prompt.length).toBe(0)
    expect(calls.message).toBe(0)
  })

  test("event-stream connect failure throws a descriptive error and never posts the prompt", async () => {
    const calls = installFetch({ event: () => new Response("nope", { status: 500 }) })
    await expect(drain(chatOf(bridge()))).rejects.toThrow("event stream failed: 500")
    expect(calls.prompt.length).toBe(0)
  })

  test("non-OK prompt_async response throws", async () => {
    installFetch({
      event: () => new Response(staticBody([connected(), idle()])),
      prompt: () => new Response("busy", { status: 409 }),
    })
    await expect(drain(chatOf(bridge()))).rejects.toThrow("session prompt failed: 409")
  })

  test("prompt body carries resolved model from history, plus agent + system", async () => {
    const calls = installFetch({
      event: () => new Response(staticBody([connected(), part("m1", "ok"), idle()])),
      message: () =>
        new Response(
          JSON.stringify([
            { info: { role: "user" } },
            { info: { role: "assistant", providerID: "anthropic", modelID: "claude" } },
          ]),
          { status: 200 },
        ),
    })
    await drain(chatOf(bridge({ agent: () => "build", system: "be terse" })))
    const body = calls.prompt[0]
    expect(body.parts).toEqual([{ type: "text", text: "hello" }])
    expect(body.model).toEqual({ providerID: "anthropic", modelID: "claude" })
    expect(body.agent).toBe("build")
    expect(body.system).toBe("be terse")
  })

  test("falls back to fallbackModel when there is no assistant history", async () => {
    const calls = installFetch({
      event: () => new Response(staticBody([connected(), idle()])),
      message: () => new Response("[]", { status: 200 }),
    })
    await drain(chatOf(bridge({ fallbackModel: { providerID: "openai", modelID: "gpt" } })))
    expect(calls.prompt[0].model).toEqual({ providerID: "openai", modelID: "gpt" })
  })

  test("does NOT include agent/system/model when absent", async () => {
    const calls = installFetch({
      event: () => new Response(staticBody([connected(), idle()])),
    })
    await drain(chatOf(bridge()))
    const body = calls.prompt[0]
    expect("agent" in body).toBe(false)
    expect("system" in body).toBe(false)
    expect("model" in body).toBe(false)
  })

  test("SSE frame split across two reads is buffered and parsed exactly once", async () => {
    const full = sse(part("m1", "joined"))
    const cut = Math.floor(full.length / 2)
    installFetch({
      event: () => new Response(staticBody([sse(connected()).slice(0), full.slice(0, cut), full.slice(cut) + sse(idle())])),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("joined")
  })

  test("comment lines, empty data, and unknown event types are ignored without ending the turn", async () => {
    installFetch({
      event: () =>
        new Response(
          staticBody([
            connected(),
            ": keep-alive\n\n", // comment frame, no data:
            "data:\n\n", // empty data
            { type: "something.unknown", properties: { foo: 1 } }, // unknown type — no-op
            part("m1", "real"),
            idle(),
          ]),
        ),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("real")
  })

  test("a malformed SSE frame is skipped and the turn keeps streaming", async () => {
    installFetch({
      event: () =>
        new Response(
          staticBody([
            connected(),
            "data: {this is not json}\n\n", // garbage frame — must not throw out of the generator
            part("m1", "survived"),
            idle(),
          ]),
        ),
    })
    const chunks = await drain(chatOf(bridge()))
    expect(textOf(chunks)).toBe("survived")
  })

  test("heartbeats bump the activity clock and prevent a false staleness timeout", async () => {
    let clock = 1_000_000
    Date.now = () => clock
    const m = manualBody()
    installFetch({ event: () => new Response(m.stream) })
    const stream = chatOf(bridge())
    const chunks: any[] = []
    const p = (async () => {
      for await (const c of stream) chunks.push(c)
    })()
    m.push(connected())
    await ticks(4) // let priming (#sessionModel + prompt_async) settle and the loop park
    m.push(part("m1", "a"))
    await ticks()
    clock += 90_000
    m.push(heartbeat())
    await ticks()
    clock += 90_000
    m.push(heartbeat())
    await ticks()
    clock += 90_000
    m.push(part("m1", "b"))
    await ticks()
    m.push(idle())
    m.close()
    await p
    expect(textOf(chunks)).toBe("ab") // never tripped the 120s check despite 270s elapsed
  })

  test("staleness fires when the connection goes silent past STALE_MS", async () => {
    let clock = 1_000_000
    Date.now = () => clock
    const m = manualBody()
    installFetch({ event: () => new Response(m.stream) })
    const stream = chatOf(bridge())
    const chunks: any[] = []
    const p = (async () => {
      for await (const c of stream) chunks.push(c)
    })()
    m.push(connected())
    await ticks(4)
    m.push(part("m1", "a"))
    await ticks()
    clock += 130_000 // no heartbeat in the gap
    m.push(part("m1", "b"))
    await expect(p).rejects.toThrow("timed out")
    expect(textOf(chunks)).toBe("a")
  })
})
