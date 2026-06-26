import { describe, expect, test, mock, beforeEach } from "bun:test"

// VoiceWorkflow is the safety-critical gate for voice-controlled code execution:
// every turn must default to plan, and only an exact "BUILD" verdict from the
// gateway model may flip that single turn to build. These tests lock that logic
// so a future refactor (e.g. switching the exact-match to includes(), or letting
// an error fall through to build) can't silently weaken it. The real @livekit/
// agents is unresolvable in this env, so shim only what workflow.ts touches.

// Shared @livekit/agents shim (process-global mock.module must agree across all
// voice test files — see _livekit-agents.shim.ts).
import { agentsShim, warnings, resetLogs } from "./_livekit-agents.shim"

mock.module("@livekit/agents", () => agentsShim)

const { VoiceWorkflow } = await import("../../src/voice/workflow")

beforeEach(resetLogs)

/** A gateway-LLM stand-in: .chat() yields `verdict` once, or throws if asked. */
function fakeIntent(verdict: string | (() => never)) {
  let calls = 0
  return {
    get calls() {
      return calls
    },
    chat() {
      calls++
      if (typeof verdict === "function") verdict()
      return (async function* () {
        yield { delta: { content: verdict } }
      })()
    },
  }
}

async function modeAfter(verdict: string | (() => never), utterance = "do the thing") {
  const intent = fakeIntent(verdict)
  const wf = new VoiceWorkflow(intent as any)
  await wf.route(utterance)
  return { mode: wf.agent(), intent, wf }
}

describe("VoiceWorkflow.route (plan/build safety gate)", () => {
  test("exact BUILD verdict flips the turn to build", async () => {
    expect((await modeAfter("BUILD")).mode).toBe("build")
  })

  test("BUILD with surrounding whitespace still flips (trimmed)", async () => {
    expect((await modeAfter("  BUILD\n")).mode).toBe("build")
  })

  test("lowercase build is accepted (case-folded)", async () => {
    expect((await modeAfter("build")).mode).toBe("build")
  })

  test("a verdict that merely CONTAINS build does not grant execution", async () => {
    expect((await modeAfter("PLAN, not BUILD")).mode).toBe("plan")
    expect((await modeAfter("BUILD if you must, but actually plan")).mode).toBe("plan")
  })

  test("PLAN verdict stays in plan", async () => {
    expect((await modeAfter("PLAN")).mode).toBe("plan")
  })

  test("empty verdict stays in plan", async () => {
    expect((await modeAfter("")).mode).toBe("plan")
  })

  test("a classifier error falls back to plan and never grants execution", async () => {
    const { mode } = await modeAfter(() => {
      throw new Error("intent model down")
    })
    expect(mode).toBe("plan")
    expect(warnings.some((w) => w.includes("intent check failed"))).toBe(true)
  })

  test("an empty/whitespace utterance short-circuits to plan without calling the model", async () => {
    const { mode, intent } = await modeAfter("BUILD", "   ")
    expect(mode).toBe("plan")
    expect(intent.calls).toBe(0) // never even asks the gateway
  })

  test("mode re-defaults to plan every turn: a confirmed build does not carry over", async () => {
    // same instance, two turns: the gateway confirms BUILD on turn 1, then a
    // plain question on turn 2 must drop back to plan (no sticky execution).
    let verdict = "BUILD"
    const intent = {
      chat() {
        const v = verdict
        return (async function* () {
          yield { delta: { content: v } }
        })()
      },
    }
    const wf = new VoiceWorkflow(intent as any)
    await wf.route("yes do it")
    expect(wf.agent()).toBe("build")
    verdict = "PLAN"
    await wf.route("what about the tests?")
    expect(wf.agent()).toBe("plan")
  })
})
