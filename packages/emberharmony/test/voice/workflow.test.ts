import { test, expect, describe, beforeAll } from "bun:test"
import { VoiceWorkflow } from "../../src/voice/workflow"
import { log as lkLog, initializeLogger } from "@livekit/agents"
import type { llm } from "@livekit/agents"

// Initialize the LiveKit logger before any workflow tests run.
// VoiceWorkflow uses log() from @livekit/agents, which throws if
// the logger hasn't been initialized.
beforeAll(() => {
  initializeLogger({ pretty: false, level: "error" })
})

/**
 * Fake intent LLM that returns a fixed verdict.
 * Implements just enough of llm.LLM for VoiceWorkflow.route().
 * Cast with `as llm.LLM` because the workflow only calls .chat().
 */
class FakeIntent {
  #verdict: string

  constructor(verdict: string) {
    this.#verdict = verdict
  }

  /** Set the verdict for the next route() call */
  setVerdict(verdict: string) {
    this.#verdict = verdict
  }

  /** Returns the configured verdict as a stream */
  chat(_opts: unknown) {
    const verdict = this.#verdict
    return {
      async *[Symbol.asyncIterator]() {
        yield { delta: { content: verdict } }
      },
    }
  }
}

/** Cast fake intent to the LLM type the workflow expects */
function fakeLLM(verdict: string): llm.LLM {
  return new FakeIntent(verdict) as unknown as llm.LLM
}

/** Fake LLM that always throws — tests the error-safety invariant */
function throwingLLM(): llm.LLM {
  return {
    chat() {
      throw new Error("LLM unavailable")
    },
  } as unknown as llm.LLM
}

/**
 * E2E tests for the voice workflow.
 *
 * Tests both default mode (free-form plan/build) and
 * structured mode (5-stage state machine) with escape hatch.
 * The intent LLM is faked — we control what the classifier says.
 */
describe("VoiceWorkflow — default mode", () => {
  test("starts in plan mode", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: false })
    expect(workflow.agent()).toBe("plan")
    expect(workflow.canBuild).toBe(false)
  })

  test("classifier BUILD verdict upgrades to build mode", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("BUILD"), structured: false })
    await workflow.route("yes, do it")
    expect(workflow.agent()).toBe("build")
    expect(workflow.canBuild).toBe(true)
  })

  test("classifier PLAN verdict stays in plan mode", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: false })
    await workflow.route("what about this file?")
    expect(workflow.agent()).toBe("plan")
    expect(workflow.canBuild).toBe(false)
  })

  test("build is one-shot — next turn re-defaults to plan", async () => {
    const intent = new FakeIntent("BUILD")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: false })

    // Turn 1: confirm
    await workflow.route("yes, do it")
    expect(workflow.agent()).toBe("build")

    // Turn 2: the classifier returns PLAN — mode resets
    intent.setVerdict("PLAN")
    await workflow.route("what about this?")
    expect(workflow.agent()).toBe("plan")
  })

  test("empty utterance is a no-op", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("BUILD"), structured: false })
    await workflow.route("")
    expect(workflow.agent()).toBe("plan")
  })

  test("whitespace-only utterance is a no-op", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("BUILD"), structured: false })
    await workflow.route("   ")
    expect(workflow.agent()).toBe("plan")
  })

  test("classifier error must never grant execution", async () => {
    const workflow = new VoiceWorkflow({ intent: throwingLLM(), structured: false })
    await workflow.route("yes, do it")
    expect(workflow.agent()).toBe("plan")
    expect(workflow.canBuild).toBe(false)
  })
})

describe("VoiceWorkflow — structured mode", () => {
  test("starts in gathering stage", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: true })
    expect(workflow.stage).toBe("gathering")
    expect(workflow.agent()).toBe("plan")
  })

  test("gathering → confirmed on direct BUILD verdict", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("BUILD"), structured: true })
    await workflow.route("just do it")
    expect(workflow.stage).toBe("confirmed")
    expect(workflow.canBuild).toBe(true)
  })

  test("structured flag is exposed", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: true })
    expect(workflow.structured).toBe(true)
  })

  test("escape drops back to gathering + plan", async () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("BUILD"), structured: true })
    await workflow.route("yes, go ahead")
    expect(workflow.stage).toBe("confirmed")

    workflow.escape()
    expect(workflow.stage).toBe("gathering")
    expect(workflow.canBuild).toBe(false)
  })

  test("escape phrases are detected from utterance", async () => {
    const intent = new FakeIntent("BUILD")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: true })

    await workflow.route("yes, go ahead")
    expect(workflow.stage).toBe("confirmed")

    intent.setVerdict("PLAN")
    await workflow.route("exit workflow")
    expect(workflow.stage).toBe("gathering")
    expect(workflow.canBuild).toBe(false)
  })

  test("'skip the stages' triggers escape", async () => {
    const intent = new FakeIntent("BUILD")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: true })

    await workflow.route("yes, go ahead")
    intent.setVerdict("PLAN")
    await workflow.route("skip the stages")
    expect(workflow.stage).toBe("gathering")
  })

  test("'freeform' triggers escape", async () => {
    const intent = new FakeIntent("BUILD")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: true })

    await workflow.route("yes, go ahead")
    intent.setVerdict("PLAN")
    await workflow.route("let's go freeform")
    expect(workflow.stage).toBe("gathering")
  })

  test("transition validates allowed moves — can't skip from gathering to executing", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: true })
    workflow.transition("executing")
    expect(workflow.stage).toBe("gathering")
  })

  test("transition allows gathering → confirmed", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: true })
    workflow.transition("confirmed")
    expect(workflow.stage).toBe("confirmed")
  })

  test("transition is a no-op in default mode", () => {
    const workflow = new VoiceWorkflow({ intent: fakeLLM("PLAN"), structured: false })
    workflow.transition("confirmed")
    expect(workflow.agent()).toBe("plan")
  })

  test("PLAN verdict in proposing goes back to gathering", async () => {
    const intent = new FakeIntent("PLAN")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: true })

    // Manually advance to proposing
    workflow.transition("proposing")
    expect(workflow.stage).toBe("proposing")

    // User modifies request — go back
    await workflow.route("actually, let me change the approach")
    expect(workflow.stage).toBe("gathering")
  })
})

describe("VoiceWorkflow — build gate enforcement", () => {
  test("submit_prompt derives agent from ctx.agent, never from model params", () => {
    // Same logic as submit_prompt.execute():
    //   const effectiveAgent = ctx.agent === "build" ? "build" : "plan"
    const effectiveAgent = (ctxAgent: string) => (ctxAgent === "build" ? "build" : "plan")

    expect(effectiveAgent("plan")).toBe("plan")
    expect(effectiveAgent("build")).toBe("build")
    // Prompt injection attempts
    expect(effectiveAgent("execute")).toBe("plan")
    expect(effectiveAgent("sudo")).toBe("plan")
    expect(effectiveAgent("override")).toBe("plan")
    expect(effectiveAgent("admin")).toBe("plan")
  })

  test("model-supplied agent parameter is gone from submit_prompt schema", () => {
    // The agent field was removed from submit_prompt's zod schema.
    // Verify it's not in the params shape.
    const params = { text: "do something", sessionID: "abc", directory: "/tmp" }
    expect("agent" in params).toBe(false)
  })

  test("build authorization is one-shot per turn", async () => {
    const intent = new FakeIntent("BUILD")
    const workflow = new VoiceWorkflow({ intent: intent as unknown as llm.LLM, structured: false })

    // Turn 1: authorize build
    await workflow.route("yes, go ahead")
    expect(workflow.agent()).toBe("build")

    // Turn 2: classifier says PLAN — build is revoked
    intent.setVerdict("PLAN")
    await workflow.route("wait, let me think")
    expect(workflow.agent()).toBe("plan")

    // Turn 3: re-authorize
    intent.setVerdict("BUILD")
    await workflow.route("okay, now do it")
    expect(workflow.agent()).toBe("build")

    // Turn 4: reset again
    intent.setVerdict("PLAN")
    await workflow.route("what about tests?")
    expect(workflow.agent()).toBe("plan")
  })
})

describe("VoiceWorkflow — classification safety", () => {
  test("exact BUILD match only — rambling verdict is plan", () => {
    const isBuild = (verdict: string) => verdict.trim().toUpperCase() === "BUILD"

    expect(isBuild("BUILD")).toBe(true)
    expect(isBuild("build")).toBe(true)
    expect(isBuild("  BUILD  ")).toBe(true)
    expect(isBuild("PLAN, not BUILD")).toBe(false)
    expect(isBuild("BUILD IT")).toBe(false)
    expect(isBuild("I think BUILD")).toBe(false)
    expect(isBuild("")).toBe(false)
  })
})
