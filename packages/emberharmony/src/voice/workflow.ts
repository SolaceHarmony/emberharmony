import { llm, log } from "@livekit/agents"

/**
 * Voice mode workflow: every spoken turn runs through the session's `plan`
 * agent (read-only — the session refuses mutating tools) unless the utterance
 * is an explicit confirmation to proceed, in which case that single turn runs
 * as the `build` agent. Confirmation is judged by a small fast model on the
 * LiveKit Inference gateway, so the heavy session model never decides its own
 * permissions. Every turn re-defaults to plan: each execution needs a fresh
 * spoken confirmation.
 */
export class VoiceWorkflow {
  #mode: "plan" | "build" = "plan"
  #intent: llm.LLM

  constructor(intent: llm.LLM) {
    this.#intent = intent
  }

  /** Agent name for the current voice turn — wired into the session bridge */
  agent(): string {
    return this.#mode
  }

  /**
   * Called when the user's turn is finalized, before the session bridge runs.
   * Flips this single turn to build mode only on explicit confirmation.
   */
  async route(utterance: string): Promise<void> {
    this.#mode = "plan"
    if (!utterance.trim()) return
    try {
      const chatCtx = llm.ChatContext.empty()
      chatCtx.addMessage({
        role: "system",
        content:
          "You route a voice-controlled coding assistant between planning and execution. " +
          "Reply with exactly one word. Reply BUILD only if the user is explicitly confirming " +
          "that the assistant should go ahead and execute work that was previously discussed or " +
          'proposed — e.g. "yes, do it", "go ahead", "sounds good, proceed", "ship it", ' +
          '"run it". Reply PLAN for everything else: questions, ideas, requests to look at ' +
          "something, hesitation, or new instructions that have not been confirmed.",
      })
      chatCtx.addMessage({ role: "user", content: utterance })
      const stream = this.#intent.chat({ chatCtx })
      let verdict = ""
      for await (const chunk of stream) {
        verdict += chunk.delta?.content ?? ""
      }
      if (/\bBUILD\b/i.test(verdict)) this.#mode = "build"
      log().info(`voice workflow: ${this.#mode} turn (intent: ${verdict.trim() || "<empty>"})`)
    } catch (error) {
      // classification failure must never grant execution — stay in plan
      this.#mode = "plan"
      log().warn(`voice workflow intent check failed, staying in plan mode: ${error}`)
    }
  }
}

export const VOICE_SYSTEM_PROMPT = [
  "The user is speaking to you by voice and hears your replies as speech.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations.",
  "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two,",
  "then ask whether to proceed — they will confirm out loud.",
].join(" ")
