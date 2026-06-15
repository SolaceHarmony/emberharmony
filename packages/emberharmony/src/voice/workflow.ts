import { llm, log } from "@livekit/agents"

/**
 * Voice workflow state machine.
 *
 * The agent moves through defined stages:
 *
 *   gathering  → you describe what you want
 *        ↓
 *   proposing  → the agent presents a plan and asks whether to proceed
 *        ↓       you confirm (voice or tap)
 *   confirmed  → one-time flip to build mode
 *        ↓
 *   executing  → the agent submits work to the attached session
 *        ↓       the session goes idle
 *   reviewing  → the agent summarizes what happened
 *        ↓       back to gathering
 *
 * The agent can't skip from gathering to executing. It must propose, and
 * you must confirm. The intent classifier can upgrade from plan to build,
 * but only when the workflow is in the confirmed stage — enforced in
 * code, not instructions. A prompt injection cannot grant build access.
 */

export type Stage = "gathering" | "proposing" | "confirmed" | "executing" | "reviewing"

export class VoiceWorkflow {
  #stage: Stage = "gathering"
  #intent: llm.LLM

  constructor(intent: llm.LLM) {
    this.#intent = intent
  }

  /** Current stage — exposed via participant attributes */
  get stage(): Stage {
    return this.#stage
  }

  /** Whether the workflow allows build-mode tool calls */
  get canBuild(): boolean {
    return this.#stage === "confirmed" || this.#stage === "executing"
  }

  /** Agent name for the current voice turn — wired into the session bridge */
  agent(): string {
    return this.canBuild ? "build" : "plan"
  }

  /**
   * Transition the workflow to a new stage.
   * Only valid transitions are allowed — invalid ones are no-ops.
   */
  transition(next: Stage): void {
    const allowed = transitions[this.#stage]
    if (!allowed.includes(next)) {
      log().warn(`workflow: invalid transition ${this.#stage} → ${next}`)
      return
    }
    log().info(`workflow: ${this.#stage} → ${next}`)
    this.#stage = next
  }

  /**
   * Called when the user's turn is finalized, before the session bridge runs.
   * Uses the intent classifier to determine whether the utterance is a
   * confirmation (upgrade to build) or a new instruction (stay in plan).
   *
   * The classifier verdict can only upgrade the workflow to build mode
   * when the stage is "confirmed" — enforced by canBuild above.
   */
  async route(utterance: string): Promise<void> {
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
      // Exact match only — a rambling verdict like "PLAN, not BUILD" must
      // never grant execution
      const confirmed = verdict.trim().toUpperCase() === "BUILD"

      if (confirmed) {
        // The classifier says this is a confirmation. If the workflow is
        // in proposing, transition to confirmed. If we're already in
        // confirmed/executing, stay there. Otherwise, stay in gathering.
        if (this.#stage === "proposing") {
          this.transition("confirmed")
        } else if (this.#stage === "confirmed" || this.#stage === "executing") {
          // Already building — keep going
        } else {
          // Premature confirmation — user confirmed before a plan was
          // proposed. Treat as gathering input.
          log().info("workflow: premature BUILD verdict, staying in gathering")
        }
      } else {
        // Not a confirmation — treat as new input. If we were proposing,
        // the user modified the request; go back to gathering.
        if (this.#stage === "proposing") {
          this.#stage = "gathering"
        }
      }

      log().info(`voice workflow: ${this.#stage} (intent: ${verdict.trim() || "<empty>"})`)
    } catch (error) {
      // Classification failure must never grant execution — stay in current stage
      log().warn(`voice workflow intent check failed: ${error}`)
    }
  }
}

/** Valid transitions from each stage */
const transitions: Record<Stage, Stage[]> = {
  gathering: ["proposing"],
  proposing: ["confirmed", "gathering"],
  confirmed: ["executing"],
  executing: ["reviewing"],
  reviewing: ["gathering"],
}

export const VOICE_SYSTEM_PROMPT = [
  "The user is speaking to you by voice and hears your replies as speech.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations.",
  "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two,",
  "then ask whether to proceed — they will confirm out loud.",
].join(" ")
