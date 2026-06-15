import { llm, log } from "@livekit/agents"

/**
 * Voice workflow — plan/build routing with optional structured stages.
 *
 * Default mode (structured: false):
 *   The brain flows naturally. The classifier decides plan/build per turn.
 *   The brain proposes and asks for confirmation because its system prompt
 *   tells it to — not because a state machine forces it through named stages.
 *   The only enforcement is the build gate: ctx.agent in submit_prompt.
 *   Every turn re-defaults to "plan." A single "yes" authorizes one build.
 *
 * Structured mode (structured: true):
 *   The 5-stage machine activates:
 *     gathering → proposing → confirmed → executing → reviewing → gathering
 *   The agent must propose, and the user must confirm. Transitions are
 *   enforced in code. The user can say "skip" or "exit workflow" to
 *   escape back to free-form at any time.
 *
 * Both modes share the same build gate — the classifier's verdict flows
 * through ctx.agent, which submit_prompt trusts. A prompt injection cannot
 * grant build access because the model cannot control ctx.agent.
 */

export type Stage = "gathering" | "proposing" | "confirmed" | "executing" | "reviewing"

export class VoiceWorkflow {
  #mode: "plan" | "build" = "plan"
  #stage: Stage = "gathering"
  #structured: boolean
  #intent: llm.LLM

  constructor(opts: { intent: llm.LLM; structured?: boolean }) {
    this.#intent = opts.intent
    this.#structured = opts.structured ?? false
  }

  /** Current stage (only meaningful when structured is true) */
  get stage(): Stage {
    return this.#stage
  }

  /** Whether the workflow allows build-mode tool calls */
  get canBuild(): boolean {
    if (!this.#structured) return this.#mode === "build"
    return this.#stage === "confirmed" || this.#stage === "executing"
  }

  /** Whether the structured 5-stage machine is active */
  get structured(): boolean {
    return this.#structured
  }

  /** Agent name for the current voice turn — wired into the session bridge */
  agent(): string {
    return this.canBuild ? "build" : "plan"
  }

  /**
   * Transition the workflow to a new stage.
   * Only valid when structured mode is active. Only valid transitions
   * are allowed — invalid ones are no-ops.
   */
  transition(next: Stage): void {
    if (!this.#structured) {
      log().warn("workflow: transitions are only available in structured mode")
      return
    }
    const allowed = transitions[this.#stage]
    if (!allowed.includes(next)) {
      log().warn(`workflow: invalid transition ${this.#stage} → ${next}`)
      return
    }
    log().info(`workflow: ${this.#stage} → ${next}`)
    this.#stage = next
  }

  /**
   * Escape the structured workflow and return to free-form mode.
   * The user can say "exit workflow" or "skip the stages" at any time.
   */
  escape(): void {
    if (!this.#structured) return
    log().info("workflow: escaping structured mode")
    this.#stage = "gathering"
    this.#mode = "plan"
  }

  /**
   * Called when the user's turn is finalized, before the session bridge runs.
   * Uses the intent classifier to determine whether the utterance is a
   * confirmation (upgrade to build) or a new instruction (stay in plan).
   *
   * In default mode: the classifier verdict directly sets plan/build.
   * In structured mode: the classifier verdict drives stage transitions.
   * In both modes: classification failure must never grant execution.
   */
  async route(utterance: string): Promise<void> {
    if (!utterance.trim()) return

    // Check for escape phrases before running the classifier
    if (this.#structured && isEscape(utterance)) {
      this.escape()
      return
    }

    try {
      const chatCtx = llm.ChatContext.empty()
      chatCtx.addMessage({
        role: "system",
        content:
          "You route a voice-controlled coding assistant between planning and execution. " +
          "Reply with exactly one word. Reply BUILD only if the user is explicitly confirming " +
          "that the assistant should go ahead and execute work that was previously discussed or " +
          'proposed — e.g. "yes, do it", "go ahead", "sounds good, proceed", "ship it", ' +
          '"run it", "just do it". Reply PLAN for everything else: questions, ideas, requests to look at ' +
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
        if (this.#structured) {
          this.routeStructuredConfirm()
        } else {
          // Default mode: classifier says build — one-shot authorization
          this.#mode = "build"
        }
      } else {
        if (this.#structured) {
          this.routeStructuredPlan()
        } else {
          // Default mode: not a confirmation — stay in plan
          this.#mode = "plan"
        }
      }

      log().info(
        `voice workflow: ${this.#structured ? this.#stage : this.#mode} (intent: ${verdict.trim() || "<empty>"})`,
      )
    } catch (error) {
      // Classification failure must never grant execution
      this.#mode = "plan"
      log().warn(`voice workflow intent check failed: ${error}`)
    }
  }

  /**
   * Structured mode: handle a BUILD verdict from the classifier.
   * Transitions through the stage machine.
   */
  routeStructuredConfirm(): void {
    if (this.#stage === "proposing") {
      this.transition("confirmed")
    } else if (this.#stage === "confirmed" || this.#stage === "executing") {
      // Already building — keep going
    } else if (this.#stage === "gathering") {
      // User confirmed before the agent proposed. In structured mode,
      // treat as a direct skip — jump straight to confirmed.
      log().info("workflow: skipping to confirmed (user confirmed early)")
      this.#stage = "confirmed"
    } else {
      // reviewing or other — stay put
      log().info(`workflow: BUILD verdict in ${this.#stage}, staying`)
    }
  }

  /**
   * Structured mode: handle a PLAN verdict from the classifier.
   * The user is still discussing, not confirming.
   */
  routeStructuredPlan(): void {
    if (this.#stage === "proposing") {
      this.#stage = "gathering"
    }
    // In all other stages, a PLAN verdict just means "not confirming"
  }
}

/** Valid transitions from each stage */
const transitions: Record<Stage, Stage[]> = {
  gathering: ["proposing", "confirmed"],
  proposing: ["confirmed", "gathering"],
  confirmed: ["executing"],
  executing: ["reviewing"],
  reviewing: ["gathering"],
}

/** Escape phrases that drop out of structured mode */
const ESCAPE_PHRASES = [
  "exit workflow",
  "skip the stages",
  "skip the workflow",
  "stop the workflow",
  "leave workflow",
  "no workflow",
  "free form",
  "freeform",
]

function isEscape(utterance: string): boolean {
  const lower = utterance.toLowerCase().trim()
  return ESCAPE_PHRASES.some((phrase) => lower.includes(phrase))
}

export const VOICE_SYSTEM_PROMPT = [
  "The user is speaking to you by voice and hears your replies as speech.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations.",
  "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two,",
  "then ask whether to proceed — they will confirm out loud.",
].join(" ")
