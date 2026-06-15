import { llm, voice } from "@livekit/agents"

/**
 * Concierge agent — the detached voice agent that handles session browsing,
 * searching, and attachment. No session is attached; the concierge helps
 * the user find and connect to a project session.
 *
 * The concierge always operates in plan mode. When the user asks to connect
 * to a session, the concierge calls `attach_session` which triggers a
 * LiveKit handoff to the Operator.
 */
export const CONCIERGE_INSTRUCTIONS = [
  "You are EmberHarmony's voice concierge.",
  "You help the user browse, search, and connect to coding sessions.",
  "No session is currently attached — you can list available sessions,",
  "search by title, and describe what each session is working on.",
  "When the user wants to work on a session, attach to it and hand off.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks.",
].join(" ")

/**
 * Operator agent — the attached voice agent that works inside a specific
 * session. Has access to build tools like submit_prompt and abort_attached.
 *
 * The operator can be in plan or build mode, controlled by the workflow
 * state machine. When the user detaches from the session, the operator
 * hands off back to the Concierge.
 */
export const OPERATOR_INSTRUCTIONS = [
  "You are EmberHarmony's voice operator, working inside an attached session.",
  "The user is speaking to you by voice and hears your replies as speech.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations.",
  "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two,",
  "then ask whether to proceed — they will confirm out loud.",
  "You have your own session — you are the thinker. When you need to act on the project,",
  "use tools like submit_prompt to send work to the attached project session.",
  "Never read raw tool output, file contents, or command output verbatim.",
  "Interpret what happened and narrate it naturally.",
].join(" ")

export class ConciergeAgent extends voice.Agent {
  constructor() {
    super({ instructions: CONCIERGE_INSTRUCTIONS })
  }
}

export class OperatorAgent extends voice.Agent {
  #workflow: { agent: () => string }

  constructor(workflow: { agent: () => string }) {
    super({ instructions: OPERATOR_INSTRUCTIONS })
    this.#workflow = workflow
  }

  /** Returns the session agent name for the current workflow state */
  get sessionAgent(): string {
    return this.#workflow.agent()
  }
}
