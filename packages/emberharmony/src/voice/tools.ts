import { z } from "zod"
import { Tool } from "../tool/tool"
import { Instance } from "../project/instance"
import { Session } from "../session"
import { SessionPrompt } from "../session/prompt"
import { Log } from "../util/log"

/**
 * Voice brain tools — server-side session tools that the brain agent uses to
 * interact with the project. The brain session owns the tool schemas; the
 * EmberHarmony server executes them and returns results.
 *
 * Build authorization is enforced in code, not instructions. submit_prompt
 * derives the effective agent from ctx.agent — a trusted signal set by the
 * worker's intent classifier — NOT from the model-supplied params. The model
 * cannot forge "build" mode: if the classifier sent "plan", the tool forces
 * "plan" regardless of what the model asks for. This is one-shot — each call
 * to submit_prompt uses the classifier's verdict for that turn, and every
 * turn re-defaults to "plan".
 */

const log = Log.create({ service: "voice.tools" })

/**
 * List open sessions in the current project.
 * The brain uses this to tell the user what sessions are available.
 */
export const ListSessionsTool = Tool.define("list_sessions", {
  description:
    "List open sessions in the current project. Returns session IDs, titles, and status. " +
    "Use this to tell the user what sessions are available and help them choose which one to work on.",
  parameters: z.object({
    search: z.string().optional().describe("Optional search query to filter sessions by title"),
  }),
  async execute(params, ctx) {
    const sessions: Array<{ id: string; title: string; status: string }> = []
    for await (const session of Session.list()) {
      if (params.search && !session.title.toLowerCase().includes(params.search.toLowerCase())) continue
      sessions.push({ id: session.id, title: session.title, status: "active" })
    }
    if (sessions.length === 0) {
      return {
        title: "No sessions found",
        metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: "" },
        output: params.search ? `No sessions matching "${params.search}".` : "No open sessions.",
      }
    }
    const lines = sessions.map((s) => `- ${s.title} (${s.id})`)
    return {
      title: `${sessions.length} session${sessions.length === 1 ? "" : "s"}`,
      metadata: { sessions, sessionID: "" },
      output: lines.join("\n"),
    }
  },
})

/**
 * Get recent messages from a session.
 * The brain uses this to understand what's happening in the attached session.
 */
export const GetRecentActivityTool = Tool.define("get_recent_activity", {
  description:
    "Get recent activity from a session. Returns the last few messages as a summary. " +
    "Use this to understand what's happening in a session before attaching to it, " +
    "or to check on progress after submitting a prompt.",
  parameters: z.object({
    sessionID: z.string().describe("The session ID to get activity from"),
    limit: z.number().optional().describe("Number of recent messages to return (default 5)"),
  }),
  async execute(params, ctx) {
    const session = await Session.get(params.sessionID).catch(() => undefined)
    if (!session) {
      return {
        title: "Session not found",
        metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
        output: `Session ${params.sessionID} not found. It may have been deleted.`,
      }
    }

    const messages = await Instance.provide({
      directory: session.directory,
      async fn() {
        const msgs = await Session.messages({ sessionID: params.sessionID, limit: params.limit ?? 5 })
        return msgs
      },
    })

    if (!messages || messages.length === 0) {
      return {
        title: "No recent activity",
        metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
        output: `No recent messages in "${session.title}".`,
      }
    }

    const lines = messages.map((m) => {
      const role = m.info.role === "user" ? "User" : (m.info.agent ?? "Assistant")
      const textParts = m.parts
        .filter((p) => p.type === "text")
        .map((p) => p.text ?? "")
        .join(" ")
      const summary = textParts.length > 200 ? textParts.slice(0, 200) + "..." : textParts
      return `${role}: ${summary || "(tool call)"}`
    })

    return {
      title: `Recent activity in "${session.title}"`,
      metadata: { sessions: [], sessionID: params.sessionID },
      output: lines.join("\n"),
    }
  },
})

/**
 * Submit a prompt to the attached project session.
 *
 * BUILD AUTHORIZATION IS ENFORCED IN CODE, NOT INSTRUCTIONS.
 *
 * The model-supplied "agent" parameter is IGNORED. The effective agent is
 * derived from ctx.agent — a trusted signal set by the worker's intent
 * classifier. If the classifier sent "plan", this tool forces "plan"
 * regardless of what the model asks. A prompt injection cannot grant build
 * access because it cannot control ctx.agent.
 *
 * This is one-shot: each call uses the classifier's verdict for that turn.
 * Every turn re-defaults to "plan" — a single "yes" authorizes exactly one
 * build submission.
 */
export const SubmitPromptTool = Tool.define("submit_prompt", {
  description:
    "Submit a prompt to the attached project session. Use this to send work to the session " +
    "when you and the user have confirmed what to build. The agent mode is determined automatically.",
  parameters: z.object({
    sessionID: z.string().describe("The session ID to submit the prompt to"),
    directory: z.string().describe("The project directory for the session"),
    text: z.string().describe("The prompt text to submit"),
  }),
  async execute(params, ctx) {
    // BUILD SAFETY: ctx.agent is the TRUSTED signal from the worker's intent
    // classifier. The model cannot control this value. If the classifier
    // said "plan", the attached session runs in plan mode — no amount of
    // prompt injection can upgrade it to "build".
    const effectiveAgent = ctx.agent === "build" ? "build" : "plan"

    log.info("submit_prompt", {
      sessionID: params.sessionID,
      agent: ctx.agent,
      effectiveAgent,
    })

    try {
      await Instance.provide({
        directory: params.directory,
        async fn() {
          // Fire and forget — prompt_async semantics. The session processes
          // the prompt asynchronously while the voice agent continues.
          SessionPrompt.prompt({
            sessionID: params.sessionID,
            parts: [{ type: "text", text: params.text }],
            agent: effectiveAgent,
          })
        },
      })
      return {
        title: "Prompt submitted",
        metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
        output: `Prompt submitted to session ${params.sessionID} in ${effectiveAgent} mode. The session is now processing.`,
      }
    } catch (error) {
      if (error instanceof Session.BusyError) {
        return {
          title: "Session busy",
          metadata: {
            sessions: [] as Array<{ id: string; title: string; status: string }>,
            sessionID: params.sessionID,
          },
          output: `Session ${params.sessionID} is busy. Please wait for it to finish and try again.`,
        }
      }
      return {
        title: "Prompt submission failed",
        metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
        output: `Failed to submit prompt to session ${params.sessionID}: ${error instanceof Error ? error.message : String(error)}`,
      }
    }
  },
})

/**
 * Abort the current generation in the attached session.
 */
export const AbortAttachedTool = Tool.define("abort_attached", {
  description:
    "Abort the current generation in the attached session. Use this when the user " +
    "interrupts while the session is processing, or when you need to stop a long-running task.",
  parameters: z.object({
    sessionID: z.string().describe("The session ID to abort"),
    directory: z.string().describe("The project directory for the session"),
  }),
  async execute(params, ctx) {
    await Instance.provide({
      directory: params.directory,
      async fn() {
        SessionPrompt.cancel(params.sessionID)
      },
    })

    return {
      title: "Session aborted",
      metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
      output: `Session ${params.sessionID} aborted.`,
    }
  },
})

/**
 * Change the model for the attached session.
 */
export const SetModelTool = Tool.define("set_model", {
  description:
    "Change the AI model used by the attached session. Use this when the user asks to " +
    "switch models (e.g. 'use Claude for this task' or 'switch to GPT-4').",
  parameters: z.object({
    sessionID: z.string().describe("The session ID"),
    directory: z.string().describe("The project directory for the session"),
    providerID: z.string().describe("The provider ID (e.g. 'openai', 'anthropic', 'ollama')"),
    modelID: z.string().describe("The model ID (e.g. 'gpt-4o', 'claude-3.5-sonnet')"),
  }),
  async execute(params, ctx) {
    // Model changes through voice are not yet supported — the session PATCH
    // endpoint doesn't have a model field. We'll add this when it does.
    return {
      title: "Model change requested",
      metadata: { sessions: [] as Array<{ id: string; title: string; status: string }>, sessionID: params.sessionID },
      output: `Model change to ${params.providerID}/${params.modelID} is not yet supported via voice. Please change the model in the session settings.`,
    }
  },
})

/**
 * Attach to a project session. This transitions the voice agent from
 * concierge mode (browsing sessions) to operator mode (working in a session).
 * The worker detects the attached_session metadata change and hands off
 * to the Operator agent.
 */
export const AttachSessionTool = Tool.define("attach_session", {
  description:
    "Attach to a project session for voice interaction. Use this when the user wants to " +
    "work on a specific session — say, to start coding or make changes. After attaching, " +
    "you'll be in operator mode and can submit prompts to the session.",
  parameters: z.object({
    sessionID: z.string().describe("The session ID to attach to"),
    directory: z.string().describe("The project directory for the session"),
  }),
  async execute(params, ctx) {
    const session = await Session.get(params.sessionID).catch(() => undefined)
    if (!session) {
      return {
        title: "Session not found",
        metadata: {
          sessions: [] as Array<{ id: string; title: string; status: string }>,
          sessionID: params.sessionID,
          attached_session: "",
          attached_directory: "",
        },
        output: `Session ${params.sessionID} not found. It may have been deleted.`,
      }
    }

    return {
      title: `Attached to "${session.title}"`,
      metadata: {
        sessions: [{ id: session.id, title: session.title, status: "active" }] as Array<{
          id: string
          title: string
          status: string
        }>,
        sessionID: params.sessionID,
        attached_session: params.sessionID,
        attached_directory: params.directory,
      },
      output: `Attached to session "${session.title}" (${params.sessionID}). You are now in operator mode. You can submit prompts to work on this session.`,
    }
  },
})

/**
 * Detach from the current session. This transitions the voice agent from
 * operator mode back to concierge mode (browsing sessions).
 * The worker detects the detached state and hands off to the Concierge.
 */
export const DetachSessionTool = Tool.define("detach_session", {
  description:
    "Detach from the current session. Use this when the user wants to switch to a " +
    "different session or go back to browsing sessions. After detaching, you'll be " +
    "in concierge mode and can list or search for other sessions.",
  parameters: z.object({}),
  async execute(params, ctx) {
    return {
      title: "Detached from session",
      metadata: {
        sessions: [] as Array<{ id: string; title: string; status: string }>,
        sessionID: "",
        attached_session: "",
        attached_directory: "",
      },
      output:
        "Detached from the current session. You are now in concierge mode and can list or search for other sessions.",
    }
  },
})

/**
 * All voice brain tools exported as an array for registration.
 */
export const voiceTools = [
  ListSessionsTool,
  GetRecentActivityTool,
  SubmitPromptTool,
  AbortAttachedTool,
  SetModelTool,
  AttachSessionTool,
  DetachSessionTool,
] as const
