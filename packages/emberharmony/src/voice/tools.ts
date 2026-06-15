import { z } from "zod"
import { Tool } from "../tool/tool"
import { Instance } from "../project/instance"
import { Session } from "../session"
import { Log } from "../util/log"

/**
 * Voice brain tools — server-side session tools that the brain agent uses to
 * interact with the project. The brain session owns the tool schemas; the
 * EmberHarmony server executes them and returns results.
 *
 * These tools are only available when the brain session's agent is set to
 * "voice". They call the EmberHarmony server API to affect application state:
 * list sessions, attach/detach, submit prompts, etc.
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
        metadata: { sessions: [], sessionID: params.sessionID },
        output: `Session ${params.sessionID} not found. It may have been deleted.`,
      }
    }

    const messages = await Instance.provide({
      directory: session.directory,
      async fn() {
        const result = await fetch(
          `http://localhost:${process.env.EMBERHARMONY_PORT || 4096}/session/${params.sessionID}/message?limit=${params.limit ?? 5}`,
          { headers: { "x-emberharmony-directory": encodeURIComponent(session.directory) } },
        )
        if (!result.ok) return []
        return result.json() as Promise<
          Array<{ info: { role: string; agent?: string }; parts: Array<{ type: string; text?: string }> }>
        >
      },
    })

    if (!messages || messages.length === 0) {
      return {
        title: "No recent activity",
        metadata: { sessions: [], sessionID: params.sessionID },
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
 * This is the only tool that directly interacts with the project session.
 */
export const SubmitPromptTool = Tool.define("submit_prompt", {
  description:
    "Submit a prompt to the attached project session. Use this to send work to the session " +
    "when you and the user have confirmed what to build. Only use this in build mode after confirmation.",
  parameters: z.object({
    sessionID: z.string().describe("The session ID to submit the prompt to"),
    directory: z.string().describe("The project directory for the session"),
    text: z.string().describe("The prompt text to submit"),
    agent: z.string().optional().describe("The agent to use (e.g. 'build' or 'plan')"),
  }),
  async execute(params, ctx) {
    const result = await Instance.provide({
      directory: params.directory,
      async fn() {
        const response = await fetch(
          `http://localhost:${process.env.EMBERHARMONY_PORT || 4096}/session/${params.sessionID}/prompt_async`,
          {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-emberharmony-directory": encodeURIComponent(params.directory),
            },
            body: JSON.stringify({
              parts: [{ type: "text", text: params.text }],
              ...(params.agent ? { agent: params.agent } : {}),
            }),
          },
        )
        return response.status
      },
    })

    if (result === 204 || result === 200) {
      return {
        title: "Prompt submitted",
        metadata: { sessions: [], sessionID: params.sessionID },
        output: `Prompt submitted to session ${params.sessionID}. The session is now processing.`,
      }
    }

    return {
      title: "Prompt submission failed",
      metadata: { sessions: [], sessionID: params.sessionID },
      output: `Failed to submit prompt to session ${params.sessionID} (status ${result}).`,
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
    const result = await Instance.provide({
      directory: params.directory,
      async fn() {
        const response = await fetch(
          `http://localhost:${process.env.EMBERHARMONY_PORT || 4096}/session/${params.sessionID}/abort`,
          {
            method: "POST",
            headers: { "x-emberharmony-directory": encodeURIComponent(params.directory) },
          },
        )
        return response.status
      },
    })

    return {
      title: "Session aborted",
      metadata: { sessions: [], sessionID: params.sessionID },
      output: result === 200 ? `Session ${params.sessionID} aborted.` : `Abort request sent (status ${result}).`,
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
      metadata: { sessions: [], sessionID: params.sessionID },
      output: `Model change to ${params.providerID}/${params.modelID} is not yet supported via voice. Please change the model in the session settings.`,
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
] as const
