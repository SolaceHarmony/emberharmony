import fs from "fs/promises"
import path from "path"
import { Global } from "../global"
import { Instance } from "../project/instance"
import { Session } from "../session"
import { Log } from "../util/log"

/**
 * The voice brain is a permanent EmberHarmony session. It lives in its own
 * project directory at ~/.local/share/emberharmony/voice/ — a real project
 * with .emberharmony/ config, message history, and a session store.
 *
 * The brain session uses SessionLLM just like the current voice bridge, but
 * targets this permanent session instead of the attached project session. It
 * gets its own system prompt (narrate, interpret, never read raw output) and
 * its own model selection. When the brain needs to act on a project session,
 * it uses tools like submit_prompt to send work there.
 */

const log = Log.create({ service: "voice.brain" })

/** The permanent voice project directory */
export const VOICE_PROJECT_DIR = path.join(Global.Path.data, "voice")

/** The .emberharmony config subdirectory inside the voice project */
export const VOICE_CONFIG_DIR = path.join(VOICE_PROJECT_DIR, ".emberharmony")

/**
 * Ensure the voice project directory exists and is initialized.
 * Creates the directory tree and a minimal config if they don't exist yet.
 * Returns the directory path.
 */
export async function ensureVoiceProject(): Promise<string> {
  await fs.mkdir(VOICE_PROJECT_DIR, { recursive: true })
  await fs.mkdir(VOICE_CONFIG_DIR, { recursive: true })

  // Write a minimal emberharmony.jsonc if it doesn't exist yet.
  // This makes Project.fromDirectory happy — it needs a config file
  // to resolve the project. The voice project doesn't need any
  // special configuration; it just needs to be a valid project root.
  const configPath = path.join(VOICE_CONFIG_DIR, "emberharmony.jsonc")
  try {
    await fs.access(configPath)
  } catch {
    log.info("initializing voice project config", { path: configPath })
    await fs.writeFile(configPath, "{}\n")
  }

  log.info("voice project ready", { directory: VOICE_PROJECT_DIR })
  return VOICE_PROJECT_DIR
}

/**
 * Find or create the permanent brain session in the voice project.
 * Runs within the voice project's Instance context so Session.create
 * uses the voice project's directory and project ID.
 *
 * Returns the session ID.
 */
export async function ensureBrainSession(): Promise<string> {
  const directory = await ensureVoiceProject()
  return Instance.provide({
    directory,
    async fn() {
      // Look for an existing brain session — it has a known title
      const sessions = Session.list()
      for await (const session of sessions) {
        if (session.title === BRAIN_SESSION_TITLE) {
          log.info("found existing brain session", { id: session.id })
          return session.id
        }
      }

      // No existing brain session — create one
      const session = await Session.create({ title: BRAIN_SESSION_TITLE })
      log.info("created brain session", { id: session.id, directory })
      return session.id
    },
  })
}

/**
 * Create a NEW voice conversation session in the voice project.
 *
 * Voice is a *project* of conversations: each launch gets its own session — an
 * individual, viewable record — not one eternal session. Continuity comes from
 * gatherRecentVoiceContext() reading the recent ones, not from reuse.
 */
export async function createVoiceSession(): Promise<string> {
  const directory = await ensureVoiceProject()
  return Instance.provide({
    directory,
    async fn() {
      const session = await Session.create({ title: `Voice — ${new Date().toISOString()}` })
      log.info("created voice session", { id: session.id, directory })
      return session.id
    },
  })
}

/**
 * Gather short "where it landed" summaries of the most recent voice
 * conversations, formatted as memory for the brain's system prompt. This is the
 * orient step: the brain knows what was discussed last time and can open
 * casually referencing it. Returns "" when there are no prior conversations.
 */
export async function gatherRecentVoiceContext(excludeID: string, limit = 5): Promise<string> {
  const directory = await ensureVoiceProject()
  return Instance.provide({
    directory,
    async fn() {
      const infos: Session.Info[] = []
      for await (const info of Session.list()) {
        if (info.id !== excludeID && !info.parentID) infos.push(info)
      }
      infos.sort((a, b) => b.time.created - a.time.created)

      const lines: string[] = []
      for (const info of infos.slice(0, limit)) {
        const last = await lastAssistantText(info.id)
        if (last) lines.push(`- ${truncate(last, 160)}`)
      }
      if (lines.length === 0) return ""

      return [
        "MEMORY — recent voice conversations with this user (most recent first):",
        ...lines,
        "",
        "When you open a new conversation, greet casually and naturally. If there is recent",
        "context above, briefly reference what you were last working on and offer to continue",
        "or start something fresh. Never read this list aloud verbatim.",
      ].join("\n")
    },
  })
}

/** Last assistant text in a session — a cheap "where it landed" signal. */
async function lastAssistantText(sessionID: string): Promise<string | undefined> {
  const msgs = await Session.messages({ sessionID, limit: 20 })
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i]!.info.role !== "assistant") continue
    const textPart = msgs[i]!.parts.find((p) => p.type === "text" && p.text.trim().length > 0)
    if (textPart) return truncate((textPart as { text: string }).text, 400)
  }
  return undefined
}

function truncate(s: string, n: number): string {
  const t = s.replace(/\s+/g, " ").trim()
  return t.length > n ? `${t.slice(0, n - 1)}…` : t
}

/** The title used to identify the brain session across restarts */
export const BRAIN_SESSION_TITLE = "Voice Brain"

/**
 * System prompt for the voice brain session.
 *
 * This is NOT the voice agent's instructions — those live in agent.ts.
 * This is the system prompt injected into every brain session prompt via
 * the SessionBridgeOptions.system field. It tells the session model to
 * narrate, interpret, and summarize rather than reading raw output verbatim.
 */
export const BRAIN_SYSTEM_PROMPT = [
  "You are a voice interface to a coding assistant.",
  "The user is speaking to you by voice and hears your replies as speech.",
  "Keep replies short and speakable: plain sentences, no markdown, no code blocks, no long enumerations.",
  "When the user asks for changes while you are in plan mode, lay out a brief plan in a sentence or two,",
  "then ask whether to proceed — they will confirm out loud.",
  "You have your own session — you are the thinker. When you need to act on a project,",
  "use tools like submit_prompt to send work to the attached project session.",
  "Never read raw tool output, file contents, or command output verbatim.",
  "Interpret what happened and narrate it naturally — like an out-of-body commentary on the work being done.",
].join(" ")
