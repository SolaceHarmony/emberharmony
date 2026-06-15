import { llm, DEFAULT_API_CONNECT_OPTIONS, type APIConnectOptions } from "@livekit/agents"

/**
 * Bridges the voice agent's LLM step into an EmberHarmony session.
 *
 * Instead of calling a model provider directly, each voice turn posts the
 * transcribed utterance to the session's prompt API and streams the reply
 * text back out of the server's SSE event feed. The session does the real
 * work (model, tools, permissions, context), the voice pipeline just speaks
 * the streamed text. User utterances and replies show up in the chat UI like
 * any other message.
 */
export interface SessionBridgeOptions {
  /** EmberHarmony server origin, e.g. http://localhost:4096 */
  serverUrl: string
  /** Project directory the session belongs to */
  directory: string
  /** Session to bridge into */
  sessionID: string
  /** Basic auth, if the server is password-protected */
  username?: string
  password?: string
  /** Model for sessions with no message history to inherit from */
  fallbackModel?: { providerID: string; modelID: string }
  /** Session agent for each voice turn (e.g. plan/build), resolved per turn */
  agent?: () => string | undefined
  /** Extra per-message system instructions attached to every voice prompt */
  system?: string
  /**
   * Whether to abort the server session when voice interrupts.
   * Defaults to true for backward compatibility, but should be set to false
   * for the brain session — the brain controls its own aborts deliberately,
   * not as a side effect of voice interruption.
   */
  abortOnInterrupt?: boolean
}

function headers(opts: SessionBridgeOptions): Record<string, string> {
  const result: Record<string, string> = {
    "x-emberharmony-directory": encodeURIComponent(opts.directory),
  }
  if (opts.password) {
    const user = opts.username ?? "emberharmony"
    result["authorization"] = `Basic ${Buffer.from(`${user}:${opts.password}`).toString("base64")}`
  }
  return result
}

async function* serverEvents(opts: SessionBridgeOptions, signal: AbortSignal): AsyncGenerator<any> {
  const response = await fetch(`${opts.serverUrl}/event`, {
    headers: { ...headers(opts), accept: "text/event-stream" },
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`event stream failed: ${response.status} ${await response.text().catch(() => "")}`)
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += value
      let boundary: number
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data:")) continue
          const data = line.slice(5).trim()
          if (!data) continue
          yield JSON.parse(data)
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

export class SessionLLM extends llm.LLM {
  constructor(readonly opts: SessionBridgeOptions) {
    super()
  }

  label(): string {
    return "emberharmony.SessionLLM"
  }

  chat({
    chatCtx,
    connOptions = DEFAULT_API_CONNECT_OPTIONS,
  }: {
    chatCtx: llm.ChatContext
    toolCtx?: llm.ToolContext
    connOptions?: APIConnectOptions
    parallelToolCalls?: boolean
    toolChoice?: llm.ToolChoice
    extraKwargs?: Record<string, unknown>
  }): SessionLLMStream {
    return new SessionLLMStream(this, this.opts, { chatCtx, connOptions })
  }
}

export class SessionLLMStream extends llm.LLMStream {
  #opts: SessionBridgeOptions

  constructor(
    sessionLLM: SessionLLM,
    opts: SessionBridgeOptions,
    { chatCtx, connOptions }: { chatCtx: llm.ChatContext; connOptions: APIConnectOptions },
  ) {
    super(sessionLLM, { chatCtx, connOptions })
    this.#opts = opts
  }

  protected async run(): Promise<void> {
    const text = this.#latestUserText()
    if (!text) return
    const signal = this.abortController.signal

    // Voice interruption aborts this stream. For the brain session, we don't
    // abort the server session on every interruption — the brain controls
    // its own aborts deliberately. For backward compatibility, sessions
    // that don't set abortOnInterrupt still abort on interruption.
    //
    // Without the abort, a busy session blocks the next turn's prompt.
    // With it, every voice interruption kills the session — causing abort
    // cascades when the brain is just narrating. The brain session uses
    // abortOnInterrupt=false and sends deliberate aborts only when the
    // attached session should stop.
    const abortOnInterrupt = this.#opts.abortOnInterrupt !== false
    if (abortOnInterrupt) {
      signal.addEventListener("abort", () => {
        fetch(`${this.#opts.serverUrl}/session/${this.#opts.sessionID}/abort`, {
          method: "POST",
          headers: headers(this.#opts),
        }).catch(() => {})
      })
    }

    const events = serverEvents(this.#opts, signal)
    // generators are lazy: pull the first event (server.connected) so the SSE
    // connection is established before the prompt is posted and no reply
    // delta can be missed
    await events.next()

    // continue with whatever model the session is already using — the server
    // default is only well-defined when the user has configured one
    const model = await this.#sessionModel(signal)

    const agent = this.#opts.agent?.()
    const response = await fetch(`${this.#opts.serverUrl}/session/${this.#opts.sessionID}/prompt_async`, {
      method: "POST",
      headers: { ...headers(this.#opts), "content-type": "application/json" },
      body: JSON.stringify({
        parts: [{ type: "text", text }],
        ...(model ? { model } : {}),
        ...(agent ? { agent } : {}),
        ...(this.#opts.system ? { system: this.#opts.system } : {}),
      }),
      signal,
    })
    if (!response.ok) {
      throw new Error(`session prompt failed: ${response.status} ${await response.text().catch(() => "")}`)
    }

    // A turn can span several assistant messages: a tool-call step finalizes
    // its message (sets time.completed) BEFORE the tool runs, then the next
    // step opens a fresh assistant message for the post-tool reply. So we
    // stream text from every assistant message in this session — under one
    // stable id so TTS speaks it as one continuous utterance — and end the
    // turn only when the session returns to idle.
    //
    // Ending on a single message's completion (the old behaviour) cut the reply
    // off the instant the first tool-call message finalized: this stream closed,
    // its abort handler POSTed /abort, and the server killed the still-running
    // tool ("operation aborted") while the continuation never streamed.
    //
    // The server emits a heartbeat every 30s, which bumps the activity clock —
    // so the staleness check only fires when the SSE connection itself goes
    // dead (no heartbeat for STALE_MS), turning a silently-dropped stream into
    // an error instead of a worker hang. A long-running tool keeps the stream
    // alive via those heartbeats, so it never trips this.
    const STALE_MS = 120_000
    let lastActivity = Date.now()
    let replyId: string | undefined
    for await (const event of events) {
      if (Date.now() - lastActivity > STALE_MS) {
        throw new Error(`session reply timed out (no session events for ${STALE_MS / 1000}s)`)
      }
      if (event.type === "message.part.updated") {
        const { part, delta } = event.properties ?? {}
        if (!part || part.sessionID !== this.#opts.sessionID) continue
        lastActivity = Date.now()
        if (part.type !== "text" || !delta) continue
        if (!replyId) replyId = part.messageID
        this.queue.put({ id: replyId!, delta: { role: "assistant", content: delta } })
      }
      if (event.type === "message.updated") {
        if (event.properties?.info?.sessionID === this.#opts.sessionID) lastActivity = Date.now()
      }
      // Heartbeats (every 30s) prove the SSE connection is alive even while a
      // long tool runs with no message events — bump activity so the staleness
      // check only trips on a genuinely dead connection, not a slow tool.
      if (event.type === "server.heartbeat") lastActivity = Date.now()
      // The whole turn — every step and tool call — is done only when the
      // session goes idle (also fires on server-side cancel, which ends the
      // turn just the same).
      if (event.type === "session.idle" && event.properties?.sessionID === this.#opts.sessionID) {
        return
      }
      if (event.type === "session.error") {
        const props = event.properties ?? {}
        if (props.sessionID && props.sessionID !== this.#opts.sessionID) continue
        throw new Error(`session error: ${JSON.stringify(props.error ?? props)}`)
      }
    }
  }

  async #sessionModel(signal: AbortSignal): Promise<{ providerID: string; modelID: string } | undefined> {
    const response = await fetch(`${this.#opts.serverUrl}/session/${this.#opts.sessionID}/message?limit=100`, {
      headers: headers(this.#opts),
      signal,
    })
    if (!response.ok) return this.#opts.fallbackModel
    const messages: Array<{ info: { role: string; providerID?: string; modelID?: string } }> | null = await response
      .json()
      .catch(() => null)
    if (!Array.isArray(messages)) return this.#opts.fallbackModel
    for (let i = messages.length - 1; i >= 0; i--) {
      const info = messages[i]!.info
      if (info.role === "assistant" && info.providerID && info.modelID) {
        return { providerID: info.providerID, modelID: info.modelID }
      }
    }
    return this.#opts.fallbackModel
  }

  #latestUserText(): string | undefined {
    const items = this.chatCtx.items
    for (let i = items.length - 1; i >= 0; i--) {
      const item = items[i]!
      if (item.type === "message" && item.role === "user") {
        return item.textContent
      }
    }
    return undefined
  }
}
