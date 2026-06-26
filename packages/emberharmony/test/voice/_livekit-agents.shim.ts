// Shared @livekit/agents shim for voice unit tests.
//
// mock.module() is process-global, so every voice test file that mocks
// "@livekit/agents" must agree on ONE shim — otherwise the last registration
// wins and a file that needs an export the winner omits (e.g. `log` vs
// `DEFAULT_API_CONNECT_OPTIONS`) fails to link. This module is that single
// superset: enough of the real surface for bridge.ts AND workflow.ts.
//
// The real package can't be imported in this env (eager native deps:
// ffmpeg-installer, a broken form-data symlink), which is the other reason we
// shim rather than load it.

/** Captured log output so tests can assert on warn/info messages. */
export const warnings: string[] = []
export const infos: string[] = []
export function resetLogs() {
  warnings.length = 0
  infos.length = 0
}

class FakeQueue {
  items: any[] = []
  resolvers: Array<(v: { value: any; done: boolean }) => void> = []
  closed = false
  put(item: any) {
    const r = this.resolvers.shift()
    if (r) r({ value: item, done: false })
    else this.items.push(item)
  }
  close() {
    this.closed = true
    let r
    while ((r = this.resolvers.shift())) r({ value: undefined, done: true })
  }
  next(): Promise<{ value: any; done: boolean }> {
    if (this.items.length) return Promise.resolve({ value: this.items.shift(), done: false })
    if (this.closed) return Promise.resolve({ value: undefined, done: true })
    return new Promise((res) => this.resolvers.push(res))
  }
}

class ChatContextShim {
  items: any[] = []
  static empty() {
    return new ChatContextShim()
  }
  addMessage({ role, content }: { role: string; content: string }) {
    this.items.push({ type: "message", role, content, textContent: content })
    return this
  }
}

class LLMShim {}

// Reimplements the @livekit/agents LLMStream queue->async-iterator contract the
// session bridge relies on: run() is overridden by SessionLLMStream, puts chunks
// on `this.queue`, and the public async iterator drains them. A genuine run()
// error surfaces after draining; an abort is a clean cancellation.
class LLMStreamShim {
  #chatCtx: any
  connOptions: any
  abortController = new AbortController()
  queue = new FakeQueue()
  #started = false
  #runPromise: Promise<void> | undefined
  constructor(_llm: any, { chatCtx, connOptions }: { chatCtx: any; connOptions: any }) {
    this.#chatCtx = chatCtx
    this.connOptions = connOptions
  }
  get chatCtx() {
    return this.#chatCtx
  }
  protected async run(): Promise<void> {}
  #ensureRun() {
    if (this.#started) return
    this.#started = true
    this.#runPromise = (async () => {
      try {
        await this.run()
      } catch (e: any) {
        if (e?.name !== "AbortError") throw e
      } finally {
        this.queue.close()
      }
    })()
  }
  async *[Symbol.asyncIterator]() {
    this.#ensureRun()
    while (true) {
      const { value, done } = await this.queue.next()
      if (done) break
      yield value
    }
    await this.#runPromise
  }
  close() {
    this.abortController.abort()
  }
}

export const agentsShim = {
  llm: { LLM: LLMShim, LLMStream: LLMStreamShim, ChatContext: ChatContextShim },
  DEFAULT_API_CONNECT_OPTIONS: { maxRetry: 0, retryIntervalMs: 0, timeoutMs: 30_000 },
  log: () => ({ info: (m: string) => infos.push(m), warn: (m: string) => warnings.push(m) }),
}

export { ChatContextShim }
