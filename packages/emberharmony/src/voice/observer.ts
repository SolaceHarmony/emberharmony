import { Bus } from "../bus"
import { Log } from "../util/log"
import { MessageV2 } from "../session/message-v2"

/**
 * SessionObserver watches an attached project session's events and feeds
 * narrated interpretations into the brain session's context.
 *
 * Not yet wired into the voice worker. When implemented, the brain session
 * will use the observer to interpret what's happening in the attached session
 * and decide what to narrate. Currently only tested — no production consumer.
 *
 * This is a server-side bus subscriber, not a worker SSE client. It runs
 * within the EmberHarmony server process and subscribes to session events
 * via the Bus.
 */

const log = Log.create({ service: "voice.observer" })

/** Interpretation fed into the brain session as a system message */
export interface Observation {
  /** What happened (e.g. "editing the auth module") */
  summary: string
  /** Whether the observation represents completion */
  done: boolean
  /** Whether the observation represents an error */
  error: boolean
}

/**
 * SessionObserver watches the attached session's events and returns
 * observations for the brain session to narrate.
 *
 * Subscribes to Bus events for the attached session and interprets
 * them as natural-language summaries. Runs inside the EmberHarmony
 * server process with full access to the session event stream.
 */
export class SessionObserver {
  #unsub: (() => void) | undefined
  #observations: Observation[] = []
  #resolve: ((observation: Observation) => void) | undefined
  #stopped = false

  constructor(private readonly attachedSessionID: string) {}

  /**
   * Start observing the attached session. Subscribes to Bus events
   * and collects interpretations until stop() is called.
   */
  start(): void {
    this.#stopped = false
    this.#unsub = Bus.subscribe(MessageV2.Event.PartUpdated, (payload) => {
      const part = payload.properties.part
      if (part.sessionID !== this.attachedSessionID) return

      const interpretation = this.interpretPart(part)
      if (interpretation) {
        this.#observations.push(interpretation)
        this.#resolve?.(interpretation)
      }
    })
  }

  /**
   * Stop observing and clean up the subscription.
   */
  stop(): void {
    this.#stopped = true
    this.#unsub?.()
    this.#unsub = undefined
    this.#resolve = undefined
  }

  /**
   * Wait for the next observation from the attached session.
   * Returns immediately if there's a queued observation, otherwise
   * resolves when the next event comes in.
   * Returns undefined if the observer has been stopped.
   */
  async next(): Promise<Observation | undefined> {
    if (this.#stopped) return undefined
    if (this.#observations.length > 0) {
      return this.#observations.shift()
    }
    return new Promise<Observation | undefined>((resolve) => {
      if (this.#stopped) {
        resolve(undefined)
        return
      }
      this.#resolve = resolve
    })
  }

  /**
   * Interpret a message part into a natural-language observation.
   * The brain never reads raw tool output — it receives narrated
   * summaries that it then interprets for the user.
   */
  private interpretPart(part: MessageV2.Part): Observation | undefined {
    switch (part.type) {
      case "text":
        return {
          summary: "The assistant is responding with text.",
          done: false,
          error: false,
        }
      case "tool": {
        const toolName = part.tool
        const state = part.state
        if (state.status === "running") {
          return {
            summary: state.title ? `Running: ${state.title}` : `Running tool: ${toolName}`,
            done: false,
            error: false,
          }
        }
        if (state.status === "completed") {
          return {
            summary: state.title ?? `Completed: ${toolName}`,
            done: true,
            error: false,
          }
        }
        if (state.status === "error") {
          return {
            summary: `Error in ${toolName}: ${state.error}`,
            done: true,
            error: true,
          }
        }
        return undefined
      }
      case "reasoning":
        return undefined
      default:
        return undefined
    }
  }
}
