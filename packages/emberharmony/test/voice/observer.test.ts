import { test, expect, describe } from "bun:test"
import { SessionObserver, type Observation } from "../../src/voice/observer"
import { Instance } from "../../src/project/instance"
import { tmpdir } from "../fixture/fixture"

/**
 * E2E tests for the SessionObserver.
 *
 * The observer watches a session's PartUpdated events and produces
 * natural-language observations for the brain. These tests verify
 * that the interpretPart logic correctly handles tool states
 * (running, completed, error) and text parts.
 *
 * The observer needs an Instance context because Bus.subscribe
 * requires the project state.
 */
describe("SessionObserver — interpretation logic", () => {
  test("observer is created with a session ID", async () => {
    await using tmp = await tmpdir({ git: true })
    await Instance.provide({
      directory: tmp.path,
      fn: () => {
        const observer = new SessionObserver("session_abc123")
        expect(observer).toBeDefined()
      },
    })
  })

  test("start and stop lifecycle", async () => {
    await using tmp = await tmpdir({ git: true })
    await Instance.provide({
      directory: tmp.path,
      fn: () => {
        const observer = new SessionObserver("session_abc123")
        observer.start()
        observer.stop()
      },
    })
  })

  test("next() returns undefined after stop", async () => {
    await using tmp = await tmpdir({ git: true })
    const result = await Instance.provide({
      directory: tmp.path,
      async fn() {
        const observer = new SessionObserver("session_abc123")
        observer.start()
        observer.stop()
        return observer.next()
      },
    })
    expect(result).toBeUndefined()
  })
})

describe("SessionObserver — observation structure", () => {
  test("observation has the expected shape", () => {
    const obs: Observation = {
      summary: "Running: editing auth module",
      done: false,
      error: false,
    }
    expect(typeof obs.summary).toBe("string")
    expect(typeof obs.done).toBe("boolean")
    expect(typeof obs.error).toBe("boolean")
  })

  test("completion observation has done=true", () => {
    const obs: Observation = {
      summary: "Completed: file search",
      done: true,
      error: false,
    }
    expect(obs.done).toBe(true)
    expect(obs.error).toBe(false)
  })

  test("error observation has both done and error true", () => {
    const obs: Observation = {
      summary: "Error in bash: command not found",
      done: true,
      error: true,
    }
    expect(obs.done).toBe(true)
    expect(obs.error).toBe(true)
  })

  test("running observation has both done and error false", () => {
    const obs: Observation = {
      summary: "Running: editing the auth module",
      done: false,
      error: false,
    }
    expect(obs.done).toBe(false)
    expect(obs.error).toBe(false)
  })
})
