import { test, expect, describe } from "bun:test"
import { Instance } from "../../src/project/instance"
import { Session } from "../../src/session"
import { tmpdir } from "../fixture/fixture"
import fs from "fs/promises"
import path from "path"
import {
  ensureVoiceProject,
  ensureBrainSession,
  VOICE_PROJECT_DIR,
  VOICE_CONFIG_DIR,
  BRAIN_SESSION_TITLE,
} from "../../src/voice/brain"

/**
 * E2E tests for the voice brain session.
 *
 * These tests create real project directories and sessions — no mocks.
 * The brain session is a permanent EmberHarmony session in the voice
 * project directory. The voice project directory lives at
 * ~/.local/share/emberharmony/voice/ and is shared across all projects.
 */
describe("Voice brain session", () => {
  test("ensureVoiceProject creates the voice project directory", async () => {
    const directory = await ensureVoiceProject()
    expect(typeof directory).toBe("string")
    expect(directory).toContain("voice")

    // The config directory should exist
    const configExists = await fs.access(VOICE_CONFIG_DIR).then(
      () => true,
      () => false,
    )
    expect(configExists).toBe(true)
  })

  test("ensureBrainSession creates a session with the expected title", async () => {
    const sessionID = await ensureBrainSession()
    expect(typeof sessionID).toBe("string")
    expect(sessionID.length).toBeGreaterThan(0)

    // Verify the session exists and has the right title
    // Must use the voice project directory as Instance context
    const session = await Instance.provide({
      directory: VOICE_PROJECT_DIR,
      fn: () => Session.get(sessionID),
    })
    expect(session.title).toBe(BRAIN_SESSION_TITLE)
  })

  test("ensureBrainSession is idempotent — returns same session on second call", async () => {
    const first = await ensureBrainSession()
    const second = await ensureBrainSession()
    expect(first).toBe(second)
  })

  test("VOICE_PROJECT_DIR points to the expected path", () => {
    expect(VOICE_PROJECT_DIR).toContain("emberharmony")
    expect(VOICE_PROJECT_DIR).toContain("voice")
  })

  test("VOICE_CONFIG_DIR is inside the voice project directory", () => {
    expect(VOICE_CONFIG_DIR).toContain(VOICE_PROJECT_DIR)
    expect(VOICE_CONFIG_DIR).toContain(".emberharmony")
  })
})
