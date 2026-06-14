import { test, expect, describe, beforeEach, afterEach } from "bun:test"
import { Voice } from "../../src/voice/token"
import { Instance } from "../../src/project/instance"
import { tmpdir } from "../fixture/fixture"

describe("Voice.settings", () => {
  test("marks unavailable when credentials are missing", async () => {
    // Don't set any LiveKit env vars — the test preload clears provider keys,
    // so Voice.settings() should report unavailable.
    await using tmp = await tmpdir()
    const settings = await Instance.provide({
      directory: tmp.path,
      fn: () => Voice.settings(),
    })

    expect(settings.available).toBe(false)
    expect(settings.url).toBeUndefined()
  })

  test("uses default model values regardless of config", async () => {
    await using tmp = await tmpdir()
    const settings = await Instance.provide({
      directory: tmp.path,
      fn: () => Voice.settings(),
    })

    // Default models are always set (from VoiceRegistry constants)
    expect(settings.stt).toBe("deepgram/nova-3:multi")
    expect(settings.tts).toBe("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    expect(settings.intent).toBe("openai/gpt-5.4-nano")
  })

  test("token throws VoiceNotConfiguredError when voice is not available", async () => {
    await using tmp = await tmpdir()
    await expect(
      Instance.provide({
        directory: tmp.path,
        fn: () =>
          Voice.token({
            roomName: "emberharmony_test",
            identity: "user_test",
          }),
      }),
    ).rejects.toThrow()
  })

  test("settings override resolves model from config", async () => {
    await using tmp = await tmpdir()
    const settings = await Instance.provide({
      directory: tmp.path,
      fn: () =>
        Voice.settings({
          stt: "deepgram/nova-2",
          tts: "elevenlabs/turbo-v2",
          intent: "anthropic/claude-3-haiku",
        }),
    })

    expect(settings.stt).toBe("deepgram/nova-2")
    expect(settings.tts).toBe("elevenlabs/turbo-v2")
    expect(settings.intent).toBe("anthropic/claude-3-haiku")
  })
})
