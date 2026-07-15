import { describe, expect, test } from "bun:test"
import { VOICE_CONTROL_INTERRUPT, VOICE_CONTROL_TOPIC } from "../../src/voice/constants"
import { parseVoiceControl } from "../../src/voice/control"

const encoder = new TextEncoder()

describe("voice control packets", () => {
  test("accepts the desktop interrupt command on the voice control topic", () => {
    const payload = encoder.encode(JSON.stringify({ type: VOICE_CONTROL_INTERRUPT }))

    expect(parseVoiceControl(VOICE_CONTROL_TOPIC, payload)).toBe(VOICE_CONTROL_INTERRUPT)
  })

  test("ignores unrelated topics and malformed payloads", () => {
    expect(parseVoiceControl("other", encoder.encode(JSON.stringify({ type: VOICE_CONTROL_INTERRUPT })))).toBeUndefined()
    expect(parseVoiceControl(VOICE_CONTROL_TOPIC, encoder.encode("{"))).toBeUndefined()
    expect(parseVoiceControl(VOICE_CONTROL_TOPIC, encoder.encode(JSON.stringify({ type: "stop" })))).toBeUndefined()
  })
})
