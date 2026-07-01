import { describe, expect, test } from "bun:test"
import {
  voiceButtonOn,
  voiceEnabled,
  voiceMicTarget,
  voiceProvider,
  type VoiceNativeStatus,
} from "./voice-state"
import { defaultVoiceSettings, type VoicePlan, type VoiceProvider } from "./voice-settings"

function plan(provider: VoiceProvider, enabled = provider !== "off"): VoicePlan {
  return {
    provider,
    enabled,
    surface: provider === "off" ? "off" : "native",
    running: false,
    runningProvider: undefined,
    micEnabled: false,
    ready: provider !== "off",
    detail: "",
  }
}

function native(provider: VoiceProvider, stored = true): VoiceNativeStatus {
  return {
    plan: plan(provider),
    settings: { ...defaultVoiceSettings, provider },
    stored,
  }
}

describe("voice state decisions", () => {
  test("desktop uses explicit Tauri provider even when LiveKit is unavailable", () => {
    expect(voiceProvider(true, native("lfm2"), { available: false })).toBe("lfm2")
    expect(voiceEnabled(true, native("lfm2"), { available: false })).toBe(true)
  })

  test("desktop uses Tauri as the only provider authority", () => {
    expect(voiceProvider(true, native("off", false), { available: true })).toBe("off")
    expect(voiceEnabled(true, native("off", false), { available: true })).toBe(false)
    expect(voiceProvider(true, native("off", false), { available: false })).toBe("off")
    expect(voiceEnabled(true, native("off", false), { available: false })).toBe(false)
  })

  test("web still follows server LiveKit availability", () => {
    expect(voiceProvider(false, undefined, { available: false })).toBe("livekit")
    expect(voiceEnabled(false, undefined, { available: false })).toBe(false)
    expect(voiceEnabled(false, undefined, { available: true })).toBe(true)
  })

  test("mic affordance stays on for enabled providers before connection", () => {
    expect(voiceButtonOn("disconnected", true)).toBe(true)
    expect(voiceButtonOn("connected", false)).toBe(true)
    expect(voiceButtonOn("disconnected", false)).toBe(false)
  })

  test("connected voice pauses mic for typed input and busy prompt turns", () => {
    expect(voiceMicTarget("disconnected", false, false)).toBeUndefined()
    expect(voiceMicTarget("connected", false, false)).toBe(true)
    expect(voiceMicTarget("connected", true, false)).toBe(false)
    expect(voiceMicTarget("connected", false, true)).toBe(false)
  })
})
