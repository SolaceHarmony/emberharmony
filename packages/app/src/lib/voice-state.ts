import type { VoicePlan, VoiceProvider, VoiceSettings } from "./voice-settings"

export type VoiceServerStatus = { available?: boolean } | undefined

export type VoiceNativeStatus = {
  plan: VoicePlan
  settings: VoiceSettings
  stored: boolean
}

export type VoiceConnectionState = "disconnected" | "connecting" | "connected" | "error"

export function voiceProvider(
  desktop: boolean,
  current: VoiceNativeStatus | undefined,
  server: VoiceServerStatus,
): VoiceProvider {
  if (!desktop) return "livekit"
  if (current?.stored === false && server?.available) return "livekit"
  return current?.plan.provider ?? "off"
}

export function voiceEnabled(
  desktop: boolean,
  current: VoiceNativeStatus | undefined,
  server: VoiceServerStatus,
): boolean {
  if (!desktop) return server?.available === true
  if (current?.stored === false && server?.available) return true
  return current?.plan.enabled ?? voiceProvider(desktop, current, server) !== "off"
}

export function voiceButtonOn(state: VoiceConnectionState, enabled: boolean): boolean {
  return state === "connected" || enabled
}

export function voiceMicTarget(state: VoiceConnectionState, dirty: boolean, busy: boolean): boolean | undefined {
  if (state !== "connected") return undefined
  return !dirty && !busy
}

export function shouldStopRuntimeForProviderChange(running: VoiceProvider | undefined, next: VoiceSettings): boolean {
  return running !== undefined && next.provider !== running
}
