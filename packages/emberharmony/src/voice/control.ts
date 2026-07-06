import { VOICE_CONTROL_INTERRUPT, VOICE_CONTROL_TOPIC } from "./constants"

const decoder = new TextDecoder()

export type VoiceControlCommand = typeof VOICE_CONTROL_INTERRUPT

export function parseVoiceControl(topic: string | undefined, payload: Uint8Array): VoiceControlCommand | undefined {
  if (topic !== VOICE_CONTROL_TOPIC) return undefined
  try {
    const data = JSON.parse(decoder.decode(payload)) as unknown
    const type = data && typeof data === "object" && "type" in data ? data.type : undefined
    if (type === VOICE_CONTROL_INTERRUPT) return VOICE_CONTROL_INTERRUPT
    return undefined
  } catch {
    return undefined
  }
}
