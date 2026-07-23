/**
 * Curated registry of LiveKit Inference voice integrations (tier 1: routed
 * through the LiveKit gateway with a single LiveKit credential, no
 * per-provider API keys). Model ids are persisted configuration values passed
 * to the configured LiveKit service.
 */
export namespace VoiceRegistry {
  export interface Option {
    /** Gateway model string, without the ":<language|voice>" suffix */
    id: string
    /** Display name for the model */
    name: string
    /** Provider display name */
    provider: string
    /** Default ":<suffix>" — language for STT, voice ID for TTS */
    defaultSuffix?: string
  }

  export const STT: Option[] = [
    { id: "deepgram/nova-3", name: "Nova 3", provider: "Deepgram", defaultSuffix: "multi" },
    { id: "deepgram/flux-general-multi", name: "Flux General (multilingual)", provider: "Deepgram" },
    {
      id: "assemblyai/universal-streaming-multilingual",
      name: "Universal Streaming (multilingual)",
      provider: "AssemblyAI",
    },
    { id: "cartesia/ink-whisper", name: "Ink Whisper", provider: "Cartesia" },
    { id: "elevenlabs/scribe_v2_realtime", name: "Scribe v2 Realtime", provider: "ElevenLabs" },
  ]

  export const TTS: Option[] = [
    {
      id: "cartesia/sonic-3",
      name: "Sonic 3",
      provider: "Cartesia",
      defaultSuffix: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    },
    { id: "elevenlabs/eleven_turbo_v2_5", name: "Eleven Turbo v2.5", provider: "ElevenLabs" },
    { id: "inworld/inworld-tts-1", name: "Inworld TTS 1", provider: "Inworld" },
    { id: "rime/mistv2", name: "Mist v2", provider: "Rime" },
    { id: "deepgram/aura-2", name: "Aura 2", provider: "Deepgram" },
  ]

  function withSuffix(option: Option) {
    return option.defaultSuffix ? `${option.id}:${option.defaultSuffix}` : option.id
  }

  export const DEFAULT_STT = withSuffix(STT[0]!)
  export const DEFAULT_TTS = withSuffix(TTS[0]!)
  /** Small fast gateway model for the plan/build voice workflow */
  export const DEFAULT_INTENT = "openai/gpt-5.4-nano"
}
