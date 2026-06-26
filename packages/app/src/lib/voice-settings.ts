// Typed wrapper over the Tauri `voice_settings_*` commands (src-tauri/src/settings.rs).
//
// The native voice layer — the provider switch and the local LFM2-Audio config —
// is persisted in the Tauri settings store and read natively (in-process) by the
// Rust voice loop. The webview reaches it through the injected `window.__TAURI__`
// global, the same pattern titlebar.tsx uses, guarded so that in the standalone
// web build (no Tauri runtime) these calls no-op to defaults.

export type VoiceProvider = "off" | "lfm2" | "livekit"
export type Lfm2Device = "cpu" | "metal"

export interface LiveKitSettings {
  url?: string
  stt?: string
  tts?: string
  intent?: string
}

export interface DelegateSettings {
  enabled: boolean
  target?: string
}

export interface Lfm2Settings {
  modelDir?: string
  device: Lfm2Device
  vadThreshold: number
  maxTokens: number
  model?: string
  seed?: number
  delegate: DelegateSettings
}

export interface VoiceSettings {
  provider: VoiceProvider
  livekit: LiveKitSettings
  lfm2: Lfm2Settings
}

export const defaultVoiceSettings: VoiceSettings = {
  provider: "off",
  livekit: {},
  lfm2: { device: "cpu", vadThreshold: 0.012, maxTokens: 512, delegate: { enabled: false } },
}

type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>

function tauriInvoke(): Invoke | undefined {
  return (window as unknown as { __TAURI__?: { core?: { invoke?: Invoke } } }).__TAURI__?.core?.invoke
}

/** True when running inside the Tauri desktop shell (native voice available). */
export function isDesktop(): boolean {
  return tauriInvoke() !== undefined
}

/** Read voice settings. In the web build (no Tauri) returns defaults. */
export async function getVoiceSettings(): Promise<VoiceSettings> {
  const invoke = tauriInvoke()
  if (!invoke) return defaultVoiceSettings
  return invoke<VoiceSettings>("voice_settings_get")
}

/** Persist the whole voice settings object. No-op in the web build. */
export async function setVoiceSettings(settings: VoiceSettings): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_settings_set", { settings })
}

/** Readiness of the active provider — mirrors the Rust `VoicePlan`. */
export interface VoicePlan {
  provider: VoiceProvider
  ready: boolean
  detail: string
}

/** Whether the configured voice provider is ready to start (native side). */
export async function getVoiceStatus(): Promise<VoicePlan> {
  const invoke = tauriInvoke()
  if (!invoke) return { provider: "off", ready: false, detail: "" }
  return invoke<VoicePlan>("voice_status")
}
