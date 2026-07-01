// Typed wrapper over the Tauri `voice_settings_*` commands (src-tauri/src/settings.rs).
//
// The native voice layer — the provider switch and the local LFM2-Audio config —
// is persisted in the Tauri settings store and read natively (in-process) by the
// Rust voice loop. The webview reaches it through the injected `window.__TAURI__`
// global, the same pattern titlebar.tsx uses, guarded so that in the standalone
// web build (no Tauri runtime) these calls no-op to defaults.

export type VoiceProvider = "off" | "lfm2" | "livekit"
export type VoiceSurface = "off" | "native" | "livekit"
export type Lfm2Device = "cpu" | "metal"
export const VOICE_SETTINGS_CHANGED = "emberharmony:voice-settings-changed"

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
  /** Download-source revision (branch/tag/commit); ignored once modelDir is set. */
  revision?: string
  seed?: number
  delegate: DelegateSettings
}

export interface VoiceSettings {
  provider: VoiceProvider
  lastProvider?: Exclude<VoiceProvider, "off">
  livekit: LiveKitSettings
  lfm2: Lfm2Settings
}

export interface VoiceSettingsState {
  settings: VoiceSettings
  stored: boolean
}

export type VoiceSettingsChangedEvent = CustomEvent<VoiceSettings | undefined>

export const defaultVoiceSettings: VoiceSettings = {
  provider: "off",
  lastProvider: "lfm2",
  livekit: {},
  // Desktop resolves the default `model` in Rust; this literal is only the
  // web-build display fallback when no Tauri runtime exists.
  lfm2: {
    model: "LiquidAI/LFM2.5-Audio-1.5B",
    device: "cpu",
    vadThreshold: 0.012,
    maxTokens: 512,
    delegate: { enabled: false },
  },
}

type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>
type ChannelCtor = new <T = unknown>(onmessage: (response: T) => void) => unknown
type TauriCore = { invoke?: Invoke; Channel?: ChannelCtor }

function tauriCore(): TauriCore | undefined {
  return (window as unknown as { __TAURI__?: { core?: TauriCore } }).__TAURI__?.core
}

function tauriInvoke(): Invoke | undefined {
  return tauriCore()?.invoke
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

/** Read voice settings plus whether the native store had an explicit value. */
export async function getVoiceSettingsState(): Promise<VoiceSettingsState> {
  const invoke = tauriInvoke()
  if (!invoke) return { settings: defaultVoiceSettings, stored: false }
  return invoke<VoiceSettingsState>("voice_settings_state")
}

/** Persist the whole voice settings object. No-op in the web build. */
export async function setVoiceSettings(settings: VoiceSettings): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_settings_set", { settings })
  window.dispatchEvent(new CustomEvent(VOICE_SETTINGS_CHANGED, { detail: settings }))
}

/** Readiness of the active provider — mirrors the Rust `VoicePlan`. */
export interface VoicePlan {
  provider: VoiceProvider
  enabled: boolean
  surface: VoiceSurface
  running: boolean
  runningProvider?: VoiceProvider
  micEnabled: boolean
  ready: boolean
  detail: string
}

/** Whether the configured voice provider is ready to start (native side). */
export async function getVoiceStatus(): Promise<VoicePlan> {
  const invoke = tauriInvoke()
  if (!invoke)
    return {
      provider: "off",
      enabled: false,
      surface: "off",
      running: false,
      runningProvider: undefined,
      micEnabled: false,
      ready: false,
      detail: "",
    }
  return invoke<VoicePlan>("voice_status")
}

export type NativeVoiceState = "loading" | "idle" | "listening" | "thinking" | "speaking"

export type NativeVoiceEvent =
  | { type: "state"; state: NativeVoiceState }
  | { type: "transcript"; role: "user" | "assistant"; text: string }
  | { type: "level"; rms: number }
  | { type: "audioClip"; wav: number[]; ms: number }
  | { type: "ended"; reason?: string }
  | { type: "error"; message: string }

export interface VoiceStartContext {
  sessionID: string
  directory: string
  agent?: string
  model?: {
    providerID: string
    modelID: string
  }
  variant?: string
  promptMode?: "plan" | "build"
}

export type VoiceStartResult = { provider: "lfm2" } | { provider: "livekit" }

/** Start the native desktop voice service. */
export async function startVoice(
  ctx: VoiceStartContext,
  onEvent: (event: NativeVoiceEvent) => void,
): Promise<VoiceStartResult> {
  const core = tauriCore()
  if (!core?.invoke || !core.Channel) throw new Error("Native voice is unavailable.")
  const Channel = core.Channel
  const channel = new Channel(onEvent)
  return core.invoke<VoiceStartResult>("voice_start", { ctx, channel })
}

/** Stop the native desktop voice service. */
export async function stopVoice(): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_stop")
}

/** Interrupt native speech/playback without ending the voice session. */
export async function interruptVoice(): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_interrupt")
}

/** Pause/resume native microphone capture without ending the voice session. */
export async function setVoiceMicEnabled(enabled: boolean): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_set_mic_enabled", { enabled })
}

/** Pause native microphone capture and interrupt voice before a typed prompt runs. */
export async function beginVoiceTypedInput(): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_begin_typed_input")
}

export interface LiveKitCredentialsStatus {
  stored: boolean
}

/** Store or clear desktop LiveKit API credentials in the native OS keychain. */
export async function setLiveKitCredentials(apiKey: string, apiSecret: string): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_livekit_credentials_set", { apiKey, apiSecret })
  window.dispatchEvent(new CustomEvent(VOICE_SETTINGS_CHANGED, { detail: undefined }))
}

/** Whether desktop LiveKit API credentials are stored natively. */
export async function getLiveKitCredentialsStatus(): Promise<LiveKitCredentialsStatus> {
  const invoke = tauriInvoke()
  if (!invoke) return { stored: false }
  return invoke<LiveKitCredentialsStatus>("voice_livekit_credentials_status")
}

// ---- model management (download / local dir / HF token) ----

export type NativeDownloadEvent =
  | { type: "started"; total: number }
  | { type: "file"; index: number; total: number; name: string }
  | { type: "done"; dir: string }
  | { type: "error"; message: string }

/**
 * Download an LFM2-Audio model snapshot (repo id or pasted HF URL + optional revision),
 * streaming per-file progress over a Channel. The terminal `done`/`error` event is
 * authoritative; on `done` the caller persists `dir` as the active `modelDir`. The HF
 * token is read natively from the keychain and never passed from here.
 */
export async function downloadVoiceModel(
  args: { source: string; revision?: string },
  onEvent: (event: NativeDownloadEvent) => void,
): Promise<void> {
  const core = tauriCore()
  if (!core?.invoke || !core.Channel) throw new Error("Native model download is unavailable.")
  const Channel = core.Channel
  const channel = new Channel(onEvent)
  await core.invoke<void>("voice_model_download", {
    source: args.source,
    revision: args.revision,
    channel,
  })
}

/** Native folder picker for a local model snapshot directory (undefined if cancelled). */
export async function pickModelDir(): Promise<string | undefined> {
  const invoke = tauriInvoke()
  if (!invoke) return undefined
  return (await invoke<string | null>("voice_pick_model_dir")) ?? undefined
}

/** Whether a Hugging Face token is stored in the OS keychain (presence only). */
export async function getHfTokenStatus(): Promise<boolean> {
  const invoke = tauriInvoke()
  if (!invoke) return false
  return invoke<boolean>("voice_hf_token_status")
}

/** Store (non-empty) or clear (empty) the Hugging Face token in the OS keychain. */
export async function setHfToken(token: string): Promise<void> {
  const invoke = tauriInvoke()
  if (!invoke) return
  await invoke<void>("voice_hf_token_set", { token })
}
