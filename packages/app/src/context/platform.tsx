import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { AsyncStorage, SyncStorage } from "@solid-primitives/storage"

export type VoiceState = {
  connected: boolean
  room: string | null
  agentStage: string | null
  agentMode: string | null
  micMuted: boolean
}

export type VoiceAdapter = {
  connect(url: string, token: string): Promise<VoiceState>
  disconnect(): Promise<VoiceState>
  toggleMute(): Promise<boolean>
  getState(): Promise<VoiceState>
  onStateChange(callback: (state: VoiceState) => void): () => void
}

export type Platform = {
  /** Platform discriminator */
  platform: "web" | "desktop"

  /** Desktop OS (Tauri only) */
  os?: "macos" | "windows" | "linux"

  /** App version */
  version?: string

  /** Open a URL in the default browser */
  openLink(url: string): void

  /** Restart the app  */
  restart(): Promise<void>

  /** Navigate back in history */
  back(): void

  /** Navigate forward in history */
  forward(): void

  /** Send a system notification (optional deep link) */
  notify(title: string, description?: string, href?: string): Promise<void>

  /** Open directory picker dialog (native on Tauri, server-backed on web) */
  openDirectoryPickerDialog?(opts?: { title?: string; multiple?: boolean }): Promise<string | string[] | null>

  /** Open native file picker dialog (Tauri only) */
  openFilePickerDialog?(opts?: { title?: string; multiple?: boolean }): Promise<string | string[] | null>

  /** Save file picker dialog (Tauri only) */
  saveFilePickerDialog?(opts?: { title?: string; defaultPath?: string }): Promise<string | null>

  /** Storage mechanism, defaults to localStorage */
  storage?: (name?: string) => SyncStorage | AsyncStorage

  /** Check for updates (Tauri only) */
  checkUpdate?(): Promise<{ updateAvailable: boolean; version?: string }>

  /** Install updates (Tauri only) */
  update?(): Promise<void>

  /** Fetch override */
  fetch?: typeof fetch

  /** Get the configured default server URL (platform-specific) */
  getDefaultServerUrl?(): Promise<string | null> | string | null

  /** Set the default server URL to use on app startup (platform-specific) */
  setDefaultServerUrl?(url: string | null): Promise<void> | void

  /** Parse markdown to HTML using native parser (desktop only, returns unprocessed code blocks) */
  parseMarkdown?(markdown: string): Promise<string>

  /** Native voice adapter (desktop only). When present, VoiceProvider uses this
   *  instead of livekit-client WebRTC for audio transport. */
  voice?: VoiceAdapter
}

export const { use: usePlatform, provider: PlatformProvider } = createSimpleContext({
  name: "Platform",
  init: (props: { value: Platform }) => {
    return props.value
  },
})
