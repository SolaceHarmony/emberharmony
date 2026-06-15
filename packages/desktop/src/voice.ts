import { invoke } from "@tauri-apps/api/core"
import { listen } from "@tauri-apps/api/event"
import type { VoiceAdapter, VoiceState } from "@thesolaceproject/emberharmony-app"

const VOICE_STATE_EVENT = "voice://state-changed"

export const voiceAdapter: VoiceAdapter = {
  async connect(url: string, token: string): Promise<VoiceState> {
    return invoke<VoiceState>("voice_connect", { url, token })
  },

  async disconnect(): Promise<VoiceState> {
    return invoke<VoiceState>("voice_disconnect")
  },

  async toggleMute(): Promise<boolean> {
    return invoke<boolean>("voice_toggle_mute")
  },

  async getState(): Promise<VoiceState> {
    return invoke<VoiceState>("voice_state")
  },

  onStateChange(callback: (state: VoiceState) => void): () => void {
    let unlisten: (() => void) | undefined
    listen<VoiceState>(VOICE_STATE_EVENT, (event) => {
      callback(event.payload)
    }).then((fn) => {
      unlisten = fn
    })
    return () => {
      unlisten?.()
    }
  },
}
