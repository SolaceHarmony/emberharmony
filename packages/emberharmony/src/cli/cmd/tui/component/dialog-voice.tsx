import { createMemo, createSignal, Show } from "solid-js"
import { useSync } from "@tui/context/sync"
import { useSDK } from "@tui/context/sdk"
import { DialogSelect } from "@tui/ui/dialog-select"
import { useDialog } from "@tui/ui/dialog"
import { useTheme } from "../context/theme"
import { TextAttributes } from "@opentui/core"
import type { VoiceConfigInfo } from "@thesolaceproject/emberharmony-sdk/v2"

/**
 * TUI voice settings dialog.
 *
 * Shows the current voice configuration — brain model, STT/TTS,
 * intent classifier, structured workflow, and LiveKit connection status.
 * Sub-dialogs handle model selection changes.
 */
export function DialogVoice() {
  const sync = useSync()
  const sdk = useSDK()
  const dialog = useDialog()
  const { theme } = useTheme()

  const voice = createMemo(() => sync.data.voice)

  async function refreshVoice() {
    const result = await sdk.client.voice.config()
    if (result.data) sync.set("voice", result.data)
  }

  const brainOptions = createMemo(() => {
    const providers = sync.data.provider
    const items: Array<{ title: string; value: string; description: string; category: string }> = []
    for (const p of providers) {
      for (const m of Object.values(p.models)) {
        items.push({
          title: m.name,
          value: `${p.id}/${m.id}`,
          description: p.name,
          category: p.name,
        })
      }
    }
    return items
  })

  return (
    <box paddingLeft={2} paddingRight={2} gap={1} paddingBottom={1}>
      <box flexDirection="row" justifyContent="space-between">
        <text fg={theme.text} attributes={TextAttributes.BOLD}>
          Voice settings
        </text>
        <text fg={theme.textMuted}>esc</text>
      </box>

      <Show when={voice()} fallback={<text fg={theme.textMuted}>Loading voice config...</text>}>
        <VoiceConfigView
          config={voice()!}
          brainOptions={brainOptions()}
          onSelectBrain={async (value: string | undefined) => {
            await sdk.client.voice.configUpdate({ voiceConfig: { brain: value } })
            await refreshVoice()
          }}
          onToggleStructured={async (structured: boolean) => {
            await sdk.client.voice.configUpdate({ voiceConfig: { structured } })
            await refreshVoice()
          }}
        />
      </Show>
    </box>
  )
}

function VoiceConfigView(props: {
  config: VoiceConfigInfo
  brainOptions: Array<{ title: string; value: string; description: string; category: string }>
  onSelectBrain: (value: string | undefined) => Promise<void>
  onToggleStructured: (structured: boolean) => Promise<void>
}) {
  const { theme } = useTheme()
  const dialog = useDialog()
  const sync = useSync()
  const cfg = createMemo(() => props.config)

  return (
    <box gap={1}>
      {/* Status */}
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>Status:</text>
        <text fg={cfg().available ? theme.success : theme.error}>
          {cfg().available ? "Connected" : "Not configured"}
        </text>
      </box>

      {/* Connection */}
      <text fg={theme.text} attributes={TextAttributes.BOLD}>
        Connection
      </text>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>URL:</text>
        <text fg={theme.text}>{cfg().url ?? "not set"}</text>
      </box>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>Credentials:</text>
        <text fg={cfg().credentials.livekit ? theme.success : theme.error}>
          {cfg().credentials.livekit ? "configured" : "missing"}
        </text>
      </box>

      {/* Models */}
      <text fg={theme.text} attributes={TextAttributes.BOLD}>
        Models
      </text>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>Brain:</text>
        <text fg={theme.text}>{cfg().brain ?? "default"}</text>
      </box>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>STT:</text>
        <text fg={theme.text}>{cfg().stt}</text>
      </box>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>TTS:</text>
        <text fg={theme.text}>{cfg().tts}</text>
      </box>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>Intent:</text>
        <text fg={theme.text}>{cfg().intent}</text>
      </box>

      {/* Workflow */}
      <text fg={theme.text} attributes={TextAttributes.BOLD}>
        Workflow
      </text>
      <box flexDirection="row" gap={1}>
        <text fg={theme.textMuted}>Structured:</text>
        <text fg={theme.text}>{cfg().structured ? "on (5-stage)" : "off (free-form)"}</text>
      </box>
    </box>
  )
}

/**
 * Opens a sub-dialog to change the brain model.
 */
export function openBrainModelSelect() {
  const sync = useSync()
  const dialog = useDialog()
  const sdk = useSDK()

  const options = createMemo(() => {
    const providers = sync.data.provider
    const items: Array<{ title: string; value: string; description: string; category: string }> = []
    for (const p of providers) {
      for (const m of Object.values(p.models)) {
        items.push({
          title: m.name,
          value: `${p.id}/${m.id}`,
          description: p.name,
          category: p.name,
        })
      }
    }
    return items
  })

  dialog.replace(() => (
    <DialogSelect
      title="Brain model"
      placeholder="Select LLM for voice brain..."
      options={options()}
      current={sync.data.voice?.brain ?? undefined}
      onSelect={async (option) => {
        await sdk.client.voice.configUpdate({ voiceConfig: { brain: option?.value } })
        const result = await sdk.client.voice.config()
        if (result.data) sync.set("voice", result.data)
      }}
    />
  ))
}

/**
 * Opens a sub-dialog to toggle the structured workflow.
 */
export function openStructuredToggle() {
  const sync = useSync()
  const dialog = useDialog()
  const sdk = useSDK()

  dialog.replace(() => (
    <DialogSelect
      title="Workflow mode"
      placeholder="Select workflow mode..."
      options={[
        { title: "Free-form", value: "off", description: "Brain flows naturally, plan/build only" },
        {
          title: "Structured (5-stage)",
          value: "on",
          description: "gathering → proposing → confirmed → executing → reviewing",
        },
      ]}
      current={sync.data.voice?.structured ? "on" : "off"}
      onSelect={async (option) => {
        await sdk.client.voice.configUpdate({ voiceConfig: { structured: option?.value === "on" } })
        const result = await sdk.client.voice.config()
        if (result.data) sync.set("voice", result.data)
      }}
    />
  ))
}
