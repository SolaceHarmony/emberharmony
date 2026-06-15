import { createMemo } from "solid-js"
import { useSync } from "@tui/context/sync"
import { useSDK } from "@tui/context/sdk"
import { DialogSelect } from "@tui/ui/dialog-select"
import { useDialog } from "@tui/ui/dialog"
import { pipe, flatMap, entries, sortBy } from "remeda"

/**
 * TUI voice settings dialog.
 *
 * Selectable options for brain model, workflow mode, and status info.
 * Selecting "Change brain model" or "Toggle structured" opens a sub-dialog.
 */
export function DialogVoice() {
  const sync = useSync()
  const sdk = useSDK()
  const dialog = useDialog()

  async function refreshVoice() {
    const result = await sdk.client.voice.config()
    if (result.data) sync.set("voice", result.data)
  }

  const options = createMemo(() => {
    const voice = sync.data.voice
    if (!voice) return []

    const items = []

    // Status section
    items.push({
      value: "status",
      title: voice.available ? "Connected" : "Not configured",
      description: voice.url ?? undefined,
      category: "Status",
      onSelect: () => {
        dialog.clear()
      },
    })

    // Brain model
    items.push({
      value: "brain",
      title: voice.brain ?? "default",
      category: "Brain model",
      onSelect: () => {
        dialog.replace(() => <DialogBrainModelSelect />)
      },
    })

    // Workflow
    items.push({
      value: "workflow",
      title: voice.structured ? "Structured (5-stage)" : "Free-form",
      category: "Workflow",
      onSelect: () => {
        dialog.replace(() => <DialogStructuredToggle />)
      },
    })

    // Info lines (non-selectable context)
    items.push({
      value: "stt",
      title: voice.stt,
      category: "STT",
      onSelect: () => {
        dialog.clear()
      },
    })
    items.push({
      value: "tts",
      title: voice.tts,
      category: "TTS",
      onSelect: () => {
        dialog.clear()
      },
    })
    if (voice.intent) {
      items.push({
        value: "intent",
        title: voice.intent,
        category: "Intent classifier",
        onSelect: () => {
          dialog.clear()
        },
      })
    }
    items.push({
      value: "credentials",
      title: voice.credentials.livekit ? "configured" : "missing",
      category: "LiveKit credentials",
      onSelect: () => {
        dialog.clear()
      },
    })

    return items
  })

  return (
    <DialogSelect
      title="Voice settings"
      placeholder="Search voice settings..."
      options={options()}
      onSelect={(option) => option.onSelect?.(dialog)}
    />
  )
}

/**
 * Sub-dialog to pick a brain model from the provider/model list.
 */
function DialogBrainModelSelect() {
  const sync = useSync()
  const sdk = useSDK()
  const dialog = useDialog()

  async function refreshVoice() {
    const result = await sdk.client.voice.config()
    if (result.data) sync.set("voice", result.data)
  }

  const options = createMemo(() =>
    pipe(
      sync.data.provider,
      sortBy(
        (p) => p.id !== "emberharmony",
        (p) => p.name,
      ),
      flatMap((p) =>
        pipe(
          p.models,
          entries(),
          flatMap(([modelID, info]) => [
            {
              value: `${p.id}/${modelID}`,
              title: info.name ?? modelID,
              description: p.name,
              category: p.name,
              onSelect: async () => {
                await sdk.client.voice.configUpdate({ voiceConfig: { brain: `${p.id}/${modelID}` } })
                await refreshVoice()
                dialog.clear()
              },
            },
          ]),
        ),
      ),
    ),
  )

  return (
    <DialogSelect
      title="Brain model"
      placeholder="Select LLM for voice brain..."
      current={sync.data.voice?.brain ?? undefined}
      options={options()}
    />
  )
}

/**
 * Sub-dialog to toggle the structured workflow mode.
 */
function DialogStructuredToggle() {
  const sync = useSync()
  const sdk = useSDK()
  const dialog = useDialog()

  async function refreshVoice() {
    const result = await sdk.client.voice.config()
    if (result.data) sync.set("voice", result.data)
  }

  const current = createMemo(() => (sync.data.voice?.structured ? "on" : "off"))

  return (
    <DialogSelect
      title="Workflow mode"
      placeholder="Select workflow mode..."
      current={current()}
      options={[
        {
          value: "off",
          title: "Free-form",
          description: "Brain flows naturally, plan/build only",
          onSelect: async () => {
            await sdk.client.voice.configUpdate({ voiceConfig: { structured: false } })
            await refreshVoice()
            dialog.clear()
          },
        },
        {
          value: "on",
          title: "Structured (5-stage)",
          description: "gathering → proposing → confirmed → executing → reviewing",
          onSelect: async () => {
            await sdk.client.voice.configUpdate({ voiceConfig: { structured: true } })
            await refreshVoice()
            dialog.clear()
          },
        },
      ]}
    />
  )
}
