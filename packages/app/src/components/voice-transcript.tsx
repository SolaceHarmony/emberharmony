import { createMemo, Show } from "solid-js"
import { useLanguage } from "@/context/language"
import { useVoice } from "@/context/voice"

/**
 * Live transcript strip shown above the prompt input while voice mode is
 * active. Shows the most recent utterance from the lk.transcription text
 * stream — the user's speech as it is transcribed, and the agent's speech as
 * it is spoken. Full replies also land in the chat itself via the session
 * bridge; this is just the low-latency "what is being said right now" view.
 */
export function VoiceTranscript() {
  const voice = useVoice()
  const language = useLanguage()

  const latest = createMemo(() => {
    const streams = voice.transcriptions()
    if (streams.length === 0) return undefined
    const stream = streams[streams.length - 1]
    if (!stream.text.trim()) return undefined
    return {
      agent: stream.participantInfo?.identity?.startsWith("agent") ?? false,
      text: stream.text,
    }
  })

  return (
    <Show when={voice.state() === "connected" && latest()}>
      {(entry) => (
        <div class="flex items-baseline gap-2 px-4 py-1.5 text-12-regular text-text-weak overflow-hidden">
          <span class="text-text-strong shrink-0">
            {entry().agent ? language.t("voice.transcript.agent") : language.t("voice.transcript.you")}
          </span>
          <span class="truncate" aria-live="polite">
            {entry().text}
          </span>
        </div>
      )}
    </Show>
  )
}
