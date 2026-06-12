/**
 * Ported from @livekit/components-react `components/participant/BarVisualizer.tsx` (Apache-2.0).
 * Keeps the upstream `lk-audio-bar-visualizer` / `lk-audio-bar` class names and
 * data attributes so upstream styling recipes apply.
 */
import { createMemo, Index, splitProps, type JSX } from "solid-js"
import type { TrackReferenceOrPlaceholder } from "@livekit/components-core"
import type { LocalAudioTrack, RemoteAudioTrack } from "livekit-client"
import { useBarAnimator } from "../hooks/bar-animator"
import { useMultibandTrackVolume } from "../hooks/volume"
import type { AgentState } from "../hooks/voice-assistant"

export type BarVisualizerOptions = {
  /** in percentage */
  maxHeight?: number
  /** in percentage */
  minHeight?: number
}

export interface BarVisualizerProps extends JSX.HTMLAttributes<HTMLDivElement> {
  /** If set, the visualizer will transition between different voice assistant states */
  state?: AgentState
  /** Number of bars that show up in the visualizer */
  barCount?: number
  track?: TrackReferenceOrPlaceholder | LocalAudioTrack | RemoteAudioTrack
  options?: BarVisualizerOptions
}

const sequencerIntervals = new Map<AgentState, number>([
  ["connecting", 2000],
  ["initializing", 2000],
  ["listening", 500],
  ["thinking", 150],
])

const getSequencerInterval = (state: AgentState | undefined, barCount: number): number | undefined => {
  if (state === undefined) return 1000
  let interval = sequencerIntervals.get(state)
  if (interval && state === "connecting") interval /= barCount
  return interval
}

/**
 * Visualizes audio signals from a track as bars. If the `state` prop is set,
 * it automatically transitions between voice assistant states.
 */
export function BarVisualizer(props: BarVisualizerProps) {
  const [local, rest] = splitProps(props, ["state", "barCount", "track", "options", "class"])
  const barCount = () => local.barCount ?? 15

  const volumeBands = useMultibandTrackVolume(() => local.track, {
    bands: barCount(),
    loPass: 100,
    hiPass: 200,
  })
  const minHeight = () => local.options?.minHeight ?? 20
  const maxHeight = () => local.options?.maxHeight ?? 100

  const highlightedIndices = useBarAnimator(
    () => local.state,
    barCount,
    () => getSequencerInterval(local.state, barCount()) ?? 100,
  )

  const highlighted = createMemo(() => new Set(highlightedIndices()))

  return (
    <div
      {...rest}
      class={`lk-audio-bar-visualizer${local.class ? ` ${local.class}` : ""}`}
      data-lk-va-state={local.state}
    >
      <Index each={volumeBands()}>
        {(volume, idx) => (
          <span
            data-lk-highlighted={highlighted().has(idx)}
            data-lk-bar-index={idx}
            classList={{ "lk-audio-bar": true, "lk-highlighted": highlighted().has(idx) }}
            style={{
              height: `${Math.min(maxHeight(), Math.max(minHeight(), volume() * 100 + 5))}%`,
            }}
          />
        )}
      </Index>
    </div>
  )
}
