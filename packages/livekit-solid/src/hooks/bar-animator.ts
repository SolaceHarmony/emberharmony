/**
 * Ported from @livekit/components-react (Apache-2.0):
 * `components/participant/animators/useBarAnimator.ts` and
 * `components/participant/animationSequences/*.ts`.
 */
import { type Accessor, createEffect, createMemo, createSignal, onCleanup } from "solid-js"
import type { AgentState } from "./voice-assistant"

export const generateConnectingSequenceBar = (columns: number): number[][] => {
  const seq = []
  for (let x = 0; x < columns; x++) {
    seq.push([x, columns - 1 - x])
  }
  return seq
}

export const generateListeningSequenceBar = (columns: number): number[][] => {
  const center = Math.floor(columns / 2)
  const noIndex = -1
  return [[center], [noIndex]]
}

export function useBarAnimator(
  state: Accessor<AgentState | undefined>,
  columns: Accessor<number>,
  interval: Accessor<number>,
): Accessor<number[]> {
  const [index, setIndex] = createSignal(0)

  const sequence = createMemo<number[][]>(() => {
    const s = state()
    const cols = columns()
    setIndex(0)
    if (s === "thinking" || s === "listening") return generateListeningSequenceBar(cols)
    if (s === "connecting" || s === "initializing") return [...generateConnectingSequenceBar(cols)]
    if (s === undefined || s === "speaking") return [new Array(cols).fill(0).map((_, idx) => idx)]
    return [[]]
  })

  createEffect(() => {
    const ms = interval()
    let startTime = performance.now()
    let frame: number
    const animate = (time: DOMHighResTimeStamp) => {
      if (time - startTime >= ms) {
        setIndex((prev) => prev + 1)
        startTime = time
      }
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    onCleanup(() => cancelAnimationFrame(frame))
  })

  return () => sequence()[index() % sequence().length] ?? []
}
