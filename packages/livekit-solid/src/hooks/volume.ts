/**
 * Ported from @livekit/components-react `hooks/useTrackVolume.ts` (Apache-2.0):
 * useMultibandTrackVolume and its frequency normalization.
 */
import { type Accessor, createEffect, createSignal, onCleanup } from "solid-js"
import { isTrackReference, type TrackReferenceOrPlaceholder } from "@livekit/components-core"
import { Track, createAudioAnalyser } from "livekit-client"
import type { LocalAudioTrack, RemoteAudioTrack } from "livekit-client"

const normalizeFrequencies = (frequencies: Float32Array) => {
  const normalizeDb = (value: number) => {
    const minDb = -100
    const maxDb = -10
    let db = 1 - (Math.max(minDb, Math.min(maxDb, value)) * -1) / 100
    db = Math.sqrt(db)
    return db
  }
  return frequencies.map((value) => {
    if (value === -Infinity) return 0
    return normalizeDb(value)
  })
}

export interface MultiBandTrackVolumeOptions {
  bands?: number
  /** cut off of frequency bins on the lower end, relative to analyserOptions.fftSize */
  loPass?: number
  /** cut off of frequency bins on the higher end, relative to analyserOptions.fftSize */
  hiPass?: number
  /** update should run every x ms */
  updateInterval?: number
  analyserOptions?: AnalyserOptions
}

const multibandDefaults = {
  bands: 5,
  loPass: 100,
  hiPass: 600,
  updateInterval: 32,
  analyserOptions: { fftSize: 2048 },
} as const satisfies MultiBandTrackVolumeOptions

export function useMultibandTrackVolume(
  trackOrTrackReference: Accessor<LocalAudioTrack | RemoteAudioTrack | TrackReferenceOrPlaceholder | undefined>,
  options: MultiBandTrackVolumeOptions = {},
): Accessor<number[]> {
  const opts = { ...multibandDefaults, ...options }
  const [frequencyBands, setFrequencyBands] = createSignal<number[]>(new Array(opts.bands).fill(0))

  createEffect(() => {
    const source = trackOrTrackReference()
    const track =
      source instanceof Track
        ? source
        : source && isTrackReference(source)
          ? (source.publication.track as LocalAudioTrack | RemoteAudioTrack | undefined)
          : undefined
    if (!track || !track.mediaStream) {
      setFrequencyBands(new Array(opts.bands).fill(0))
      return
    }
    const { analyser, cleanup } = createAudioAnalyser(track, opts.analyserOptions)
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Float32Array(bufferLength)

    const updateVolume = () => {
      analyser.getFloatFrequencyData(dataArray)
      const frequencies = new Float32Array(dataArray).slice(opts.loPass, opts.hiPass)
      const normalizedFrequencies = normalizeFrequencies(frequencies)
      const totalBins = normalizedFrequencies.length
      const chunks: number[] = []
      for (let i = 0; i < opts.bands; i++) {
        const startIndex = Math.floor((i * totalBins) / opts.bands)
        const endIndex = Math.floor(((i + 1) * totalBins) / opts.bands)
        const chunk = normalizedFrequencies.slice(startIndex, endIndex)
        if (chunk.length === 0) {
          chunks.push(0)
        } else {
          let sum = 0
          for (const val of chunk) sum += val
          chunks.push(sum / chunk.length)
        }
      }
      setFrequencyBands(chunks)
    }

    const interval = setInterval(updateVolume, opts.updateInterval)
    onCleanup(() => {
      cleanup()
      clearInterval(interval)
    })
  })

  return frequencyBands
}
