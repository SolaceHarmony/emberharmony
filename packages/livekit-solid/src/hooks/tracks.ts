/**
 * Ported from @livekit/components-react hooks (Apache-2.0):
 * useTracks (plain Track.Source[] form, no placeholder support),
 * useTrackTranscription, useTextStream, useTranscriptions.
 */
import { type Accessor, createMemo, createSignal, createEffect, onCleanup } from "solid-js"
import {
  addMediaTimestampToTranscription,
  dedupeSegments,
  DataTopic,
  ParticipantAgentAttributes,
  setupTextStream,
  trackReferencesObservable,
  trackSyncTimeObserver,
  trackTranscriptionObserver,
  type ReceivedTranscriptionSegment,
  type TextStreamData,
  type TrackReference,
  type TrackReferenceOrPlaceholder,
} from "@livekit/components-core"
import { ConnectionState, type Room, type RoomEvent, type Track, type TranscriptionSegment } from "livekit-client"
import { map } from "rxjs"
import { useEnsureRoom } from "../context"
import { observableState } from "../observable"
import { useConnectionState } from "./participants"

export interface UseTracksOptions {
  updateOnlyOn?: RoomEvent[]
  onlySubscribed?: boolean
  room?: Room
}

export function useTracks(sources: Track.Source[], options: UseTracksOptions = {}): Accessor<TrackReference[]> {
  const room = useEnsureRoom(options.room)
  const bundle = observableState(
    () =>
      trackReferencesObservable(room, sources, {
        additionalRoomEvents: options.updateOnlyOn,
        onlySubscribed: options.onlySubscribed,
      }),
    () => ({ trackReferences: [] as TrackReference[], participants: [] }),
  )
  return () => bundle().trackReferences
}

function useTrackSyncTime(ref: Accessor<TrackReferenceOrPlaceholder | undefined>) {
  return observableState(
    () => {
      const track = ref()?.publication?.track
      return track
        ? trackSyncTimeObserver(track).pipe(map((timestamp) => ({ timestamp, rtpTimestamp: track.rtpTimestamp })))
        : undefined
    },
    () => ({
      timestamp: Date.now(),
      rtpTimestamp: ref()?.publication?.track?.rtpTimestamp,
    }),
  )
}

export interface TrackTranscriptionOptions {
  /** how many transcription segments should be buffered in state */
  bufferSize?: number
  /** optional callback for retrieving newly incoming transcriptions only */
  onTranscription?: (newSegments: TranscriptionSegment[]) => void
}

const TRACK_TRANSCRIPTION_DEFAULTS = {
  bufferSize: 100,
} as const satisfies TrackTranscriptionOptions

export function useTrackTranscription(
  trackRef: Accessor<TrackReferenceOrPlaceholder | undefined>,
  options?: TrackTranscriptionOptions,
) {
  const opts = { ...TRACK_TRANSCRIPTION_DEFAULTS, ...options }
  const [segments, setSegments] = createSignal<ReceivedTranscriptionSegment[]>([])
  const syncTimestamps = useTrackSyncTime(trackRef)

  createEffect(() => {
    const publication = trackRef()?.publication
    setSegments([])
    if (!publication) return
    const subscription = trackTranscriptionObserver(publication).subscribe(([newSegments]) => {
      opts.onTranscription?.(newSegments)
      setSegments((prev) =>
        dedupeSegments(
          prev,
          newSegments.map((s) => addMediaTimestampToTranscription(s, syncTimestamps())),
          opts.bufferSize,
        ),
      )
    })
    onCleanup(() => subscription.unsubscribe())
  })

  return { segments }
}

export function useTextStream(topic: string, room?: Room): Accessor<TextStreamData[]> {
  const r = useEnsureRoom(room)
  const connectionState = useConnectionState(r)
  return observableState(
    () => (connectionState() === ConnectionState.Disconnected ? undefined : setupTextStream(r, topic)),
    () => [] as TextStreamData[],
  )
}

export interface UseTranscriptionsOptions {
  room?: Room
  participantIdentities?: Accessor<string[] | undefined>
  trackSids?: Accessor<string[] | undefined>
}

export function useTranscriptions(opts: UseTranscriptionsOptions = {}): Accessor<TextStreamData[]> {
  const textStreams = useTextStream(DataTopic.TRANSCRIPTION, opts.room)
  return createMemo(() => {
    const identities = opts.participantIdentities?.()
    const sids = opts.trackSids?.()
    return textStreams()
      .filter((stream) => (identities ? identities.includes(stream.participantInfo.identity) : true))
      .filter((stream) =>
        sids
          ? sids.includes(stream.streamInfo.attributes?.[ParticipantAgentAttributes.TranscribedTrackId] ?? "")
          : true,
      )
  })
}
