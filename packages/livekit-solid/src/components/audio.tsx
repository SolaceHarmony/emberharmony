/**
 * Ported from @livekit/components-react (Apache-2.0):
 * `components/RoomAudioRenderer.tsx` and `components/participant/AudioTrack.tsx`.
 * The React useMediaTrackBySourceOrName indirection is collapsed into a direct
 * setupMediaTrack subscription + attach/detach effect.
 */
import { createEffect, createMemo, Index, onCleanup } from "solid-js"
import { getTrackReferenceId, setupMediaTrack, type TrackReference } from "@livekit/components-core"
import { RemoteAudioTrack, RemoteTrackPublication, Track, type Room } from "livekit-client"
import { observableState } from "../observable"
import { useTracks } from "../hooks/tracks"

export interface AudioTrackProps {
  /** The track reference of the track from which the audio is to be rendered. */
  trackRef: TrackReference
  /** Sets the volume of the audio track. By default, the range is between `0.0` and `1.0`. */
  volume?: number
  /** Mutes the audio track if set to `true` (the server stops sending track data). */
  muted?: boolean
}

export function AudioTrack(props: AudioTrackProps) {
  let element!: HTMLAudioElement

  const publication = observableState(
    () => setupMediaTrack(props.trackRef).trackObserver,
    () => props.trackRef.publication,
  )

  createEffect(() => {
    const track = publication()?.track
    if (!track) return
    if (!(props.trackRef.participant.isLocal && track.kind === "audio")) {
      track.attach(element)
    }
    onCleanup(() => track.detach(element))
  })

  createEffect(() => {
    const track = publication()?.track
    if (track === undefined || props.volume === undefined) return
    if (track instanceof RemoteAudioTrack) track.setVolume(props.volume)
  })

  createEffect(() => {
    const pub = publication()
    if (pub === undefined || props.muted === undefined) return
    if (pub instanceof RemoteTrackPublication) pub.setEnabled(!props.muted)
  })

  return <audio ref={element} data-lk-source={publication()?.source} />
}

export interface RoomAudioRendererProps {
  room?: Room
  /** Sets the volume for all audio tracks rendered by this component (0.0 to 1.0). */
  volume?: number
  /** If set to `true`, mutes all audio tracks rendered by the component. */
  muted?: boolean
}

/**
 * Drop-in solution for adding audio to a LiveKit app: renders all remote
 * audio tracks so microphones and screen share audio are audible.
 */
export function RoomAudioRenderer(props: RoomAudioRendererProps) {
  const tracks = useTracks([Track.Source.Microphone, Track.Source.ScreenShareAudio, Track.Source.Unknown], {
    updateOnlyOn: [],
    onlySubscribed: true,
    room: props.room,
  })
  const audioTracks = createMemo(() =>
    tracks().filter((ref) => !ref.participant.isLocal && ref.publication.kind === Track.Kind.Audio),
  )
  return (
    <div style={{ display: "none" }}>
      <Index each={audioTracks()}>
        {(trackRef) => <AudioTrack trackRef={trackRef()} volume={props.volume} muted={props.muted} />}
      </Index>
    </div>
  )
}

export { getTrackReferenceId }
