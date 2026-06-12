/**
 * Ported from @livekit/components-react hooks (Apache-2.0):
 * useConnectionStatus, useRemoteParticipants, useParticipantAttributes,
 * useParticipantTracks, useLocalParticipant.
 */
import { type Accessor, createMemo } from "solid-js"
import {
  connectedParticipantsObserver,
  connectionStateObserver,
  observeParticipantMedia,
  participantAttributesObserver,
  participantTracksObservable,
  type TrackReference,
} from "@livekit/components-core"
import type { Participant, RemoteParticipant, Room, RoomEvent, Track } from "livekit-client"
import { useEnsureRoom } from "../context"
import { observableState } from "../observable"

export function useConnectionState(room?: Room) {
  const r = useEnsureRoom(room)
  return observableState(
    () => connectionStateObserver(r),
    () => r.state,
  )
}

export interface UseRemoteParticipantsOptions {
  updateOnlyOn?: RoomEvent[]
  room?: Room
}

export function useRemoteParticipants(options: UseRemoteParticipantsOptions = {}): Accessor<RemoteParticipant[]> {
  const room = useEnsureRoom(options.room)
  return observableState(
    () => connectedParticipantsObserver(room, { additionalRoomEvents: options.updateOnlyOn }),
    () => [] as RemoteParticipant[],
  )
}

export function useParticipantAttributes(participant: Accessor<Participant | undefined>) {
  return observableState(
    () => {
      const p = participant()
      return p ? participantAttributesObserver(p) : undefined
    },
    () => ({ attributes: participant()?.attributes }),
  )
}

export function useParticipantTracks(
  sources: Track.Source[],
  participantIdentity: Accessor<string | undefined>,
  room?: Room,
): Accessor<TrackReference[]> {
  const participants = useRemoteParticipants({ room, updateOnlyOn: [] })
  const participant = createMemo(() => {
    const identity = participantIdentity()
    if (!identity) return undefined
    return participants().find((p) => p.identity === identity)
  })
  return observableState(
    () => {
      const p = participant()
      return p ? participantTracksObservable(p, { sources }) : undefined
    },
    () => [] as TrackReference[],
  )
}

export function useLocalParticipant(room?: Room) {
  const r = useEnsureRoom(room)
  const media = observableState(
    () => observeParticipantMedia(r.localParticipant),
    () => ({
      isCameraEnabled: r.localParticipant.isCameraEnabled,
      isMicrophoneEnabled: r.localParticipant.isMicrophoneEnabled,
      isScreenShareEnabled: r.localParticipant.isScreenShareEnabled,
      cameraTrack: undefined,
      microphoneTrack: undefined,
      participant: r.localParticipant,
    }),
  )
  return {
    localParticipant: () => media().participant,
    isMicrophoneEnabled: () => media().isMicrophoneEnabled,
    microphoneTrack: () => media().microphoneTrack,
  }
}
