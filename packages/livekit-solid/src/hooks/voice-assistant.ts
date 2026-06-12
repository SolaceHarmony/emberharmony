/**
 * Ported from @livekit/components-react `hooks/useVoiceAssistant.ts` and the
 * `AgentState` type from `hooks/useAgent.ts` (Apache-2.0).
 */
import { type Accessor, createMemo } from "solid-js"
import {
  ParticipantAgentAttributes,
  type ReceivedTranscriptionSegment,
  type TrackReference,
} from "@livekit/components-core"
import { ConnectionState, ParticipantKind, Track } from "livekit-client"
import type { RemoteParticipant, Room } from "livekit-client"
import {
  useConnectionState,
  useParticipantAttributes,
  useParticipantTracks,
  useRemoteParticipants,
} from "./participants"
import { useTrackTranscription } from "./tracks"

export type AgentState =
  | "disconnected"
  | "connecting"
  | "pre-connect-buffering"
  | "failed"
  | "initializing"
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"

export interface VoiceAssistant {
  /** The agent participant. */
  agent: Accessor<RemoteParticipant | undefined>
  /** The current state of the agent. */
  state: Accessor<AgentState>
  /** The microphone track published by the agent or associated avatar worker (if any). */
  audioTrack: Accessor<TrackReference | undefined>
  /** The camera track published by the agent or associated avatar worker (if any). */
  videoTrack: Accessor<TrackReference | undefined>
  /** The transcriptions of the agent's microphone track (if any). */
  agentTranscriptions: Accessor<ReceivedTranscriptionSegment[]>
  /** The agent's participant attributes. */
  agentAttributes: Accessor<RemoteParticipant["attributes"] | undefined>
}

const state_attribute = ParticipantAgentAttributes.AgentState

/**
 * Looks for the first agent-participant in the room.
 * @remarks Requires an agent running with livekit-agents >= 0.9.0
 */
export function useVoiceAssistant(room?: Room): VoiceAssistant {
  const remoteParticipants = useRemoteParticipants({ room })
  const agent = createMemo(() =>
    remoteParticipants().find(
      (p) => p.kind === ParticipantKind.AGENT && !(ParticipantAgentAttributes.PublishOnBehalf in p.attributes),
    ),
  )
  const worker = createMemo(() =>
    remoteParticipants().find(
      (p) =>
        p.kind === ParticipantKind.AGENT &&
        p.attributes[ParticipantAgentAttributes.PublishOnBehalf] === agent()?.identity,
    ),
  )
  const agentTracks = useParticipantTracks(
    [Track.Source.Microphone, Track.Source.Camera],
    () => agent()?.identity,
    room,
  )
  const workerTracks = useParticipantTracks(
    [Track.Source.Microphone, Track.Source.Camera],
    () => worker()?.identity,
    room,
  )
  const audioTrack = createMemo(
    () =>
      agentTracks().find((t) => t.source === Track.Source.Microphone) ??
      workerTracks().find((t) => t.source === Track.Source.Microphone),
  )
  const videoTrack = createMemo(
    () =>
      agentTracks().find((t) => t.source === Track.Source.Camera) ??
      workerTracks().find((t) => t.source === Track.Source.Camera),
  )
  const { segments: agentTranscriptions } = useTrackTranscription(audioTrack)
  const connectionState = useConnectionState(room)
  const attributeState = useParticipantAttributes(agent)

  const state = createMemo<AgentState>(() => {
    const attributes = attributeState().attributes
    if (connectionState() === ConnectionState.Disconnected) return "disconnected"
    if (connectionState() === ConnectionState.Connecting || !agent() || !attributes?.[state_attribute])
      return "connecting"
    return attributes[state_attribute] as AgentState
  })

  return {
    agent,
    state,
    audioTrack,
    videoTrack,
    agentTranscriptions,
    agentAttributes: () => attributeState().attributes,
  }
}
