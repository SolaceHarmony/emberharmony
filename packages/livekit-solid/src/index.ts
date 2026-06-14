export { RoomContext, useRoomContext, useMaybeRoomContext, useEnsureRoom } from "./context"
export { observableState } from "./observable"
export {
  useConnectionState,
  useRemoteParticipants,
  useParticipantAttributes,
  useParticipantTracks,
  useLocalParticipant,
  type UseRemoteParticipantsOptions,
} from "./hooks/participants"
export {
  useTracks,
  useTrackTranscription,
  useTextStream,
  useTranscriptions,
  type UseTracksOptions,
  type TrackTranscriptionOptions,
  type UseTranscriptionsOptions,
} from "./hooks/tracks"
export { useVoiceAssistant, type AgentState, type VoiceAssistant } from "./hooks/voice-assistant"
export { useMultibandTrackVolume, type MultiBandTrackVolumeOptions } from "./hooks/volume"
export { useBarAnimator } from "./hooks/bar-animator"
export { AudioTrack, RoomAudioRenderer, type AudioTrackProps, type RoomAudioRendererProps } from "./components/audio"
export { BarVisualizer, type BarVisualizerProps, type BarVisualizerOptions } from "./components/bar-visualizer"
// Re-export the upstream types that appear in this package's public hook
// signatures (useVoiceAssistant().audioTrack, useTranscriptions()), so
// consumers can type values bridged out of the hooks without depending on
// @livekit/components-core directly.
export type { TrackReference, TextStreamData } from "@livekit/components-core"
