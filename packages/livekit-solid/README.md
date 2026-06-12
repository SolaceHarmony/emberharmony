# @thesolaceproject/livekit-components-solid

SolidJS port of [`@livekit/components-react`](https://github.com/livekit/components-js) (v2.9.21), vendored into EmberHarmony because LiveKit only ships React components and this codebase is SolidJS.

The framework-agnostic state layer, [`@livekit/components-core`](https://www.npmjs.com/package/@livekit/components-core), is consumed as a regular npm dependency — it does all the heavy lifting (RxJS observables over room/participant/track state). This package translates the thin React wrapper on top of it to Solid primitives:

| Upstream (React)                          | Here (Solid)                              |
| ----------------------------------------- | ----------------------------------------- |
| `useObservableState` (useState/useEffect) | `observableState` (signal + effect)       |
| `RoomContext` / `useEnsureRoom`           | `RoomContext` / `useEnsureRoom`           |
| `useConnectionState`                      | `useConnectionState` → `Accessor`         |
| `useRemoteParticipants`                   | `useRemoteParticipants` → `Accessor`      |
| `useParticipantAttributes`                | `useParticipantAttributes` → `Accessor`   |
| `useParticipantTracks`                    | `useParticipantTracks` → `Accessor`       |
| `useLocalParticipant`                     | `useLocalParticipant` → accessors         |
| `useTracks`                               | `useTracks` (plain `Track.Source[]` only) |
| `useTrackTranscription`                   | `useTrackTranscription`                   |
| `useTextStream` / `useTranscriptions`     | `useTextStream` / `useTranscriptions`     |
| `useVoiceAssistant`                       | `useVoiceAssistant` → accessors           |
| `useMultibandTrackVolume`                 | `useMultibandTrackVolume`                 |
| `useBarAnimator` + animation sequences    | `useBarAnimator`                          |
| `<RoomAudioRenderer>` / `<AudioTrack>`    | `<RoomAudioRenderer>` / `<AudioTrack>`    |
| `<BarVisualizer>`                         | `<BarVisualizer>`                         |

Reactive inputs that change over time (participant identity, track reference, agent state) are passed as Solid accessors instead of plain values; hook returns are accessors.

Intentionally not ported (yet): video components, prefabs/layouts, `useTracks` placeholder support, chat, persistent user choices. Port them from the upstream source on demand — the cloned repo lives at `/tmp/livekit-components` during development, or re-clone `livekit/components-js`.

Licensed Apache-2.0, same as upstream — see `LICENSE` and `NOTICE`.
