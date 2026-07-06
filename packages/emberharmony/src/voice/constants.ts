// Standalone voice constants with no heavy imports, so the agent worker bundle
// can reference them without pulling in token.ts's server-side dependency tree
// (livekit-server-sdk, Config, Auth, zod).

/** LiveKit agent name used for explicit dispatch of the EmberHarmony voice agent. */
export const VOICE_AGENT_NAME = "emberharmony-voice"

/** Reliable LiveKit data topic used by the desktop kernel to control the agent. */
export const VOICE_CONTROL_TOPIC = "emberharmony.voice.control"

/** Interrupt the current agent speech/generation without disconnecting the room. */
export const VOICE_CONTROL_INTERRUPT = "interrupt"
