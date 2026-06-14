// Standalone voice constants with no heavy imports, so the agent worker bundle
// can reference them without pulling in token.ts's server-side dependency tree
// (livekit-server-sdk, Config, Auth, zod).

/** LiveKit agent name used for explicit dispatch of the EmberHarmony voice agent. */
export const VOICE_AGENT_NAME = "emberharmony-voice"
