# LiveKit Voice Integration Journal

## Overview

Adding voice input/output to EmberHarmony using LiveKit as the realtime media transport layer. The desktop app gets a microphone toggle in the prompt input bar. When activated, the user's speech is transcribed (STT), sent through EmberHarmony's existing LLM session pipeline (same tools, permissions, context), and the response is spoken back (TTS).

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  Tauri App (SolidJS)    │     │  LiveKit Server   │     │  Agent Worker (Node.js)  │
│                         │     │  (Cloud or self)  │     │  @livekit/agents          │
│  livekit-client         │◄───►│                   │◄───►│                           │
│  Room.connect()         │ WSS │  Routes WebRTC    │     │  voice.AgentSession       │
│  TokenSource.endpoint() │     │  media between    │     │  voice.Agent              │
│  setMicrophoneEnabled() │     │  participants     │     │  inference.STT/LLM/TTS    │
│  track.attach() <audio> │     │                   │     │  silero.VAD               │
│                         │     │  Token verify     │     │                           │
│  POST /voice/token ─────┤     │                   │     │  Bridges to EmberHarmony  │
│  GET /event (SSE) ◄─────┤     │                   │     │  REST API for sessions    │
│  POST /session/:id/msg ──┤     │                   │     │  SSE for text deltas      │
└─────────────────────────┘     └──────────────────┘     └──────────────────────────┘
```

### Flow

1. User clicks mic toggle in prompt toolbar
2. App requests LiveKit token from `POST /voice/token` on EmberHarmony server
3. App connects to LiveKit room via `livekit-client`
4. App enables microphone; audio streams to LiveKit room
5. Agent worker (already dispatched or auto-dispatched) joins the room
6. Agent's STT transcribes user speech
7. Agent sends transcription as a message to EmberHarmony's existing session API
8. Agent subscribes to SSE events for text delta streaming
9. Agent feeds text deltas to TTS; audio streams back to user via LiveKit
10. Text responses also appear in the chat UI (existing message flow)

### Key Design Decision: STT→LLM→TTS Pipeline

**Phase 1 uses the STT→LLM→TTS pipeline** (not OpenAI Realtime API) because:
- Reuses EmberHarmony's entire tool/permission/context system via the REST API
- No code duplication — voice sessions are just sessions with speech I/O
- Can swap to Realtime API later (Phase 2) for lower latency

---

## Packages Required

### Browser (packages/app, packages/desktop)

| Package | Version | Purpose |
|---------|---------|---------|
| `livekit-client` | 2.19.2 | WebRTC room connection, mic, audio playback |

### Server (packages/emberharmony)

| Package | Version | Purpose |
|---------|---------|---------|
| `livekit-server-sdk` | 2.15.4 | JWT token generation, room management |
| `@livekit/protocol` | 1.46.6 | Protobuf types for room config |

### Agent Worker (separate process or same server)

| Package | Version | Purpose |
|---------|---------|---------|
| `@livekit/agents` | 1.4.5 | Agent framework, worker, CLI |
| `@livekit/agents-plugin-silero` | 1.4.5 | Voice activity detection |
| `@livekit/agents-plugin-livekit` | 1.4.5 | Turn detection, noise cancellation |
| `@livekit/rtc-node` | 0.13.29 | Native RTC bindings (required by agents) |
| `@livekit/agents-plugin-openai` | 1.4.5 | OpenAI Realtime API (Phase 2) |
| `@livekit/plugins-ai-coustics` | latest | Noise cancellation enhancement |

### Note: No SolidJS component library

LiveKit only ships React components (`@livekit/components-react`). We build our own SolidJS reactive wrappers around `livekit-client` directly, using the same patterns as our other contexts (`createSimpleContext`, `createStore`, `createEffect`).

---

## Files to Modify

### Critical Path (Phase 1 — minimum viable voice)

#### 1. Workspace catalog — `package.json` (root)

Add to `workspaces.catalog`:
```json
"livekit-client": "2.19.2",
"livekit-server-sdk": "2.15.4",
"@livekit/protocol": "1.46.6"
```

#### 2. App package — `packages/app/package.json`

Add dependency:
```json
"livekit-client": "catalog:"
```

#### 3. Desktop package — `packages/desktop/package.json`

Add dependency:
```json
"livekit-client": "catalog:"
```

#### 4. Server package — `packages/emberharmony/package.json`

Add dependency:
```json
"livekit-server-sdk": "catalog:"
```

#### 5. Voice context — `packages/app/src/context/voice.tsx` (NEW)

```typescript
// Key structure:
// - createSimpleContext pattern (same as sdk.tsx, platform.tsx)
// - Room state: connecting | connected | disconnected | error
// - Mic state: muted | unmuted | unavailable
// - Agent state: listening | thinking | speaking | idle
// - Methods: connect(), disconnect(), toggleMute()
// - Uses livekit-client Room, TokenSource, RoomEvent
// - Subscribes to EmberHarmony SSE for text deltas (bridges to TTS)
```

Exports:
- `VoiceProvider` — context provider component
- `useVoice()` — returns `{ state, micState, agentState, connect, disconnect, toggleMute, room }`

#### 6. Platform type — `packages/app/src/context/platform.tsx`

Add to `Platform` type:
```typescript
/** Request microphone permission (desktop: Tauri dialog, web: browser API) */
requestMicrophonePermission?(): Promise<boolean>

/** Check if LiveKit voice is available (server configured) */
voiceAvailable?(): Promise<boolean>
```

#### 7. Desktop platform impl — `packages/desktop/src/index.tsx`

Implement `requestMicrophonePermission` and `voiceAvailable` in `createPlatform()`.

For `requestMicrophonePermission`: use `@tauri-apps/plugin-dialog` to show a native permission dialog, then `navigator.mediaDevices.getUserMedia({ audio: true })` to actually request the permission.

For `voiceAvailable`: check if the server has `LIVEKIT_URL` configured via a health endpoint.

#### 8. Web platform impl — `packages/app/src/entry.tsx`

Implement `requestMicrophonePermission` using `navigator.mediaDevices.getUserMedia({ audio: true })`.
Implement `voiceAvailable` same as desktop.

#### 9. Prompt input — `packages/app/src/components/prompt-input.tsx`

Add a microphone toggle button in the bottom toolbar. Location: right side, between the file-attach area and the submit button (lines ~2046-2062).

```typescript
// Import voice context
import { useVoice } from "@/context/voice"

// In the component:
const voice = useVoice()

// Button rendering (inside the right-side toolbar div):
<IconButton
  icon={voice.state === "connected" ? "mic" : "mic-off"}
  variant={voice.state === "connected" ? "primary" : "ghost"}
  onClick={voice.toggleMute}
  disabled={voice.state === "connecting"}
  title={t("voice.toggle")}
/>
```

When voice is active and user is speaking, the text transcription fills into the prompt input (same as the existing `speech.ts` utility but using LiveKit STT instead).

#### 10. App provider tree — `packages/app/src/app.tsx`

Wrap session route with `VoiceProvider`:
```typescript
// Inside the session route, between PromptProvider and CommentsProvider:
<VoiceProvider>
  {/* existing session content */}
</VoiceProvider>
```

#### 11. Server voice route — `packages/emberharmony/src/server/routes/voice.ts` (NEW)

```typescript
// Hono router with endpoints:
// POST /voice/token — generate LiveKit JWT token
//   Body: { sessionID: string, agentName?: string }
//   Response: { token: string, url: string, roomName: string }
//
// GET /voice/status — check if voice is configured
//   Response: { available: boolean, url: string | null }
```

Token generation uses `livekit-server-sdk`:
```typescript
import { AccessToken } from "livekit-server-sdk"
import { RoomConfiguration, RoomAgentDispatch } from "@livekit/protocol"

const token = new AccessToken(API_KEY, API_SECRET, {
  identity: `user_${userID}`,
  name: userName,
  ttl: "15m",
})
token.addGrant({ room: roomName, roomJoin: true, canPublish: true, canSubscribe: true, canPublishData: true })
token.roomConfig = new RoomConfiguration({
  agents: [new RoomAgentDispatch({ agentName: "emberharmony-voice" })],
})
const jwt = await token.toJwt()
```

#### 12. Server route registration — `packages/emberharmony/src/server/server.ts`

Add route:
```typescript
import VoiceRoutes from "./routes/voice"
// ...
.route("/voice", VoiceRoutes())
```

#### 13. Feature flags — `packages/emberharmony/src/flag/flag.ts`

Add:
```typescript
export const LIVEKIT_URL = value("EMBERHARMONY_LIVEKIT_URL", "")
export const LIVEKIT_API_KEY = value("EMBERHARMONY_LIVEKIT_API_KEY", "")
export const LIVEKIT_API_SECRET = value("EMBERHARMONY_LIVEKIT_API_SECRET", "")
export const VOICE_ENABLED = truthy("EMBERHARMONY_VOICE_ENABLED")
```

#### 14. Tauri CSP — `packages/desktop/src-tauri/tauri.conf.json`

Update CSP `connect-src` to add LiveKit WebSocket URLs:
```
connect-src 'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:* wss://localhost:* https://*.solace.ofharmony.ai https://github.com/SolaceHarmony/ tauri://localhost http://tauri.localhost wss://*.livekit.cloud https://*.livekit.cloud;
```

Update `media-src` to add `blob:` for WebRTC audio streams:
```
media-src 'self' data: blob:;
```

Also update `packages/desktop/src-tauri/tauri.prod.conf.json` with the same changes.

#### 15. Tauri capabilities — `packages/desktop/src-tauri/capabilities/default.json`

Add LiveKit URLs to the HTTP allow-list:
```json
{
  "url": "https://*.livekit.cloud/*"
},
{
  "url": "wss://*.livekit.cloud/*"
}
```

#### 16. Server CSP — `packages/emberharmony/src/server/server.ts`

Update the `secureHeaders` middleware CSP to include LiveKit URLs in `connectSrc` and `mediaSrc`:
```typescript
connectSrc: [...existing, "wss://*.livekit.cloud", "https://*.livekit.cloud"],
mediaSrc: ["'self'", "data:", "blob:"],
```

#### 17. macOS entitlements — `packages/desktop/src-tauri/entitlements.plist`

Already has `com.apple.security.device.audio-input` — no change needed.

#### 18. i18n — all locale files

Add voice keys to every locale file in:
- `packages/app/src/i18n/en.ts` (and all other locales)
- `packages/desktop/src/i18n/en.ts` (and all other locales)

Keys to add:
```typescript
"voice": {
  "toggle": "Voice mode",
  "connecting": "Connecting...",
  "connected": "Voice connected",
  "disconnected": "Voice disconnected",
  "muted": "Muted",
  "unmuted": "Unmuted",
  "unavailable": "Voice unavailable",
  "permission_denied": "Microphone permission denied",
  "error": "Voice error"
}
```

---

## Agent Worker Design

### Phase 1: Bridge Agent (STT → EmberHarmony API → TTS)

The agent runs as a separate Node.js process alongside the EmberHarmony server. It:

1. Connects to a LiveKit room when dispatched
2. Uses LiveKit Inference STT (Deepgram Nova-3) for speech-to-text
3. Sends transcribed text to EmberHarmony's `POST /session/:sessionID/message` API
4. Subscribes to EmberHarmony's `GET /event` SSE stream for `MessageV2.Event.PartUpdated` events
5. Feeds text deltas to TTS (Cartesia Sonic-3 via LiveKit Inference)
6. Audio streams back to the user via LiveKit room

**File:** `packages/emberharmony/src/voice/agent.ts`

```typescript
import { defineAgent, inference, voice, cli, ServerOptions } from "@livekit/agents"
import * as silero from "@livekit/agents-plugin-silero"
import * as livekit from "@livekit/agents-plugin-livekit"

class EmberHarmonyAgent extends voice.Agent {
  constructor() {
    super({
      instructions: "You are EmberHarmony, a helpful AI coding assistant.",
    })
  }
}

export default defineAgent({
  prewarm: async (proc) => {
    proc.userData.vad = await silero.VAD.load()
  },
  entry: async (ctx) => {
    const vad = ctx.proc.userData.vad
    const session = new voice.AgentSession({
      stt: new inference.STT({ model: "deepgram/nova-3", language: "multi" }),
      llm: new inference.LLM({ model: "openai/gpt-5.2-chat-latest" }),
      tts: new inference.TTS({ model: "cartesia/sonic-3", voice: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc" }),
      vad,
      turnDetection: new livekit.turnDetector.MultilingualModel(),
    })

    await session.start({ agent: new EmberHarmonyAgent(), room: ctx.room })
    await ctx.connect()
    await session.generateReply({ instructions: "Greet the user." })
  },
})

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url), agentName: "emberharmony-voice" }))
```

**Phase 2 enhancement:** Replace the STT/LLM/TTS pipeline with OpenAI Realtime API for lower latency:
```typescript
llm: new openai.realtime.RealtimeModel({ voice: "coral" })
// Remove stt, tts, vad, turnDetection options
```

### Token Server — `packages/emberharmony/src/voice/token.ts`

```typescript
import { AccessToken } from "livekit-server-sdk"
import { RoomConfiguration, RoomAgentDispatch } from "@livekit/protocol"
import { LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET } from "../flag/flag"

export async function createVoiceToken(opts: {
  roomName: string
  identity: string
  name?: string
  agentName?: string
}): Promise<{ token: string; url: string }> {
  const token = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
    identity: opts.identity,
    name: opts.name,
    ttl: "15m",
  })
  token.addGrant({
    room: opts.roomName,
    roomJoin: true,
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
  })
  if (opts.agentName) {
    token.roomConfig = new RoomConfiguration({
      agents: [new RoomAgentDispatch({ agentName: opts.agentName })],
    })
  }
  return { token: await token.toJwt(), url: LIVEKIT_URL }
}
```

### Server Startup — how the agent process runs

The agent can be started in two ways:

**Option A: Separate process** (recommended for Phase 1)
```bash
# Start the agent worker separately
LIVEKIT_URL=wss://... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \
  bun packages/emberharmony/src/voice/agent.ts dev
```

**Option B: Embedded in the EmberHarmony server** (future)
Spawn the agent as a child process from `packages/emberharmony/src/cli/cmd/serve.ts`, similar to how the web dev server is spawned in `web.ts`.

---

## Voice Context Provider Design

### `packages/app/src/context/voice.tsx`

```typescript
import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { createEffect, createSignal, onCleanup } from "solid-js"
import { Room, RoomEvent, ConnectionState, Track } from "livekit-client"
import { useSDK } from "./sdk"
import { usePlatform } from "./platform"

type VoiceState = "disconnected" | "connecting" | "connected" | "error"
type MicState = "muted" | "unmuted" | "unavailable"
type AgentState = "disconnected" | "listening" | "thinking" | "speaking"

export const { use: useVoice, provider: VoiceProvider } = createSimpleContext({
  name: "Voice",
  init: (props: { sessionID: string }) => {
    const platform = usePlatform()
    const sdk = useSDK()
    const [state, setState] = createSignal<VoiceState>("disconnected")
    const [micState, setMicState] = createSignal<MicState>("unavailable")
    const [agentState, setAgentState] = createSignal<AgentState>("disconnected")
    const [room, setRoom] = createSignal<Room | null>(null)

    const connect = async () => {
      setState("connecting")
      // 1. Get token from EmberHarmony server
      const resp = await sdk.client.voice.token({
        sessionID: props.sessionID,
      })
      // 2. Create and connect room
      const r = new Room({ adaptiveStream: true, dynacast: true })
      r.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const element = track.attach()
          document.body.appendChild(element)
        }
      })
      r.on(RoomEvent.Disconnected, () => setState("disconnected"))
      await r.connect(resp.url, resp.token)
      setRoom(r)
      // 3. Enable microphone
      await r.localParticipant.setMicrophoneEnabled(true)
      setMicState("unmuted")
      setState("connected")
    }

    const disconnect = () => {
      room()?.disconnect()
      setRoom(null)
      setMicState("unavailable")
      setAgentState("disconnected")
      setState("disconnected")
    }

    const toggleMute = async () => {
      const r = room()
      if (!r) return
      const enabled = micState() === "unmuted"
      await r.localParticipant.setMicrophoneEnabled(!enabled)
      setMicState(enabled ? "muted" : "unmuted")
    }

    onCleanup(() => { room()?.disconnect() })

    return { state, micState, agentState, connect, disconnect, toggleMute, room }
  },
})
```

---

## Existing Speech Utility

`packages/app/src/utils/speech.ts` already implements a complete Web Speech API wrapper (`createSpeechRecognition`). It's **not currently used by any component**. 

**Plan for it:**
- Keep it as a **fallback** for when no LiveKit server is configured — basic dictation into the prompt input
- The voice toggle button shows a different icon/state when LiveKit is available vs. when only browser STT is available
- LiveKit voice mode = full duplex (speak and hear), browser STT = dictation only (text into prompt)

---

## Environment Variables

```
# LiveKit server URL (e.g., wss://my-project.livekit.cloud)
EMBERHARMONY_LIVEKIT_URL=

# LiveKit API key and secret (for token generation)
EMBERHARMONY_LIVEKIT_API_KEY=
EMBERHARMONY_LIVEKIT_API_SECRET=

# Feature flag to enable/disable voice
EMBERHARMONY_VOICE_ENABLED=true

# Voice agent configuration (optional, for STT/LLM/TTS model selection)
EMBERHARMONY_VOICE_STT_MODEL=deepgram/nova-3
EMBERHARMONY_VOICE_LLM_MODEL=openai/gpt-5.2-chat-latest
EMBERHARMONY_VOICE_TTS_MODEL=cartesia/sonic-3
EMBERHARMONY_VOICE_TTS_VOICE=9626c31c-bec5-4cca-baa8-f8ba9e84c8bc
```

---

## Implementation Order

### Phase 1: Minimum Viable Voice (desktop only)

1. **Install dependencies** — add `livekit-client`, `livekit-server-sdk` to workspace catalog and packages
2. **Create token endpoint** — `POST /voice/token` route in EmberHarmony server
3. **Create voice context** — `packages/app/src/context/voice.tsx` with Room connect/disconnect/mute
4. **Add voice toggle to prompt input** — mic button in toolbar, shows connection state
5. **Update Tauri CSP** — add `wss://*.livekit.cloud`, `blob:` to media-src
6. **Update Tauri capabilities** — add LiveKit URLs to HTTP allow-list
7. **Add i18n keys** — voice-related translations in all locale files
8. **Create agent worker** — `packages/emberharmony/src/voice/agent.ts` with STT→LLM→TTS pipeline
9. **Test end-to-end** — desktop app connects to LiveKit, speaks, hears response

### Phase 2: Enhanced Voice (post-MVP)

10. **OpenAI Realtime API** — lower latency speech-to-speech via `openai.realtime.RealtimeModel`
11. **Web Speech API fallback** — wire up `speech.ts` for dictation-only mode when no LiveKit configured
12. **Visual voice indicators** — animated waveform/pulse in the prompt input during listening/speaking
13. **Transcript sync** — show voice transcript in the chat as messages arrive
14. **Noise cancellation** — add `@livekit/plugins-ai-coustics` for background noise suppression
15. **Mobile support** — adapt voice UI for smaller screens

---

## Tauri/WebView Gotchas

1. **Audio autoplay**: Browsers require user gesture before audio playback. Use `room.startAudio()` in the voice toggle click handler (which IS a user gesture).
2. **Microphone permissions**: macOS entitlements already include `com.apple.security.device.audio-input`. The Tauri webview should handle `navigator.mediaDevices.getUserMedia` natively.
3. **WebRTC support**: Both WebKit (macOS) and WebView2/Edge (Windows) support WebRTC. Linux (WebKitGTK) may need `libwebkit2gtk-4.1-dev` (already a build dependency).
4. **CSP**: Must allow `wss://` and `blob:` in CSP directives for LiveKit WebRTC connections.
5. **`isBrowserSupported()`**: Call this before attempting connection. Some Linux WebKitGTK builds may not support all WebRTC features.

---

## Reference Files

| Purpose | Path |
|---------|------|
| Voice context (NEW) | `packages/app/src/context/voice.tsx` |
| Prompt input (mic button) | `packages/app/src/components/prompt-input.tsx` |
| Platform type (add voice methods) | `packages/app/src/context/platform.tsx` |
| Desktop platform impl | `packages/desktop/src/index.tsx` |
| Web platform impl | `packages/app/src/entry.tsx` |
| App provider tree | `packages/app/src/app.tsx` |
| Session page | `packages/app/src/pages/session.tsx` |
| Speech fallback utility | `packages/app/src/utils/speech.ts` |
| Voice route (NEW) | `packages/emberharmony/src/server/routes/voice.ts` |
| Voice token util (NEW) | `packages/emberharmony/src/voice/token.ts` |
| Voice agent (NEW) | `packages/emberharmony/src/voice/agent.ts` |
| Server route registration | `packages/emberharmony/src/server/server.ts` |
| Feature flags | `packages/emberharmony/src/flag/flag.ts` |
| Tauri CSP | `packages/desktop/src-tauri/tauri.conf.json` |
| Tauri prod CSP | `packages/desktop/src-tauri/tauri.prod.conf.json` |
| Tauri capabilities | `packages/desktop/src-tauri/capabilities/default.json` |
| macOS entitlements | `packages/desktop/src-tauri/entitlements.plist` |
| App i18n (en) | `packages/app/src/i18n/en.ts` |
| Desktop i18n (en) | `packages/desktop/src/i18n/en.ts` |
| Root workspace catalog | `package.json` |
| App deps | `packages/app/package.json` |
| Desktop deps | `packages/desktop/package.json` |
| Server deps | `packages/emberharmony/package.json` |
| Server startup | `packages/emberharmony/src/cli/cmd/serve.ts` |