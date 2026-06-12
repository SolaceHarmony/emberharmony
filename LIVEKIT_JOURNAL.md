# LiveKit Voice Integration Journal

## Status (2026-06-12): Phase 1 implemented

Everything in the Phase 1 critical path below is built and typechecks. What exists now:

- **Deps**: `livekit-client` (app, desktop), `livekit-server-sdk` + `@livekit/protocol` + `@livekit/agents` stack (emberharmony) in the workspace catalog.
- **Server**: `Flag.EMBERHARMONY_LIVEKIT_URL/API_KEY/API_SECRET` (each falls back to the standard `LIVEKIT_*` env var names, so a stock LiveKit `.env` works as-is); voice is available whenever credentials are configured, with `EMBERHARMONY_VOICE_DISABLE=true` as the kill switch (opt-out instead of the originally planned `EMBERHARMONY_VOICE_ENABLED` opt-in, matching the codebase's `DISABLE_*` flag convention); token util at `packages/emberharmony/src/voice/token.ts`; routes at `packages/emberharmony/src/server/routes/voice.ts` (`GET /voice/status`, `POST /voice/token`), registered in `server.ts`; server CSP updated. Verified by curl: status reports availability, token returns a JWT with room grant + `emberharmony-voice` agent dispatch, and the unconfigured path returns `VoiceNotConfiguredError`.
- **SDK**: regenerated; clients call `client.voice.status()` and `client.voice.token({ sessionID })`.
- **App**: `packages/app/src/context/voice.tsx` (`useVoice`: state/micState/agentState, connect/disconnect/toggleMute, agent state tracked via the `lk.agent.state` participant attribute); `VoiceProvider` wraps the session route in `app.tsx`; mic toggle button in `prompt-input.tsx` (visible only when the server reports voice available); `microphone`/`microphone-off` icons added to the UI icon set; `voice.*` i18n keys in `en.ts` (other locales fall back to English).
- **Tauri**: `tauri.conf.json` CSP allows `wss://*.livekit.cloud` + `blob:` media; capabilities allow LiveKit HTTP/WSS. (`tauri.prod.conf.json` has no CSP section — it inherits from the base config, so no change needed there.)
- **Agent worker**: `packages/emberharmony/src/voice/agent.ts` (STT→LLM→TTS pipeline via LiveKit Inference, silero VAD, multilingual turn detection). Run `bun run src/voice/agent.ts download-files` once, then `bun run voice-agent` (the `dev` subcommand) with the `EMBERHARMONY_LIVEKIT_*` env vars set. Verified: boots, loads models, attempts worker registration.

End-to-end verified against the real LiveKit Cloud project (credentials from `.env`): token issued → simulated participant joined → agent auto-dispatched → went thinking → published TTS greeting audio → listening.

## Status (2026-06-12, later): vendored SolidJS components package

LiveKit only ships React components, so `packages/livekit-solid` (`@thesolaceproject/livekit-components-solid`) now holds a SolidJS port of the parts of `@livekit/components-react` v2.9.21 we need, built on the framework-agnostic `@livekit/components-core` npm package (which does the real work via RxJS observables — the port translates the thin React wrapper to signals/effects). Apache-2.0, with LICENSE/NOTICE attribution; see that package's README for the port table and what was intentionally left unported. To port more surface, re-clone `livekit/components-js` and translate from `packages/react/src`.

Ported: `RoomContext`, `observableState` (the signal⇄observable bridge, unit-tested), `useConnectionState`, `useRemoteParticipants`, `useParticipantAttributes`, `useParticipantTracks`, `useLocalParticipant`, `useTracks`, `useTrackTranscription`, `useTextStream`, `useTranscriptions`, `useVoiceAssistant`, `useMultibandTrackVolume`, `useBarAnimator`, `<RoomAudioRenderer>`, `<AudioTrack>`, `<BarVisualizer>`.

The app's `voice.tsx` was rewired on top of it: one `Room` per provider exposed via `RoomContext`, `<RoomAudioRenderer>` replaces the manual track attach/detach, agent state comes from `useVoiceAssistant`, mic state derives from `useLocalParticipant`, and live transcriptions are exposed on the context (`transcriptions`) for UI use. The prompt input shows a `<BarVisualizer>` (agent state + audio-reactive bars, styled in `packages/app/src/index.css`) while voice is connected.

## Status (2026-06-12, later still): session bridge + desktop focus

Desktop (Tauri) is the primary target for voice. Changes:

- **Session bridge** (`packages/emberharmony/src/voice/bridge.ts`): `SessionLLM` — a custom `llm.LLM` for the agent pipeline. Each voice turn posts the transcribed utterance to `POST /session/:id/prompt_async` and streams the reply text out of the `GET /event` SSE feed (`message.part.updated` deltas, completion via `message.updated` with `time.completed`). The reply is identified as the first message in the session that streams text deltas (user parts are created whole, never stream). The session does the real work — model, tools, permissions, context — and voice turns show up in the chat UI like typed ones. The bridge reuses whatever model the session last used (the app always prompts with an explicit model; server-side `defaultModel()` is only a fallback and isn't well-defined unless the user configured one). Verified live: ~2s to first delta through a seeded session.
- **Dispatch metadata**: `POST /voice/token` embeds `{sessionID, directory, serverUrl}` in the `RoomAgentDispatch` metadata; the agent refuses to start without it. `serverUrl` is taken from the request origin (overridable with `EMBERHARMONY_VOICE_SERVER_URL` on the worker). Basic-auth servers are supported via the `EMBERHARMONY_SERVER_USERNAME/PASSWORD` env on the worker.
- **Agent** (`agent.ts`): pipeline is now STT → `SessionLLM` → TTS (`EMBERHARMONY_VOICE_LLM_MODEL` is gone); greets via `session.say(...)` since `generateReply` has no meaning against a session bridge.
- **Desktop mic**: `packages/desktop/src-tauri/Info.plist` adds `NSMicrophoneUsageDescription` (required on macOS or TCC kills the bundled app at first `getUserMedia`); entitlements already had `audio-input`; wry 0.55 auto-grants WKWebView media-capture requests.
- **Live transcript strip** (`packages/app/src/components/voice-transcript.tsx`): shows the current utterance (user STT / agent speech) above the prompt input while voice is connected, fed from the `lk.transcription` text stream via the vendored `useTranscriptions`.

Verified end-to-end (against real LiveKit Cloud + local Ollama): bridge turn streams the exact expected reply through the session; room-level dispatch parses metadata, greets over TTS, settles into listening.

Known gaps / next:
- Interrupting the agent mid-reply stops the voice stream but does not abort server-side generation (the reply still completes in the chat UI). Decide whether to call `POST /session/:id/abort` on interruption.
- An empty session with no configured default model fails the first voice turn ("no providers found") — voice currently assumes it joins a session that has been used at least once.
- Human pass on the desktop app: mic permission prompt, greeting audible, full spoken round-trip.
- Phase 2 (Realtime API, noise cancellation, mobile) unchanged.

---

## Plan: Voice as a configurable provider (replace env vars with UI-managed config)

### Why voice doesn't fit the existing provider model

EmberHarmony's `provider` config is single-source: a provider supplies *the* LLM for a session, and exactly one is active per prompt. Voice is orthogonal — you run an Ollama (or Anthropic, etc.) session *and* a voice stack at the same time. The session bridge already enforces this separation: **the voice stack is ears, mouth, and transport; the session model stays the brain.** Voice provider selection must never change which LLM answers.

So voice gets its own registry — a **voice stack** with independently selectable parts:

| Part | Role | Examples |
|------|------|----------|
| Transport | WebRTC rooms, agent dispatch | LiveKit (cloud or self-hosted) |
| STT | speech → text | **Deepgram Nova-3 (blessed default** — the pairing Cerebras used for their demo, and what we already default to), AssemblyAI |
| TTS | text → speech | Cartesia Sonic-3 (current default), ElevenLabs, Inworld, Rime |
| Realtime (Phase 2) | STT+LLM+TTS fused | OpenAI Realtime — *only this kind would override the session brain; flag it clearly in the registry* |

### Two integration tiers

1. **LiveKit Inference (default, what we use today)** — one credential (the LiveKit API key/secret), STT/TTS chosen as model strings (`deepgram/nova-3:multi`, `cartesia/sonic-3:<voice>`) routed through LiveKit's gateway. Zero per-provider keys; the curated picker is just a list of gateway-blessed model strings.
2. **Direct (BYO keys)** — the `@livekit/agents` plugin ecosystem (deepgram, cartesia, elevenlabs, assemblyai, …) with the provider's own API key. Bypasses the gateway: per-provider billing, sometimes lower latency, more knobs. Requires the worker to load the matching plugin package.

Ship tier 1 configurable first; tier 2 is additive (registry entries gain a `plugin` field + key requirement).

### Data model

- **Non-secret config** — new `voice` section in `Config.Info` (`emberharmony.json`, global or per-project), replacing the `EMBERHARMONY_VOICE_*` env vars:
  ```jsonc
  "voice": {
    "disabled": false,
    "livekit": { "url": "wss://<project>.livekit.cloud" },
    "stt": "deepgram/nova-3:multi",          // tier 1 model string, or "plugin:deepgram/nova-3" for tier 2
    "tts": "cartesia/sonic-3:<voiceID>"
  }
  ```
- **Secrets** — reuse the existing `Auth` store (`PUT /auth/:providerID`, same place `auth login` writes):
  - `livekit` → API key + secret. `Auth.Api` holds a single `key`; extend with an optional `secret` field (backwards-compatible discriminated-union member or an optional column on `Api`).
  - `deepgram`, `cartesia`, `elevenlabs`, … → ordinary `{type:"api", key}` entries, only needed for tier 2.
- **Precedence**: config/auth > `EMBERHARMONY_LIVEKIT_*`/`LIVEKIT_*` env (env stays as CI/dev fallback and for the standalone worker).

### Server surface

- `GET /voice/config` — effective settings + the blessed registry (STT/TTS options with display names) + per-entry `hasCredentials` booleans (never the secrets themselves).
- `PATCH /voice/config` — update the `voice` config section.
- `GET /voice/status` — unchanged (availability for the mic button).
- Regenerate the SDK so the app gets `client.voice.config()` / typed registry.

### Worker wiring

Today the agent worker is a hand-started process reading env vars. Plan (Option B from the original journal): `serve.ts` spawns the worker as a child process when voice is configured, injecting resolved config+credentials via env at spawn. Config changes restart the worker. This keeps a single configuration path (UI → config/auth → worker) and means users never touch env vars. The standalone `bun run voice-agent` stays for development.

### UI

Settings → **Voice** panel (desktop-first):
- Enable toggle (writes `voice.disabled`)
- LiveKit connection: URL field + key/secret credential fields (stored via auth route) + a **Test connection** button (round-trips `/voice/status` + a token mint)
- STT picker and TTS picker (+ TTS voice ID) from the registry, grouped by tier; tier-2 entries show a key field inline when `hasCredentials` is false
- Existing mic button/visualizer/transcript untouched — they already key off `/voice/status`

### Sequencing

1. Config schema (`voice` section) + `Auth.Api.secret` + resolution in `Voice.available()`/`token()`/worker env (replaces flag reads, env as fallback)
2. `/voice/config` GET/PATCH + registry + SDK regen
3. Worker spawn from `serve.ts`
4. Settings panel in the app
5. Tier 2 (direct plugin providers with BYO keys)

### Status (2026-06-12): steps 1–4 implemented

- **Schema**: `voice` section in `Config.Info` (`disabled`, `livekit.url`, `stt`, `tts`); `Auth.Api` gained an optional `secret`. `Voice.settings()` in `voice/token.ts` resolves config + the `livekit` auth entry first, `EMBERHARMONY_LIVEKIT_*`/`LIVEKIT_*`/`EMBERHARMONY_VOICE_*` env as fallback.
- **Registry**: `voice/registry.ts` — curated tier-1 gateway model strings (ids verified against `@livekit/agents` inference typings). Defaults: Deepgram Nova-3 STT, Cartesia Sonic-3 TTS.
- **Endpoints**: `GET /voice/config` (effective settings + registry + `credentials.livekit` boolean, never secrets) and `PATCH /voice/config` (writes the global config via `Config.updateGlobal`, restarts a serve-managed worker, and responds from the merged result — `updateGlobal` disposes instance caches asynchronously, so reading back through `Config.get()` in the same request races a stale cache). Credentials go through the existing `PUT /auth/livekit` with `{type:"api", key, secret}`. SDK regenerated.
- **Worker spawn**: `voice/worker.ts` + `serve.ts` — serve spawns the agent worker with resolved settings injected as env, health-check port `0` (ephemeral, so it never collides with a manually started worker on 8081), stdout/stderr inherited (worker logs and failures surface in serve output), and SIGINT/SIGTERM/exit handlers so the child dies with serve (the idle `await` in serve never resolves; without handlers a signal orphans the worker — observed before the fix). Compiled-CLI builds can't spawn the worker yet (agent source isn't bundled); it warns and you run `bun run voice-agent` manually.
- **Settings UI**: `settings-voice.tsx`, registered as a "Voice" tab in `dialog-settings.tsx` — enable toggle, LiveKit URL + key/secret fields (saved via the auth route; placeholders show `••••` when stored), Test-connection button (round-trips `/voice/status`), and STT/TTS pickers from the registry.

Verified sandboxed (XDG_CONFIG_HOME/XDG_DATA_HOME redirected, zero env vars): fresh server reports unavailable → URL via PATCH + credentials via auth → status available, token mints, `voice` section lands in the global `emberharmony.jsonc` and secrets in `auth.json` (0600) → serve spawns the worker, it registers with LiveKit Cloud, and shuts down cleanly on SIGTERM.

Remaining for this plan: tier 2 (BYO-key plugin providers), bundling the agent into the compiled CLI so worker spawn works outside source checkouts, and desktop CSP for self-hosted LiveKit servers (connect-src currently allowlists `*.livekit.cloud` + localhost only — a saved self-hosted URL is blocked by the webview until the CSP becomes config-driven).

## Status (2026-06-12, evening): desktop round-trip verified live + plan/build voice workflow

Full hands-free round-trip confirmed by a human on the desktop app: spoken question → Deepgram STT → session bridge → reply spoken back via Cartesia, with live transcript strip, visualizer states, tool execution by voice (ran Bash from a spoken command), and multi-turn context. Issues found live and fixed:

- **WKWebView audio unlock**: audio elements played silently (healthy srcObject, not paused, volume 1 — inaudible) until an `AudioContext` was created. `connect()` now resumes an AudioContext inside the click gesture and no longer swallows `startAudio()` failures; an `AudioPlaybackStatusChanged` handler retries if playback gets blocked later.
- **Worker env**: LiveKit Inference STT/TTS read the standard `LIVEKIT_*` env names — the serve spawn now injects both naming schemes. (Symptom: agent entry died with "apiKey is required", so the agent joined but never spoke.)
- **Interruption pile-up**: interrupting the agent left the server session generating; the next voice turn was rejected as busy and the LLM stream waited forever ("job is unresponsive"). The bridge now POSTs `/session/:id/abort` when its stream is aborted, and a staleness watchdog (server heartbeats guarantee wakeups) errors out instead of hanging.
- **Reconnect dispatch**: token `roomConfig` agent dispatch only fires at room creation; reconnecting into a lingering room summons no agent. Fixed durably after PR review: the token route now checks the room via `RoomServiceClient` and explicitly dispatches via `AgentDispatchClient` when the room exists without an agent participant. Live-verified: agent present on first join and on immediate reconnect.
- **Tauri CSP**: added `ipc: http://ipc.localhost` to connect-src (plugin IPC was falling back to postMessage).
- **Settings toggle race**: the Voice switch mounted during config load reads "off"; clicking it then persisted `disabled: true`. The switch now renders only after config loads.

### Voice workflow: plan by default, build on spoken confirmation

`voice/workflow.ts` — every spoken turn runs the session's **plan** agent (read-only) unless a small fast gateway model (`openai/gpt-5.4-nano`, configurable as `voice.intent`) classifies the utterance as an explicit confirmation ("yes, do it", "sounds good, ship it") — that single turn runs as **build**, then the next turn re-defaults to plan, so every execution needs a fresh spoken yes. Classification failure always falls back to plan — an error can never grant execution. The hook is `voice.Agent.onUserTurnCompleted`, which runs before the bridge; the bridge passes `agent` and a `system` voice-etiquette prompt (short speakable replies, propose-then-ask) on every prompt. Verified against the live gateway: confirmations route to build, requests/questions/hesitation stay in plan.

### Configurability audit (PR gate)

Sandboxed (XDG redirected, zero env): URL, API key+secret, STT, TTS, intent model, and the disable switch all set via UI/API → persisted (`emberharmony.jsonc` + `auth.json` 0600) → survive server restart → worker env injection carries all of them. Env vars remain only as dev/CI fallback. Internal-only env (worker port, worker→server URL) is plumbing, not user config.

Follow-up fixes (2026-06-12, after the first signed local desktop build):
- `VoiceWorker.restart()` now starts the worker even when serve booted unconfigured — configuring voice through the Settings panel brings the worker up live, no serve restart needed.
- Empty-session first turn fixed: the mic button passes the app's currently selected model through `POST /voice/token` → dispatch metadata → `SessionLLM.fallbackModel`, used only when the session has no assistant message to inherit a model from.
- Local desktop build: `bun run build:local` in packages/desktop works with the keychain's Developer ID identity (override `APPLE_SIGNING_IDENTITY` if `.env` names a different team; notarization is skipped for local builds and not needed to run on the build machine). Verified the bundled app carries `NSMicrophoneUsageDescription`.
- Mic-button availability is fetched once per session mount; after configuring voice in settings, re-open the session view to see the button. Worth wiring to config-change events later.

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
# Falls back to LIVEKIT_URL if unset (same for the key/secret below)
EMBERHARMONY_LIVEKIT_URL=

# LiveKit API key and secret (for token generation)
EMBERHARMONY_LIVEKIT_API_KEY=
EMBERHARMONY_LIVEKIT_API_SECRET=

# Voice is on whenever credentials are configured; set this to turn it off
EMBERHARMONY_VOICE_DISABLE=false

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