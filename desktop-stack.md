# Desktop Voice Kernel Layout

The desktop voice layer should be one native Tauri/Rust kernel with two provider backends:
local LFM2-Audio and native LiveKit/WebRTC. SolidJS is the control surface and renderer; it
does not own a LiveKit `Room`, microphone truth, provider availability, stop semantics, or
audio buffering.

Rust has language-level `async`/`await`; Tokio is the closest practical equivalent to Python's
`asyncio` here. Tauri already runs on Tokio, so the right split is:

- Tokio tasks for control-plane work: Tauri commands, settings, keychain credentials, LiveKit
  signaling, reconnects, token/config minting, and event fanout.
- Dedicated OS threads for realtime work: audio callbacks, turn detection, LFM2 inference,
  decode/playback, and native WebRTC/media workers.
- Bounded non-blocking buffers between them: no unbounded queues, no frontend state as a kernel,
  no sidecar process IPC for voice.

Tokio should not own the realtime hot path. The async runtime schedules control work and applies
backpressure; the model loop, audio callbacks, and media callbacks run on named threads with
bounded queues between them.

Tauri commands are still the webview-to-Rust app bridge. The part that must not exist is a
forked voice worker or process-level IPC below that bridge. Once a voice command reaches Rust,
provider lifecycle and audio flow stay inside the Tauri process.

## Unified Kernel Loop

```mermaid
flowchart TB
  subgraph Webview["SolidJS webview"]
    Button["voice button, stop button, typed input, settings"]
    Render["transcript, speaking state, level meter, errors"]
  end

  subgraph Kernel["Tauri desktop voice kernel"]
    Cmd["voice_* command handlers"]
    CmdQ["bounded tokio::sync::mpsc<RuntimeCommand>\nStart/Status try_send, critical controls send().await"]
    Loop["VoiceRuntime::kernel_loop\nsingle active VoiceSession"]
    Snap["tokio::sync::watch<RuntimeSnapshot>\nrunning, provider, mic, session"]
    UiQ["bounded UiEvents queue\nTauri Channel<VoiceEvent>"]
    Threads["ThreadManager\nnamed OS threads, join, reap, cancel"]
    Settings["Tauri settings + keychain\nprovider, model dir, LiveKit URL/keys"]
  end

  subgraph LFM2["VoiceSession::Lfm2"]
    LfmMain["voice-session thread\nliquid_audio::session_loop"]
    InCb["audio input callback\nnon-blocking push"]
    MicRing["bounded SPSC mic PCM ring"]
    Turn["turn detector / barge-in\nspeaker reference gate"]
    UttQ["bounded utterance queue\ncapacity 1"]
    Infer["lfm2-inference thread\nowns model, processor, ChatState, Mimi"]
    ModelEvents["bounded model event queue\ntext, audio, state"]
    SpkRing["bounded speaker PCM ring"]
    OutCb["audio output callback\nnon-blocking drain"]
  end

  subgraph LiveKit["VoiceSession::Livekit"]
    LkCmd["bounded tokio::sync::mpsc<LiveKitCommand>\nStop, Interrupt, Mic"]
    LkLoop["voice-livekit-session thread\nblock_on tokio::select loop"]
    UserRoom["native user room\nPlatformAudio + microphone track"]
    WebRTC["LiveKit WebRTC media threads\njitter, packets, AEC/NS/AGC path"]
    AgentRoom["native agent room\nsubscribes to user mic"]
    AgentPipe["RealtimePipeline\nLFM2 model -> NativeAudioSource"]
    LkRef["assistant playback reference\nRMS + echo hold gate"]
    RemoteMon["NativeAudioStream monitor\nassistant RMS/state"]
  end

  Button --> Cmd --> CmdQ --> Loop
  Settings --> Loop
  Loop --> Snap --> Render
  Loop --> UiQ --> Render
  Loop --> Threads

  Threads --> LfmMain
  LfmMain --> InCb --> MicRing --> Turn --> UttQ --> Infer
  Infer --> ModelEvents --> UiQ
  Infer --> SpkRing --> OutCb
  SpkRing -. "playback reference" .-> Turn

  Threads --> LkLoop
  Loop --> LkCmd --> LkLoop
  LkLoop --> UserRoom --> WebRTC
  WebRTC --> AgentRoom --> AgentPipe --> WebRTC
  AgentPipe -. "reference threshold" .-> LkRef -.-> AgentRoom
  WebRTC --> RemoteMon --> UiQ

  Loop -. "Stop" .-> LfmMain
  Loop -. "Stop" .-> LkLoop
  Loop -. "Typed input" .-> LfmMain
  Loop -. "Typed input" .-> LkLoop
```

This is the intended desktop shape: one process, one kernel loop, one active provider at a time.
Tokio is Rust's closest equivalent to Python `asyncio` for command handling, settings, LiveKit
signaling, `select!`, `mpsc`, `watch`, and oneshot replies. The realtime audio/model work stays
on named OS threads and native callback/media threads. The boundary between those worlds is a
bounded buffer, not a fork, not HTTP, and not SolidJS state.

## Single Process Map

```mermaid
flowchart TB
  subgraph App["one macOS app process"]
    subgraph Webview["SolidJS webview"]
      UI["prompt input, voice button, settings, transcript, meters"]
    end

    subgraph Rust["Tauri Rust desktop kernel"]
      Cmd["voice_* commands\nintent boundary"]
      CmdQ["bounded RuntimeCommand mpsc\ntry_send for start/status\nasync send for critical controls\noneshot replies"]
      Loop["VoiceRuntime::kernel_loop\nsingle provider router"]
      Snap["watch<RuntimeSnapshot>\nrunning, provider, mic"]
      Events["bounded UiEvents queue\nChannel<VoiceEvent> to UI"]
      Threads["ThreadManager\nnamed OS threads, join, reap"]
    end

    subgraph Local["LFM2-Audio provider"]
      LfmThread["voice-session thread\nliquid_audio::VoiceRuntime"]
      LfmPipe["RealtimePipeline thread\nmodel, Mimi, ChatState"]
      LfmAudio["audio callback threads\nmic/speaker PCM rings"]
    end

    subgraph Remote["LiveKit provider"]
      LkThread["voice-livekit-session thread\nblock_on tokio::select loop"]
      LkMedia["native WebRTC media threads\nroom, tracks, packets"]
      LkAgent["native agent pipeline\nLFM2 model -> NativeAudioSource"]
      LkAudio["PlatformAudio\nAEC, NS, AGC, microphone track"]
    end
  end

  UI --> Cmd --> CmdQ --> Loop
  Loop --> Snap
  Loop --> Events --> UI
  Loop --> Threads
  Threads --> LfmThread
  Threads --> LkThread
  LfmThread <--> LfmAudio
  LfmThread <--> LfmPipe
  LfmPipe --> Events
  LkThread <--> LkMedia
  LkThread <--> LkAgent
  LkAudio --> LkMedia
  LkMedia --> LkAgent
  LkAgent --> LkMedia
  LkMedia --> Events
```

There is no forked voice worker in this map. There is also no HTTP or process IPC layer between
Tauri and the active voice provider. The only webview boundary is the normal Tauri command/event
surface: SolidJS sends intent, and Rust owns the provider loop.

## Kernel Ownership

```mermaid
flowchart TB
  UI["SolidJS webview\nintent + rendering only"]
  Cmd["Tauri commands\nvoice_start / voice_stop / interrupt / mic / settings"]
  Events["Tauri Channel<VoiceEvent>\nstate, transcript, level, error, ended"]
  Settings["Tauri settings + keychain\nprovider, model dir, LiveKit URL + API credentials"]

  subgraph Tauri["Tauri desktop process"]
    subgraph Kernel["VoiceRuntime kernel"]
      Queue["bounded tokio::sync::mpsc<RuntimeCommand>\nStart / Stop / Interrupt / Mic / ApplySettings / InvalidateProvider / TypedInput"]
      Snap["tokio::sync::watch<RuntimeSnapshot>\nrunning, provider, mic, session"]
      Router["single active VoiceSession\nLfm2 | Livekit"]
      Threads["ThreadManager\nspawn, join, reap, cancellation"]
      Cancel["cancellation token / atomic stop flag\nshared with active session"]
    end

    subgraph LFM2["VoiceSession::Lfm2"]
      MicCb["audio input callback thread"]
      MicRing["bounded SPSC mic PCM ring\nnon-blocking push, drop on full"]
      Vad["turn detector / barge-in thread"]
      UttQ["bounded utterance queue\ncapacity 1"]
      Infer["persistent inference thread\nowns model + processor + ChatState + Mimi"]
      LfmEvents["bounded model event queue\ntext, audio, state"]
      SpkRing["bounded SPSC speaker PCM ring"]
      SpkCb["audio output callback thread"]
    end

    subgraph LK["VoiceSession::Livekit"]
      LkCmd["bounded tokio::sync::mpsc<LiveKitCommand>\nStop / Interrupt / Mic"]
      LkCtrl["ThreadManager-owned LiveKit service\nTokio two-room select loop"]
      LkUser["user participant\nPlatformAudio + LocalAudioTrack microphone"]
      LkAgent["native agent participant\nRealtimePipeline + NativeAudioSource"]
      LkMedia["native WebRTC media worker threads"]
      LkMic["NativeAudioStream task\nuser mic PCM -> LFM2 utterances"]
      LkRemote["assistant audio monitor\nRMS + playback/reference audio"]
      LkTimeout["agent-audio timeout\nfail if native agent track does not subscribe"]
    end
  end

  UI --> Cmd --> Queue --> Router
  Settings --> Router
  Router --> Snap
  Snap --> Events --> UI
  Router --> Threads
  Router --> Cancel

  Router --> LFM2
  MicCb --> MicRing --> Vad --> UttQ --> Infer
  Cancel --> Vad
  Cancel --> Infer
  Infer --> LfmEvents --> Events
  Infer --> SpkRing --> SpkCb

  Router --> LK
  LkCmd --> LkCtrl
  Cancel --> LkCtrl
  LkCtrl <--> LkMedia
  LkUser --> LkMedia
  LkMedia --> LkMic --> LkAgent
  LkAgent --> LkMedia
  LkMedia --> LkRemote --> Events
  LkCtrl --> LkTimeout --> Events
```

## Provider Kernel Loop

```mermaid
flowchart TB
  subgraph UI["SolidJS webview"]
    Button["mic button / stop button / voice switch"]
    Transcript["transcript, level meter, speaking state"]
  end

  subgraph Tauri["Tauri desktop process"]
    Command["voice_* command futures"]
    Queue["bounded Tokio mpsc<RuntimeCommand>\nStart uses try_send\nStop/Interrupt/Mic/TypedInput/Settings use send().await"]
    Kernel["VoiceRuntime::kernel_loop\nsingle active session router"]
    Snapshot["watch<RuntimeSnapshot>\nrunning/provider/mic/session"]
    Events["bounded UiEvents queue\nTauri Channel<VoiceEvent>"]
    Threads["ThreadManager\nspawn named threads, join, reap"]

    subgraph LFM2["LFM2 provider thread tree"]
      LfmSession["voice-session thread\nsession loop + cancellation"]
      LfmInput["input callback thread\nPCM -> bounded ring"]
      Turn["turn detector / barge-in\nbounded utterance send"]
      Infer["lfm2-inference thread\nmodel + processor + ChatState + Mimi"]
      LfmOutput["output callback thread\nbounded speaker ring -> device"]
    end

    subgraph LiveKit["Native LiveKit provider thread tree"]
      LkService["voice-livekit-session thread\nblock_on Tokio select loop"]
      Rooms["native LiveKit rooms\nuser participant + agent participant"]
      WebRTC["libwebrtc/media worker threads\ntracks, packets, jitter, AEC path"]
      Agent["native LFM2 agent pipeline\nRealtimePipeline -> NativeAudioSource"]
      Monitor["assistant audio monitor\nRMS/state/events"]
    end

    subgraph Model["Model management"]
      Download["voice-model-download thread\nHF snapshot acquisition, one at a time"]
    end
  end

  Button --> Command --> Queue --> Kernel
  Kernel --> Snapshot --> Transcript
  Kernel --> Events --> Transcript
  Kernel --> Threads

  Threads --> LfmSession
  LfmSession --> LfmInput
  LfmInput --> Turn --> Infer --> LfmOutput
  Infer --> Events

  Threads --> LkService
  LkService --> Rooms
  Rooms <--> WebRTC
  WebRTC --> Agent
  Agent --> WebRTC
  WebRTC --> Monitor --> Events
  Threads --> Download
```

Both providers hang off the same Tauri-owned kernel. LiveKit is not a separate frontend mode; it
is another native session implementation under `VoiceSession`. The UI can choose LFM2 or LiveKit,
but it does not become the owner of the microphone, room, tracks, model loop, or stop semantics.

## Runtime Thread/Task Topology

```mermaid
flowchart LR
  UI["SolidJS\nbuttons, switch, visualizer, transcript"]

  subgraph Process["Tauri desktop process"]
    subgraph Tokio["Tokio / tauri::async_runtime"]
      Cmds["voice_* command futures"]
      Kernel["VoiceRuntime::kernel_loop\nbounded RuntimeCommand receiver"]
      Snap["watch<RuntimeSnapshot>"]
      Fanout["UiEvents task\nbounded UI queue -> Channel<VoiceEvent>"]
    end

    subgraph Managed["ThreadManager-owned OS threads"]
      LfmMain["voice-session\nliquid_audio::session_loop"]
      LKLoop["voice-livekit-session\nblock_on LiveKit tokio::select! loop"]
      LKAgentEvents["voice-livekit-agent-events\nRealtimePipeline events -> NativeAudioSource/UI"]
      LKAgentMic["voice-livekit-agent-mic\nNativeAudioStream -> LFM2 utterances"]
      LKMonitor["voice-livekit-audio-monitor\nassistant RMS/state"]
      StopJoin["voice-lfm2-stop\nblocking stop/join"]
      LKStop["voice-livekit-stop\nsignal stop + join"]
      Download["voice-model-download\nHF snapshot acquisition"]
    end

    subgraph LfmInner["LFM2 realtime worker threads"]
      Consumer["voice-consumer\nmodel events -> UI + speaker ring"]
      Infer["lfm2-inference\nowns LFM2 model, processor, ChatState, Mimi"]
    end

    subgraph Native["Native callback / media threads"]
      InCb["CoreAudio/CPAL input callback"]
      OutCb["CoreAudio/CPAL output callback"]
      RTC["LiveKit WebRTC media threads"]
    end
  end

  UI --> Cmds
  Cmds --> Kernel
  Kernel --> Snap
  Kernel --> Fanout --> UI

  Kernel -- "StartLfm2" --> LfmMain
  Kernel -- "Stop LFM2" --> StopJoin
  LfmMain --> Consumer
  LfmMain --> Infer
  InCb -- "bounded SPSC PCM ring" --> LfmMain
  LfmMain -- "bounded utterance queue, cap 1" --> Infer
  Infer -- "bounded model event queue" --> Consumer
  Consumer -- "bounded SPSC speaker ring" --> OutCb

  Kernel -- "StartLivekit" --> LKLoop
  Kernel -- "LiveKitCommand mpsc, cap 16\nasync send for Interrupt/Mic" --> LKLoop
  Kernel -- "Stop LiveKit" --> LKStop
  LKLoop <--> RTC
  LKLoop --> LKAgentEvents
  RTC --> LKAgentMic --> LKAgentEvents
  RTC --> LKMonitor --> Fanout
  Cmds -- "voice_model_download" --> Download
```

No `fork`, no sidecar, and no HTTP/IPC boundary exists below the Tauri command bridge in this
layout. The two acceptable forms of concurrency are: Tokio tasks for async control/network work,
and named native threads/callbacks for realtime work. Every handoff is bounded; if a queue is full,
the producer gets backpressure or cancellation instead of growing latency.

The shared named-thread owner lives in `src/voice/threads.rs`. `runtime.rs`, `model.rs`, and the
LiveKit helper loops all use that same `ThreadManager` instead of carrying local handle lists.

## Kernel Loop

```mermaid
sequenceDiagram
  participant UI as SolidJS
  participant Tauri as Tauri command handler
  participant Q as bounded RuntimeCommand queue
  participant K as VoiceRuntime loop
  participant S as active VoiceSession
  participant E as VoiceEvent channel

  UI->>Tauri: voice_start(provider context)
  Tauri->>Q: try_send(Start)
  Q->>K: receive Start
  K->>K: load settings + keychain, select provider
  K->>S: spawn LFM2 or LiveKit session under ThreadManager
  S->>E: State(Listening)
  E->>UI: render mic on / active provider

  UI->>Tauri: voice_stop()
  Tauri->>Q: bounded async send Stop
  Q->>K: receive Stop
  K->>S: cancel + close tracks + flush rings
  S->>K: join/reap session workers
  K->>E: Ended("stopped")

  UI->>Tauri: typed input begins
  Tauri->>Q: bounded async send TypedInput
  Q->>K: receive typed-input pause
  K->>S: disable capture and/or interrupt active reply
  S->>E: State(Idle)

  UI->>Tauri: provider/settings changed
  Tauri->>Q: bounded async send ApplySettings
  Q->>K: receive ApplySettings
  K->>S: compare active session config key
  K->>S: stop if provider, model, audio, or LiveKit config is stale
  K->>E: State(Idle)

  UI->>Tauri: LiveKit credentials changed
  Tauri->>Q: bounded async send InvalidateProvider(Livekit)
  Q->>K: receive InvalidateProvider
  K->>S: stop active native LiveKit session
```

Stop is not a UI affordance. It is a kernel command. The stop button must cancel generation,
mute or close capture, flush pending PCM, close LiveKit tracks/room when selected, join owned
threads, and emit one terminal event. If a user types while the mic is on, the same kernel path
should pause capture and interrupt any speaking turn before the typed request runs, so voice does
not keep listening to the user's keyboard-driven interaction or its own output.

## LFM2 Model Loop

```mermaid
sequenceDiagram
  participant Mic as Input Callback Thread
  participant Ring as Mic SPSC Ring
  participant VAD as VAD / Turn Thread
  participant Worker as LFM2 Inference Thread
  participant Model as LFM2 generate_interleaved
  participant Mimi as Mimi Decoder
  participant Out as Speaker SPSC Ring
  participant UI as VoiceEvent Channel

  Mic->>Ring: push PCM frames, never block callback
  VAD->>Ring: read windows
  VAD->>VAD: speech/silence/barge-in detection
  VAD->>Worker: bounded send Utterance
  Worker->>Model: ChatState.add_audio(), generate_interleaved_cancellable()
  Model->>UI: text tokens / transcript deltas
  Model->>Mimi: audio codebooks
  Mimi->>Out: decoded 24 kHz PCM chunks
  Worker->>Worker: append text/audio_out/modality to ChatState
  Out->>Out: output callback drains PCM without blocking
```

The LFM2 thread owns the model, processor, generation state, ChatState, and Mimi decoder. The
audio callbacks do not call into the model. They only push/pull bounded PCM buffers. Barge-in
sets cancellation, clears stale playback, and lets the inference loop return `Interrupted`
instead of stacking old turns.

## LiveKit Native Loop

```mermaid
sequenceDiagram
  participant K as VoiceRuntime
  participant Grant as Native token/config builder
  participant Ctrl as Managed LiveKit Service Thread
  participant Agent as Native LiveKit Agent Participant
  participant Ref as Assistant Playback Reference
  participant Media as WebRTC Media Threads
  participant Mic as Native Mic Track
  participant Remote as Remote Audio Monitor
  participant UI as VoiceEvent Channel

  K->>Grant: build user + agent grants from Tauri settings, keychain, and local LFM2 model
  Grant->>Ctrl: ThreadManager spawn, url, room, user token, agent token
  Ctrl->>Media: user Room::connect()
  Ctrl->>Agent: agent Room::connect() + RealtimePipeline::spawn()
  Mic->>Media: publish user microphone track
  Media->>Agent: subscribed user mic frames via NativeAudioStream
  Agent->>Media: publish assistant PCM via NativeAudioSource
  Agent->>Ref: update assistant RMS + short echo hold
  Ref->>Agent: raise mic VAD threshold during assistant playback
  Media->>Remote: assistant audio frames
  Remote->>UI: level/state events
  Ctrl->>UI: error + ended if native agent track does not subscribe
  K->>Ctrl: Interrupt
  Ctrl->>Agent: RealtimePipelineHandle::interrupt()
  Ctrl->>Media: optional reliable data packet, keep room alive
  K->>Ctrl: Stop
  Ctrl->>Media: unpublish/close tracks, disconnect room
```

LiveKit is still a provider, but it is not a frontend-owned provider. Rust owns room connection,
token/config, mic publication, interrupt packets, stop teardown, and remote-audio monitoring.
LFM2 uses its speaker/playback PCM as reference audio for VAD gating, so the assistant's own
output has to clear a higher echo floor before it is treated as user barge-in. LiveKit enables
WebRTC echo cancellation/noise suppression/AGC through `PlatformAudio`, and the native LiveKit
agent path also mirrors the local playback-reference gate by raising its mic VAD threshold while
assistant PCM is being published through `NativeAudioSource`. A future true AEC pass for the local
CPAL path would be sample-accurate cancellation, not a change in ownership.

## Channel Choices

```text
start/status commands tokio::sync::mpsc, bounded, try_send/backpressure
critical controls     same bounded mpsc, async send().await for Stop/Interrupt/Mic/TypedInput
state snapshot        tokio::sync::watch
UI event stream       tauri::ipc::Channel<VoiceEvent> fed from bounded internal queues
provider completion   internal RuntimeCommand::Reap self-wake from LFM2/LiveKit threads
model downloads       Tauri-managed ModelDownloadRuntime, one active HF snapshot worker
delegated turns       BridgeState-owned async task, one active handoff per voice session
LFM2 utterances       crossbeam_channel::bounded, capacity 1
LFM2 model events     crossbeam_channel::bounded
PCM mic/speaker       custom bounded SPSC PcmRing, atomics + non-blocking callbacks
shutdown/cancel       Arc<AtomicBool> at sync/realtime boundaries
one-shot replies      tokio::sync::oneshot
```

The architecture is therefore not "make everything async." It is Tokio for orchestration, native
threads for realtime work, bounded buffers for audio and events, and SolidJS as a display/control
surface only.
