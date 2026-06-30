You’re right about the desired shape. The voice layer should be one native desktop kernel with two provider backends, not “SolidJS owns LiveKit, Rust owns LFM2.”

Rust’s closest `asyncio` equivalent is **Tokio**. Tauri already has it. But for realtime audio, the split should be:

- **Tokio async tasks** for control-plane work: commands, config, LiveKit signaling, reconnects, token/config loading.
- **Dedicated OS threads** for hot realtime work: LFM2 inference, audio callbacks, WebRTC/media callbacks.
- **Bounded non-blocking channels/rings** between them: not unbounded queues, not frontend state as coordination.

Current gap: LFM2 mostly follows this. LiveKit does not. LiveKit is still JS/webview-owned via `livekit-client`.

**Target Desktop Voice Kernel**
```mermaid
flowchart TB
  UI["SolidJS UI\nbuttons, meters, transcript only"]
  TauriCmd["Tauri voice commands\nvoice_start / stop / interrupt / mic"]
  Kernel["VoiceKernel\nsingle desktop-owned service"]
  Settings["Tauri settings + keychain\nprovider, model, LiveKit creds"]
  Events["Tauri Channel<VoiceEvent>\nUI render stream"]

  UI --> TauriCmd
  TauriCmd --> Kernel
  Settings --> Kernel
  Kernel --> Events
  Events --> UI

  Kernel --> CmdQ["bounded command channel\nStart / Stop / Interrupt / Mic / ProviderChange"]
  Kernel --> State["watch/broadcast state\nrunning, provider, mic, error"]

  CmdQ --> Router["Provider router"]

  Router --> LFM2["LFM2 native session"]
  Router --> LK["LiveKit native session"]

  subgraph LFM2Path["LFM2 Provider - native local model"]
    MicIn["audio input callback thread"]
    MicRing["lock-free mic ring"]
    VAD["VAD / turn detector thread"]
    UttQ["bounded utterance queue"]
    Infer["lfm2-inference thread\nowns model + ChatState + Mimi"]
    OutRing["lock-free speaker ring"]
    SpkOut["audio output callback thread"]

    MicIn --> MicRing --> VAD --> UttQ --> Infer
    Infer --> OutRing --> SpkOut
    Infer --> LFMEvents["text/audio/state events"]
  end

  subgraph LiveKitPath["LiveKit Provider - native Rust/WebRTC client"]
    LKCtrl["Tokio signaling task\nconnect/reconnect/token/room state"]
    LKMedia["WebRTC/media worker threads"]
    LKMicRing["lock-free mic ring"]
    LKOutRing["lock-free remote-audio ring"]
    LKEvents["agent state/transcript/level events"]

    LKCtrl <--> LKMedia
    LKMicRing --> LKMedia
    LKMedia --> LKOutRing
    LKMedia --> LKEvents
  end

  LFMEvents --> Kernel
  LKEvents --> Kernel
```

**LFM2 Kernel Loop**
```mermaid
sequenceDiagram
  participant Mic as Mic Callback Thread
  participant Ring as Mic Ring Buffer
  participant VAD as VAD/Session Loop
  participant Worker as LFM2 Inference Thread
  participant Mimi as Mimi Decoder
  participant Speaker as Speaker Ring
  participant UI as Tauri VoiceEvent Channel

  Mic->>Ring: push PCM frames, never block
  VAD->>Ring: read windows
  VAD->>VAD: detect speech/silence/barge-in
  VAD->>Worker: bounded send Utterance
  Worker->>Worker: ChatState.add_audio()
  Worker->>Worker: generate_interleaved_cancellable()
  Worker->>UI: text/state events
  Worker->>Mimi: audio codes
  Mimi->>Speaker: PCM chunks
  Worker->>Worker: append text/audio_out/modality to ChatState
  Speaker->>Speaker: output callback drains PCM
```

**LiveKit Native Loop**
```mermaid
sequenceDiagram
  participant Kernel as Tauri VoiceKernel
  participant Ctrl as Tokio LiveKit Control Task
  participant Media as Native WebRTC Media Threads
  participant Mic as Mic Ring
  participant Out as Speaker Ring
  participant UI as VoiceEvent Channel

  Kernel->>Ctrl: start room with config/creds
  Ctrl->>Media: establish WebRTC transport
  Mic->>Media: publish local audio track
  Media->>Out: remote agent audio PCM
  Media->>UI: transcription / agent state / levels
  Kernel->>Ctrl: interrupt / stop / mic enabled
  Ctrl->>Media: mute / disconnect / close tracks
```

The rule I’d use:

- SolidJS never owns a `Room`.
- SolidJS never owns mic truth.
- SolidJS never decides provider availability.
- SolidJS sends intent and renders state.
- Tauri owns provider lifecycle.
- LFM2 and LiveKit both become `VoiceProviderSession` implementations under one `VoiceKernel`.

Current code still violates that on the LiveKit side: [voice.tsx](/Volumes/stuff/Projects/agentsdevelopment/emberharmony/packages/app/src/context/voice.tsx:4) imports `livekit-client`, constructs `new Room`, connects it, mutes it, and renders `RoomAudioRenderer`. That keeps SolidJS as the LiveKit media kernel.

For the buffer model, I would not keep the current shape as-is. The current LFM2 path uses `crossbeam_channel::unbounded` for inference events and `Arc<Mutex<Vec/VecDeque>>` around CPAL buffers. The target should be bounded:

- `tokio::sync::mpsc` for command/control.
- `tokio::sync::watch` for current state.
- `tokio::sync::broadcast` or bounded `mpsc` for UI event fanout.
- `ringbuf`/`rtrb` style lock-free SPSC rings for mic and speaker PCM.
- bounded utterance queue, probably size `1`, so barge-in replaces or cancels instead of piling up stale speech.

So yes: Rust has the async equivalent. But the correct architecture is not “make everything async.” It is **Tokio for orchestration, native threads for realtime work, lock-free bounded buffers for audio, and SolidJS as a display/control surface only.**
