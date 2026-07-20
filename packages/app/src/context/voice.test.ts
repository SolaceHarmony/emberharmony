import { describe, expect, test } from "bun:test"

const source = () => Bun.file(new URL("./voice.tsx", import.meta.url)).text()
const repo = new URL("../../../../", import.meta.url)
const root = (path: string) => Bun.file(new URL(path, repo)).text()
const local = (path: string) => Bun.file(new URL(`crates/liquid-audio/${path}`, repo))
const liquid = (path: string) => local(path).text()

function between(text: string, start: string, end: string): string {
  const from = text.indexOf(start)
  const to = text.indexOf(end, from)
  expect(from).toBeGreaterThanOrEqual(0)
  expect(to).toBeGreaterThan(from)
  return text.slice(from, to)
}

describe("desktop voice context boundary", () => {
  test("desktop provider returns the Tauri voice client before constructing a LiveKit room", async () => {
    const text = await source()
    const desktop = text.indexOf("if (isDesktop())")
    const room = text.indexOf("new Room(")

    expect(desktop).toBeGreaterThanOrEqual(0)
    expect(room).toBeGreaterThanOrEqual(0)
    expect(desktop).toBeLessThan(room)
  })

  test("desktop voice start does not pass a LiveKit grant back to SolidJS", async () => {
    const text = await source()

    expect(text).not.toContain("result.grant")
    expect(text).not.toContain("return result.grant")
  })

  test("desktop voice client never uses the LiveKit browser room or server voice status", async () => {
    const text = await source()
    const desktop = between(text, "function createDesktopVoice()", "function createWebVoice")

    expect(desktop).toContain("startVoice(")
    expect(desktop).toContain("stopVoice()")
    expect(desktop).toContain("interruptVoice()")
    expect(desktop).toContain("beginVoiceTypedInput()")
    expect(desktop).toContain("setVoiceMicEnabled(enabled)")
    expect(desktop).not.toContain("new Room(")
    expect(desktop).not.toContain("RoomAudioRenderer")
    expect(desktop).not.toContain("sdk.client.voice.status")
    expect(desktop).not.toContain("sdk.client.voice.token")
    expect(desktop).not.toContain("room.connect(")
    expect(desktop).not.toContain("localParticipant")
  })

  test("desktop connect lets the native kernel reject unready providers", async () => {
    const text = await source()
    const connect = between(text, "async function connect(sessionID: string", "async function disconnect()")

    expect(connect).toContain("const current = await refresh()")
    expect(connect).toContain("const plan = current?.plan")
    expect(connect).toContain("const result = await startVoice(")
    expect(connect).toContain("await refresh().catch(() => undefined)")
    expect(connect).not.toContain("if (!plan?.ready)")
    expect(connect).not.toContain("Voice is not ready")
    expect(connect).not.toContain("mark(true, true, plan.provider)")
  })

  test("desktop connect does not synthesize connected state after start succeeds", async () => {
    const text = await source()
    const connect = between(text, "async function connect(sessionID: string", "async function disconnect()")
    const start = connect.slice(connect.indexOf("const result = await startVoice("))

    expect(start).toContain("await refresh().catch(() => undefined)")
    expect(start).not.toContain('setState("connected")')
    expect(start).not.toContain('setAgent("listening")')
  })

  test("platform capture callbacks hand borrowed blocks directly to the native dock", async () => {
    const api = await liquid("src/voice_api.rs")
    const runtime = await liquid("src/runtime/voice_runtime.rs")
    const native = await liquid("src/native_voice.rs")
    const input = between(runtime, "fn start_input(", "fn start_output")
    const dock = between(native, "impl NativeCaptureSink", "impl CaptureSink for NativeCaptureSink")

    expect(api).toContain("pub trait CaptureSink: Send")
    expect(api).toContain("fn max_callback_frames(&self) -> u32")
    expect(api).toContain("fn write_f32(&mut self, input: &[f32], channels: usize)")
    expect(api).toContain("fn write_i16(&mut self, input: &[i16], channels: usize)")
    expect(api).toContain("fn write_u16(&mut self, input: &[u16], channels: usize)")
    expect(input).toContain("dev.build_input_stream(")
    expect(input).toContain("cpal::BufferSize::Fixed(expected_request_frames)")
    expect(input).toContain("sealed != expected_max_callback_frames")
    expect(input).toContain("gate_capture_callback(")
    expect(input).toContain("callback_fault.fetch_or(DEVICE_FAULT_INPUT")
    expect(input).toContain("sink.$write(data, channels)")
    expect(dock).toContain("lfm_capture_producer_write_interleaved(")
    expect(dock).toContain("input.as_ptr().cast()")
    expect(dock).not.toContain("to_vec")
    expect(input).not.toContain("Mutex")
    expect(input).not.toContain("send(")
    expect(input).not.toContain("recv")
    expect(runtime).not.toContain("wait_for_input")
    expect(runtime).not.toContain("recv_timeout")
    expect(native).not.toContain("CaptureReservation")
  })

  test("native voice progress is a retained edge-resumed continuation", async () => {
    const text = await liquid("src/runtime/voice_runtime.rs")
    const live = between(text, "impl LiveTask", "impl Drop for LiveTask")

    expect(text).toContain("CoroutineRuntime::with_config(")
    expect(text).toContain(".owner_state_service_factory(|setup|")
    expect(text).toContain("let events = setup.realtime_notifier()?")
    expect(text).toContain("engine.mount_events(events)")
    expect(live).toContain(".advance_events(")
    expect(live).toContain("EngineProgress::Continue")
    expect(live).toContain("EngineProgress::Dormant")
    expect(live).toContain("ServiceOutcome::Dormant")
    expect(text).not.toContain("wait_for_input")
    expect(text).not.toContain("recv_timeout")
    expect(text).not.toContain("VadTask")
    expect(text).not.toContain("FrameTask")
  })


  test("native control commands resume the retained mic continuation immediately", async () => {
    const text = await liquid("src/runtime/voice_runtime.rs")
    const runtime = between(text, "pub struct VoiceRuntime", "impl Drop for VoiceRuntime")

    expect(runtime).toContain("runtime: Arc<CoroutineRuntime>")
    expect(runtime).toContain("service: Option<CoroutineService>")
    expect(runtime).toContain("control: SharedRealtimeNotifier")
    expect(runtime).toContain(".owner_state_service_factory(|setup|")
    expect(runtime).toContain("pub fn interrupt(&self) -> Result<(), String>")
    expect(runtime).toContain("native interrupt edge was rejected")
    expect(runtime).toContain("pub fn set_mic_enabled(&self, on: bool) -> Result<(), String>")
    expect(runtime).toContain("native mic-control edge was rejected")
    expect(runtime).toContain("pub fn stop(mut self)")
    expect(runtime).not.toContain("wake_control")
  })

  test("desktop LFM2 uses liquid-audio direct platform callbacks", async () => {
    const audio = await liquid("src/runtime/voice_runtime.rs")
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const session = between(runtime, "impl Lfm2Session", "fn is_finished")
    const input = between(audio, "fn start_input(", "fn start_output")
    const output = between(audio, "fn start_output(", "#[cfg(test)]")

    expect(audio).toContain("pub fn prepare(")
    expect(audio).toContain("start_input(")
    expect(audio).toContain("capture,")
    expect(audio).toContain("struct OwnerSession")
    expect(audio).toContain("streams: Option<DeviceStreams>")
    expect(audio).not.toContain("thread_local!")
    expect(input).toContain("dev.build_input_stream(")
    expect(input).toContain("sink.$write(data, channels)")
    expect(output).toContain("dev.build_output_stream(")
    expect(output).toContain("source.$write(data, channels, reset)")
    expect(session).toContain("let live = Lfm2Runtime::prepare(")
    expect(session).not.toContain(".await")
    expect(session).not.toContain("tokio::time")
    expect(runtime).not.toContain("ExternalAudioInput")
    expect(runtime).not.toContain("ExternalAudioOutput")
    expect(runtime).not.toContain("NativeAudioStream")
    expect(runtime).not.toContain("local_webrtc_input")
    expect(runtime).not.toContain("start_local_webrtc_input")
    expect(runtime).not.toContain("start_local_webrtc_output")
    expect(runtime).not.toContain("prepare_with_io")
  })

  test("desktop enables liquid-audio platform audio callbacks", async () => {
    const cargo = await root("packages/desktop/src-tauri/Cargo.toml")
    const lib = await liquid("src/lib.rs")
    const voice = await liquid("src/runtime/voice_runtime.rs")

    expect(cargo.match(/liquid-audio = \{ path = "\.\.\/\.\.\/\.\.\/crates\/liquid-audio", features = \["audio-io"\] \}/g)).toHaveLength(2)
    expect(cargo).not.toContain('features = ["metal"]')
    expect(cargo).not.toContain("candle-core")
    expect(lib).toContain("pub mod voice_runtime")
    expect(lib).not.toContain('#[cfg(feature = "audio-io")]\npub mod voice_runtime')
    expect(voice).toContain('#[cfg(feature = "audio-io")]\nuse cpal::traits')
    expect(voice).toContain("cpal::default_host()")
    expect(voice).toContain("liquid-audio was built without platform audio support")
  })









  test("native Sesame owns separate microphone and playback evidence state", async () => {
    const runtime = await liquid("src/runtime/voice_runtime.rs")
    const header = await liquid("native/include/lfm_sesame_detector.h")
    const detector = await liquid("native/src/runtime/lfm_sesame_detector.cpp")

    expect(header).toContain("LFM_SESAME_STREAM_MIC 1u")
    expect(header).toContain("LFM_SESAME_STREAM_PLAYBACK 2u")
    expect(header).toContain("LFM_SESAME_MIC_THRESHOLD 50u")
    expect(header).toContain("LFM_SESAME_PLAYBACK_THRESHOLD 10u")
    expect(detector).toContain("StreamState mic")
    expect(detector).toContain("StreamState playback")
    expect(detector).toContain("lfm_sesame_selected_magnitudes(")
    expect(detector).toContain("lfm_sesame_magnitudes_to_bytes(")
    expect(detector).toContain("lfm_sesame_classify_selected_bytes(")
    expect(runtime).not.toContain("vad_threshold")
    expect(runtime).not.toContain("PlaybackReference")
    expect(runtime).not.toContain("PLAYBACK_ECHO_MULTIPLIER")
  })

  test("native LFM2 rejects delegation until native prompt control exists", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")

    expect(text).toContain("Delegation requires a configurable native system-prompt command")
    expect(text).toContain("Native LFM2 will not instantiate the Candle engine as a fallback")
    expect(text).not.toContain("LFM2_CONVERSE_SYSTEM_PROMPT")
    expect(text).not.toContain("engine.with_system_prompt")
    expect(text).toContain("settings.lfm2.delegate.enabled")
    expect(text).toContain(".delegate")
    expect(text).toContain(".target")
    expect(text).toContain('eq_ignore_ascii_case("DELEGATE:")')
    expect(text).toContain('line.get(.."DELEGATE:".len())')
    expect(text).not.toContain('let marker = line.find("DELEGATE:")')
  })

  test("desktop voice control is routed through a bounded Tauri kernel queue", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")

    expect(text).toContain("const VOICE_COMMAND_CAP")
    expect(text).toContain("mpsc::channel(VOICE_COMMAND_CAP)")
    expect(text).toContain("watch::channel(RuntimeSnapshot::default())")
    expect(text).toContain("enum RuntimeCommand")
    expect(text).not.toContain("session: Mutex<Option<VoiceSession>>")
  })

  test("critical desktop voice controls use bounded async delivery instead of full-queue rejection", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const critical = between(text, "async fn request_critical", "impl Default for VoiceRuntime")
    const ordinary = between(text, "async fn request<T>", "async fn request_critical")

    expect(critical).toContain(".send(cmd(reply))")
    expect(critical).toContain(".await")
    expect(critical).toContain('"voice kernel stopped"')
    expect(ordinary).toContain(".try_send(cmd(reply))")
    expect(ordinary).toContain('"voice kernel command queue is full"')
    expect(text).toContain("pub async fn stop(&self) -> Result<(), String>")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::Stop { reply })")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::Interrupt { reply })")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::SetMicEnabled { enabled, reply })")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::BeginTypedInput { reply })")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::ApplySettings { settings, reply })")
    expect(text).toContain("self.request_critical(|reply| RuntimeCommand::InvalidateProvider { provider, reply })")
  })

  test("desktop voice events fan out through a bounded native buffer", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")

    expect(text).toContain("type UiChannel = Arc<UiEvents>")
    expect(text).toContain("const UI_EVENT_CAP")
    expect(text).toContain("struct UiEvents")
    expect(text).toContain("mpsc::channel(UI_EVENT_CAP)")
    expect(text).toContain("self.tx.try_send(event)")
    expect(text).toContain("impl Drop for UiEvents")
    expect(text).toContain("self.task.abort()")
    expect(text).toContain("UiEvents::new(channel)")
    expect(text).not.toContain("Arc<Mutex<tauri::ipc::Channel<VoiceEvent>>>")
    expect(text).not.toContain(".map(|channel| channel.send(event).is_ok())")
  })

  test("retained service completion wakes the Tauri kernel snapshot", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")

    expect(text).toContain("tauri::async_runtime::spawn(kernel_loop(")
    expect(text).toContain("async fn kernel_loop(")
    expect(text).toContain("while let Some(cmd) = rx.recv().await")
    expect(text).not.toContain("blocking_recv()")
    expect(text).not.toContain('.name("voice-kernel".into())')
    expect(text).toContain("Reap")
    expect(text).toContain("fn wake_kernel(commands: &mpsc::Sender<RuntimeCommand>)")
    expect(text).toContain("let _ = commands.try_send(RuntimeCommand::Reap)")
    expect(text).toContain("RuntimeCommand::Reap =>")
    expect(text).toContain("wake: mpsc::Sender<RuntimeCommand>")
    expect(text).toContain("wake_kernel(&sink_wake)")
    expect(text).not.toContain('threads.spawn("voice-session", move ||')
    expect(text).not.toContain('threads.spawn("voice-livekit-session", move ||')
  })

  test("provider reaping follows retained LFM2 service completion", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const cleanup = between(text, "fn cleanup_finished", "fn publish_snapshot")
    const lfm2 = between(text, "impl Lfm2Session", "impl Drop for Lfm2Session")
    const lfm2Finished = between(lfm2, "fn is_finished(&self) -> bool", "fn interrupt")

    expect(cleanup).toContain("VoiceSession::is_finished")
    expect(lfm2Finished).toContain("Lfm2Runtime::is_finished")
    expect(text).toContain("struct VoiceSession(Lfm2Session)")
    expect(text).not.toContain("LiveKitSession")
  })

  test("desktop mic state follows the native runtime snapshot", async () => {
    const text = await source()
    const desktop = between(text, "function createDesktopVoice()", "function createWebVoice")

    expect(desktop).toContain("() => native()?.plan")
    expect(desktop).toContain("if (plan.running)")
    expect(desktop).toContain('return native()?.plan.micEnabled ? "unmuted" : "muted"')
    expect(desktop).toContain('state() !== "connected"')
    expect(desktop).toContain('state() !== "connecting"')
    expect(desktop).toContain('agent() !== "thinking"')
    expect(desktop).toContain('agent() !== "speaking"')
    expect(desktop).toContain("clear()")
    expect(desktop).not.toContain("function mark(")
    expect(desktop).not.toContain("setMic(")
    expect(desktop).not.toContain("runningProvider: active, micEnabled")
  })

  test("desktop mic controls refresh native runtime truth after commands", async () => {
    const text = await source()
    const desktop = between(text, "function createDesktopVoice()", "function createWebVoice")
    const mic = between(desktop, "async function setMicEnabled", "async function beginTypedInput")
    const typed = between(desktop, "async function beginTypedInput", "const micState")

    expect(mic).toContain("await setVoiceMicEnabled(enabled)")
    expect(mic).toContain("await refresh().catch(() => undefined)")
    expect(mic).not.toContain("setMic(")
    expect(mic).not.toContain("mark(true")
    expect(typed).toContain("await beginVoiceTypedInput()")
    expect(typed).toContain("await refresh().catch(() => undefined)")
    expect(typed).not.toContain("setMic(")
    expect(typed).not.toContain("mark(true")
  })

  test("prompt voice meter uses native level before LiveKit media tracks", async () => {
    const text = await root("packages/app/src/components/prompt-input.tsx")
    const visualizer = between(
      text,
      '<Show when={voice.state() === "connected"}>',
      '<Show when={store.mode === "normal" && params.id}>',
    )

    expect(visualizer).toContain("voice.agentLevel() !== undefined")
    expect(visualizer).toContain("<NativeVoiceMeter level={voice.agentLevel() ?? 0}")
    expect(visualizer).toContain("voice.agentAudioTrack()")
    expect(visualizer).toContain("<BarVisualizer")
    expect(visualizer.indexOf("voice.agentLevel() !== undefined")).toBeLessThan(
      visualizer.indexOf("voice.agentAudioTrack()"),
    )
  })

  test("desktop voice commands await the Tauri runtime kernel", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const lfm2 = await liquid("src/runtime/voice_runtime.rs")
    const lfm2Session = between(runtime, "struct Lfm2Session", "impl Lfm2Session")
    const lfm2Stop = between(runtime, "fn stop_lfm2(", "fn f32_to_i16")

    expect(text).toContain("runtime.snapshot().await")
    expect(text).toContain("runtime.start_lfm2(ctx, settings, channel, bridge).await?")
    expect(text).toContain(".start_livekit(ctx.clone(), settings, grant, channel, bridge)")
    expect(text).toContain("runtime.stop().await")
    expect(runtime).toContain("session.stop(threads)?")
    expect(runtime).toContain("let live = Lfm2Runtime::prepare(")
    expect(runtime).not.toContain('threads.spawn("voice-session", move ||')
    expect(runtime).toContain("fn stop_lfm2(threads: &ThreadManager, session: Lfm2Session)")
    expect(runtime).toContain("stop_lfm2(threads, self.0)")
    expect(lfm2Stop).toContain("let stopped = session.stop();")
    expect(lfm2Stop).toContain("let joined = threads.wait();")
    expect(lfm2Stop).toContain("match (stopped, joined)")
    expect(lfm2Stop).not.toContain("session.stop()?")
    expect(runtime).not.toContain('threads.spawn("voice-lfm2-stop"')
    expect(runtime).not.toContain("LiveKitSession")
    expect(runtime).toContain("LiveKit voice inference was removed")
    expect(runtime).toContain("threads.wait()")
    expect(lfm2).toContain("pub fn prepare(")
    expect(lfm2).toContain("service: Option<CoroutineService>")
    expect(lfm2).toContain(".owner_state_service_factory(|setup|")
    expect(lfm2Session).not.toContain("mic: Option<AsyncTask>")
    expect(lfm2Session).not.toContain("ExternalAudio")
    expect(lfm2Session).not.toContain("NativeAudioStream")
    expect(runtime).not.toContain("VoiceSession::Livekit(session) => session.stop().await")
    expect(runtime).not.toContain("LiveKit stop task failed")
    expect(text).toContain("runtime.begin_typed_input().await")
    expect(runtime).not.toContain('"voice-session-stop"')
    expect(runtime).not.toContain("tauri::async_runtime::spawn_blocking(move || session.stop())")
    expect(text).not.toContain("runtime.active_provider()")
  })

  test("typed input is a single kernel command that pauses mic and interrupts voice", async () => {
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const lfm2 = await liquid("src/runtime/voice_runtime.rs")
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const lib = await root("packages/desktop/src-tauri/src/lib.rs")
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const voice = await source()

    expect(runtime).toContain("pub async fn begin_typed_input(&self)")
    expect(runtime).toContain("RuntimeCommand::BeginTypedInput")
    expect(runtime).toContain("session.set_mic_enabled(false)")
    expect(runtime).toContain("session.interrupt()")
    expect(runtime).toContain("fn emit_ready(&self) -> Result<(), String>")
    expect(runtime).toContain("self.emit_ready()")
    expect(control).toContain("voice_begin_typed_input")
    expect(control).toContain("runtime.begin_typed_input().await")
    expect(lib).toContain("voice::control::voice_begin_typed_input")
    expect(bridge).toContain('invoke<void>("voice_begin_typed_input")')
    expect(voice).toContain("beginTypedInput: () => Promise<void>")
    expect(voice).toContain("await beginVoiceTypedInput()")
    expect(voice).toContain('event.state === "idle"')
    expect(voice).toContain("!running()")
    expect(voice).toContain('"disconnected"')
    expect(voice).toContain('"connected"')
    expect(runtime).toContain("VoiceState::Idle")
    expect(runtime).toContain("VoiceEvent::Level { rms: 0.0 }")
    expect(runtime).not.toContain("reset_livekit_audio_state_or_done")
    expect(lfm2).toContain("emit_ready(sink, stop, mic_enabled)")
    expect(lfm2).toContain("RuntimeEvent::Level(0.0)")
    expect(lfm2).toContain("VoiceEvent::TurnComplete | VoiceEvent::Interrupted")
    expect(lfm2).toContain("fn ready_state(mic_enabled: &AtomicBool) -> SessionState")
    expect(lfm2).toContain("fn emit_ready")
    expect(lfm2).toContain("terminal_turn_state_follows_mic_enabled")
    expect(lfm2).toContain("ready_transition_clears_output_level")
    expect(lfm2).toContain("mic_enabled.load(Ordering::SeqCst)")
    expect(lfm2).toContain("SessionState::Idle")
  })

  test("native voice delegation does not trust the webview selected agent for build access", async () => {
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")

    expect(control).toContain('agent: Some("plan".to_string())')
    expect(control).toContain("session_bridge_defaults_voice_delegation_to_plan")
    expect(control).toContain("session_bridge_does_not_trust_webview_model_override")
    expect(control).toContain("desktop kernel must not trust the webview")
    expect(control).toContain("model: parse_model_ref(target)")
    expect(control).not.toContain("agent: ctx.prompt_mode.clone().or_else(|| ctx.agent.clone())")
    expect(control).not.toContain("parse_model_ref(target).or_else")
    expect(control).not.toContain("ctx.model.as_ref().map")
  })

  test("native LFM2 start only touches the session bridge when Tauri settings enable delegation", async () => {
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const start = between(control, "pub async fn voice_start", "async fn lfm2_bridge_config")
    const branch = between(start, "VoiceProvider::Lfm2 =>", "VoiceProvider::Livekit =>")
    const livekit = between(start, "VoiceProvider::Livekit =>", "VoiceProvider::Off =>")
    const helper = between(control, "async fn lfm2_bridge_config", "fn session_bridge_config")

    expect(branch).toContain("let bridge = lfm2_bridge_config(&settings, &ctx, &server).await?")
    expect(branch).not.toContain("server.status.clone().await")
    expect(livekit).toContain("let bridge = lfm2_bridge_config(&settings, &ctx, &server).await?")
    expect(livekit).toContain(".start_livekit(ctx.clone(), settings, grant, channel, bridge)")
    expect(livekit).not.toContain("server.status.clone().await")
    expect(helper).toContain("settings.lfm2.delegate.enabled")
    expect(helper).toContain(".delegate")
    expect(helper).toContain(".target")
    expect(helper).toContain(".map(str::trim)")
    expect(helper).toContain(".filter(|target| !target.is_empty())")
    expect(helper).toContain("return Ok(None)")
    expect(helper).toContain("server")
    expect(helper).toContain(".status")
    expect(helper).toContain(".await")
    expect(helper.indexOf("return Ok(None)")).toBeLessThan(helper.indexOf(".status"))
    expect(control).not.toContain("pub delegate_target")
  })

  test("desktop start context does not carry the native delegate target from SolidJS", async () => {
    const voice = await source()
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const desktop = between(voice, "function createDesktopVoice()", "function createWebVoice")
    const start = between(bridge, "export interface VoiceStartContext", "export type VoiceStartResult")

    expect(start).not.toContain("delegateTarget")
    expect(desktop).not.toContain("const delegateTarget")
    expect(desktop).not.toContain("delegateTarget,")
    expect(desktop).toContain("startVoice(")
    expect(desktop).toContain("directory: sdk.directory")
  })

  test("desktop provider settings reconciliation is owned by the Tauri runtime", async () => {
    const voice = await source()
    const settings = await root("packages/desktop/src-tauri/src/settings.rs")
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const refresh = between(voice, "const refreshSettings = () =>", "window.addEventListener")
    const key = between(runtime, "struct SessionSettingsKey", "struct VoiceSession")

    expect(voice).not.toContain("shouldStopRuntimeForProviderChange")
    expect(refresh).toContain("refresh().catch(() => {})")
    expect(refresh).not.toContain("disconnect()")
    expect(refresh).not.toContain("stopVoice()")
    expect(settings).toContain("runtime: State<'_, VoiceRuntime>")
    expect(settings).toContain("runtime.apply_settings(settings.clone()).await?")
    expect(settings.indexOf("runtime.apply_settings(settings.clone()).await?")).toBeLessThan(
      settings.indexOf("store.set(VOICE_KEY, value)"),
    )
    expect(runtime).toContain("pub async fn apply_settings(&self, settings: VoiceSettings)")
    expect(runtime).toContain("RuntimeCommand::ApplySettings")
    expect(runtime).toContain("apply_settings_to_session(&mut session, &threads, settings)")
    expect(runtime).toContain("!session.matches_settings(&settings)")
    expect(runtime).toContain("SessionSettingsKey::lfm2(&settings)")
    expect(key).toContain("provider: VoiceProvider")
    expect(key).toContain("lfm2: Lfm2Settings")
    expect(key).not.toContain("LiveKitSettings")
    expect(key).not.toContain("last_provider")
    expect(runtime).toContain("return stop_session(session, threads)")
  })








  test("native voice delegation cancels server handoff when the UI event queue closes", async () => {
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const bridge = between(runtime, "impl BridgeState", "fn delegate_task")
    const helper = between(runtime, "fn send_or_cancel", "fn send_runtime")
    const task = between(bridge, 'self.threads.spawn("voice-delegate-turn"', "fn reap_delegate")

    expect(helper).toContain("cancel.cancel()")
    expect(task).toContain("tauri::async_runtime::block_on(async move")
    expect(task).toContain("if !send_scoped(")
    expect(task).toContain("return;")
    expect(task).toContain("let delta_cancel = cancel.clone()")
    expect(task).toContain("run_turn(cfg, task, cancel.clone()")
    expect(task).toContain("send_scoped(")
    expect(task).toContain("VoiceEvent::Transcript")
    expect(task).toContain("VoiceEvent::Error")
    expect(task).toContain("VoiceState::Listening")
    expect(task).toContain("Err(message)")
    expect(task).toContain("DelegateDone")
    expect(runtime).toContain("impl Drop for DelegateDone")
    expect(runtime).toContain("wake_kernel(&self.wake)")
    expect(task).toContain("self.task = Some(DelegateTask { done })")
    expect(task).not.toContain("let _ = send(")
    expect(task).not.toContain("send(\n                        &delta_channel")
  })












  test("desktop LiveKit token and credential ownership lives in Tauri", async () => {
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const livekit = await root("packages/desktop/src-tauri/src/voice/livekit.rs")
    const lib = await root("packages/desktop/src-tauri/src/lib.rs")
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const settings = await root("packages/app/src/components/settings-voice.tsx")

    expect(control).toContain("livekit::configured(&settings)?")
    expect(control).toContain("let local_ready = local_model_ready(&settings)?")
    expect(control).toContain("livekit::configured(&settings)? && local_ready")
    expect(control).toContain("livekit::grant(&settings, &ctx).await?")
    expect(control).toContain("LIVEKIT_READY_DETAIL")
    expect(control).not.toContain("waiting for the external emberharmony-voice agent")
    expect(control).not.toContain("LiveKitTokenRequest")
    expect(control).not.toContain("/voice/token")
    expect(livekit).toContain("AccessToken::with_api_key")
    expect(livekit).toContain("VideoGrants")
    expect(livekit).toContain("let agent_identity = format!")
    expect(livekit).toContain("agent_token")
    expect(livekit).not.toContain("LIVEKIT_AGENT_NAME")
    expect(livekit).not.toContain("RoomConfiguration")
    expect(livekit).not.toContain("RoomAgentDispatch")
    expect(livekit).not.toContain("AgentDispatchClient")
    expect(livekit).not.toContain("RoomClient")
    expect(livekit).not.toContain("ensure_agent_dispatched")
    expect(livekit).not.toContain("CreateAgentDispatchRequest")
    expect(livekit).toContain("voice_livekit_credentials_set")
    expect(livekit).toContain("voice_livekit_credentials_status")
    expect(livekit).toContain("runtime: State<'_, VoiceRuntime>")
    expect(livekit).toContain("runtime.invalidate_provider(VoiceProvider::Livekit).await")
    expect(lib).toContain("voice::livekit::voice_livekit_credentials_set")
    expect(lib).toContain("voice::livekit::voice_livekit_credentials_status")
    expect(bridge).toContain("voice_livekit_credentials_set")
    expect(bridge).toContain("voice_livekit_credentials_status")
    expect(bridge).toContain("window.dispatchEvent(new CustomEvent(VOICE_SETTINGS_CHANGED, { detail: undefined }))")
    expect(settings).toContain("getLiveKitCredentialsStatus")
    expect(settings).toContain("setLiveKitCredentials(key, secret)")
    expect(settings).toContain("const status = await getVoiceStatus()")
  })

  test("native session bridge decodes complete SSE frames before UTF-8 text", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/session.rs")

    expect(text).toContain("buffer: Vec<u8>")
    expect(text).toContain("stream.buffer.extend_from_slice(&chunk)")
    expect(text).toContain("fn sse_boundary(buffer: &[u8]) -> Option<(usize, usize)>")
    expect(text).toContain("std::str::from_utf8(&bytes[..boundary])")
    expect(text).toContain("sse_stream_preserves_utf8_across_network_chunks")
    expect(text).toContain("drain_event_accepts_crlf_sse_boundaries")
    expect(text).not.toContain("String::from_utf8_lossy(&chunk)")
  })

  test("native session bridge bounds delegated SSE and HTTP error bodies", async () => {
    const text = await root("packages/desktop/src-tauri/src/voice/session.rs")

    expect(text).toContain("const SESSION_BRIDGE_CONNECT_TIMEOUT_SECS")
    expect(text).toContain("const SESSION_BRIDGE_READ_TIMEOUT_SECS")
    expect(text).toContain("const SESSION_BRIDGE_ERROR_BODY_CAP")
    expect(text).toContain("const SESSION_BRIDGE_SSE_BUFFER_CAP")
    expect(text).toContain("tokio::time::timeout(")
    expect(text).toContain("session event stream did not connect")
    expect(text).toContain("struct CancelSignal")
    expect(text).toContain("generation.fetch_add(1, Ordering::AcqRel)")
    expect(text).toContain("notified.as_mut().enable()")
    expect(text).toContain("_ = &mut cancelled")
    expect(text).not.toContain("wait_cancel")
    expect(text).toContain("stream.buffer.len().saturating_add(chunk.len())")
    expect(text).toContain("session event frame exceeded")
    expect(text).toContain("async fn capped_error_body")
    expect(text).toContain("fn utf8_prefix")
    expect(text).toContain("error.valid_up_to()")
    expect(text).toContain("utf8_prefix_cuts_at_complete_char_boundary")
    expect(text).toContain("sse_stream_rejects_unbounded_frame_without_boundary")
    expect(text).not.toContain("response.text().await")
  })

  test("desktop sidecar opts out of the legacy voice worker runtime", async () => {
    const lib = await root("packages/desktop/src-tauri/src/lib.rs")
    const worker = await root("packages/emberharmony/src/voice/worker.ts")
    const build = await root("packages/desktop/scripts/build-local.ts")

    expect(lib).toContain('env("EMBERHARMONY_DESKTOP_NATIVE_VOICE", "1")')
    expect(lib).not.toContain("EMBERHARMONY_VOICE_RUNTIME_DIR")
    expect(lib).not.toContain("resources/voice")
    expect(lib).not.toContain("agent.js")
    expect(lib).not.toContain("voice runtime not bundled")
    expect(worker).toContain('process.env["EMBERHARMONY_DESKTOP_NATIVE_VOICE"] === "1"')
    expect(worker).toContain("legacy voice agent worker not started")
    expect(build).toContain("desktop bundle no longer ships a separate LiveKit Node voice runtime")
  })

  test("settings keep the selected native provider separate from voice enablement", async () => {
    const settings = await root("packages/app/src/components/settings-voice.tsx")
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const rust = await root("packages/desktop/src-tauri/src/settings.rs")

    expect(bridge).toContain("lastProvider?: Exclude<VoiceProvider")
    expect(bridge).toContain('lastProvider: "lfm2"')
    expect(settings).toContain("const rememberedProvider = ()")
    expect(settings).toContain("if (!enabled())")
    expect(settings).toContain("const settings = { ...base, lastProvider: next }")
    expect(settings).toContain("const settings = { ...base, provider: next, lastProvider }")
    expect(rust).toContain("pub last_provider: Option<VoiceProvider>")
    expect(rust).toContain("fn default_last_provider() -> Option<VoiceProvider>")
    expect(rust).toContain('assert_eq!(json["lastProvider"], "lfm2")')
    expect(rust).toContain('assert_eq!(json["lfm2"]["engine"], "lfm2Interleaved")')
    expect(rust).toContain("stored_voice_settings_without_engine_use_native_lfm2_default")
  })

  test("desktop voice settings save LiveKit configuration only through Tauri", async () => {
    const settings = await root("packages/app/src/components/settings-voice.tsx")
    const config = between(settings, "const [config, { refetch }]", "const [tauriVoice")
    const provider = between(settings, "const provider = (): VoiceProvider =>", "const enabled = ()")
    const toggle = between(settings, "async function toggleVoice", "async function selectProvider")
    const select = between(settings, "async function selectProvider", "async function updateLfm2")
    const update = between(settings, "async function update(patch", "async function saveConnection")
    const save = between(settings, "async function saveConnection", "async function testConnection")

    expect(config).toContain("desktop")
    expect(config).toContain("Promise.resolve(undefined)")
    expect(provider).toContain('if (!desktop && livekitConfigured()) return "livekit"')
    expect(toggle).toContain('if (!desktop && previous === "livekit" && next !== "livekit")')
    expect(toggle).toContain('if (!desktop && next === "livekit")')
    expect(select).toContain('if (!desktop && previous === "livekit" && next !== "livekit")')
    expect(select).toContain('if (!desktop && next === "livekit")')
    expect(update).toContain("if (desktop)")
    expect(update).toContain('await updateLiveKit(patch as Partial<VoiceSettings["livekit"]>)')
    expect(update).toContain("globalSDK.client.voice")
    expect(settings).toContain("const showNativeModel = ()")
    expect(settings).toContain('activeProvider() === "lfm2" || (desktop && activeProvider() === "livekit")')
    expect(settings).toContain("const showLegacyLiveKitModels = () => !desktop")
    expect(settings).toContain("const statusPoll = desktop ? setInterval(() => refetchStatus(), 1000) : undefined")
    expect(settings).toContain("clearInterval(statusPoll)")
    expect(settings).toContain("<Show when={showNativeModel()}>")
    expect(settings).toContain("<Show when={showLegacyLiveKitModels()}>")
    expect(save).toContain("await updateLiveKit({ url: nextUrl || undefined })")
    expect(save).toContain("await setLiveKitCredentials(key, secret)")
    expect(save).toContain("if (!desktop)")
    expect(save).toContain("globalSDK.client.auth.set")
    expect(save).toContain("globalSDK.client.voice.configUpdate")
    expect(settings).toContain("if (status.ready)")
    expect(settings).not.toContain('status.provider === "livekit" && status.ready')
    expect(settings).not.toContain("transitional worker")
    expect(settings).not.toContain("sidecar config")
  })

  test("desktop local model download persists the selected native snapshot before reporting success", async () => {
    const settings = await root("packages/app/src/components/settings-voice.tsx")
    const voice = await root("packages/app/src/lib/voice-settings.ts")
    const download = between(settings, "async function downloadModel", "const updateDelegate")

    expect(voice).toContain('export const DEFAULT_MOSHI_MODEL = "kyutai/moshiko-candle-bf16"')
    expect(download).toContain("const engine = localEngine()")
    expect(download).toContain("const revision = hfRevision()")
    expect(download).toContain("await downloadVoiceModel({ source, revision }")
    expect(download).toContain('case "done":')
    expect(download).toContain('engine === "moshiRealtime"')
    expect(download).toContain("await updateLfm2({ moshiModelDir: event.dir })")
    expect(download).toContain("await updateLfm2({ modelDir: event.dir })")
    expect(download).toContain("setMoshiModelDirEdit(event.dir)")
    expect(download).toContain("setModelDirEdit(event.dir)")
    expect(download).toContain("if (!saved) return")
    expect(download).toContain(
      'showToast({ title: language.t("settings.voice.download.done"), description: event.dir })',
    )
    expect(download.indexOf("await updateLfm2({ modelDir: event.dir })")).toBeLessThan(
      download.indexOf("if (!saved) return"),
    )
    expect(download.indexOf("await updateLfm2({ moshiModelDir: event.dir })")).toBeLessThan(
      download.indexOf("if (!saved) return"),
    )
    expect(download.indexOf("if (!saved) return")).toBeLessThan(
      download.indexOf('showToast({ title: language.t("settings.voice.download.done"), description: event.dir })'),
    )
  })

  test("native model directory settings accept Hugging Face cache repo roots", async () => {
    const rust = await root("packages/desktop/src-tauri/src/settings.rs")
    const resolver = between(rust, "fn hf_snapshot_dir", "pub fn expand_user_path")
    const lfm2 = between(rust, "pub fn lfm2_model_dir", "pub fn moshi_model_dir")
    const moshi = between(rust, "pub fn moshi_model_dir", "/// The active LFM2-Audio directory")

    expect(resolver).toContain('dir.join("snapshots")')
    expect(resolver).toContain('dir.join("refs").join("main")')
    expect(resolver).toContain("snapshots.join(rev.trim())")
    expect(lfm2).toContain(".map(hf_snapshot_dir)")
    expect(moshi).toContain(".map(hf_snapshot_dir)")
  })


  test("desktop LFM2 construction opens only the opaque native model", async () => {
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const lfm2 = between(runtime, "struct ResidentLfm2Key", "/// One active native voice service")
    const build = between(runtime, "fn build_engine(", "#[derive(Debug, Clone, PartialEq, Serialize)]")

    expect(lfm2).toContain("ResidentCache<ResidentLfm2Key, NativeVoiceModel>")
    expect(lfm2).toContain("NativeVoiceModel::open_with_config(dir, runtime)")
    expect(lfm2).not.toContain("LFM2AudioModel")
    expect(lfm2).not.toContain("LFM2AudioProcessor")
    expect(lfm2).not.toContain("from_pretrained")
    expect(build).toContain("let engine: NativeLfm2VoiceEngine = model.engine(")
    expect(build).toContain("capture_max_callback_frames,")
    expect(build).not.toContain("Lfm2VoiceEngine::new")
    expect(build).not.toContain("from_pretrained")
  })

  test("desktop LFM2 model downloads are owned by a bounded native runtime", async () => {
    const model = await root("packages/desktop/src-tauri/src/voice/model.rs")
    const lib = await root("packages/desktop/src-tauri/src/lib.rs")
    const module = await root("packages/desktop/src-tauri/src/voice/mod.rs")
    const threads = await root("packages/desktop/src-tauri/src/voice/threads.rs")
    const runtime = between(model, "pub struct ModelDownloadRuntime", "/// Download a model snapshot")
    const command = between(model, "pub async fn voice_model_download", "/// Open a native folder picker")
    const manager = between(threads, "pub(crate) struct ThreadManager", "impl Drop for ThreadManager")

    expect(lib).toContain("app.manage(voice::model::ModelDownloadRuntime::default())")
    expect(module).toContain("mod threads")
    expect(model).toContain("use super::threads::ThreadManager")
    expect(runtime).toContain("threads: ThreadManager")
    expect(runtime).toContain("fn spawn(&self")
    expect(runtime).toContain("self.threads.spawn_if_idle(")
    expect(runtime).toContain("voice model download already running")
    expect(runtime).toContain('"voice-model-download"')
    expect(manager).toContain("pub(crate) fn spawn_if_idle(")
    expect(manager).toContain("self.reap()?")
    expect(manager).toContain("if !handles.is_empty()")
    expect(manager).toContain("ThreadBuilder::new()")
    expect(manager).toContain(".name(name.into())")
    expect(command).toContain("runtime: State<'_, ModelDownloadRuntime>")
    expect(command).toContain(".path()")
    expect(command).toContain("app_cache_dir()")
    expect(command).toContain("snapshot_download_to")
    expect(command).toContain("token.as_deref()")
    expect(command).toContain("runtime.spawn(move ||")
    expect(model).not.toContain("Mutex<Vec<JoinHandle<()>>>")
    expect(model).not.toContain("thread::{Builder as ThreadBuilder, JoinHandle}")
    expect(model).not.toContain("std::thread::Builder::new()")
  })

  test("desktop LFM2 audio counters are exposed through native voice status", async () => {
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const settings = await root("packages/app/src/components/settings-voice.tsx")
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")
    const audio = await liquid("src/runtime/voice_runtime.rs")

    expect(audio).toContain("pub struct AudioStatsSnapshot")
    expect(audio).toContain("decoded_samples")
    expect(audio).toContain("played_samples")
    expect(runtime).toContain("pub audio_stats: Option<AudioStatsSnapshot>")
    expect(control).toContain('#[serde(rename = "audioStats")]')
    expect(control).toContain("p.audio_stats = active.audio_stats")
    expect(control).toContain("pub enum VoiceEngineMode")
    expect(control).toContain("pub engine: Option<VoiceEngineMode>")
    expect(control).toContain("local_engine_mode")
    expect(bridge).toContain("audioStats?:")
    expect(bridge).toContain("engine?: VoiceEngineMode")
    expect(settings).toContain("const audioStatsText = ()")
    expect(settings).toContain("const engineText = ()")
    expect(settings).toContain('language.t("settings.voice.audioStats"')
    expect(settings).toContain("settings.voice.engine.moshiRealtime")
  })

  test("desktop speaker probe reports native WebRTC speaker state", async () => {
    const bridge = await root("packages/app/src/lib/voice-settings.ts")
    const settings = await root("packages/app/src/components/settings-voice.tsx")
    const control = await root("packages/desktop/src-tauri/src/voice/control.rs")
    const runtime = await root("packages/desktop/src-tauri/src/voice/runtime.rs")

    expect(runtime).toContain("pub struct VoiceAudioProbeReport")
    expect(runtime).toContain("adm_playout_enabled")
    expect(runtime).toContain("playout_initialized")
    expect(runtime).toContain("playout_devices: self.audio.playout_devices().count()")
    expect(control).toContain("Result<super::runtime::VoiceAudioProbeReport, String>")
    expect(bridge).toContain("export interface VoiceAudioProbeReport")
    expect(bridge).toContain("playoutInitialized: boolean")
    expect(settings).toContain('language.t("settings.voice.toast.speakerReport"')
  })

  test("prompt input pauses native voice while text input or prompt execution is active", async () => {
    const text = await root("packages/app/src/components/prompt-input.tsx")

    expect(text).toContain("voiceMicTarget(voice.state(), prompt.dirty(), working())")
    expect(text).toContain("voice.beginTypedInput().catch(() => {})")
    expect(text).toContain('if (voice.state() === "connected") await voice.beginTypedInput().catch(() => {})')
    expect(text).toContain("if (voiceMic() === false) voice.setMicEnabled(true).catch(() => {})")
  })

  test("prompt input stop button interrupts voice before aborting prompt work", async () => {
    const text = await root("packages/app/src/components/prompt-input.tsx")
    const abort = between(text, "const abort = async () =>", "const addToHistory")

    expect(abort).toContain("const speaking = voice.turnActive()")
    expect(abort).toContain("await voice.interrupt().catch(() => {})")
    expect(abort.indexOf("await voice.interrupt().catch(() => {})")).toBeLessThan(abort.indexOf("sdk.client.session"))
  })

  test("desktop voice docs describe the implemented Tauri kernel command surface", async () => {
    const stack = await root("desktop-stack.md")
    const frontend = await root("packages/desktop/src-tauri/src/voice/FRONTEND_DESIGN.md")
    const architecture = await root("packages/desktop/src-tauri/src/voice/VOICE_ARCHITECTURE.md")

    expect(stack).toContain("The shared named-thread owner lives in `src/voice/threads.rs`")
    expect(stack).toContain("voice-model-download thread")
    expect(stack).toContain("voice-livekit-agent-events")
    expect(stack).toContain("voice-livekit-agent-mic")
    expect(stack).toContain("voice-livekit-audio-monitor")
    expect(stack).not.toContain("LKRemote")
    expect(frontend).toContain("voice_start(app, runtime, server, ctx, channel)")
    expect(frontend).toContain("voice_interrupt(runtime)")
    expect(frontend).toContain("voice_begin_typed_input(runtime)")
    expect(frontend).toContain("SessionCtx { sessionID, directory, model?, agent?, variant?, promptMode? }")
    expect(frontend).toContain("It does not carry `delegateTarget`")
    expect(frontend).not.toContain("voice_start_live")
    expect(frontend).not.toContain("voice_stop_live")
    expect(frontend).not.toContain("voice_abort_turn")
    expect(frontend).not.toContain("voice_generate_turn")
    expect(frontend).not.toContain("delegateTarget?")
    expect(architecture).toContain("Rust owns platform callback endpoints")
    expect(architecture).toContain("The safe kcoro Rust seam owns host continuations")
    expect(architecture).toContain("Tauri may:")
    expect(architecture).toContain("persist device/model/audio settings")
    expect(architecture).toContain("desktop commands/settings/events")
    expect(architecture).toContain("Device and model selection are runtime settings")
    expect(architecture).not.toContain("voice_settings_set persists, then RuntimeCommand::ApplySettings")
  })
})
