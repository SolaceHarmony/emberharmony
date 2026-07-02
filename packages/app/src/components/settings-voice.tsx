import { Component, createMemo, createResource, createSignal, onCleanup, Show, type JSX } from "solid-js"
import { Button } from "@thesolaceproject/emberharmony-ui/button"
import { Select } from "@thesolaceproject/emberharmony-ui/select"
import { Switch } from "@thesolaceproject/emberharmony-ui/switch"
import { TextField } from "@thesolaceproject/emberharmony-ui/text-field"
import { showToast } from "@thesolaceproject/emberharmony-ui/toast"
import { ScrollFade } from "@thesolaceproject/emberharmony-ui/scroll-fade"
import { Link } from "./link"
import { useLanguage } from "@/context/language"
import { useGlobalSDK } from "@/context/global-sdk"
import {
  DEFAULT_LFM2_MODEL,
  DEFAULT_MOSHI_MODEL,
  defaultVoiceSettings,
  downloadVoiceModel,
  getHfTokenStatus,
  getLiveKitCredentialsStatus,
  getVoiceSettingsState,
  getVoiceStatus,
  isDesktop,
  pickModelDir,
  playVoiceAudioProbe,
  setHfToken,
  setLiveKitCredentials,
  setVoiceSettings,
  type DelegateSettings,
  type Lfm2Device,
  type Lfm2Settings,
  type VoiceEngineMode,
  type VoiceProvider,
  type VoiceSettings,
} from "@/lib/voice-settings"

export const SettingsVoice: Component = () => {
  const language = useLanguage()
  const globalSDK = useGlobalSDK()
  const desktop = isDesktop()

  // The Tauri settings/keychain are the desktop voice source of truth. The
  // legacy server config remains only for the non-desktop web voice provider.
  const [config, { refetch }] = createResource(() =>
    desktop
      ? Promise.resolve(undefined)
      : globalSDK.client.voice
          .config()
          .then((x) => x.data)
          .catch(() => undefined),
  )
  const [tauriVoice, { refetch: refetchTauri, mutate: setTauriVoice }] = createResource(getVoiceSettingsState)
  const [voiceStatus, { refetch: refetchStatus }] = createResource(getVoiceStatus)
  const [nativeLivekit, { refetch: refetchNativeLivekit }] = createResource(getLiveKitCredentialsStatus)
  const statusPoll = desktop ? setInterval(() => refetchStatus(), 1000) : undefined
  onCleanup(() => {
    if (statusPoll === undefined) return
    clearInterval(statusPoll)
  })

  const [url, setUrl] = createSignal<string | undefined>(undefined)
  const [apiKey, setApiKey] = createSignal("")
  const [apiSecret, setApiSecret] = createSignal("")
  const [saving, setSaving] = createSignal(false)
  const [testing, setTesting] = createSignal(false)
  const [probing, setProbing] = createSignal(false)
  const [override, setOverride] = createSignal<VoiceProvider>()

  // LFM2 model management — download / HF token / native dir picker.
  const [hfTokenInput, setHfTokenInput] = createSignal("")
  const [savingToken, setSavingToken] = createSignal(false)
  const [modelDirEdit, setModelDirEdit] = createSignal<string | undefined>(undefined)
  const [moshiModelDirEdit, setMoshiModelDirEdit] = createSignal<string | undefined>(undefined)
  const [downloading, setDownloading] = createSignal(false)
  const [downloadMsg, setDownloadMsg] = createSignal<string | undefined>(undefined)
  const [hfTokenStored, { refetch: refetchToken }] = createResource(getHfTokenStatus)

  const effectiveUrl = () => url() ?? voice().livekit.url ?? (!desktop ? config()?.url : undefined) ?? ""
  const voice = (): VoiceSettings => tauriVoice()?.settings ?? defaultVoiceSettings
  const lfm2 = (): Lfm2Settings => voice().lfm2
  const localEngine = (): VoiceEngineMode => lfm2().engine ?? "moshiRealtime"
  const modelDirValue = () => modelDirEdit() ?? lfm2().modelDir ?? ""
  const moshiModelDirValue = () => moshiModelDirEdit() ?? lfm2().moshiModelDir ?? ""
  const selectedModelDirValue = () => (localEngine() === "moshiRealtime" ? moshiModelDirValue() : modelDirValue())
  const hfModel = () =>
    (localEngine() === "moshiRealtime"
      ? (lfm2().moshiModel ?? DEFAULT_MOSHI_MODEL)
      : (lfm2().model ?? DEFAULT_LFM2_MODEL)
    ).trim()
  const hfRevision = () => (localEngine() === "moshiRealtime" ? lfm2().moshiRevision : lfm2().revision)
  const hfUrl = () => {
    const m = hfModel()
    return m.startsWith("http") ? m : `https://huggingface.co/${m}`
  }
  const livekitCredentialsStored = () => nativeLivekit()?.stored || (!desktop && config()?.credentials?.livekit)
  const livekitConfigured = () => Boolean(effectiveUrl() || livekitCredentialsStored())

  // Effective provider: an explicit pick wins; otherwise the stored provider. In
  // web mode only, old LiveKit config can still surface the legacy provider.
  const provider = (): VoiceProvider => {
    const picked = override()
    if (picked) return picked
    const stored = tauriVoice()
    if (stored?.stored) return stored.settings.provider
    if (!desktop && livekitConfigured()) return "livekit"
    return stored?.settings.provider ?? "off"
  }

  const enabled = () => provider() !== "off"
  const rememberedProvider = (): Exclude<VoiceProvider, "off"> | undefined => {
    const last = voice().lastProvider
    if (last === "lfm2" || last === "livekit") return last
    return undefined
  }
  const defaultProvider = (): Exclude<VoiceProvider, "off"> => {
    const stored = voice().provider
    if (stored === "lfm2" || stored === "livekit") return stored
    const remembered = rememberedProvider()
    if (remembered) return remembered
    if (desktop) return "lfm2"
    return "livekit"
  }
  const activeProvider = (): Exclude<VoiceProvider, "off"> => {
    const current = provider()
    if (current === "lfm2" || current === "livekit") return current
    return defaultProvider()
  }

  async function changeProvider(next: VoiceProvider) {
    const previous = provider()
    setOverride(next)
    const base = voice()
    const lastProvider = next === "off" ? (activeProvider() ?? base.lastProvider) : next
    const settings = { ...base, provider: next, lastProvider }
    const ok = await setVoiceSettings(settings)
      .then(() => {
        setTauriVoice({ settings, stored: true })
        setOverride(undefined)
        return true
      })
      .catch((err) => {
        setOverride(previous)
        showToast({
          title: language.t("settings.voice.toast.saveFailed"),
          description: err instanceof Error ? err.message : String(err),
        })
        return false
      })
    refetchTauri()
    refetchStatus()
    return ok
  }

  async function toggleVoice(checked: boolean) {
    const previous = provider()
    const next = checked ? activeProvider() : "off"
    if (!(await changeProvider(next))) return
    if (!desktop && previous === "livekit" && next !== "livekit") await update({ disabled: true })
    if (!desktop && next === "livekit") await update({ disabled: false })
  }

  async function selectProvider(next: Exclude<VoiceProvider, "off">) {
    const previous = provider()
    if (!enabled()) {
      const base = voice()
      const settings = { ...base, lastProvider: next }
      await setVoiceSettings(settings)
        .then(() => setTauriVoice({ settings, stored: true }))
        .catch((err) =>
          showToast({
            title: language.t("settings.voice.toast.saveFailed"),
            description: err instanceof Error ? err.message : String(err),
          }),
        )
      refetchTauri()
      refetchStatus()
      return
    }
    if (!(await changeProvider(next))) return
    if (!desktop && previous === "livekit" && next !== "livekit") await update({ disabled: true })
    if (!desktop && next === "livekit") await update({ disabled: false })
  }

  async function updateLfm2(patch: Partial<Lfm2Settings>) {
    const base = voice()
    const settings = { ...base, lfm2: { ...base.lfm2, ...patch } }
    const ok = await setVoiceSettings(settings)
      .then(() => {
        setTauriVoice({ settings, stored: true })
        return true
      })
      .catch((err) => {
        showToast({
          title: language.t("settings.voice.toast.saveFailed"),
          description: err instanceof Error ? err.message : String(err),
        })
        return false
      })
    refetchTauri()
    refetchStatus()
    return ok
  }

  async function updateLiveKit(patch: Partial<VoiceSettings["livekit"]>) {
    const base = voice()
    const settings = { ...base, livekit: { ...base.livekit, ...patch } }
    await setVoiceSettings(settings).then(() => setTauriVoice({ settings, stored: true }))
    await refetchTauri()
    await refetchStatus()
  }

  async function saveToken() {
    setSavingToken(true)
    try {
      await setHfToken(hfTokenInput().trim())
      setHfTokenInput("")
      await refetchToken()
    } catch (err) {
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setSavingToken(false)
    }
  }

  async function browseModelDir() {
    try {
      const dir = await pickModelDir()
      if (!dir) return
      if (localEngine() === "moshiRealtime") {
        setMoshiModelDirEdit(dir)
        await updateLfm2({ moshiModelDir: dir })
        return
      }
      setModelDirEdit(dir)
      await updateLfm2({ modelDir: dir })
    } catch (err) {
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  // Explicit, fail-hard download: progress streams over a Channel; the model only becomes
  // active on a clean `done`. Nothing here downloads silently and nothing auto-starts.
  async function downloadModel() {
    // The shown model (pre-filled default unless the user overrode it) — never empty, so
    // one click downloads the recommended model without anyone hunting for a repo id.
    const source = hfModel()
    const engine = localEngine()
    const revision = hfRevision()
    setDownloading(true)
    setDownloadMsg(`${language.t("settings.voice.download.downloading")}…`)
    try {
      await downloadVoiceModel({ source, revision }, (event) => {
        switch (event.type) {
          case "started":
            setDownloadMsg(`${language.t("settings.voice.download.downloading")} 0/${event.total}`)
            break
          case "file":
            setDownloadMsg(
              `${language.t("settings.voice.download.downloading")} ${event.index}/${event.total}: ${event.name}`,
            )
            break
          case "done":
            void (async () => {
              const saved =
                engine === "moshiRealtime"
                  ? await updateLfm2({ moshiModelDir: event.dir })
                  : await updateLfm2({ modelDir: event.dir })
              if (engine === "moshiRealtime") setMoshiModelDirEdit(event.dir)
              else setModelDirEdit(event.dir)
              setDownloading(false)
              setDownloadMsg(undefined)
              if (!saved) return
              showToast({ title: language.t("settings.voice.download.done"), description: event.dir })
            })()
            break
          case "error":
            setDownloading(false)
            setDownloadMsg(undefined)
            showToast({ title: language.t("settings.voice.download.failed"), description: event.message })
            break
        }
      })
    } catch (err) {
      setDownloading(false)
      setDownloadMsg(undefined)
      showToast({
        title: language.t("settings.voice.download.failed"),
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const updateDelegate = (patch: Partial<DelegateSettings>) =>
    updateLfm2({ delegate: { ...lfm2().delegate, ...patch } })

  async function update(patch: Record<string, unknown>) {
    if (desktop) {
      await updateLiveKit(patch as Partial<VoiceSettings["livekit"]>)
      return
    }
    await globalSDK.client.voice
      .configUpdate({ voiceConfig: patch })
      .then(() => refetch())
      .catch((err) => {
        showToast({
          title: language.t("settings.voice.toast.saveFailed"),
          description: err instanceof Error ? err.message : String(err),
        })
      })
  }

  async function saveConnection() {
    const key = apiKey().trim()
    const secret = apiSecret().trim()
    if ((key && !secret) || (!key && secret)) {
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: language.t("settings.voice.toast.credentialsIncomplete"),
      })
      return
    }
    setSaving(true)
    try {
      const nextUrl = effectiveUrl().trim()
      if (url() !== undefined || nextUrl !== (voice().livekit.url ?? "").trim()) {
        await updateLiveKit({ url: nextUrl || undefined })
      }
      if (key && secret) {
        await setLiveKitCredentials(key, secret)
        setApiKey("")
        setApiSecret("")
        if (!desktop) {
          await globalSDK.client.auth.set({
            providerID: "livekit",
            auth: { type: "api", key, secret },
          })
          await globalSDK.client.voice.configUpdate({ voiceConfig: {} })
        }
      }
      if (!desktop && url() !== undefined && url() !== config()?.url) {
        await globalSDK.client.voice.configUpdate({ voiceConfig: { livekit: { url: nextUrl } } })
      }
      await refetchNativeLivekit()
      await refetchStatus()
      if (!desktop) await refetch()
      showToast({ title: language.t("settings.voice.toast.saved") })
    } catch (err) {
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setSaving(false)
    }
  }

  async function testConnection() {
    setTesting(true)
    try {
      if (desktop) {
        const status = await getVoiceStatus()
        if (status.ready) {
          showToast({ title: language.t("settings.voice.toast.testOk"), description: status.detail })
          return
        }
        showToast({
          title: language.t("settings.voice.toast.testFailed"),
          description: status.detail || language.t("settings.voice.toast.testUnavailable"),
        })
        return
      }
      const status = await globalSDK.client.voice.status().then((x) => x.data)
      if (status?.available) {
        showToast({ title: language.t("settings.voice.toast.testOk"), description: status.url ?? undefined })
      } else {
        showToast({
          title: language.t("settings.voice.toast.testFailed"),
          description: language.t("settings.voice.toast.testUnavailable"),
        })
      }
    } catch (err) {
      showToast({
        title: language.t("settings.voice.toast.testFailed"),
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setTesting(false)
    }
  }

  async function testSpeaker() {
    setProbing(true)
    await playVoiceAudioProbe()
      .then((report) =>
        showToast({
          title: language.t("settings.voice.toast.speakerOk"),
          description: report
            ? language.t("settings.voice.toast.speakerReport", {
                device: report.playoutDevice ?? "default output",
                rate: report.sampleRate,
                frames: report.webrtcFrames,
                adm: report.admPlayoutEnabled && report.playoutInitialized ? "ready" : "not ready",
              })
            : undefined,
        }),
      )
      .catch((err) =>
        showToast({
          title: language.t("settings.voice.toast.speakerFailed"),
          description: err instanceof Error ? err.message : String(err),
        }),
      )
      .finally(() => setProbing(false))
  }

  type RegistryOption = { id: string; name: string; provider: string; defaultSuffix?: string }
  const modelValue = (option: RegistryOption) =>
    option.defaultSuffix ? `${option.id}:${option.defaultSuffix}` : option.id
  const sttOptions = createMemo(() => config()?.registry.stt ?? [])
  const ttsOptions = createMemo(() => config()?.registry.tts ?? [])
  const currentStt = createMemo(() => sttOptions().find((o) => config()?.stt.split(":")[0] === o.id))
  const currentTts = createMemo(() => ttsOptions().find((o) => config()?.tts.split(":")[0] === o.id))

  type ProviderOption = { id: Exclude<VoiceProvider, "off">; label: string }
  const providerOptions = (): ProviderOption[] => [
    { id: "lfm2", label: language.t("settings.voice.provider.lfm2") },
    { id: "livekit", label: language.t("settings.voice.provider.livekit") },
  ]
  const currentProvider = () => providerOptions().find((o) => o.id === activeProvider())
  const showNativeModel = () =>
    enabled() && (activeProvider() === "lfm2" || (desktop && activeProvider() === "livekit"))
  const showLegacyLiveKitModels = () => !desktop && enabled() && activeProvider() === "livekit"
  const audioStatsText = () => {
    const stats = voiceStatus()?.audioStats
    if (!stats) return undefined
    return language.t("settings.voice.audioStats", {
      decoded: stats.decodedSamples.toLocaleString(),
      queued: stats.queuedSamples.toLocaleString(),
      played: stats.playedSamples.toLocaleString(),
      dropped: stats.droppedSamples.toLocaleString(),
      underruns: stats.underrunFrames.toLocaleString(),
    })
  }
  const engineText = () => {
    const engine = voiceStatus()?.engine
    if (!engine) return undefined
    return engine === "moshiRealtime"
      ? language.t("settings.voice.engine.moshiRealtime")
      : language.t("settings.voice.engine.lfm2Interleaved")
  }

  type EngineOption = { id: VoiceEngineMode; label: string }
  const engineOptions = (): EngineOption[] => [
    { id: "lfm2Interleaved", label: language.t("settings.voice.engineOption.lfm2Interleaved") },
    { id: "moshiRealtime", label: language.t("settings.voice.engineOption.moshiRealtime") },
  ]
  const currentEngine = () => engineOptions().find((o) => o.id === localEngine())

  type DeviceOption = { id: Lfm2Device; label: string }
  const deviceOptions: DeviceOption[] = [
    { id: "cpu", label: "CPU" },
    { id: "metal", label: "Metal (Apple GPU)" },
  ]
  const currentDevice = () => deviceOptions.find((o) => o.id === lfm2().device)

  const numberFromInput = (e: FocusEvent): number | undefined => {
    const raw = (e.currentTarget as HTMLInputElement).value.trim()
    if (raw === "") return undefined
    const n = Number(raw)
    return Number.isFinite(n) ? n : undefined
  }

  return (
    <ScrollFade class="h-full overflow-y-auto px-8">
      <div class="sticky top-0 z-10 bg-[linear-gradient(to_bottom,var(--surface-raised-stronger-non-alpha)_calc(100%_-_24px),transparent)]">
        <div class="flex flex-col gap-1 pt-6 pb-8">
          <h2 class="text-16-medium text-text-strong">{language.t("settings.voice.title")}</h2>
        </div>
      </div>

      <div class="flex flex-col gap-8 w-full pb-8">
        <div class="flex flex-col gap-1">
          <div class="bg-surface-raised-base px-4 rounded-lg">
            <SettingsRow
              title={language.t("settings.voice.row.enabled.title")}
              description={language.t("settings.voice.row.enabled.description")}
            >
              <Switch hideLabel checked={enabled()} onChange={toggleVoice}>
                {language.t("settings.voice.row.enabled.title")}
              </Switch>
            </SettingsRow>
          </div>
        </div>

        {/* provider switch — provider choice is separate from enablement */}
        <div class="flex flex-col gap-1">
          <div class="bg-surface-raised-base px-4 rounded-lg">
            <SettingsRow
              title={language.t("settings.voice.row.provider.title")}
              description={language.t("settings.voice.row.provider.description")}
            >
              <Select
                options={providerOptions()}
                current={currentProvider()}
                value={(o) => o.id}
                label={(o) => o.label}
                onSelect={(option) => option && selectProvider(option.id)}
                variant="secondary"
                size="small"
                triggerVariant="settings"
              />
            </SettingsRow>
          </div>
          <Show when={desktop}>
            <div class="bg-surface-raised-base px-4 rounded-lg">
              <SettingsRow
                title={language.t("settings.voice.row.speaker.title")}
                description={language.t("settings.voice.row.speaker.description")}
              >
                <Button variant="secondary" size="small" disabled={probing()} onClick={testSpeaker}>
                  {probing()
                    ? language.t("settings.voice.action.testingSpeaker")
                    : language.t("settings.voice.action.testSpeaker")}
                </Button>
              </SettingsRow>
            </div>
          </Show>
          <Show when={audioStatsText()}>
            {(text) => <div class="text-12-regular text-text-weak px-1">{text()}</div>}
          </Show>
          <Show when={engineText()}>
            {(text) => <div class="text-12-regular text-text-weak px-1">{text()}</div>}
          </Show>
        </div>

        <Show when={!enabled()}>
          <div class="text-12-regular text-text-weak px-1">{language.t("settings.voice.off.hint")}</div>
        </Show>

        {/* ---- Local voice model/delegation (native, Tauri store) ---- */}
        <Show when={showNativeModel()}>
          <div class="flex flex-col gap-1">
            <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.lfm2")}</h3>
            <Show when={voiceStatus()?.provider === "lfm2" ? voiceStatus() : undefined}>
              {(s) => (
                <div class={`text-12-regular pb-2 px-1 ${s().ready ? "text-text-weak" : "text-text-strong"}`}>
                  {s().detail}
                </div>
              )}
            </Show>
            <div class="bg-surface-raised-base px-4 rounded-lg">
              <SettingsRow
                title={language.t("settings.voice.row.engine.title")}
                description={language.t("settings.voice.row.engine.description")}
              >
                <Select
                  options={engineOptions()}
                  current={currentEngine()}
                  value={(o) => o.id}
                  label={(o) => o.label}
                  onSelect={(option) => option && updateLfm2({ engine: option.id })}
                  variant="secondary"
                  size="small"
                  triggerVariant="settings"
                />
              </SettingsRow>
            </div>
            <div class="bg-surface-raised-base px-4 py-3 rounded-lg flex flex-col gap-3">
              <Show
                when={localEngine() === "moshiRealtime"}
                fallback={
                  <>
                    <TextField
                      label={language.t("settings.voice.row.model.title")}
                      description={language.t("settings.voice.row.model.description")}
                      placeholder={DEFAULT_LFM2_MODEL}
                      defaultValue={lfm2().model ?? DEFAULT_LFM2_MODEL}
                      onFocusOut={(e: FocusEvent) => {
                        const value = (e.currentTarget as HTMLInputElement).value.trim()
                        updateLfm2({ model: value || undefined })
                      }}
                    />
                    <Link href={hfUrl()}>{language.t("settings.voice.row.model.viewOnHf")}</Link>
                    <TextField
                      label={language.t("settings.voice.row.revision.title")}
                      description={language.t("settings.voice.row.revision.description")}
                      placeholder="main"
                      defaultValue={lfm2().revision ?? ""}
                      onFocusOut={(e: FocusEvent) => {
                        const value = (e.currentTarget as HTMLInputElement).value.trim()
                        updateLfm2({ revision: value || undefined })
                      }}
                    />
                  </>
                }
              >
                <TextField
                  label={language.t("settings.voice.row.model.title")}
                  description={language.t("settings.voice.row.moshiModel.description")}
                  placeholder={DEFAULT_MOSHI_MODEL}
                  defaultValue={lfm2().moshiModel ?? DEFAULT_MOSHI_MODEL}
                  onFocusOut={(e: FocusEvent) => {
                    const value = (e.currentTarget as HTMLInputElement).value.trim()
                    updateLfm2({ moshiModel: value || undefined })
                  }}
                />
                <Link href={hfUrl()}>{language.t("settings.voice.row.model.viewOnHf")}</Link>
                <TextField
                  label={language.t("settings.voice.row.revision.title")}
                  description={language.t("settings.voice.row.revision.description")}
                  placeholder="main"
                  defaultValue={lfm2().moshiRevision ?? ""}
                  onFocusOut={(e: FocusEvent) => {
                    const value = (e.currentTarget as HTMLInputElement).value.trim()
                    updateLfm2({ moshiRevision: value || undefined })
                  }}
                />
              </Show>
              <TextField
                label={language.t("settings.voice.row.hfToken.title")}
                description={language.t("settings.voice.row.hfToken.description")}
                type="password"
                placeholder={hfTokenStored() ? "••••••••" : "hf_…"}
                value={hfTokenInput()}
                onChange={setHfTokenInput}
              />
              <div class="flex items-center gap-2">
                <Button variant="secondary" size="small" disabled={savingToken()} onClick={saveToken}>
                  {language.t("settings.voice.action.saveToken")}
                </Button>
                <Show when={hfTokenStored()}>
                  <span class="text-12-regular text-text-weak">{language.t("settings.voice.row.hfToken.stored")}</span>
                </Show>
              </div>
              <div class="flex items-end gap-2">
                <div class="flex-1 min-w-0">
                  <TextField
                    label={language.t("settings.voice.row.modelDir.title")}
                    description={
                      localEngine() === "moshiRealtime"
                        ? language.t("settings.voice.row.moshiModelDir.description")
                        : language.t("settings.voice.row.modelDir.description")
                    }
                    placeholder="/path/to/huggingface/snapshot"
                    value={selectedModelDirValue()}
                    onChange={(value) =>
                      localEngine() === "moshiRealtime" ? setMoshiModelDirEdit(value) : setModelDirEdit(value)
                    }
                    onFocusOut={() => {
                      const dir = selectedModelDirValue().trim() || undefined
                      if (localEngine() === "moshiRealtime") updateLfm2({ moshiModelDir: dir })
                      else updateLfm2({ modelDir: dir })
                    }}
                  />
                </div>
                <Show when={desktop}>
                  <Button variant="secondary" size="small" onClick={browseModelDir}>
                    {language.t("settings.voice.row.modelDir.browse")}
                  </Button>
                </Show>
              </div>
              <Show when={desktop}>
                <div class="flex items-center gap-2 pt-1">
                  <Button variant="primary" size="small" disabled={downloading()} onClick={downloadModel}>
                    {language.t("settings.voice.action.download")}
                  </Button>
                  <Show when={downloadMsg()}>
                    {(msg) => <span class="text-12-regular text-text-weak truncate">{msg()}</span>}
                  </Show>
                </div>
              </Show>
            </div>
            <div class="bg-surface-raised-base px-4 rounded-lg mt-2">
              <SettingsRow
                title={language.t("settings.voice.row.device.title")}
                description={language.t("settings.voice.row.device.description")}
              >
                <Select
                  options={deviceOptions}
                  current={currentDevice()}
                  value={(o) => o.id}
                  label={(o) => o.label}
                  onSelect={(option) => option && updateLfm2({ device: option.id })}
                  variant="secondary"
                  size="small"
                  triggerVariant="settings"
                />
              </SettingsRow>
              <SettingsRow
                title={language.t("settings.voice.row.vadThreshold.title")}
                description={language.t("settings.voice.row.vadThreshold.description")}
              >
                <TextField
                  hideLabel
                  label={language.t("settings.voice.row.vadThreshold.title")}
                  defaultValue={String(lfm2().vadThreshold)}
                  onFocusOut={(e: FocusEvent) => {
                    const n = numberFromInput(e)
                    if (n !== undefined && n > 0) updateLfm2({ vadThreshold: n })
                  }}
                />
              </SettingsRow>
              <Show when={localEngine() === "lfm2Interleaved"}>
                <SettingsRow
                  title={language.t("settings.voice.row.maxTokens.title")}
                  description={language.t("settings.voice.row.maxTokens.description")}
                >
                  <TextField
                    hideLabel
                    label={language.t("settings.voice.row.maxTokens.title")}
                    defaultValue={String(lfm2().maxTokens)}
                    onFocusOut={(e: FocusEvent) => {
                      const n = numberFromInput(e)
                      if (n !== undefined && n >= 1) updateLfm2({ maxTokens: Math.floor(n) })
                    }}
                  />
                </SettingsRow>
              </Show>
            </div>
          </div>

          <Show when={localEngine() === "lfm2Interleaved"}>
            <div class="flex flex-col gap-1">
            <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.delegate")}</h3>
            <div class="bg-surface-raised-base px-4 rounded-lg">
              <SettingsRow
                title={language.t("settings.voice.row.delegate.title")}
                description={language.t("settings.voice.row.delegate.description")}
              >
                <Switch
                  hideLabel
                  checked={lfm2().delegate.enabled}
                  onChange={(checked) => updateDelegate({ enabled: checked })}
                >
                  {language.t("settings.voice.row.delegate.title")}
                </Switch>
              </SettingsRow>
              <Show when={lfm2().delegate.enabled}>
                <SettingsRow
                  title={language.t("settings.voice.row.delegateTarget.title")}
                  description={language.t("settings.voice.row.delegateTarget.description")}
                >
                  <TextField
                    hideLabel
                    label={language.t("settings.voice.row.delegateTarget.title")}
                    defaultValue={lfm2().delegate.target ?? ""}
                    onFocusOut={(e: FocusEvent) => {
                      const value = (e.currentTarget as HTMLInputElement).value.trim()
                      updateDelegate({ target: value || undefined })
                    }}
                  />
                </SettingsRow>
              </Show>
            </div>
            </div>
          </Show>

          <Show when={!desktop}>
            <div class="text-12-regular text-text-weak px-1">{language.t("settings.voice.lfm2.desktopOnly")}</div>
          </Show>
        </Show>

        {/* ---- LiveKit provider (native Tauri config + keychain credentials) ---- */}
        <Show when={enabled() && activeProvider() === "livekit"}>
          <div class="flex flex-col gap-1">
            <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.connection")}</h3>
            <Show when={voiceStatus()?.provider === "livekit" ? voiceStatus() : undefined}>
              {(s) => (
                <div class={`text-12-regular pb-2 px-1 ${s().ready ? "text-text-weak" : "text-text-strong"}`}>
                  {s().detail}
                </div>
              )}
            </Show>
            <div class="bg-surface-raised-base px-4 py-3 rounded-lg flex flex-col gap-3">
              <TextField
                label={language.t("settings.voice.row.url.title")}
                description={language.t("settings.voice.row.url.description")}
                placeholder="wss://<project>.livekit.cloud"
                value={effectiveUrl()}
                onChange={setUrl}
              />
              <TextField
                label={language.t("settings.voice.row.apiKey.title")}
                type="password"
                placeholder={livekitCredentialsStored() ? "••••••••" : "API…"}
                value={apiKey()}
                onChange={setApiKey}
              />
              <TextField
                label={language.t("settings.voice.row.apiSecret.title")}
                description={language.t("settings.voice.row.credentials.description")}
                type="password"
                placeholder={livekitCredentialsStored() ? "••••••••" : ""}
                value={apiSecret()}
                onChange={setApiSecret}
              />
              <div class="flex items-center gap-2 pt-1">
                <Button variant="primary" size="small" disabled={saving()} onClick={saveConnection}>
                  {language.t("settings.voice.action.save")}
                </Button>
                <Button variant="secondary" size="small" disabled={testing()} onClick={testConnection}>
                  {language.t("settings.voice.action.test")}
                </Button>
              </div>
            </div>
          </div>

          <Show when={showLegacyLiveKitModels()}>
            <div class="flex flex-col gap-1">
              <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.models")}</h3>
              <div class="bg-surface-raised-base px-4 rounded-lg">
                <SettingsRow
                  title={language.t("settings.voice.row.stt.title")}
                  description={language.t("settings.voice.row.stt.description")}
                >
                  <Select
                    options={sttOptions()}
                    current={currentStt()}
                    value={(o) => o.id}
                    label={(o) => `${o.provider} ${o.name}`}
                    onSelect={(option) => option && update({ stt: modelValue(option) })}
                    variant="secondary"
                    size="small"
                    triggerVariant="settings"
                  />
                </SettingsRow>
                <SettingsRow
                  title={language.t("settings.voice.row.tts.title")}
                  description={language.t("settings.voice.row.tts.description")}
                >
                  <Select
                    options={ttsOptions()}
                    current={currentTts()}
                    value={(o) => o.id}
                    label={(o) => `${o.provider} ${o.name}`}
                    onSelect={(option) => option && update({ tts: modelValue(option) })}
                    variant="secondary"
                    size="small"
                    triggerVariant="settings"
                  />
                </SettingsRow>
                <SettingsRow
                  title={language.t("settings.voice.row.intent.title")}
                  description={language.t("settings.voice.row.intent.description")}
                >
                  <Show when={config()}>
                    {(cfg) => (
                      <TextField
                        hideLabel
                        label={language.t("settings.voice.row.intent.title")}
                        defaultValue={cfg().intent}
                        onFocusOut={(e: FocusEvent) => {
                          const value = (e.currentTarget as HTMLInputElement).value.trim()
                          if (value && value !== cfg().intent) update({ intent: value })
                        }}
                      />
                    )}
                  </Show>
                </SettingsRow>
              </div>
            </div>
          </Show>

          <Show when={config() && !livekitCredentialsStored() && !config()!.available}>
            <div class="text-12-regular text-text-weak px-1">{language.t("settings.voice.hint.credentials")}</div>
          </Show>
        </Show>
      </div>
    </ScrollFade>
  )
}

interface SettingsRowProps {
  title: string
  description: string | JSX.Element
  children: JSX.Element
}

const SettingsRow: Component<SettingsRowProps> = (props) => {
  return (
    <div class="flex flex-wrap items-center justify-between gap-4 py-3 border-b border-border-weak-base last:border-none">
      <div class="flex flex-col gap-0.5 min-w-0">
        <span class="text-14-medium text-text-strong">{props.title}</span>
        <span class="text-12-regular text-text-weak">{props.description}</span>
      </div>
      <div class="flex-shrink-0">{props.children}</div>
    </div>
  )
}
