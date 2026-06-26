import { Component, createMemo, createResource, createSignal, Show, type JSX } from "solid-js"
import { Button } from "@thesolaceproject/emberharmony-ui/button"
import { Select } from "@thesolaceproject/emberharmony-ui/select"
import { Switch } from "@thesolaceproject/emberharmony-ui/switch"
import { TextField } from "@thesolaceproject/emberharmony-ui/text-field"
import { showToast } from "@thesolaceproject/emberharmony-ui/toast"
import { ScrollFade } from "@thesolaceproject/emberharmony-ui/scroll-fade"
import { useLanguage } from "@/context/language"
import { useGlobalSDK } from "@/context/global-sdk"
import {
  defaultVoiceSettings,
  getVoiceSettings,
  getVoiceStatus,
  isDesktop,
  setVoiceSettings,
  type DelegateSettings,
  type Lfm2Device,
  type Lfm2Settings,
  type VoiceProvider,
  type VoiceSettings,
} from "@/lib/voice-settings"

export const SettingsVoice: Component = () => {
  const language = useLanguage()
  const globalSDK = useGlobalSDK()
  const desktop = isDesktop()

  // LiveKit settings live in the sidecar config + credentials store (the sidecar
  // dispatches the LiveKit agent). The provider switch + the local LFM2 config
  // live in the Tauri settings store, read natively by the Rust voice loop.
  const [config, { refetch }] = createResource(() =>
    globalSDK.client.voice
      .config()
      .then((x) => x.data)
      .catch(() => undefined),
  )
  const [tauriVoice, { refetch: refetchTauri }] = createResource(getVoiceSettings)
  const [voiceStatus, { refetch: refetchStatus }] = createResource(getVoiceStatus)

  const [url, setUrl] = createSignal<string | undefined>(undefined)
  const [apiKey, setApiKey] = createSignal("")
  const [apiSecret, setApiSecret] = createSignal("")
  const [saving, setSaving] = createSignal(false)
  const [testing, setTesting] = createSignal(false)
  const [override, setOverride] = createSignal<VoiceProvider>()

  const effectiveUrl = () => url() ?? config()?.url ?? ""
  const lfm2 = (): Lfm2Settings => tauriVoice()?.lfm2 ?? defaultVoiceSettings.lfm2
  const livekitConfigured = () => Boolean(config()?.url || config()?.credentials?.livekit)

  // Effective provider: an explicit pick wins; otherwise the stored provider, and
  // as a migration nicety existing LiveKit users default to "livekit" so their
  // config stays visible even before they touch the new switch.
  const provider = (): VoiceProvider => {
    const picked = override()
    if (picked) return picked
    const stored = tauriVoice()?.provider
    if (stored && stored !== "off") return stored
    if (livekitConfigured()) return "livekit"
    return stored ?? "off"
  }

  async function changeProvider(next: VoiceProvider) {
    setOverride(next)
    const base = tauriVoice() ?? defaultVoiceSettings
    await setVoiceSettings({ ...base, provider: next }).catch((err) =>
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: err instanceof Error ? err.message : String(err),
      }),
    )
    refetchTauri()
    refetchStatus()
  }

  async function updateLfm2(patch: Partial<Lfm2Settings>) {
    const base = tauriVoice() ?? defaultVoiceSettings
    await setVoiceSettings({ ...base, lfm2: { ...base.lfm2, ...patch } }).catch((err) =>
      showToast({
        title: language.t("settings.voice.toast.saveFailed"),
        description: err instanceof Error ? err.message : String(err),
      }),
    )
    refetchTauri()
    refetchStatus()
  }

  const updateDelegate = (patch: Partial<DelegateSettings>) =>
    updateLfm2({ delegate: { ...lfm2().delegate, ...patch } })

  async function update(patch: Record<string, unknown>) {
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
      if (url() !== undefined && url() !== config()?.url) {
        await globalSDK.client.voice.configUpdate({ voiceConfig: { livekit: { url: url() } } })
      }
      if (key && secret) {
        await globalSDK.client.auth.set({
          providerID: "livekit",
          auth: { type: "api", key, secret },
        })
        setApiKey("")
        setApiSecret("")
        // the auth route only stores credentials; an empty voice config patch
        // nudges serve to (re)start the agent worker with them
        await globalSDK.client.voice.configUpdate({ voiceConfig: {} })
      }
      await refetch()
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

  type RegistryOption = { id: string; name: string; provider: string; defaultSuffix?: string }
  const modelValue = (option: RegistryOption) =>
    option.defaultSuffix ? `${option.id}:${option.defaultSuffix}` : option.id
  const sttOptions = createMemo(() => config()?.registry.stt ?? [])
  const ttsOptions = createMemo(() => config()?.registry.tts ?? [])
  const currentStt = createMemo(() => sttOptions().find((o) => config()?.stt.split(":")[0] === o.id))
  const currentTts = createMemo(() => ttsOptions().find((o) => config()?.tts.split(":")[0] === o.id))

  type ProviderOption = { id: VoiceProvider; label: string }
  const providerOptions = (): ProviderOption[] => [
    { id: "off", label: language.t("settings.voice.provider.off") },
    { id: "lfm2", label: language.t("settings.voice.provider.lfm2") },
    { id: "livekit", label: language.t("settings.voice.provider.livekit") },
  ]
  const currentProvider = () => providerOptions().find((o) => o.id === provider())

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
        {/* provider switch — the two-provider surface */}
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
                onSelect={(option) => option && changeProvider(option.id)}
                variant="secondary"
                size="small"
                triggerVariant="settings"
              />
            </SettingsRow>
          </div>
        </div>

        <Show when={provider() === "off"}>
          <div class="text-12-regular text-text-weak px-1">{language.t("settings.voice.off.hint")}</div>
        </Show>

        {/* ---- Local LFM2-Audio provider (native, Tauri store) ---- */}
        <Show when={provider() === "lfm2"}>
          <div class="flex flex-col gap-1">
            <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.lfm2")}</h3>
            <Show when={voiceStatus()?.provider === "lfm2" ? voiceStatus() : undefined}>
              {(s) => (
                <div class={`text-12-regular pb-2 px-1 ${s().ready ? "text-text-weak" : "text-text-strong"}`}>
                  {s().detail}
                </div>
              )}
            </Show>
            <div class="bg-surface-raised-base px-4 py-3 rounded-lg flex flex-col gap-3">
              <TextField
                label={language.t("settings.voice.row.modelDir.title")}
                description={language.t("settings.voice.row.modelDir.description")}
                placeholder="/path/to/lfm2-audio/model"
                value={lfm2().modelDir ?? ""}
                onFocusOut={(e: FocusEvent) => {
                  const value = (e.currentTarget as HTMLInputElement).value.trim()
                  updateLfm2({ modelDir: value || undefined })
                }}
              />
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
                  value={String(lfm2().vadThreshold)}
                  onFocusOut={(e: FocusEvent) => {
                    const n = numberFromInput(e)
                    if (n !== undefined && n > 0) updateLfm2({ vadThreshold: n })
                  }}
                />
              </SettingsRow>
              <SettingsRow
                title={language.t("settings.voice.row.maxTokens.title")}
                description={language.t("settings.voice.row.maxTokens.description")}
              >
                <TextField
                  hideLabel
                  label={language.t("settings.voice.row.maxTokens.title")}
                  value={String(lfm2().maxTokens)}
                  onFocusOut={(e: FocusEvent) => {
                    const n = numberFromInput(e)
                    if (n !== undefined && n >= 1) updateLfm2({ maxTokens: Math.floor(n) })
                  }}
                />
              </SettingsRow>
            </div>
          </div>

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
                    value={lfm2().delegate.target ?? ""}
                    onFocusOut={(e: FocusEvent) => {
                      const value = (e.currentTarget as HTMLInputElement).value.trim()
                      updateDelegate({ target: value || undefined })
                    }}
                  />
                </SettingsRow>
              </Show>
            </div>
          </div>

          <Show when={!desktop}>
            <div class="text-12-regular text-text-weak px-1">{language.t("settings.voice.lfm2.desktopOnly")}</div>
          </Show>
        </Show>

        {/* ---- LiveKit provider (sidecar config + credentials store) ---- */}
        <Show when={provider() === "livekit"}>
          <div class="flex flex-col gap-1">
            <div class="bg-surface-raised-base px-4 rounded-lg">
              <SettingsRow
                title={language.t("settings.voice.row.enabled.title")}
                description={language.t("settings.voice.row.enabled.description")}
              >
                {/* render only once config is loaded — a Switch mounted in the
                    loading state reads as "off" and a click then persists
                    disabled:true even though the user meant to enable */}
                <Show when={config()}>
                  {(cfg) => (
                    <Switch hideLabel checked={!cfg().disabled} onChange={(checked) => update({ disabled: !checked })}>
                      {language.t("settings.voice.row.enabled.title")}
                    </Switch>
                  )}
                </Show>
              </SettingsRow>
            </div>
          </div>

          <div class="flex flex-col gap-1">
            <h3 class="text-14-medium text-text-strong pb-2">{language.t("settings.voice.section.connection")}</h3>
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
                placeholder={config()?.credentials.livekit ? "••••••••" : "API…"}
                value={apiKey()}
                onChange={setApiKey}
              />
              <TextField
                label={language.t("settings.voice.row.apiSecret.title")}
                description={language.t("settings.voice.row.credentials.description")}
                type="password"
                placeholder={config()?.credentials.livekit ? "••••••••" : ""}
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
                      value={cfg().intent}
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

          <Show when={config() && !config()!.credentials.livekit && !config()!.available}>
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
