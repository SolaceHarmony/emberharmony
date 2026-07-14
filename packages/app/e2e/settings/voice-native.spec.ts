import { type Page } from "@playwright/test"
import { test, expect } from "../fixtures"
import { modKey } from "../utils"

type Provider = "lfm2" | "livekit"
type Call = { cmd: string; args?: unknown }
type ModeSampling = {
  textTemperature: number
  textTopK: number
  audioTemperature: number
  audioTopK: number
  maxTokens: number
}
type Settings = {
  provider: "off" | Provider
  lastProvider?: Provider
  livekit: { url?: string }
  lfm2: {
    engine: "lfm2Interleaved"
    model: string
    modelDir?: string
    device: "metal"
    vadThreshold: number
    asr: ModeSampling
    tts: ModeSampling
    interleaved: ModeSampling
    delegate: { enabled: boolean }
  }
}

async function install(page: Page, lastProvider: Provider) {
  await page.addInitScript(
    (input: { lastProvider: Provider }) => {
      const calls: Call[] = []
      const events: unknown[] = []
      const livekit = { credentialsStored: false }
      const settings: Settings = {
        provider: "off",
        lastProvider: input.lastProvider,
        livekit: { url: "wss://livekit.invalid" },
        lfm2: {
          engine: "lfm2Interleaved",
          model: "LiquidAI/LFM2.5-Audio-1.5B",
          modelDir: "/tmp/lfm2-audio-e2e",
          device: "metal",
          vadThreshold: 0.012,
          asr: { textTemperature: 0, textTopK: 0, audioTemperature: 0, audioTopK: 0, maxTokens: 100 },
          tts: { textTemperature: 0.7, textTopK: 0, audioTemperature: 0.8, audioTopK: 64, maxTokens: 1024 },
          interleaved: { textTemperature: 1.0, textTopK: 0, audioTemperature: 1.0, audioTopK: 4, maxTokens: 8192 },
          delegate: { enabled: false },
        },
      }
      const status = () => ({
        provider: settings.provider,
        enabled: settings.provider !== "off",
        surface: settings.provider === "off" ? "off" : "native",
        running: false,
        runningProvider: undefined,
        micEnabled: false,
        ready: settings.provider !== "off",
        detail: settings.provider === "off" ? "Voice is off." : "native voice ready",
      })

      class Channel {
        onmessage: (event: unknown) => void

        constructor(onmessage: (event: unknown) => void) {
          this.onmessage = onmessage
        }

        send(event: unknown) {
          this.onmessage(event)
        }
      }

      const invoke = async (cmd: string, args?: Record<string, unknown>) => {
        const safe =
          cmd === "voice_model_download"
            ? { source: args?.source, revision: args?.revision }
            : cmd === "voice_settings_set"
              ? { settings: args?.settings }
              : cmd === "voice_livekit_credentials_set"
                ? { apiKey: args?.apiKey, apiSecret: args?.apiSecret }
                : args
        calls.push({ cmd, args: safe })
        if (cmd === "voice_settings_state") return { settings, stored: true }
        if (cmd === "voice_settings_get") return settings
        if (cmd === "voice_status") return status()
        if (cmd === "voice_livekit_credentials_status") return { stored: livekit.credentialsStored }
        if (cmd === "voice_livekit_credentials_set") {
          livekit.credentialsStored = Boolean(args?.apiKey && args?.apiSecret)
          return undefined
        }
        if (cmd === "voice_settings_set") {
          Object.assign(settings, args?.settings as Settings)
          return undefined
        }
        if (cmd === "voice_model_download") {
          const channel = args?.channel as { send?: (event: unknown) => void; onmessage?: (event: unknown) => void }
          const send = (event: unknown) => {
            if (channel.send) {
              channel.send(event)
              return
            }
            channel.onmessage?.(event)
          }
          setTimeout(() => {
            send({ type: "started", total: 1 })
            send({ type: "file", index: 1, total: 1, name: "model.safetensors" })
            send({ type: "done", dir: "/tmp/lfm2-audio-downloaded" })
          }, 0)
          return undefined
        }
        if (cmd === "voice_hf_token_status") return false
        if (cmd === "voice_start") return { provider: settings.provider }
        if (
          cmd === "voice_stop" ||
          cmd === "voice_interrupt" ||
          cmd === "voice_begin_typed_input" ||
          cmd === "voice_set_mic_enabled"
        ) {
          return undefined
        }
        throw new Error(`unexpected tauri command: ${cmd}`)
      }

      Object.assign(window, {
        __voiceSettingsCalls: calls,
        __voiceSettingsEvents: events,
        __voiceSettingsState: settings,
        __TAURI__: { core: { invoke, Channel } },
      })
      window.addEventListener("emberharmony:voice-settings-changed", (event) => {
        const detail = (event as CustomEvent).detail
        events.push(detail == null ? "__credential_refresh__" : detail)
      })
    },
    { lastProvider },
  )
}

async function openVoiceSettings(page: Page) {
  const dialog = page.getByRole("dialog")
  await page.keyboard.press(`${modKey}+Comma`).catch(() => undefined)
  const opened = await dialog
    .waitFor({ state: "visible", timeout: 3000 })
    .then(() => true)
    .catch(() => false)
  if (!opened) {
    await page.getByRole("button", { name: "Settings" }).first().click({ timeout: 5000 })
    await expect(dialog).toBeVisible()
  }
  await dialog.getByRole("tab", { name: "Voice" }).click({ timeout: 5000 })
  await expect(dialog.getByRole("heading", { name: "Voice" })).toBeVisible()
  return dialog
}

async function state(page: Page): Promise<Settings> {
  return page.evaluate(() => ({ ...(window as unknown as { __voiceSettingsState?: Settings }).__voiceSettingsState! }))
}

async function calls(page: Page): Promise<Call[]> {
  return page.evaluate(() => (window as unknown as { __voiceSettingsCalls?: Call[] }).__voiceSettingsCalls ?? [])
}

async function events(page: Page): Promise<unknown[]> {
  return page.evaluate(() => (window as unknown as { __voiceSettingsEvents?: unknown[] }).__voiceSettingsEvents ?? [])
}

for (const provider of ["lfm2", "livekit"] as const) {
  test(`settings voice switch re-enables remembered native ${provider} provider`, async ({ page, gotoSession }) => {
    test.setTimeout(45_000)
    await install(page, provider)
    await gotoSession()

    const dialog = await openVoiceSettings(page)
    const toggle = dialog.getByRole("switch", { name: "Voice mode" }).first()
    await expect(toggle).toBeVisible()
    await expect(toggle).not.toBeChecked()

    await toggle.press("Space", { timeout: 5000 })

    await expect
      .poll(async () => state(page), { timeout: 5000 })
      .toMatchObject({
        provider,
        lastProvider: provider,
      })
    await expect
      .poll(
        async () =>
          (await calls(page)).some(
            (call) =>
              call.cmd === "voice_settings_set" &&
              (call.args as { settings?: Settings } | undefined)?.settings?.provider === provider &&
              (call.args as { settings?: Settings } | undefined)?.settings?.lastProvider === provider,
          ),
        { timeout: 5000 },
      )
      .toBe(true)
  })
}

for (const provider of ["lfm2", "livekit"] as const) {
  test(`settings downloads native LFM2 model for remembered ${provider} provider`, async ({ page, gotoSession }) => {
    test.setTimeout(45_000)
    const legacy: string[] = []
    await page.route("**/voice/config**", (route) => {
      legacy.push(`${route.request().method()} ${route.request().url()}`)
      return route.fulfill({ status: 500, body: "legacy voice config must not be used for native model download" })
    })
    await install(page, provider)
    await gotoSession()

    const dialog = await openVoiceSettings(page)
    const toggle = dialog.getByRole("switch", { name: "Voice mode" }).first()
    await expect(toggle).not.toBeChecked()
    await toggle.press("Space", { timeout: 5000 })

    await expect(dialog.getByRole("button", { name: "Download model" })).toBeVisible()
    if (provider === "livekit") {
      await expect(dialog.getByText("Speech models", { exact: true })).toHaveCount(0)
      await expect(dialog.getByText("Speech to text", { exact: true })).toHaveCount(0)
      await expect(dialog.getByText("Text to speech", { exact: true })).toHaveCount(0)
    }

    await dialog.getByRole("button", { name: "Download model" }).click()

    await expect
      .poll(
        async () =>
          (await calls(page)).some(
            (call) =>
              call.cmd === "voice_model_download" &&
              (call.args as { source?: string } | undefined)?.source === "LiquidAI/LFM2.5-Audio-1.5B",
          ),
        { timeout: 5000 },
      )
      .toBe(true)
    await expect
      .poll(async () => state(page), { timeout: 5000 })
      .toMatchObject({ provider, lfm2: { modelDir: "/tmp/lfm2-audio-downloaded" } })
    await expect
      .poll(
        async () =>
          (await calls(page)).some(
            (call) =>
              call.cmd === "voice_settings_set" &&
              (call.args as { settings?: Settings } | undefined)?.settings?.lfm2.modelDir ===
                "/tmp/lfm2-audio-downloaded",
          ),
        { timeout: 5000 },
      )
      .toBe(true)
    await expect
      .poll(
        async () =>
          (await events(page)).some(
            (event) =>
              (event as { lfm2?: { modelDir?: string } } | undefined)?.lfm2?.modelDir === "/tmp/lfm2-audio-downloaded",
          ),
        { timeout: 5000 },
      )
      .toBe(true)
    expect(legacy).toEqual([])
  })
}

test("settings save native LiveKit connection through Tauri only", async ({ page, gotoSession }) => {
  test.setTimeout(45_000)
  const legacy: string[] = []
  await page.route("**/voice/config**", (route) => {
    legacy.push(`${route.request().method()} ${route.request().url()}`)
    return route.fulfill({ status: 500, body: "legacy voice config must not be used in desktop voice settings" })
  })
  await page.route("**/auth/livekit**", (route) => {
    legacy.push(`${route.request().method()} ${route.request().url()}`)
    return route.fulfill({ status: 500, body: "legacy auth must not be used in desktop voice settings" })
  })
  await install(page, "livekit")
  await gotoSession()

  const dialog = await openVoiceSettings(page)
  const toggle = dialog.getByRole("switch", { name: "Voice mode" }).first()
  await expect(toggle).not.toBeChecked()
  await toggle.press("Space", { timeout: 5000 })

  await dialog.getByLabel("Server URL").fill("wss://native-livekit.invalid")
  await dialog.getByLabel("API key").fill("native-key")
  await dialog.getByLabel("API secret").fill("native-secret")
  await dialog.getByRole("button", { name: "Save connection" }).click()

  await expect
    .poll(
      async () =>
        (await calls(page)).some(
          (call) =>
            call.cmd === "voice_settings_set" &&
            (call.args as { settings?: Settings } | undefined)?.settings?.livekit.url ===
              "wss://native-livekit.invalid",
        ),
      { timeout: 5000 },
    )
    .toBe(true)
  await expect
    .poll(
      async () =>
        (await calls(page)).some(
          (call) =>
            call.cmd === "voice_livekit_credentials_set" &&
            (call.args as { apiKey?: string; apiSecret?: string } | undefined)?.apiKey === "native-key" &&
            (call.args as { apiKey?: string; apiSecret?: string } | undefined)?.apiSecret === "native-secret",
        ),
      { timeout: 5000 },
    )
    .toBe(true)
  await expect
    .poll(
      async () => {
        const values = await events(page)
        return values.some((value) => value === "__credential_refresh__")
      },
      { timeout: 5000 },
    )
    .toBe(true)
  expect(legacy).toEqual([])
})
