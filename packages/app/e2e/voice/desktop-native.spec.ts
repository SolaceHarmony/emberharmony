import { type Page } from "@playwright/test"
import { test, expect } from "../fixtures"
import { modKey, promptSelector } from "../utils"

type Provider = "lfm2" | "livekit"
type Call = { cmd: string; args?: unknown }
type Runtime = { running: boolean; runningProvider?: Provider; micEnabled: boolean }
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

async function install(page: Page, provider: Provider, options?: { ready?: boolean; detail?: string }) {
  await page.addInitScript(
    (input: { provider: Provider; ready: boolean; detail: string }) => {
      const calls: Call[] = []
      const state: { channel?: { send: (event: unknown) => void } } = {}
      const runtime: Runtime = { running: false, micEnabled: false }
      const settings: Settings = {
        provider: input.provider,
        lastProvider: input.provider,
        livekit: { url: "wss://livekit.invalid" },
        lfm2: {
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
      const plan = () => ({
        provider: settings.provider,
        enabled: settings.provider !== "off",
        surface: settings.provider === "off" ? "off" : "native",
        running: runtime.running,
        runningProvider: runtime.runningProvider,
        micEnabled: runtime.micEnabled,
        ready: settings.provider !== "off" && input.ready,
        detail: settings.provider === "off" ? "Voice is off." : input.detail,
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

      const record = (cmd: string, args?: Record<string, unknown>) => {
        const safe =
          cmd === "voice_start"
            ? { ctx: args?.ctx }
            : cmd === "voice_set_mic_enabled"
              ? { enabled: args?.enabled }
              : undefined
        calls.push({ cmd, args: safe })
      }

      const invoke = async (cmd: string, args?: Record<string, unknown>) => {
        record(cmd, args)
        if (cmd === "voice_settings_state") return { settings, stored: true }
        if (cmd === "voice_settings_get") return settings
        if (cmd === "voice_status") return plan()
        if (cmd === "voice_settings_set") {
          Object.assign(settings, args?.settings as Settings)
          if (runtime.running && settings.provider !== runtime.runningProvider) {
            runtime.running = false
            runtime.runningProvider = undefined
            runtime.micEnabled = false
            state.channel?.send({ type: "ended" })
          }
          return undefined
        }
        if (cmd === "voice_start") {
          if (!plan().ready) throw new Error(input.detail)
          state.channel = args?.channel as { send: (event: unknown) => void }
          runtime.running = true
          runtime.runningProvider = input.provider
          runtime.micEnabled = true
          state.channel.send({ type: "state", state: "listening" })
          state.channel.send({ type: "level", rms: 0.02 })
          return { provider: input.provider }
        }
        if (cmd === "voice_interrupt") {
          state.channel?.send({ type: "state", state: "listening" })
          state.channel?.send({ type: "level", rms: 0 })
          return undefined
        }
        if (cmd === "voice_begin_typed_input") {
          runtime.micEnabled = false
          state.channel?.send({ type: "state", state: "idle" })
          state.channel?.send({ type: "level", rms: 0 })
          return undefined
        }
        if (cmd === "voice_stop") {
          runtime.running = false
          runtime.runningProvider = undefined
          runtime.micEnabled = false
          state.channel?.send({ type: "ended" })
          return undefined
        }
        if (cmd === "voice_set_mic_enabled") {
          runtime.micEnabled = args?.enabled === true
          return undefined
        }
        throw new Error(`unexpected tauri command: ${cmd}`)
      }

      Object.assign(window, {
        __voiceCalls: calls,
        __voiceRuntime: runtime,
        __voiceSend: (event: unknown) => state.channel?.send(event),
        __voiceStopSilently: () => {
          runtime.running = false
          runtime.runningProvider = undefined
          runtime.micEnabled = false
          window.dispatchEvent(new CustomEvent("emberharmony:voice-settings-changed"))
        },
        __voiceApplySettings: async (next: Settings) => {
          await invoke("voice_settings_set", { settings: next })
          window.dispatchEvent(new CustomEvent("emberharmony:voice-settings-changed", { detail: next }))
        },
        __TAURI__: { core: { invoke, Channel } },
      })
    },
    { provider, ready: options?.ready ?? true, detail: options?.detail ?? "native voice e2e ready" },
  )
}

async function calls(page: Page): Promise<Call[]> {
  return page.evaluate(() => (window as unknown as { __voiceCalls?: Call[] }).__voiceCalls ?? [])
}

async function runtime(page: Page): Promise<Runtime> {
  return page.evaluate(() => ({ ...(window as unknown as { __voiceRuntime?: Runtime }).__voiceRuntime! }))
}

async function send(page: Page, event: unknown) {
  await page.evaluate((value) => {
    ;(window as unknown as { __voiceSend?: (event: unknown) => void }).__voiceSend?.(value)
  }, event)
}

async function applySettings(page: Page, settings: Settings) {
  await page.evaluate((value) => {
    return (
      window as unknown as { __voiceApplySettings?: (settings: Settings) => Promise<void> }
    ).__voiceApplySettings?.(value)
  }, settings)
}

async function stopSilently(page: Page) {
  await page.evaluate(() => {
    ;(window as unknown as { __voiceStopSilently?: () => void }).__voiceStopSilently?.()
  })
}

for (const provider of ["lfm2", "livekit"] as const) {
  test(`desktop ${provider} keeps the native mic affordance on and asks the kernel when not ready`, async ({
    page,
    sdk,
    gotoSession,
  }) => {
    await install(page, provider, {
      ready: false,
      detail: provider === "lfm2" ? "No local model." : "Enter your LiveKit URL, API key, and API secret.",
    })

    const title = `e2e native voice not ready ${provider} ${Date.now()}`
    const created = await sdk.session.create({ title }).then((r) => r.data)
    if (!created?.id) throw new Error("Session create did not return an id")

    try {
      await gotoSession(created.id)

      const voice = page.getByRole("button", { name: "Voice mode" })
      await expect(voice).toBeVisible()
      await expect(voice).toHaveAttribute("aria-pressed", "true")

      await voice.click()

      await expect.poll(async () => (await calls(page)).filter((call) => call.cmd === "voice_start").length).toBe(1)
      const start = (await calls(page)).find((call) => call.cmd === "voice_start")
      expect(start?.args).toMatchObject({ ctx: { sessionID: created.id } })
      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: false,
          micEnabled: false,
        })
      await expect(voice).toHaveAttribute("aria-pressed", "true")
    } finally {
      await sdk.session.delete({ sessionID: created.id }).catch(() => undefined)
    }
  })

  test(`desktop ${provider} voice uses the native Tauri kernel from the prompt`, async ({ page, sdk, gotoSession }) => {
    await install(page, provider)

    const title = `e2e native voice ${provider} ${Date.now()}`
    const created = await sdk.session.create({ title }).then((r) => r.data)
    if (!created?.id) throw new Error("Session create did not return an id")

    try {
      await gotoSession(created.id)

      const voice = page.getByRole("button", { name: "Voice mode" })
      await expect(voice).toBeVisible()
      await voice.click()

      await expect.poll(async () => (await calls(page)).filter((call) => call.cmd === "voice_start").length).toBe(1)
      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: true,
          runningProvider: provider,
          micEnabled: true,
        })
      await expect(voice).toHaveAttribute("aria-pressed", "true")

      const start = (await calls(page)).find((call) => call.cmd === "voice_start")
      expect(start?.args).toMatchObject({ ctx: { sessionID: created.id } })

      await send(page, { type: "state", state: "speaking" })
      await expect(page.getByRole("button", { name: "Stop" })).toBeEnabled()

      await send(page, { type: "state", state: "listening" })
      await expect(page.getByRole("button", { name: "Send" })).toBeDisabled()

      const prompt = page.locator(promptSelector)
      await prompt.click()
      await page.keyboard.type("hello while voice is active")

      await expect
        .poll(async () => (await calls(page)).some((call) => call.cmd === "voice_begin_typed_input"))
        .toBe(true)
      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: true,
          runningProvider: provider,
          micEnabled: false,
        })
      await expect(voice).toHaveAttribute("aria-pressed", "true")

      await page.keyboard.press(`${modKey}+A`)
      await page.keyboard.press("Backspace")

      await expect
        .poll(
          async () =>
            (await calls(page)).some(
              (call) =>
                call.cmd === "voice_set_mic_enabled" &&
                (call.args as { enabled?: boolean } | undefined)?.enabled === true,
            ),
          { timeout: 5000 },
        )
        .toBe(true)
      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: true,
          runningProvider: provider,
          micEnabled: true,
        })

      await send(page, { type: "state", state: "speaking" })
      await page.getByRole("button", { name: "Stop" }).click()

      await expect.poll(async () => (await calls(page)).some((call) => call.cmd === "voice_interrupt")).toBe(true)
      await expect(page.getByRole("button", { name: "Stop" })).toBeHidden()

      await voice.click()
      await expect.poll(async () => (await calls(page)).some((call) => call.cmd === "voice_stop")).toBe(true)
      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: false,
          micEnabled: false,
        })
    } finally {
      await sdk.session.delete({ sessionID: created.id }).catch(() => undefined)
    }
  })

  test(`desktop ${provider} voice keeps delegation target in native Tauri settings`, async ({
    page,
    sdk,
    gotoSession,
  }) => {
    await install(page, provider)

    const title = `e2e native voice delegate ${provider} ${Date.now()}`
    const created = await sdk.session.create({ title }).then((r) => r.data)
    if (!created?.id) throw new Error("Session create did not return an id")

    try {
      await gotoSession(created.id)
      await applySettings(page, {
        provider,
        lastProvider: provider,
        livekit: { url: "wss://livekit.invalid" },
        lfm2: {
          model: "LiquidAI/LFM2.5-Audio-1.5B",
          modelDir: "/tmp/lfm2-audio-e2e",
          device: "metal",
          vadThreshold: 0.012,
          asr: { textTemperature: 0, textTopK: 0, audioTemperature: 0, audioTopK: 0, maxTokens: 100 },
          tts: { textTemperature: 0.7, textTopK: 0, audioTemperature: 0.8, audioTopK: 64, maxTokens: 1024 },
          interleaved: { textTemperature: 1.0, textTopK: 0, audioTemperature: 1.0, audioTopK: 4, maxTokens: 8192 },
          delegate: { enabled: false },
        },
      })

      const voice = page.getByRole("button", { name: "Voice mode" })
      await expect(voice).toBeVisible()
      await voice.click()

      await expect.poll(async () => (await calls(page)).filter((call) => call.cmd === "voice_start").length).toBe(1)

      const start = (await calls(page)).find((call) => call.cmd === "voice_start")
      const ctx = (start?.args as { ctx?: Record<string, unknown> } | undefined)?.ctx
      expect(ctx).toMatchObject({ sessionID: created.id })
      expect(ctx && "delegateTarget" in ctx).toBe(false)
    } finally {
      await sdk.session.delete({ sessionID: created.id }).catch(() => undefined)
    }
  })

  test(`desktop ${provider} voice stops when Tauri settings invalidate the native session`, async ({
    page,
    sdk,
    gotoSession,
  }) => {
    await install(page, provider)

    const title = `e2e native voice settings invalidate ${provider} ${Date.now()}`
    const created = await sdk.session.create({ title }).then((r) => r.data)
    if (!created?.id) throw new Error("Session create did not return an id")

    try {
      await gotoSession(created.id)

      const voice = page.getByRole("button", { name: "Voice mode" })
      await expect(voice).toBeVisible()
      await voice.click()

      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: true,
          runningProvider: provider,
          micEnabled: true,
        })
      await expect(voice).toHaveAttribute("aria-pressed", "true")

      const before = (await calls(page)).filter((call) => call.cmd === "voice_stop").length
      await applySettings(page, {
        provider: "off",
        lastProvider: provider,
        livekit: { url: "wss://livekit.invalid" },
        lfm2: {
          model: "LiquidAI/LFM2.5-Audio-1.5B",
          modelDir: "/tmp/lfm2-audio-e2e",
          device: "metal",
          vadThreshold: 0.012,
          asr: { textTemperature: 0, textTopK: 0, audioTemperature: 0, audioTopK: 0, maxTokens: 100 },
          tts: { textTemperature: 0.7, textTopK: 0, audioTemperature: 0.8, audioTopK: 64, maxTokens: 1024 },
          interleaved: { textTemperature: 1.0, textTopK: 0, audioTemperature: 1.0, audioTopK: 4, maxTokens: 8192 },
          delegate: { enabled: false },
        },
      })

      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: false,
          micEnabled: false,
        })
      await expect(voice).toHaveAttribute("aria-pressed", "false")
      expect((await calls(page)).filter((call) => call.cmd === "voice_stop").length).toBe(before)
    } finally {
      await sdk.session.delete({ sessionID: created.id }).catch(() => undefined)
    }
  })

  test(`desktop ${provider} voice clears stale speaking UI from the native runtime snapshot`, async ({
    page,
    sdk,
    gotoSession,
  }) => {
    await install(page, provider)

    const title = `e2e native voice snapshot clears ${provider} ${Date.now()}`
    const created = await sdk.session.create({ title }).then((r) => r.data)
    if (!created?.id) throw new Error("Session create did not return an id")

    try {
      await gotoSession(created.id)

      const voice = page.getByRole("button", { name: "Voice mode" })
      await expect(voice).toBeVisible()
      await voice.click()

      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: true,
          runningProvider: provider,
          micEnabled: true,
        })

      await send(page, { type: "state", state: "speaking" })
      const stop = page.getByRole("button", { name: "Stop" })
      await expect(stop).toBeEnabled()

      await stopSilently(page)

      await expect
        .poll(async () => runtime(page))
        .toMatchObject({
          running: false,
          micEnabled: false,
        })
      await expect(stop).toBeHidden()
      await expect(page.getByRole("button", { name: "Send" })).toBeDisabled()
      await expect(voice).toHaveAttribute("aria-pressed", "true")
    } finally {
      await sdk.session.delete({ sessionID: created.id }).catch(() => undefined)
    }
  })
}
