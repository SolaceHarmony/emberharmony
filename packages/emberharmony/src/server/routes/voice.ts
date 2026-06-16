import { Hono } from "hono"
import { describeRoute, validator, resolver } from "hono-openapi"
import z from "zod"
import { Auth } from "../../auth"
import { Config } from "../../config/config"
import { Voice } from "../../voice/token"
import { VoiceRegistry } from "../../voice/registry"
import { VoiceWorker } from "../../voice/worker"
import { Instance } from "../../project/instance"
import {
  ensureVoiceProject,
  createVoiceSession,
  gatherRecentVoiceContext,
  BRAIN_SYSTEM_PROMPT,
  VOICE_PROJECT_DIR,
} from "../../voice/brain"
import { errors } from "../error"
import { lazy } from "../../util/lazy"

const RegistryOption = z
  .object({
    id: z.string(),
    name: z.string(),
    provider: z.string(),
    defaultSuffix: z.string().optional(),
  })
  .meta({ ref: "VoiceRegistryOption" })

const VoiceConfigInfo = z
  .object({
    available: z.boolean(),
    disabled: z.boolean(),
    url: z.string().nullable(),
    brain: z.string().nullable(),
    stt: z.string(),
    tts: z.string(),
    intent: z.string(),
    structured: z.boolean(),
    registry: z.object({
      stt: RegistryOption.array(),
      tts: RegistryOption.array(),
    }),
    credentials: z.object({
      livekit: z.boolean(),
    }),
  })
  .meta({ ref: "VoiceConfigInfo" })

async function configInfo(override?: Config.Voice) {
  const settings = await Voice.settings(override)
  const auth = await Auth.get(Voice.AUTH_PROVIDER_ID)
  return {
    available: settings.available,
    disabled: settings.disabled,
    url: settings.url ?? null,
    brain: settings.brain ?? null,
    stt: settings.stt,
    tts: settings.tts,
    intent: settings.intent,
    structured: settings.structured,
    registry: { stt: VoiceRegistry.STT, tts: VoiceRegistry.TTS },
    credentials: {
      livekit: Boolean(auth?.type === "api" && auth.key && auth.secret),
    },
  }
}

export const VoiceRoutes = lazy(() =>
  new Hono()
    .get(
      "/status",
      describeRoute({
        summary: "Get voice status",
        description: "Check whether LiveKit voice is configured on this server.",
        operationId: "voice.status",
        responses: {
          200: {
            description: "Voice availability",
            content: {
              "application/json": {
                schema: resolver(
                  z.object({
                    available: z.boolean(),
                    url: z.string().nullable(),
                  }),
                ),
              },
            },
          },
        },
      }),
      async (c) => {
        const settings = await Voice.settings()
        return c.json({ available: settings.available, url: settings.available ? (settings.url ?? null) : null })
      },
    )
    .get(
      "/config",
      describeRoute({
        summary: "Get voice configuration",
        description:
          "Effective voice settings, the registry of supported STT/TTS integrations, and which credentials are stored (never the secrets themselves).",
        operationId: "voice.config",
        responses: {
          200: {
            description: "Voice configuration",
            content: {
              "application/json": {
                schema: resolver(VoiceConfigInfo),
              },
            },
          },
        },
      }),
      async (c) => {
        return c.json(await configInfo())
      },
    )
    .patch(
      "/config",
      describeRoute({
        summary: "Update voice configuration",
        description:
          "Update voice settings in the global config. LiveKit credentials are stored separately via the auth API (PUT /auth/livekit).",
        operationId: "voice.configUpdate",
        responses: {
          200: {
            description: "Updated voice configuration",
            content: {
              "application/json": {
                schema: resolver(VoiceConfigInfo),
              },
            },
          },
          ...errors(400),
        },
      }),
      validator("json", Config.Voice),
      async (c) => {
        const body = c.req.valid("json")
        const merged = await Config.updateGlobal({ voice: body })
        // pick up the new settings immediately when serve manages the worker
        await VoiceWorker.restart(merged.voice ?? {})
        // updateGlobal disposes instance caches asynchronously; respond from
        // the merged result instead of racing the stale cache
        return c.json(await configInfo(merged.voice ?? {}))
      },
    )
    .post(
      "/token",
      describeRoute({
        summary: "Create voice token",
        description: "Generate a LiveKit access token for a voice session.",
        operationId: "voice.token",
        responses: {
          200: {
            description: "LiveKit access token",
            content: {
              "application/json": {
                schema: resolver(
                  z.object({
                    token: z.string(),
                    url: z.string(),
                    roomName: z.string(),
                    directory: z.string(),
                    projectID: z.string(),
                  }),
                ),
              },
            },
          },
          ...errors(400),
        },
      }),
      validator(
        "json",
        z.object({
          sessionID: z.string().optional(),
          agentName: z.string().optional(),
          model: z
            .object({ providerID: z.string(), modelID: z.string() })
            .optional()
            .describe("Model to use for voice turns when the session has no message history yet"),
        }),
      ),
      async (c) => {
        const body = c.req.valid("json")
        // One room per project: emberharmony_voice_{projectID}. When no project
        // context is active (e.g. connecting from the TUI or desktop without a
        // session open), fall back to the permanent voice brain project so the
        // agent always has a valid session to work with.
        let directory: string
        let projectID: string
        try {
          directory = Instance.directory
          projectID = Instance.project.id
        } catch {
          // No Instance context (TUI, desktop without session) — use voice project
          directory = await ensureVoiceProject()
          projectID = "voice"
        }
        const roomName = `emberharmony_voice_${projectID}`
        const agentName = body.agentName ?? Voice.AGENT_NAME
        const resolved = await Voice.settings()
        // the agent worker uses this to bridge the voice conversation into
        // the EmberHarmony session (same tools, permissions, and context)
        const agentMetadata = JSON.stringify({
          projectID,
          directory,
          serverUrl: new URL(c.req.url).origin,
          model: body.model,
          brainModel: resolved.brain,
          structured: resolved.structured,
        })
        const result = await Voice.token({
          roomName,
          identity: `user_${projectID}`,
          agentName,
          agentMetadata,
        })
        await Voice.ensureAgentDispatched({ roomName, agentName, metadata: agentMetadata })
        return c.json({ token: result.token, url: result.url, roomName, directory, projectID })
      },
    )
    .get(
      "/brain",
      describeRoute({
        summary: "Start a voice conversation",
        description:
          "Create a NEW conversation session in the voice project and return it with the brain " +
          "system prompt (which folds in memory of recent conversations). The voice agent worker " +
          "calls this at startup to get the session it should target.",
        operationId: "voice.brain",
        responses: {
          200: {
            description: "Voice brain session info",
            content: {
              "application/json": {
                schema: resolver(
                  z.object({
                    sessionID: z.string(),
                    directory: z.string(),
                    system: z.string(),
                  }),
                ),
              },
            },
          },
        },
      }),
      async (c) => {
        // Each conversation is a NEW session in the voice project. Continuity
        // comes from recent-conversation memory folded into the system prompt
        // (the orient step), not from reusing one eternal session.
        const sessionID = await createVoiceSession()
        const recent = await gatherRecentVoiceContext(sessionID)
        const system = recent ? `${BRAIN_SYSTEM_PROMPT}\n\n${recent}` : BRAIN_SYSTEM_PROMPT
        return c.json({ sessionID, directory: VOICE_PROJECT_DIR, system })
      },
    ),
)
