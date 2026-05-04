import path from "path"

const dir = process.env.EMBERHARMONY_E2E_PROJECT_DIR ?? process.cwd()
const title = process.env.EMBERHARMONY_E2E_SESSION_TITLE ?? "E2E Session"
const text = process.env.EMBERHARMONY_E2E_MESSAGE ?? "Seeded for UI e2e"
const model = process.env.EMBERHARMONY_E2E_MODEL ?? "mock/mock-model"
const parts = model.split("/")
const providerID = parts[0] ?? "mock"
const modelID = parts[1] ?? "mock-model"
const now = Date.now()

// Write a mock provider config so E2E tests have models available without real credentials.
// This config is only written when the config file does not already exist in the directory,
// so it won't overwrite real project configurations.
const configPath = path.join(dir, "emberharmony.json")
const configExists = await Bun.file(configPath).exists()
if (!configExists) {
  const mockConfig = {
    $schema: "https://solace.ofharmony.ai/config.json",
    provider: {
      mock: {
        name: "Mock Provider",
        npm: "@ai-sdk/openai-compatible",
        env: [],
        models: {
          "mock-model": {
            name: "Mock Model",
            tool_call: true,
            limit: { context: 16000, output: 4096 },
          },
          "mock-model-2": {
            name: "Mock Model 2",
            tool_call: true,
            limit: { context: 16000, output: 4096 },
          },
        },
        options: {
          apiKey: "mock-key",
          // baseURL is never called by model-picker/visibility E2E tests
          // (they only list models, they don't send chat requests).
          // Port 4097 is an arbitrary unused port; the URL is a valid placeholder.
          baseURL: "http://127.0.0.1:4097/v1",
        },
      },
    },
  }
  await Bun.write(configPath, JSON.stringify(mockConfig, null, 2))
}

const seed = async () => {
  const { Instance } = await import("../src/project/instance")
  const { InstanceBootstrap } = await import("../src/project/bootstrap")
  const { Session } = await import("../src/session")
  const { Identifier } = await import("../src/id/id")
  const { Project } = await import("../src/project/project")

  await Instance.provide({
    directory: dir,
    init: InstanceBootstrap,
    fn: async () => {
      const session = await Session.create({ title })
      const messageID = Identifier.descending("message")
      const partID = Identifier.descending("part")
      const message = {
        id: messageID,
        sessionID: session.id,
        role: "user" as const,
        time: { created: now },
        agent: "build",
        model: {
          providerID,
          modelID,
        },
      }
      const part = {
        id: partID,
        sessionID: session.id,
        messageID,
        type: "text" as const,
        text,
        time: { start: now },
      }
      await Session.updateMessage(message)
      await Session.updatePart(part)
      await Project.update({ projectID: Instance.project.id, name: "E2E Project" })
    },
  })
}

await seed()
