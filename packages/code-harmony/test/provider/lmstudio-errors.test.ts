import { describe, expect, test } from "bun:test"
import path from "path"
import { tmpdir } from "../fixture/fixture"
import { Instance } from "../../src/project/instance"
import { Provider } from "../../src/provider/provider"
import { Env } from "../../src/env"
import { streamText } from "ai"
import { MessageV2 } from "../../src/session/message-v2"

describe("lmstudio errors", () => {
  test("connection refused surfaces providerID + url", async () => {
    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "code-harmony.json"),
          JSON.stringify({
            $schema: "https://solace.ofharmony.ai/config.json",
            provider: {
              lmstudio: {
                api: "http://127.0.0.1:65534/v1",
                options: {
                  timeout: 500,
                },
              },
            },
          }),
        )
      },
    })

	    await Instance.provide({
	      directory: tmp.path,
	      init: async () => {
	        Env.set("LMSTUDIO_API_KEY", "test")
	      },
	      fn: async () => {
	        const model = await Provider.getModel("lmstudio", "openai/gpt-oss-20b")
	        const lang = await Provider.getLanguage(model)
	        const err = await streamText({
	          model: lang,
	          messages: [
	            {
	              role: "user",
	              content: "hi",
	            },
	          ],
	        }).text.then(() => undefined).catch((e: unknown) => e)

	        expect(err).toBeDefined()
	        const obj = MessageV2.fromError(err, { providerID: "lmstudio" })
	        expect(MessageV2.APIError.isInstance(obj)).toBe(true)
	        expect((obj as MessageV2.APIError).data.message).toContain("lmstudio")
	        expect((obj as MessageV2.APIError).data.message).toContain("65534")
	        expect(String((obj as MessageV2.APIError).data.metadata?.url ?? "")).toContain("65534")
	      },
	    })
	  })

  test("context window error is converted into an actionable message", async () => {
    const srv = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch(req) {
        const url = new URL(req.url)
        if (url.pathname === "/v1/models") {
          return Response.json({
            object: "list",
            data: [{ id: "openai/gpt-oss-20b", object: "model" }],
          })
        }
        if (url.pathname === "/v1/chat/completions") {
          return Response.json(
            {
              error: {
                message:
                  "The number of tokens to keep from the initial prompt is greater than the context length. Try to load the model with a larger context length, or provide a shorter input",
              },
            },
            { status: 400 },
          )
        }
        return new Response("not found", { status: 404 })
      },
    })

    try {
      await using tmp = await tmpdir({
        init: async (dir) => {
          await Bun.write(
            path.join(dir, "code-harmony.json"),
            JSON.stringify({
              $schema: "https://solace.ofharmony.ai/config.json",
              provider: {
                lmstudio: {
                  api: `http://127.0.0.1:${srv.port}/v1`,
                },
              },
            }),
          )
        },
      })

      await Instance.provide({
        directory: tmp.path,
        init: async () => {
          Env.set("LMSTUDIO_API_KEY", "test")
        },
	        fn: async () => {
	          const model = await Provider.getModel("lmstudio", "openai/gpt-oss-20b")
	          const lang = await Provider.getLanguage(model)
	          const err = await streamText({
	            model: lang,
	            messages: [{ role: "user", content: "hi" }],
	          }).text.then(() => undefined).catch((e: unknown) => e)

	          expect(err).toBeDefined()
	          const obj = MessageV2.fromError(err, { providerID: "lmstudio" })
	          expect(obj.name).toBe("APIError")
	          expect(String(obj.data.message)).toContain("LM Studio returned a context window error")
	        },
      })
    } finally {
      srv.stop(true)
    }
  })
})
