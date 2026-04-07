import { describe, expect, test } from "bun:test"
import { Provider } from "../../src/provider/provider"
import { MessageV2 } from "../../src/session/message-v2"
import { APICallError } from "ai"

describe("lmstudio errors", () => {
  test("connection refused surfaces providerID + url", () => {
    const err = new Provider.RequestFailedError({
      providerID: "lmstudio",
      url: "http://127.0.0.1:65534/v1/chat/completions",
      error: "connect ECONNREFUSED 127.0.0.1:65534",
    })

    const obj = MessageV2.fromError(err, { providerID: "lmstudio" })
    expect(MessageV2.APIError.isInstance(obj)).toBe(true)
    expect((obj as MessageV2.APIError).data.message).toContain("lmstudio")
    expect((obj as MessageV2.APIError).data.message).toContain("65534")
    expect(String((obj as MessageV2.APIError).data.metadata?.url ?? "")).toContain("65534")
  })

  test("context window error is converted into an actionable message", () => {
    const err = new APICallError({
      message: "Bad Request",
      url: "http://127.0.0.1:1234/v1/chat/completions",
      requestBodyValues: {},
      statusCode: 400,
      responseHeaders: { "content-type": "application/json" },
      responseBody: JSON.stringify({
        error: {
          message:
            "The number of tokens to keep from the initial prompt is greater than the context length. Try to load the model with a larger context length, or provide a shorter input",
        },
      }),
      isRetryable: false,
    })

    const obj = MessageV2.fromError(err, { providerID: "lmstudio" })
    expect(obj.name).toBe("APIError")
    expect(String(obj.data.message)).toContain("LM Studio returned a context window error")
  })
})
