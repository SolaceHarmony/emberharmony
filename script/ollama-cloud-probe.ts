const main = async () => {
  const key = process.env.OLLAMA_API_KEY
  const base = (process.env.OLLAMA_BASE_URL ?? "https://ollama.com/v1").replace(/\/+$/, "")
  const url = base + "/chat/completions"

  const cloud = base === "https://ollama.com/v1"
  if (cloud && !key) {
    console.error("Missing OLLAMA_API_KEY (required for Ollama Cloud).")
    process.exit(1)
  }

  const model = process.argv[2] ?? "gemini-3-pro-preview"

  const todowrite = await Bun.file("packages/code-harmony/src/tool/todowrite.txt")
    .text()
    .catch(() => "")

  const mk = (n: number) => "x".repeat(n)

  const base = {
    model,
    stream: false,
    max_tokens: 16,
    temperature: 0,
    messages: [{ role: "user", content: "hi" }],
  }

  const tool = (desc: string, params: Record<string, unknown>) => ({
    tools: [
      {
        type: "function",
        function: {
          name: "ping",
          description: desc,
          parameters: params,
        },
      },
    ],
    tool_choice: "auto",
  })

  const cases = [
    {
      name: "no-tools",
      body: base,
    },
    {
      name: "tools-min",
      body: {
        ...base,
        ...tool("ping", { type: "object", properties: {}, additionalProperties: false }),
      },
    },
    {
      name: "tools-schema",
      body: {
        ...base,
        ...tool("ping", {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          type: "object",
          properties: {},
          additionalProperties: false,
        }),
      },
    },
    {
      name: "tools-desc-8k",
      body: {
        ...base,
        ...tool(mk(8192), { type: "object", properties: {}, additionalProperties: false }),
      },
    },
    {
      name: "tools-desc-16k",
      body: {
        ...base,
        ...tool(mk(16384), { type: "object", properties: {}, additionalProperties: false }),
      },
    },
    {
      name: "tools-todowrite",
      body: {
        ...base,
        ...tool(todowrite, {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          type: "object",
          properties: {
            todos: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  content: { type: "string" },
                  status: { type: "string" },
                  priority: { type: "string" },
                  id: { type: "string" },
                },
                required: ["content", "status", "priority", "id"],
                additionalProperties: false,
              },
            },
          },
          required: ["todos"],
          additionalProperties: false,
        }),
      },
    },
  ] as const

  console.log("base\t" + base)
  for (const item of cases) {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(key ? { authorization: `Bearer ${key}` } : {}),
      },
      body: JSON.stringify(item.body),
    })

    const text = await res.text()
    const line = text.replace(/\s+/g, " ").slice(0, 200)
    console.log([item.name, String(res.status), line].join("\t"))
  }
}

await main()
