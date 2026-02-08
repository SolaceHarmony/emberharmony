export * from "./client.js"
export * from "./server.js"

import { createCodeHarmonyClient, createOpencodeClient } from "./client.js"
import { createCodeHarmonyServer, createOpencodeServer } from "./server.js"
import type { ServerOptions } from "./server.js"

export async function createCodeHarmony(options?: ServerOptions) {
  const server = await createCodeHarmonyServer({
    ...options,
  })

  const client = createCodeHarmonyClient({
    baseUrl: server.url,
  })

  return {
    client,
    server,
  }
}

// Backwards compatibility for older consumers.
export const createOpencode = async (options?: ServerOptions) => {
  const server = await createOpencodeServer(options)
  const client = createOpencodeClient({ baseUrl: server.url })
  return { client, server }
}
