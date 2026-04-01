export * from "./client.js"
export * from "./server.js"

import { createCodeHarmonyClient } from "./client.js"
import { createCodeHarmonyServer } from "./server.js"
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
