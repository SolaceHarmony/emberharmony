export * from "./client.js"
export * from "./server.js"

import { createEmberHarmonyClient } from "./client.js"
import { createEmberHarmonyServer } from "./server.js"
import type { ServerOptions } from "./server.js"

export async function createEmberHarmony(options?: ServerOptions) {
  const server = await createEmberHarmonyServer({
    ...options,
  })

  const client = createEmberHarmonyClient({
    baseUrl: server.url,
  })

  return {
    client,
    server,
  }
}
