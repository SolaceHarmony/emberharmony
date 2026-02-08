export * from "./gen/types.gen.js"

import { createClient } from "./gen/client/client.gen.js"
import { type Config } from "./gen/client/types.gen.js"
import { OpencodeClient } from "./gen/sdk.gen.js"
export { type Config as OpencodeClientConfig, OpencodeClient }
export { type Config as CodeHarmonyClientConfig, OpencodeClient as CodeHarmonyClient }

export function createCodeHarmonyClient(config?: Config & { directory?: string }) {
  const cfg = { ...config }

  const headers = { ...(cfg.headers ?? {}) } as Record<string, string>
  if (cfg.directory) {
    headers["x-opencode-directory"] = cfg.directory
    headers["x-code-harmony-directory"] = cfg.directory
  }

  const client = createClient({ ...cfg, headers })
  return new OpencodeClient({ client })
}

// Backwards compatibility for older consumers.
export const createOpencodeClient = createCodeHarmonyClient
