export * from "./gen/types.gen.js"

import { createClient } from "./gen/client/client.gen.js"
import { type Config } from "./gen/client/types.gen.js"
import { CodeHarmonyClient } from "./gen/sdk.gen.js"
export { type Config as CodeHarmonyClientConfig, CodeHarmonyClient }

export function createCodeHarmonyClient(config?: Config & { directory?: string }) {
  const cfg = { ...config }

  const headers = { ...(cfg.headers ?? {}) } as Record<string, string>
  if (cfg.directory) {
    const isNonASCII = /[^\x00-\x7F]/.test(cfg.directory)
    const encodedDirectory = isNonASCII ? encodeURIComponent(cfg.directory) : cfg.directory
    headers["x-code-harmony-directory"] = encodedDirectory
  }

  const client = createClient({ ...cfg, headers })
  return new CodeHarmonyClient({ client })
}
