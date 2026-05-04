#!/usr/bin/env bun
/**
 * Update the vendored models-snapshot.ts from models.dev/api.json.
 *
 * This script must be run explicitly by a code owner when the snapshot needs
 * to be refreshed. The resulting diff should be reviewed in a PR before merge.
 *
 * Usage:
 *   bun packages/emberharmony/script/update-models-snapshot.ts
 *
 * Supply a local file instead of fetching (useful for offline testing):
 *   MODELS_DEV_API_JSON=/path/to/api.json bun packages/emberharmony/script/update-models-snapshot.ts
 */

import path from "path"
import { fileURLToPath } from "url"
import { createHash } from "crypto"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const dir = path.resolve(__dirname, "..")

const snapshotPath = path.join(dir, "src/provider/models-snapshot.ts")

const data = process.env.MODELS_DEV_API_JSON
  ? await Bun.file(process.env.MODELS_DEV_API_JSON).text()
  : await fetch("https://models.dev/api.json", {
      signal: AbortSignal.timeout(30_000),
    }).then((r) => {
      if (!r.ok) throw new Error(`models.dev responded with ${r.status} ${r.statusText}`)
      return r.text()
    })

// Validate JSON before writing
JSON.parse(data)

const hash = createHash("sha256").update(data).digest("hex")
const formatted = JSON.stringify(JSON.parse(data), null, 2)

await Bun.write(
  snapshotPath,
  `// Vendored static snapshot — do not edit manually.\n// Update by running: bun packages/emberharmony/script/update-models-snapshot.ts\n// sha256: ${hash}\nexport const snapshot = ${formatted} as const\n`,
)

console.log(`Updated models-snapshot.ts (sha256: ${hash})`)
console.log("Review the diff carefully before committing — check for unexpected new providers or changed URLs.")
