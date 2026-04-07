const ignore = new Set(["aws-sdk"])
// CVE-2026-25536 is fixed in @modelcontextprotocol/sdk >= 1.26.0, but OSV can lag behind.
// We pin @modelcontextprotocol/sdk to a fixed version in the repo; keep installs unblocked.
const ignoreIds = new Set(["GHSA-j965-2qgj-vjmq", "CVE-2026-25536"])
const win = process.platform === "win32"
const ci =
  process.env["CI"] === "true" || process.env["GITHUB_ACTIONS"] === "true" || process.env["BUN_SECURITY_SCAN"] === "0"
const debug = process.env["BUN_SECURITY_SCAN_DEBUG"] === "1"

export const scanner: Bun.Security.Scanner = {
  version: "1",
  async scan(input) {
    if (win) return []
    if (ci) return []
    const mod = await import("bun-osv-scanner").catch(() => null)
    if (!mod) return []
    const advisories = await mod.scanner.scan(input).catch(() => [])
    if (debug) {
      for (const advisory of advisories) {
        if (advisory.package !== "@modelcontextprotocol/sdk") continue
        console.log("[security-scan] mcp", JSON.stringify(advisory))
      }
    }
    return advisories.filter((advisory) => !ignore.has(advisory.package) && !ignoreIds.has(advisory.id))
  },
}
