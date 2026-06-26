const ignore = new Set([
  "aws-sdk",
  // Transitive serveStatic middleware bypasses — low severity, no upstream fix.
  // @hono/node-server via @modelcontextprotocol/sdk; srvx via nitro.
  "@hono/node-server", "srvx",
])
// CVE-2026-25536 is fixed in @modelcontextprotocol/sdk >= 1.26.0, but OSV can lag behind.
// We pin @modelcontextprotocol/sdk to a fixed version in the repo; keep installs unblocked.
const ignoreIds = new Set([
  "GHSA-j965-2qgj-vjmq", "CVE-2026-25536",
  // undici WebSocket CVEs — transitive via @actions/github, which uses undici
  // for HTTP (the GitHub API), not its WebSocket client, so these WS-only DoS
  // issues aren't reachable. undici is also pinned to a fixed release (see
  // overrides); these older GHSA advisories stay suppressed in case OSV lags.
  "GHSA-2mjp-6q6p-2qxm", "GHSA-f269-vfmq-vjvj", "GHSA-vrm6-8vpv-qv8q",
  "GHSA-4992-7rv2-5pvq", "GHSA-phc3-fgpg-7m6h", "GHSA-v9p9-hfj2-hcw8",
  // uuid v3/v5/v6 buffer bounds check — transitive via @actions/artifact › @azure/core-http (CI-only).
  // @azure/core-http is deprecated and pins uuid ^8; the fix only exists in uuid >= 11.1.1.
  // Not applicable: core-http only calls uuid.v4() without a buf argument.
  "GHSA-w5hq-g745-h8pq", "CVE-2026-41907",
  // @ai-sdk/provider-utils uncontrolled resource consumption — no fixed release as of 2026-06-12.
  // Remove this suppression once a patched @ai-sdk/provider-utils ships and the pin is bumped.
  "GHSA-866g-f22w-33x8", "CVE-2026-8769",
])
// postcss CVE-2026-41305 / GHSA-qx2v-qp2m-jg93 — fixed in postcss >= 8.5.10.
// OSV's affected-range data is stale and flags 8.5.x as affected. We gate the
// suppression on every installed postcss being >= 8.5.10: if the root override
// (currently pinning all resolutions to 8.5.14) were ever removed and a pre-8.5.10
// copy snuck back in via tw-to-css or another transitive dep, this would surface it.
const postcssIgnoreIds = new Set(["CVE-2026-41305", "GHSA-qx2v-qp2m-jg93"])
const win = process.platform === "win32"
const ci =
  process.env["CI"] === "true" || process.env["GITHUB_ACTIONS"] === "true" || process.env["BUN_SECURITY_SCAN"] === "0"
// A non-interactive install can't answer bun's "Continue anyway? [y/N]" warning
// prompt, so a piped/scripted/Docker run (or any CI that doesn't set the flags
// above) would HANG on it or auto-abort. Treat "no TTY" as non-interactive and
// skip the scan — matching the CI-skip behavior. A real terminal (both stdin
// and stdout are TTYs) still gets scanned and prompted.
const interactive = Boolean(process.stdin.isTTY) && Boolean(process.stdout.isTTY)
const debug = process.env["BUN_SECURITY_SCAN_DEBUG"] === "1"

export const scanner: Bun.Security.Scanner = {
  version: "1",
  async scan(input) {
    if (win) return []
    if (ci) return []
    if (!interactive) return []
    const mod = await import("bun-osv-scanner").catch(() => null)
    if (!mod) return []
    const advisories = await mod.scanner.scan(input).catch(() => [])
    if (debug) {
      for (const advisory of advisories) {
        if (advisory.package !== "@modelcontextprotocol/sdk") continue
        console.log("[security-scan] mcp", JSON.stringify(advisory))
      }
    }
    return advisories.filter((advisory) => {
      if (ignore.has(advisory.package)) return false
      if (ignoreIds.has(advisory.id)) return false
      if (postcssIgnoreIds.has(advisory.id) && advisory.package === "postcss") {
        // Only suppress when every installed postcss is >= 8.5.10.
        // If the root override is removed and a vulnerable copy sneaks in, this surfaces it.
        const hasVulnerable = input.packages.some((pkg) => pkg.name === "postcss" && !atLeast(pkg.version, "8.5.10"))
        return hasVulnerable
      }
      return true
    })
  },
}

function atLeast(version: string, min: string) {
  const v = version.split(".").map(Number)
  const m = min.split(".").map(Number)
  for (let i = 0; i < Math.max(v.length, m.length); i++) {
    const a = v[i] ?? 0
    const b = m[i] ?? 0
    if (a > b) return true
    if (a < b) return false
  }
  return true
}
