const ignore = new Set(["aws-sdk"])
const ignoreIds = new Set(["GHSA-j965-2qgj-vjmq"])
const win = process.platform === "win32"
const ci = process.env["CI"] === "true" || process.env["GITHUB_ACTIONS"] === "true" || process.env["BUN_SECURITY_SCAN"] === "0"

export const scanner: Bun.Security.Scanner = {
  version: "1",
  async scan(input) {
    if (win) return []
    if (ci) return []
    const mod = await import("bun-osv-scanner").catch(() => null)
    if (!mod) return []
    const advisories = await mod.scanner.scan(input).catch(() => [])
    return advisories.filter((advisory) => !ignore.has(advisory.package) && !ignoreIds.has(advisory.id))
  },
}
