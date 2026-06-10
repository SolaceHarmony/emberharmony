import { BusEvent } from "@/bus/bus-event"
import os from "os"
import path from "path"
import { $ } from "bun"
import z from "zod"
import { NamedError } from "@thesolaceproject/emberharmony-util/error"
import { Log } from "../util/log"
import { iife } from "@/util/iife"
import { Flag } from "../flag/flag"

declare global {
  const EMBERHARMONY_VERSION: string
  const EMBERHARMONY_CHANNEL: string
  const EMBERHARMONY_TAG: string
}

export namespace Installation {
  const log = Log.create({ service: "installation" })

  export type Method = Awaited<ReturnType<typeof method>>

  export const Event = {
    Updated: BusEvent.define(
      "installation.updated",
      z.object({
        version: z.string(),
      }),
    ),
    UpdateAvailable: BusEvent.define(
      "installation.update-available",
      z.object({
        version: z.string(),
      }),
    ),
  }

  export const Info = z
    .object({
      version: z.string(),
      latest: z.string(),
    })
    .meta({
      ref: "InstallationInfo",
    })
  export type Info = z.infer<typeof Info>

  export async function info() {
    return {
      version: VERSION,
      latest: await latest(),
    }
  }

  export function isPreview() {
    return CHANNEL !== "latest"
  }

  export function isLocal() {
    return CHANNEL === "local"
  }

  const npmName = "@thesolaceproject/emberharmony"

  export async function method() {
    if (process.execPath.includes(path.join(".emberharmony", "bin"))) return "curl"
    if (process.execPath.includes(path.join(".local", "bin"))) return "curl"
    const exec = process.execPath.toLowerCase()

    const checks = [
      {
        name: "npm" as const,
        command: () => $`npm list -g --depth=0`.throws(false).quiet().text(),
      },
      {
        name: "yarn" as const,
        command: () => $`yarn global list`.throws(false).quiet().text(),
      },
      {
        name: "pnpm" as const,
        command: () => $`pnpm list -g --depth=0`.throws(false).quiet().text(),
      },
      {
        name: "bun" as const,
        command: () => $`bun pm ls -g`.throws(false).quiet().text(),
      },
    ]

    checks.sort((a, b) => {
      const aMatches = exec.includes(a.name)
      const bMatches = exec.includes(b.name)
      if (aMatches && !bMatches) return -1
      if (!aMatches && bMatches) return 1
      return 0
    })

    const names = [npmName, "@thesolaceproject/emberharmony", "emberharmony"]

    for (const check of checks) {
      const output = await check.command()
      if (names.some((name) => output.includes(name))) {
        return check.name
      }
    }

    return "unknown"
  }

  export const UpgradeFailedError = NamedError.create(
    "UpgradeFailedError",
    z.object({
      stderr: z.string(),
    }),
  )

  export async function upgrade(method: Method, target: string) {
    let cmd
    switch (method) {
      case "curl": {
        // The tag is interpolated into a URL whose response gets executed,
        // and URL clients normalize ../ segments — a crafted tag could
        // otherwise address an arbitrary repository. Enforce the same
        // release-tag character allowlist as build.ts and the install script.
        if (!/^[A-Za-z0-9._-]+$/.test(target)) {
          throw new Error(`invalid release tag "${target}"`)
        }
        // The installer takes the release TAG verbatim — tags name releases;
        // versions live in version.json and never identify a release. The
        // installer script is fetched at the target tag's ref so the install
        // is reproducible, and fetched explicitly rather than `curl | bash`:
        // in a pipe, bash exits 0 on empty input when curl fails, silently
        // no-op'ing the upgrade while reporting success.
        const res = await fetch(`https://raw.githubusercontent.com/SolaceHarmony/emberharmony/${target}/install`, {
          headers: { "User-Agent": USER_AGENT },
        })
        if (!res.ok) {
          throw new Error(`failed to fetch installer for ${target}: ${res.status} ${res.statusText}`)
        }
        const installer = path.join(os.tmpdir(), `emberharmony-install-${target}.sh`)
        await Bun.write(installer, await res.text())
        cmd = $`bash ${installer}`.env({
          ...process.env,
          TAG: target,
        })
        break
      }
      case "npm":
        cmd = $`npm install -g ${npmName}@${target}`
        break
      case "pnpm":
        cmd = $`pnpm install -g ${npmName}@${target}`
        break
      case "bun":
        cmd = $`bun install -g ${npmName}@${target}`
        break
      case "yarn":
        cmd = $`yarn global add ${npmName}@${target}`
        break
      default:
        throw new Error(`Unknown method: ${method}`)
    }
    const result = await cmd.quiet().throws(false)
    if (result.exitCode !== 0) {
      throw new UpgradeFailedError({
        stderr: result.stderr.toString("utf8"),
      })
    }
    log.info("upgraded", {
      method,
      target,
      stdout: result.stdout.toString(),
      stderr: result.stderr.toString(),
    })
    await $`${process.execPath} --version`.nothrow().quiet().text()
  }

  export const VERSION = typeof EMBERHARMONY_VERSION === "string" ? EMBERHARMONY_VERSION : "local"
  export const CHANNEL = typeof EMBERHARMONY_CHANNEL === "string" ? EMBERHARMONY_CHANNEL : "local"
  // The GitHub release tag this binary was built for ("" for npm/local builds).
  // GitHub-released binaries are identified by their tag, not VERSION: the tag
  // is human-chosen and intentionally unrelated to version.json.
  export const TAG = typeof EMBERHARMONY_TAG === "string" ? EMBERHARMONY_TAG : ""
  export const USER_AGENT = `emberharmony/${CHANNEL}/${VERSION}/${Flag.EMBERHARMONY_CLIENT}`

  // What this install reports as its current release identity, for comparison
  // against latest(): npm-managed installs are versioned by the npm package
  // version; GitHub (curl) installs by the release tag embedded at build time.
  export function installed(installMethod: Method) {
    if (installMethod === "npm" || installMethod === "pnpm" || installMethod === "bun" || installMethod === "yarn") {
      return VERSION
    }
    return TAG || VERSION
  }

  export async function latest(installMethod?: Method) {
    const detectedMethod = installMethod || (await method())

    if (detectedMethod === "npm" || detectedMethod === "bun" || detectedMethod === "pnpm" || detectedMethod === "yarn") {
      const registry = await iife(async () => {
        const r = (await $`npm config get registry`.quiet().nothrow().text()).trim()
        const reg = r || "https://registry.npmjs.org"
        return reg.endsWith("/") ? reg.slice(0, -1) : reg
      })
      const channel = CHANNEL
      const name = encodeURIComponent(npmName)
      return fetch(`${registry}/${name}/${channel}`)
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText)
          return res.json()
        })
        .then((data: any) => data.version)
    }

    // GitHub-released binaries follow release tags verbatim. Dev-channel
    // builds never appear in /releases/latest (dev-target releases are always
    // prereleases), so they follow the newest prerelease targeting dev.
    if (CHANNEL === "dev") {
      return fetch("https://api.github.com/repos/SolaceHarmony/emberharmony/releases?per_page=30", {
        headers: { "User-Agent": USER_AGENT },
      })
        .then((res) => {
          if (!res.ok) throw new Error(res.statusText)
          return res.json()
        })
        .then((data: any[]) => {
          const release = data.find((item) => item.prerelease && !item.draft && item.target_commitish === "dev")
          if (!release) throw new Error("no dev-target prerelease found on GitHub")
          return release.tag_name as string
        })
    }

    return fetch("https://api.github.com/repos/SolaceHarmony/emberharmony/releases/latest", {
      headers: { "User-Agent": USER_AGENT },
    })
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText)
        return res.json()
      })
      .then((data: any) => data.tag_name as string)
  }
}
