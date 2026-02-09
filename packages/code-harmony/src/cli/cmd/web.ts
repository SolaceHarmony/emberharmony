import { Server } from "../../server/server"
import { UI } from "../ui"
import { cmd } from "./cmd"
import { withNetworkOptions, resolveNetworkOptions } from "../network"
import { Flag } from "../../flag/flag"
import open from "open"
import { networkInterfaces } from "os"
import path from "path"

function getNetworkIPs() {
  const nets = networkInterfaces()
  const results: string[] = []

  for (const name of Object.keys(nets)) {
    const net = nets[name]
    if (!net) continue

    for (const netInfo of net) {
      // Skip internal and non-IPv4 addresses
      if (netInfo.internal || netInfo.family !== "IPv4") continue

      // Skip Docker bridge networks (typically 172.x.x.x)
      if (netInfo.address.startsWith("172.")) continue

      results.push(netInfo.address)
    }
  }

  return results
}

export const WebCommand = cmd({
  command: "web",
  builder: (yargs) => withNetworkOptions(yargs),
  describe: "start code-harmony server and open web interface",
  handler: async (args) => {
    if (!Flag.CODE_HARMONY_SERVER_PASSWORD) {
      UI.println(UI.Style.TEXT_WARNING_BOLD + "!  " + "CODE_HARMONY_SERVER_PASSWORD is not set; server is unsecured.")
    }
    const opts = await resolveNetworkOptions(args)
    const server = Server.listen(opts)
    UI.empty()
    UI.println(UI.logo("  "))
    UI.empty()

    const localhostApi = `http://127.0.0.1:${server.port}`
    if (opts.hostname === "0.0.0.0") {
      UI.println(UI.Style.TEXT_INFO_BOLD + "  Local access:      ", UI.Style.TEXT_NORMAL, localhostApi)
      const networkIPs = getNetworkIPs()
      if (networkIPs.length > 0) {
        for (const ip of networkIPs) {
          UI.println(UI.Style.TEXT_INFO_BOLD + "  Network access:    ", UI.Style.TEXT_NORMAL, `http://${ip}:${server.port}`)
        }
      }

      if (opts.mdns) {
        UI.println(UI.Style.TEXT_INFO_BOLD + "  mDNS:              ", UI.Style.TEXT_NORMAL, `code-harmony.local:${server.port}`)
      }
    }

    const mode = process.env.CODE_HARMONY_WEB_UI ?? "auto"
    const noopen = process.env.CODE_HARMONY_WEB_NO_OPEN === "1"

    const appdir = path.resolve(process.cwd(), "..", "app")
    const apppkg = path.join(appdir, "package.json")
    const hasapp = await Bun.file(apppkg).exists()
    const bunbin = Bun.which("bun")

    const useLocal = mode === "local" || (mode === "auto" && hasapp && !!bunbin)
    if (!useLocal) {
      const url = server.url.toString()
      UI.println(UI.Style.TEXT_INFO_BOLD + "  Web interface:     ", UI.Style.TEXT_NORMAL, url)
      if (!noopen) open(url).catch(() => {})
      await new Promise(() => {})
      await server.stop()
      return
    }

    const probe = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch() {
        return new Response("")
      },
    })
    const uiport = probe.port
    probe.stop(true)

    const url = `http://127.0.0.1:${uiport}/`
    UI.println(UI.Style.TEXT_INFO_BOLD + "  Web interface:     ", UI.Style.TEXT_NORMAL, url)

    const child = Bun.spawn({
      cmd: ["bun", "run", "dev", "--", "--host", "127.0.0.1", "--port", String(uiport), "--strictPort"],
      cwd: appdir,
      env: {
        ...process.env,
        VITE_CODE_HARMONY_SERVER_HOST: "127.0.0.1",
        VITE_CODE_HARMONY_SERVER_PORT: String(server.port),
      },
      stdio: ["ignore", "inherit", "inherit"],
    })

    // Give the UI dev server a moment to bind before opening the browser.
    for (const delay of [50, 100, 250, 500, 1000, 2000]) {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 250)
      const ok = await fetch(url, { signal: ctrl.signal })
        .then(() => true)
        .catch(() => false)
      clearTimeout(timer)
      if (ok) break
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
    if (!noopen) open(url).catch(() => {})

    const stop = async () => {
      child.kill()
      await server.stop(true)
      process.exit(0)
    }
    process.once("SIGINT", () => void stop())
    process.once("SIGTERM", () => void stop())

    await child.exited
    await server.stop(true)
  },
})
