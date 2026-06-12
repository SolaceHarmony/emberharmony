import { Server } from "../../server/server"
import { cmd } from "./cmd"
import { withNetworkOptions, resolveNetworkOptions } from "../network"
import { Flag } from "../../flag/flag"
import { bootstrap } from "../bootstrap"
import { VoiceWorker } from "../../voice/worker"

export const ServeCommand = cmd({
  command: "serve",
  builder: (yargs) => withNetworkOptions(yargs),
  describe: "starts a headless emberharmony server",
  handler: async (args) => {
    if (!Flag.EMBERHARMONY_SERVER_PASSWORD) {
      console.log("Warning: EMBERHARMONY_SERVER_PASSWORD is not set; server is unsecured.")
    }
    const opts = await resolveNetworkOptions(args)
    const server = Server.listen(opts)
    console.log(`emberharmony server listening on http://${server.hostname}:${server.port}`)

    // the worker is a child process; it talks to this server over loopback
    const localhost = server.hostname === "0.0.0.0" || server.hostname === "::" ? "127.0.0.1" : server.hostname
    await bootstrap(process.cwd(), async () => {
      const started = await VoiceWorker.start(`http://${localhost}:${server.port}`)
      if (started) console.log("voice agent worker started")
    })

    // the idle promise below never resolves; without these handlers a signal
    // would kill the server and orphan the worker child process
    const shutdown = async () => {
      VoiceWorker.stop()
      await server.stop()
      process.exit(0)
    }
    process.on("SIGINT", shutdown)
    process.on("SIGTERM", shutdown)
    process.on("exit", () => VoiceWorker.stop())

    await new Promise(() => {})
  },
})
