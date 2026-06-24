import { createServer } from "node:net"
import { unlinkSync } from "node:fs"

const socketPath = process.env["EMBERHARMONY_VOICE_IPC_SOCKET"]
if (!socketPath) {
  console.error("EMBERHARMONY_VOICE_IPC_SOCKET not set")
  process.exit(1)
}

const server = createServer((conn) => {
  conn.on("data", (data) => {
    if (data.toString().trim() === "shutdown") {
      conn.write("ok")
      conn.end()
      server.close()
      unlinkSync(socketPath)
      process.exit(0)
    }
  })
  conn.on("error", () => {})
})

server.listen(socketPath, () => {
  console.log("fake voice agent ready on " + socketPath)
})
