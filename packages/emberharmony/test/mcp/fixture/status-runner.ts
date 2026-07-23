import { readdir } from "node:fs/promises"
import path from "node:path"
import { MCP } from "../../../src/mcp"
import { Instance } from "../../../src/project/instance"

const dir = process.argv[2]
if (!dir) throw new Error("missing fixture directory")

await Instance.provide({
  directory: dir,
  fn: async () => {
    const before = (await MCP.status()).slow.status
    await Promise.all([MCP.connect("slow"), MCP.connect("slow")])
    const starts = (await readdir(path.join(dir, "starts"))).length
    const after = (await MCP.status()).slow.status
    console.log(JSON.stringify({ before, starts, after }))
    await Instance.dispose()
  },
})
process.exit(0)
