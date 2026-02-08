const args = process.argv.slice(2)

const opts = {
  url: process.env.OPENCODE_SERVER_URL ?? "http://localhost:4096",
  id: "",
  cmd: "",
  title: "",
  cwd: "",
  list: false,
  help: false,
  args: [] as string[],
  env: {} as Record<string, string>,
}

const queue = [...args]
while (queue.length) {
  const item = queue.shift()
  if (!item) break

  if (item === "--help" || item === "-h") {
    opts.help = true
    continue
  }
  if (item === "--list") {
    opts.list = true
    continue
  }
  if (item === "--id") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --id")
      process.exit(1)
    }
    opts.id = value
    continue
  }
  if (item.startsWith("--id=")) {
    opts.id = item.slice(5)
    continue
  }
  if (item === "--url") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --url")
      process.exit(1)
    }
    opts.url = value
    continue
  }
  if (item.startsWith("--url=")) {
    opts.url = item.slice(6)
    continue
  }
  if (item === "--cmd" || item === "--command") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --cmd")
      process.exit(1)
    }
    opts.cmd = value
    continue
  }
  if (item.startsWith("--cmd=")) {
    opts.cmd = item.slice(6)
    continue
  }
  if (item === "--cwd") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --cwd")
      process.exit(1)
    }
    opts.cwd = value
    continue
  }
  if (item.startsWith("--cwd=")) {
    opts.cwd = item.slice(6)
    continue
  }
  if (item === "--title") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --title")
      process.exit(1)
    }
    opts.title = value
    continue
  }
  if (item.startsWith("--title=")) {
    opts.title = item.slice(8)
    continue
  }
  if (item === "--env") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --env")
      process.exit(1)
    }
    const index = value.indexOf("=")
    if (index === -1) {
      console.error("invalid --env value, expected KEY=VALUE")
      process.exit(1)
    }
    opts.env[value.slice(0, index)] = value.slice(index + 1)
    continue
  }
  if (item.startsWith("--env=")) {
    const value = item.slice(6)
    const index = value.indexOf("=")
    if (index === -1) {
      console.error("invalid --env value, expected KEY=VALUE")
      process.exit(1)
    }
    opts.env[value.slice(0, index)] = value.slice(index + 1)
    continue
  }
  if (item === "--arg") {
    const value = queue.shift()
    if (!value) {
      console.error("missing value for --arg")
      process.exit(1)
    }
    opts.args.push(value)
    continue
  }
  if (item === "--") {
    opts.args.push(...queue)
    break
  }
  if (item.startsWith("--")) {
    console.error(`unknown flag: ${item}`)
    process.exit(1)
  }
  if (!opts.cmd) {
    opts.cmd = item
    continue
  }
  opts.args.push(item)
}

if (opts.help) {
  console.log(`pty wrapper

Usage:
  bun script/pty-wrapper.ts --list
  bun script/pty-wrapper.ts --id <ptyID>
  bun script/pty-wrapper.ts [command] [--arg <arg>...] [--cwd <dir>] [--title <name>]

Options:
  --url <url>      Base URL (default: http://localhost:4096)
  --id <ptyID>     Connect to an existing PTY session
  --list           List active PTY sessions
  --cmd <command>  Command to spawn (default: server picks)
  --arg <arg>      Extra command arg (repeatable)
  --cwd <dir>      Working directory
  --title <name>   PTY title
  --env KEY=VALUE  Add env var (repeatable)
`)
  process.exit(0)
}

const base = opts.url.endsWith("/") ? opts.url.slice(0, -1) : opts.url
const auth = (() => {
  const pass = process.env.OPENCODE_SERVER_PASSWORD
  if (!pass) return undefined
  const user = process.env.OPENCODE_SERVER_USERNAME ?? "opencode"
  return `Basic ${btoa(`${user}:${pass}`)}`
})()

const headers: Record<string, string> = auth ? { Authorization: auth } : {}

const request = async (path: string, init: RequestInit) => {
  const res = await fetch(base + path, init).catch(() => undefined)
  if (!res) {
    console.error("request failed")
    process.exit(1)
  }
  if (!res.ok) {
    const text = await res.text()
    console.error(`${res.status} ${res.statusText}`)
    if (text) console.error(text)
    process.exit(1)
  }
  return res
}

const list = async () => {
  const res = await request("/pty", { method: "GET", headers })
  const data = await res.json()
  console.log(JSON.stringify(data, null, 2))
}

const create = async () => {
  const body: Record<string, unknown> = {}
  if (opts.cmd) body.command = opts.cmd
  if (opts.args.length) body.args = opts.args
  if (opts.cwd) body.cwd = opts.cwd
  if (opts.title) body.title = opts.title
  if (Object.keys(opts.env).length) body.env = opts.env
  const res = await request("/pty", {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  return data
}

const update = async (id: string, cols?: number, rows?: number) => {
  if (!cols || !rows) return
  await request(`/pty/${id}`, {
    method: "PUT",
    headers: {
      ...headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      size: { cols, rows },
    }),
  })
}

const ws = (url: string) => {
  if (url.startsWith("https://")) return "wss://" + url.slice(8)
  if (url.startsWith("http://")) return "ws://" + url.slice(7)
  return "ws://" + url
}

const decode = (value: unknown) => {
  if (typeof value === "string") return value
  if (value instanceof ArrayBuffer) return new TextDecoder().decode(new Uint8Array(value))
  if (value instanceof Uint8Array) return new TextDecoder().decode(value)
  return String(value)
}

const connect = async (id: string) => {
  await update(id, process.stdout.columns, process.stdout.rows)

  const socket = new WebSocket(`${ws(base)}/pty/${id}/connect`, {
    headers,
  })

  const cleanup = () => {
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false)
    }
    process.stdin.pause()
  }

  socket.addEventListener("close", () => {
    cleanup()
    process.exit(0)
  })

  socket.addEventListener("error", (event) => {
    console.error("socket error", event)
    cleanup()
    process.exit(1)
  })

  socket.addEventListener("message", (event) => {
    const text = decode(event.data)
    process.stdout.write(text)
  })

  socket.addEventListener("open", () => {
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(true)
    }
    process.stdin.resume()
    process.stdin.on("data", (chunk) => {
      if (socket.readyState !== WebSocket.OPEN) return
      socket.send(chunk.toString())
    })
  })

  process.stdout.on("resize", () => {
    update(id, process.stdout.columns, process.stdout.rows)
  })

  process.on("SIGINT", () => {
    cleanup()
    socket.close()
  })
}

const run = async () => {
  if (opts.list) {
    await list()
    return
  }

  const info = opts.id ? { id: opts.id } : await create()
  const id = info.id
  if (!id) {
    console.error("missing PTY id")
    process.exit(1)
  }
  console.log(`connected: ${id}`)
  await connect(id)
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
})
