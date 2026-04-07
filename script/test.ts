import path from "path"

const main = async () => {
  const root = path.resolve(import.meta.dir, "..")
  const args = process.argv.slice(2)
  const pkgs = await Array.fromAsync(new Bun.Glob("packages/**/package.json").scan({ cwd: root, onlyFiles: true }))

  const script = (json: unknown) => {
    if (!json) return undefined
    if (typeof json !== "object") return undefined
    if (!("scripts" in json)) return undefined

    const scripts = (json as Record<string, unknown>).scripts
    if (!scripts) return undefined
    if (typeof scripts !== "object") return undefined
    if (!("test" in scripts)) return undefined

    const test = (scripts as Record<string, unknown>).test
    if (typeof test !== "string") return undefined
    return test
  }

  const dirs = [] as string[]
  for (const file of pkgs) {
    const abs = path.join(root, file)
    const json = await Bun.file(abs).json().catch(() => undefined)
    const cmd = script(json)
    if (!cmd) continue
    if (!cmd.includes("bun test")) continue
    dirs.push(path.dirname(abs))
  }

  const uniq = [...new Set(dirs)].sort()
  if (uniq.length === 0) {
    console.log("No packages with a bun test script were found.")
    return
  }

  const failures = [] as string[]
  for (const dir of uniq) {
    const rel = path.relative(root, dir) || "."
    const cmd = ["bun", "test", ...args]
    console.log(`\n[${rel}] ${cmd.join(" ")}`)

    const proc = Bun.spawn(cmd, { cwd: dir, stdout: "inherit", stderr: "inherit" })
    const code = await proc.exited
    if (code === 0) continue
    failures.push(`${rel} (exit ${code})`)
  }

  if (failures.length === 0) return
  console.error("\nFailed:\n" + failures.join("\n"))
  process.exit(1)
}

await main()

