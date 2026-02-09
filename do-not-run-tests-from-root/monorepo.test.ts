import { describe, expect, test } from "bun:test"
import path from "path"

describe("monorepo", () => {
  test("runs bun unit tests in each package as separate bun processes", async () => {
    const root = path.resolve(import.meta.dir, "..")
    const pkgs = await Array.fromAsync(new Bun.Glob("packages/**/package.json").scan({ cwd: root, onlyFiles: true }))
    const dirs = [] as string[]

    const cmd = (json: unknown) => {
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

    for (const file of pkgs) {
      const abs = path.join(root, file)
      const json = await Bun.file(abs).json().catch(() => undefined)
      const test = cmd(json)
      if (!test) continue
      if (!test.includes("bun test")) continue
      dirs.push(path.dirname(abs))
    }

    const uniq = [...new Set(dirs)].sort()
    expect(uniq.length).toBeGreaterThan(0)

    const failures = [] as string[]
    for (const dir of uniq) {
      const args = ["bun", "test"]
      const proc = Bun.spawn(args, { cwd: dir, stdout: "pipe", stderr: "pipe" })
      const stdout = await new Response(proc.stdout).text()
      const stderr = await new Response(proc.stderr).text()
      const code = await proc.exited
      if (code === 0) continue
      failures.push(
        [
          `cwd: ${dir}`,
          `cmd: ${args.join(" ")}`,
          `exit: ${code}`,
          stdout.trim() ? `stdout:\n${stdout.trim()}` : "",
          stderr.trim() ? `stderr:\n${stderr.trim()}` : "",
        ]
          .filter((x) => x !== "")
          .join("\n"),
      )
    }

    expect(failures.join("\n\n")).toBe("")
  }, 120_000)
})
