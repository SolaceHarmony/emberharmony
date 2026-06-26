import { test, expect } from "bun:test"
import fs from "fs/promises"
import { Log } from "../../src/util/log"

// The test preload runs Log.init({ print: false, dev: true }), so log output
// goes to the dev.log file returned by Log.file(). Read that file back and pull
// the line carrying our unique marker.
async function logged(marker: string, fn: () => void): Promise<string> {
  fn()
  const path = Log.file()
  for (let i = 0; i < 50; i++) {
    const content = await fs.readFile(path, "utf8").catch(() => "")
    const line = content
      .split("\n")
      .reverse()
      .find((l) => l.includes(marker))
    if (line) return line
    await Bun.sleep(5)
  }
  throw new Error(`log line with marker ${marker} not found in ${path}`)
}

let n = 0
const freshLogger = () => Log.create({ service: `log-trunc-test-${n++}` })

test("caps an oversized object field and marks the elision", async () => {
  const huge = "x".repeat(60_000)
  const line = await logged("trunc-obj-marker", () => {
    freshLogger().error("trunc-obj-marker", { error: { requestBodyValues: huge, statusCode: 429 } })
  })
  expect(line.length).toBeLessThan(4096) // 60KB payload must not reach the line
  expect(line).toContain("…[+")
  expect(line).not.toContain(huge)
  expect(line).toContain("error=")
})

test("leaves small fields untouched", async () => {
  const line = await logged("trunc-small-marker", () => {
    freshLogger().info("trunc-small-marker", { statusCode: 429, model: "anthropic/claude-haiku-4.5" })
  })
  expect(line).toContain("statusCode=429")
  expect(line).toContain("model=anthropic/claude-haiku-4.5")
  expect(line).not.toContain("…[+")
})

test("caps a giant primitive string field too", async () => {
  const huge = "y".repeat(50_000)
  const line = await logged("trunc-prim-marker", () => {
    freshLogger().warn("trunc-prim-marker", { blob: huge })
  })
  expect(line.length).toBeLessThan(4096)
  expect(line).toContain("…[+")
})

test("renders circular objects without collapsing to [object Object]", async () => {
  const circular: any = { a: 1 }
  circular.self = circular
  const line = await logged("trunc-circular-marker", () => {
    freshLogger().info("trunc-circular-marker", { data: circular })
  })
  expect(line).not.toContain("[object Object]")
  expect(line).toContain("[Circular]")
  expect(line).toContain('"a":1')
})

test("renders BigInt fields instead of throwing or collapsing", async () => {
  const line = await logged("trunc-bigint-marker", () => {
    freshLogger().info("trunc-bigint-marker", { data: { tokens: 10n } })
  })
  expect(line).not.toContain("[object Object]")
  expect(line).toContain("10n")
})
