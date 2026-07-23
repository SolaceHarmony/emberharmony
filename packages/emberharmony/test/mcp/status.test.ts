import { expect, test } from "bun:test"
import { mkdir } from "node:fs/promises"
import path from "node:path"
import { tmpdir } from "../fixture/fixture"

test("enabled MCPs report connecting and deduplicate concurrent connects", async () => {
  await using tmp = await tmpdir({
    init: async (dir) => {
      const starts = path.join(dir, "starts")
      await mkdir(starts)
      await Bun.write(
        path.join(dir, "emberharmony.json"),
        JSON.stringify({
          $schema: "https://solace.ofharmony.ai/config.json",
          mcp: {
            slow: {
              type: "local",
              command: [
                process.execPath,
                "-e",
                `await Bun.write(${JSON.stringify(starts)} + "/" + process.pid, ""); await Bun.sleep(250)`,
              ],
              timeout: 100,
            },
          },
        }),
      )
    },
  })

  const proc = Bun.spawn([process.execPath, path.join(import.meta.dir, "fixture/status-runner.ts"), tmp.path], {
    env: {
      ...process.env,
      EMBERHARMONY_TEST_HOME: path.join(tmp.path, "home"),
      EMBERHARMONY_DISABLE_MODELS_FETCH: "true",
      XDG_DATA_HOME: path.join(tmp.path, "share"),
      XDG_CACHE_HOME: path.join(tmp.path, "cache"),
      XDG_CONFIG_HOME: path.join(tmp.path, "config"),
      XDG_STATE_HOME: path.join(tmp.path, "state"),
    },
    stdout: "pipe",
    stderr: "pipe",
  })
  const result = await Promise.all([proc.exited, new Response(proc.stdout).text(), new Response(proc.stderr).text()])

  expect(result[0], result[2]).toBe(0)
  expect(result[1].trim()).toBe('{"before":"connecting","starts":1,"after":"failed"}')
})
