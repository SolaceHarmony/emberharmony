import { test, expect } from "bun:test"
import fs from "fs/promises"
import path from "path"
import { Storage } from "../../src/storage/storage"
import { Global } from "../../src/global"

const targetOf = (key: string[]) => path.join(Global.Path.data, "storage", ...key) + ".json"
const exists = (p: string) => fs.access(p).then(() => true, () => false)

test("writeBehind: read-your-writes before flush, then durable on disk via atomic rename", async () => {
  const key = ["test-wb", "ryow-" + Date.now()]
  const target = targetOf(key)
  await fs.unlink(target).catch(() => {})

  Storage.writeBehind(key, { hello: "world", n: 1 })
  // immediately readable from the overlay, before the background flush
  expect(await Storage.read<{ hello: string; n: number }>(key)).toEqual({ hello: "world", n: 1 })

  // eventually durable on disk
  for (let i = 0; i < 200 && !(await exists(target)); i++) await Bun.sleep(10)
  expect(JSON.parse(await fs.readFile(target, "utf8"))).toEqual({ hello: "world", n: 1 })
  await fs.unlink(target).catch(() => {})
})

test("a durable write supersedes an in-flight writeBehind (no stale clobber)", async () => {
  const key = ["test-wb", "supersede-" + Date.now()]
  const target = targetOf(key)
  await fs.unlink(target).catch(() => {})

  Storage.writeBehind(key, { v: "old" })
  await Storage.write(key, { v: "new" })
  await Bun.sleep(150) // give any stale flush a chance to (wrongly) run

  expect(JSON.parse(await fs.readFile(target, "utf8"))).toEqual({ v: "new" })
  expect(await Storage.read<{ v: string }>(key)).toEqual({ v: "new" })
  await fs.unlink(target).catch(() => {})
})

test("list() includes write-behind entries (read-after-create consistency)", async () => {
  const prefix = ["test-wb-list", "p-" + Date.now()]
  const key = [...prefix, "child"]
  const target = targetOf(key)
  await fs.unlink(target).catch(() => {})

  Storage.writeBehind(key, { ok: true })
  // immediately listable (from overlay if not yet flushed, else from disk) —
  // there is no gap because flush writes the file before clearing pending
  const keys = await Storage.list(prefix)
  expect(keys.some((k) => k.join("/") === key.join("/"))).toBe(true)

  for (let i = 0; i < 200 && !(await exists(target)); i++) await Bun.sleep(10)
  await fs.unlink(target).catch(() => {})
})

test("remove() of a write-behind key does not resurrect the file", async () => {
  const key = ["test-wb", "remove-" + Date.now()]
  const target = targetOf(key)
  await fs.unlink(target).catch(() => {})

  Storage.writeBehind(key, { v: 1 })
  await Storage.remove(key) // serialized with flush via Lock.write
  await Bun.sleep(250) // give any scheduled flush a chance to (wrongly) recreate it

  expect(await exists(target)).toBe(false)
  let threw = false
  try {
    await Storage.read(key)
  } catch {
    threw = true
  }
  expect(threw).toBe(true)
})

test("update sees a pending writeBehind value", async () => {
  const key = ["test-wb", "update-" + Date.now()]
  const target = targetOf(key)
  await fs.unlink(target).catch(() => {})

  Storage.writeBehind(key, { count: 1 })
  const updated = await Storage.update<{ count: number }>(key, (d) => {
    d.count += 1
  })
  expect(updated.count).toBe(2)

  await Bun.sleep(50)
  expect(JSON.parse(await fs.readFile(target, "utf8")).count).toBe(2)
  await fs.unlink(target).catch(() => {})
})
