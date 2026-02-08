#!/usr/bin/env bun
import { mkdir, cp, stat } from "node:fs/promises"
import { dirname, join } from "node:path"

function redact(value: string) {
  const clean = value.trim()
  if (clean.length <= 12) return "[redacted]"
  return `${clean.slice(0, 6)}…${clean.slice(-4)}`
}

function lines(text: string, index: number) {
  const head = text.slice(0, index)
  const n = head.split("\n").length
  const last = head.lastIndexOf("\n")
  const col = index - (last === -1 ? 0 : last + 1) + 1
  return { line: n, col }
}

function patterns() {
  const list = [
    { name: "GitHub PAT", re: /\bgithub_pat_[A-Za-z0-9_]{10,}\b/g },
    { name: "Anthropic API key", re: /\bsk-ant-[A-Za-z0-9_-]{10,}\b/g },
    { name: "Stripe publishable key", re: /\bpk_(?:live|test)_[A-Za-z0-9]{10,}\b/g },
    { name: "Stripe secret key", re: /\bsk_(?:live|test)_[A-Za-z0-9]{10,}\b/g },
    { name: "Cerebras API key", re: /\bcsk-[A-Za-z0-9]{10,}\b/g },
    { name: "NPM token", re: /\bnpm_[A-Za-z0-9]{10,}\b/g },
    { name: "OpenCode-style key", re: /\boc_(?:live_sk|test_sk|live|test|prod|dev)_[A-Za-z0-9]{10,}\b/g },
    // Nix SRI hashes are not secrets, but scanners sometimes flag them.
    { name: "Nix SRI hash", re: /\bsha256-[A-Za-z0-9+/]{10,}={0,2}\b/g },
  ] as const
  return list
}

async function allowlist(root: string) {
  const file = join(root, "security/allowlist/manifest.json")
  const text = await Bun.file(file).text().catch(() => "")
  if (!text) return []

  const data = JSON.parse(text) as { ignore?: Array<{ file?: string; kind?: string }> }
  const list = data.ignore ?? []
  return list
    .map((x) => ({
      file: (x.file ?? "").replaceAll("\\", "/"),
      kind: x.kind ?? "",
    }))
    .filter((x) => x.file && x.kind)
}

async function main() {
  const root = process.cwd()
  const ts = new Date().toISOString().replace(/[:.]/g, "-")
  const base = (Bun.env.CODE_HARMONY_QUARANTINE_DIR ?? "").trim()
  const home = (Bun.env.HOME ?? "").trim()
  const dir = base
    ? base
    : home
      ? join(home, "Documents", "code-harmony-quarantine")
      : join(root, ".quarantine")
  const out = join(dir, ts)
  await mkdir(out, { recursive: true })
  const allow = await allowlist(root)

  const skip = [
    ".git/",
    ".quarantine/",
    "security/allowlist/",
    ".git.old-",
    "node_modules/",
    "dist/",
    "target/",
    ".next/",
    ".turbo/",
    ".cache/",
  ]

  const glob = new Bun.Glob("**/*")
  const hits: Array<{
    file: string
    kind: string
    line: number
    col: number
    sample: string
  }> = []

  const files: string[] = []
  for await (const rel of glob.scan({ cwd: root, onlyFiles: true, dot: true })) {
    const p = rel.replaceAll("\\", "/")
    if (skip.some((s) => p.startsWith(s) || p.includes(`/${s}`))) continue
    files.push(rel)
  }

  for (const rel of files) {
    const p = join(root, rel)
    const info = await stat(p).catch(() => null)
    if (!info) continue
    if (!info.isFile()) continue
    if (info.size > 2_000_000) continue

    const text = await Bun.file(p).text().catch(() => "")
    if (!text) continue

    const found = patterns()
      .flatMap((pat) => {
        const items: Array<{ kind: string; index: number; value: string }> = []
        const re = new RegExp(pat.re.source, pat.re.flags)
        for (const m of text.matchAll(re)) {
          if (typeof m.index !== "number") continue
          items.push({ kind: pat.name, index: m.index, value: m[0] })
        }
        return items
      })
      .filter((m) => {
        const file = rel.replaceAll("\\", "/")
        return !allow.some((a) => a.file === file && a.kind === m.kind)
      })
      .slice(0, 50)

    if (!found.length) continue

    const dest = join(out, rel)
    await mkdir(dirname(dest), { recursive: true })
    await cp(p, dest)

    for (const m of found) {
      const pos = lines(text, m.index)
      hits.push({
        file: rel.replaceAll("\\", "/"),
        kind: m.kind,
        line: pos.line,
        col: pos.col,
        sample: redact(m.value),
      })
    }
  }

  const report = join(out, "report.json")
  await Bun.write(report, JSON.stringify({ root, out, hits }, null, 2) + "\n")

  const text = [
    `Quarantine: ${out}`,
    `Matches: ${hits.length}`,
    "",
    ...hits.map((h) => `${h.file}:${h.line}:${h.col}  ${h.kind}  ${h.sample}`),
    "",
    `JSON: ${report}`,
    "",
  ].join("\n")
  await Bun.write(join(out, "report.txt"), text)

  process.stdout.write(text)
}

main()
