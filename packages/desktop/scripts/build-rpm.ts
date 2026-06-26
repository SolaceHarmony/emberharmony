#!/usr/bin/env bun
/**
 * Build the Linux `.rpm` with the native, streaming `rpmbuild` instead of
 * Tauri's in-memory `rpm` crate.
 *
 * Tauri's bundler (the `rpm-rs` crate) reads the ENTIRE payload into RAM, builds
 * the cpio + compresses it in memory, and writes the package at the end. On a
 * ~950 MB / 10.6k-file bundle that means ~1.6 GB RSS and 55+ minutes (it
 * swap-thrashes a small VM and was the cause of CI's x86_64 ~58-min rpm hang).
 *
 * The `.deb` bundler already shells out to `dpkg-deb`, which STREAMS files off
 * disk — that's why the deb is fast and cheap. This script does the same for
 * rpm: it reuses the deb's already-staged buildroot (`bundle/deb/<pkg>/data`)
 * and runs `rpmbuild` against it, streaming to a zstd payload on disk. Measured
 * ~37 s for the same bundle the crate couldn't finish in an hour.
 *
 * Requirements: Linux host with `rpmbuild` (Fedora: `rpm-build`; Debian/Ubuntu:
 * `rpm`). Must run AFTER `tauri build --bundles deb` has produced the staging.
 *
 * Usage:  bun ./scripts/build-rpm.ts [bundleDir]
 *   bundleDir defaults to ../src-tauri/target/release/bundle
 */
import { $ } from "bun"
import { existsSync } from "node:fs"
import { mkdir, mkdtemp, readdir, rm, stat, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

if (process.platform !== "linux") {
  console.log("[rpm] not Linux — skipping (rpm is a Linux-only bundle)")
  process.exit(0)
}
if ((await $`which rpmbuild`.nothrow().quiet()).exitCode !== 0) {
  throw new Error("[rpm] rpmbuild not found — install it (Fedora: `dnf install rpm-build`, Debian/Ubuntu: `apt install rpm`)")
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const bundleDir = path.resolve(process.argv[2] ?? path.join(scriptDir, "..", "src-tauri/target/release/bundle"))
const debDir = path.join(bundleDir, "deb")
const outDir = path.join(bundleDir, "rpm")

// --- locate the deb staging (the dir containing a `data/` install tree) -------
const debStaging = existsSync(debDir)
  ? (await readdir(debDir, { withFileTypes: true }))
      .filter((e) => e.isDirectory())
      .map((e) => path.join(debDir, e.name))
      .find((d) => existsSync(path.join(d, "data")) && existsSync(path.join(d, "control", "control")))
  : undefined
if (!debStaging) {
  throw new Error(`[rpm] no deb staging found under ${debDir} — run \`tauri build --bundles deb\` first`)
}
const buildroot = path.join(debStaging, "data")

// --- metadata from the deb control (adapts to dev/prod automatically) ---------
const control = await Bun.file(path.join(debStaging, "control", "control")).text()
const field = (k: string) => control.match(new RegExp(`^${k}:\\s*(.+)$`, "m"))?.[1]?.trim()
const name = field("Package") ?? "emberharmony"
const version = field("Version") ?? "0.0.0"
const summary = field("Description")?.split("\n")[0] ?? "EmberHarmony"
// deb arch -> rpm arch
const rpmArch = ({ arm64: "aarch64", amd64: "x86_64" } as Record<string, string>)[field("Architecture") ?? ""] ?? "noarch"
// The app's runtime libraries, in rpm-world package names (the install target is
// always an rpm distro regardless of where we build).
const requires = ["webkit2gtk4.1", "gtk3"]

// --- generate the %files list -------------------------------------------------
// Own the app's own paths only — never claim shared system dirs (/usr/bin,
// /usr/share/icons/hicolor, …). The app tree under /usr/lib/<AppDir> is owned
// recursively by one entry; the binary/desktop/icon files are listed
// individually. Paths with spaces ("EmberHarmony Dev") are quoted.
const q = (p: string) => `"${p}"`
const files: string[] = []
for (const d of await readdir(path.join(buildroot, "usr/lib"), { withFileTypes: true }).catch(() => []))
  if (d.isDirectory()) files.push(q(`/usr/lib/${d.name}`))
for (const sub of ["usr/bin", "usr/share/applications"])
  for (const e of await readdir(path.join(buildroot, sub), { withFileTypes: true }).catch(() => []))
    if (!e.isDirectory()) files.push(q(`/${sub}/${e.name}`))
// icon files live under shared hicolor dirs — list the files, not the dirs
const iconRoot = path.join(buildroot, "usr/share/icons")
if (existsSync(iconRoot)) {
  const icons = (await $`find ${iconRoot} -type f`.text()).trim().split("\n").filter(Boolean)
  for (const f of icons) files.push(q(f.slice(buildroot.length)))
}
if (files.length === 0) throw new Error(`[rpm] no installable files found under ${buildroot}`)

// --- spec: disable the slow passes that scan all 10.6k files ------------------
//   __os_install_post nil  -> skip brp-* (strip, build-id, mangle) post-processing
//   __requires/provides_exclude .*  -> skip automatic ELF dependency generation
//   _binary_payload w3.zstdio       -> stream-compress the payload with zstd-3
const work = await mkdtemp(path.join(os.tmpdir(), "emberharmony-rpm-"))
try {
  const fileListPath = path.join(work, "files.list")
  await writeFile(fileListPath, files.join("\n") + "\n")
  const specPath = path.join(work, `${name}.spec`)
  await writeFile(
    specPath,
    [
      "%global _build_id_links none",
      "%global __os_install_post %{nil}",
      "%global __requires_exclude .*",
      "%global __provides_exclude .*",
      "%define _binary_payload w3.zstdio",
      `Name: ${name}`,
      `Version: ${version}`,
      "Release: 1",
      `Summary: ${summary}`,
      "License: MIT",
      `BuildArch: ${rpmArch}`,
      `Requires: ${requires.join(", ")}`,
      "%description",
      summary,
      `%files -f ${fileListPath}`,
      "%changelog",
      "",
    ].join("\n"),
  )

  await mkdir(outDir, { recursive: true })
  console.log(`[rpm] rpmbuild → ${name}-${version}-1.${rpmArch}.rpm (streaming from ${path.relative(bundleDir, buildroot)})`)
  const start = Bun.nanoseconds()
  await $`rpmbuild -bb ${specPath} --buildroot ${buildroot} --define ${`_topdir ${path.join(work, "top")}`} --define ${`_rpmdir ${outDir}`} --define ${"_rpmfilename %{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}.rpm"}`.quiet()
  const secs = ((Bun.nanoseconds() - start) / 1e9).toFixed(1)

  const rpmPath = path.join(outDir, `${name}-${version}-1.${rpmArch}.rpm`)
  if (!existsSync(rpmPath)) throw new Error(`[rpm] rpmbuild reported success but ${rpmPath} is missing`)
  const sizeMB = ((await stat(rpmPath)).size / 1024 / 1024).toFixed(0)
  console.log(`[rpm] done in ${secs}s -> ${path.relative(bundleDir, rpmPath)} (${sizeMB} MB)`)
} finally {
  // always remove the temp work dir, even if rpmbuild or a later step throws
  await rm(work, { recursive: true, force: true })
}
