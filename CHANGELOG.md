# Changelog

All notable changes to EmberHarmony will be documented in this file.

This project is a fork of [opencode](https://github.com/opencode-ai/opencode),
rebranded and maintained by [The Solace Project](https://github.com/SolaceHarmony).

## [1.4.6] - 2026-06-14

### Fixed

- **Voice broke on every tool use ("operation aborted")** — the session bridge
  ended a voice reply as soon as the first assistant message completed. But a
  tool-call step finalizes its message (`time.completed`, `finish: "tool-calls"`)
  *before* the tool runs, then opens a new assistant message for the result. So
  the bridge returned mid-tool, its stream closed, the abort handler POSTed
  `/session/abort`, and the server killed the running tool — voice then went
  silent (though still listening) because the continuation never streamed. The
  bridge now streams text from every assistant message in the turn (under one
  stable id, so TTS stays continuous) and ends only when the session goes idle.
- **Voice-runtime build aborted on a flaky HuggingFace model download** — the
  assembler's `download-files` step had no retry, so a single transient
  HuggingFace error (rate limiting surfaces as
  "tokenizerConfig.tokenizer_class undefined" when a non-JSON response is parsed
  as the turn-detector tokenizer config) failed the whole build. The download
  now retries with backoff and only fails loudly after exhausting attempts.
- **macOS notarization rejected the unsigned voice-runtime binaries** — the CI
  step that signs the bundled native binaries passed
  `--identity "$APPLE_SIGNING_IDENTITY"`, but that secret is intentionally unset
  in this repo (it is optional; `tauri-action` auto-derives the identity from
  the imported `APPLE_CERTIFICATE`). With an empty identity the signer took its
  ad-hoc "skip" path, so every nested `.node`/`.dylib`/`ffmpeg` shipped unsigned
  and notarization failed with "not signed with a valid Developer ID
  certificate". The step now derives the Developer ID identity from the imported
  cert in its keychain and fails loudly if none is found, instead of skipping.
- **Windows desktop build failed packaging the voice runtime** — the bundled
  runtime's `node_modules` was installed with bun, whose standalone install
  deep-nests `@livekit/agents`' genuinely-conflicting `@opentelemetry`
  dependencies (OTel 1.x *and* 2.x are both required) four to five
  `node_modules` levels deep, producing paths over Windows' 260-char `MAX_PATH`
  limit that the NSIS bundler can't open ("The system cannot find the file
  specified"). The voice-runtime assembler now installs with **npm**, whose
  hoisting collapses the same unavoidable conflicts into a shallow, shippable
  tree (longest path ~167 chars, down from ~205). 1.4.5 built clean on
  macOS/Linux but never produced a Windows bundle.

### Changed

- **Linux desktop ships `.deb` + `.rpm` only (AppImage dropped).** AppImage
  bundling stalls on x86_64 — the AppImage runtime/FUSE step hangs with no
  timeout, burning the full 60-minute build limit (arm64 took ~18 min but
  eventually finished it). It also wasn't in the configured bundle targets. The
  build now pins `--bundles deb,rpm` on Linux and drops the custom
  `truly-portable-appimage` tauri-cli, which also removes a ~5-minute
  per-Linux-build `cargo install` step.
- The voice-runtime assembler now reads the `@livekit/*` versions from the
  workspace catalog and the Bun version from `packageManager`, instead of
  hardcoding them — so the bundled runtime can no longer silently drift from
  what the worker is compiled against.

## [1.4.5] - 2026-06-14

### Changed

- **`@parcel/watcher` 2.5.1 → 2.5.6** — bumped the file-watcher and all seven
  pinned platform binary packages (`darwin-arm64`, `darwin-x64`,
  `linux-{arm64,x64}-{glibc,musl}`, `win32-x64`) plus the meta package, keeping
  the lockfile's full platform set aligned at 2.5.6.

## [1.4.4] - 2026-06-14

### Fixed

- **macOS desktop release failed notarization** — the bundled voice runtime
  ships prebuilt binaries (`bun`, `@ffmpeg-installer`, `@livekit/rtc-ffi`,
  `onnxruntime-node`, `sharp`/`libvips`) that arrive from npm/GitHub unsigned or
  third-party-signed. Tauri signs the app, its main binary, and the sidecar but
  seals nested resource code without signing it, so notarization rejected every
  macOS build of 1.4.3 with "not signed with a valid Developer ID certificate" /
  "signature does not include a secure timestamp". The build now re-signs every
  Mach-O in the runtime — found by magic bytes, so nothing is missed by
  name/extension — with the Developer ID cert + a secure timestamp + hardened
  runtime before `tauri build` seals the app (`scripts/sign-voice-runtime.ts`,
  wired into the local build and a CI keychain step). `--preserve-metadata`
  keeps each binary's entitlements, so `bun` retains the `allow-jit` /
  `disable-library-validation` it needs to run and to load the (now same-team)
  native libs.

## [1.4.3] - 2026-06-13

### Fixed

- **Voice mode now works in the packaged desktop app** — the LiveKit agents
  framework forks `node_modules` scripts and dynamically imports the agent file
  by path, so the voice worker could not run inside the compiled single-file
  CLI sidecar (which has no on-disk `node_modules`); voice was silently dead in
  every installed build. The desktop app now ships a self-contained voice
  runtime (a Bun binary, the bundled worker, the pruned native deps, and the
  pre-downloaded Silero VAD + turn-detector ONNX models) as a Tauri resource,
  and `serve` spawns that runtime instead. The bundle is assembled per platform
  at build time and pruned from ~647 MB to ~305 MB (dropping `onnxruntime-web`
  and `typescript`, and stripping source maps, type decls, docs, and tests).
- **Voice worker failed to load on paths containing a space** — the agents
  job-processor loaded the agent with `import(pathToFileURL(file).pathname)`,
  which keeps percent-encoding (`%20`) but drops the `file://` scheme, so any
  install path with a space (the dev app bundle, Windows `Program Files`, a
  user home with a space) resolved to a literal `%20` and failed with "Cannot
  find module". The bundled framework loader is now patched to use `.href`,
  matching the framework's own `download.js`.

## [1.4.2] - 2026-06-13

### Fixed

- **Desktop app missing from releases** — the `attach-assets` release step
  globbed only the top level of the downloaded artifacts, but tauri-action
  writes its bundles nested under `…/release/bundle/<type>/`, so the `.dmg`,
  `.exe`, `.deb`, `.rpm`, and `.AppImage` were never found, renamed, or
  uploaded — every prior release shipped CLI archives only. The step now finds
  each bundle recursively and uploads it under the canonical
  `emberharmony-desktop-*` name. (The macOS `.app`/`.dmg` are signed and
  notarized in CI, so they open without Gatekeeper friction — unlike the raw
  CLI binary, which is unsigned and meant to be installed via npm or the
  install script, not double-clicked.)

## [1.4.1] - 2026-06-13

### Fixed

- **`npm i -g @thesolaceproject/emberharmony` failed during postinstall** —
  both the postinstall and the runtime `bin/emberharmony` wrapper derived the
  platform binary package name by stripping the `@thesolaceproject/` scope,
  then looked for an unscoped `emberharmony-<platform>-<arch>` that is never
  published (the real packages are scoped, e.g.
  `@thesolaceproject/emberharmony-darwin-arm64`). The postinstall exited 1 and
  broke the install; the wrapper's `node_modules` walk likewise couldn't find a
  scoped package. Both now keep the full scoped name, and the wrapper descends
  into `node_modules/<scope>/`. (Latent since the package was scoped; only
  surfaced now that the CLI is installed from npm rather than built locally.)

## [1.4.0] - 2026-06-12

### Added

- **Voice mode (LiveKit)** — hands-free voice conversations with sessions on the
  desktop app. Speak to a session and hear the reply; the spoken turn runs
  through the same model, tools, permissions, and context as a typed prompt,
  and both sides appear in the chat. Includes a live transcript strip and an
  agent-state visualizer in the prompt input.
  - **Plan/build workflow** — spoken turns run the read-only `plan` agent by
    default; a small fast model classifies explicit confirmations ("yes, do
    it") and only then runs that single turn as `build`. A failed
    classification can never grant execution.
  - **Session bridge** — the voice agent worker's LLM step posts each
    utterance to the session API and streams the reply back over SSE, so the
    session stays the brain and voice is only ears, mouth, and transport.
  - **Voice settings panel** — configure the LiveKit connection (credentials
    stored in the local auth store, never config files), STT/TTS models, and
    the intent model entirely from the UI. No environment variables required;
    settings persist and the agent worker is managed by `emberharmony serve`.
  - Voice follows session and project switches, reconnecting into the new
    session's room so context always matches the visible session.
- **`@thesolaceproject/livekit-components-solid`** — vendored SolidJS port of
  `@livekit/components-react`, built on the framework-agnostic
  `@livekit/components-core` (Apache-2.0, with attribution).

### Security

- **Biased cryptographic random fixed** — base62 ID generation in `id.ts` and
  `util/identifier.ts` used `byte % 62`, biasing toward the first 8 of 62
  characters; both now use rejection sampling.
- **HTML double-unescaping fixed** — the markdown code-block decoder decoded
  `&amp;` before `&quot;`/`&#39;`, which could double-unescape an escaped
  entity into a real quote; `&amp;` now decodes last.
- **CLI argument handling hardened** — `run` reassembles argv for prompt /
  command text without shell-escaping (it never reaches a shell), choosing a
  quote character that preserves embedded quotes and backslashes so JSON args
  and Windows paths survive intact.
- **Dependency CVEs patched** — esbuild bumped to ≥ 0.28.1 (binary integrity
  verification, GHSA-gv7w-rqvm-qjhr) across the workspace and the VS Code SDK;
  earlier in the cycle hono → 4.12.21, devalue, js-cookie, qs, ws, react-router,
  and esbuild were bumped past fatal advisories blocking installs.

### Changed

- Desktop microphone support: `NSMicrophoneUsageDescription` added to the macOS
  bundle; CSP and Tauri capabilities allow LiveKit and Tauri IPC.

## [1.3.0] - 2026-05-30

Reconstructed from git history (`v1.2.2..1.3.0`); grouped by theme rather than
per-commit.

### Changed

- **Rebrand to the ember-flame identity** — pixel-perfect re-skin of app and
  desktop assets, with all 15 translated READMEs re-synced to the current
  English structure.
- **Stripped inherited opencode/SST scaffolding** — removed the SST runtime
  integration, Docker/AUR publishing, and unused upstream workflows and config.
- Desktop builds drop Intel macOS (`x86_64-apple-darwin`); the local desktop
  build is now self-contained with a declared-requirements preflight.

### Added

- **CI/CD overhaul** — reusable build workflow, per-merge CI, and CodeQL
  analysis (including buildless Rust extraction). Tag-driven releases: the tag
  names the artifacts, `version.json` names the build.
- Mock/CI-safe provider for E2E model-picker tests; model-dependent E2E tests
  skip on CI without provider credentials.

### Security

- **Supply chain** — vendor `models-snapshot.ts` instead of fetching from
  models.dev at build time.
- **Dependency CVEs** — nitro → 3.0.260429-beta (CVE-2026-44373); h3, astro,
  and wrangler bumped to close 4 CVEs; turbo, tar, and rand advisories patched;
  tauri, rustls-webpki, dompurify, and Rust crates updated.
- **Code-scanning fixes** — server-side request forgery, uncontrolled command
  line, biased cryptographic random, and missing workflow permissions; all
  GitHub Actions pinned and CI locked out of default-token writes.

## [1.2.2] - 2026-04-09

### Security

- **13 upstream workflows removed** — inherited from opencode fork, exposed
  `ANTHROPIC_API_KEY` and `EMBERHARMONY_API_KEY` to fork PRs via
  `pull_request_target` without fork detection, and fetched unverified code
  via `curl | bash` from the dev branch
- **11 CVEs patched** in dependencies:
  - hono 4.11.7 → 4.12.12 (serveStatic file access, cookie injection, prototype pollution)
  - vite 7.1.11 → 7.3.2 (arbitrary file read via WebSocket, `server.fs.deny` bypass)
  - drizzle-orm 0.41.0 → 0.45.2 (SQL injection via unescaped identifiers)
  - fast-xml-parser 5.3.4 → 5.5.11 (entity expansion DoS, regex injection bypass)
  - h3 → 1.15.11 (SSE injection, middleware bypass)
  - undici → 7.22.0 (WebSocket DoS, request smuggling)
  - file-type → 22.0.0 (infinite loop on malformed ASF input)
- **Code injection closed** — removed `new Function()` eval in debug agent CLI
- **Path traversal hardened** — `path.resolve()` normalization on server directory param
- **CORS restricted** — enterprise API endpoint locked to known origins
- **Open redirects blocked** — `window.location.href` assignments validate HTTPS
- **CSP headers added** — `secureHeaders` middleware on Hono server
- **GitHub Actions pinned** — all 13 workflow files use commit SHA references

### Changed

- README rewritten for EmberHarmony brand — removed dead links, stub package
  manager commands, and incorrect upstream references; added provider support
  section documenting Ollama auto-discovery

## [1.2.1] - 2026-04-07

### Fixed

- CI: quote scoped npm package name in publish workflow (unquoted `@` broke YAML parsing)
- CI: make publish job wait for all Tauri desktop builds before finalizing the release

## [1.2.0] - 2026-04-06

### Added

- Auto-discover local Ollama models — the provider queries `localhost:11434/api/tags`
  on startup and registers every installed model with zero configuration. Display
  names include parameter size and quantization level (e.g. `llama3.2 · 3B · Q4_K_M`).

## [1.1.1] - 2026-04-06

### Fixed

- Replace opencode square SVG mark/splash with ember flame icon
- Deep link protocol `codeharmony://` → `emberharmony://` (app + desktop)
- VS Code extension command IDs `codeharmony.*` → `emberharmony.*`
- Legal text references updated to EMBERHARMONY
- Use raw `g_object_get_data` FFI for Linux pinch-zoom disable — the typed
  `ObjectExt::data::<T>()` wrapper silently returns None for C-attached data
- Drop direct `gtk` crate dependency; use webkit2gtk glib re-export instead
- Remove redundant sidecar port lookup in desktop lib.rs
- Install script backward compat now checks `OPENCODE_INSTALL_DIR`
- Test snapshot path updated from code-harmony to emberharmony

## [1.1.0] - 2026-04-06

### Added

- EmberHarmony brand identity — full rebrand from upstream opencode
- Ember flame ASCII logo with figlet "standard" font wordmark for CLI splash
- Unified dev stack launcher (`bun run dev:stack`) — starts backend + Vite UI concurrently
- EH-1 default theme (renamed from EH-1) with updated preload and barrel exports
- `@thesolaceproject/emberharmony-plugin` package scope

### Changed

- All package names moved to `@thesolaceproject/emberharmony-*` scope
- CLI binary renamed to `emberharmony`
- GitHub workflows, Nix derivations, and CI configs updated for new naming
- Brand assets (logos, wordmarks, icons) renamed from code-harmony to emberharmony
- Desktop app product name set to "EmberHarmony"

### Security

- **minimatch** 10.0.3 → 10.2.5 — fixes 3 ReDoS vulnerabilities
- **dompurify** 3.3.1 → 3.3.3 — fixes XSS, mutation-XSS, prototype pollution, and URI validation bypass
- **astro** 5.15.9 → 5.18.1 — fixes remote allowlist bypass via unanchored wildcard
- **tauri** 2.9.5 → 2.10.3 — resolves transitive Rust CVEs:
  - rustls-webpki (faulty CRL matching)
  - tar (symlink chmod + PAX header handling)
  - quinn-proto (unauthenticated DoS via QUIC parameter parsing)
  - time (stack exhaustion DoS)
  - bytes (integer overflow in BytesMut::reserve)
- **react** 18.2.0 → 18.3.1, **@types/react** 18.0.25 → 18.3.28 — resolves peer dependency warnings

### Known Issues

- **glib** 0.18.5 (Linux-only) — unmaintained GTK3 binding with iterator unsoundness advisory. Requires gtk4 migration to resolve; not exploitable on macOS/Windows.
- **solid-js** pinned at 1.9.10 — version 1.9.12 removes `RequestEvent.locals` type, breaking console-app. Needs code migration before bumping.

## [1.0.0] - Initial Fork

Forked from [opencode](https://github.com/opencode-ai/opencode). Established
repository under SolaceHarmony org with CI/CD, desktop signing, and publishing
infrastructure.
