# Changelog

All notable changes to EmberHarmony will be documented in this file.

This project is a fork of [opencode](https://github.com/opencode-ai/opencode),
rebranded and maintained by [The Solace Project](https://github.com/SolaceHarmony).

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
- EH-1 default theme (renamed from OC-1) with updated preload and barrel exports
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
