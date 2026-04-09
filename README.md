<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony logo">
    </picture>
  </a>
</p>
<p align="center">The open source AI coding agent.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/publish.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/publish.yml?style=flat-square&branch=dev" /></a>
</p>

[![EmberHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## What is EmberHarmony?

EmberHarmony is an open source AI coding agent that runs in your terminal. It's provider-agnostic — use it with Claude, OpenAI, Google, local models via Ollama, or any OpenAI-compatible endpoint. It features a rich TUI, built-in LSP support, and a client/server architecture that lets you drive it remotely.

**Key differences from other AI coding tools:**

- 100% open source (MIT)
- Not coupled to any single provider — works with Claude, OpenAI, Google, Ollama, and more
- Out-of-the-box LSP support for intelligent code navigation
- Rich terminal UI pushing the limits of what's possible in the terminal
- Client/server architecture — run on your machine, drive from anywhere

### Installation

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Local Build + Install

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Desktop App (Beta)

EmberHarmony is also available as a desktop application. Download directly from the [releases page](https://github.com/SolaceHarmony/emberharmony/releases).

| Platform              | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, or AppImage               |

#### Installation Directory

The install script respects the following priority order:

1. `$EMBERHARMONY_INSTALL_DIR` — custom installation directory
2. `$XDG_BIN_DIR` — XDG Base Directory compliant path
3. `$HOME/bin` — standard user binary directory (if exists)
4. `$HOME/.emberharmony/bin` — default fallback

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agents

EmberHarmony includes two built-in agents you can switch between with the `Tab` key.

- **build** — default, full-access agent for development work
- **plan** — read-only agent for analysis and code exploration
  - Denies file edits by default
  - Asks permission before running bash commands
  - Ideal for exploring unfamiliar codebases or planning changes

A **general** subagent is also available for complex searches and multistep tasks. It's used internally and can be invoked with `@general` in messages.

### Provider Support

EmberHarmony works with any OpenAI-compatible API. Built-in support for:

- **Anthropic** (Claude) — Opus, Sonnet, Haiku
- **OpenAI** — GPT-4o, o1, o3
- **Google** — Gemini Pro, Flash
- **Ollama** — local models auto-discovered on startup
- **Any OpenAI-compatible endpoint** — LM Studio, vLLM, Together, Groq, etc.

Local Ollama models are detected automatically when Ollama is running — no configuration needed.

### Contributing

If you're interested in contributing to EmberHarmony, please read our [contributing guide](./CONTRIBUTING.md) before submitting a pull request.

### Building on EmberHarmony

If you are working on a project related to EmberHarmony that uses "emberharmony" in its name, please add a note in your README clarifying that it is not built by The Solace Project and is not affiliated with us.

### Acknowledgments

EmberHarmony is a fork of [opencode](https://github.com/opencode-ai/opencode) by the [SST](https://sst.dev) team. We are deeply grateful for their foundational work in building an exceptional open source AI coding agent. This project builds on their vision and engineering.

### Maintainer

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Community:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
