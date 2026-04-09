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

### Zero-Config Local Models

EmberHarmony automatically discovers every model installed in your local [Ollama](https://ollama.com) instance. No API keys, no configuration files, no manual setup. If Ollama is running, your models appear:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Switch between cloud and local models mid-conversation. Run sensitive code analysis entirely on your machine. Use cloud models when you need frontier capability, local models when you need privacy or offline access.

**Key differences from other AI coding tools:**

- **Local-first** — automatic Ollama model discovery, zero config, no keys needed
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

| Provider | Models | Config needed |
|----------|--------|---------------|
| **Ollama (local)** | Auto-discovered from `localhost:11434` | None — just run Ollama |
| **Ollama Cloud** | Hosted Ollama models | API key |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API key |
| **OpenAI** | GPT-4o, o1, o3 | API key |
| **Google** | Gemini Pro, Flash | API key |
| **Any OpenAI-compatible** | LM Studio, vLLM, Together, Groq, etc. | Endpoint + key |

#### Ollama Local Setup

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

To use a non-default Ollama address, add to `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

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
