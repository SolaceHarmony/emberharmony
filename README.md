<p align="center">
  <a href="https://solace.ofharmony.ai">
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

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh.md">简体中文</a> |
  <a href="README.zht.md">繁體中文</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.it.md">Italiano</a> |
  <a href="README.da.md">Dansk</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.pl.md">Polski</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.ar.md">العربية</a> |
  <a href="README.no.md">Norsk</a> |
  <a href="README.br.md">Português (Brasil)</a> |
  <a href="README.th.md">ไทย</a>
</p>

[![EmberHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://solace.ofharmony.ai)

---

### Installation

```bash
# YOLO
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# Package managers
npm i -g @thesolaceproject/emberharmony@latest        # or bun/pnpm/yarn
scoop install emberharmony             # Windows
choco install emberharmony             # Windows
paru -S emberharmony-bin               # Arch Linux
mise use -g emberharmony               # Any OS
nix run nixpkgs#emberharmony           # or github:SolaceHarmony/emberharmony for latest dev branch
```

> [!TIP]
> Remove versions older than 0.1.x before installing.

#### Local Build + Install (No CI)

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.0.0.tgz
```

### Desktop App (BETA)

EmberHarmony is also available as a desktop application. Download directly from the [releases page](https://github.com/SolaceHarmony/emberharmony/releases) or [solace.ofharmony.ai/download](https://github.com/SolaceHarmony/emberharmony/releases).

| Platform              | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, or AppImage               |

```bash
# Windows (Scoop)
scoop bucket add extras; scoop install extras@thesolaceproject/emberharmony-desktop
```

#### Installation Directory

The install script respects the following priority order for the installation path:

1. `$EMBERHARMONY_INSTALL_DIR` - Custom installation directory (preferred)
2. `$EMBERHARMONY_INSTALL_DIR` - Backward compat
3. `$XDG_BIN_DIR` - XDG Base Directory Specification compliant path
4. `$HOME/bin` - Standard user binary directory (if exists or can be created)
5. `$HOME/.emberharmony/bin` - Default fallback

```bash
# Examples
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
XDG_BIN_DIR=$HOME/.local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agents

EmberHarmony includes two built-in agents you can switch between with the `Tab` key.

- **build** - Default, full access agent for development work
- **plan** - Read-only agent for analysis and code exploration
  - Denies file edits by default
  - Asks permission before running bash commands
  - Ideal for exploring unfamiliar codebases or planning changes

Also, included is a **general** subagent for complex searches and multistep tasks.
This is used internally and can be invoked using `@general` in messages.

Learn more about [agents](https://solace.ofharmony.ai/docs/agents).

### Documentation

For more info on how to configure EmberHarmony [**head over to our docs**](https://solace.ofharmony.ai/docs).

### Contributing

If you're interested in contributing to EmberHarmony, please read our [contributing docs](./CONTRIBUTING.md) before submitting a pull request.

### Building on EmberHarmony

If you are working on a project that's related to EmberHarmony and is using "emberharmony" or "emberharmony" as a part of its name, please add a note in your README to clarify that it is not built by The Solace Project and is not affiliated with us in any way.

### FAQ

#### How is this different from Claude Code?

It's very similar to Claude Code in terms of capability. Here are the key differences:

- 100% open source
- Not coupled to any provider. EmberHarmony can be used with Claude, OpenAI, Google, or even local models. As models evolve the gaps between them will close and pricing will drop, so being provider-agnostic is important.
- Out of the box LSP support
- A focus on TUI — we are going to push the limits of what's possible in the terminal.
- A client/server architecture. This for example can allow EmberHarmony to run on your computer, while you can drive it remotely from a mobile app. The TUI frontend is just one of the possible clients.

### Acknowledgments

EmberHarmony is a fork of [EmberHarmony](https://github.com/sst/emberharmony) by the [SST](https://sst.dev) team. We are deeply grateful for their foundational work in building an exceptional open source AI coding agent. This project would not exist without their vision and engineering.

### Maintainer

**Sydney Renee** — sydney@solace.ofharmony.ai
[The Solace Project](https://solace.ofharmony.ai)

---

**Join our community** [Discord](https://discord.gg/EdF8f7JR) | [GitHub Discussions](https://github.com/SolaceHarmony/emberharmony/discussions)
