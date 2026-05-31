<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony 标志">
    </picture>
  </a>
</p>
<p align="center">开源 AI 编程代理。</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="构建状态" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony 终端界面](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## 什么是 EmberHarmony？

EmberHarmony 是一款在你的终端中运行的开源 AI 编程代理。它与提供商无关——你可以搭配 Claude、OpenAI、Google、通过 Ollama 运行的本地模型，或任何兼容 OpenAI 的端点来使用它。它具备功能丰富的 TUI、内置的 LSP 支持，以及让你能够远程驱动它的客户端/服务器架构。

### 零配置的本地模型

EmberHarmony 会自动发现你本地 [Ollama](https://ollama.com) 实例中安装的每一个模型。无需 API 密钥，无需配置文件，无需手动设置。只要 Ollama 正在运行，你的模型就会出现：

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

在对话过程中随时在云端模型与本地模型之间切换。完全在你自己的机器上运行敏感代码分析。需要前沿能力时使用云端模型，需要隐私或离线访问时使用本地模型。

**与其他 AI 编程工具的关键区别：**

- **本地优先**——自动发现 Ollama 模型，零配置，无需密钥
- 100% 开源（MIT）
- 不绑定任何单一提供商——可搭配 Claude、OpenAI、Google、Ollama 等使用
- 开箱即用的 LSP 支持，实现智能代码导航
- 功能丰富的终端界面，挑战终端中可实现的极限
- 客户端/服务器架构——在你的机器上运行，从任何地方驱动

### 安装

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### 本地构建 + 安装

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### 桌面应用（Beta）

EmberHarmony 也提供桌面应用程序。可直接从[发布页面](https://github.com/SolaceHarmony/emberharmony/releases)下载。

| 平台                  | 下载                                      |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`、`.rpm` 或 AppImage                |

#### 安装目录

安装脚本遵循以下优先级顺序：

1. `$EMBERHARMONY_INSTALL_DIR` — 自定义安装目录
2. `$XDG_BIN_DIR` — 符合 XDG 基础目录规范的路径
3. `$HOME/bin` — 标准用户二进制目录（如果存在）
4. `$HOME/.emberharmony/bin` — 默认回退路径

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### 代理（Agents）

EmberHarmony 内置两个代理，你可以用 `Tab` 键在它们之间切换。

- **build** — 默认的、具有完全访问权限的代理，用于开发工作
- **plan** — 只读代理，用于分析和代码探索
  - 默认拒绝文件编辑
  - 运行 bash 命令前会请求许可
  - 适合探索陌生的代码库或规划变更

还提供了一个 **general** 子代理，用于复杂搜索和多步骤任务。它在内部使用，并可在消息中通过 `@general` 调用。

### 提供商支持

EmberHarmony 可与任何兼容 OpenAI 的 API 配合使用。内置支持：

| 提供商 | 模型 | 所需配置 |
|----------|--------|---------------|
| **Ollama（本地）** | 从 `localhost:11434` 自动发现 | 无——只需运行 Ollama |
| **Ollama Cloud** | 托管的 Ollama 模型 | API 密钥 |
| **Anthropic** | Claude Opus、Sonnet、Haiku | API 密钥 |
| **OpenAI** | GPT-4o、o1、o3 | API 密钥 |
| **Google** | Gemini Pro、Flash | API 密钥 |
| **任何兼容 OpenAI 的服务** | LM Studio、vLLM、Together、Groq 等 | 端点 + 密钥 |

#### Ollama 本地设置

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

要使用非默认的 Ollama 地址，请添加到 `~/.config/emberharmony/config.json`：
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### 贡献

如果你有兴趣为 EmberHarmony 做贡献，请在提交拉取请求前阅读我们的[贡献指南](./CONTRIBUTING.md)。

### 基于 EmberHarmony 构建

如果你正在开发一个与 EmberHarmony 相关、并且名称中使用了 “emberharmony” 的项目，请在你的 README 中添加说明，澄清它并非由 The Solace Project 构建，且与我们没有关联。

### 致谢

EmberHarmony 是 opencode upstream 团队的 [opencode](https://github.com/anomalyco/opencode) 的一个分支。我们对他们在构建一款卓越的开源 AI 编程代理方面所打下的基础工作深表感激。本项目正是建立在他们的愿景与工程之上。

### 维护者

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**社区：** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
