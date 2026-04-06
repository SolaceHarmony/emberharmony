<p align="center">
  <a href="https://solace.ofharmony.ai">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony logo">
    </picture>
  </a>
</p>
<p align="center">開源的 AI Coding Agent。</p>
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
  <a href="README.br.md">Português (Brasil)</a>
</p>

[![EmberHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://solace.ofharmony.ai)

---

### 安裝

```bash
# 直接安裝 (YOLO)
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# 套件管理員
npm i -g @thesolaceproject/emberharmony@latest        # 也可使用 bun/pnpm/yarn
scoop install emberharmony             # Windows
choco install emberharmony             # Windows
paru -S emberharmony-bin               # Arch Linux
mise use -g emberharmony               # 任何作業系統
nix run nixpkgs#emberharmony           # 或使用 github:SolaceHarmony/emberharmony 以取得最新開發分支
```

> [!TIP]
> 安裝前請先移除 0.1.x 以前的舊版本。

### 桌面應用程式 (BETA)

EmberHarmony 也提供桌面版應用程式。您可以直接從 [發佈頁面 (releases page)](https://github.com/SolaceHarmony/emberharmony/releases) 或 [solace.ofharmony.ai/download](https://github.com/SolaceHarmony/emberharmony/releases) 下載。

| 平台                  | 下載連結                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, 或 AppImage               |

```bash
# Windows (Scoop)
scoop bucket add extras; scoop install extras@thesolaceproject/emberharmony-desktop
```

#### 安裝目錄

安裝腳本會依據以下優先順序決定安裝路徑：

1. `$EMBERHARMONY_INSTALL_DIR` - 自定義安裝目錄
2. `$XDG_BIN_DIR` - 符合 XDG 基礎目錄規範的路徑
3. `$HOME/bin` - 標準使用者執行檔目錄 (若存在或可建立)
4. `$HOME/.emberharmony/bin` - 預設備用路徑

```bash
# 範例
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
XDG_BIN_DIR=$HOME/.local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agents

EmberHarmony 內建了兩種 Agent，您可以使用 `Tab` 鍵快速切換。

- **build** - 預設模式，具備完整權限的 Agent，適用於開發工作。
- **plan** - 唯讀模式，適用於程式碼分析與探索。
  - 預設禁止修改檔案。
  - 執行 bash 指令前會詢問權限。
  - 非常適合用來探索陌生的程式碼庫或規劃變更。

此外，EmberHarmony 還包含一個 **general** 子 Agent，用於處理複雜搜尋與多步驟任務。此 Agent 供系統內部使用，亦可透過在訊息中輸入 `@general` 來呼叫。

了解更多關於 [Agents](https://solace.ofharmony.ai/docs/agents) 的資訊。

### 線上文件

關於如何設定 EmberHarmony 的詳細資訊，請參閱我們的 [**官方文件**](https://solace.ofharmony.ai/docs)。

### 參與貢獻

如果您有興趣參與 EmberHarmony 的開發，請在提交 Pull Request 前先閱讀我們的 [貢獻指南 (Contributing Docs)](./CONTRIBUTING.md)。

### 基於 EmberHarmony 進行開發

如果您正在開發與 EmberHarmony 相關的專案，並在名稱中使用了 "emberharmony"（例如 "emberharmony-dashboard" 或 "emberharmony-mobile"），請在您的 README 中加入聲明，說明該專案並非由 EmberHarmony 團隊開發，且與我們沒有任何隸屬關係。

### 常見問題 (FAQ)

#### 這跟 Claude Code 有什麼不同？

在功能面上與 Claude Code 非常相似。以下是關鍵差異：

- 100% 開源。
- 不綁定特定的服務提供商。雖然我們推薦使用透過提供的模型，但 EmberHarmony 也可搭配 Claude, OpenAI, Google 甚至本地模型使用。隨著模型不斷演進，彼此間的差距會縮小且價格會下降，因此具備「不限廠商 (provider-agnostic)」的特性至關重要。
- 內建 LSP (語言伺服器協定) 支援。
- 專注於終端機介面 (TUI)。EmberHarmony我們將不斷挑戰終端機介面的極限。
- 客戶端/伺服器架構 (Client/Server Architecture)。這讓 EmberHarmony 能夠在您的電腦上運行的同時，由行動裝置進行遠端操控。這意味著 TUI 前端只是眾多可能的客戶端之一。


### Acknowledgments

EmberHarmony is a fork of [EmberHarmony](https://github.com/sst/emberharmony) by the [SST](https://sst.dev) team. We are deeply grateful for their foundational work in building an exceptional open source AI coding agent.

### Maintainer

**Sydney Renee** — sydney@solace.ofharmony.ai
[The Solace Project](https://solace.ofharmony.ai)

---

**加入我們的社群** [Discord](https://discord.gg/EdF8f7JR) | [GitHub Discussions](https://github.com/SolaceHarmony/emberharmony/discussions)
