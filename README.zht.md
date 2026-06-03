<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony 標誌">
    </picture>
  </a>
</p>
<p align="center">開源的 AI 程式設計代理。</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="建置狀態" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony 終端機 UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## 什麼是 EmberHarmony？

EmberHarmony 是一個在你的終端機中執行的開源 AI 程式設計代理。它與供應商無關——可搭配 Claude、OpenAI、Google、透過 Ollama 執行的本地模型，或任何相容於 OpenAI 的端點使用。它具備豐富的 TUI、內建的 LSP 支援，以及一個讓你能夠遠端操控它的主從式（client/server）架構。

### 零設定的本地模型

EmberHarmony 會自動探索你本地 [Ollama](https://ollama.com) 執行個體中已安裝的每一個模型。不需要 API 金鑰、不需要設定檔、不需要手動設定。只要 Ollama 正在執行，你的模型就會出現：

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

在對話進行到一半時，於雲端與本地模型之間切換。完全在你的機器上執行敏感的程式碼分析。當你需要前沿能力時使用雲端模型，當你需要隱私或離線存取時使用本地模型。

**與其他 AI 程式設計工具的主要差異：**

- **本地優先（Local-first）**——自動探索 Ollama 模型、零設定、不需要金鑰
- 100% 開源（MIT）
- 不綁定任何單一供應商——可搭配 Claude、OpenAI、Google、Ollama 等使用
- 開箱即用的 LSP 支援，提供智慧型程式碼導覽
- 豐富的終端機 UI，挑戰終端機可能性的極限
- 主從式架構——在你的機器上執行，從任何地方操控

### 安裝

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### 本地建置 + 安裝

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### 桌面應用程式（Beta）

EmberHarmony 也提供桌面應用程式版本。可直接從[發行頁面](https://github.com/SolaceHarmony/emberharmony/releases)下載。

| 平台                  | 下載                                       |
| --------------------- | ----------------------------------------- |
| macOS（Apple Silicon）| `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS（Intel）        | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`、`.rpm` 或 AppImage                |

#### 安裝目錄

安裝指令碼會依循以下優先順序：

1. `$EMBERHARMONY_INSTALL_DIR` — 自訂的安裝目錄
2. `$XDG_BIN_DIR` — 符合 XDG Base Directory 規範的路徑
3. `$HOME/bin` — 標準的使用者二進位目錄（若存在）
4. `$HOME/.emberharmony/bin` — 預設的後備選項

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### 代理（Agents）

EmberHarmony 內建兩個代理，你可以用 `Tab` 鍵在它們之間切換。

- **build** — 預設、具完整存取權限的代理，用於開發工作
- **plan** — 唯讀代理，用於分析與程式碼探索
  - 預設拒絕檔案編輯
  - 在執行 bash 指令前會徵求許可
  - 適合用於探索不熟悉的程式碼庫或規劃變更

另外還提供一個 **general** 子代理，用於複雜的搜尋與多步驟任務。它在內部使用，也可以在訊息中以 `@general` 來叫用。

### 供應商支援

EmberHarmony 可搭配任何相容於 OpenAI 的 API 使用。內建支援：

| 供應商 | 模型 | 所需設定 |
|----------|--------|---------------|
| **Ollama（本地）** | 從 `localhost:11434` 自動探索 | 無——只要執行 Ollama |
| **Ollama Cloud** | 託管的 Ollama 模型 | API 金鑰 |
| **Anthropic** | Claude Opus、Sonnet、Haiku | API 金鑰 |
| **OpenAI** | GPT-4o、o1、o3 | API 金鑰 |
| **Google** | Gemini Pro、Flash | API 金鑰 |
| **任何相容於 OpenAI 的服務** | LM Studio、vLLM、Together、Groq 等 | 端點 + 金鑰 |

#### Ollama 本地設定

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

若要使用非預設的 Ollama 位址，請新增至 `~/.config/emberharmony/config.json`：
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### 貢獻

如果你有興趣為 EmberHarmony 做出貢獻，請在提交 pull request 之前先閱讀我們的[貢獻指南](./CONTRIBUTING.md)。

### 以 EmberHarmony 為基礎進行開發

如果你正在進行的專案與 EmberHarmony 相關，且名稱中使用了「emberharmony」，請在你的 README 中加上說明，闡明該專案並非由 The Solace Project 所建置，且與我們無任何關聯。

### 致謝

EmberHarmony 是 opencode upstream 團隊所開發的 [opencode](https://github.com/anomalyco/opencode) 的一個分支（fork）。我們由衷感謝他們在打造一個卓越的開源 AI 程式設計代理上所奠定的基礎工作。本專案正是建立在他們的願景與工程之上。

### 維護者

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**社群：** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
