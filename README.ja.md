<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony ロゴ">
    </picture>
  </a>
</p>
<p align="center">オープンソースの AI コーディングエージェント。</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="ビルドステータス" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony ターミナル UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## EmberHarmony とは?

EmberHarmony は、ターミナル上で動作するオープンソースの AI コーディングエージェントです。プロバイダーに依存せず、Claude、OpenAI、Google、Ollama 経由のローカルモデル、または任意の OpenAI 互換エンドポイントと組み合わせて利用できます。リッチな TUI、組み込みの LSP サポート、そしてリモートから操作できるクライアント/サーバーアーキテクチャを備えています。

### ゼロコンフィグのローカルモデル

EmberHarmony は、ローカルの [Ollama](https://ollama.com) インスタンスにインストールされたすべてのモデルを自動的に検出します。API キーも、設定ファイルも、手動セットアップも不要です。Ollama が起動していれば、あなたのモデルが表示されます:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

会話の途中でクラウドモデルとローカルモデルを切り替えられます。機密性の高いコード解析を、すべて自分のマシン上で実行できます。フロンティアレベルの能力が必要なときはクラウドモデルを、プライバシーやオフラインアクセスが必要なときはローカルモデルを使いましょう。

**他の AI コーディングツールとの主な違い:**

- **ローカルファースト** — Ollama モデルの自動検出、ゼロコンフィグ、キー不要
- 100% オープンソース (MIT)
- 単一のプロバイダーに縛られない — Claude、OpenAI、Google、Ollama などと連携
- インテリジェントなコードナビゲーションのための LSP サポートを標準装備
- ターミナルで可能なことの限界に挑むリッチなターミナル UI
- クライアント/サーバーアーキテクチャ — 自分のマシンで動かし、どこからでも操作

### インストール

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### ローカルビルド + インストール

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### デスクトップアプリ (ベータ)

EmberHarmony はデスクトップアプリケーションとしても利用できます。[リリースページ](https://github.com/SolaceHarmony/emberharmony/releases)から直接ダウンロードしてください。

| プラットフォーム      | ダウンロード                              |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`、`.rpm`、または AppImage           |

#### インストールディレクトリ

インストールスクリプトは、以下の優先順位に従います:

1. `$EMBERHARMONY_INSTALL_DIR` — カスタムインストールディレクトリ
2. `$XDG_BIN_DIR` — XDG Base Directory に準拠したパス
3. `$HOME/bin` — 標準のユーザーバイナリディレクトリ (存在する場合)
4. `$HOME/.emberharmony/bin` — デフォルトのフォールバック

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### エージェント

EmberHarmony には、`Tab` キーで切り替えられる 2 つの組み込みエージェントが含まれています。

- **build** — デフォルトの、開発作業向けのフルアクセスエージェント
- **plan** — 解析やコード探索のための読み取り専用エージェント
  - デフォルトでファイル編集を拒否します
  - bash コマンドを実行する前に許可を求めます
  - 不慣れなコードベースの探索や変更の計画に最適です

複雑な検索や多段階のタスク向けに、**general** サブエージェントも利用できます。これは内部的に使用され、メッセージ内で `@general` と記述することで呼び出せます。

### プロバイダーサポート

EmberHarmony は、任意の OpenAI 互換 API と連携します。以下が標準でサポートされています:

| プロバイダー | モデル | 必要な設定 |
|----------|--------|---------------|
| **Ollama (ローカル)** | `localhost:11434` から自動検出 | なし — Ollama を起動するだけ |
| **Ollama Cloud** | ホスト型 Ollama モデル | API キー |
| **Anthropic** | Claude Opus、Sonnet、Haiku | API キー |
| **OpenAI** | GPT-4o、o1、o3 | API キー |
| **Google** | Gemini Pro、Flash | API キー |
| **任意の OpenAI 互換** | LM Studio、vLLM、Together、Groq など | エンドポイント + キー |

#### Ollama ローカルセットアップ

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

デフォルト以外の Ollama アドレスを使用するには、`~/.config/emberharmony/config.json` に以下を追加します:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### コントリビューション

EmberHarmony へのコントリビューションに興味がある方は、プルリクエストを送信する前に[コントリビューションガイド](./CONTRIBUTING.md)をお読みください。

### EmberHarmony 上での開発

EmberHarmony に関連するプロジェクトで、その名前に「emberharmony」を使用している場合は、それが The Solace Project によって作られたものではなく、私たちとは無関係であることを明確にする注記を README に追加してください。

### 謝辞

EmberHarmony は、opencode upstream チームによる [opencode](https://github.com/anomalyco/opencode) のフォークです。優れたオープンソース AI コーディングエージェントを構築するという彼らの基礎的な仕事に、私たちは深く感謝しています。本プロジェクトは、彼らのビジョンとエンジニアリングの上に築かれています。

### メンテナー

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**コミュニティ:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
