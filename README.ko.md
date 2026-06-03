<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony 로고">
    </picture>
  </a>
</p>
<p align="center">오픈 소스 AI 코딩 에이전트.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony 터미널 UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## EmberHarmony란 무엇인가요?

EmberHarmony는 터미널에서 실행되는 오픈 소스 AI 코딩 에이전트입니다. 특정 제공자에 종속되지 않습니다 — Claude, OpenAI, Google, Ollama를 통한 로컬 모델, 또는 OpenAI 호환 엔드포인트와 함께 사용할 수 있습니다. 풍부한 TUI, 내장 LSP 지원, 그리고 원격으로 조작할 수 있게 해주는 클라이언트/서버 아키텍처를 갖추고 있습니다.

### 설정이 필요 없는 로컬 모델

EmberHarmony는 로컬 [Ollama](https://ollama.com) 인스턴스에 설치된 모든 모델을 자동으로 검색합니다. API 키도, 설정 파일도, 수동 설정도 필요 없습니다. Ollama가 실행 중이라면 여러분의 모델이 나타납니다:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

대화 도중에 클라우드 모델과 로컬 모델을 전환하세요. 민감한 코드 분석을 전적으로 여러분의 컴퓨터에서 실행하세요. 최첨단 성능이 필요할 때는 클라우드 모델을, 프라이버시나 오프라인 접근이 필요할 때는 로컬 모델을 사용하세요.

**다른 AI 코딩 도구와의 주요 차이점:**

- **로컬 우선** — 자동 Ollama 모델 검색, 설정 불필요, 키 불필요
- 100% 오픈 소스 (MIT)
- 어느 단일 제공자에도 종속되지 않음 — Claude, OpenAI, Google, Ollama 등과 동작
- 지능형 코드 탐색을 위한 즉시 사용 가능한 LSP 지원
- 터미널에서 가능한 것의 한계를 밀어붙이는 풍부한 터미널 UI
- 클라이언트/서버 아키텍처 — 여러분의 컴퓨터에서 실행하고, 어디서든 조작

### 설치

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### 로컬 빌드 + 설치

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### 데스크톱 앱 (베타)

EmberHarmony는 데스크톱 애플리케이션으로도 제공됩니다. [릴리스 페이지](https://github.com/SolaceHarmony/emberharmony/releases)에서 직접 다운로드하세요.

| 플랫폼                | 다운로드                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, 또는 AppImage             |

#### 설치 디렉터리

설치 스크립트는 다음 우선순위를 따릅니다:

1. `$EMBERHARMONY_INSTALL_DIR` — 사용자 지정 설치 디렉터리
2. `$XDG_BIN_DIR` — XDG Base Directory를 준수하는 경로
3. `$HOME/bin` — 표준 사용자 바이너리 디렉터리 (존재하는 경우)
4. `$HOME/.emberharmony/bin` — 기본 폴백

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### 에이전트

EmberHarmony에는 `Tab` 키로 전환할 수 있는 두 개의 내장 에이전트가 포함되어 있습니다.

- **build** — 기본값, 개발 작업을 위한 전체 접근 권한 에이전트
- **plan** — 분석 및 코드 탐색을 위한 읽기 전용 에이전트
  - 기본적으로 파일 편집을 거부함
  - bash 명령을 실행하기 전에 권한을 요청함
  - 익숙하지 않은 코드베이스를 탐색하거나 변경을 계획하기에 이상적임

복잡한 검색과 다단계 작업을 위한 **general** 서브에이전트도 사용할 수 있습니다. 이는 내부적으로 사용되며 메시지에서 `@general`로 호출할 수 있습니다.

### 제공자 지원

EmberHarmony는 모든 OpenAI 호환 API와 동작합니다. 다음에 대한 내장 지원:

| 제공자 | 모델 | 필요한 설정 |
|----------|--------|---------------|
| **Ollama (로컬)** | `localhost:11434`에서 자동 검색됨 | 없음 — Ollama만 실행하면 됨 |
| **Ollama Cloud** | 호스팅된 Ollama 모델 | API 키 |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API 키 |
| **OpenAI** | GPT-4o, o1, o3 | API 키 |
| **Google** | Gemini Pro, Flash | API 키 |
| **모든 OpenAI 호환** | LM Studio, vLLM, Together, Groq 등 | 엔드포인트 + 키 |

#### Ollama 로컬 설정

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

기본값이 아닌 Ollama 주소를 사용하려면 `~/.config/emberharmony/config.json`에 추가하세요:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### 기여하기

EmberHarmony에 기여하는 데 관심이 있다면, 풀 리퀘스트를 제출하기 전에 [기여 가이드](./CONTRIBUTING.md)를 읽어 주세요.

### EmberHarmony 기반으로 빌드하기

이름에 "emberharmony"를 사용하는 EmberHarmony 관련 프로젝트를 작업하고 있다면, 해당 프로젝트가 The Solace Project에서 만든 것이 아니며 저희와 제휴 관계가 없음을 명확히 하는 안내를 README에 추가해 주세요.

### 감사의 말

EmberHarmony는 opencode upstream 팀의 [opencode](https://github.com/anomalyco/opencode)에서 포크한 프로젝트입니다. 뛰어난 오픈 소스 AI 코딩 에이전트를 구축한 그들의 기초 작업에 깊이 감사드립니다. 이 프로젝트는 그들의 비전과 엔지니어링 위에 세워졌습니다.

### 메인테이너

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**커뮤니티:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
