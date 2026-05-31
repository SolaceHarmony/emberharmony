<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Логотип EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">Открытый ИИ-агент для написания кода.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Статус сборки" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Терминальный интерфейс EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Что такое EmberHarmony?

EmberHarmony — это открытый ИИ-агент для написания кода, который работает в вашем терминале. Он не привязан к конкретному провайдеру — используйте его с Claude, OpenAI, Google, локальными моделями через Ollama или любым OpenAI-совместимым эндпоинтом. Он включает богатый TUI, встроенную поддержку LSP и клиент-серверную архитектуру, позволяющую управлять им удалённо.

### Локальные модели без настройки

EmberHarmony автоматически обнаруживает каждую модель, установленную в вашем локальном экземпляре [Ollama](https://ollama.com). Никаких API-ключей, файлов конфигурации или ручной настройки. Если Ollama запущен, ваши модели появляются:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Переключайтесь между облачными и локальными моделями прямо посреди разговора. Запускайте анализ конфиденциального кода полностью на своей машине. Используйте облачные модели, когда вам нужны передовые возможности, и локальные модели, когда вам нужна конфиденциальность или работа без сети.

**Ключевые отличия от других ИИ-инструментов для написания кода:**

- **Локальность прежде всего** — автоматическое обнаружение моделей Ollama, нулевая настройка, ключи не нужны
- 100% открытый исходный код (MIT)
- Не привязан ни к одному провайдеру — работает с Claude, OpenAI, Google, Ollama и другими
- Поддержка LSP из коробки для интеллектуальной навигации по коду
- Богатый терминальный интерфейс, раздвигающий границы возможного в терминале
- Клиент-серверная архитектура — запускайте на своей машине, управляйте откуда угодно

### Установка

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Локальная сборка + установка

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Настольное приложение (Beta)

EmberHarmony также доступен в виде настольного приложения. Скачайте его напрямую со [страницы релизов](https://github.com/SolaceHarmony/emberharmony/releases).

| Платформа             | Загрузка                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm` или AppImage               |

#### Каталог установки

Скрипт установки соблюдает следующий порядок приоритета:

1. `$EMBERHARMONY_INSTALL_DIR` — пользовательский каталог установки
2. `$XDG_BIN_DIR` — путь, совместимый с XDG Base Directory
3. `$HOME/bin` — стандартный пользовательский каталог для бинарных файлов (если существует)
4. `$HOME/.emberharmony/bin` — запасной вариант по умолчанию

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Агенты

EmberHarmony включает два встроенных агента, между которыми можно переключаться клавишей `Tab`.

- **build** — агент по умолчанию с полным доступом для работы над разработкой
- **plan** — агент только для чтения, предназначенный для анализа и изучения кода
  - По умолчанию запрещает редактирование файлов
  - Запрашивает разрешение перед выполнением команд bash
  - Идеален для изучения незнакомых кодовых баз или планирования изменений

Также доступен субагент **general** для сложных поисков и многошаговых задач. Он используется внутренне и может быть вызван с помощью `@general` в сообщениях.

### Поддержка провайдеров

EmberHarmony работает с любым OpenAI-совместимым API. Встроенная поддержка:

| Провайдер | Модели | Требуемая настройка |
|----------|--------|---------------|
| **Ollama (локально)** | Автоматически обнаруживаются на `localhost:11434` | Нет — просто запустите Ollama |
| **Ollama Cloud** | Размещённые модели Ollama | API-ключ |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API-ключ |
| **OpenAI** | GPT-4o, o1, o3 | API-ключ |
| **Google** | Gemini Pro, Flash | API-ключ |
| **Любой OpenAI-совместимый** | LM Studio, vLLM, Together, Groq и т. д. | Эндпоинт + ключ |

#### Локальная настройка Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Чтобы использовать нестандартный адрес Ollama, добавьте в `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Участие в разработке

Если вы заинтересованы в том, чтобы внести вклад в EmberHarmony, пожалуйста, прочитайте наше [руководство для участников](./CONTRIBUTING.md), прежде чем отправлять pull request.

### Разработка на основе EmberHarmony

Если вы работаете над проектом, связанным с EmberHarmony, который использует «emberharmony» в своём названии, пожалуйста, добавьте в свой README примечание, поясняющее, что он создан не The Solace Project и не связан с нами.

### Благодарности

EmberHarmony является форком [opencode](https://github.com/anomalyco/opencode) от команды opencode upstream. Мы глубоко благодарны за их основополагающую работу по созданию исключительного открытого ИИ-агента для написания кода. Этот проект развивает их видение и инженерные решения.

### Сопровождающий

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Сообщество:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
