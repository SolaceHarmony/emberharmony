<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony Logo">
    </picture>
  </a>
</p>
<p align="center">Der quelloffene KI-Coding-Agent.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony Terminal-UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Was ist EmberHarmony?

EmberHarmony ist ein quelloffener KI-Coding-Agent, der in deinem Terminal läuft. Er ist anbieterunabhängig — nutze ihn mit Claude, OpenAI, Google, lokalen Modellen über Ollama oder jedem OpenAI-kompatiblen Endpunkt. Er bietet eine umfangreiche TUI, integrierte LSP-Unterstützung und eine Client/Server-Architektur, mit der du ihn aus der Ferne steuern kannst.

### Lokale Modelle ohne Konfiguration

EmberHarmony erkennt automatisch jedes Modell, das in deiner lokalen [Ollama](https://ollama.com)-Instanz installiert ist. Keine API-Schlüssel, keine Konfigurationsdateien, keine manuelle Einrichtung. Wenn Ollama läuft, erscheinen deine Modelle:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Wechsle mitten im Gespräch zwischen Cloud- und lokalen Modellen. Führe sensible Code-Analysen vollständig auf deinem Rechner aus. Nutze Cloud-Modelle, wenn du Spitzenfähigkeiten brauchst, und lokale Modelle, wenn du Privatsphäre oder Offline-Zugriff benötigst.

**Wesentliche Unterschiede zu anderen KI-Coding-Tools:**

- **Local-First** — automatische Erkennung von Ollama-Modellen, keine Konfiguration, keine Schlüssel nötig
- 100 % Open Source (MIT)
- An keinen einzelnen Anbieter gebunden — funktioniert mit Claude, OpenAI, Google, Ollama und mehr
- Sofort einsatzbereite LSP-Unterstützung für intelligente Code-Navigation
- Umfangreiche Terminal-UI, die die Grenzen des im Terminal Möglichen auslotet
- Client/Server-Architektur — auf deinem Rechner ausführen, von überall steuern

### Installation

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Lokaler Build + Installation

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Desktop-App (Beta)

EmberHarmony ist auch als Desktop-Anwendung verfügbar. Lade sie direkt von der [Releases-Seite](https://github.com/SolaceHarmony/emberharmony/releases) herunter.

| Plattform             | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm` oder AppImage              |

#### Installationsverzeichnis

Das Installationsskript beachtet die folgende Prioritätsreihenfolge:

1. `$EMBERHARMONY_INSTALL_DIR` — benutzerdefiniertes Installationsverzeichnis
2. `$XDG_BIN_DIR` — Pfad gemäß XDG Base Directory
3. `$HOME/bin` — standardmäßiges Benutzer-Binärverzeichnis (falls vorhanden)
4. `$HOME/.emberharmony/bin` — Standard-Fallback

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agenten

EmberHarmony enthält zwei integrierte Agenten, zwischen denen du mit der `Tab`-Taste wechseln kannst.

- **build** — Standard-Agent mit vollem Zugriff für Entwicklungsarbeiten
- **plan** — schreibgeschützter Agent für Analyse und Code-Exploration
  - Verweigert standardmäßig Datei-Bearbeitungen
  - Fragt um Erlaubnis, bevor Bash-Befehle ausgeführt werden
  - Ideal zum Erkunden unbekannter Codebasen oder zum Planen von Änderungen

Ein **general**-Subagent steht ebenfalls für komplexe Suchen und mehrstufige Aufgaben zur Verfügung. Er wird intern verwendet und kann in Nachrichten mit `@general` aufgerufen werden.

### Anbieter-Unterstützung

EmberHarmony funktioniert mit jeder OpenAI-kompatiblen API. Integrierte Unterstützung für:

| Anbieter | Modelle | Erforderliche Konfiguration |
|----------|--------|---------------|
| **Ollama (lokal)** | Automatisch von `localhost:11434` erkannt | Keine — einfach Ollama ausführen |
| **Ollama Cloud** | Gehostete Ollama-Modelle | API-Schlüssel |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API-Schlüssel |
| **OpenAI** | GPT-4o, o1, o3 | API-Schlüssel |
| **Google** | Gemini Pro, Flash | API-Schlüssel |
| **Beliebige OpenAI-kompatible** | LM Studio, vLLM, Together, Groq usw. | Endpunkt + Schlüssel |

#### Lokale Ollama-Einrichtung

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Um eine abweichende Ollama-Adresse zu verwenden, füge Folgendes zu `~/.config/emberharmony/config.json` hinzu:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Mitwirken

Wenn du daran interessiert bist, zu EmberHarmony beizutragen, lies bitte unseren [Leitfaden zum Mitwirken](./CONTRIBUTING.md), bevor du einen Pull Request einreichst.

### Auf EmberHarmony aufbauen

Wenn du an einem Projekt im Zusammenhang mit EmberHarmony arbeitest, das „emberharmony“ im Namen trägt, füge bitte einen Hinweis in deiner README hinzu, der klarstellt, dass es nicht von The Solace Project entwickelt wurde und nicht mit uns verbunden ist.

### Danksagungen

EmberHarmony ist ein Fork von [opencode](https://github.com/anomalyco/opencode) des opencode upstream-Teams. Wir sind zutiefst dankbar für ihre grundlegende Arbeit beim Aufbau eines herausragenden quelloffenen KI-Coding-Agenten. Dieses Projekt baut auf ihrer Vision und ihrer Ingenieursarbeit auf.

### Maintainer

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Community:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
