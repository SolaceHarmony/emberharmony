<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Logo di EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">L'agente di coding AI open source.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Interfaccia da terminale di EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Cos'è EmberHarmony?

EmberHarmony è un agente di coding AI open source che gira nel tuo terminale. È indipendente dal provider — usalo con Claude, OpenAI, Google, modelli locali tramite Ollama o qualsiasi endpoint compatibile con OpenAI. Offre una ricca TUI, supporto LSP integrato e un'architettura client/server che ti permette di pilotarlo da remoto.

### Modelli locali a configurazione zero

EmberHarmony rileva automaticamente ogni modello installato nella tua istanza locale di [Ollama](https://ollama.com). Nessuna chiave API, nessun file di configurazione, nessuna impostazione manuale. Se Ollama è in esecuzione, i tuoi modelli compaiono:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Passa dai modelli cloud a quelli locali a metà conversazione. Esegui analisi di codice sensibile interamente sulla tua macchina. Usa i modelli cloud quando hai bisogno di capacità di frontiera, i modelli locali quando hai bisogno di privacy o accesso offline.

**Differenze chiave rispetto agli altri strumenti di coding AI:**

- **Local-first** — rilevamento automatico dei modelli Ollama, configurazione zero, nessuna chiave necessaria
- 100% open source (MIT)
- Non vincolato a un singolo provider — funziona con Claude, OpenAI, Google, Ollama e altri ancora
- Supporto LSP pronto all'uso per una navigazione intelligente del codice
- Ricca interfaccia da terminale che spinge al limite ciò che è possibile fare nel terminale
- Architettura client/server — eseguilo sulla tua macchina, pilotalo da ovunque

### Installazione

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Build + installazione locale

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### App desktop (Beta)

EmberHarmony è disponibile anche come applicazione desktop. Scaricala direttamente dalla [pagina delle release](https://github.com/SolaceHarmony/emberharmony/releases).

| Piattaforma           | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, o AppImage                |

#### Directory di installazione

Lo script di installazione rispetta il seguente ordine di priorità:

1. `$EMBERHARMONY_INSTALL_DIR` — directory di installazione personalizzata
2. `$XDG_BIN_DIR` — percorso conforme alla XDG Base Directory
3. `$HOME/bin` — directory standard dei binari utente (se esiste)
4. `$HOME/.emberharmony/bin` — fallback predefinito

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agenti

EmberHarmony include due agenti integrati tra cui puoi passare con il tasto `Tab`.

- **build** — agente predefinito ad accesso completo per il lavoro di sviluppo
- **plan** — agente di sola lettura per l'analisi e l'esplorazione del codice
  - Nega le modifiche ai file in modo predefinito
  - Chiede il permesso prima di eseguire comandi bash
  - Ideale per esplorare codebase sconosciute o pianificare modifiche

È disponibile anche un subagente **general** per ricerche complesse e attività in più passaggi. Viene usato internamente e può essere invocato con `@general` nei messaggi.

### Supporto dei provider

EmberHarmony funziona con qualsiasi API compatibile con OpenAI. Supporto integrato per:

| Provider | Modelli | Configurazione necessaria |
|----------|--------|---------------|
| **Ollama (locale)** | Rilevati automaticamente da `localhost:11434` | Nessuna — basta avviare Ollama |
| **Ollama Cloud** | Modelli Ollama ospitati | Chiave API |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Chiave API |
| **OpenAI** | GPT-4o, o1, o3 | Chiave API |
| **Google** | Gemini Pro, Flash | Chiave API |
| **Qualsiasi compatibile con OpenAI** | LM Studio, vLLM, Together, Groq, ecc. | Endpoint + chiave |

#### Configurazione locale di Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Per usare un indirizzo Ollama non predefinito, aggiungi a `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Contribuire

Se sei interessato a contribuire a EmberHarmony, leggi la nostra [guida ai contributi](./CONTRIBUTING.md) prima di inviare una pull request.

### Costruire su EmberHarmony

Se stai lavorando a un progetto legato a EmberHarmony che usa "emberharmony" nel proprio nome, aggiungi una nota nel tuo README per chiarire che non è realizzato da The Solace Project e non è affiliato con noi.

### Ringraziamenti

EmberHarmony è un fork di [opencode](https://github.com/anomalyco/opencode) realizzato dal team opencode upstream. Siamo profondamente grati per il loro lavoro fondamentale nella creazione di un eccezionale agente di coding AI open source. Questo progetto si basa sulla loro visione e sulla loro ingegneria.

### Manutentore

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Community:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
