<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony-logo">
    </picture>
  </a>
</p>
<p align="center">Den åpne kildekode-baserte AI-kodeagenten.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Byggstatus" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony terminal-UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Hva er EmberHarmony?

EmberHarmony er en åpen kildekode-basert AI-kodeagent som kjører i terminalen din. Den er leverandøruavhengig — bruk den med Claude, OpenAI, Google, lokale modeller via Ollama, eller et hvilket som helst OpenAI-kompatibelt endepunkt. Den har en rik TUI, innebygd LSP-støtte og en klient/tjener-arkitektur som lar deg styre den eksternt.

### Lokale modeller uten konfigurasjon

EmberHarmony oppdager automatisk hver modell som er installert i din lokale [Ollama](https://ollama.com)-instans. Ingen API-nøkler, ingen konfigurasjonsfiler, ingen manuell oppsett. Hvis Ollama kjører, dukker modellene dine opp:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Bytt mellom sky- og lokale modeller midt i en samtale. Kjør sensitiv kodeanalyse fullstendig på din egen maskin. Bruk skymodeller når du trenger banebrytende kapasitet, og lokale modeller når du trenger personvern eller tilgang uten nett.

**Viktige forskjeller fra andre AI-kodeverktøy:**

- **Lokalt først** — automatisk oppdagelse av Ollama-modeller, ingen konfigurasjon, ingen nøkler nødvendig
- 100 % åpen kildekode (MIT)
- Ikke bundet til én enkelt leverandør — fungerer med Claude, OpenAI, Google, Ollama og mer
- Innebygd LSP-støtte for intelligent kodenavigasjon
- Rikt terminalgrensesnitt som flytter grensene for hva som er mulig i terminalen
- Klient/tjener-arkitektur — kjør på din egen maskin, styr fra hvor som helst

### Installasjon

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Lokal bygging + installasjon

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Skrivebordsapp (Beta)

EmberHarmony er også tilgjengelig som en skrivebordsapplikasjon. Last ned direkte fra [utgivelsessiden](https://github.com/SolaceHarmony/emberharmony/releases).

| Plattform             | Nedlasting                                |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, eller AppImage            |

#### Installasjonskatalog

Installasjonsskriptet følger denne prioritetsrekkefølgen:

1. `$EMBERHARMONY_INSTALL_DIR` — egendefinert installasjonskatalog
2. `$XDG_BIN_DIR` — sti i samsvar med XDG Base Directory
3. `$HOME/bin` — standard binærkatalog for brukeren (hvis den finnes)
4. `$HOME/.emberharmony/bin` — standard reserveløsning

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agenter

EmberHarmony inkluderer to innebygde agenter som du kan bytte mellom med `Tab`-tasten.

- **build** — standardagent med full tilgang for utviklingsarbeid
- **plan** — skrivebeskyttet agent for analyse og kodeutforskning
  - Nekter filredigeringer som standard
  - Ber om tillatelse før kjøring av bash-kommandoer
  - Ideell for å utforske ukjente kodebaser eller planlegge endringer

En **general**-underagent er også tilgjengelig for komplekse søk og oppgaver med flere trinn. Den brukes internt og kan kalles med `@general` i meldinger.

### Leverandørstøtte

EmberHarmony fungerer med ethvert OpenAI-kompatibelt API. Innebygd støtte for:

| Leverandør | Modeller | Konfigurasjon nødvendig |
|----------|--------|---------------|
| **Ollama (lokal)** | Auto-oppdaget fra `localhost:11434` | Ingen — bare kjør Ollama |
| **Ollama Cloud** | Hostede Ollama-modeller | API-nøkkel |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API-nøkkel |
| **OpenAI** | GPT-4o, o1, o3 | API-nøkkel |
| **Google** | Gemini Pro, Flash | API-nøkkel |
| **Alt OpenAI-kompatibelt** | LM Studio, vLLM, Together, Groq, osv. | Endepunkt + nøkkel |

#### Lokalt Ollama-oppsett

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

For å bruke en Ollama-adresse som ikke er standard, legg til i `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Bidra

Hvis du er interessert i å bidra til EmberHarmony, vennligst les [bidragsveiledningen](./CONTRIBUTING.md) før du sender inn en pull request.

### Bygge videre på EmberHarmony

Hvis du jobber med et prosjekt relatert til EmberHarmony som bruker "emberharmony" i navnet, vennligst legg til en merknad i README-filen din som klargjør at det ikke er bygget av The Solace Project og ikke er tilknyttet oss.

### Anerkjennelser

EmberHarmony er en fork av [opencode](https://github.com/anomalyco/opencode) av [SST](https://sst.dev)-teamet. Vi er dypt takknemlige for deres grunnleggende arbeid med å bygge en eksepsjonell åpen kildekode-basert AI-kodeagent. Dette prosjektet bygger videre på deres visjon og ingeniørarbeid.

### Vedlikeholder

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Fellesskap:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
