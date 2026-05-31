<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="EmberHarmony-logo">
    </picture>
  </a>
</p>
<p align="center">Den open source AI-kodningsagent.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![EmberHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Hvad er EmberHarmony?

EmberHarmony er en open source AI-kodningsagent, der kører i din terminal. Den er udbyder-agnostisk — brug den med Claude, OpenAI, Google, lokale modeller via Ollama eller et hvilket som helst OpenAI-kompatibelt endpoint. Den har en righoldig TUI, indbygget LSP-understøttelse og en klient/server-arkitektur, der lader dig styre den eksternt.

### Lokale modeller uden konfiguration

EmberHarmony opdager automatisk hver model, der er installeret i din lokale [Ollama](https://ollama.com)-instans. Ingen API-nøgler, ingen konfigurationsfiler, ingen manuel opsætning. Hvis Ollama kører, dukker dine modeller op:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Skift mellem cloud- og lokale modeller midt i en samtale. Kør følsom kodeanalyse helt på din egen maskine. Brug cloud-modeller, når du har brug for frontier-kapacitet, og lokale modeller, når du har brug for privatliv eller offline-adgang.

**Vigtige forskelle fra andre AI-kodningsværktøjer:**

- **Lokal-først** — automatisk opdagelse af Ollama-modeller, ingen konfiguration, ingen nøgler nødvendige
- 100 % open source (MIT)
- Ikke bundet til en enkelt udbyder — virker med Claude, OpenAI, Google, Ollama og mere
- Indbygget LSP-understøttelse til intelligent kodenavigation
- Righoldig terminal-UI, der flytter grænserne for, hvad der er muligt i terminalen
- Klient/server-arkitektur — kør på din maskine, styr den hvorfra som helst

### Installation

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Lokal build + installation

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Desktop-app (Beta)

EmberHarmony er også tilgængelig som en desktop-applikation. Download den direkte fra [releases-siden](https://github.com/SolaceHarmony/emberharmony/releases).

| Platform              | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, eller AppImage            |

#### Installationsmappe

Installationsscriptet respekterer følgende prioritetsrækkefølge:

1. `$EMBERHARMONY_INSTALL_DIR` — brugerdefineret installationsmappe
2. `$XDG_BIN_DIR` — sti i overensstemmelse med XDG Base Directory
3. `$HOME/bin` — standard brugerbinærmappe (hvis den findes)
4. `$HOME/.emberharmony/bin` — standard-fallback

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agenter

EmberHarmony indeholder to indbyggede agenter, som du kan skifte imellem med `Tab`-tasten.

- **build** — standard, fuld-adgangs-agent til udviklingsarbejde
- **plan** — skrivebeskyttet agent til analyse og kodeudforskning
  - Afviser filredigeringer som standard
  - Beder om tilladelse, før der køres bash-kommandoer
  - Ideel til at udforske ukendte kodebaser eller planlægge ændringer

En **general**-underagent er også tilgængelig til komplekse søgninger og opgaver i flere trin. Den bruges internt og kan kaldes med `@general` i beskeder.

### Udbyderunderstøttelse

EmberHarmony virker med ethvert OpenAI-kompatibelt API. Indbygget understøttelse for:

| Provider | Models | Config needed |
|----------|--------|---------------|
| **Ollama (lokal)** | Auto-opdaget fra `localhost:11434` | Ingen — bare kør Ollama |
| **Ollama Cloud** | Hostede Ollama-modeller | API-nøgle |
| **Anthropic** | Claude Opus, Sonnet, Haiku | API-nøgle |
| **OpenAI** | GPT-4o, o1, o3 | API-nøgle |
| **Google** | Gemini Pro, Flash | API-nøgle |
| **Enhver OpenAI-kompatibel** | LM Studio, vLLM, Together, Groq, osv. | Endpoint + nøgle |

#### Lokal Ollama-opsætning

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

For at bruge en Ollama-adresse, der ikke er standard, tilføj til `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Bidrag

Hvis du er interesseret i at bidrage til EmberHarmony, så læs venligst vores [bidragsguide](./CONTRIBUTING.md), før du indsender en pull request.

### At bygge videre på EmberHarmony

Hvis du arbejder på et projekt relateret til EmberHarmony, der bruger "emberharmony" i sit navn, så tilføj venligst en note i din README, der præciserer, at det ikke er bygget af The Solace Project og ikke er tilknyttet os.

### Anerkendelser

EmberHarmony er en fork af [opencode](https://github.com/anomalyco/opencode) af opencode upstream-teamet. Vi er dybt taknemmelige for deres grundlæggende arbejde med at bygge en enestående open source AI-kodningsagent. Dette projekt bygger videre på deres vision og ingeniørarbejde.

### Vedligeholder

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Fællesskab:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
