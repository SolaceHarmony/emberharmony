<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Logo EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">Otwartoźródłowy agent AI do kodowania.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Status kompilacji" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Interfejs terminalowy EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Czym jest EmberHarmony?

EmberHarmony to otwartoźródłowy agent AI do kodowania, który działa w Twoim terminalu. Jest niezależny od dostawcy — używaj go z Claude, OpenAI, Google, modelami lokalnymi przez Ollama lub dowolnym punktem końcowym zgodnym z OpenAI. Oferuje bogaty interfejs TUI, wbudowane wsparcie LSP oraz architekturę klient/serwer, która pozwala sterować nim zdalnie.

### Modele lokalne bez konfiguracji

EmberHarmony automatycznie wykrywa każdy model zainstalowany w Twojej lokalnej instancji [Ollama](https://ollama.com). Bez kluczy API, bez plików konfiguracyjnych, bez ręcznej konfiguracji. Jeśli Ollama działa, Twoje modele się pojawiają:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Przełączaj się między modelami chmurowymi a lokalnymi w trakcie rozmowy. Uruchamiaj wrażliwą analizę kodu w całości na swoim komputerze. Korzystaj z modeli chmurowych, gdy potrzebujesz najnowocześniejszych możliwości, a z modeli lokalnych, gdy potrzebujesz prywatności lub dostępu offline.

**Kluczowe różnice względem innych narzędzi AI do kodowania:**

- **Lokalne na pierwszym miejscu** — automatyczne wykrywanie modeli Ollama, brak konfiguracji, bez potrzeby kluczy
- W 100% otwartoźródłowy (MIT)
- Niezwiązany z żadnym pojedynczym dostawcą — działa z Claude, OpenAI, Google, Ollama i innymi
- Wsparcie LSP od razu po instalacji dla inteligentnej nawigacji po kodzie
- Bogaty interfejs terminalowy przesuwający granice możliwości terminala
- Architektura klient/serwer — uruchamiaj na swoim komputerze, steruj z dowolnego miejsca

### Instalacja

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Lokalna kompilacja + instalacja

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Aplikacja desktopowa (Beta)

EmberHarmony jest również dostępny jako aplikacja desktopowa. Pobierz bezpośrednio ze [strony wydań](https://github.com/SolaceHarmony/emberharmony/releases).

| Platforma             | Pobranie                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm` lub AppImage               |

#### Katalog instalacji

Skrypt instalacyjny respektuje następującą kolejność priorytetów:

1. `$EMBERHARMONY_INSTALL_DIR` — niestandardowy katalog instalacji
2. `$XDG_BIN_DIR` — ścieżka zgodna ze standardem XDG Base Directory
3. `$HOME/bin` — standardowy katalog binariów użytkownika (jeśli istnieje)
4. `$HOME/.emberharmony/bin` — domyślny wariant awaryjny

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agenci

EmberHarmony zawiera dwóch wbudowanych agentów, między którymi możesz przełączać się klawiszem `Tab`.

- **build** — domyślny agent z pełnym dostępem do pracy programistycznej
- **plan** — agent tylko do odczytu, do analizy i eksploracji kodu
  - Domyślnie odmawia edycji plików
  - Pyta o pozwolenie przed uruchomieniem poleceń bash
  - Idealny do eksploracji nieznanych baz kodu lub planowania zmian

Dostępny jest również subagent **general** do złożonych wyszukiwań i wieloetapowych zadań. Jest używany wewnętrznie i można go wywołać za pomocą `@general` w wiadomościach.

### Wsparcie dostawców

EmberHarmony działa z dowolnym API zgodnym z OpenAI. Wbudowane wsparcie dla:

| Dostawca | Modele | Wymagana konfiguracja |
|----------|--------|---------------|
| **Ollama (lokalnie)** | Automatycznie wykrywane z `localhost:11434` | Brak — wystarczy uruchomić Ollama |
| **Ollama Cloud** | Hostowane modele Ollama | Klucz API |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Klucz API |
| **OpenAI** | GPT-4o, o1, o3 | Klucz API |
| **Google** | Gemini Pro, Flash | Klucz API |
| **Dowolny zgodny z OpenAI** | LM Studio, vLLM, Together, Groq itp. | Punkt końcowy + klucz |

#### Lokalna konfiguracja Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Aby użyć niestandardowego adresu Ollama, dodaj do `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Wkład w projekt

Jeśli jesteś zainteresowany wniesieniem wkładu w EmberHarmony, prosimy o zapoznanie się z naszym [przewodnikiem dla współtwórców](./CONTRIBUTING.md) przed zgłoszeniem pull requesta.

### Budowanie na bazie EmberHarmony

Jeśli pracujesz nad projektem związanym z EmberHarmony, który używa nazwy „emberharmony" w swojej nazwie, prosimy o dodanie w swoim README adnotacji wyjaśniającej, że nie został on stworzony przez The Solace Project i nie jest z nami powiązany.

### Podziękowania

EmberHarmony to fork [opencode](https://github.com/anomalyco/opencode) autorstwa zespołu [SST](https://sst.dev). Jesteśmy głęboko wdzięczni za ich fundamentalną pracę przy budowie wyjątkowego otwartoźródłowego agenta AI do kodowania. Ten projekt opiera się na ich wizji i inżynierii.

### Opiekun

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Społeczność:** [Discord](https://discord.gg/EdF8f7JR) | [Zgłoszenia](https://github.com/SolaceHarmony/emberharmony/issues) | [Wydania](https://github.com/SolaceHarmony/emberharmony/releases)
