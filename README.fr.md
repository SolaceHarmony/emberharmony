<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Logo EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">L'agent de codage IA open source.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Interface terminal d'EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## Qu'est-ce qu'EmberHarmony ?

EmberHarmony est un agent de codage IA open source qui s'exécute dans votre terminal. Il est agnostique vis-à-vis des fournisseurs — utilisez-le avec Claude, OpenAI, Google, des modèles locaux via Ollama, ou tout point de terminaison compatible OpenAI. Il offre une TUI riche, une prise en charge intégrée de LSP, et une architecture client/serveur qui vous permet de le piloter à distance.

### Modèles locaux sans configuration

EmberHarmony découvre automatiquement chaque modèle installé dans votre instance locale [Ollama](https://ollama.com). Aucune clé API, aucun fichier de configuration, aucune installation manuelle. Si Ollama est en cours d'exécution, vos modèles apparaissent :

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Basculez entre modèles cloud et locaux en pleine conversation. Effectuez des analyses de code sensibles entièrement sur votre machine. Utilisez les modèles cloud quand vous avez besoin de capacités de pointe, et les modèles locaux quand vous avez besoin de confidentialité ou d'un accès hors ligne.

**Principales différences avec les autres outils de codage IA :**

- **Local d'abord** — découverte automatique des modèles Ollama, zéro configuration, aucune clé nécessaire
- 100 % open source (MIT)
- Non couplé à un fournisseur unique — fonctionne avec Claude, OpenAI, Google, Ollama, et plus encore
- Prise en charge LSP immédiate pour une navigation intelligente dans le code
- Interface terminal riche qui repousse les limites de ce qui est possible dans le terminal
- Architecture client/serveur — exécutez sur votre machine, pilotez depuis n'importe où

### Installation

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Build local + installation

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Application de bureau (Bêta)

EmberHarmony est également disponible sous forme d'application de bureau. Téléchargez-la directement depuis la [page des releases](https://github.com/SolaceHarmony/emberharmony/releases).

| Plateforme            | Téléchargement                            |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, ou AppImage               |

#### Répertoire d'installation

Le script d'installation respecte l'ordre de priorité suivant :

1. `$EMBERHARMONY_INSTALL_DIR` — répertoire d'installation personnalisé
2. `$XDG_BIN_DIR` — chemin conforme à la spécification XDG Base Directory
3. `$HOME/bin` — répertoire binaire utilisateur standard (s'il existe)
4. `$HOME/.emberharmony/bin` — solution de repli par défaut

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agents

EmberHarmony inclut deux agents intégrés entre lesquels vous pouvez basculer avec la touche `Tab`.

- **build** — agent par défaut à accès complet pour le travail de développement
- **plan** — agent en lecture seule pour l'analyse et l'exploration du code
  - Refuse les modifications de fichiers par défaut
  - Demande la permission avant d'exécuter des commandes bash
  - Idéal pour explorer des bases de code inconnues ou planifier des changements

Un sous-agent **general** est également disponible pour les recherches complexes et les tâches multi-étapes. Il est utilisé en interne et peut être invoqué avec `@general` dans les messages.

### Prise en charge des fournisseurs

EmberHarmony fonctionne avec toute API compatible OpenAI. Prise en charge intégrée pour :

| Fournisseur | Modèles | Configuration requise |
|----------|--------|---------------|
| **Ollama (local)** | Découverts automatiquement depuis `localhost:11434` | Aucune — lancez simplement Ollama |
| **Ollama Cloud** | Modèles Ollama hébergés | Clé API |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Clé API |
| **OpenAI** | GPT-4o, o1, o3 | Clé API |
| **Google** | Gemini Pro, Flash | Clé API |
| **Tout fournisseur compatible OpenAI** | LM Studio, vLLM, Together, Groq, etc. | Point de terminaison + clé |

#### Configuration locale d'Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Pour utiliser une adresse Ollama non par défaut, ajoutez ceci à `~/.config/emberharmony/config.json` :
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Contribuer

Si vous souhaitez contribuer à EmberHarmony, veuillez lire notre [guide de contribution](./CONTRIBUTING.md) avant de soumettre une pull request.

### Construire sur EmberHarmony

Si vous travaillez sur un projet lié à EmberHarmony qui utilise « emberharmony » dans son nom, veuillez ajouter une note dans votre README précisant qu'il n'est pas développé par The Solace Project et qu'il n'est pas affilié à nous.

### Remerciements

EmberHarmony est un fork d'[opencode](https://github.com/anomalyco/opencode) par l'équipe [SST](https://sst.dev). Nous sommes profondément reconnaissants pour leur travail fondateur dans la construction d'un agent de codage IA open source exceptionnel. Ce projet s'appuie sur leur vision et leur ingénierie.

### Mainteneuse

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Communauté :** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
