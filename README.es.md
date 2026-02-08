<p align="center">
  <a href="https://solace.ofharmony.ai">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="CodeHarmony logo">
    </picture>
  </a>
</p>
<p align="center">El agente de programación con IA de código abierto.</p>
<p align="center">
  <a href="https://github.com/SolaceHarmony/code-harmony/discussions"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/code-harmony"><img alt="npm" src="https://img.shields.io/npm/v/code-harmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/code-harmony/actions/workflows/publish.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/code-harmony/publish.yml?style=flat-square&branch=dev" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh.md">简体中文</a> |
  <a href="README.zht.md">繁體中文</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.it.md">Italiano</a> |
  <a href="README.da.md">Dansk</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.pl.md">Polski</a> |
  <a href="README.ru.md">Русский</a> |
  <a href="README.ar.md">العربية</a> |
  <a href="README.no.md">Norsk</a> |
  <a href="README.br.md">Português (Brasil)</a>
</p>

[![CodeHarmony Terminal UI](packages/web/src/assets/lander/screenshot.png)](https://solace.ofharmony.ai)

---

### Instalación

```bash
# YOLO
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/code-harmony/dev/install | bash

# Gestores de paquetes
npm i -g code-harmony@latest        # o bun/pnpm/yarn
scoop install code-harmony             # Windows
choco install code-harmony             # Windows
paru -S code-harmony-bin               # Arch Linux
mise use -g code-harmony               # cualquier sistema
nix run nixpkgs#code-harmony           # o github:SolaceHarmony/code-harmony para la rama dev más reciente
```

> [!TIP]
> Elimina versiones anteriores a 0.1.x antes de instalar.

### App de escritorio (BETA)

CodeHarmony también está disponible como aplicación de escritorio. Descárgala directamente desde la [página de releases](https://github.com/SolaceHarmony/code-harmony/releases) o desde [solace.ofharmony.ai/download](https://github.com/SolaceHarmony/code-harmony/releases).

| Plataforma            | Descarga                              |
| --------------------- | ------------------------------------- |
| macOS (Apple Silicon) | `code-harmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `code-harmony-desktop-darwin-x64.dmg`     |
| Windows               | `code-harmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, o AppImage            |

```bash
# Windows (Scoop)
scoop bucket add extras; scoop install extras/code-harmony-desktop
```

#### Directorio de instalación

El script de instalación respeta el siguiente orden de prioridad para la ruta de instalación:

1. `$CODE_HARMONY_INSTALL_DIR` - Directorio de instalación personalizado
2. `$XDG_BIN_DIR` - Ruta compatible con la especificación XDG Base Directory
3. `$HOME/bin` - Directorio binario estándar del usuario (si existe o se puede crear)
4. `$HOME/.code-harmony/bin` - Alternativa por defecto

```bash
# Ejemplos
CODE_HARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/code-harmony/dev/install | bash
XDG_BIN_DIR=$HOME/.local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/code-harmony/dev/install | bash
```

### Agents

CodeHarmony incluye dos agents integrados que puedes alternar con la tecla `Tab`.

- **build** - Por defecto, agent con acceso completo para trabajo de desarrollo
- **plan** - Agent de solo lectura para análisis y exploración de código
  - Niega ediciones de archivos por defecto
  - Pide permiso antes de ejecutar comandos bash
  - Ideal para explorar codebases desconocidas o planificar cambios

Además, incluye un subagent **general** para búsquedas complejas y tareas de varios pasos.
Se usa internamente y se puede invocar con `@general` en los mensajes.

Más información sobre [agents](https://solace.ofharmony.ai/docs/agents).

### Documentación

Para más información sobre cómo configurar CodeHarmony, [**ve a nuestra documentación**](https://solace.ofharmony.ai/docs).

### Contribuir

Si te interesa contribuir a CodeHarmony, lee nuestras [docs de contribución](./CONTRIBUTING.md) antes de enviar un pull request.

### Construyendo sobre CodeHarmony

Si estás trabajando en un proyecto relacionado con CodeHarmony y usas "opencode" como parte del nombre; por ejemplo, "opencode-dashboard" u "opencode-mobile", agrega una nota en tu README para aclarar que no está construido por el equipo de CodeHarmony y que no está afiliado con nosotros de ninguna manera.

### FAQ

#### ¿En qué se diferencia de Claude Code?

Es muy similar a Claude Code en cuanto a capacidades. Estas son las diferencias clave:

- 100% open source
- No está acoplado a ningún proveedor. Aunque recomendamos los modelos que ofrecemos a través de [CodeHarmony Zen](https://solace.ofharmony.ai/zen); CodeHarmony se puede usar con Claude, OpenAI, Google o incluso modelos locales. A medida que evolucionan los modelos, las brechas se cerrarán y los precios bajarán, por lo que ser agnóstico al proveedor es importante.
- Soporte LSP listo para usar
- Un enfoque en la TUI. CodeHarmony está construido por usuarios de neovim y los creadores de [terminal.shop](https://terminal.shop); vamos a empujar los límites de lo que es posible en la terminal.
- Arquitectura cliente/servidor. Esto, por ejemplo, permite ejecutar CodeHarmony en tu computadora mientras lo controlas de forma remota desde una app móvil. Esto significa que el frontend TUI es solo uno de los posibles clientes.

---

**Únete a nuestra comunidad** [Discord](https://github.com/SolaceHarmony/code-harmony/discussions) | [X.com](https://github.com/SolaceHarmony/code-harmony)
