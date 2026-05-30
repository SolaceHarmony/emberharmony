<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Logo de EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">El agente de codificación con IA de código abierto.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Estado de la compilación" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Interfaz de terminal de EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## ¿Qué es EmberHarmony?

EmberHarmony es un agente de codificación con IA de código abierto que se ejecuta en tu terminal. Es independiente del proveedor: úsalo con Claude, OpenAI, Google, modelos locales mediante Ollama o cualquier endpoint compatible con OpenAI. Cuenta con una TUI rica, soporte integrado para LSP y una arquitectura cliente/servidor que te permite controlarlo de forma remota.

### Modelos locales sin configuración

EmberHarmony descubre automáticamente cada modelo instalado en tu instancia local de [Ollama](https://ollama.com). Sin claves de API, sin archivos de configuración, sin instalación manual. Si Ollama está en ejecución, tus modelos aparecen:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Cambia entre modelos en la nube y locales en medio de una conversación. Ejecuta análisis de código sensible enteramente en tu máquina. Usa modelos en la nube cuando necesites capacidad de vanguardia, y modelos locales cuando necesites privacidad o acceso sin conexión.

**Diferencias clave frente a otras herramientas de codificación con IA:**

- **Local primero** — descubrimiento automático de modelos de Ollama, cero configuración, sin necesidad de claves
- 100 % de código abierto (MIT)
- No está acoplado a ningún proveedor único — funciona con Claude, OpenAI, Google, Ollama y más
- Soporte LSP listo para usar para una navegación de código inteligente
- Interfaz de terminal rica que lleva al límite lo posible en la terminal
- Arquitectura cliente/servidor — ejecútalo en tu máquina y contrólalo desde cualquier lugar

### Instalación

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Compilación + instalación local

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Aplicación de escritorio (Beta)

EmberHarmony también está disponible como aplicación de escritorio. Descárgala directamente desde la [página de versiones](https://github.com/SolaceHarmony/emberharmony/releases).

| Plataforma            | Descarga                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, o AppImage                |

#### Directorio de instalación

El script de instalación respeta el siguiente orden de prioridad:

1. `$EMBERHARMONY_INSTALL_DIR` — directorio de instalación personalizado
2. `$XDG_BIN_DIR` — ruta conforme al estándar XDG Base Directory
3. `$HOME/bin` — directorio estándar de binarios del usuario (si existe)
4. `$HOME/.emberharmony/bin` — alternativa predeterminada

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agentes

EmberHarmony incluye dos agentes integrados entre los que puedes alternar con la tecla `Tab`.

- **build** — predeterminado, agente con acceso completo para el trabajo de desarrollo
- **plan** — agente de solo lectura para análisis y exploración de código
  - Deniega las ediciones de archivos de forma predeterminada
  - Pide permiso antes de ejecutar comandos de bash
  - Ideal para explorar bases de código desconocidas o planificar cambios

También está disponible un subagente **general** para búsquedas complejas y tareas de varios pasos. Se usa internamente y puede invocarse con `@general` en los mensajes.

### Compatibilidad con proveedores

EmberHarmony funciona con cualquier API compatible con OpenAI. Soporte integrado para:

| Proveedor | Modelos | Configuración necesaria |
|----------|--------|---------------|
| **Ollama (local)** | Descubiertos automáticamente desde `localhost:11434` | Ninguna — solo ejecuta Ollama |
| **Ollama Cloud** | Modelos de Ollama alojados | Clave de API |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Clave de API |
| **OpenAI** | GPT-4o, o1, o3 | Clave de API |
| **Google** | Gemini Pro, Flash | Clave de API |
| **Cualquiera compatible con OpenAI** | LM Studio, vLLM, Together, Groq, etc. | Endpoint + clave |

#### Configuración local de Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Para usar una dirección de Ollama no predeterminada, añade a `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Contribuir

Si te interesa contribuir a EmberHarmony, lee nuestra [guía de contribución](./CONTRIBUTING.md) antes de enviar un pull request.

### Desarrollar sobre EmberHarmony

Si estás trabajando en un proyecto relacionado con EmberHarmony que usa "emberharmony" en su nombre, añade una nota en tu README aclarando que no está desarrollado por The Solace Project y que no está afiliado con nosotros.

### Agradecimientos

EmberHarmony es un fork de [opencode](https://github.com/anomalyco/opencode) del equipo de [SST](https://sst.dev). Estamos profundamente agradecidos por su trabajo fundacional al construir un agente de codificación con IA de código abierto excepcional. Este proyecto se basa en su visión y su ingeniería.

### Responsable del mantenimiento

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Comunidad:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Versiones](https://github.com/SolaceHarmony/emberharmony/releases)
