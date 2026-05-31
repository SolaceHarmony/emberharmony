<p align="center">
  <a href="https://github.com/SolaceHarmony/emberharmony">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="Logo do EmberHarmony">
    </picture>
  </a>
</p>
<p align="center">O agente de programação com IA de código aberto.</p>
<p align="center">
  <a href="https://discord.gg/EdF8f7JR"><img alt="Discord" src="https://img.shields.io/discord/1391832426048651334?style=flat-square&label=discord" /></a>
  <a href="https://www.npmjs.com/package/@thesolaceproject/emberharmony"><img alt="npm" src="https://img.shields.io/npm/v/%40thesolaceproject%2Femberharmony?style=flat-square" /></a>
  <a href="https://github.com/SolaceHarmony/emberharmony/actions/workflows/ci.yml"><img alt="Status da build" src="https://img.shields.io/github/actions/workflow/status/SolaceHarmony/emberharmony/ci.yml?style=flat-square&branch=main" /></a>
</p>

[![Interface de Terminal do EmberHarmony](packages/web/src/assets/lander/screenshot.png)](https://github.com/SolaceHarmony/emberharmony)

---

## O que é o EmberHarmony?

O EmberHarmony é um agente de programação com IA de código aberto que roda no seu terminal. Ele é agnóstico em relação a provedores — use-o com Claude, OpenAI, Google, modelos locais via Ollama ou qualquer endpoint compatível com a OpenAI. Ele conta com uma TUI rica, suporte embutido a LSP e uma arquitetura cliente/servidor que permite controlá-lo remotamente.

### Modelos Locais sem Configuração

O EmberHarmony descobre automaticamente todos os modelos instalados na sua instância local do [Ollama](https://ollama.com). Sem chaves de API, sem arquivos de configuração, sem configuração manual. Se o Ollama estiver em execução, seus modelos aparecem:

```
ollama (custom): 19 models
  gemma3:latest           · 4.3B  · Q4_K_M
  llama3.2:latest         · 3.2B  · Q4_K_M
  deepseek-r1:14b         · 14.0B · Q4_K_M
  qwen3:8b                · 8.2B  · Q4_K_M
  ...
```

Alterne entre modelos na nuvem e locais no meio de uma conversa. Execute análises de código sensíveis inteiramente na sua máquina. Use modelos na nuvem quando precisar de capacidade de ponta, e modelos locais quando precisar de privacidade ou acesso offline.

**Principais diferenças em relação a outras ferramentas de programação com IA:**

- **Local em primeiro lugar** — descoberta automática de modelos Ollama, zero configuração, sem necessidade de chaves
- 100% código aberto (MIT)
- Não acoplado a nenhum provedor único — funciona com Claude, OpenAI, Google, Ollama e mais
- Suporte a LSP pronto para uso para navegação inteligente de código
- Interface de terminal rica que empurra os limites do que é possível no terminal
- Arquitetura cliente/servidor — execute na sua máquina, controle de qualquer lugar

### Instalação

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash

# npm / bun
npm i -g @thesolaceproject/emberharmony@latest
```

#### Build + Instalação Local

```bash
bun install
npm run pack:local
# prints a .tgz path you can install, e.g.
# npm i -g /absolute/path/to/emberharmony-1.2.2.tgz
```

### Aplicativo Desktop (Beta)

O EmberHarmony também está disponível como aplicativo desktop. Baixe diretamente da [página de releases](https://github.com/SolaceHarmony/emberharmony/releases).

| Plataforma            | Download                                  |
| --------------------- | ----------------------------------------- |
| macOS (Apple Silicon) | `emberharmony-desktop-darwin-aarch64.dmg` |
| macOS (Intel)         | `emberharmony-desktop-darwin-x64.dmg`     |
| Windows               | `emberharmony-desktop-windows-x64.exe`    |
| Linux                 | `.deb`, `.rpm`, ou AppImage               |

#### Diretório de Instalação

O script de instalação respeita a seguinte ordem de prioridade:

1. `$EMBERHARMONY_INSTALL_DIR` — diretório de instalação personalizado
2. `$XDG_BIN_DIR` — caminho compatível com a especificação XDG Base Directory
3. `$HOME/bin` — diretório binário padrão do usuário (se existir)
4. `$HOME/.emberharmony/bin` — fallback padrão

```bash
EMBERHARMONY_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/SolaceHarmony/emberharmony/dev/install | bash
```

### Agentes

O EmberHarmony inclui dois agentes integrados entre os quais você pode alternar com a tecla `Tab`.

- **build** — padrão, agente com acesso total para trabalho de desenvolvimento
- **plan** — agente somente leitura para análise e exploração de código
  - Nega edições de arquivos por padrão
  - Pede permissão antes de executar comandos bash
  - Ideal para explorar codebases desconhecidas ou planejar mudanças

Um subagente **general** também está disponível para buscas complexas e tarefas em várias etapas. Ele é usado internamente e pode ser invocado com `@general` nas mensagens.

### Suporte a Provedores

O EmberHarmony funciona com qualquer API compatível com a OpenAI. Suporte embutido para:

| Provedor | Modelos | Configuração necessária |
|----------|--------|---------------|
| **Ollama (local)** | Descoberto automaticamente em `localhost:11434` | Nenhuma — basta executar o Ollama |
| **Ollama Cloud** | Modelos Ollama hospedados | Chave de API |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Chave de API |
| **OpenAI** | GPT-4o, o1, o3 | Chave de API |
| **Google** | Gemini Pro, Flash | Chave de API |
| **Qualquer um compatível com a OpenAI** | LM Studio, vLLM, Together, Groq, etc. | Endpoint + chave |

#### Configuração Local do Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull llama3.2

# 3. Run EmberHarmony — models appear automatically
emberharmony
```

Para usar um endereço Ollama diferente do padrão, adicione a `~/.config/emberharmony/config.json`:
```json
{
  "provider": {
    "ollama": {
      "options": { "baseURL": "http://192.168.1.100:11434" }
    }
  }
}
```

### Contribuindo

Se você tiver interesse em contribuir com o EmberHarmony, leia nosso [guia de contribuição](./CONTRIBUTING.md) antes de enviar um pull request.

### Construindo sobre o EmberHarmony

Se você estiver trabalhando em um projeto relacionado ao EmberHarmony que use "emberharmony" em seu nome, adicione uma nota no seu README esclarecendo que ele não é construído pelo The Solace Project e não é afiliado a nós.

### Agradecimentos

O EmberHarmony é um fork do [opencode](https://github.com/anomalyco/opencode) criado pela equipe opencode upstream. Somos profundamente gratos pelo trabalho fundamental deles na construção de um agente de programação com IA de código aberto excepcional. Este projeto se baseia na visão e na engenharia deles.

### Mantenedora

**Sydney Renee** — [The Solace Project](https://github.com/SolaceHarmony)

---

**Comunidade:** [Discord](https://discord.gg/EdF8f7JR) | [Issues](https://github.com/SolaceHarmony/emberharmony/issues) | [Releases](https://github.com/SolaceHarmony/emberharmony/releases)
