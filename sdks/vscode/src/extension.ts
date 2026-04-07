import * as vscode from "vscode"

const terminalName = "EmberHarmony"

export function activate(context: vscode.ExtensionContext) {
  const openNew = vscode.commands.registerCommand("emberharmony.openNewTerminal", async () => {
    await openTerminal()
  })

  const open = vscode.commands.registerCommand("emberharmony.openTerminal", async () => {
    const existing = vscode.window.terminals.find((t) => t.name === terminalName)
    if (existing) {
      existing.show()
      return
    }
    await openTerminal()
  })

  const addFile = vscode.commands.registerCommand("emberharmony.addFilepathToTerminal", async () => {
    const fileRef = getActiveFile()
    if (!fileRef) return

    const terminal = vscode.window.activeTerminal
    if (!terminal) return
    if (terminal.name !== terminalName) return

    // @ts-expect-error VS Code's terminal env typing isn't strong enough here.
    const env = terminal.creationOptions.env as Record<string, unknown> | undefined
    const portRaw =
      (typeof env?.["_EXTENSION_EMBERHARMONY_PORT"] === "string" && env?._EXTENSION_EMBERHARMONY_PORT) ||
      (typeof env?.["_EXTENSION_EMBERHARMONY_PORT"] === "string" && env?._EXTENSION_EMBERHARMONY_PORT)
    const port = portRaw ? parseInt(portRaw) : NaN

    if (!Number.isFinite(port)) {
      terminal.sendText(fileRef, false)
      terminal.show()
      return
    }

    await appendPrompt(port, fileRef)
    terminal.show()
  })

  context.subscriptions.push(openNew, open, addFile)

  async function openTerminal() {
    const port = Math.floor(Math.random() * (65535 - 16384 + 1)) + 16384
    const terminal = vscode.window.createTerminal({
      name: terminalName,
      iconPath: {
        light: vscode.Uri.file(context.asAbsolutePath("images/button-dark.svg")),
        dark: vscode.Uri.file(context.asAbsolutePath("images/button-light.svg")),
      },
      location: {
        viewColumn: vscode.ViewColumn.Beside,
        preserveFocus: false,
      },
      env: {
        _EXTENSION_EMBERHARMONY_PORT: port.toString(),
        // Backwards compatibility for older tooling/scripts.
        _EXTENSION_EMBERHARMONY_PORT: port.toString(),
        EMBERHARMONY_CALLER: "vscode",
        EMBERHARMONY_CALLER: "vscode",
      },
    })

    terminal.show()
    terminal.sendText(`emberharmony --port ${port}`)

    const fileRef = getActiveFile()
    if (!fileRef) return

    const connected = await (async () => {
      for (const _ of Array.from({ length: 10 })) {
        await new Promise((r) => setTimeout(r, 200))
        const ok = await fetch(`http://localhost:${port}/app`).then(
          () => true,
          () => false,
        )
        if (ok) return true
      }
      return false
    })()

    if (!connected) return

    await appendPrompt(port, `In ${fileRef}`)
    terminal.show()
  }

  async function appendPrompt(port: number, text: string) {
    await fetch(`http://localhost:${port}/tui/append-prompt`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    })
  }

  function getActiveFile() {
    const activeEditor = vscode.window.activeTextEditor
    if (!activeEditor) return

    const document = activeEditor.document
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri)
    if (!workspaceFolder) return

    const relativePath = vscode.workspace.asRelativePath(document.uri)
    const ref = `@${relativePath}`

    const selection = activeEditor.selection
    if (selection.isEmpty) return ref

    const startLine = selection.start.line + 1
    const endLine = selection.end.line + 1
    if (startLine === endLine) return `${ref}#L${startLine}`
    return `${ref}#L${startLine}-${endLine}`
  }
}

export function deactivate() {}
