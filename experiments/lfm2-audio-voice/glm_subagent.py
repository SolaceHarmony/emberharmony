#!/usr/bin/env python
"""
GLM subagent — the "hard work" delegate for the LFM2-Audio local voice agent.

The architecture (see README.md): a small, local LFM2-Audio model is the
conversational front. Its ONE non-trivial tool is to hand a task to this
subagent, which runs GLM-5.1 (via the ollama-cloud OpenAI-compatible API) to do
the actual work and return a short, speakable result.

This module is deliberately standalone: no LiveKit, no EmberHarmony session
bridge. It is "just another model" reached over an API, with a primitive tool
loop (a single `bash` tool) so GLM can actually do work when allowed.

Auth: reads the ollama-cloud API key from $OLLAMA_API_KEY, else from
~/.local/share/emberharmony/auth.json ("ollama-cloud".key). The key is never
logged or printed.

Safety: bash execution is OFF by default. Pass allow_exec=True (CLI: --exec) to
let the subagent run shell commands. Without it, the subagent reasons/answers
but cannot touch the machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("GLM_BASE_URL", "https://ollama.com/v1")
MODEL = os.environ.get("GLM_MODEL", "glm-5.1")
AUTH_JSON = Path(os.environ.get("EMBERHARMONY_AUTH", Path.home() / ".local/share/emberharmony/auth.json"))

SUBAGENT_SYSTEM = (
    "You are a capable engineering subagent invoked by a voice assistant to do the hard work. "
    "You receive a task, do it, and return a SHORT, plain-spoken summary the voice assistant can read "
    "aloud — no markdown, no code fences, two or three sentences at most. "
    "If you have the bash tool, use it to actually inspect or change things; if you do not, reason it through "
    "and give the best concrete answer or plan. State clearly what you did or found."
)


def _load_key() -> str:
    """ollama-cloud API key from env or the EmberHarmony auth store. Never printed."""
    key = os.environ.get("OLLAMA_API_KEY")
    if key:
        return key.strip()
    try:
        data = json.loads(AUTH_JSON.read_text())
    except FileNotFoundError as e:
        raise RuntimeError(
            f"no OLLAMA_API_KEY set and auth store not found at {AUTH_JSON}"
        ) from e
    entry = data.get("ollama-cloud") or data.get("ollama_cloud")
    if not isinstance(entry, dict) or not entry.get("key"):
        raise RuntimeError(f"no ollama-cloud key found in {AUTH_JSON}")
    return str(entry["key"]).strip()


def _chat(messages: list[dict], tools: list[dict] | None = None, timeout: float = 120.0) -> dict:
    """One OpenAI-compatible chat-completions call to ollama-cloud. Returns the raw message dict."""
    body: dict = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {_load_key()}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"GLM API HTTP {e.code}: {detail}") from e
    choice = payload.get("choices", [{}])[0]
    return choice.get("message", {})


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and get back stdout/stderr. Use for inspecting or changing files, running tests, git, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"},
            },
            "required": ["command"],
        },
    },
}


def _run_bash(command: str, cwd: str, timeout: float = 120.0) -> str:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > 6000:
        out = out[:6000] + "\n…[truncated]"
    return f"[exit {proc.returncode}]\n{out}".strip()


def run_subagent(
    task: str,
    *,
    allow_exec: bool = False,
    cwd: str | None = None,
    max_steps: int = 8,
    verbose: bool = False,
) -> str:
    """Run GLM on `task` and return a short, speakable result string."""
    cwd = cwd or os.getcwd()
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": task},
    ]
    tools = [BASH_TOOL] if allow_exec else None

    for _ in range(max_steps):
        msg = _chat(messages, tools=tools)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip() or "(the subagent returned nothing)"
        # Echo the assistant turn (with its tool calls) back into the history.
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "bash" and allow_exec:
                if verbose:
                    print(f"  [subagent bash] {args.get('command','')}", file=sys.stderr)
                result = _run_bash(args.get("command", ""), cwd=cwd)
            else:
                result = f"tool '{name}' is not available (execution disabled)"
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
    return "(the subagent hit its step limit without finishing)"


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--exec"]
    allow_exec = "--exec" in sys.argv
    if not args:
        print('usage: python glm_subagent.py [--exec] "the task"', file=sys.stderr)
        return 2
    task = " ".join(args)
    print(f"[subagent] model={MODEL} exec={allow_exec}", file=sys.stderr)
    print(run_subagent(task, allow_exec=allow_exec, verbose=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
