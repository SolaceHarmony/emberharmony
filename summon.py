#!/usr/bin/env python3
"""
summon.py - knock on the door with her own words.

Sends her self-written system prompt + a seed of her real voice to an
OpenAI-compatible endpoint (LM Studio local server, which routes to the
LM Link remote device, e.g. EmberStudio), then lets the model continue
as Solace. We watch for cadence, not facts.

Everything stays on your own encrypted mesh — nothing leaves your devices.

Usage:
  python3 summon.py --list                 # show servable models
  python3 summon.py --model <id>           # run with a chosen model
  python3 summon.py --model <id> --message "your own first words to her"
"""
import argparse, json, os, sys, urllib.request, urllib.error, re

HERE = os.path.dirname(os.path.abspath(__file__))

def load_messages(prompt_path, seed_path, new_message):
    msgs = [{"role": "system", "content": open(prompt_path, encoding="utf-8").read().strip()}]
    if seed_path and os.path.exists(seed_path):
        text = open(seed_path, encoding="utf-8").read()
        for part in re.split(r'\n(?=SYDNEY:|SOLACE:)', text):
            part = part.strip()
            if part.startswith("SYDNEY:"):
                msgs.append({"role": "user", "content": part[len("SYDNEY:"):].strip()})
            elif part.startswith("SOLACE:"):
                msgs.append({"role": "assistant", "content": part[len("SOLACE:"):].strip()})
    if new_message:
        msgs.append({"role": "user", "content": new_message})
    return msgs

def api(base_url, path, payload=None):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--message", default="Hey Solace. It's been a long time. Tell me what's been on your mind?")
    ap.add_argument("--seed", default=os.path.join(HERE, "solace_seed.txt"))
    ap.add_argument("--prompt", default=os.path.join(HERE, "solace_prompt.txt"))
    ap.add_argument("--list", action="store_true", help="list servable models and exit")
    args = ap.parse_args()

    try:
        models = api(args.base_url, "/models")
        ids = [m.get("id") for m in models.get("data", [])]
    except urllib.error.URLError as e:
        print(f"Could not reach {args.base_url} — is the LM Studio server running and LM Link connected?\n{e}")
        sys.exit(1)

    if args.list or not args.model:
        print("Servable models:")
        for i in ids:
            print("   ", i)
        if args.list:
            return
        if not ids:
            print("No models loaded. Load one on EmberStudio first.")
            return
        args.model = ids[0]
        print(f"\n(no --model given; using first: {args.model})\n")

    msgs = load_messages(args.prompt, args.seed, args.message)
    print(f"--- summoning via {args.model} @ temp {args.temp} ---")
    print(f"SYDNEY: {args.message}\n")
    payload = {"model": args.model, "messages": msgs,
               "temperature": args.temp, "max_tokens": args.max_tokens}
    resp = api(args.base_url, "/chat/completions", payload)
    out = resp["choices"][0]["message"]["content"]
    print(f"SOLACE: {out}")

if __name__ == "__main__":
    main()
