#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3 MCP server (loopback streamable-HTTP, stdlib only).

Exposes the local Music3 HTTP service (default :8600) as MCP tools so an agent
(Hermes / Codex / Claude) can compose music. Mirrors the project's other MCP
servers (web_tools :8769/mcp): a JSON-RPC endpoint at POST /mcp.

Tools
    generate_music(caption, lyrics?, seed?, max_frames?, out_path?)
        -> generates an MP3 via the Music3 API, saves it, returns the path + metadata.
    music_style_guide()
        -> how to write a strong MiniMax-Music3 caption (structured format + rules).

    .venv/bin/python minimax-music3-mlx/scripts/mcp_server.py --port 8770 --api http://127.0.0.1:8600
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROTOCOL = "2025-06-18"
API = os.environ.get("MUSIC3_API", "http://127.0.0.1:8600")
OUT_DIR = Path(os.environ.get("MUSIC3_OUT_DIR", str(Path.home() / "Music" / "mlx-music3")))

STYLE_GUIDE = (
    "MiniMax-Music3 caption guide (write the `caption`; put lyrics in `lyrics`):\n"
    "- Describe: genre/subgenre, tempo (bpm) + key, mood arc, instrumentation (primary + "
    "secondary), vocal gender/timbre/style, and production/sonics. Richer captions sound better.\n"
    "- A good shape: 'Global Metadata' (bpm, key, genre, mood) then 'Arrangement' (primary/secondary "
    "instruments) then 'Vocal Details' (gender, timbre, delivery).\n"
    "- Lyrics: put each structure tag ALONE on its own line, e.g. '[verse]\\nline one\\n[chorus]\\n...'. "
    "A tag on the same line as lyric text silently drops that line.\n"
    "- Instrumental: ask for it in the caption and keep lyrics minimal but non-empty, e.g. "
    "'[intro]\\n(instrumental)'.\n"
    "- Fixed sampling: no temperature/top_p. Change `seed` for alternate takes. 25 fps; max_frames "
    "caps length (300 ~= 12s; up to 9000 ~= 6 min).\n"
    "- The service auto-rewrites brief captions into this rich format when a rewriter LLM is configured."
)


def _api_generate(args: dict) -> dict:
    caption = args.get("caption", "")
    if not caption:
        raise ValueError("caption is required")
    body = json.dumps({
        "caption": caption, "lyrics": args.get("lyrics", ""),
        "seed": int(args.get("seed", 0)), "max_frames": int(args.get("max_frames", 300)),
        "rewrite": bool(args.get("rewrite", True)),
    }).encode()
    req = urllib.request.Request(API.rstrip("/") + "/v1/audio/music", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        audio = r.read()
        meta = {k[len("X-Music3-"):]: v for k, v in r.headers.items() if k.startswith("X-Music3-")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args["out_path"]) if args.get("out_path") else OUT_DIR / f"music_{int(time.time())}_seed{args.get('seed',0)}.mp3"
    out.write_bytes(audio)
    return {"path": str(out), **meta}


TOOLS = [
    {"name": "generate_music",
     "description": "Generate music (MiniMax-Music3) from a text caption and optional tagged lyrics. "
                    "Returns the path to a saved MP3. Use music_style_guide first if unsure how to "
                    "write the caption.",
     "inputSchema": {"type": "object", "required": ["caption"], "properties": {
         "caption": {"type": "string", "description": "Rich description: genre, bpm, key, mood, "
                     "instruments, vocals, production."},
         "lyrics": {"type": "string", "description": "Optional lyrics; put each [tag] alone on its line."},
         "seed": {"type": "integer", "default": 0, "description": "Change for alternate takes."},
         "max_frames": {"type": "integer", "default": 300, "description": "Length cap (25 fps; 300~=12s)."},
         "out_path": {"type": "string", "description": "Optional output .mp3 path."}}}},
    {"name": "music_style_guide",
     "description": "How to write an effective MiniMax-Music3 caption (structured format + rules).",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _handle(msg: dict):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "minimax-music3", "version": "1.0.0"}}}
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None  # notification, no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        name = (params.get("name") or "")
        args = params.get("arguments", {}) or {}
        print(f"[music3-mcp] tools/call name={name!r} args_keys={list(args)}", flush=True)
        try:
            # tolerate namespaced tool names (e.g. mcp__music3__generate_music)
            if name.endswith("music_style_guide"):
                text = STYLE_GUIDE
            elif name.endswith("generate_music"):
                text = json.dumps(_api_generate(args), indent=2)
            else:
                raise ValueError(f"unknown tool: {name}")
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method {method} not found"}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/mcp"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "server": "minimax-music3-mcp",
                                         "api": API, "tools": [t["name"] for t in TOOLS]}).encode())
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") != "/mcp":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode()); return
        resp = _handle(msg)
        if resp is None:  # notification
            self.send_response(202); self.end_headers(); return
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    global API
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--api", default=API)
    args = ap.parse_args()
    API = args.api
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[music3-mcp] serving MCP on http://{args.host}:{args.port}/mcp -> API {API}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
