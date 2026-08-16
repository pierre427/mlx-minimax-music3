#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3 MCP server (loopback streamable-HTTP, stdlib only).

Exposes the local Music3 HTTP service (default :8600) as MCP tools so an agent
(Hermes / Codex / Claude) can compose music. Mirrors the project's other MCP
servers (web_tools :8769/mcp): a JSON-RPC endpoint at POST /mcp.

Tools
    generate_music(caption, lyrics?, duration_seconds?, seed?, out_path?)
        -> generates an MP3 via the Music3 API, saves it, returns the path + metadata.
    music_style_guide()
        -> how to write a strong MiniMax-Music3 caption (structured format + rules).

    .venv/bin/python minimax-music3-mlx/scripts/mcp_server.py --port 8770 --api http://127.0.0.1:8600
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

from minimax_music3_mlx.constants import MAX_AUDIO_FRAMES  # noqa: E402
from minimax_music3_mlx.duration import extract_duration_from_text, parse_duration_seconds  # noqa: E402

PROTOCOL = "2025-06-18"
API = os.environ.get("MUSIC3_API", "http://127.0.0.1:8600")
OUT_DIR = Path(os.environ.get("MUSIC3_OUT_DIR", str(Path.home() / "Music" / "mlx-music3")))
JOB_STORE = Path(os.environ.get(
    "MUSIC3_JOB_STORE",
    str(Path(__file__).resolve().parents[1] / "state" / "music3_jobs.json"),
))
MAX_RETAINED_JOBS = 64
_jobs: dict[str, dict] = {}
_jobs_lock = threading.RLock()
_job_queue: queue.Queue[str] = queue.Queue()

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
    "- Fixed sampling: no temperature/top_p. Change `seed` for alternate takes. The production "
    "decoder always uses the 9,000-frame safety ceiling.\n"
    "- If the user requests a length, pass `duration_seconds` (for example 270 for 4:30). The "
    "service injects that target into the structured caption and prevents early model EOS before "
    "95% of it. Music3 supports requested targets up to five minutes.\n"
    "- The service always expands even a brief caption into this rich format when the rewriter LLM "
    "is available; explicit constraints and duration are preserved."
)


def _normalize_generate_args(args: dict) -> dict:
    normalized = dict(args)
    if "max_frames" in normalized and int(normalized["max_frames"]) != MAX_AUDIO_FRAMES:
        raise ValueError(
            "max_frames no longer requests song length and is fixed at 9000; "
            "pass duration_seconds instead (for example 270 for 4:30)"
        )
    normalized.pop("max_frames", None)
    raw_duration = normalized.pop("target_duration", normalized.get("duration_seconds"))
    if raw_duration is None:
        raw_duration = extract_duration_from_text(normalized.get("caption", ""))
    duration_seconds = parse_duration_seconds(raw_duration)
    if duration_seconds is None:
        normalized.pop("duration_seconds", None)
    else:
        normalized["duration_seconds"] = duration_seconds
    return normalized


def _api_generate(args: dict) -> dict:
    args = _normalize_generate_args(args)
    caption = args.get("caption", "")
    if not caption:
        raise ValueError("caption is required")
    body = json.dumps({
        "caption": caption, "lyrics": args.get("lyrics", ""),
        "seed": int(args.get("seed", 0)), "max_frames": MAX_AUDIO_FRAMES,
        "target_seconds": args.get("duration_seconds"),
        "rewrite": True,
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


def _now() -> float:
    return round(time.time(), 3)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _persist_jobs_locked() -> None:
    # Retention is metadata-only: generated audio is never removed by the MCP.
    terminal = sorted(
        (job for job in _jobs.values() if job["status"] in {"succeeded", "failed"}),
        key=lambda job: job.get("finished_at", 0),
        reverse=True,
    )
    retained_ids = {job["job_id"] for job in terminal[:MAX_RETAINED_JOBS]}
    retained_ids.update(job_id for job_id, job in _jobs.items()
                        if job["status"] in {"queued", "running"})
    for job_id in tuple(_jobs):
        if job_id not in retained_ids:
            del _jobs[job_id]
    _atomic_write_json(JOB_STORE, {"version": 1, "jobs": _jobs})


def _load_jobs() -> None:
    with _jobs_lock:
        _jobs.clear()
        try:
            saved = json.loads(JOB_STORE.read_text())
            jobs = saved.get("jobs", {})
            if not isinstance(jobs, dict):
                raise ValueError("jobs must be an object")
        except FileNotFoundError:
            return
        except Exception as exc:
            # Preserve a corrupt store for diagnosis and start with no phantom jobs.
            print(f"[music3-mcp] ignoring unreadable job store: {exc}", flush=True)
            return
        _jobs.update(jobs)
        recovered = False
        for job in _jobs.values():
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="failed",
                    finished_at=_now(),
                    error={
                        "code": "mcp_restarted_before_completion",
                        "message": "Music3 MCP restarted before this job completed; the output state is unknown.",
                    },
                )
                recovered = True
        if recovered:
            _persist_jobs_locked()


def _public_job(job: dict) -> dict:
    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }
    if job["status"] == "succeeded":
        result["result"] = job["result"]
    elif job["status"] == "failed":
        result["error"] = job["error"]
    else:
        result["next_action"] = (
            "Generation is still in progress. Do not retry generate_music. "
            "Call get_music_job later with this job_id."
        )
    return result


def _submit_job(args: dict) -> dict:
    args = _normalize_generate_args(args)
    if not args.get("caption", ""):
        raise ValueError("caption is required")
    job_id = f"music_{uuid.uuid4().hex}"
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "submitted_at": _now(),
            # Persist only the operational request. Do not expose caption/lyrics in status responses.
            "request": dict(args),
        }
        _persist_jobs_locked()
    _job_queue.put(job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "next_action": (
            "Music generation has been queued. Reply to the user that it is running; "
            "call get_music_job with this job_id only when they ask for progress or the result."
        ),
    }


def _job_worker() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None or job["status"] != "queued":
                    continue
                job.update(status="running", started_at=_now())
                _persist_jobs_locked()
                args = dict(job["request"])
            try:
                result = _api_generate(args)
            except Exception as exc:
                error = {"code": "generation_failed", "message": str(exc)}
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        job.update(status="failed", finished_at=_now(), error=error)
                        _persist_jobs_locked()
                print(f"[music3-mcp] job {job_id} failed: {exc}", flush=True)
            else:
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        job.update(status="succeeded", finished_at=_now(), result=result)
                        _persist_jobs_locked()
                print(f"[music3-mcp] job {job_id} succeeded: {result['path']}", flush=True)
        finally:
            _job_queue.task_done()


TOOLS = [
    {"name": "generate_music",
     "description": "Queue MiniMax-Music3 generation and return a durable job_id immediately. This "
                    "does not wait for rendering. Tell the user generation is running; use "
                    "get_music_job later with job_id to retrieve the saved MP3 path or failure. "
                    "A brief user description is welcome: the service automatically expands it into "
                    "a detailed structured caption while preserving explicit constraints and requested "
                    "duration. Use music_style_guide first only when planning help is useful.",
     "inputSchema": {"type": "object", "required": ["caption"], "properties": {
         "caption": {"type": "string", "description": "The user's musical request. It may be brief; "
                     "the caption-rewriter expands genre, mood, arrangement, instruments, vocals, and "
                     "production while preserving explicit details."},
         "lyrics": {"type": "string", "description": "Optional lyrics; put each [tag] alone on its line."},
         "seed": {"type": "integer", "default": 0, "description": "Change for alternate takes."},
         "duration_seconds": {"type": "number", "minimum": 1, "maximum": 300,
                              "description": "Requested song length in seconds; use 270 for 4:30. "
                                             "The service expands the caption around this target and "
                                             "enforces a 95% minimum. Omit only when the model should "
                                             "choose the length."},
         "out_path": {"type": "string", "description": "Optional output .mp3 path."}}}},
    {"name": "get_music_job",
     "description": "Get the durable status/result of a previously queued Music3 job. A succeeded "
                    "job contains the saved MP3 path and generation metadata. For queued/running jobs, "
                    "do not retry generation; report that it is still running.",
     "inputSchema": {"type": "object", "required": ["job_id"], "properties": {
         "job_id": {"type": "string", "description": "job_id returned by generate_music."}}}},
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
                text = json.dumps(_submit_job(args), indent=2)
            elif name.endswith("get_music_job"):
                job_id = args.get("job_id", "")
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    text = json.dumps(
                        _public_job(job) if job else {
                            "error": {
                                "code": "unknown_job",
                                "message": f"No retained Music3 job exists for job_id {job_id!r}.",
                            }
                        },
                        indent=2,
                    )
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
    _load_jobs()
    threading.Thread(target=_job_worker, name="music3-job-worker", daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[music3-mcp] serving MCP on http://{args.host}:{args.port}/mcp -> API {API}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
