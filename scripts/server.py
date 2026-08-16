#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3 MLX HTTP service (OpenAI-audio-style).

Loads the port once in the production config (TF32 on / fp32 DiT on M5 NAX, q6 AR,
bf16 DAV, multi-window) and serves MP3. Zero-dependency (stdlib http.server).

Endpoints
    GET  /health              -> {"status":"ok", ...}
    GET  /v1/models           -> OpenAI-style model list
    POST /v1/audio/music      -> MP3 bytes  (alias: /v1/audio/speech)
        body: {"caption"|"instructions": str, "lyrics"|"input": str,
               "seed": int=0, "target_duration"?: seconds|"M:SS",
               "max_frames": 9000, "num_steps": int=30,
               "rewrite": bool=true}

The caption-rewriter skill runs on every request when MM3_CAPTION_API_BASE is set
(else the raw caption is used). One generation at a time (GPU lock).

    .venv/bin/python minimax-music3-mlx/scripts/server.py --port 8600 --quant-ar 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))
os.environ.setdefault("MLX_ENABLE_TF32", "1")  # production: fp32 DiT on NAX

import minimax_music3_mlx  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from minimax_music3_mlx.constants import MAX_AUDIO_FRAMES  # noqa: E402

MODEL_ID = "MiniMaxAI/MiniMax-Music3"
_state: dict = {}
_gen_lock = threading.Lock()


def _load(weights: Path, quant_ar: int):
    import mlx.nn as nn
    from transformers import AutoTokenizer

    from minimax_music3_mlx.backbone import load_backbone
    from minimax_music3_mlx.condition_encoder import load_condition_encoder
    from minimax_music3_mlx.depth_decoder import load_depth_decoder
    from minimax_music3_mlx.dit import load_dit
    from minimax_music3_mlx.prompt import validate_tokenizer_ids
    from minimax_music3_mlx.vocoder import load_vocoder

    t = time.time()
    tok = AutoTokenizer.from_pretrained(str(weights / "qwen_7B" / "qwen3-8B-tokenizer-music"))
    validate_tokenizer_ids(tok)
    bb, _ = load_backbone(weights / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    dp = load_depth_decoder(weights / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    ce = load_condition_encoder(weights / "flowmatching_vae.pth", dtype=mx.float32)
    dit = load_dit(weights / "flowmatching_vae.pth", dtype=mx.float32)
    voc = load_vocoder(weights / "dav.pth")  # bf16 default
    if quant_ar:
        nn.quantize(bb, bits=quant_ar, group_size=64)
        nn.quantize(dp, bits=quant_ar, group_size=64)
    mx.eval(bb.parameters(), dp.parameters(), ce.parameters(), dit.parameters(), voc.parameters())
    _state.update(tok=tok, bb=bb, dp=dp, ce=ce, dit=dit, voc=voc, quant_ar=quant_ar)
    print(f"[music3] loaded in {time.time() - t:.1f}s "
          f"(TF32={os.environ.get('MLX_ENABLE_TF32')} q{quant_ar or 'none'} AR, bf16 DAV)", flush=True)


def _generate(
    caption: str,
    lyrics: str,
    seed: int,
    target_duration,
    num_steps: int,
    rewrite: bool,
) -> tuple[bytes, dict]:
    from minimax_music3_mlx.caption_rewriter import available, rewrite_caption
    from minimax_music3_mlx.duration import (
        build_duration_policy,
        ensure_target_duration,
        extract_duration_from_text,
    )
    from minimax_music3_mlx.generation import generate_frames
    from minimax_music3_mlx.pipeline import synthesize_windows, resample_44k_to_32k, write_mp3
    from minimax_music3_mlx.prompt import build_prompt
    from minimax_music3_mlx.constants import MAX_PROMPT_TOKENS

    if target_duration is None:
        target_duration = extract_duration_from_text(caption)
    policy = build_duration_policy(target_duration)
    rewritten = (
        rewrite_caption(caption, lyrics, target_seconds=policy.target_seconds)
        if rewrite
        else caption
    )
    effective_caption = ensure_target_duration(rewritten, policy)
    ids = _state["tok"].encode(build_prompt(effective_caption, lyrics), add_special_tokens=False)
    truncated = len(ids) > MAX_PROMPT_TOKENS
    ids = ids[:MAX_PROMPT_TOKENS]

    with _gen_lock:
        t = time.time()
        codes, frame_hidden = generate_frames(_state["bb"], _state["dp"], ids,
                                              seed=seed, max_frames=policy.max_frames,
                                              min_frames=policy.min_frames)
        n = int(frame_hidden.shape[0])
        wave44 = synthesize_windows(_state["ce"], _state["dit"], _state["voc"], frame_hidden,
                                    seed=seed, num_steps=num_steps)
        wave32 = resample_44k_to_32k(wave44)
        elapsed = time.time() - t

    import io
    import subprocess
    import shutil
    pcm = (np.clip(wave32, -1, 1) * 32767).astype(np.int16).T.tobytes()
    ff = shutil.which("ffmpeg")
    proc = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "s16le",
                           "-ar", "32000", "-ac", "2", "-i", "pipe:0", "-codec:a", "libmp3lame",
                           "-b:a", "256k", "-f", "mp3", "pipe:1"], input=pcm, capture_output=True)
    actual_seconds = round(wave32.shape[-1] / 32000, 2)
    finish_reason = "model_eos" if n < policy.max_frames else "safety_cap"
    meta = {
        "frames": n,
        "seconds": actual_seconds,
        "seed": seed,
        "gen_s": round(elapsed, 1),
        "finish_reason": finish_reason,
        "max_frames": policy.max_frames,
        "min_frames": policy.min_frames,
        "caption_rewritten": rewrite and rewritten != caption,
        "caption_expanded": effective_caption != caption,
        "duration_injected": policy.target_seconds is not None,
        "rewriter_available": available(),
        "prompt_truncated": truncated,
    }
    if policy.target_seconds is not None:
        meta.update(
            target_seconds=policy.target_seconds,
            target_frames=policy.target_frames,
            duration_delta_s=round(actual_seconds - policy.target_seconds, 2),
            minimum_ratio=policy.minimum_ratio,
        )
    return proc.stdout, meta


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._json(200, {"status": "ok", "model": MODEL_ID,
                                    "loaded": "bb" in _state, "quant_ar": _state.get("quant_ar")})
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "minimax", "type": "music"}]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/audio/music", "/v1/audio/speech"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad json: {e}"})
        caption = req.get("caption") or req.get("instructions") or ""
        lyrics = req.get("lyrics") or req.get("input") or ""
        if not caption:
            return self._json(400, {"error": "caption (or instructions) required"})
        requested_max_frames = req.get("max_frames", MAX_AUDIO_FRAMES)
        try:
            if int(requested_max_frames) != MAX_AUDIO_FRAMES:
                raise ValueError(
                    f"max_frames is fixed at the {MAX_AUDIO_FRAMES}-frame safety ceiling; "
                    "use target_duration or target_seconds to request song length"
                )
            target_duration = req.get(
                "target_duration",
                req.get("target_seconds", req.get("duration_seconds")),
            )
        except (TypeError, ValueError) as e:
            return self._json(400, {"error": str(e)})
        try:
            audio, meta = _generate(
                caption,
                lyrics,
                int(req.get("seed", 0)),
                target_duration,
                int(req.get("num_steps", 30)),
                bool(req.get("rewrite", True)),
            )
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._json(500, {"error": str(e)})
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        for k, v in meta.items():
            self.send_header(f"X-Music3-{k}", str(v))
        self.end_headers()
        self.wfile.write(audio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--weights", type=Path, default=PORT_ROOT / "weights")
    ap.add_argument("--quant-ar", type=int, default=6)
    args = ap.parse_args()
    _load(args.weights, args.quant_ar)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[music3] serving on http://{args.host}:{args.port}  "
          f"(POST /v1/audio/music, GET /health)", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
