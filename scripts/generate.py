#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate music with the MLX MiniMax-Music3 port.

    .venv/bin/python minimax-music3-mlx/scripts/generate.py \
        --caption "bpm is 120. key is C. upbeat acoustic pop." \
        --lyrics "[verse] sunlight on the open road\n[chorus] we are golden" \
        --out out.wav --max-frames 150 --seed 0

Single-window path (<= 200 AR frames). Backbone runs bf16, DiT/vocoder fp32.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

# Production entry point: run the fp32 DiT on M5 NAX kernels (TF32 validated corr 1.0
# vs pinned-fp32). Parity tests keep TF32 off via the package default. Override with
# MLX_ENABLE_TF32=0 in the env for the exact-parity path.
os.environ.setdefault("MLX_ENABLE_TF32", "1")

import minimax_music3_mlx  # noqa: E402,F401
import mlx.core as mx  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caption", required=True)
    p.add_argument("--lyrics", required=True)
    p.add_argument("--out", type=Path, default=Path("minimax_music3_mlx.mp3"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--weights", type=Path, default=PORT_ROOT / "weights")
    p.add_argument("--quant-ar", type=int, default=0, help="quantize backbone+depth to N bits (e.g. 6)")
    p.add_argument("--no-rewrite", action="store_true", help="skip the caption-rewriter skill")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from minimax_music3_mlx.backbone import load_backbone
    from minimax_music3_mlx.caption_rewriter import available as rewriter_available
    from minimax_music3_mlx.caption_rewriter import rewrite_caption
    from minimax_music3_mlx.condition_encoder import load_condition_encoder
    from minimax_music3_mlx.constants import MAX_PROMPT_TOKENS
    from minimax_music3_mlx.depth_decoder import load_depth_decoder
    from minimax_music3_mlx.dit import load_dit
    from minimax_music3_mlx.pipeline import generate_music, write_audio
    from minimax_music3_mlx.prompt import build_prompt, validate_tokenizer_ids
    from minimax_music3_mlx.vocoder import load_vocoder

    W = args.weights
    tok = AutoTokenizer.from_pretrained(str(W / "qwen_7B" / "qwen3-8B-tokenizer-music"))
    validate_tokenizer_ids(tok)

    lyrics = args.lyrics.replace("\\n", "\n")
    caption = args.caption
    if not args.no_rewrite:  # caption-rewriter skill runs for every request (per requirement)
        caption = rewrite_caption(caption, lyrics)
        print(f"caption-rewriter: {'applied' if rewriter_available() else 'skipped (no endpoint)'}")

    ids = tok.encode(build_prompt(caption, lyrics), add_special_tokens=False)
    if len(ids) > MAX_PROMPT_TOKENS:  # cookbook: prompt caps at 5000 tokens
        print(f"warning: prompt {len(ids)} tokens > {MAX_PROMPT_TOKENS} cap; truncating")
        ids = ids[:MAX_PROMPT_TOKENS]

    t0 = time.time()
    bb, _ = load_backbone(W / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    dp = load_depth_decoder(W / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    ce = load_condition_encoder(W / "flowmatching_vae.pth", dtype=mx.float32)
    dit = load_dit(W / "flowmatching_vae.pth", dtype=mx.float32)
    voc = load_vocoder(W / "dav.pth")  # bf16 default (corr 0.99993 vs fp32)
    if args.quant_ar:
        import mlx.nn as nn
        nn.quantize(bb, bits=args.quant_ar, group_size=64)
        nn.quantize(dp, bits=args.quant_ar, group_size=64)
        mx.eval(bb.parameters(), dp.parameters())
    print(f"loaded in {time.time() - t0:.1f}s"
          + (f" (q{args.quant_ar} AR)" if args.quant_ar else ""))

    t1 = time.time()
    wave32k, n = generate_music(bb, dp, ce, dit, voc, ids, seed=args.seed,
                                max_frames=args.max_frames, num_steps=args.num_steps)
    out = write_audio(args.out, wave32k)  # MP3 (deletes any .wav)
    print(f"{n} frames -> {wave32k.shape[-1] / 32000:.2f}s @ 32 kHz "
          f"in {time.time() - t1:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
