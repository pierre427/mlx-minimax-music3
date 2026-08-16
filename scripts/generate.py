#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate music with the MLX MiniMax-Music3 port.

    .venv/bin/python minimax-music3-mlx/scripts/generate.py \
        --caption "bpm is 120. key is C. upbeat acoustic pop." \
        --lyrics "[verse] sunlight on the open road\n[chorus] we are golden" \
        --target-duration 4:00 --out out.mp3 --seed 0

The production ceiling is 9,000 frames. A target duration is injected into the
expanded caption and protects the first 95% from early model EOS.
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
    p.add_argument("--target-duration", help="requested duration in seconds or M:SS (up to 5:00)")
    p.add_argument("--max-frames", type=int, default=9000,
                   help="developer safety ceiling; production default is always 9000")
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
    from minimax_music3_mlx.duration import (
        build_duration_policy,
        ensure_target_duration,
        extract_duration_from_text,
    )
    from minimax_music3_mlx.pipeline import generate_music, write_audio
    from minimax_music3_mlx.prompt import build_prompt, validate_tokenizer_ids
    from minimax_music3_mlx.vocoder import load_vocoder

    W = args.weights
    tok = AutoTokenizer.from_pretrained(str(W / "qwen_7B" / "qwen3-8B-tokenizer-music"))
    validate_tokenizer_ids(tok)

    lyrics = args.lyrics.replace("\\n", "\n")
    requested_duration = args.target_duration
    if requested_duration is None:
        requested_duration = extract_duration_from_text(args.caption)
    policy = build_duration_policy(requested_duration, max_frames=args.max_frames)
    caption = args.caption
    if not args.no_rewrite:  # caption-rewriter skill runs for every request (per requirement)
        expanded = rewrite_caption(caption, lyrics, target_seconds=policy.target_seconds)
        print(f"caption-rewriter: {'applied' if expanded != caption else 'raw fallback'} "
              f"(available={rewriter_available()})")
        caption = expanded
    caption = ensure_target_duration(caption, policy)

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
                                max_frames=policy.max_frames, min_frames=policy.min_frames,
                                num_steps=args.num_steps)
    out = write_audio(args.out, wave32k)  # MP3 (deletes any .wav)
    finish_reason = "model_eos" if n < policy.max_frames else "safety_cap"
    print(f"{n} frames -> {wave32k.shape[-1] / 32000:.2f}s @ 32 kHz "
          f"in {time.time() - t1:.1f}s ({finish_reason}, min_frames={policy.min_frames}, "
          f"max_frames={policy.max_frames}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
