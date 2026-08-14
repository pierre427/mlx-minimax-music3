#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Reference-rank fidelity harness for the AR half (fable #2 gate).

Generates a reference code trajectory with the trusted unquantized model, then
teacher-forces that trajectory through each candidate config and measures where
each reference code lands in that config's own guided logits. A faithful config
keeps the reference codes near the top (rank ~0) and, critically, inside the
top-50 samplable set; degradation pushes ranks up and out of top-50.

The "reference" trajectory is our unquantized model's own generation (MiniMax's
shipped code trajectories aren't public), so this measures quantization fidelity
*relative to the fp32-parity baseline* — exactly the quant decision this gates.

    .venv/bin/python minimax-music3-mlx/scripts/fidelity_rank.py --frames 80 --bits 8 6 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

CAPTION = ("A lively traditional Irish jig in 6/8 at fast tempo, key of D major, fiddle and "
           "tin whistle, button accordion, bodhran drum, spirited pub-session feel.")
LYRICS = "[verse]\nUp with the lark and away we go\nDown by the river where the wild winds blow\n"


def _load(W, bits):
    from minimax_music3_mlx.backbone import load_backbone
    from minimax_music3_mlx.depth_decoder import load_depth_decoder
    bb, _ = load_backbone(W / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    dp = load_depth_decoder(W / "qwen_7B" / "qwen_7B", dtype=mx.bfloat16)
    if bits:
        nn.quantize(bb, bits=bits, group_size=64)
        nn.quantize(dp, bits=bits, group_size=64)
    mx.eval(bb.parameters(), dp.parameters())
    return bb, dp


def _stats(ranks: np.ndarray) -> dict:
    r = ranks.reshape(-1)
    return {
        "mean": float(r.mean()), "median": float(np.median(r)),
        "top1": float((r == 0).mean()), "top5": float((r < 5).mean()),
        "top50": float((r < 50).mean()), "out_of_top50": float((r >= 50).mean()),
        "c0_top50": float((ranks[:, 0] < 50).mean()),        # c0 vocab is 16384
        "depth_top50": float((ranks[:, 1:] < 50).mean()),    # residual codebooks are 1024
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bits", type=int, nargs="*", default=[8, 6, 4])
    ap.add_argument("--weights", type=Path, default=PORT_ROOT / "weights")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from minimax_music3_mlx.generation import generate_frames, replay_ranks
    from minimax_music3_mlx.prompt import build_prompt

    tok = AutoTokenizer.from_pretrained(str(args.weights / "qwen_7B" / "qwen3-8B-tokenizer-music"))
    ids = tok.encode(build_prompt(CAPTION, LYRICS), add_special_tokens=False)

    # 1) reference trajectory from the trusted unquantized model
    bb, dp = _load(args.weights, bits=0)
    ref_codes, _ = generate_frames(bb, dp, ids, seed=args.seed, max_frames=args.frames)
    print(f"reference trajectory: {ref_codes.shape[0]} frames x 8 codes (unquantized bf16)\n")

    # 2) baseline self-replay (unquantized) + each quant level
    configs = [("bf16", 0)] + [(f"q{b}", b) for b in args.bits]
    rows = []
    for name, bits in configs:
        model_bb, model_dp = (bb, dp) if bits == 0 else _load(args.weights, bits=bits)
        ranks = replay_ranks(model_bb, model_dp, ids, ref_codes)
        rows.append((name, _stats(ranks)))

    # 3) report
    hdr = f"{'config':<7}{'mean':>7}{'med':>5}{'top1':>7}{'top5':>7}{'top50':>7}{'OUT50':>7}{'c0≤50':>7}{'dep≤50':>7}"
    print(hdr); print("-" * len(hdr))
    base = rows[0][1]
    for name, s in rows:
        print(f"{name:<7}{s['mean']:>7.1f}{s['median']:>5.0f}{s['top1']:>7.1%}{s['top5']:>7.1%}"
              f"{s['top50']:>7.1%}{s['out_of_top50']:>7.1%}{s['c0_top50']:>7.1%}{s['depth_top50']:>7.1%}")
    print("\nGate: a faithful quant keeps OUT50 (codes that fell out of the samplable")
    print("top-50) near the bf16 baseline; a spike there means drift that WILL change audio.")
    print(f"bf16 baseline OUT50={base['out_of_top50']:.1%} (this is the reference floor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
