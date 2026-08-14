# SPDX-License-Identifier: Apache-2.0
"""M-opt gate: reference-rank fidelity of the AR-half quantization (fable #2).

Generates a short reference trajectory with the unquantized model, teacher-forces
it through the q6 model, and asserts the reference codes stay overwhelmingly
inside the samplable top-50 (i.e. quantization does not silently change which
codes the model would pick). This is the gate that lets q6 ship.

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_fidelity_rank.py -q -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401

CKPT = PORT_ROOT / "weights" / "qwen_7B" / "qwen_7B"
TOK = PORT_ROOT / "weights" / "qwen_7B" / "qwen3-8B-tokenizer-music"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1", reason="loads the 8B backbone; set RUN_HEAVY=1"
)


def _load(bits):
    import mlx.core as mx
    import mlx.nn as nn

    from minimax_music3_mlx.backbone import load_backbone
    from minimax_music3_mlx.depth_decoder import load_depth_decoder

    bb, _ = load_backbone(CKPT, dtype=mx.bfloat16)
    dp = load_depth_decoder(CKPT, dtype=mx.bfloat16)
    if bits:
        nn.quantize(bb, bits=bits, group_size=64)
        nn.quantize(dp, bits=bits, group_size=64)
    mx.eval(bb.parameters(), dp.parameters())
    return bb, dp


def test_q6_keeps_reference_codes_samplable():
    if not CKPT.exists() or not TOK.exists():
        pytest.skip("weights missing")
    from transformers import AutoTokenizer

    from minimax_music3_mlx.generation import generate_frames, replay_ranks
    from minimax_music3_mlx.prompt import build_prompt

    tok = AutoTokenizer.from_pretrained(str(TOK))
    ids = tok.encode(build_prompt("bpm 120, D major, Irish jig, fiddle, tin whistle",
                                  "[verse]\nup and away we go\n"), add_special_tokens=False)

    bb, dp = _load(0)
    ref, _ = generate_frames(bb, dp, ids, seed=0, max_frames=48)
    base = replay_ranks(bb, dp, ids, ref)
    assert (base >= 50).mean() == 0.0, "unquantized self-replay must keep all codes in top-50"

    q6bb, q6dp = _load(6)
    q6 = replay_ranks(q6bb, q6dp, ids, ref)
    out50 = float((q6 >= 50).mean())
    print(f"\nq6 out-of-top-50 = {out50:.2%} (bf16 floor 0%); mean rank "
          f"{q6.mean():.1f} vs {base.mean():.1f}")
    # q6 should keep >=97% of the reference trajectory samplable (measured ~99.5%).
    assert out50 < 0.03, f"q6 drift too high: {out50:.2%} of reference codes left top-50"
