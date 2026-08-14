# SPDX-License-Identifier: Apache-2.0
"""M2 parity: MLX Qwen3 backbone last-hidden vs the torch reference (fp32).

Oracle = transformers `Qwen3ForCausalLM` loaded from the `language_model/`
re-export (the same backbone weights as `qwen_7B/` minus the audio keys), run on
CPU in fp32 with output_hidden_states. We diff the final normed hidden `[1,T,4096]`.

Heavy (loads two 8B models in fp32 ~64 GB); opt-in via RUN_HEAVY=1.
The package __init__ pins MLX_ENABLE_TF32=0, which is what makes this pass at 1e-4.

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_backbone_parity.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401  (pins MLX_ENABLE_TF32=0 before any mlx import)

BACKBONE_CKPT = PORT_ROOT / "weights" / "qwen_7B" / "qwen_7B"
REF_CKPT = PORT_ROOT / "weights" / "language_model"

# <|im_start|><|caption_start|>Basic: bpm is 9 2 .
IDS = [151644, 151671, 15944, 25, 97724, 374, 220, 24, 17, 13]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1",
    reason="heavy (two 8B models); set RUN_HEAVY=1 to run",
)


def test_backbone_last_hidden_parity_fp32():
    import mlx.core as mx

    from minimax_music3_mlx.backbone import last_hidden, load_backbone

    if not BACKBONE_CKPT.exists() or not REF_CKPT.exists():
        pytest.skip("weights not present")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model, _ = load_backbone(BACKBONE_CKPT, dtype=mx.float32)
    h_mlx = np.array(last_hidden(model, mx.array([IDS])).astype(mx.float32))
    del model

    tm = transformers.Qwen3ForCausalLM.from_pretrained(
        str(REF_CKPT), torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        h_ref = (
            tm(torch.tensor([IDS]), output_hidden_states=True)
            .hidden_states[-1]
            .float()
            .numpy()
        )

    max_abs = float(np.abs(h_mlx - h_ref).max())
    cos = (h_mlx * h_ref).sum(-1) / (
        np.linalg.norm(h_mlx, axis=-1) * np.linalg.norm(h_ref, axis=-1) + 1e-9
    )
    assert cos.min() > 0.9999, f"cosine drift: {cos.min()}"
    assert max_abs < 1e-4, f"max_abs {max_abs:.3e} exceeds 1e-4 (TF32 pinned off?)"
