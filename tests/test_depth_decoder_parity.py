# SPDX-License-Identifier: Apache-2.0
"""M3 parity: MLX RVQ depth decoder vs the sglang torch reference (fp32).

Oracle = sglang's `RVQDepthDecoder` (pure torch, loaded standalone from the
reference tree via its own `load_checkpoint_state`). We feed both models an
identical fixed `inputs_embeds` and diff: the normed hidden `[1,steps,4096]`,
every `audio_heads[i]` logit row, and the standalone `projection`.

Only the audio shards are read (via the checkpoint index), so this is light
(~a few hundred MB), not the full 14 GB backbone. Package __init__ pins
MLX_ENABLE_TF32=0.

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_depth_decoder_parity.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401  (pins MLX_ENABLE_TF32=0 before mlx import)

CKPT = PORT_ROOT / "weights" / "qwen_7B" / "qwen_7B"
REF_RVQ_PY = Path(
    "/Users/Shared/src/sglang-omni/sglang_omni/models/minimax_music3/rvq_decoder.py"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1",
    reason="loads checkpoint tensors; set RUN_HEAVY=1 to run",
)


def _load_ref_module():
    spec = importlib.util.spec_from_file_location("_ref_rvq", REF_RVQ_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _audio_state_torch():
    """Load model.audio_decoder.* (+ audio_extra_embedding) as torch fp32 tensors,
    reading only the shards the index says hold them."""
    import torch
    from safetensors import safe_open

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    wanted = {
        k: v
        for k, v in wmap.items()
        if k.startswith("model.audio_decoder.") or k == "model.audio_extra_embedding.weight"
    }
    shards = sorted({v for v in wanted.values()})
    state: dict[str, "torch.Tensor"] = {}
    for shard in shards:
        with safe_open(str(CKPT / shard), framework="pt") as f:
            for k in f.keys():
                if k in wanted:
                    state[k] = f.get_tensor(k).float()
    return state


def test_depth_decoder_parity_fp32():
    import mlx.core as mx
    import torch

    if not CKPT.exists() or not REF_RVQ_PY.exists():
        pytest.skip("weights or reference missing")

    cfg = json.loads((CKPT / "config.json").read_text())
    steps = cfg["audio_num_codebooks"]  # 8 depth positions (global + c0..c6 embeds)

    # ---- reference (torch) ----
    ref = _load_ref_module()
    torch_dec = ref.RVQDepthDecoder(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["decoder_num_layers"],
        num_heads=cfg["decoder_num_heads"],
        intermediate_size=cfg["decoder_intermediate_size"],
        audio_vocab_size=cfg["audio_vocab_size"],
        num_codebooks=cfg["audio_num_codebooks"],
    ).eval()
    torch_dec.load_checkpoint_state(_audio_state_torch())

    # ---- MLX ----
    from minimax_music3_mlx.depth_decoder import load_depth_decoder

    mlx_dec = load_depth_decoder(CKPT, dtype=mx.float32)

    # ---- identical fixed input ----
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, steps, cfg["hidden_size"]), dtype=np.float64).astype(np.float32)

    with torch.no_grad():
        h_ref = torch_dec(torch.from_numpy(x)).numpy()
        heads_ref = [torch_dec.audio_heads[i](torch.from_numpy(h_ref)).numpy()
                     for i in range(cfg["audio_num_codebooks"] - 1)]
        proj_ref = torch_dec.projection(torch.from_numpy(x)).numpy()

    h_mlx = np.array(mlx_dec(mx.array(x)).astype(mx.float32))
    heads_mlx = [np.array(mlx_dec.head_logits(mx.array(h_mlx), i).astype(mx.float32))
                 for i in range(cfg["audio_num_codebooks"] - 1)]
    proj_mlx = np.array(mlx_dec.projection(mx.array(x)).astype(mx.float32))

    hd = float(np.abs(h_mlx - h_ref).max())
    pd = float(np.abs(proj_mlx - proj_ref).max())
    hmax = max(float(np.abs(a - b).max()) for a, b in zip(heads_mlx, heads_ref))
    print(f"\nhidden max_abs={hd:.3e}  projection max_abs={pd:.3e}  heads max_abs={hmax:.3e}")

    assert hd < 1e-4, f"hidden max_abs {hd:.3e}"
    assert pd < 1e-4, f"projection max_abs {pd:.3e}"
    assert hmax < 1e-4, f"head-logits max_abs {hmax:.3e}"
