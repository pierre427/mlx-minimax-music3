# SPDX-License-Identifier: Apache-2.0
"""M5 parity: MLX condition encoder vs a torch transcription of the diffusers
`MiniMaxMusic3ConditionEncoder.forward` (fp32), using the same flowmatching_vae.pth
weights. Also checks aligned_mel_length (689 @ 200 frames, 344 @ 100).

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_condition_encoder_parity.py -q -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401  (pin TF32)

PTH = PORT_ROOT / "weights" / "flowmatching_vae.pth"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1", reason="loads flowmatching_vae.pth; set RUN_HEAVY=1"
)


def test_aligned_mel_length():
    from minimax_music3_mlx.condition_encoder import aligned_mel_length

    assert aligned_mel_length(200) == 689
    assert aligned_mel_length(100) == 344


def _torch_reference(sd, hidden_np):
    import torch
    import torch.nn.functional as F

    b, frames, _ = hidden_np.shape
    h = torch.from_numpy(hidden_np).transpose(1, 2).reshape(b, 8, 4096, frames)
    w = torch.softmax(sd["cond_layer_logits"].float(), dim=0)
    h = torch.einsum("blht,l->bht", h, w)
    h = sd["cond_layer_scale"].float() * h
    h = F.conv1d(h, sd["latent_conditioners.0.weight"].float(),
                 sd["latent_conditioners.0.bias"].float(), padding=1)
    out_len = max(1, int(frames * 44100 / 24000 * 960 / 512))
    h = F.interpolate(h, size=out_len, mode="nearest")
    return h.transpose(1, 2).numpy()


@pytest.mark.parametrize("frames", [200, 100, 53])
def test_condition_encoder_parity(frames):
    import mlx.core as mx
    import torch

    if not PTH.exists():
        pytest.skip("flowmatching_vae.pth missing")

    from minimax_music3_mlx.condition_encoder import load_condition_encoder

    enc = load_condition_encoder(PTH, dtype=mx.float32)
    sd = torch.load(str(PTH), map_location="cpu", weights_only=True, mmap=True)

    rng = np.random.default_rng(frames)
    hidden = rng.standard_normal((1, frames, 32768)).astype(np.float32)

    got = np.array(enc(mx.array(hidden)).astype(mx.float32))
    ref = _torch_reference(sd, hidden)
    assert got.shape == ref.shape == (1, max(1, int(frames * 44100 / 24000 * 960 / 512)), 2048)
    d = float(np.abs(got - ref).max())
    print(f"\nframes={frames} out_len={got.shape[1]} max_abs={d:.3e}")
    assert d < 1e-4
