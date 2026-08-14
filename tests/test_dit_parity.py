# SPDX-License-Identifier: Apache-2.0
"""M6 parity: MLX flow-matching DiT vs a torch transcription of the diffusers
`MiniMaxMusic3Transformer1DModel.forward` (fp32), same flowmatching_vae.pth weights.

Compares the single-step predicted velocity for a fixed (latent, t, condition).
Runs in fp32 with TF32 pinned off (package init) — the fidelity path the model
requires.

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_dit_parity.py -q -s
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401  (pin TF32)

PTH = PORT_ROOT / "weights" / "flowmatching_vae.pth"
NUM_LAYERS = 36

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1", reason="loads 9GB DiT; set RUN_HEAVY=1"
)


def _torch_reference(sd, hidden, t, cond, num_layers):
    import torch
    import torch.nn.functional as F

    P = "diffusion_transformer."
    W = lambda n: sd[P + n].float()  # noqa: E731
    B, _, L = hidden.shape

    zeros = torch.zeros_like(hidden)
    x = torch.cat([hidden, zeros, cond.transpose(1, 2)], dim=1)
    x = F.conv1d(x, W("preprocess_conv.weight")) + x
    x = x.transpose(1, 2)

    angles = 2.0 * math.pi * t.unsqueeze(-1) @ W("timestep_features.weight").T
    fourier = torch.cat([angles.cos(), angles.sin()], dim=-1)
    temb = F.linear(fourier, W("to_timestep_embed.0.weight"), W("to_timestep_embed.0.bias"))
    temb = F.silu(temb)
    temb = F.linear(temb, W("to_timestep_embed.2.weight"), W("to_timestep_embed.2.bias"))

    x = F.linear(x, W("transformer.project_in.weight"))
    x = torch.cat([temb.unsqueeze(1), x], dim=1)
    seq = x.shape[1]

    rotary_dim, theta = 32, 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos, sin = freqs.cos(), freqs.sin()

    def rope(h):  # h: [B, seq, heads, hd]
        c = cos[:, None, :].to(h.dtype)
        s = sin[:, None, :].to(h.dtype)
        rot = h[..., :rotary_dim]
        a, b = rot.chunk(2, dim=-1)
        rh = torch.cat([-b, a], dim=-1)
        rot = rot * c + rh * s
        return torch.cat([rot, h[..., rotary_dim:]], dim=-1)

    for i in range(num_layers):
        lp = f"transformer.layers.{i}."
        h = F.layer_norm(x, (x.shape[-1],), W(lp + "pre_norm.gamma"), W(lp + "pre_norm.beta"))
        qkv = F.linear(h, W(lp + "self_attn.to_qkv.weight"))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, seq, 32, 64); k = k.view(B, seq, 32, 64); v = v.view(B, seq, 32, 64)
        q, k = rope(q), rope(k)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        attn = attn.transpose(1, 2).reshape(B, seq, 2048)
        x = x + F.linear(attn, W(lp + "self_attn.to_out.weight"))
        h2 = F.layer_norm(x, (x.shape[-1],), W(lp + "ff_norm.gamma"), W(lp + "ff_norm.beta"))
        ff = F.linear(h2, W(lp + "ff.ff.0.proj.weight"), W(lp + "ff.ff.0.proj.bias"))
        gate_states, gate = ff.chunk(2, dim=-1)
        x = x + F.linear(gate_states * F.silu(gate), W(lp + "ff.ff.2.weight"), W(lp + "ff.ff.2.bias"))

    x = F.linear(x[:, 1:], W("transformer.project_out.weight"))
    x = x.transpose(1, 2)
    x = F.conv1d(x, W("postprocess_conv.weight")) + x
    return x.numpy()


@pytest.mark.parametrize("L", [16, 47])
def test_dit_velocity_parity(L):
    import mlx.core as mx
    import torch

    if not PTH.exists():
        pytest.skip("flowmatching_vae.pth missing")

    from minimax_music3_mlx.dit import load_dit

    model = load_dit(PTH, dtype=mx.float32, num_layers=NUM_LAYERS)
    sd = torch.load(str(PTH), map_location="cpu", weights_only=True, mmap=True)

    rng = np.random.default_rng(L)
    hidden = rng.standard_normal((1, 128, L)).astype(np.float32)
    t = np.array([0.37], dtype=np.float32)
    cond = rng.standard_normal((1, L, 2048)).astype(np.float32)

    got = np.array(model(mx.array(hidden), mx.array(t), mx.array(cond)).astype(mx.float32))
    with torch.no_grad():
        ref = _torch_reference(sd, torch.from_numpy(hidden), torch.from_numpy(t),
                               torch.from_numpy(cond), NUM_LAYERS)
    assert got.shape == ref.shape == (1, 128, L)
    d = float(np.abs(got - ref).max())
    rel = d / float(np.abs(ref).max())
    print(f"\nL={L} velocity max_abs={d:.3e} rel={rel:.3e}")
    assert d < 1e-3, f"max_abs {d:.3e}"  # 36-layer fp32 accumulation; expect ~1e-4
