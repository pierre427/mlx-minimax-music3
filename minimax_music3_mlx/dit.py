# SPDX-License-Identifier: Apache-2.0
"""M6: flow-matching diffusion transformer (the DiT).

Denoises Flow-VAE audio latents [B,128,L] conditioned on the latent-aligned
frame hiddens [B,L,2048] from the condition encoder. Predicts the flow-matching
velocity. Matches diffusers `MiniMaxMusic3Transformer1DModel`.

Notable, non-standard bits (all flagged in the wiki):
  * timestep enters as a *prepended* Fourier token (no AdaLN); dropped after blocks
  * `MiniMaxMusic3FourierEmbedding` uses a TRAINED random-projection weight [128,1]
  * partial RoPE: only the leading rotary_dim=32 of each 64-d head rotates (theta 1e4)
  * x-transformers LayerNorm with real weight AND bias (`gamma`/`beta` — load beta)
  * SiLU-GLU feed-forward; residual pre/post Conv1d(k1)
Runs in fp32 (bf16 measured 9.3 dB drift); TF32 is pinned off in the package init.

Weights: `flowmatching_vae.pth` under `diffusion_transformer.*`.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _partial_rope_tables(seq_len: int, rotary_dim: int, theta: float, dtype):
    inv_freq = 1.0 / (theta ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
    steps = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(steps, inv_freq)            # [seq, rotary_dim/2]
    freqs = mx.concatenate([freqs, freqs], axis=-1)  # [seq, rotary_dim]
    return mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype)


def _apply_partial_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, seq, heads, head_dim]; cos/sin: [seq, rotary_dim]
    rotary_dim = cos.shape[-1]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    first, second = rot[..., :half], rot[..., half:]
    rotate_half = mx.concatenate([-second, first], axis=-1)
    rot = rot * cos + rotate_half * sin
    return mx.concatenate([rot, x[..., rotary_dim:]], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = heads * head_dim
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, s, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(b, s, self.heads, self.head_dim)
        k = k.reshape(b, s, self.heads, self.head_dim)
        v = v.reshape(b, s, self.heads, self.head_dim)
        q = _apply_partial_rope(q, cos, sin)
        k = _apply_partial_rope(k, cos, sin)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner * 2)
        self.ff_out = nn.Linear(ff_inner, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        gate_states, gate = mx.split(self.ff_in(self.norm2(x)), 2, axis=-1)
        return x + self.ff_out(gate_states * nn.silu(gate))


class DiT(nn.Module):
    def __init__(self, in_channels=128, condition_dim=2048, num_layers=36,
                 num_attention_heads=32, attention_head_dim=64, ff_inner_dim=8192,
                 rotary_dim=32, fourier_embedding_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.rotary_dim = rotary_dim
        self.rope_theta = 10000.0
        inner = num_attention_heads * attention_head_dim
        concat = 2 * in_channels + condition_dim

        self.fourier_weight = mx.zeros((fourier_embedding_dim // 2, 1))  # trained
        self.time_linear1 = nn.Linear(fourier_embedding_dim, inner)
        self.time_linear2 = nn.Linear(inner, inner)

        self.preprocess = nn.Linear(concat, concat, bias=False)   # Conv1d k1 as matmul
        self.proj_in = nn.Linear(concat, inner, bias=False)
        self.blocks = [
            TransformerBlock(inner, num_attention_heads, attention_head_dim, ff_inner_dim)
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner, in_channels, bias=False)
        self.postprocess = nn.Linear(in_channels, in_channels, bias=False)
        self._rope_cache: dict = {}  # (fable #6) cos/sin are schedule-independent

    def _rope(self, seq_len: int, dtype):
        key = (seq_len, dtype)
        tab = self._rope_cache.get(key)
        if tab is None:
            tab = _partial_rope_tables(seq_len, self.rotary_dim, self.rope_theta, dtype)
            self._rope_cache[key] = tab
        return tab

    def _timestep_embed(self, t: mx.array) -> mx.array:
        angles = 2.0 * math.pi * t[:, None] @ self.fourier_weight.T  # [B, fourier/2]
        fourier = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)  # [B, fourier]
        return self.time_linear2(nn.silu(self.time_linear1(fourier)))  # [B, inner]

    def __call__(self, hidden: mx.array, timestep: mx.array, cond: mx.array) -> mx.array:
        """hidden [B,128,L], timestep [B], cond [B,L,2048] -> velocity [B,128,L]."""
        # channels-last working layout [B, L, C]
        x = hidden.transpose(0, 2, 1)                    # [B, L, 128]
        zeros = mx.zeros_like(x)
        x = mx.concatenate([x, zeros, cond], axis=-1)    # [B, L, 2304]
        x = self.preprocess(x) + x
        temb = self._timestep_embed(timestep)            # [B, inner]
        x = self.proj_in(x)                              # [B, L, inner]
        x = mx.concatenate([temb[:, None, :], x], axis=1)  # prepend timestep token
        cos, sin = self._rope(x.shape[1], x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.proj_out(x[:, 1:])                      # drop timestep token -> [B, L, 128]
        x = self.postprocess(x) + x
        return x.transpose(0, 2, 1)                      # [B, 128, L]


def load_dit(pth_path: str | Path, *, dtype: mx.Dtype = mx.float32,
             num_layers: int = 36) -> DiT:
    import torch

    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True, mmap=True)
    P = "diffusion_transformer."

    def g(name):
        return mx.array(sd[P + name].float().numpy()).astype(dtype)

    model = DiT(num_layers=num_layers)
    w = {
        "fourier_weight": g("timestep_features.weight"),
        "time_linear1.weight": g("to_timestep_embed.0.weight"),
        "time_linear1.bias": g("to_timestep_embed.0.bias"),
        "time_linear2.weight": g("to_timestep_embed.2.weight"),
        "time_linear2.bias": g("to_timestep_embed.2.bias"),
        "preprocess.weight": g("preprocess_conv.weight").squeeze(-1),   # [C,C,1]->[C,C]
        "postprocess.weight": g("postprocess_conv.weight").squeeze(-1),
        "proj_in.weight": g("transformer.project_in.weight"),
        "proj_out.weight": g("transformer.project_out.weight"),
    }
    for i in range(num_layers):
        lp = f"transformer.layers.{i}."
        bp = f"blocks.{i}."
        w[bp + "norm1.weight"] = g(lp + "pre_norm.gamma")
        w[bp + "norm1.bias"] = g(lp + "pre_norm.beta")
        w[bp + "norm2.weight"] = g(lp + "ff_norm.gamma")
        w[bp + "norm2.bias"] = g(lp + "ff_norm.beta")
        w[bp + "attn.to_qkv.weight"] = g(lp + "self_attn.to_qkv.weight")
        w[bp + "attn.to_out.weight"] = g(lp + "self_attn.to_out.weight")
        w[bp + "ff_in.weight"] = g(lp + "ff.ff.0.proj.weight")
        w[bp + "ff_in.bias"] = g(lp + "ff.ff.0.proj.bias")
        w[bp + "ff_out.weight"] = g(lp + "ff.ff.2.weight")
        w[bp + "ff_out.bias"] = g(lp + "ff.ff.2.bias")

    model.load_weights(list(w.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.eval()
    return model
