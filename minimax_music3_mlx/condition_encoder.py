# SPDX-License-Identifier: Apache-2.0
"""M5: condition encoder — AR frame hiddens -> latent-aligned DiT conditioning.

Each frame carries 8 hidden states of 4096 (global + 7 depth); they are mixed with
learned softmax weights, scaled, projected by a Conv1d(4096->2048, k3, p1), then
nearest-neighbour resampled from the 24 kHz/960-hop AR frame rate to the
44.1 kHz/512-hop latent rate. Matches diffusers `MiniMaxMusic3ConditionEncoder`
(== sglang `condition`/`aligned_condition`).

Weights come from `flowmatching_vae.pth`:
  cond_layer_logits -> layer_weight_logits, cond_layer_scale -> layer_scale,
  latent_conditioners.0.{weight,bias} -> proj.{weight,bias}.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# latent_length = int(frames * out_sr/in_sr * in_hop/out_hop)
_IN_SR, _OUT_SR, _IN_HOP, _OUT_HOP = 24000, 44100, 960, 512


def aligned_mel_length(num_frames: int) -> int:
    return max(1, int(num_frames * _OUT_SR / _IN_SR * _IN_HOP / _OUT_HOP))


def _nearest_indices(num_frames: int, out_len: int) -> mx.array:
    # torch F.interpolate(mode="nearest"): src = floor(dst * in/out)
    idx = np.floor(np.arange(out_len) * (num_frames / out_len)).astype(np.int32)
    return mx.array(np.clip(idx, 0, num_frames - 1))


class ConditionEncoder(nn.Module):
    def __init__(self, condition_hidden_dim: int = 4096, num_condition_layers: int = 8,
                 out_dim: int = 2048):
        super().__init__()
        self.num_layers = num_condition_layers
        self.hidden_dim = condition_hidden_dim
        self.layer_weight_logits = mx.zeros(num_condition_layers)
        self.layer_scale = mx.ones(1)
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """[B, frames, num_layers*hidden] -> [B, latent_length, out_dim]."""
        b, frames, _ = hidden_states.shape
        h = hidden_states.reshape(b, frames, self.num_layers, self.hidden_dim)
        w = mx.softmax(self.layer_weight_logits, axis=0).astype(h.dtype)
        h = (h * w[None, None, :, None]).sum(axis=2)          # [B, frames, hidden]
        h = self.layer_scale.astype(h.dtype) * h
        h = self.proj(h)                                       # [B, frames, out_dim] (channels-last)
        out_len = aligned_mel_length(frames)
        return h[:, _nearest_indices(frames, out_len), :]      # nearest resample over frames


def load_condition_encoder(
    pth_path: str | Path, *, dtype: mx.Dtype = mx.float32
) -> ConditionEncoder:
    import torch

    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True, mmap=True)

    def g(name):
        return mx.array(sd[name].float().numpy()).astype(dtype)

    model = ConditionEncoder()
    weights = {
        "layer_weight_logits": g("cond_layer_logits"),
        "layer_scale": g("cond_layer_scale"),
        # torch Conv1d weight [out, in, k] -> mlx Conv1d weight [out, k, in]
        "proj.weight": g("latent_conditioners.0.weight").transpose(0, 2, 1),
        "proj.bias": g("latent_conditioners.0.bias"),
    }
    model.load_weights(list(weights.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.eval()
    return model
