# SPDX-License-Identifier: Apache-2.0
"""M7: DAV vocoder — Flow-VAE latent -> 44.1 kHz stereo waveform.

A DAC-style decoder: dec_in_proj(64->1024,k1) -> conv_in(1024->1536,k7) -> 4
upsampling blocks (strides 8/8/4/2 = 512x), each Snake -> wnConvTranspose ->
3x ResidualUnit(dil 1/3/9) -> snake_out -> conv_out(->1,k7) -> tanh. Stereo is
carried as two folded latent-channel halves. Matches diffusers
`MiniMaxMusic3Vocoder` / sglang DAV `Decoder`.

The channels-last conv primitives (WNConv1d/WNConvTranspose1d/Snake1d/
ResidualUnit) are adapted from the sibling port minimax-h3-mlx
(minimax_h3_mlx/audio_vae.py) — its DAC building blocks are identical to Music3's;
Music3 uses the plain DAC decoder (no BigVGAN AMPBlock / anti-alias filters).

Weight norm (weight_g/weight_v) is folded to a plain weight at load. Weights come
from `dav.pth` (`dec_in_proj.*`, `decoder.model.*`); the encoder + normalizing
`flow` posterior in that checkpoint are unused at inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class WNConv1d(nn.Module):
    """Conv1d over (N, L, C); weight (C_out, k, C_in). Weight-norm folded at load."""

    def __init__(self, in_c, out_c, k, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.stride, self.padding, self.dilation = stride, padding, dilation
        s = 1.0 / math.sqrt(in_c * k)
        self.weight = mx.random.uniform(-s, s, (out_c, k, in_c))
        if bias:
            self.bias = mx.zeros((out_c,))

    def __call__(self, x):
        out = mx.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return out + self.bias if "bias" in self else out


class WNConvTranspose1d(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride, self.padding = stride, padding
        s = 1.0 / math.sqrt(in_c * k)
        self.weight = mx.random.uniform(-s, s, (out_c, k, in_c))
        if bias:
            self.bias = mx.zeros((out_c,))

    def __call__(self, x):
        out = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        return out + self.bias if "bias" in self else out


class Snake1d(nn.Module):
    """x + (alpha + 1e-9)^-1 * sin(alpha * x)^2, per-channel alpha (channels-last)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x):
        return x + mx.reciprocal(self.alpha + 1e-9) * mx.square(mx.sin(self.alpha * x))


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = WNConv1d(dim, dim, 7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = WNConv1d(dim, dim, 1)

    def __call__(self, x):
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        if y.shape[1] != x.shape[1]:  # centre-crop shortcut (no-op with matched padding)
            pad = (x.shape[1] - y.shape[1]) // 2
            x = x[:, pad : x.shape[1] - pad, :]
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.snake1 = Snake1d(in_dim)
        self.conv_t = WNConvTranspose1d(in_dim, out_dim, 2 * stride, stride=stride,
                                        padding=math.ceil(stride / 2))
        self.res1 = ResidualUnit(out_dim, 1)
        self.res2 = ResidualUnit(out_dim, 3)
        self.res3 = ResidualUnit(out_dim, 9)

    def __call__(self, x):
        return self.res3(self.res2(self.res1(self.conv_t(self.snake1(x)))))


class Vocoder(nn.Module):
    def __init__(self, latent_channels=128, decoder_input_dim=1024,
                 decoder_hidden_dim=1536, upsampling_ratios=(8, 8, 4, 2)):
        super().__init__()
        self.half = latent_channels // 2
        self.dec_in_proj = WNConv1d(self.half, decoder_input_dim, 1)  # plain (no wn in ckpt)
        self.conv_in = WNConv1d(decoder_input_dim, decoder_hidden_dim, 7, padding=3)
        blocks = []
        for i, stride in enumerate(upsampling_ratios):
            in_dim = decoder_hidden_dim // (2**i)
            out_dim = decoder_hidden_dim // (2 ** (i + 1))
            blocks.append(DecoderBlock(in_dim, out_dim, stride))
        self.blocks = blocks
        self.snake_out = Snake1d(out_dim)
        self.conv_out = WNConv1d(out_dim, 1, 7, padding=3)

    def __call__(self, latents: mx.array) -> mx.array:
        """latents [B, 128, T] -> waveform [B, 2, T*512] in [-1, 1]."""
        latents = latents.astype(self.dec_in_proj.weight.dtype)  # accept fp32 DiT latent into bf16 vocoder
        b, _, t = latents.shape
        x = latents.reshape(b * 2, self.half, t).transpose(0, 2, 1)  # [2B, T, 64]
        x = self.conv_in(self.dec_in_proj(x))
        for block in self.blocks:
            x = block(x)
        x = mx.tanh(self.conv_out(self.snake_out(x)))  # [2B, T*512, 1]
        return x.transpose(0, 2, 1).reshape(b, 2, -1)


# ---- weight loading (fold weight-norm; torch conv layout -> mlx channels-last) ----


def _fold(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    # weight_norm dim=0: norm of v over all axes except 0, per-slice scale g.
    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True))
    return (g.astype(np.float64) * v.astype(np.float64) / norm).astype(np.float32)


def load_vocoder(pth_path: str | Path, *, dtype: mx.Dtype = mx.bfloat16) -> Vocoder:
    # bf16 is the default (1.63x faster, waveform corr 0.99993 vs fp32). Pass
    # dtype=mx.float32 for the exact-parity path (tests/test_vocoder_parity.py).
    import torch

    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True, mmap=True)
    npd = {k: v.float().numpy() for k, v in sd.items()
           if k.startswith(("dec_in_proj.", "decoder."))}

    def conv_w(name):  # folded, torch [out,in,k] -> mlx [out,k,in]
        w = _fold(npd[name + ".weight_g"], npd[name + ".weight_v"])
        return mx.array(w.transpose(0, 2, 1)).astype(dtype)

    def convT_w(name):  # folded, torch ConvTranspose [in,out,k] -> mlx [out,k,in]
        w = _fold(npd[name + ".weight_g"], npd[name + ".weight_v"])
        return mx.array(w.transpose(1, 2, 0)).astype(dtype)

    def plain_conv_w(name):  # torch [out,in,k] -> mlx [out,k,in]
        return mx.array(npd[name + ".weight"].transpose(0, 2, 1)).astype(dtype)

    def bias(name):
        return mx.array(npd[name + ".bias"]).astype(dtype)

    def alpha(name):  # ckpt [1,C,1] -> mlx [1,1,C]
        return mx.array(npd[name + ".alpha"].transpose(0, 2, 1)).astype(dtype)

    model = Vocoder()
    w = {
        "dec_in_proj.weight": plain_conv_w("dec_in_proj"),
        "dec_in_proj.bias": bias("dec_in_proj"),
        "conv_in.weight": conv_w("decoder.model.0"),
        "conv_in.bias": bias("decoder.model.0"),
        "snake_out.alpha": alpha("decoder.model.5"),
        "conv_out.weight": conv_w("decoder.model.6"),
        "conv_out.bias": bias("decoder.model.6"),
    }
    for bi in range(4):  # blocks <- decoder.model.{1..4}
        mp = f"decoder.model.{bi + 1}."
        bp = f"blocks.{bi}."
        w[bp + "snake1.alpha"] = alpha(mp + "block.0")
        w[bp + "conv_t.weight"] = convT_w(mp + "block.1")
        w[bp + "conv_t.bias"] = bias(mp + "block.1")
        for ri, rp in enumerate(("res1", "res2", "res3")):
            rb = mp + f"block.{ri + 2}.block."   # ResidualUnit.block.{0..3}
            dp = bp + rp + "."
            w[dp + "snake1.alpha"] = alpha(rb + "0")
            w[dp + "conv1.weight"] = conv_w(rb + "1")
            w[dp + "conv1.bias"] = bias(rb + "1")
            w[dp + "snake2.alpha"] = alpha(rb + "2")
            w[dp + "conv2.weight"] = conv_w(rb + "3")
            w[dp + "conv2.bias"] = bias(rb + "3")

    model.load_weights(list(w.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.eval()
    return model
