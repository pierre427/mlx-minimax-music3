# SPDX-License-Identifier: Apache-2.0
"""M3: MiniMax-Music3 RVQ depth decoder (the "local LM").

Within each audio frame it autoregressively predicts the seven residual RVQ
codebooks c1..c7 from the global hidden state and the codes sampled so far, and
exposes the per-step hidden states that (concatenated with the global hidden)
form the 32768-d frame condition for the flow-matching stage.

Structure follows the diffusers `MiniMaxMusic3RVQDepthDecoder` (split q/k/v,
`audio_embeddings` owned here). `forward` takes already-`projection`-ed
depth-sequence embeddings and returns normed hidden `[B, steps, 4096]`; the
caller applies `projection` and reads `audio_heads[i]` on the last step.

Weights load from the raw backbone checkpoint:
  * `model.audio_extra_embedding.weight` -> `audio_embeddings.weight`
  * `model.audio_decoder.*`              -> the rest (q/k/v/o_proj -> to_q/k/v/out,
    mlp.{gate,up,down}_proj kept, *_layernorm/norm/pos_embedding/projection/audio_heads).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((s, s), -mx.inf, dtype=q.dtype), k=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        return self.to_out(out)


class DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(dim, eps=1e-6)
        self.attn = DepthAttention(dim, heads)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=1e-6)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.input_layernorm(x))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class RVQDepthDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = [
            DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.audio_heads = [
            nn.Linear(hidden_size, audio_vocab_size, bias=False)
            for _ in range(num_codebooks - 1)
        ]

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        """inputs_embeds: already-projected depth sequence [B, steps, hidden]."""
        positions = mx.arange(inputs_embeds.shape[1])
        h = inputs_embeds + self.pos_embedding(positions)[None]
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)

    def head_logits(self, hidden: mx.array, codebook_idx: int) -> mx.array:
        """Logits for c{codebook_idx+1} from the step's normed hidden."""
        return self.audio_heads[codebook_idx](hidden)


# ---- weight loading ----------------------------------------------------------

_AUD = "model.audio_decoder."
_EMB = "model.audio_extra_embedding.weight"


def _translate(raw: dict[str, mx.array], num_layers: int, dtype: mx.Dtype) -> dict:
    """Raw backbone-checkpoint keys -> this module's parameter tree."""
    w: dict[str, mx.array] = {}
    g = lambda k: raw[k].astype(dtype)  # noqa: E731
    w["audio_embeddings.weight"] = g(_EMB)
    w["projection.weight"] = g(_AUD + "projection.weight")
    w["pos_embedding.weight"] = g(_AUD + "pos_embedding.weight")
    w["norm.weight"] = g(_AUD + "norm.weight")
    for i in range(num_layers):
        lp, dp = f"{_AUD}layers.{i}.", f"layers.{i}."
        w[dp + "input_layernorm.weight"] = g(lp + "input_layernorm.weight")
        w[dp + "post_attention_layernorm.weight"] = g(lp + "post_attention_layernorm.weight")
        w[dp + "attn.to_q.weight"] = g(lp + "self_attn.q_proj.weight")
        w[dp + "attn.to_k.weight"] = g(lp + "self_attn.k_proj.weight")
        w[dp + "attn.to_v.weight"] = g(lp + "self_attn.v_proj.weight")
        w[dp + "attn.to_out.weight"] = g(lp + "self_attn.o_proj.weight")
        w[dp + "gate_proj.weight"] = g(lp + "mlp.gate_proj.weight")
        w[dp + "up_proj.weight"] = g(lp + "mlp.up_proj.weight")
        w[dp + "down_proj.weight"] = g(lp + "mlp.down_proj.weight")
    # audio_heads.{i}.weight — count them from the raw keys
    n_heads = sum(1 for k in raw if k.startswith(_AUD + "audio_heads."))
    for i in range(n_heads):
        w[f"audio_heads.{i}.weight"] = g(f"{_AUD}audio_heads.{i}.weight")
    return w


def load_depth_decoder(
    ckpt_dir: str | Path,
    *,
    dtype: mx.Dtype = mx.float32,
) -> RVQDepthDecoder:
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "config.json").read_text())
    model = RVQDepthDecoder(
        hidden_size=config["hidden_size"],
        num_layers=config["decoder_num_layers"],
        num_attention_heads=config["decoder_num_heads"],
        intermediate_size=config["decoder_intermediate_size"],
        audio_vocab_size=config["audio_vocab_size"],
        num_codebooks=config["audio_num_codebooks"],
    )
    raw: dict[str, mx.array] = {}
    for shard in sorted(glob.glob(str(ckpt_dir / "*.safetensors"))):
        for k, v in mx.load(shard).items():
            if k == _EMB or k.startswith(_AUD):
                raw[k] = v
    weights = _translate(raw, config["decoder_num_layers"], dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.eval()
    return model
