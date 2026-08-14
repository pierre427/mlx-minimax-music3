# SPDX-License-Identifier: Apache-2.0
"""M2: MiniMax-Music3 autoregressive backbone on mlx-lm's Qwen3.

The backbone in `qwen_7B/qwen_7B/` is a standard dense Qwen3-8B (the `mixtral`
config label collapses to dense at `num_local_experts=1`). We reuse the unified
tree's `mlx_lm.models.qwen3` verbatim and only:

  * translate the `mixtral`-labelled config into Qwen3 `ModelArgs`,
  * load the checkpoint while dropping the audio-only tensors
    (`model.audio_extra_embedding`, `model.audio_decoder.*`) that belong to the
    RVQ depth decoder (M3),
  * expose the final normed hidden `[B, T, 4096]` (the AR→acoustic bridge) and
    the c0 logits over the masked audio vocabulary.

mlx-lm's `Qwen3Model.__call__` already accepts `input_embeddings=`, which the AR
decode loop needs (it feeds fused audio-frame embeddings, not token ids).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Reuse upstream mlx-lm's Qwen3 implementation (a declared dependency:
# `pip install mlx-lm`). Verified against mlx-lm v0.31.3, whose Qwen3Model
# supports the input_embeddings path the AR loop needs. Set MM3_MLX_LM_UNIFIED
# to point at a local mlx-lm checkout/fork instead.
import os as _os  # noqa: E402
_override = _os.environ.get("MM3_MLX_LM_UNIFIED")
if _override and _override not in sys.path:
    sys.path.insert(0, _override)

from mlx_lm.models.qwen3 import Model as Qwen3Model  # noqa: E402
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args  # noqa: E402

# Tensors in the backbone checkpoint that are NOT part of the Qwen3 graph.
_AUDIO_KEY_PREFIXES = ("model.audio_extra_embedding", "model.audio_decoder")


def build_args(config: dict) -> Qwen3Args:
    """Map the qwen_7B config.json (labelled `mixtral`) to Qwen3 ModelArgs."""
    return Qwen3Args(
        model_type="qwen3",  # override the mixtral label; num_local_experts=1 => dense
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        num_key_value_heads=config["num_key_value_heads"],
        max_position_embeddings=config["max_position_embeddings"],
        rope_theta=config["rope_theta"],
        head_dim=config["head_dim"],
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )


def _load_backbone_weights(ckpt_dir: Path, dtype: mx.Dtype) -> dict:
    """Load every shard, drop audio-only tensors, cast to `dtype`."""
    weights: dict[str, mx.array] = {}
    shards = sorted(glob.glob(str(ckpt_dir / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {ckpt_dir}")
    for shard in shards:
        for k, v in mx.load(shard).items():
            if k.startswith(_AUDIO_KEY_PREFIXES):
                continue
            weights[k] = v.astype(dtype)
    return weights


def load_backbone(
    ckpt_dir: str | Path,
    *,
    dtype: mx.Dtype = mx.float32,
) -> tuple[Qwen3Model, Qwen3Args]:
    """Instantiate the Qwen3 backbone and load weights (fp32 by default for parity).

    Returns the mlx-lm `Model` (has `.model` for last-hidden, `.__call__` for logits).
    """
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "config.json").read_text())
    args = build_args(config)

    model = Qwen3Model(args)
    weights = _load_backbone_weights(ckpt_dir, dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.eval()
    return model, args


def last_hidden(model: Qwen3Model, input_ids: mx.array) -> mx.array:
    """Final normed hidden `[B, T, hidden]` — the AR→acoustic bridge feature."""
    return model.model(input_ids)  # Qwen3Model returns self.norm(h)


def c0_logits(model: Qwen3Model, input_ids: mx.array) -> mx.array:
    """Full-vocab logits `[B, T, vocab]`; c0 masking is applied by the sampler (M4)."""
    return model(input_ids)


def step(
    model: Qwen3Model,
    *,
    cache=None,
    input_ids: mx.array | None = None,
    input_embeddings: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """One backbone pass. Returns (hidden, logits), each `[B, T, ...]`.

    Feed `input_ids` for the prompt prefill, or `input_embeddings` for the
    audio-frame feedback during decode (mlx-lm's Qwen3Model uses the embeds
    directly and advances RoPE from the cache offset).
    """
    placeholder = input_ids if input_ids is not None else input_embeddings
    hidden = model.model(placeholder, cache=cache, input_embeddings=input_embeddings)
    logits = (
        model.model.embed_tokens.as_linear(hidden)
        if model.args.tie_word_embeddings
        else model.lm_head(hidden)
    )
    return hidden, logits
