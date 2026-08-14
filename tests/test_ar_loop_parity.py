# SPDX-License-Identifier: Apache-2.0
"""M2b parity: the AR-loop *glue* vs torch transcriptions of the sglang functions.

M2/M3/M4 are individually parity-verified, so the new surface in the loop is the
assembly glue: c0 narrowing (select_c0_logits), c0 CFG (apply_cfg), the feedback
embedding (embed_audio_frames) and the depth-sequence assembly / frame_hidden.
Each is checked against a torch transcription of the exact sglang code; the depth
path is driven with forced codes so both frameworks build the identical sequence.
The backbone KV-cache stepping is covered by M2; the full multi-frame run vs
reference clips is validated at M8.

Reads only the embed table + audio shards (not the full backbone).
Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_ar_loop_parity.py -q -s
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

import minimax_music3_mlx  # noqa: E402,F401  (pin TF32 before mlx)

CKPT = PORT_ROOT / "weights" / "qwen_7B" / "qwen_7B"
REF_RVQ_PY = Path("/Users/Shared/src/sglang-omni/sglang_omni/models/minimax_music3/rvq_decoder.py")

STOP, OFFSET, C0V, SCALE, TOPK, VOCAB = 151670, 151675, 16384, 1.5, 50, 200000

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1", reason="loads embed+audio tensors; set RUN_HEAVY=1"
)


def _np_tensor(name):
    from safetensors import safe_open  # torch framework: handles bf16 -> fp32

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    with safe_open(str(CKPT / shard), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def _audio_state_torch():
    import torch
    from safetensors import safe_open

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())
    wanted = {k: v for k, v in index["weight_map"].items()
              if k.startswith("model.audio_decoder.") or k == "model.audio_extra_embedding.weight"}
    state = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(str(CKPT / shard), framework="pt") as f:
            for k in f.keys():
                if k in wanted:
                    state[k] = f.get_tensor(k).float()
    return state


def _load_ref_rvq():
    spec = importlib.util.spec_from_file_location("_ref_rvq", REF_RVQ_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ctx():
    if not CKPT.exists() or not REF_RVQ_PY.exists():
        pytest.skip("weights/reference missing")
    import mlx.core as mx
    import mlx.nn as nn
    import torch

    cfg = json.loads((CKPT / "config.json").read_text())
    embed_w = _np_tensor("model.embed_tokens.weight")  # [200000, 4096]

    from minimax_music3_mlx.depth_decoder import load_depth_decoder

    mlx_depth = load_depth_decoder(CKPT, dtype=mx.float32)
    embed_mlx = nn.Embedding(embed_w.shape[0], embed_w.shape[1])
    embed_mlx.weight = mx.array(embed_w)

    ref = _load_ref_rvq()
    torch_depth = ref.RVQDepthDecoder(
        hidden_size=cfg["hidden_size"], num_layers=cfg["decoder_num_layers"],
        num_heads=cfg["decoder_num_heads"], intermediate_size=cfg["decoder_intermediate_size"],
        audio_vocab_size=cfg["audio_vocab_size"], num_codebooks=cfg["audio_num_codebooks"],
    ).eval()
    torch_depth.load_checkpoint_state(_audio_state_torch())
    aud_w = _np_tensor("model.audio_extra_embedding.weight")  # [7168, 4096]
    return dict(mx=mx, torch=torch, embed_w=embed_w, aud_w=aud_w,
               mlx_depth=mlx_depth, embed_mlx=embed_mlx, torch_depth=torch_depth)


def test_select_c0_logits(ctx):
    mx = ctx["mx"]
    from minimax_music3_mlx.generation import _c0_logit_ids

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, VOCAB)).astype(np.float32)
    got = np.array(mx.take(mx.array(logits), _c0_logit_ids(), axis=-1))
    ids = np.concatenate([[STOP], np.arange(OFFSET, OFFSET + C0V)])
    ref = logits[:, ids]
    assert np.abs(got - ref).max() < 1e-4


def test_apply_c0_cfg(ctx):
    from minimax_music3_mlx.generation import _apply_c0_cfg

    rng = np.random.default_rng(1)
    narrowed = rng.standard_normal((2, 1 + C0V)).astype(np.float64)
    got = _apply_c0_cfg(narrowed, SCALE, TOPK)
    cond, uncond = narrowed[0], narrowed[1]
    guided = uncond + (cond - uncond) * SCALE
    thr = np.sort(cond)[-TOPK]
    ref = np.where(cond < thr, -np.inf, guided)
    fin = np.isfinite(ref)
    assert np.abs(got[fin] - ref[fin]).max() < 1e-9
    assert np.array_equal(np.isfinite(got), fin)


def test_embed_audio_frame(ctx):
    mx = ctx["mx"]
    from minimax_music3_mlx.generation import _embed_audio_frame

    codes = [7777] + np.random.default_rng(2).integers(0, 1024, size=7).tolist()
    got = np.array(_embed_audio_frame(ctx["mlx_depth"], ctx["embed_mlx"], codes).astype(mx.float32))
    ew, aw = ctx["embed_w"], ctx["aud_w"]
    ref = ew[codes[0] + OFFSET].astype(np.float64).copy()
    for i in range(1, 8):
        ref += aw[codes[i] + (i - 1) * 1024].astype(np.float64)
    ref *= 8.0 ** -0.5
    assert np.abs(got.astype(np.float64) - ref).max() < 1e-4


def test_depth_assembly_frame_hidden(ctx):
    mx, torch = ctx["mx"], ctx["torch"]
    from minimax_music3_mlx.generation import _depth_decode

    rng = np.random.default_rng(3)
    hidden2 = rng.standard_normal((2, 4096)).astype(np.float32)
    c0 = 4242
    forced = [c0] + rng.integers(0, 1024, size=7).tolist()

    _, dh_mlx = _depth_decode(ctx["mlx_depth"], ctx["embed_mlx"], mx.array(hidden2),
                              c0, 0, 0, forced=forced)
    dh_mlx = np.array(dh_mlx.astype(mx.float32))

    # torch transcription of the sglang _depth_decode_eager assembly (forced codes)
    td = ctx["torch_depth"]
    ew, aw = torch.from_numpy(ctx["embed_w"]), torch.from_numpy(ctx["aud_w"])
    h2 = torch.from_numpy(hidden2)
    with torch.no_grad():
        c0_embed = ew[[c0 + OFFSET, c0 + OFFSET]]
        seq = [td.projection(h2).unsqueeze(1), td.projection(c0_embed).unsqueeze(1)]
        parts = []
        for index in range(1, 8):
            out = td(torch.cat(seq, dim=1))[:, -1]
            parts.append(out[0])
            if index < 7:
                emb = aw[forced[index] + (index - 1) * 1024]
                seq.append(td.projection(torch.stack([emb, emb])).unsqueeze(1))
        dh_ref = torch.cat(parts).numpy()

    d = float(np.abs(dh_mlx - dh_ref).max())
    print(f"\ndepth frame_hidden max_abs={d:.3e}")
    assert d < 1e-4
