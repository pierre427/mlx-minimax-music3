# SPDX-License-Identifier: Apache-2.0
"""M2b: the MiniMax-Music3 autoregressive decode loop.

Assembles the backbone (M2), RVQ depth decoder (M3) and seeded sampler (M4) into
the frame-by-frame token generator. Each audio frame:

  1. backbone step -> global hidden + c0 logits, for a (cond, uncond) CFG pair;
  2. c0: narrow to legal ids, CFG-guide (uncond + 1.5*(cond-uncond)) restricted to
     cond's top-50, then seeded top-50 Gumbel sample. Column 0 = stop;
  3. depth-decode c1..c7 over both rows with depth-CFG, taking the cond codes;
  4. frame_hidden = cat(cond global hidden, 7 cond depth hiddens) = 32768;
  5. re-embed all 8 codes (embed_audio_frames) and feed the SAME vector back to
     both rows for the next step.

Positions for the seeded sampler are frame*8 + codebook_index (unique per draw).
Faithful to sglang model_runner.py / sglang_model.py. See wiki
ports/minimax-music3-mlx.md.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from . import backbone as _bb
from .constants import AR_CFG_SCALE, AR_CFG_TOP_K, MAX_AUDIO_FRAMES
from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from .sampler import sample_topk_seeded

_C0_VOCAB = 16384
_STOP = SPECIAL_TOKEN_IDS["<|audio_end|>"]
_CFG_TOKEN = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
_FRAME_SCALE = 8.0**-0.5  # num_codebooks**-0.5


def derive_sampling_seed(public_seed: int, namespace: str = "minimax-ttm-ar") -> int:
    digest = hashlib.blake2b(f"{namespace}:{public_seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _c0_logit_ids() -> mx.array:
    """Legal c0 columns in vocab order: stop first, then the 16384 code range."""
    return mx.concatenate(
        [mx.array([_STOP], dtype=mx.int32),
         mx.arange(AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + _C0_VOCAB, dtype=mx.int32)]
    )


def uncond_ids(prompt_ids: list[int]) -> list[int]:
    """CFG-null prompt: keep <|im_start|>, <|im_end|>, <|audio_start|>; replace the
    caption+lyrics span [1:-2] with <|audio_cfg|>."""
    u = list(prompt_ids)
    u[1:-2] = [_CFG_TOKEN] * (len(u) - 3)
    return u


def _apply_c0_cfg(narrowed: np.ndarray, scale: float, top_k: int) -> np.ndarray:
    """narrowed: [2, 1+16384] fp64 (cond, uncond) -> guided [1+16384].

    guided = uncond + scale*(cond-uncond), then -inf below cond's top-k threshold
    (a second, separate restriction from the sampler's own top-k)."""
    cond, uncond = narrowed[0], narrowed[1]
    guided = uncond + (cond - uncond) * scale
    kth = np.partition(cond, -top_k)[-top_k]
    return np.where(cond < kth, -np.inf, guided)


def _depth_decode(
    depth, embed_tokens, hidden2: mx.array, c0: int, seed: int, frame_pos: int,
    *, forced: list[int] | None = None,
) -> tuple[list[int], mx.array]:
    """Decode c1..c7 over the (cond, uncond) pair; return the 8 cond codes and the
    concatenated 7 cond depth hiddens [7*hidden].

    `forced` (8 codes) replays a fixed trajectory instead of sampling — used by the
    parity harness so both frameworks assemble the identical depth sequence.
    """
    proj = depth.projection
    audio_emb = depth.audio_embeddings
    c0_tok = mx.array([c0 + AUDIO_CODE_OFFSET, c0 + AUDIO_CODE_OFFSET])
    seq = [proj(hidden2)[:, None, :], proj(embed_tokens(c0_tok))[:, None, :]]
    codes = [c0]
    hidden_parts: list[mx.array] = []
    for index in range(1, 8):
        out = depth(mx.concatenate(seq, axis=1))[:, -1]  # [2, hidden]
        hidden_parts.append(out[0])
        if forced is not None:
            code = int(forced[index])
        else:
            head = depth.audio_heads[index - 1](out)  # [2, 1024]
            h = np.array(head.astype(mx.float32)).astype(np.float64)  # one sync, not two
            cond, uncond = h[0], h[1]
            guided = uncond + (cond - uncond) * AR_CFG_SCALE
            code = int(sample_topk_seeded(guided[None], np.array([seed]), np.array([frame_pos + index]))[0])
        codes.append(code)
        if index < 7:
            emb = audio_emb(mx.array([code + (index - 1) * 1024]))  # [1, hidden]
            emb2 = mx.concatenate([emb, emb], axis=0)  # both rows share the code
            seq.append(proj(emb2)[:, None, :])
    return codes, mx.concatenate(hidden_parts)


def _embed_audio_frame(depth, embed_tokens, codes: list[int]) -> mx.array:
    """embed_audio_frames: one feedback vector [hidden] from all 8 codes."""
    c0 = embed_tokens(mx.array([codes[0] + AUDIO_CODE_OFFSET]))[0]
    offsets = mx.arange(7) * 1024
    extra = depth.audio_embeddings(mx.array(codes[1:]) + offsets).sum(axis=0)
    return (c0 + extra) * _FRAME_SCALE


def replay_ranks(backbone, depth, prompt_ids: list[int], forced_codes: np.ndarray, *,
                 head_slice: bool = False) -> np.ndarray:
    """Teacher-forced fidelity harness (fable #2 gate; mirrors sglang
    model_runner._record_reference_ranks).

    Force `forced_codes` [F, 8] (a reference trajectory) through the model and
    return the RANK of each forced code within the model's own guided logits
    [F, 8] — 0 means "the model's top choice". A faithful model (or a good AR-half
    quantization) ranks the reference codes near the front; degradation pushes
    ranks up and eventually out of the top-50 samplable set.
    """
    embed_tokens = backbone.model.embed_tokens
    ids_lookup = _c0_logit_ids()
    c0_head = mx.take(backbone.lm_head.weight, ids_lookup, axis=0) if head_slice else None

    def _c0_from(h):
        return (h @ c0_head.T) if head_slice else mx.take(backbone.lm_head(h), ids_lookup, axis=-1)

    ids = mx.array([list(prompt_ids), uncond_ids(prompt_ids)])
    cache = make_prompt_cache(backbone)
    h_last = backbone.model(ids, cache=cache)[:, -1]
    narrowed_mx = _c0_from(h_last)

    proj, audio_emb = depth.projection, depth.audio_embeddings
    F = int(forced_codes.shape[0])
    ranks = np.zeros((F, 8), dtype=np.int64)
    for f in range(F):
        narrowed = np.array(narrowed_mx.astype(mx.float32)).astype(np.float64)
        c0_logits = _apply_c0_cfg(narrowed, AR_CFG_SCALE, AR_CFG_TOP_K)  # [1+16384]
        fc = [int(v) for v in forced_codes[f]]
        c0_col = fc[0] + 1  # column 0 == stop; code k at column k+1
        ranks[f, 0] = int((c0_logits > c0_logits[c0_col]).sum())

        c0_tok = mx.array([fc[0] + AUDIO_CODE_OFFSET, fc[0] + AUDIO_CODE_OFFSET])
        seq = [proj(h_last)[:, None, :], proj(embed_tokens(c0_tok))[:, None, :]]
        for index in range(1, 8):
            out = depth(mx.concatenate(seq, axis=1))[:, -1]
            head = np.array(depth.audio_heads[index - 1](out).astype(mx.float32)).astype(np.float64)
            guided = head[1] + (head[0] - head[1]) * AR_CFG_SCALE  # [1024]
            ranks[f, index] = int((guided > guided[fc[index]]).sum())
            if index < 7:
                emb = audio_emb(mx.array([fc[index] + (index - 1) * 1024]))
                seq.append(proj(mx.concatenate([emb, emb], axis=0))[:, None, :])

        emb = _embed_audio_frame(depth, embed_tokens, fc)
        emb2 = mx.concatenate([emb[None], emb[None]], axis=0)[:, None, :]
        h_last = backbone.model(emb2, cache=cache, input_embeddings=emb2)[:, -1]
        narrowed_mx = _c0_from(h_last)
    return ranks


def generate_frames(
    backbone,
    depth,
    prompt_ids: list[int],
    *,
    seed: int = 0,
    max_frames: int = MAX_AUDIO_FRAMES,
    on_frame=None,
    head_slice: bool = False,
):
    """Run the AR loop. Returns (codes [F, 8] int, frame_hidden [F, 32768] mx.array).

    `on_frame(frame_idx, codes8, c0_logits, frame_hidden)` is called per frame for
    parity harnesses that want the intermediate tensors.

    `head_slice` (fable #4): the loop only ever reads the 16385 legal c0 columns of
    the 200k-vocab head, so pre-slice the head weight and compute only those logits
    — 9x less head compute + ~10% less AR weight-streaming, bit-identical result.
    """
    embed_tokens = backbone.model.embed_tokens
    ids_lookup = _c0_logit_ids()
    # sliced c0 head [16385, hidden] — only used when head_slice is on.
    c0_head = mx.take(backbone.lm_head.weight, ids_lookup, axis=0) if head_slice else None

    def _c0_from(hidden_row):  # -> [rows, 16385] mx.array
        if head_slice:
            return hidden_row @ c0_head.T
        return mx.take(backbone.lm_head(hidden_row), ids_lookup, axis=-1)

    ids = mx.array([list(prompt_ids), uncond_ids(prompt_ids)])  # [2, L]
    cache = make_prompt_cache(backbone)
    hidden = backbone.model(ids, cache=cache)
    h_last = hidden[:, -1]
    narrowed_mx = _c0_from(h_last)

    sampling_seed = derive_sampling_seed(seed)
    frame_pos = 0
    codes_out: list[list[int]] = []
    fh_out: list[mx.array] = []

    for _ in range(max_frames):
        narrowed = np.array(narrowed_mx.astype(mx.float32)).astype(np.float64)
        c0_logits = _apply_c0_cfg(narrowed, AR_CFG_SCALE, AR_CFG_TOP_K)
        sampled = int(sample_topk_seeded(c0_logits[None], np.array([sampling_seed]), np.array([frame_pos]))[0])
        if sampled == 0:  # stop
            break
        c0 = sampled - 1
        codes8, depth_hidden = _depth_decode(depth, embed_tokens, h_last, c0, sampling_seed, frame_pos)
        frame_hidden = mx.concatenate([h_last[0], depth_hidden])  # [32768]
        mx.eval(frame_hidden)
        if on_frame is not None:
            on_frame(len(codes_out), codes8, c0_logits, frame_hidden)
        codes_out.append(codes8)
        fh_out.append(frame_hidden)

        emb = _embed_audio_frame(depth, embed_tokens, codes8)
        emb2 = mx.concatenate([emb[None], emb[None]], axis=0)[:, None, :]  # [2,1,hidden]
        hidden = backbone.model(emb2, cache=cache, input_embeddings=emb2)
        h_last = hidden[:, -1]
        narrowed_mx = _c0_from(h_last)
        frame_pos += 8

    codes = np.array(codes_out, dtype=np.int64) if codes_out else np.zeros((0, 8), np.int64)
    frame_hidden = mx.stack(fh_out) if fh_out else mx.zeros((0, 32768))
    return codes, frame_hidden
