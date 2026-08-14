# SPDX-License-Identifier: Apache-2.0
"""M4: deterministic seeded top-k sampling + CFG combine for the AR stage.

Faithful reimplementation of sglang's `multinomial_with_seed`
(python/sglang/srt/layers/sampler.py + kernels/ops/sampling/murmur_hash.py):
a MurmurHash3-based counter RNG produces one uniform per (seed, position,
token_id), turned into Gumbel noise; the argmax of `logits + gumbel` over the
top-k-masked vocabulary is the sample. This makes sampling deterministic and
batch-invariant (the draw depends only on the row's seed+position, not on batch
composition).

Runs on CPU in float64 — the reference notes float64 is critical for the gumbel
noise, and Metal/MLX has no float64. Sampling is a tiny per-draw op (a few rows x
<=16384 logits), so this is not a throughput concern. Accepts mlx or numpy
logits; returns numpy int64 indices.

Bit-manipulation is validated against an independent arbitrary-precision
reference in tests/test_sampler_parity.py.
"""

from __future__ import annotations

import numpy as np

from .constants import AR_CFG_SCALE, AR_CFG_TOP_K, DEFAULT_DIT_CFG_SCALE  # noqa: F401

_U32 = np.uint32(0xFFFFFFFF)
_UINT32_MAX = float(np.iinfo(np.uint32).max)


def _rotl32(x: np.ndarray, r: int) -> np.ndarray:
    # native uint32 wrapping: bits shifted off the top are exactly the ones that
    # reappear via (x >> (32 - r)), so no explicit mask is needed.
    return ((x << np.uint32(r)) | (x >> np.uint32(32 - r))).astype(np.uint32)


def _murmur3_mix(h: np.ndarray, k: np.ndarray) -> np.ndarray:
    c1 = np.uint32(0xCC9E2D51)
    c2 = np.uint32(0x1B873593)
    k = (k * c1).astype(np.uint32)
    k = _rotl32(k, 15)
    k = (k * c2).astype(np.uint32)
    h = (h ^ k).astype(np.uint32)
    h = _rotl32(h, 13)
    h = (h * np.uint32(5) + np.uint32(0xE6546B64)).astype(np.uint32)
    return h


def _fmix32(h: np.ndarray) -> np.ndarray:
    h = (h ^ (h >> np.uint32(16))).astype(np.uint32)
    h = (h * np.uint32(0x85EBCA6B)).astype(np.uint32)
    h = (h ^ (h >> np.uint32(13))).astype(np.uint32)
    h = (h * np.uint32(0xC2B2AE35)).astype(np.uint32)
    h = (h ^ (h >> np.uint32(16))).astype(np.uint32)
    return h


def murmur_hash32(
    seed: np.ndarray, positions: np.ndarray, col_indices: np.ndarray
) -> np.ndarray:
    """[n] seed(uint64), [n] positions, [m] col_indices -> [n, m] uint32 hash."""
    seed = seed.astype(np.uint64)
    seed_lo = (seed & np.uint64(0xFFFFFFFF)).astype(np.uint32)[:, None]
    seed_hi = ((seed >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)[:, None]
    pos = positions.astype(np.uint32)[:, None]
    col = col_indices.astype(np.uint32)[None, :]

    h = np.zeros((seed.shape[0], col_indices.shape[0]), dtype=np.uint32)
    h = _murmur3_mix(h, np.broadcast_to(seed_lo, h.shape).astype(np.uint32))
    h = _murmur3_mix(h, np.broadcast_to(seed_hi, h.shape).astype(np.uint32))
    h = _murmur3_mix(h, np.broadcast_to(pos, h.shape).astype(np.uint32))
    h = _murmur3_mix(h, np.broadcast_to(col, h.shape).astype(np.uint32))
    h = (h ^ np.uint32(16)).astype(np.uint32)
    return _fmix32(h)


def _gumbel_from_hash(hashed: np.ndarray) -> np.ndarray:
    """Match multinomial_with_seed's float64 gumbel noise exactly."""
    x = hashed.astype(np.float64) / _UINT32_MAX
    # x.log().clamp(min=finfo.min, max=-2^-32).neg()  ->  -clamp(log x, ...)
    x = np.log(x)
    x = np.clip(x, np.finfo(np.float64).min, -(2.0**-32))
    x = -x
    # x.log().neg()  ->  -log(that)  == gumbel noise
    return -np.log(x)


def _to_np(a) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.array(a)  # mlx arrays convert via __array__


def multinomial_with_seed(
    logits, seed: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """[n, m] logits, [n] seed, [n] positions -> [n] argmax(logits + gumbel)."""
    lg = _to_np(logits).astype(np.float64)
    n, m = lg.shape
    hashed = murmur_hash32(np.asarray(seed), np.asarray(positions), np.arange(m))
    x = _gumbel_from_hash(hashed) + lg
    return np.argmax(x, axis=-1).astype(np.int64)


def _topk_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
    """-inf everything below the row's top-k threshold (matches sample_topk_seeded)."""
    vals = np.nan_to_num(logits.astype(np.float64), nan=-1e9, posinf=1e9, neginf=-1e9)
    if top_k >= vals.shape[-1]:
        return vals
    kth = np.partition(vals, -top_k, axis=-1)[..., -top_k, None]
    return np.where(vals < kth, -np.inf, vals)


def sample_topk_seeded(
    logits,
    seed: np.ndarray,
    positions: np.ndarray,
    *,
    top_k: int = AR_CFG_TOP_K,
) -> np.ndarray:
    """Top-k mask then seeded Gumbel-argmax. [n,m] -> [n] sampled indices."""
    masked = _topk_mask(_to_np(logits), top_k)
    return multinomial_with_seed(masked, np.asarray(seed), np.asarray(positions))


# ---- classifier-free guidance combine ---------------------------------------
# Exact c0 double-top-k masking and the depth-CFG wiring are exercised end-to-end
# in the AR loop (M2b); these are the arithmetic combines they use.


def cfg_combine(cond_logits, uncond_logits, scale: float) -> np.ndarray:
    """uncond + scale * (cond - uncond), elementwise (Music3 AR/depth CFG form)."""
    c = _to_np(cond_logits).astype(np.float64)
    u = _to_np(uncond_logits).astype(np.float64)
    return u + scale * (c - u)
