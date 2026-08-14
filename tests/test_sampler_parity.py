# SPDX-License-Identifier: Apache-2.0
"""M4 parity: seeded top-k sampler vs an independent spec reference.

The sglang murmur-hash sampler kernel is triton/CUDA-only, so it can't run on
this Mac. Instead we pin correctness against an INDEPENDENT arbitrary-precision
Python-int implementation of the exact documented algorithm (unambiguous, no
uint32-wrapping subtlety), plus a pure-Python float64 gumbel-argmax. Also checks
the properties the model relies on: determinism and batch-invariance.

Run: .venv/bin/python -m pytest minimax-music3-mlx/tests/test_sampler_parity.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

from minimax_music3_mlx import sampler as S  # noqa: E402

M32 = 0xFFFFFFFF


# ---- independent arbitrary-precision reference -------------------------------


def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def _mix(h, k):
    k = (k * 0xCC9E2D51) & M32
    k = _rotl(k, 15)
    k = (k * 0x1B873593) & M32
    h ^= k
    h = _rotl(h, 13)
    h = (h * 5 + 0xE6546B64) & M32
    return h


def _fmix(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def _hash1(seed, pos, col):
    h = 0
    h = _mix(h, seed & M32)
    h = _mix(h, (seed >> 32) & M32)
    h = _mix(h, pos & M32)
    h = _mix(h, col & M32)
    h ^= 16
    return _fmix(h)


def _sample1(logits_row, seed, pos):
    # pure-Python float64 gumbel-argmax over the row
    best_i, best_v = -1, -math.inf
    for col, lg in enumerate(logits_row):
        u = _hash1(seed, pos, col) / float(M32)
        g = -math.log(-max(min(math.log(u), -(2.0**-32)), np.finfo(np.float64).min))
        v = g + float(lg)
        if v > best_v:
            best_v, best_i = v, col
    return best_i


# ---- tests -------------------------------------------------------------------


def test_murmur_hash_matches_bigint_reference():
    seeds = np.array([0, 1, 42, 2**40 + 7, 0xDEADBEEFCAFE], dtype=np.uint64)
    positions = np.array([0, 3, 100, 7, 65535], dtype=np.uint64)
    cols = np.arange(2000, dtype=np.uint64)
    got = S.murmur_hash32(seeds, positions, cols)
    for r, (sd, ps) in enumerate(zip(seeds.tolist(), positions.tolist())):
        for c in [0, 1, 2, 49, 137, 1023, 1999]:
            assert int(got[r, c]) == _hash1(sd, ps, int(c)), (sd, ps, c)


def test_full_gumbel_argmax_matches_reference():
    rng = np.random.default_rng(7)
    vocab = 512
    logits = rng.standard_normal((6, vocab)).astype(np.float32)
    seeds = np.array([0, 1, 2, 99, 12345, 2**33], dtype=np.uint64)
    positions = np.array([0, 1, 5, 5, 900, 3], dtype=np.uint64)
    got = S.multinomial_with_seed(logits, seeds, positions)
    for r in range(logits.shape[0]):
        assert int(got[r]) == _sample1(
            logits[r].tolist(), int(seeds[r]), int(positions[r])
        ), r


def test_topk_restricts_support():
    # with top_k=50, the sampled index must be within the row's top-50 by logit.
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((4, 16384)).astype(np.float32)
    seeds = np.array([5, 6, 7, 8], dtype=np.uint64)
    positions = np.array([10, 11, 12, 13], dtype=np.uint64)
    idx = S.sample_topk_seeded(logits, seeds, positions, top_k=50)
    for r in range(logits.shape[0]):
        top50 = set(np.argsort(logits[r])[-50:].tolist())
        assert int(idx[r]) in top50, r


def test_determinism_and_batch_invariance():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 1024)).astype(np.float32)
    seeds = np.array([11, 22, 33, 44, 55], dtype=np.uint64)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.uint64)

    a = S.sample_topk_seeded(logits, seeds, positions)
    b = S.sample_topk_seeded(logits, seeds, positions)
    assert np.array_equal(a, b), "not deterministic"

    # permute the batch; each row's draw must be unchanged (depends only on its
    # own seed+position, not batch composition).
    perm = np.array([3, 0, 4, 1, 2])
    c = S.sample_topk_seeded(logits[perm], seeds[perm], positions[perm])
    assert np.array_equal(c, a[perm]), "not batch-invariant"


def test_accepts_mlx_input():
    import minimax_music3_mlx  # noqa: F401  (pin TF32 before mlx import)
    import mlx.core as mx

    rng = np.random.default_rng(9)
    logits = rng.standard_normal((2, 1024)).astype(np.float32)
    seeds = np.array([1, 2], dtype=np.uint64)
    positions = np.array([0, 1], dtype=np.uint64)
    from_np = S.sample_topk_seeded(logits, seeds, positions)
    from_mlx = S.sample_topk_seeded(mx.array(logits), seeds, positions)
    assert np.array_equal(from_np, from_mlx)
