# SPDX-License-Identifier: Apache-2.0
"""Chunk boundary + overlap helpers for multi-window synthesis.

Vendored verbatim from sglang-omni (models/minimax_music3/chunking.py). Fixed
200-frame windows with 50% (100-frame) overlap; the acoustic stage pins each
window's leading 172 mel-frames to the previous window's latent tail for
continuity, then crops in sample space so cropped chunks tile without overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import AR_CHUNK_FRAMES, AR_CHUNK_HOP_FRAMES


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: int
    end: int
    is_first: bool
    is_last: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_windows(frames: int) -> list[ChunkWindow]:
    """Fixed-size windows with a 50% overlap."""
    frames = int(frames)
    if frames == 0:
        return []
    if frames <= AR_CHUNK_FRAMES:
        return [ChunkWindow(0, 0, frames, True, True)]
    windows: list[ChunkWindow] = []
    index = 0
    start = 0
    while start < frames:
        end = min(start + AR_CHUNK_FRAMES, frames)
        windows.append(ChunkWindow(index=index, start=start, end=end,
                                   is_first=index == 0, is_last=end >= frames))
        if end >= frames:
            break
        index += 1
        start += AR_CHUNK_HOP_FRAMES
    return windows


_DAV_HOP_SAMPLES = 512
_WINDOW_MEL_FRAMES = 344
_BLEND_MEL_FRAMES = _WINDOW_MEL_FRAMES // 4


def overlap_mel_length() -> int:
    return _WINDOW_MEL_FRAMES // 2


def crop_sample_bounds(window: ChunkWindow) -> tuple[int, int]:
    """(left_samples, right_samples) to trim from a decoded chunk."""
    left = 0 if window.is_first else _BLEND_MEL_FRAMES * _DAV_HOP_SAMPLES
    right = (0 if window.is_last
             else (_WINDOW_MEL_FRAMES - _BLEND_MEL_FRAMES) * _DAV_HOP_SAMPLES)
    return left, right


__all__ = ["ChunkWindow", "chunk_windows", "crop_sample_bounds", "overlap_mel_length"]
