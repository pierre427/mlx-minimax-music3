# SPDX-License-Identifier: Apache-2.0
"""No-model tests for duration parsing, prompt injection, and early-EOS policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

from minimax_music3_mlx import caption_rewriter  # noqa: E402
from minimax_music3_mlx.duration import (  # noqa: E402
    build_duration_policy,
    ensure_target_duration,
    extract_duration_from_text,
    parse_duration_seconds,
)
from minimax_music3_mlx.generation import _mask_stop_before_minimum  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (270, 270.0),
        ("270", 270.0),
        ("4:30", 270.0),
        ("00:04:30", 270.0),
        ("4m 30s", 270.0),
        ("4.5 minutes", 270.0),
        (None, None),
        ("", None),
    ],
)
def test_parse_duration_seconds(raw, expected):
    assert parse_duration_seconds(raw) == expected


@pytest.mark.parametrize("raw", [True, 0, "4:75", "not a duration", 301, float("inf")])
def test_parse_duration_rejects_invalid_or_unsupported_targets(raw):
    with pytest.raises(ValueError):
        parse_duration_seconds(raw)


def test_duration_policy_uses_fixed_ceiling_and_95_percent_minimum():
    policy = build_duration_policy("4:30")
    assert policy.target_seconds == 270
    assert policy.target_frames == 6750
    assert policy.min_frames == 6412
    assert policy.max_frames == 9000
    assert policy.target_label == "4:30"


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Make this a 4:30 dark country song", 270.0),
        ("A four-minute concept expressed as 4 minutes", 240.0),
        ("Extended instrumental lasting 4 minutes 30 seconds", 270.0),
        ("A concise 90-second cue", 90.0),
        ("Dark country at 88 BPM in 4/4", None),
    ],
)
def test_extract_explicit_duration_from_caption(caption, expected):
    assert extract_duration_from_text(caption) == expected


def test_target_duration_is_injected_once_under_global_metadata():
    policy = build_duration_policy(270)
    caption = "### Global Metadata\nDark country blues.\n### Vocal Details\nFemale duet."
    expanded = ensure_target_duration(caption, policy)
    lines = expanded.splitlines()
    assert lines[1].startswith("Target Duration: approximately 4:30 (270 seconds)")
    assert ensure_target_duration(expanded, policy) == expanded


def test_stop_token_is_masked_only_before_minimum_without_mutating_input():
    logits = np.array([9.0, 2.0, 1.0], dtype=np.float64)
    masked = _mask_stop_before_minimum(logits, generated_frames=99, min_frames=100)
    assert np.isneginf(masked[0])
    assert logits[0] == 9.0
    assert masked is not logits
    unmasked = _mask_stop_before_minimum(logits, generated_frames=100, min_frames=100)
    assert unmasked is logits


def test_rewriter_receives_duration_at_every_stage(monkeypatch):
    calls: list[list[dict]] = []
    outputs = iter([
        "general-pop-ballad",
        "",
        "### Global Metadata\nTarget Duration: approximately 4:30 (270 seconds).\n"
        "### Vocal Details\nA restrained vocal performance.\n"
        "### Arrangement\nA complete section-aware long-form arrangement.",
    ])

    def fake_chat(messages, **_kwargs):
        calls.append(messages)
        return next(outputs)

    monkeypatch.setenv("MM3_CAPTION_API_BASE", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(caption_rewriter, "_llm_chat", fake_chat)
    result = caption_rewriter.rewrite_caption(
        "dark country blues",
        "[verse]\noriginal lyric\n[chorus]\noriginal refrain",
        target_seconds=270,
    )
    assert "Target Duration: approximately 4:30" in result
    assert len(calls) == 3
    for messages in calls:
        assert "Target duration: 4:30 (270 seconds)" in messages[-1]["content"]
        assert "original lyric" not in messages[-1]["content"]
