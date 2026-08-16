# SPDX-License-Identifier: Apache-2.0
"""Duration policy for MiniMax Music 3 requests.

The checkpoint owns the musical ending and may emit ``<|audio_end|>`` before
the decode ceiling.  A requested duration therefore has two jobs:

* become an explicit structured-caption constraint so the model can plan the
  arrangement around it; and
* provide a conservative minimum frame count before early EOS is allowed.

The production ceiling remains the reference contract's 9,000 frames.  The
model card advertises complete songs up to five minutes, so requested targets
are validated against that supported range rather than the slightly larger
six-minute safety ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .constants import MAX_AUDIO_FRAMES

AUDIO_FRAMES_PER_SECOND = 25
MAX_TARGET_SECONDS = 5 * 60
MIN_TARGET_SECONDS = 1.0
DEFAULT_MINIMUM_RATIO = 0.95

_COLON_DURATION_RE = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?\s*$")
_UNIT_DURATION_RE = re.compile(
    r"^\s*(?:(?P<hours>\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours))?\s*"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes))?\s*"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds))?\s*$",
    re.IGNORECASE,
)
_TEXT_COLON_DURATION_RE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?!\d)")
_TEXT_MINUTE_DURATION_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*[- ]?"
    r"(?:m|min|mins|minute|minutes)\b"
    r"(?:\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
    re.IGNORECASE,
)
_TEXT_SECOND_DURATION_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*[- ]?(?:s|sec|secs|second|seconds)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DurationPolicy:
    """Normalized production duration controls."""

    target_seconds: float | None
    target_frames: int | None
    min_frames: int
    max_frames: int = MAX_AUDIO_FRAMES
    minimum_ratio: float = DEFAULT_MINIMUM_RATIO

    @property
    def target_label(self) -> str | None:
        return format_duration(self.target_seconds) if self.target_seconds is not None else None


def parse_duration_seconds(value: Any) -> float | None:
    """Parse seconds or common human duration strings such as ``4:30``.

    ``None`` and an empty string mean that the model should choose the length.
    Requested targets are limited to the model card's supported five minutes.
    """

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("target duration must be seconds or a duration string, not boolean")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        try:
            seconds = float(raw)
        except ValueError:
            colon = _COLON_DURATION_RE.fullmatch(raw)
            if colon:
                first, second, third = colon.groups()
                if third is None:
                    minutes = int(first)
                    secs = float(second)
                    if secs >= 60:
                        raise ValueError(f"invalid duration seconds field: {value!r}")
                    seconds = minutes * 60 + secs
                else:
                    hours = int(first)
                    minutes = int(second)
                    secs = float(third)
                    if minutes >= 60 or secs >= 60:
                        raise ValueError(f"invalid duration time fields: {value!r}")
                    seconds = hours * 3600 + minutes * 60 + secs
            else:
                units = _UNIT_DURATION_RE.fullmatch(raw)
                if not units or not any(units.groupdict().values()):
                    raise ValueError(
                        "target duration must be seconds or a value like '4:30' or '4m 30s'"
                    )
                seconds = (
                    float(units.group("hours") or 0) * 3600
                    + float(units.group("minutes") or 0) * 60
                    + float(units.group("seconds") or 0)
                )
    else:
        raise ValueError("target duration must be numeric seconds or a duration string")

    if not math.isfinite(seconds):
        raise ValueError("target duration must be finite")
    if seconds < MIN_TARGET_SECONDS:
        raise ValueError(f"target duration must be at least {MIN_TARGET_SECONDS:g} second")
    if seconds > MAX_TARGET_SECONDS:
        raise ValueError(
            f"target duration {seconds:g}s exceeds MiniMax Music 3's supported "
            f"five-minute target ({MAX_TARGET_SECONDS}s)"
        )
    return seconds


def extract_duration_from_text(text: str) -> float | None:
    """Extract an explicit human duration from a natural-language caption.

    This is a resilience fallback for clients that leave the structured duration
    field empty.  Bare numbers are intentionally ignored so BPM, key signatures,
    years, and other musical details cannot be mistaken for song length.
    """

    if not text:
        return None
    colon = _TEXT_COLON_DURATION_RE.search(text)
    if colon:
        return parse_duration_seconds(f"{colon.group(1)}:{colon.group(2)}")
    minutes = _TEXT_MINUTE_DURATION_RE.search(text)
    if minutes:
        seconds = float(minutes.group(1)) * 60 + float(minutes.group(2) or 0)
        return parse_duration_seconds(seconds)
    seconds = _TEXT_SECOND_DURATION_RE.search(text)
    if seconds:
        return parse_duration_seconds(float(seconds.group(1)))
    return None


def build_duration_policy(
    value: Any,
    *,
    minimum_ratio: float = DEFAULT_MINIMUM_RATIO,
    max_frames: int = MAX_AUDIO_FRAMES,
) -> DurationPolicy:
    """Build the target/minimum/safety-ceiling policy for one generation."""

    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be between 0 and 1")
    if max_frames < 1 or max_frames > MAX_AUDIO_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_AUDIO_FRAMES}")
    seconds = parse_duration_seconds(value)
    if seconds is None:
        return DurationPolicy(None, None, 0, max_frames=max_frames, minimum_ratio=minimum_ratio)
    target_frames = max(1, round(seconds * AUDIO_FRAMES_PER_SECOND))
    min_frames = min(target_frames, math.floor(target_frames * minimum_ratio))
    if min_frames > max_frames:
        raise ValueError("requested duration minimum exceeds the generation ceiling")
    return DurationPolicy(
        seconds,
        target_frames,
        min_frames,
        max_frames=max_frames,
        minimum_ratio=minimum_ratio,
    )


def format_duration(seconds: float | None) -> str:
    """Render a compact user/model-facing duration label."""

    if seconds is None:
        return "model-selected"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def ensure_target_duration(caption: str, policy: DurationPolicy) -> str:
    """Insert one canonical target-duration constraint into a caption.

    The line is placed immediately below a Global Metadata heading when one is
    present.  A rewriter is asked to include the same line itself; this guard
    covers LLM omissions and fail-safe raw-caption fallback without duplicating
    a compliant line.
    """

    if policy.target_seconds is None:
        return caption
    line = (
        f"Target Duration: approximately {policy.target_label} "
        f"({policy.target_seconds:g} seconds); plan the full arrangement, section pacing, "
        "instrumental development, and outro to fill this duration."
    )
    target_line = re.compile(r"^\s*(?:#{1,6}\s*)?target duration\s*:", re.IGNORECASE)
    # Replace, rather than trust, an LLM-rendered duration line so a rewrite can
    # never round or silently alter the user's requested value.
    lines = [existing for existing in caption.splitlines() if not target_line.match(existing)]
    for index, existing in enumerate(lines):
        if re.fullmatch(r"\s*(?:#{1,6}\s*)?global metadata\s*:?[\s]*", existing, re.IGNORECASE):
            lines.insert(index + 1, line)
            return "\n".join(lines)
    return f"{line}\n{caption}" if caption else line


__all__ = [
    "AUDIO_FRAMES_PER_SECOND",
    "DEFAULT_MINIMUM_RATIO",
    "DurationPolicy",
    "MAX_TARGET_SECONDS",
    "build_duration_policy",
    "ensure_target_duration",
    "extract_duration_from_text",
    "format_duration",
    "parse_duration_seconds",
]
