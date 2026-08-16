# SPDX-License-Identifier: Apache-2.0
"""Caption rewriter — mandatory prompt preprocessor (per user requirement).

Runs MiniMax's `music-caption-rewriter` skill (bundled under caption_skill/) to
turn a brief user caption + tagged lyrics into the model's preferred structured
caption (Global Metadata / Vocal Details / Arrangement). The skill is a
natural-language reasoning workflow over bundled genre templates — it needs an
LLM. We drive it via any OpenAI-compatible chat endpoint using progressive
disclosure: genre-router -> one/two family indexes -> up to three templates ->
render.

Config (env):
    MM3_CAPTION_API_BASE   OpenAI-compatible base url (e.g. http://127.0.0.1:8189/v1)
    MM3_CAPTION_MODEL      model name to request
    MM3_CAPTION_API_KEY    bearer token (default "none")
    MM3_CAPTION_DISABLE=1  bypass the rewriter (use the raw caption)

Fail-safe: if no endpoint is configured/reachable, or anything errors, the raw
caption is returned unchanged (with a one-time warning) so generation never breaks.
Only bracketed lyric tags are treated as directives; lyric text is never quoted
or reproduced (skill contract).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from .duration import format_duration

logger = logging.getLogger(__name__)

_SKILL = Path(__file__).resolve().parent / "caption_skill"
_REF = _SKILL / "references"
_TPL = _SKILL / "templates"
_FAMILIES = {p.stem[len("index-"):] for p in _REF.glob("index-*.md")}
_warned = False


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _llm_chat(messages: list[dict], *, base: str, model: str, key: str, timeout: float) -> str:
    # /no_think keeps reasoning models (Qwen3 etc.) from spending the budget on
    # hidden reasoning and returning empty content.
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.7,
                       "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.load(r)["choices"][0].get("message", {})
    # tolerate thinking models: content may be null with the answer in reasoning_content
    content = msg.get("content") or msg.get("reasoning_content") or ""
    # strip any leftover <think>...</think> block
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _pick(text: str, options: set[str], limit: int) -> list[str]:
    """Extract known tokens (family keys / template ids) from LLM output, in order."""
    found: list[str] = []
    for tok in re.findall(r"[a-z0-9][a-z0-9-]+(?:_\d{4})?", text.lower()):
        if tok in options and tok not in found:
            found.append(tok)
            if len(found) >= limit:
                break
    return found


def available() -> bool:
    return bool(os.environ.get("MM3_CAPTION_API_BASE")) and os.environ.get("MM3_CAPTION_DISABLE") != "1"


def rewrite_caption(
    caption: str,
    lyrics: str = "",
    *,
    target_seconds: float | None = None,
    timeout: float = 60.0,
) -> str:
    """Rewrite `caption` into a structured MiniMax caption. Returns the original
    caption unchanged on any failure or when no LLM endpoint is configured.

    A requested duration is a first-class constraint: every rewriter stage sees
    it and the final renderer must use it to scale section pacing and the outro.
    """
    global _warned
    if os.environ.get("MM3_CAPTION_DISABLE") == "1":
        return caption
    base = os.environ.get("MM3_CAPTION_API_BASE")
    if not base:
        if not _warned:
            logger.warning("caption-rewriter: MM3_CAPTION_API_BASE unset — using raw caption "
                           "(set it to an OpenAI-compatible endpoint to enable the skill)")
            _warned = True
        return caption
    model = os.environ.get("MM3_CAPTION_MODEL", "default")
    key = os.environ.get("MM3_CAPTION_API_KEY", "none")

    try:
        skill = _read(_SKILL / "SKILL.md")
        router = _read(_REF / "genre-router.md")
        # tags only from lyrics (never the lyric text itself)
        tags = " ".join(re.findall(r"\[[^\]]+\]", lyrics))
        duration = (
            f"{format_duration(target_seconds)} ({target_seconds:g} seconds)"
            if target_seconds is not None
            else "model-selected"
        )
        brief = (
            f"Caption: {caption}\n"
            f"Lyric section tags: {tags or '(none)'}\n"
            f"Target duration: {duration}"
        )
        sys_msg = {"role": "system", "content": skill}

        # Stage 1: route to family/families
        s1 = _llm_chat([sys_msg,
            {"role": "user", "content": f"{brief}\n\n--- genre-router.md ---\n{router}\n\n"
             f"Return ONLY the primary family key (and a secondary only for an explicit fusion), "
             f"from: {sorted(_FAMILIES)}"}],
            base=base, model=model, key=key, timeout=timeout)
        fams = _pick(s1, _FAMILIES, 2) or ["general-pop-ballad"]

        # Stage 2: pick up to 3 template ids from the chosen index(es)
        index_text = "\n\n".join(_read(_REF / f"index-{f}.md") for f in fams)
        tpl_ids = {p.stem for p in _TPL.glob("*.txt")}
        s2 = _llm_chat([sys_msg,
            {"role": "user", "content": f"{brief}\n\n--- family index cards ---\n{index_text}\n\n"
             "Choose up to THREE template IDs with distinct roles (primary style, plus optional "
             "instrumentation/vocal/production references). Return only the IDs."}],
            base=base, model=model, key=key, timeout=timeout)
        chosen = _pick(s2, tpl_ids, 3)

        # Stage 3: render the final structured caption
        tpl_text = "\n\n".join(f"--- {tid} ---\n{_read(_TPL / (tid + '.txt'))}" for tid in chosen)
        s3 = _llm_chat([sys_msg,
            {"role": "user", "content": f"{brief}\n\n--- selected templates ---\n{tpl_text}\n\n"
             "Now render ONE final structured caption for this request, following the skill's "
             "output format (Global Metadata, Vocal Details, Arrangement). Preserve every explicit "
             "user constraint and any instrumental request. When a target duration is supplied, add "
             "a 'Target Duration:' line under Global Metadata and scale the section-by-section "
             "arrangement, instrumental development, and outro to fill it. Output ONLY the caption "
             "text."}],
            base=base, model=model, key=key, timeout=timeout)
        out = s3.strip()
        if len(out) < 40:  # implausible rewrite -> keep original
            return caption
        logger.info("caption-rewriter: rewrote via families=%s templates=%s (%d->%d chars)",
                    fams, chosen, len(caption), len(out))
        return out
    except Exception as e:  # never break generation
        if not _warned:
            logger.warning("caption-rewriter: endpoint error (%s) — using raw caption", e)
            _warned = True
        return caption
