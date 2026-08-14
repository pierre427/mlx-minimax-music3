# SPDX-License-Identifier: Apache-2.0
"""M1 parity: our vendored prompt/tokenizer logic vs the sglang-omni reference.

Two oracles:
  1. String parity — load the reference prompt.py directly from the sglang tree
     (pure stdlib, no deps) and diff clean_caption / normalize_lyrics /
     build_prompt over a battery of inputs including the shipped demo caption+lyrics.
  2. Tokenizer id parity — validate the 9 special-token ids against the actual
     HF music tokenizer, and confirm build_prompt tokenizes without splitting any
     special token.

Run: .venv/bin/python -m pytest minimax-music3-mlx/tests/test_prompt_parity.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

from minimax_music3_mlx import prompt as port_prompt  # noqa: E402

REF_PROMPT_PY = Path(
    "/Users/Shared/src/sglang-omni/sglang_omni/models/minimax_music3/prompt.py"
)
TOKENIZER_DIR = PORT_ROOT / "weights" / "qwen_7B" / "qwen3-8B-tokenizer-music"


def _load_reference():
    spec = importlib.util.spec_from_file_location("_ref_prompt", REF_PROMPT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- inputs: the shipped demo + adversarial edge cases -----------------------

DEMO_CAPTION = (
    "Global Metadata\n"
    "Basic Attributes: bpm is 92. key is E, and scale is minor. Electric Blues / Blues Rock.\n"
    "**Vocal Details**\n"
    "* Vocal Gender: Male\n"
    "Sonics: warm and <|mood gritty|> with • bullet residue and    quad-space."
)
DEMO_LYRICS = (
    "[verse] I'm learning how to fill up [pre-chorus] Breathe a little deeper "
    "[BASS-QUARTET-RUMBLES-IN] Every heartbeat ^ washed away the pain"
)

EDGE_CASES_CAPTION = [
    "",
    "no tags at all, plain text",
    "<|tempo fast|> then <|key C minor|> and a lone <|bareword|>",
    "# Heading\n## Sub\n- item one\n+ item two\n*emph* and **bold** text",
    "line1\n\n\nline2\n\n\n\nline3",  # multiple blank lines -> collapse {2,}
    "trailing spaces here   \nand a --- rule\n___\n***",
    "unicode café — dash, quotes “x” and 里程碑",
]
EDGE_CASES_LYRICS = [
    "",
    "[intro] [verse] stacked leading tags then words that stay",
    "no brackets just words with ^ carets ^ inside",
    "[Chorus] Mixed CASE Tag] weird ] spacing [ bracket",
    "[a][b][c] contiguous [d] e",
    "line with ] and [ both ^ markers ^ present",
]


@pytest.fixture(scope="module")
def ref():
    if not REF_PROMPT_PY.exists():
        pytest.skip(f"reference not found: {REF_PROMPT_PY}")
    return _load_reference()


def test_special_token_table_matches_reference(ref):
    assert port_prompt.SPECIAL_TOKEN_IDS == ref.SPECIAL_TOKEN_IDS
    assert port_prompt.AUDIO_CODE_OFFSET == ref.AUDIO_CODE_OFFSET


def test_clean_caption_parity(ref):
    for c in [DEMO_CAPTION, *EDGE_CASES_CAPTION]:
        assert port_prompt.clean_caption(c) == ref.clean_caption(c), repr(c)


def test_normalize_lyrics_parity(ref):
    for lyr in [DEMO_LYRICS, *EDGE_CASES_LYRICS]:
        assert port_prompt.normalize_lyrics(lyr) == ref.normalize_lyrics(lyr), repr(lyr)


def test_build_prompt_parity(ref):
    for c in [DEMO_CAPTION, *EDGE_CASES_CAPTION]:
        for lyr in [DEMO_LYRICS, *EDGE_CASES_LYRICS]:
            assert port_prompt.build_prompt(c, lyr) == ref.build_prompt(c, lyr)


# ---- tokenizer id parity -----------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR.exists():
        pytest.skip(f"tokenizer dir not found: {TOKENIZER_DIR}")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def test_validate_tokenizer_ids(tokenizer):
    # Raises on mismatch; passing means all 9 ids line up with the checkpoint.
    port_prompt.validate_tokenizer_ids(tokenizer)


def test_special_tokens_are_atomic(tokenizer):
    # Each special token must encode to exactly its single reserved id.
    for token, expected in port_prompt.SPECIAL_TOKEN_IDS.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert ids == [expected], f"{token!r} -> {ids}, expected [{expected}]"
