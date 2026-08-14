#!/bin/bash
# Fetch MiniMax's 1000 caption templates for the caption-rewriter skill.
# The skill logic (SKILL.md, genre-router, indexes) ships in this repo; the
# templates are not re-hosted — this pulls them from the upstream MiniMax repo.
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/minimax_music3_mlx/caption_skill/templates"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Fetching caption templates from MiniMax-AI/MiniMax-Music3 ..."
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/MiniMax-AI/MiniMax-Music3 "$TMP/repo"
git -C "$TMP/repo" sparse-checkout init --no-cone
git -C "$TMP/repo" sparse-checkout set 'skills/music-caption-rewriter/templates/*'
git -C "$TMP/repo" checkout HEAD

mkdir -p "$DEST"
cp "$TMP/repo/skills/music-caption-rewriter/templates/"*.txt "$DEST/"
echo "Done: $(ls "$DEST"/*.txt | wc -l | tr -d ' ') templates -> $DEST"
echo "(The rewriter also works without templates — it synthesizes from the index cards.)"
