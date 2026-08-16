# mlx-minimax-music3

An Apple-Silicon ([MLX](https://github.com/ml-explore/mlx)) port of
[**MiniMaxAI/MiniMax-Music3**](https://huggingface.co/MiniMaxAI/MiniMax-Music3) —
open-weights **text → music** generation running fully on-device on Apple Silicon
(built and validated on an M5 Max).

Give it a **caption** (style/mood/instrumentation) and optional **lyrics** with
section tags; it generates a 32 kHz stereo track. Every stage is a from-scratch
MLX reimplementation, parity-verified against the reference at fp32.

> Text → music only. The open weights don't include the hosted platform's
> audio-reference "cover / style-transfer" feature (the reference serving code
> declares `supports_reference_audio=False`).

## What it is

A hybrid **autoregressive + flow-matching** pipeline:

```
caption + lyrics
  → Qwen3-8B AR backbone  → per-frame global hidden + first RVQ code (c0)
  → 4-layer RVQ depth decoder → residual codes c1..c7  → 32768-d frame hidden
  → condition encoder      → latent-aligned conditioning
  → flow-matching DiT (36L, 30-step Euler, CFG 1.7, fp32)  → VAE latent
  → DAV vocoder (DAC decoder, ×512)  → 44.1 kHz stereo
  → multi-window overlap-add → resample → 32 kHz MP3
```

The AR backbone reuses upstream **mlx-lm**'s Qwen3 (verified against v0.31.3); the
rest is implemented here.

## Parity

Nine `pytest` suites diff each module against the reference (sglang-omni /
diffusers) at fp32:

| Module | max abs error |
|---|---|
| prompt/tokenizer | exact (byte-for-byte) |
| Qwen3-8B backbone (last hidden) | 4.6e-5 |
| RVQ depth decoder | 2.9e-5 |
| seeded sampler | bit-exact (MurmurHash3 Gumbel-argmax) |
| condition encoder | 5e-6 |
| flow-matching DiT (fp32) | 1.9e-5 |
| DAV vocoder | ~1e-6 |

A reference-rank harness (`scripts/fidelity_rank.py`) validates AR-half
quantization by teacher-forcing a reference trajectory and measuring where each
code ranks under the quantized model.

## Performance (M5 Max, 150-frame / 6 s clip)

The two levers that matter, measured end-to-end (not microbenchmarks):

| Config | AR | DiT solve | total |
|---|---|---|---|
| baseline (fp32 DiT, bf16 backbone) | 14.8 s | 12.2 s | 27.8 s |
| **+ `MLX_ENABLE_TF32=1`** (fp32 DiT → M5 NAX kernels) | 14.8 s | **4.7 s** | 20.3 s |
| **+ q6 AR** (backbone + depth) | **5.7 s** | 4.7 s | **11.0 s** |

**~2.5× end-to-end** at negligible quality cost (TF32 correlates 1.0 with the
reference's own default; q6 keeps 99.5% of reference codes samplable). The fp32
DiT is fidelity-critical — don't run it bf16 (measured ~9 dB drift). `MLX_ENABLE_TF32=1`
is the production default; parity tests pin it off.

## Install

```bash
git clone https://github.com/pierre427/mlx-minimax-music3
cd mlx-minimax-music3
pip install -e .          # mlx, mlx-lm, transformers, torch, scipy, ...
brew install ffmpeg       # MP3 encoding
./fetch_skill.sh          # optional: caption-rewriter templates

# download the weights (57 GB) next to the package, or pass --weights
huggingface-cli download MiniMaxAI/MiniMax-Music3 --local-dir weights
```

## Usage

**CLI:**
```bash
python scripts/generate.py \
  --caption "upbeat Irish jig, 6/8, fiddle, tin whistle, bodhran, D major" \
  --lyrics $'[verse]\nup and away we go\n[chorus]\ndance till the morning' \
  --target-duration 4:30 --seed 3 --quant-ar 6 --out jig.mp3
```

**HTTP API** (OpenAI-audio style):
```bash
python scripts/server.py --port 8600 --quant-ar 6
curl -s -X POST http://127.0.0.1:8600/v1/audio/music -H "Content-Type: application/json" \
  -d '{"caption":"warm acoustic folk waltz, fingerpicked guitar, wistful","lyrics":"[verse]\nsoft evening light","target_duration":"4:30","seed":1}' \
  --output out.mp3
```

Song duration is a semantic target, not a generation ceiling. Production always
uses the full 9,000-frame safety cap and suppresses early model EOS until 95% of
an explicit target. The API accepts `target_duration`, `target_seconds`, or
`duration_seconds`; CLI durations may be seconds or `M:SS`. If no duration field
is supplied, an explicit length in the caption such as “a 4:30 song” is inferred.

**MCP** (for agents — Claude / Codex / etc.):
```bash
python scripts/mcp_server.py --port 8770 --api http://127.0.0.1:8600
# tools: generate_music, get_music_job, music_style_guide
```

`generate_music` accepts `duration_seconds` and returns a durable job ID
immediately. Poll `get_music_job` for `queued`, `running`, `succeeded`, or
`failed`; successful jobs contain the saved MP3 path and generation metadata.

## Caption rewriter (optional)

Bundles MiniMax's `music-caption-rewriter` skill to expand even a very brief
caption into the model's preferred structured format. The rewriter preserves an
explicit duration and scales its arrangement guidance toward that target. It
needs an OpenAI-compatible LLM endpoint; without one the raw caption plus the
canonical duration constraint is used, so generation never breaks:

```bash
export MM3_CAPTION_API_BASE=http://127.0.0.1:8000/v1
export MM3_CAPTION_MODEL=your-local-model
```

## Notes

- Lyrics: put each `[tag]` **alone on its own line** — a tag on the same line as
  lyric text silently drops that line (a model quirk).
- Sampling is fixed (no temperature/top_p); change `seed` for alternate takes.
- Requested duration is capped at five minutes. The separate 9,000-frame limit
  is a runaway safety ceiling and is always supplied in production.

## License & credits

Apache-2.0. This repo contains **no model weights**. See [NOTICE](NOTICE) for
attribution — it builds on MiniMax-Music3, sglang-omni, diffusers, minimax-h3-mlx,
and mlx / mlx-lm.
