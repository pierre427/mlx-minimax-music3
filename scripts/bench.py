#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Instrumented / traced generation for the MLX MiniMax-Music3 port.

Times every stage (load, AR loop + per-frame, condition encoder, DiT solve +
per-step, vocoder, resample), records peak GPU memory, and writes a JSON trace +
a summary table. Used to measure the fable optimization sweep one lever at a time.

Optimization flags (each maps to a fable recommendation; baseline = none set):
    --tf32            MLX_ENABLE_TF32=1 for the fp32 DiT -> M5 NAX kernels   (fable #1)
    --quant-ar Q      quantize backbone + depth decoder to Q bits (e.g. 6)   (fable #2)
    --head-slice      slice lm_head to the 16385 legal c0 columns            (fable #4)
    --precompute      precompute DiT temb/RoPE/nearest-index tables          (fable #6)
    --compile         mx.compile the depth step and DiT block                (fable #7)
    --cpu-dav         run the DAV vocoder + resample on the CPU stream        (fable #5)

The workload (prompt, seed, frame count, DiT steps) is fixed so runs are
comparable. Irish-jig caption + original lyrics (no copyrighted text).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

# --- optimization flags must be resolved BEFORE mlx is imported (TF32 latches) ---
_ARGV = sys.argv[1:]
if "--tf32" in _ARGV:
    os.environ["MLX_ENABLE_TF32"] = "1"           # override the package default
else:
    os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

# Irish jig, fast 6/8, trad instrumentation. Lyrics below are original (self-authored).
CAPTION = (
    "A lively traditional Irish jig in 6/8 time at a fast tempo around 120 bpm, key of D major. "
    "Bright and celebratory folk dance music with fiddle carrying the melody, tin whistle "
    "ornaments, button accordion, acoustic guitar, and a driving bodhran drum. Energetic, "
    "spirited, and danceable with a rollicking pub-session feel."
)
LYRICS = (
    "[verse]\n"
    "Up with the lark and away we go\n"
    "Down by the river where the wild winds blow\n"
    "[chorus]\n"
    "Dance till the morning, spin round the floor\n"
    "Kick up your heels and call for more\n"
)


def _peak_mem_gb() -> float:
    for fn in ("get_peak_memory", "get_active_memory"):
        f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", object()), fn, None)
        if f is not None:
            try:
                return f() / 1e9
            except Exception:
                pass
    return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--weights", type=Path, default=PORT_ROOT / "weights")
    p.add_argument("--out-wav", type=Path, default=None)
    p.add_argument("--trace-dir", type=Path,
                   default=Path("/private/tmp/claude-501/-Users-pierrelamy-Desktop-mlx-uag/"
                               "0f6784e9-87a2-470e-889a-dd39fc3fcf4b/scratchpad/mm3_traces"))
    # optimization toggles (parsed for the record; effects applied in the port where wired)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--quant-ar", type=int, default=0)
    p.add_argument("--head-slice", action="store_true")
    p.add_argument("--precompute", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cpu-dav", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from minimax_music3_mlx.backbone import load_backbone
    from minimax_music3_mlx.condition_encoder import load_condition_encoder
    from minimax_music3_mlx.depth_decoder import load_depth_decoder
    from minimax_music3_mlx.dit import load_dit
    from minimax_music3_mlx.generation import generate_frames
    from minimax_music3_mlx.pipeline import resample_44k_to_32k, solve_latent, write_wav
    from minimax_music3_mlx.prompt import build_prompt, validate_tokenizer_ids
    from minimax_music3_mlx.vocoder import load_vocoder

    W = args.weights
    trace: dict = {"label": args.label, "flags": {
        "tf32": args.tf32, "quant_ar": args.quant_ar, "head_slice": args.head_slice,
        "precompute": args.precompute, "compile": args.compile, "cpu_dav": args.cpu_dav},
        "config": {"seed": args.seed, "max_frames": args.max_frames, "num_steps": args.num_steps},
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"), "stages": {}}

    tok = AutoTokenizer.from_pretrained(str(W / "qwen_7B" / "qwen3-8B-tokenizer-music"))
    validate_tokenizer_ids(tok)
    ids = tok.encode(build_prompt(CAPTION, LYRICS), add_special_tokens=False)
    trace["prompt_tokens"] = len(ids)

    # ---- load ----
    bb_dtype = mx.bfloat16
    t = time.perf_counter()
    bb, _ = load_backbone(W / "qwen_7B" / "qwen_7B", dtype=bb_dtype)
    dp = load_depth_decoder(W / "qwen_7B" / "qwen_7B", dtype=bb_dtype)
    ce = load_condition_encoder(W / "flowmatching_vae.pth", dtype=mx.float32)
    dit = load_dit(W / "flowmatching_vae.pth", dtype=mx.float32)
    voc = load_vocoder(W / "dav.pth", dtype=mx.float32)
    mx.eval(bb.parameters(), dp.parameters(), ce.parameters(), dit.parameters(), voc.parameters())
    trace["stages"]["load_s"] = time.perf_counter() - t

    if args.quant_ar:
        import mlx.nn as nn
        t = time.perf_counter()
        nn.quantize(bb, bits=args.quant_ar, group_size=64)
        nn.quantize(dp, bits=args.quant_ar, group_size=64)
        mx.eval(bb.parameters(), dp.parameters())
        trace["stages"]["quantize_s"] = time.perf_counter() - t

    # ---- AR loop (per-frame timestamps) ----
    frame_ts: list[float] = []
    ar_t0 = time.perf_counter()

    def on_frame(_i, _codes, _c0, _fh):
        frame_ts.append(time.perf_counter())

    codes, frame_hidden = generate_frames(bb, dp, ids, seed=args.seed,
                                          max_frames=args.max_frames, on_frame=on_frame,
                                          head_slice=args.head_slice)
    ar_total = time.perf_counter() - ar_t0
    n_frames = int(frame_hidden.shape[0])
    per_frame = np.diff([ar_t0] + frame_ts) if frame_ts else np.array([])
    trace["stages"]["ar_total_s"] = ar_total
    trace["stages"]["ar_frames"] = n_frames
    trace["stages"]["ar_ms_per_frame_mean"] = float(per_frame.mean() * 1000) if len(per_frame) else None
    trace["stages"]["ar_ms_per_frame_p50"] = float(np.median(per_frame) * 1000) if len(per_frame) else None

    # ---- condition encoder ----
    t = time.perf_counter()
    align = ce(frame_hidden[None])
    mx.eval(align)
    trace["stages"]["condition_s"] = time.perf_counter() - t
    mel_len = int(align.shape[1])
    trace["mel_len"] = mel_len

    # ---- DiT solve (per-step) ----
    dt = 1.0 / args.num_steps
    key = mx.random.key(args.seed)
    x = mx.random.normal((1, 128, mel_len), key=key).astype(align.dtype)
    cond_cfg = mx.concatenate([align, mx.zeros_like(align)], axis=0)
    step_ms: list[float] = []
    solve_t0 = time.perf_counter()
    for step in range(args.num_steps):
        s0 = time.perf_counter()
        tt = mx.full((2,), step / args.num_steps, dtype=x.dtype)
        d = dit(mx.broadcast_to(x, (2, 128, mel_len)), tt, cond_cfg)
        d = 1.7 * d[0:1] + (1.0 - 1.7) * d[1:2]
        x = x + dt * d
        mx.eval(x)
        step_ms.append((time.perf_counter() - s0) * 1000)
    trace["stages"]["dit_solve_s"] = time.perf_counter() - solve_t0
    trace["stages"]["dit_ms_per_step_mean"] = float(np.mean(step_ms))
    latent = x

    # ---- vocoder ----
    voc_dev = mx.cpu if args.cpu_dav else mx.gpu
    t = time.perf_counter()
    with mx.stream(voc_dev):
        wave44 = voc(latent)
        mx.eval(wave44)
    trace["stages"]["vocoder_s"] = time.perf_counter() - t

    # ---- resample ----
    wave44_np = np.array(wave44[0].astype(mx.float32))
    t = time.perf_counter()
    wave32 = resample_44k_to_32k(wave44_np)
    trace["stages"]["resample_s"] = time.perf_counter() - t

    trace["peak_mem_gb"] = _peak_mem_gb()
    trace["audio"] = {"samples": int(wave32.shape[-1]), "seconds": wave32.shape[-1] / 32000,
                      "range": [float(wave32.min()), float(wave32.max())],
                      "rms": float(np.sqrt((wave32.astype(np.float64) ** 2).mean()))}
    # end-to-end compute wall (AR + condition + solve + vocoder + resample)
    st = trace["stages"]
    trace["stages"]["gen_total_s"] = (st["ar_total_s"] + st["condition_s"]
                                      + st["dit_solve_s"] + st["vocoder_s"] + st["resample_s"])

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    tp = args.trace_dir / f"trace_{args.label}.json"
    tp.write_text(json.dumps(trace, indent=2))
    if args.out_wav:
        write_wav(args.out_wav, wave32)

    # ---- summary ----
    print(f"\n=== {args.label} (TF32={trace['mlx_enable_tf32']} quant_ar={args.quant_ar}) ===")
    print(f"  load           {st['load_s']:6.1f}s")
    print(f"  AR  ({n_frames}f)     {st['ar_total_s']:6.1f}s   ({st['ar_ms_per_frame_mean']:.1f} ms/frame)")
    print(f"  condition      {st['condition_s']:6.3f}s")
    print(f"  DiT solve      {st['dit_solve_s']:6.1f}s   ({st['dit_ms_per_step_mean']:.1f} ms/step, mel={mel_len})")
    print(f"  vocoder        {st['vocoder_s']:6.2f}s   ({'CPU' if args.cpu_dav else 'GPU'})")
    print(f"  resample       {st['resample_s']:6.3f}s")
    print(f"  --------------------------")
    print(f"  gen total      {st['gen_total_s']:6.1f}s   -> {trace['audio']['seconds']:.2f}s audio")
    print(f"  peak mem       {trace['peak_mem_gb']:6.1f} GB")
    print(f"  audio rms={trace['audio']['rms']:.4f} range={trace['audio']['range']}")
    print(f"  trace -> {tp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
