# SPDX-License-Identifier: Apache-2.0
"""M8: end-to-end MiniMax-Music3 pipeline — prompt -> 32 kHz stereo WAV.

Chains the AR loop (M2b) -> condition encoder (M5) -> flow-matching DiT solve
(M6, 30-step uniform Euler with CFG 1.7) -> DAV vocoder (M7) -> resample
44.1->32 kHz -> WAV.

The flow-matching solve is the sglang closed form (scheduler collapses to plain
Euler at shift=1.0): x0 ~ N(0,1); for step k, t=k/N, x += (1/N)*v where
v = cfg*v_cond + (1-cfg)*v_uncond, v_uncond conditioned on zeros.

This module does the single-window path (<= 200 AR frames). Multi-window
overlap-add for long clips (chunking.py / crop_sample_bounds) is a follow-on.
The DiT runs fp32 (fidelity); the backbone may run bf16.
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from .chunking import chunk_windows, crop_sample_bounds, overlap_mel_length
from .constants import (
    DAV_SAMPLE_RATE,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
    MAX_AUDIO_FRAMES,
    OUTPUT_SAMPLE_RATE,
)
from .generation import generate_frames


def derive_dit_seed(seed: int, chunk_idx: int) -> int:
    """Per-chunk DiT noise seed (matches sglang acoustic._derive_seed(seed,'dit',idx))."""
    d = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    d.update(int(seed).to_bytes(8, "little", signed=False))
    for part in ("dit", chunk_idx):
        raw = str(part).encode("utf-8")
        d.update(len(raw).to_bytes(4, "little"))
        d.update(raw)
    return int.from_bytes(d.digest(), "little") & ((1 << 63) - 1)


def solve_latent(dit, align: mx.array, *, num_steps: int = DEFAULT_DIT_STEPS,
                 cfg_scale: float = DEFAULT_DIT_CFG_SCALE, key: mx.array | None = None,
                 initial_latent: mx.array | None = None,
                 initial_condition: mx.array | None = None) -> mx.array:
    """align [1, mel_len, 2048] -> VAE latent [1, 128, mel_len] via CFG Euler solve.

    For windows after the first, `initial_latent` [1,128,left] pins the leading
    `left` mel-frames of the trajectory to the previous window's saved tail (and
    `initial_condition` [1,left,2048] overwrites that region's conditioning), giving
    seamless cross-window continuity. Matches the sglang DiT solve.
    """
    mel_len = align.shape[1]
    if key is None:
        key = mx.random.key(0)
    x = mx.random.normal((1, 128, mel_len), key=key).astype(align.dtype)

    left = 0
    latent_prompt = noise_prompt = None
    if initial_latent is not None:
        left = min(initial_latent.shape[-1], mel_len)
        if left > 0:
            latent_prompt = initial_latent[..., :left]
            noise_prompt = x[..., :left]
            if initial_condition is not None:
                align = mx.concatenate([initial_condition[:, :left, :], align[:, left:, :]], axis=1)

    cond_cfg = mx.concatenate([align, mx.zeros_like(align)], axis=0)  # [2, mel_len, 2048]
    dt = 1.0 / num_steps
    for step in range(num_steps):
        tval = step / num_steps
        if left and latent_prompt is not None:
            blended = (1.0 - (1.0 - 1e-6) * tval) * noise_prompt + tval * latent_prompt
            x = mx.concatenate([blended, x[..., left:]], axis=-1)
        t = mx.full((2,), tval, dtype=x.dtype)
        d = dit(mx.broadcast_to(x, (2, 128, mel_len)), t, cond_cfg)
        d = cfg_scale * d[0:1] + (1.0 - cfg_scale) * d[1:2]
        x = x + dt * d
        mx.eval(x)
    if left and latent_prompt is not None:
        x = mx.concatenate([latent_prompt, x[..., left:]], axis=-1)
    return x


def synthesize(cond_encoder, dit, vocoder, frame_hidden: mx.array, *,
               num_steps: int = DEFAULT_DIT_STEPS, cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
               key: mx.array | None = None) -> mx.array:
    """frame_hidden [F, 32768] -> 44.1 kHz stereo waveform [1, 2, mel_len*512]."""
    align = cond_encoder(frame_hidden[None])              # [1, mel_len, 2048]
    latent = solve_latent(dit, align, num_steps=num_steps, cfg_scale=cfg_scale, key=key)
    return vocoder(latent)


def resample_44k_to_32k(wave_np: np.ndarray) -> np.ndarray:
    """[.., samples] 44.1 kHz -> 32 kHz. Prefers torchaudio (matches reference)."""
    try:
        import torch
        import torchaudio.functional as AF

        out = AF.resample(torch.from_numpy(wave_np).float(), DAV_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        return out.numpy()
    except Exception:
        from scipy.signal import resample_poly  # 44100/32000 = 441/320

        return resample_poly(wave_np, up=320, down=441, axis=-1).astype(np.float32)


def _pcm16(wave_stereo: np.ndarray) -> bytes:
    return (np.clip(wave_stereo, -1.0, 1.0) * 32767.0).astype(np.int16).T.tobytes()


def write_wav(path: str | Path, wave_stereo: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> None:
    """wave_stereo [2, samples] float [-1,1] -> 16-bit PCM stereo WAV."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(_pcm16(wave_stereo))


def write_mp3(path: str | Path, wave_stereo: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE,
              bitrate: str = "256k") -> None:
    """wave_stereo [2, samples] float [-1,1] -> MP3 (pipes PCM to ffmpeg libmp3lame)."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot encode MP3")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(sample_rate), "-ac", "2", "-i", "pipe:0",
           "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)]
    subprocess.run(cmd, input=_pcm16(wave_stereo), check=True)


def write_audio(path: str | Path, wave_stereo: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> Path:
    """Write MP3 (default output format). If given a .wav path, writes the .mp3
    sibling and removes the .wav. Returns the actual output path."""
    path = Path(path)
    mp3_path = path.with_suffix(".mp3")
    write_mp3(mp3_path, wave_stereo, sample_rate)
    if path.suffix.lower() == ".wav" and path.exists():
        path.unlink()  # delete the wav once the mp3 is done
    return mp3_path


def synthesize_windows(cond_encoder, dit, vocoder, frame_hidden: mx.array, *, seed: int = 0,
                       num_steps: int = DEFAULT_DIT_STEPS, cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
                       on_window=None) -> np.ndarray:
    """Multi-window acoustic synthesis with latent-space continuity + sample-space
    crop. frame_hidden [F, 32768] -> 44.1 kHz stereo waveform [2, samples]."""
    frames = int(frame_hidden.shape[0])
    windows = chunk_windows(frames)
    ov = overlap_mel_length()
    last_latent = last_condition = None
    chunks: list[np.ndarray] = []
    for win in windows:
        hidden_win = frame_hidden[win.start:win.end]                 # [Fw, 32768]
        align = cond_encoder(hidden_win[None])                       # [1, mel, 2048]
        key = mx.random.key(derive_dit_seed(seed, win.index))
        latent = solve_latent(dit, align, num_steps=num_steps, cfg_scale=cfg_scale, key=key,
                              initial_latent=last_latent, initial_condition=last_condition)
        wave = vocoder(latent)                                        # [1, 2, mel*512]
        mel = latent.shape[-1]
        s = max(0, mel - 2 * ov)
        e = max(s, mel - ov)
        last_latent = latent[..., s:e]
        last_condition = align[:, s:e, :]
        left, right = crop_sample_bounds(win)
        w = np.array(wave[0].astype(mx.float32))                     # [2, mel*512]
        end = w.shape[-1] - right if right else w.shape[-1]
        chunks.append(w[:, left:end])
        if on_window is not None:
            on_window(win, latent, w)
    return np.concatenate(chunks, axis=-1)


def generate_music(backbone, depth, cond_encoder, dit, vocoder, prompt_ids, *,
                   seed: int = 0, max_frames: int = MAX_AUDIO_FRAMES, min_frames: int = 0,
                   num_steps: int = DEFAULT_DIT_STEPS,
                   cfg_scale: float = DEFAULT_DIT_CFG_SCALE, out_path: str | Path | None = None,
                   multiwindow: bool = True):
    """Full pipeline (multi-window by default). Returns (wave_32k [2, samples], num_frames)."""
    codes, frame_hidden = generate_frames(
        backbone,
        depth,
        prompt_ids,
        seed=seed,
        max_frames=max_frames,
        min_frames=min_frames,
    )
    if frame_hidden.shape[0] == 0:
        raise RuntimeError("AR stage produced zero frames")
    if multiwindow:
        wave44k_np = synthesize_windows(cond_encoder, dit, vocoder, frame_hidden, seed=seed,
                                        num_steps=num_steps, cfg_scale=cfg_scale)
    else:
        wave44k = synthesize(cond_encoder, dit, vocoder, frame_hidden, num_steps=num_steps,
                             cfg_scale=cfg_scale, key=mx.random.key(seed))
        wave44k_np = np.array(wave44k[0].astype(mx.float32))
    wave32k = resample_44k_to_32k(wave44k_np)
    if out_path is not None:
        write_wav(out_path, wave32k)
    return wave32k, frame_hidden.shape[0]
