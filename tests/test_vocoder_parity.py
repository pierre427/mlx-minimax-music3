# SPDX-License-Identifier: Apache-2.0
"""M7 parity: MLX DAV vocoder vs the sglang torch DAV decoder (fp32).

Oracle = sglang's `MiniMaxMusic3DAV` (pure torch, loaded standalone from the
reference tree) with its own weight-norm convs, loaded from dav.pth via
`select_decoder_state`. We diff the stereo waveform for a fixed latent, which
also validates the weight-norm folding (torch computes g*v/||v|| on the fly;
we fold it once at load).

Run: RUN_HEAVY=1 .venv/bin/python -m pytest minimax-music3-mlx/tests/test_vocoder_parity.py -q -s
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT_ROOT))

import minimax_music3_mlx  # noqa: E402,F401  (pin TF32)

PTH = PORT_ROOT / "weights" / "dav.pth"
REF_DAV_PY = Path("/Users/Shared/src/sglang-omni/sglang_omni/models/minimax_music3/dav.py")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY") != "1", reason="loads dav.pth; set RUN_HEAVY=1"
)


def _load_ref():
    spec = importlib.util.spec_from_file_location("_ref_dav", REF_DAV_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("T", [16, 43])
def test_vocoder_waveform_parity(T):
    import mlx.core as mx
    import torch

    if not PTH.exists() or not REF_DAV_PY.exists():
        pytest.skip("dav.pth or reference missing")

    ref = _load_ref()
    dav = ref.MiniMaxMusic3DAV().eval()
    state = torch.load(str(PTH), map_location="cpu", weights_only=True)
    dav.load_state_dict(ref.select_decoder_state(state), strict=True)

    from minimax_music3_mlx.vocoder import load_vocoder

    voc = load_vocoder(PTH, dtype=mx.float32)

    rng = np.random.default_rng(T)
    latent = rng.standard_normal((1, 128, T)).astype(np.float32)

    with torch.no_grad():
        ref_wave = dav(torch.from_numpy(latent)).numpy()
    got = np.array(voc(mx.array(latent)).astype(mx.float32))

    assert got.shape == ref_wave.shape == (1, 2, T * 512)
    d = float(np.abs(got - ref_wave).max())
    print(f"\nT={T} samples={T*512} waveform max_abs={d:.3e} "
          f"(range ref [{ref_wave.min():.3f},{ref_wave.max():.3f}])")
    assert d < 1e-3, f"waveform max_abs {d:.3e}"
