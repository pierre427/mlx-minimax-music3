"""MLX (Apple Silicon) port of MiniMaxAI/MiniMax-Music3 — text-to-music.

See wiki ports/minimax-music3-mlx.md for the architecture map and build plan.
"""

import os

# On M5 the MLX default enables TF32 for fp32 matmuls, which drops ~1e-2 error on
# high-magnitude Qwen3 activations and would accumulate across the 30-step fp32
# Euler solve in the flow-matching DiT. Pin it off for numeric fidelity (only the
# fp32 paths are affected; a bf16 backbone is unchanged). Verified: this is what
# takes M2 backbone parity from max_abs 1.5e-2 -> 4.6e-5. See lab lesson
# tf32-default-fp32-gemm-m5. Set MLX_ENABLE_TF32=1 in the env to override.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
