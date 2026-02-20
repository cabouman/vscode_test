"""Utilities for generating random camera-shake blur kernels (PSFs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ShakeParams:
    size: int
    steps: int
    damping: float
    jitter: float
    kick_prob: float
    kick_strength: float


def random_shake_params(rng: Optional[np.random.Generator] = None) -> ShakeParams:
    """Draw a random but bounded set of shake parameters."""
    rng = rng or np.random.default_rng()
    size = int(rng.choice(np.arange(21, 71, 2)))
    return ShakeParams(
        size=size,
        steps=int(rng.integers(120, 420)),
        damping=float(rng.uniform(0.84, 0.97)),
        jitter=float(rng.uniform(0.07, 0.2)),
        kick_prob=float(rng.uniform(0.025, 0.11)),
        kick_strength=float(rng.uniform(0.5, 1.8)),
    )


def _deposit_bilinear(canvas: np.ndarray, x: float, y: float) -> None:
    """Deposit point energy into the four nearest pixels."""
    h, w = canvas.shape
    if not (0 <= x < w - 1 and 0 <= y < h - 1):
        return

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0

    canvas[y0, x0] += (1.0 - dx) * (1.0 - dy)
    canvas[y0 + 1, x0] += (1.0 - dx) * dy
    canvas[y0, x0 + 1] += dx * (1.0 - dy)
    canvas[y0 + 1, x0 + 1] += dx * dy


def _sample_trajectory(
    size: int,
    steps: int,
    damping: float,
    jitter: float,
    kick_prob: float,
    kick_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bounded 2D camera shake path with inertia and random impulses."""
    center = (size - 1) / 2.0
    pos = np.array([center, center], dtype=np.float64)
    vel = np.zeros(2, dtype=np.float64)

    margin = 1.0
    points = np.empty((steps, 2), dtype=np.float64)

    for i in range(steps):
        vel *= damping
        vel += rng.normal(0.0, jitter, size=2)

        if rng.random() < kick_prob:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            vel += kick_strength * np.array([np.cos(angle), np.sin(angle)])

        pos += vel

        # Reflective bounds keep the trajectory mostly inside the kernel grid.
        for axis in (0, 1):
            low = margin
            high = size - 1 - margin
            if pos[axis] < low:
                pos[axis] = low + (low - pos[axis])
                vel[axis] *= -0.6
            elif pos[axis] > high:
                pos[axis] = high - (pos[axis] - high)
                vel[axis] *= -0.6

        points[i] = pos

    return points


def generate_shake_kernel(
    size: int = 45,
    steps: int = 240,
    damping: float = 0.92,
    jitter: float = 0.13,
    kick_prob: float = 0.06,
    kick_strength: float = 1.1,
    seed: Optional[int] = None,
    return_trajectory: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a random camera-shake PSF normalized to unit sum.

    The model uses a random trajectory with inertia and occasional impulses.
    The kernel is the accumulated dwell time along that path.
    """
    if size % 2 == 0 or size < 5:
        raise ValueError("size must be an odd integer >= 5")
    if steps < 16:
        raise ValueError("steps must be >= 16")
    if not (0.0 < damping < 1.0):
        raise ValueError("damping must be in (0, 1)")
    if jitter <= 0:
        raise ValueError("jitter must be > 0")
    if not (0.0 <= kick_prob <= 1.0):
        raise ValueError("kick_prob must be in [0, 1]")

    rng = np.random.default_rng(seed)
    trajectory = _sample_trajectory(
        size=size,
        steps=steps,
        damping=damping,
        jitter=jitter,
        kick_prob=kick_prob,
        kick_strength=kick_strength,
        rng=rng,
    )

    kernel = np.zeros((size, size), dtype=np.float64)
    for x, y in trajectory:
        _deposit_bilinear(kernel, x, y)

    total = kernel.sum()
    if total <= 0:
        # Fallback delta kernel (should be rare).
        kernel[size // 2, size // 2] = 1.0
    else:
        kernel /= total

    if return_trajectory:
        return kernel.astype(np.float32), trajectory
    return kernel.astype(np.float32)
