"""CLI utility to visualize random camera-shake kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .shake_kernel import generate_shake_kernel, random_shake_params
except ImportError:  # pragma: no cover
    from shake_kernel import generate_shake_kernel, random_shake_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random camera-shake PSFs.")
    parser.add_argument("--num", type=int, default=6, help="Number of kernels to display.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of showing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cols = min(3, args.num)
    rows = int(np.ceil(args.num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(args.num):
        params = random_shake_params(rng)
        kernel, trajectory = generate_shake_kernel(
            size=params.size,
            steps=params.steps,
            damping=params.damping,
            jitter=params.jitter,
            kick_prob=params.kick_prob,
            kick_strength=params.kick_strength,
            seed=int(rng.integers(0, 2**31 - 1)),
            return_trajectory=True,
        )

        ax = axes[i]
        ax.imshow(kernel, cmap="magma")
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="cyan", alpha=0.3, linewidth=0.8)
        ax.set_title(f"size={params.size}, steps={params.steps}")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(args.num, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Random Camera-Shake PSFs")
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
