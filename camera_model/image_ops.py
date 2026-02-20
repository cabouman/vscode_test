"""Image operations for applying blur kernels."""

from __future__ import annotations

import numpy as np


def blur_image_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D blur kernel to an RGB or grayscale image using FFT convolution."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (H, W, C)")
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")

    kernel = kernel.astype(np.float32)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        raise ValueError("kernel sum must be > 0")
    kernel = kernel / kernel_sum

    if image.ndim == 2:
        out = _fft_conv2d(image.astype(np.float32), kernel)
        return np.clip(out, 0, 255).astype(np.uint8)

    channels = []
    for c in range(image.shape[2]):
        channel = _fft_conv2d(image[:, :, c].astype(np.float32), kernel)
        channels.append(channel)
    merged = np.stack(channels, axis=2)
    return np.clip(merged, 0, 255).astype(np.uint8)


def _fft_conv2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    kh, kw = kernel.shape
    fh = h + kh - 1
    fw = w + kw - 1

    channel_f = np.fft.rfft2(channel, s=(fh, fw))
    kernel_f = np.fft.rfft2(kernel, s=(fh, fw))
    out_full = np.fft.irfft2(channel_f * kernel_f, s=(fh, fw))

    y0 = kh // 2
    x0 = kw // 2
    return out_full[y0 : y0 + h, x0 : x0 + w]
