"""Streamlit app to preview random camera-shake blur kernels."""

from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

try:
    from .image_ops import blur_image_fft
    from .shake_kernel import ShakeParams, generate_shake_kernel, random_shake_params
except ImportError:  # pragma: no cover
    from image_ops import blur_image_fft
    from shake_kernel import ShakeParams, generate_shake_kernel, random_shake_params


def _kernel_figure(kernel: np.ndarray, title: str = "Blur Kernel (PSF)"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(kernel, cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _load_image(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def _download_button(image_array: np.ndarray) -> None:
    out_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    st.download_button(
        label="Download blurred image",
        data=buf.getvalue(),
        file_name="blurred_camera_shake.png",
        mime="image/png",
    )


def _manual_params() -> ShakeParams:
    size = st.sidebar.slider("Kernel size (odd)", min_value=9, max_value=91, value=45, step=2)
    steps = st.sidebar.slider("Trajectory steps", min_value=60, max_value=600, value=240, step=20)
    damping = st.sidebar.slider("Damping", min_value=0.80, max_value=0.99, value=0.92, step=0.01)
    jitter = st.sidebar.slider("Jitter", min_value=0.03, max_value=0.30, value=0.13, step=0.01)
    kick_prob = st.sidebar.slider("Kick probability", min_value=0.0, max_value=0.2, value=0.06, step=0.01)
    kick_strength = st.sidebar.slider("Kick strength", min_value=0.2, max_value=2.5, value=1.1, step=0.1)
    return ShakeParams(
        size=size,
        steps=steps,
        damping=damping,
        jitter=jitter,
        kick_prob=kick_prob,
        kick_strength=kick_strength,
    )


def main() -> None:
    st.set_page_config(page_title="Camera Shake Kernel Lab", layout="wide")
    st.title("Camera Shake Kernel Lab")
    st.write("Upload an image, generate a random shake PSF, and preview the blurred result.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    image = _load_image(uploaded_file)

    st.sidebar.header("Kernel Controls")
    random_mode = st.sidebar.checkbox("Random parameter mode", value=True)
    seed = st.sidebar.number_input("Seed (optional)", min_value=0, max_value=2**31 - 1, value=0)
    use_seed = st.sidebar.checkbox("Use seed", value=False)

    if random_mode:
        params = random_shake_params(np.random.default_rng(seed if use_seed else None))
    else:
        params = _manual_params()

    if "kernel" not in st.session_state:
        st.session_state.draw_count = 0
        st.session_state.kernel = generate_shake_kernel(
            size=params.size,
            steps=params.steps,
            damping=params.damping,
            jitter=params.jitter,
            kick_prob=params.kick_prob,
            kick_strength=params.kick_strength,
            seed=seed if use_seed else None,
        )
        st.session_state.last_params = params
        st.session_state.blurred = None

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Generate random kernel"):
            st.session_state.draw_count += 1
            draw_seed = (seed + st.session_state.draw_count) if use_seed else None
            active_params = random_shake_params(np.random.default_rng(draw_seed)) if random_mode else params
            st.session_state.kernel = generate_shake_kernel(
                size=active_params.size,
                steps=active_params.steps,
                damping=active_params.damping,
                jitter=active_params.jitter,
                kick_prob=active_params.kick_prob,
                kick_strength=active_params.kick_strength,
                seed=draw_seed,
            )
            st.session_state.last_params = active_params
            st.session_state.blurred = None

    with col_btn2:
        apply_clicked = st.button("Apply kernel to image")

    kernel = st.session_state.kernel
    last_params = st.session_state.last_params

    st.caption(
        "Current kernel params: "
        f"size={last_params.size}, steps={last_params.steps}, damping={last_params.damping:.2f}, "
        f"jitter={last_params.jitter:.2f}, kick_prob={last_params.kick_prob:.2f}, "
        f"kick_strength={last_params.kick_strength:.2f}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Kernel")
        st.pyplot(_kernel_figure(kernel), clear_figure=True)

    with right:
        st.subheader("Image Preview")
        if image is None:
            st.info("Upload an image to preview blur.")
        else:
            st.image(image, caption="Original", use_container_width=True)
            if apply_clicked:
                with st.spinner("Applying blur..."):
                    st.session_state.blurred = blur_image_fft(image, kernel)
            if st.session_state.blurred is not None:
                st.image(
                    st.session_state.blurred,
                    caption="Blurred with generated kernel",
                    use_container_width=True,
                )
                _download_button(st.session_state.blurred)


if __name__ == "__main__":
    main()
