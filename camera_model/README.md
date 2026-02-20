# Camera Model Utilities

This module provides a first-pass camera shake blur simulator:

- Random camera-shake PSF (blur kernel) generation
- Kernel visualization utility
- Streamlit web app to upload an image, generate kernels, and preview blur

## Setup

From the repository root:

```bash
bash camera_model/scripts/install.sh
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r camera_model/requirements.txt
```

## Visualize random kernels

```bash
python3 camera_model/visualize.py --num 6
```

Optional save instead of interactive display:

```bash
python3 camera_model/visualize.py --num 9 --save camera_model/kernel_gallery.png
```

## Run web app

```bash
bash camera_model/scripts/run_app.sh
```

Then open the local URL shown by Streamlit.
