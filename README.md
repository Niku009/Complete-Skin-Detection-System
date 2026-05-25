# Skinwise

> A small web app that reads a face the way an attentive friend might.
> You upload one selfie, four AI models look at it, and you get back a
> calm, paper textured report on your skin.

[Live demo](#deploy-to-streamlit-cloud) · [Quick start](#quick-start) · [How it works](#how-it-works)

![hero](assets/hero-preview.png)

---

## The story behind it

This project started as a college experiment that lived inside a folder
called `simple_detection_system`. Four trained models had been stitched
together: one to spot dark circles, one to find acne, one for facial
redness and eye bags, and one to classify skin type. The math worked, but
the rest had drifted. The Streamlit page mixed dark dashboard colors with
loud labels, the launcher scripts were brittle, and the live deploy on
Streamlit Cloud refused to start because the requirements file pinned a
combination of TensorFlow, Keras, Torch and Keras-CV versions that no
longer co-installed on modern Python.

The most painful surprise came later. Even after the dependencies were
sorted, two of the four models would not load. Keras kept reporting that
every layer "received 0 variables". The weights file was clearly present.
The architecture matched. Something quieter was going on.

This rewrite is the version of the project that survived all of that.
The UI is now warm cream and bold orange, modeled after a folded paper
sunrise. The code is split into small files that each do one thing. The
weights load no matter which version of TensorFlow you happen to have.
And the whole thing fits on Streamlit Cloud's free tier.

## What it does

You drag a photo onto the page. Within a few seconds you see:

- **Dark circles**, with bounding boxes around each detected region
- **Acne spots**, again with bounding boxes
- **Facial redness** as a yes or no signal, plus a confidence percentage
- **Eye bags** alongside redness from the same classifier
- **Skin type**, one of Dry, Normal or Oily, with a confidence score

A downloadable JPEG carries the annotated image away for your own
records. Everything runs in the browser session and nothing is stored on
the server.

## How it works

Under the hood there are four very different models behind the same
upload button.

| Signal              | Model           | Framework        |
| ------------------- | --------------- | ---------------- |
| Dark circles        | YOLOv8          | Ultralytics      |
| Acne spots          | YOLOv8 (XS)     | Keras-CV         |
| Redness, eye bags   | EfficientNet-B0 | PyTorch + timm   |
| Skin type           | ResNet50        | TensorFlow/Keras |

When the page boots, `app.py` shows the hero and the file uploader. The
four model loaders sit behind `@st.cache_resource`, so they only fire on
the first actual upload. Each loader imports its framework lazily, which
means a missing or broken framework cannot crash the rest of the page.
If anything goes wrong, it is captured in a diagnostic panel that the
user can expand instead of staring at a red trace.

The pipeline itself lives in `src/detection.py`. Each detector takes the
RGB image and a shared annotated copy, draws boxes onto the copy, and
writes its numbers into a small `AnalysisResult` dataclass. The UI then
renders that dataclass into four metric cards and two detail cards.

## The weight loader, a short tangent

If you only read one technical paragraph, read this one.

The four models were trained somewhere a long time ago. Two of them, the
ResNet50 for skin type and the Keras-CV YOLOv8 for acne, were saved as
`.weights.h5` files. Modern Keras opens those files using strict name
based loading: it asks the model for the name of each layer, then looks
that name up in the file.

That sounds reasonable. The catch is that the layers inside these two
files were named generically (`conv2d`, `conv2d_1`, `batch_normalization`
and so on). A fresh `keras.applications.ResNet50()` produces layers
named after the architecture (`conv1_conv`, `conv2_block1_1_conv`). Same
shapes, same order, different labels. Keras then reports that every
layer "expected N variables, received 0 variables" and gives up.

The fix is in `src/weight_loader.py`. It walks the model in topological
order, walks the h5 file using the same generic naming scheme that Keras
itself uses when no explicit names are supplied, and calls
`layer.set_weights()` directly for each matched pair. Weight free ops
(reshapes, lambdas, anchor generators) are recognised and skipped on
purpose. The result is 110 ResNet50 layers and 160 YOLOv8 layers
loaded cleanly, with no Keras version drama at all.

## Project layout

```
.
├── app.py                  # Streamlit entry point (slim glue layer)
├── src/
│   ├── config.py           # Paths, palette, weight metadata
│   ├── tf_compat.py        # Keras 2 import shim (tf_keras OR tensorflow.keras)
│   ├── styles.py           # Cream and orange CSS
│   ├── ui.py               # Header, hero, results, debug panel
│   ├── models.py           # Cached, lazy model loaders
│   ├── detection.py        # 4-model inference pipeline
│   ├── weight_loader.py    # Topological .weights.h5 loader
│   └── utils.py            # Temp files, image IO, weight auto-download
├── weights/                # .pt, .pth, .h5 (auto-downloaded on first run)
├── assets/
├── .streamlit/config.toml  # Theme matched to the UI
├── runtime.txt             # python-3.11
├── requirements.txt        # Pinned for Streamlit Cloud
├── requirements-dev.txt    # Alt pins for local Python 3.12 / 3.13
└── Dockerfile              # For self-hosting
```

## Quick start

If you only want to see it run locally:

```bash
git clone https://github.com/Niku009/Complete-Skin-Detection-System
cd Complete-Skin-Detection-System

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Pick the right requirements file for your Python version.
# Streamlit Cloud uses Python 3.11, so requirements.txt targets that.
# Locally on 3.12 / 3.13, use the dev file (it has the tf-keras backport).
pip install -r requirements-dev.txt

streamlit run app.py
```

The app opens at http://localhost:8501.

On the first launch, missing model weights are downloaded automatically
into `weights/` via [`gdown`](https://pypi.org/project/gdown/). If you
already have the four files, drop them in `weights/` and the download is
skipped:

```
weights/
├── DarkCircideWeights.pt
├── yolo_acne_detection.weights.h5
├── skin_redness_model_weights.pth
└── skin_type_weights.weights.h5
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Visit https://streamlit.io/cloud and click **New app**.
3. Point it at this repo. Set the branch to `main`, the file to
   `app.py`, and the Python version to `3.11`.
4. Click **Deploy**.

That is enough to get a working deploy. On the first run, the auto
download step will pull the weights from Google Drive into the container
under `weights/`. Cold starts after that are about ten to twenty
seconds.

Optional but recommended: in **Settings, Secrets** add the Drive file
IDs for each weight. This switches from folder level download (slower
and occasionally rate limited) to direct per file download:

```toml
WEIGHT_ID_DARK_CIRCLE = "..."
WEIGHT_ID_ACNE        = "..."
WEIGHT_ID_REDNESS     = "..."
WEIGHT_ID_SKIN_TYPE   = "..."
```

The folder ID itself can be overridden with `GDRIVE_FOLDER_ID` if you
mirror the weights into your own Drive folder.

## Why the rewrite was worth it

The original repo was a deeply nested `simple_detection_system/` folder
with a 790 line `app.py`, two competing requirements files, batch
scripts, PowerShell scripts and four markdown files at the root. The
auto download function claimed to fetch weights but only printed a
warning. The Streamlit Cloud deploy never finished.

The new repo:

- Flat, predictable layout with a thin entry point
- Real weight auto download via `gdown`, with optional per file IDs
- Safe temp file handling using `tempfile`, no path traversal from
  user filenames
- Null safe image loading
- Lazy framework imports so one missing dependency cannot kill the page
- A small `tf_compat.py` shim that handles the Keras 2 versus Keras 3
  divide without leaking through the rest of the code
- A custom topological weight loader that bypasses Keras strict name
  matching for `.weights.h5` files
- A warm, paper themed UI tuned for skin care rather than dashboards
- Diagnostic panel that surfaces silent failures with the actual error
  text, not a generic "model not loaded"

## Disclaimer

Skinwise is for informational and educational use only. It is not a
medical device. If something on your skin worries you, see a
dermatologist.

## License

This project includes pre trained models. Check the individual model
licenses before any commercial use.
