"""Helper utilities: safe temp files, image helpers, weight download."""
from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .config import (
    GDRIVE_FOLDER_ID,
    WEIGHT_FILES,
    WEIGHT_GDRIVE_IDS,
    WEIGHTS_DIR,
    has_weight,
    missing_weights,
    weight_path,
)


# ---------------------------------------------------------------------------
# Safe temp files
# ---------------------------------------------------------------------------
@contextmanager
def temp_image_path(uploaded_file, suffix_hint: str | None = None) -> Iterator[str]:
    """Write a Streamlit UploadedFile to a sandboxed temp path and clean up.

    Uses ``tempfile`` so user-controlled filenames cannot escape the directory
    or collide between concurrent users.
    """
    suffix = suffix_hint or Path(getattr(uploaded_file, "name", "img.jpg")).suffix
    if not suffix or suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    tmp_dir = Path(tempfile.gettempdir()) / "skin_detection"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        yield str(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Image loading (robust)
# ---------------------------------------------------------------------------
def load_image_rgb(path: str) -> np.ndarray | None:
    """Load an image as RGB numpy array. Returns None if invalid."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.asarray(im)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Weight auto-download (real, not a placeholder)
# ---------------------------------------------------------------------------
def ensure_model_weights(progress_cb=None) -> tuple[list[str], list[str]]:
    """Download any missing model weights using gdown.

    Returns ``(downloaded, still_missing)`` lists of model keys.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = missing_weights()
    if not missing:
        return [], []

    try:
        import gdown  # lazy import; only needed on first launch
    except Exception:
        return [], missing

    downloaded: list[str] = []
    still_missing: list[str] = []

    # 1) Per-file download (preferred, more reliable on cloud)
    for key in missing:
        file_id = WEIGHT_GDRIVE_IDS.get(key)
        target = weight_path(key)
        if not file_id:
            still_missing.append(key)
            continue
        try:
            if progress_cb:
                progress_cb(f"Downloading {WEIGHT_FILES[key]}…")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(target), quiet=True, fuzzy=True)
            if target.exists() and target.stat().st_size > 0:
                downloaded.append(key)
            else:
                still_missing.append(key)
        except Exception:
            still_missing.append(key)

    # 2) Folder-level fallback for anything still missing
    if still_missing and GDRIVE_FOLDER_ID:
        try:
            if progress_cb:
                progress_cb("Downloading remaining weights from Drive folder…")
            folder_url = (
                f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
            )
            gdown.download_folder(
                folder_url,
                output=str(WEIGHTS_DIR),
                quiet=True,
                use_cookies=False,
            )
            recovered = []
            for key in still_missing:
                if has_weight(key):
                    recovered.append(key)
                    downloaded.append(key)
            still_missing = [k for k in still_missing if k not in recovered]
        except Exception:
            pass

    return downloaded, still_missing


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def severity_from_count(n: int) -> tuple[str, str]:
    """Map a detection count to (label, css class)."""
    if n == 0:
        return "Clear", "b-g"
    if n <= 2:
        return "Mild", "b-y"
    if n <= 5:
        return "Moderate", "b-o"
    return "Severe", "b-r"


def severity_from_conf(detected: bool, conf: float) -> tuple[str, str]:
    if not detected:
        return "Clear", "b-g"
    if conf >= 0.75:
        return "Severe", "b-r"
    if conf >= 0.50:
        return "Moderate", "b-o"
    return "Mild", "b-y"


def pretty_pct(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
    except Exception:
        return "—"
