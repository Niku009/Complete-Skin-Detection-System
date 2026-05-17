"""Central configuration: paths, weight metadata, palette."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "assets"


# --- Model weight metadata ---------------------------------------------
# Each model has a filename and an optional Google Drive file ID for
# auto-download on first launch (set GDRIVE_FOLDER_ID to fall back to
# folder-level download if individual IDs aren't provided).
WEIGHT_FILES: dict[str, str] = {
    "dark_circle": "DarkCircideWeights.pt",
    "acne": "yolo_acne_detection.weights.h5",
    "redness": "skin_redness_model_weights.pth",
    "skin_type": "skin_type_weights.weights.h5",
}

# Override per-file IDs via environment variables if you upload weights
# to your own Drive (recommended for reliable cloud deploys).
WEIGHT_GDRIVE_IDS: dict[str, str | None] = {
    "dark_circle": os.environ.get("WEIGHT_ID_DARK_CIRCLE"),
    "acne": os.environ.get("WEIGHT_ID_ACNE"),
    "redness": os.environ.get("WEIGHT_ID_REDNESS"),
    "skin_type": os.environ.get("WEIGHT_ID_SKIN_TYPE"),
}

# Fallback: try to pull the entire Drive folder when individual IDs
# are missing. Override with GDRIVE_FOLDER_ID env var.
GDRIVE_FOLDER_ID: str = os.environ.get(
    "GDRIVE_FOLDER_ID",
    "15TlaZmuvhIw2c-j-AxRIp9FDi5manbUt",
)


def weight_path(model_key: str) -> Path:
    return WEIGHTS_DIR / WEIGHT_FILES[model_key]


def has_weight(model_key: str) -> bool:
    return weight_path(model_key).exists()


def missing_weights() -> list[str]:
    return [k for k in WEIGHT_FILES if not has_weight(k)]


# --- Palette (single source of truth shared with CSS) -------------------
PALETTE = {
    "cream":   "#FAF5EC",
    "cream2":  "#FFF8EC",
    "paper":   "#FFFBF4",
    "orange":  "#E8632C",
    "orange2": "#F08350",
    "orangeD": "#C24A18",
    "ink":     "#2D1810",
    "muted":   "#7A6E5D",
    "border":  "#E8DCC4",
    "success": "#5A8F4A",
    "warn":    "#C97A1B",
    "danger":  "#B83A2A",
}
