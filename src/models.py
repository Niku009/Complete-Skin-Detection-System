"""Model loaders. Every loader returns ``None`` instead of crashing.

Heavy ML frameworks are imported lazily so the Streamlit page can render
even when a particular framework is missing on the deployment image.

When a loader fails, the exception is captured in ``LOAD_ERRORS`` so the UI
can surface it instead of leaving the user guessing.
"""
from __future__ import annotations

import os
import traceback
import warnings

import streamlit as st

from .config import has_weight, weight_path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# Diagnostics: track why a loader returned None so it can be shown in the UI.
# ---------------------------------------------------------------------------
LOAD_ERRORS: dict[str, str] = {}


def _record(name: str, exc: BaseException) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    LOAD_ERRORS[name] = msg
    print(f"[model-load-fail] {name}: {msg}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_torch_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# YOLO (PyTorch) — dark circles
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_dark_circle_model():
    if not has_weight("dark_circle"):
        LOAD_ERRORS["dark_circle"] = "weight file missing"
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(weight_path("dark_circle")))
    except Exception as e:
        _record("dark_circle", e)
        return None


# ---------------------------------------------------------------------------
# Keras-CV YOLO — acne
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_acne_model():
    """Load the keras-cv YOLOv8 acne detector.

    Same generic-name h5 quirk as the skin-type model — we fall back to the
    topology + type-counter loader after building the architecture.
    """
    if not has_weight("acne"):
        LOAD_ERRORS["acne"] = "weight file missing"
        return None
    try:
        import keras_cv

        from .weight_loader import load_weights_topological

        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone", include_rescaling=True
        )
        model = keras_cv.models.YOLOV8Detector(
            num_classes=1,
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=5,
        )

        # Try the standard loader first — works if names happen to match.
        try:
            model.load_weights(str(weight_path("acne")))
            return model
        except Exception:
            pass

        stats = load_weights_topological(model, str(weight_path("acne")))
        print(f"[acne] loaded={stats['loaded']} skipped={stats['skipped']}")
        for w in stats["warnings"][:5]:
            print(f"  · {w}")
        if stats["loaded"] == 0:
            raise RuntimeError(
                "no weights assigned — h5 structure unexpected: "
                + "; ".join(stats["warnings"][:3])
            )
        return model
    except Exception as e:
        _record("acne", e)
        return None


# ---------------------------------------------------------------------------
# EfficientNet (PyTorch) — redness + eye bags
# ---------------------------------------------------------------------------
class _SkinConditionClassifier:  # placeholder so type-checkers can see it
    pass


def _build_redness_classifier(num_classes: int = 2):
    import timm
    import torch.nn as nn

    class SkinConditionClassifier(nn.Module):
        def __init__(self, num_classes: int = 2, pretrained: bool = False):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnet_b0", pretrained=pretrained
            )
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.backbone(x)

    return SkinConditionClassifier(num_classes=num_classes, pretrained=False)


@st.cache_resource(show_spinner=False)
def load_redness_model():
    if not has_weight("redness"):
        LOAD_ERRORS["redness"] = "weight file missing"
        return None
    try:
        import torch
        device = get_torch_device()
        model = _build_redness_classifier(num_classes=2).to(device)
        checkpoint = torch.load(
            str(weight_path("redness")),
            map_location=device,
            weights_only=False,
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception as e:
        _record("redness", e)
        return None


@st.cache_resource(show_spinner=False)
def get_redness_transform():
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    except Exception as e:
        _record("redness_transform", e)
        return None


# ---------------------------------------------------------------------------
# ResNet50 (TensorFlow) — skin type
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_skin_type_model():
    """Load the ResNet50 skin-type classifier.

    The .weights.h5 file was produced in a Keras session where layers got
    **generic auto-names** (``conv2d``, ``conv2d_1`` …) rather than the
    canonical ResNet50 names (``conv1_conv`` …). Both Keras 2 and Keras 3's
    built-in loaders are name-strict and silently mismatch, so we use a
    custom topology + type-counter loader (see ``weight_loader.py``).
    """
    if not has_weight("skin_type"):
        LOAD_ERRORS["skin_type"] = "weight file missing"
        return None
    try:
        from .tf_compat import Sequential, applications, layers, regularizers
        from .weight_loader import load_weights_topological

        ResNet50 = applications.ResNet50

        IMG_SIZE = 224
        resnet = ResNet50(
            weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        resnet.trainable = False
        model = Sequential(
            [
                layers.Rescaling(1.0 / 127.5, offset=-1),
                resnet,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(0.001),
                ),
                layers.Dropout(0.5),
                layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(0.001),
                ),
                layers.Dropout(0.3),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))

        stats = load_weights_topological(model, str(weight_path("skin_type")))
        print(f"[skin_type] loaded={stats['loaded']} skipped={stats['skipped']}")
        for w in stats["warnings"][:5]:
            print(f"  · {w}")
        if stats["loaded"] == 0:
            raise RuntimeError(
                "no weights assigned — h5 structure unexpected: "
                + "; ".join(stats["warnings"][:3])
            )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    except Exception as e:
        _record("skin_type", e)
        return None


# ---------------------------------------------------------------------------
# Unified availability summary (for the UI)
# ---------------------------------------------------------------------------
def model_availability() -> dict[str, bool]:
    """Cheap pre-load check based on weight files (used for the top strip)."""
    return {
        "dark_circle": has_weight("dark_circle"),
        "acne": has_weight("acne"),
        "redness": has_weight("redness"),
        "skin_type": has_weight("skin_type"),
    }


def runtime_status() -> dict[str, dict]:
    """Force every loader to run once and return ``{key: {loaded, error}}``.

    Use this to drive a debug panel — exposes silent failures so the user
    can see exactly why a detector returned nothing.
    """
    loaders = {
        "dark_circle": load_dark_circle_model,
        "acne":        load_acne_model,
        "redness":     load_redness_model,
        "skin_type":   load_skin_type_model,
    }
    status: dict[str, dict] = {}
    for key, fn in loaders.items():
        model = fn()
        status[key] = {
            "loaded": model is not None,
            "error":  LOAD_ERRORS.get(key, ""),
        }
    return status
