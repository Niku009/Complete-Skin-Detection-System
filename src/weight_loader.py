"""Custom .weights.h5 loader for Keras models whose weights were saved with
**generic auto-named layers** (e.g. ``conv2d``, ``conv2d_1`` …) rather than
the canonical names produced by ``keras.applications.ResNet50``.

Keras 2 / Keras 3's built-in ``model.load_weights()`` for ``.weights.h5``
files is strictly name-based and silently mismatches in that case (every
layer reports "received 0 variables"). This module re-maps weights by
**type + topological-creation order**, which is exactly how Keras
auto-names layers anyway, so the assignment is unambiguous.

Used by the skin-type ResNet50 model.
"""
from __future__ import annotations

from typing import Iterable

import h5py
import numpy as np


# Map Keras layer class names -> generic snake_case names used by Keras
# when no explicit name= is supplied. These match the h5 layer keys.
TYPE_MAP = {
    # No-weight ops (skipped silently)
    "TFOpLambda":             None,
    "SlicingOpLambda":        None,
    "Lambda":                 None,
    "InputLayer":             None,
    "Concatenate":            None,
    "Reshape":                None,
    "Flatten":                None,
    "Permute":                None,
    "UpSampling2D":           None,
    "Cropping2D":             None,
    "NonMaxSuppression":      None,
    "YOLOV8LabelEncoder":     None,
    "AnchorGenerator":        None,
    "Conv2D":                 "conv2d",
    "BatchNormalization":     "batch_normalization",
    "Activation":             "activation",
    "ReLU":                   "re_lu",
    "ZeroPadding2D":          "zero_padding2d",
    "MaxPooling2D":           "max_pooling2d",
    "AveragePooling2D":       "average_pooling2d",
    "Add":                    "add",
    "GlobalAveragePooling2D": "global_average_pooling2d",
    "GlobalMaxPooling2D":     "global_max_pooling2d",
    "Dense":                  "dense",
    "Dropout":                "dropout",
    "Rescaling":              "rescaling",
    "Normalization":          "normalization",
    "Functional":             "functional",
    "Model":                  "functional",
    "Sequential":             "sequential",
}


_SENTINEL_MISSING = "__missing__"


def _generic_name_for(layer) -> str | None:
    cls = type(layer).__name__
    return TYPE_MAP.get(cls, _SENTINEL_MISSING)


def _read_layer_weights(h5_layer_group) -> list[np.ndarray]:
    """Return a layer's weight tensors in numeric-key order."""
    if "vars" not in h5_layer_group:
        return []
    vars_g = h5_layer_group["vars"]
    keys = sorted(vars_g.keys(), key=lambda k: int(k) if k.isdigit() else 1_000_000)
    return [np.asarray(vars_g[k]) for k in keys]


def _load_scope(layers: Iterable, h5_group) -> tuple[int, int, list[str]]:
    """Recursively load weights for ``layers`` under ``h5_group['layers']``.

    Returns ``(loaded_count, skipped_count, warnings)``.
    """
    loaded = 0
    skipped = 0
    warnings: list[str] = []

    if "layers" not in h5_group:
        return loaded, skipped, warnings
    h5_layers = h5_group["layers"]

    counters: dict[str, int] = {}

    for layer in layers:
        base = _generic_name_for(layer)
        if base is None:
            # Explicitly known weight-free op — skip silently.
            continue
        if base is _SENTINEL_MISSING:
            # Unknown layer type — warn and skip.
            skipped += 1
            warnings.append(f"no generic-name mapping for {type(layer).__name__}")
            continue

        idx = counters.get(base, 0)
        counters[base] = idx + 1
        h5_name = base if idx == 0 else f"{base}_{idx}"

        if h5_name not in h5_layers:
            # Try a small probe: maybe it's stored under another suffix order.
            candidates = [n for n in h5_layers.keys() if n.startswith(base)]
            warnings.append(
                f"layer {layer.name} -> missing h5 entry '{h5_name}' "
                f"(have {candidates[:3]}…)"
            )
            skipped += 1
            continue

        h5_layer = h5_layers[h5_name]

        # If this is a sub-model (Functional/Sequential), recurse.
        if hasattr(layer, "layers") and layer.layers:
            sub_loaded, sub_skipped, sub_warn = _load_scope(layer.layers, h5_layer)
            loaded += sub_loaded
            skipped += sub_skipped
            warnings.extend(sub_warn)
            continue

        weights = _read_layer_weights(h5_layer)
        if not weights:
            # Layer with no learnable variables — perfectly fine.
            continue

        # Validate shape compatibility before assigning.
        current = layer.weights
        if len(current) != len(weights):
            warnings.append(
                f"{layer.name}: variable count {len(current)} vs h5 {len(weights)}"
            )
            skipped += 1
            continue
        if any(tuple(cw.shape) != tuple(w.shape) for cw, w in zip(current, weights)):
            warnings.append(f"{layer.name}: shape mismatch")
            skipped += 1
            continue

        try:
            layer.set_weights(weights)
            loaded += 1
        except Exception as e:
            warnings.append(f"{layer.name}: set_weights raised {e}")
            skipped += 1

    return loaded, skipped, warnings


def load_weights_topological(model, h5_path: str) -> dict:
    """Load a ``.weights.h5`` file into ``model`` by walking both in topology
    order and matching by generic layer name (``type + counter``).

    Returns a dict with statistics:
      - ``loaded``:  number of leaf layers whose weights were set
      - ``skipped``: number of leaf layers we couldn't match
      - ``warnings``: human-readable diagnostics
    """
    with h5py.File(h5_path, "r") as f:
        if "layers" not in f:
            return {"loaded": 0, "skipped": 0, "warnings": ["h5 has no /layers"]}
        loaded, skipped, warnings = _load_scope(model.layers, f)

    return {"loaded": loaded, "skipped": skipped, "warnings": warnings}
