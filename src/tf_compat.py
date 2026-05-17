"""Keras 2 compatibility shim.

Streamlit Cloud (Python 3.11 + TensorFlow 2.15) ships Keras 2 natively as
``tensorflow.keras`` and does **not** publish a ``tf-keras`` wheel for that
TF version. Local development on newer Python (3.12+) typically gets
TensorFlow 2.16+, where ``tensorflow.keras`` is Keras 3 and the Keras 2
implementation lives in the ``tf-keras`` package.

This module hides that difference: import ``Sequential``, ``layers``,
``regularizers`` and ``applications`` from here and the rest of the codebase
stays portable across both environments.
"""
from __future__ import annotations

import os

# keras-cv reads `import keras` internally. On TF 2.16+ that resolves to
# Keras 3 unless we flip this switch before TF is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    import tf_keras as _keras            # backport on TF 2.16+
    SOURCE = "tf_keras"
except ImportError:                       # pragma: no cover
    import tensorflow.keras as _keras    # native Keras 2 on TF 2.15
    SOURCE = "tensorflow.keras"

Sequential   = _keras.Sequential
layers       = _keras.layers
regularizers = _keras.regularizers
applications = _keras.applications
