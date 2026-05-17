"""Keras-2 compatibility shim.

We install ``tf-keras`` (the standalone Keras-2 backport) in BOTH
environments so ``import tf_keras`` always succeeds. This bypasses Keras-3
if it happens to exist in the environment.

  Python 3.11 + TF 2.15 (Streamlit Cloud) -> tf-keras==2.15.0
  Python 3.12+ local (TF 2.16+)           -> tf-keras==2.16.0

The env vars MUST be set before any tensorflow/keras import. We set them
here at module level; app.py also sets them at the very top as an extra
safety net for cases where streamlit imports this module via a cached path.
"""
from __future__ import annotations

import os

# Must precede every TF / keras-cv import.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tf_keras as _keras  # noqa: E402  (tf_keras in both requirements files)

SOURCE = "tf_keras"

Sequential   = _keras.Sequential
layers       = _keras.layers
regularizers = _keras.regularizers
applications = _keras.applications
