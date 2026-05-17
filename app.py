"""Skinwise: AI Skin Analysis.

Entry point for Streamlit. Keeps glue logic only; heavy lifting lives in
``src/``.
"""
from __future__ import annotations

# IMPORTANT: must run before any TF / keras-cv import so the Keras 2
# backport is used on TF 2.16+ (Streamlit Cloud's TF 2.15 is unaffected).
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import streamlit as st

from src import models, ui
from src.detection import run_pipeline
from src.styles import inject_global_styles
from src.utils import (
    ensure_model_weights,
    load_image_rgb,
    temp_image_path,
)


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Skinwise — AI Skin Analysis",
    page_icon="◯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_styles()


# ---------------------------------------------------------------------------
# Auto-download weights on first launch (gdown). Quiet success path.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _bootstrap_weights() -> tuple[list[str], list[str]]:
    return ensure_model_weights()


with st.spinner("Preparing models on first launch…"):
    _downloaded, _still_missing = _bootstrap_weights()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
ui.render_header()
ui.render_hero()

if _still_missing:
    st.markdown(
        f"""
<div class="info-box">
  <strong>Heads up:</strong> {len(_still_missing)} model weight(s) couldn't
  be downloaded automatically. The app still runs — affected detectors will
  be skipped. To enable them, place these files in <code>weights/</code> or
  set Google Drive file IDs as environment variables:<br/>
  <code>{", ".join(_still_missing)}</code>
</div>
        """,
        unsafe_allow_html=True,
    )

availability = models.model_availability()
ui.render_model_status(availability)

# Force loaders to run and expose any silent failures in a debug panel.
runtime = models.runtime_status()

# ── Upload + advanced settings ────────────────────────────────────────────
ui.render_upload_intro()

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    uploaded_file = st.file_uploader(
        "Image upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

with st.expander("Advanced settings"):
    confidence = st.slider(
        "Detection confidence threshold (YOLO)",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher means stricter detection — fewer but more confident boxes.",
    )

# ---------------------------------------------------------------------------
# Main analysis flow
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    with temp_image_path(uploaded_file) as tmp_path:
        image_rgb = load_image_rgb(tmp_path)
        if image_rgb is None:
            st.error(
                "Could not read that image. Please try a different JPG or PNG."
            )
        else:
            ui.section_title(
                "Step 01",
                "Image Preview",
                "Original photograph and the AI-annotated overlay, side by side.",
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    '<div class="img-label">Original</div>',
                    unsafe_allow_html=True,
                )
                st.image(image_rgb, use_column_width=True)

            with st.spinner("Analyzing your skin with four AI models…"):
                result, annotated = run_pipeline(
                    image_rgb=image_rgb,
                    image_path=tmp_path,
                    confidence=confidence,
                )

            with col2:
                st.markdown(
                    '<div class="img-label">AI Detection</div>',
                    unsafe_allow_html=True,
                )
                st.image(annotated, use_column_width=True)

            # Surface any per-model errors without blocking the report.
            for err in result.errors:
                st.warning(err)

            # Diagnostic panel — shows which models actually loaded and the
            # raw inference output. Helpful when no boxes appear.
            ui.render_debug_panel(runtime, result=result)

            # Results summary
            st.markdown("<hr/>", unsafe_allow_html=True)
            ui.section_title(
                "Step 02",
                "Detection Summary",
                "A quick read of the four analyses.",
            )
            ui.render_results_summary(result)

            # Detailed analysis
            st.markdown("<hr/>", unsafe_allow_html=True)
            ui.section_title(
                "Step 03",
                "Detailed Breakdown",
                "Per-model results with confidence bars.",
            )
            ui.render_results_detail(result, confidence)

            # Download
            st.markdown("<hr/>", unsafe_allow_html=True)
            ui.section_title(
                "Step 04",
                "Save Your Report",
                "Download the annotated image for your records.",
            )
            result_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode(".jpg", result_bgr)
            if ok:
                st.download_button(
                    label="Download annotated analysis",
                    data=buffer.tobytes(),
                    file_name=f"skinwise_analysis_{uploaded_file.name}",
                    mime="image/jpeg",
                    use_container_width=True,
                )

else:
    ui.render_empty_state()
    ui.render_what_it_detects()
    # Even before upload, expose the load status so the user can see what's
    # actually ready to run.
    ui.render_debug_panel(runtime)

ui.render_footer()
