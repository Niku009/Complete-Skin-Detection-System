"""Reusable UI fragments rendered with Streamlit markdown."""
from __future__ import annotations

import streamlit as st

from .detection import AnalysisResult
from .utils import pretty_pct, severity_from_conf, severity_from_count


# ---------------------------------------------------------------------------
# Header / Hero / Footer
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
<div class="brandbar">
  <div class="logo">
    <div class="mark">S</div>
    <div class="name">Skinwise<small>AI Skin Analysis</small></div>
  </div>
  <div class="nav">
    <span>Analyze</span>
    <span>Models</span>
    <span>About</span>
  </div>
  <div class="pill">Beta</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-inner">
    <span class="hero-eyebrow">AI Powered · 4 Vision Models</span>
    <h1>Your skin, <em>read with care.</em></h1>
    <p class="hero-sub">
      Upload one selfie and four specialised computer-vision models read your
      face for dark circles, acne, redness, eye bags and skin type — all in a
      single, quiet pass.
    </p>
    <div class="hero-pills">
      <span class="hero-pill"><span class="dot"></span>Dark Circles</span>
      <span class="hero-pill"><span class="dot"></span>Acne</span>
      <span class="hero-pill"><span class="dot"></span>Redness</span>
      <span class="hero-pill"><span class="dot"></span>Eye Bags</span>
      <span class="hero-pill"><span class="dot"></span>Skin Type</span>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
<div class="foot">
  <strong>Skinwise</strong> · YOLOv8 · KerasCV · EfficientNet · ResNet50<br/>
  Results are informational and not a medical diagnosis.
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section title
# ---------------------------------------------------------------------------
def section_title(eyebrow: str, title: str, sub: str) -> None:
    st.markdown(
        f"""
<div class="sec">
  <div class="sec-eyebrow">{eyebrow}</div>
  <div class="sec-title">{title}</div>
  <div class="sec-sub">{sub}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Upload prompt
# ---------------------------------------------------------------------------
def render_upload_intro() -> None:
    st.markdown(
        """
<div class="upload-card">
  <div class="upload-title">Upload a clear selfie</div>
  <div class="upload-sub">
    JPG · PNG · JPEG · Good lighting and a face-forward angle work best.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Results: 4 summary cards + 2 detail cards
# ---------------------------------------------------------------------------
def render_results_summary(r: AnalysisResult) -> None:
    dc_label, dc_class = severity_from_count(r.dark_circles)
    ac_label, ac_class = severity_from_count(r.acne)
    rd_label, rd_class = severity_from_conf(r.redness, r.redness_conf)

    skin_badges = {
        "Dry":     ('b-bl', 'Hydration needed'),
        "Normal":  ('b-g',  'Balanced'),
        "Oily":    ('b-y',  'Excess sebum'),
        "Unknown": ('b-au', 'Not available'),
    }
    sk_class, sk_sub = skin_badges.get(r.skin_type, skin_badges['Unknown'])

    st.markdown(
        f"""
<div class="mgrid">
  <div class="mcard">
    <div class="mcard-ico">◔</div>
    <div class="mcard-lbl">Dark Circles</div>
    <div class="mcard-val">{r.dark_circles}</div>
    <span class="bdg {dc_class}">{dc_label}</span>
  </div>
  <div class="mcard">
    <div class="mcard-ico">●</div>
    <div class="mcard-lbl">Acne Spots</div>
    <div class="mcard-val">{r.acne}</div>
    <span class="bdg {ac_class}">{ac_label}</span>
  </div>
  <div class="mcard">
    <div class="mcard-ico">❀</div>
    <div class="mcard-lbl">Facial Redness</div>
    <div class="mcard-val">{"Yes" if r.redness else "No"}</div>
    <span class="bdg {rd_class}">{rd_label}</span>
  </div>
  <div class="mcard">
    <div class="mcard-ico">◐</div>
    <div class="mcard-lbl">Skin Type</div>
    <div class="mcard-val">{r.skin_type}</div>
    <span class="bdg {sk_class}">{sk_sub}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_results_detail(r: AnalysisResult, confidence: float) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
<div class="card">
  <div class="card-h"><span class="ico">◧</span> Object Detection (YOLO)</div>
  <div class="drow">
    <div class="dkey"><span class="ddot" style="background: var(--success)"></span>Dark Circles</div>
    <span class="dval">{r.dark_circles} detected</span>
  </div>
  <div class="drow">
    <div class="dkey"><span class="ddot" style="background: var(--orange)"></span>Acne Spots</div>
    <span class="dval">{r.acne} detected</span>
  </div>
  <div style="margin-top:14px; padding-top:14px; border-top: 1px solid var(--border);">
    <div style="font-size:10px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;">
      Confidence threshold
    </div>
    <div class="cbar">
      <span>{confidence:.0%}</span>
      <div class="cbar-bg"><div class="cbar-f" style="width:{confidence*100:.0f}%"></div></div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        r_pct = pretty_pct(r.redness_conf)
        b_pct = pretty_pct(r.bags_conf)
        sk_pct = pretty_pct(r.skin_type_conf)
        st.markdown(
            f"""
<div class="card">
  <div class="card-h"><span class="ico">◇</span> Classification Models</div>
  <div class="drow">
    <div class="dkey"><span class="ddot" style="background: var(--danger)"></span>Facial Redness</div>
    <span class="dval">{"Yes" if r.redness else "No"} · {r_pct}</span>
  </div>
  <div class="cbar">
    <div class="cbar-bg"><div class="cbar-f" style="width:{r.redness_conf*100:.0f}%"></div></div>
    <span>{r_pct}</span>
  </div>
  <div class="drow">
    <div class="dkey"><span class="ddot" style="background: #355C8F"></span>Eye Bags</div>
    <span class="dval">{"Yes" if r.bags else "No"} · {b_pct}</span>
  </div>
  <div class="cbar">
    <div class="cbar-bg"><div class="cbar-f" style="width:{r.bags_conf*100:.0f}%"></div></div>
    <span>{b_pct}</span>
  </div>
  <div class="drow">
    <div class="dkey"><span class="ddot" style="background: var(--orange)"></span>Skin Type</div>
    <span class="dval">{r.skin_type} · {sk_pct}</span>
  </div>
  <div class="cbar">
    <div class="cbar-bg"><div class="cbar-f" style="width:{r.skin_type_conf*100:.0f}%"></div></div>
    <span>{sk_pct}</span>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Empty state (before upload) + What it detects card
# ---------------------------------------------------------------------------
def render_empty_state() -> None:
    st.markdown(
        """
<div class="empty">
  <div class="empty-ico">◭</div>
  <div class="empty-title">Begin with a single photo</div>
  <div class="empty-sub">
    Upload a clear, well-lit selfie. The four AI models will read your skin
    and present a complete analysis in seconds.<br/><br/>
    <em>Supports JPG · PNG · JPEG</em>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_what_it_detects() -> None:
    with st.expander("What does it detect?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
<div class="card">
  <div class="card-h"><span class="ico">◧</span> Object Detection</div>
  <div class="feat">
    <div class="num">1</div>
    <div class="body"><b>Dark Circles</b> · YOLOv8 trained for periorbital shadow detection.</div>
  </div>
  <div class="feat">
    <div class="num">2</div>
    <div class="body"><b>Acne Spots</b> · KerasCV YOLOv8 (XS) bounding boxes for active blemishes.</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
<div class="card">
  <div class="card-h"><span class="ico">◇</span> Classification</div>
  <div class="feat">
    <div class="num">3</div>
    <div class="body"><b>Redness &amp; Eye Bags</b> · EfficientNet-B0 multi-label classifier.</div>
  </div>
  <div class="feat">
    <div class="num">4</div>
    <div class="body"><b>Skin Type</b> · ResNet50 classifier across Dry · Normal · Oily.</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Model availability strip (shown at top so the user knows what's loaded)
# ---------------------------------------------------------------------------
PRETTY_NAMES = {
    "dark_circle": "Dark Circles",
    "acne":        "Acne",
    "redness":     "Redness",
    "skin_type":   "Skin Type",
}


def render_model_status(availability: dict[str, bool]) -> None:
    pills = []
    for key, ok in availability.items():
        cls = "b-g" if ok else "b-au"
        dot = "●" if ok else "○"
        pills.append(
            f'<span class="bdg {cls}" style="margin-right:6px">'
            f'{dot}  {PRETTY_NAMES[key]}</span>'
        )
    st.markdown(
        f'<div style="margin: -8px 0 12px">{" ".join(pills)}</div>',
        unsafe_allow_html=True,
    )


def render_debug_panel(status: dict[str, dict], result=None) -> None:
    """Diagnostic expander: which models actually loaded and what they returned."""
    any_failure = any(not info["loaded"] for info in status.values())
    with st.expander(
        "Diagnostics (model load + inference status)",
        expanded=any_failure or result is not None,
    ):
        for key, info in status.items():
            name = PRETTY_NAMES[key]
            if info["loaded"]:
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<span class='bdg b-g'>● {name}</span> &nbsp; loaded OK"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                err = info.get("error", "unknown reason")
                st.markdown(
                    f"<div style='padding:4px 0'>"
                    f"<span class='bdg b-r'>○ {name}</span> &nbsp; "
                    f"<code style='font-size:11px'>{err}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if result is not None:
            st.markdown("---")
            st.markdown("**Last inference output**")
            st.json(result.as_dict())
