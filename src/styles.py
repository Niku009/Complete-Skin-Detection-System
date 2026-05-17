"""Global CSS injected into Streamlit. Inspired by a warm cream + orange,
hand-pressed-paper aesthetic with an organic curved horizon."""
from __future__ import annotations

import streamlit as st

from .config import PALETTE


def _css() -> str:
    p = PALETTE
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..600;1,9..144,300..500&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {{
  --cream:   {p['cream']};
  --cream2:  {p['cream2']};
  --paper:   {p['paper']};
  --orange:  {p['orange']};
  --orange2: {p['orange2']};
  --orangeD: {p['orangeD']};
  --ink:     {p['ink']};
  --muted:   {p['muted']};
  --border:  {p['border']};
  --success: {p['success']};
  --warn:    {p['warn']};
  --danger:  {p['danger']};
  --serif:   'Fraunces', 'Cormorant Garamond', Georgia, serif;
  --sans:    'Inter', system-ui, -apple-system, sans-serif;
}}

/* ── Hide chrome ───────────────────────────────────────────── */
#MainMenu, header, footer {{ visibility: hidden; height: 0; }}
.stDeployButton {{ display: none !important; }}

/* ── Base canvas: warm cream paper with subtle sun glow ─────
   Background lives on <body> so .stApp keeps its native
   overflow/scroll behavior. Do NOT set position/overflow on
   .stApp or its direct children — Streamlit relies on those. */
html, body {{
  background:
    radial-gradient(ellipse 900px 500px at 50% -120px,
      var(--cream2) 0%, rgba(255,248,236,0) 60%),
    radial-gradient(ellipse 600px 300px at 50% -50px,
      rgba(255,255,255,.85) 0%, rgba(255,255,255,0) 55%),
    var(--cream) !important;
  color: var(--ink) !important;
}}
.stApp {{
  background: transparent !important;
  font-family: var(--sans) !important;
  color: var(--ink) !important;
}}

/* Paper grain — purely decorative, never blocks input or scroll */
body::before {{
  content: "";
  position: fixed; inset: 0;
  background-image:
    radial-gradient(rgba(180,140,90,.05) 1px, transparent 1px),
    radial-gradient(rgba(180,140,90,.04) 1px, transparent 1px);
  background-size: 3px 3px, 7px 7px;
  background-position: 0 0, 2px 4px;
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: multiply;
  opacity: .85;
}}

.block-container {{
  padding-top: 0.8rem !important;
  padding-bottom: 4rem !important;
  max-width: 1180px !important;
  position: relative;
  z-index: 1;
}}

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3, h4 {{
  font-family: var(--serif) !important;
  color: var(--ink) !important;
  font-weight: 400 !important;
  letter-spacing: -0.01em !important;
}}
p, span, div, label {{ color: var(--ink); }}

/* ── Header bar ───────────────────────────────────────────── */
.brandbar {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 18px;
  background: rgba(255, 251, 244, .65);
  border: 1px solid var(--border);
  border-radius: 999px;
  backdrop-filter: blur(8px);
  margin-bottom: 18px;
}}
.brandbar .logo {{
  display: flex; align-items: center; gap: 12px;
}}
.brandbar .mark {{
  width: 34px; height: 34px; border-radius: 12px;
  background: linear-gradient(135deg, var(--orange), var(--orange2));
  display: grid; place-items: center;
  color: white; font-family: var(--serif); font-size: 17px; font-weight: 600;
  box-shadow: 0 6px 18px rgba(232,99,44,.25);
}}
.brandbar .name {{
  font-family: var(--serif); font-size: 19px; font-weight: 500; color: var(--ink);
  line-height: 1; letter-spacing: -.01em;
}}
.brandbar .name small {{
  display: block; font-family: var(--sans); font-size: 9px;
  letter-spacing: 2.2px; color: var(--muted); text-transform: uppercase;
  font-weight: 600; margin-top: 4px;
}}
.brandbar .nav {{
  display: flex; gap: 22px; font-size: 12px; font-weight: 500;
  letter-spacing: .04em; color: var(--muted); text-transform: uppercase;
}}
.brandbar .nav span {{ cursor: default; }}
.brandbar .pill {{
  font-size: 11px; padding: 6px 12px; border-radius: 999px;
  background: var(--orange); color: white; font-weight: 600; letter-spacing: .04em;
}}

/* ── Hero with orange horizon curve ───────────────────────── */
.hero {{
  position: relative;
  border-radius: 26px;
  overflow: hidden;
  padding: 36px 28px 0;
  background:
    radial-gradient(ellipse 720px 320px at 50% -40px,
      rgba(255,255,255,.95) 0%, rgba(255,255,255,0) 60%),
    linear-gradient(180deg, var(--cream2), var(--cream));
  border: 1px solid var(--border);
  min-height: 380px;
  margin-bottom: 22px;
}}
.hero::after {{
  /* The bold orange dune */
  content: "";
  position: absolute;
  left: -8%; right: -8%; bottom: -160px;
  height: 260px;
  background: linear-gradient(180deg, var(--orange) 0%, var(--orangeD) 100%);
  border-radius: 50% 50% 0 0 / 100% 100% 0 0;
  box-shadow:
    0 -25px 60px rgba(232,99,44,.18),
    inset 0 20px 60px rgba(255,255,255,.15);
  z-index: 0;
}}
.hero::before {{
  /* Sun glow ring */
  content: "";
  position: absolute; top: 14px; left: 50%;
  transform: translateX(-50%);
  width: 200px; height: 200px; border-radius: 50%;
  background:
    radial-gradient(circle, rgba(255,255,255,.95) 0%,
      rgba(255,240,210,.7) 35%, rgba(255,224,170,0) 70%);
  filter: blur(2px);
  z-index: 0;
}}
.hero-inner {{
  position: relative; z-index: 2;
  text-align: center; max-width: 720px; margin: 0 auto;
  padding-bottom: 90px;
}}
.hero-eyebrow {{
  display: inline-block;
  font-size: 9.5px; font-weight: 700; letter-spacing: 3.5px;
  color: var(--orange); text-transform: uppercase;
  padding: 5px 13px; border: 1px solid rgba(232,99,44,.3);
  border-radius: 999px; background: rgba(255,255,255,.55);
  margin-bottom: 14px;
}}
.hero h1 {{
  font-family: var(--serif) !important;
  font-size: clamp(30px, 4.4vw, 52px) !important;
  font-weight: 350 !important;
  line-height: 1.08 !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 14px !important;
  color: var(--ink) !important;
}}
.hero h1 em {{
  font-style: italic;
  color: var(--orange);
  font-weight: 400;
}}
.hero-sub {{
  font-size: 14px !important;
  line-height: 1.65 !important;
  color: var(--muted) !important;
  max-width: 520px; margin: 0 auto 18px !important;
  font-weight: 400 !important;
}}
.hero-pills {{
  display: flex; flex-wrap: wrap; gap: 7px; justify-content: center;
}}
.hero-pill {{
  background: rgba(255,255,255,.7);
  border: 1px solid var(--border);
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 11.5px; color: var(--ink); font-weight: 500;
  display: inline-flex; align-items: center; gap: 6px;
}}
.hero-pill .dot {{ width:6px; height:6px; border-radius:50%; background: var(--orange); }}

/* ── Section titles ───────────────────────────────────────── */
.sec {{ margin: 32px 0 18px; }}
.sec-eyebrow {{
  font-size: 10px; font-weight: 700; letter-spacing: 3px;
  color: var(--orange); text-transform: uppercase;
}}
.sec-title {{
  font-family: var(--serif) !important;
  font-size: clamp(24px, 3vw, 34px) !important;
  font-weight: 400 !important;
  margin: 4px 0 6px !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
}}
.sec-sub {{
  font-size: 13.5px !important; color: var(--muted) !important;
}}

/* ── Card primitive ───────────────────────────────────────── */
.card {{
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px;
  box-shadow: 0 1px 0 rgba(255,255,255,.6) inset,
              0 8px 26px -18px rgba(80,40,10,.15);
}}
.card-h {{
  font-family: var(--serif) !important;
  font-size: 17px !important; color: var(--ink) !important;
  margin: 0 0 14px !important; font-weight: 500 !important;
  display: flex; align-items: center; gap: 8px;
}}
.card-h .ico {{
  width: 28px; height: 28px; border-radius: 9px;
  background: rgba(232,99,44,.1);
  display: inline-grid; place-items: center;
  color: var(--orange); font-size: 14px;
}}

/* ── Upload card ──────────────────────────────────────────── */
.upload-card {{
  background:
    linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,248,236,.65)),
    var(--paper);
  border: 1px dashed rgba(232,99,44,.4);
  border-radius: 22px;
  padding: 16px 20px 4px;
  margin: 8px 0 12px;
  text-align: center;
}}
.upload-title {{
  font-family: var(--serif) !important; font-size: 20px !important;
  color: var(--ink) !important; font-weight: 500 !important;
  margin: 0 0 4px !important;
}}
.upload-sub {{
  font-size: 12.5px !important; color: var(--muted) !important;
  margin-bottom: 10px !important;
}}

/* Streamlit file uploader override */
[data-testid="stFileUploader"] > div {{
  background: rgba(255,255,255,.55) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
  transition: border-color .2s, background .2s !important;
}}
[data-testid="stFileUploader"] > div:hover {{
  border-color: var(--orange) !important;
  background: rgba(255,248,236,.7) !important;
}}
[data-testid="stFileDropzone"] p,
[data-testid="stFileUploader"] label {{
  color: var(--muted) !important;
}}
[data-testid="stFileUploader"] section button {{
  background: var(--ink) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {{
  background: var(--paper) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}
[data-testid="stExpander"] summary {{
  color: var(--ink) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 14px 18px !important;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--orange) !important; }}

/* ── Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] label {{ color: var(--muted) !important; font-size: 12px !important; }}
[data-testid="stSlider"] [role="slider"] {{
  background: var(--orange) !important; border-color: var(--orange) !important;
}}

/* ── Image ────────────────────────────────────────────────── */
[data-testid="stImage"] img {{
  border-radius: 16px !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 10px 30px -18px rgba(80,40,10,.25);
}}
.img-label {{
  font-size: 10px; font-weight: 700; letter-spacing: 2.5px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}}

/* ── Metric grid (results) ────────────────────────────────── */
.mgrid {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
  margin: 8px 0 16px;
}}
@media(max-width: 820px) {{
  .mgrid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.mcard {{
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 20px 16px;
  text-align: center;
  position: relative; overflow: hidden;
}}
.mcard::before {{
  content: "";
  position: absolute; top:0; left:0; right:0; height: 3px;
  background: linear-gradient(90deg, var(--orange), var(--orange2));
  opacity: .8;
}}
.mcard-ico {{
  width: 42px; height: 42px; margin: 4px auto 12px;
  border-radius: 14px; display: grid; place-items: center;
  background: rgba(232,99,44,.10);
  color: var(--orange); font-size: 20px;
}}
.mcard-lbl {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}}
.mcard-val {{
  font-family: var(--serif); font-size: 34px; font-weight: 500;
  color: var(--ink); line-height: 1; margin-bottom: 10px;
}}

/* Badges */
.bdg {{
  display: inline-block; padding: 4px 11px; border-radius: 999px;
  font-size: 10.5px; font-weight: 600; letter-spacing: .04em;
}}
.b-g  {{ background: #E8F0E1; color: var(--success); }}
.b-y  {{ background: #FBF1D9; color: #8A6A1A; }}
.b-o  {{ background: #FBE1D0; color: var(--orangeD); }}
.b-r  {{ background: #F7DDD8; color: var(--danger); }}
.b-bl {{ background: #E0EAF5; color: #355C8F; }}
.b-au {{ background: #F2E9D9; color: var(--muted); }}

/* ── Detail card rows ─────────────────────────────────────── */
.drow {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}}
.drow:last-child {{ border-bottom: none; }}
.dkey {{
  display: flex; align-items: center; gap: 9px;
  font-size: 13px !important; color: var(--ink) !important;
}}
.ddot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.dval {{ font-size: 13px !important; font-weight: 600 !important; color: var(--ink) !important; }}

/* Confidence bars */
.cbar {{
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; color: var(--muted);
  margin: 4px 0 10px;
}}
.cbar-bg {{
  flex: 1; height: 4px; background: var(--border);
  border-radius: 2px; overflow: hidden; max-width: 220px;
}}
.cbar-f  {{
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, var(--orange), var(--orange2));
}}

/* ── Info / Alerts ────────────────────────────────────────── */
.info-box {{
  background: rgba(232,99,44,.07);
  border: 1px solid rgba(232,99,44,.25);
  border-radius: 14px;
  padding: 14px 18px;
  margin: 12px 0;
  font-size: 13.5px;
  color: var(--ink);
}}
.info-box strong {{ color: var(--orangeD); }}

[data-testid="stAlert"] {{
  background: var(--paper) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
}}

/* ── Download button ──────────────────────────────────────── */
[data-testid="stDownloadButton"] > button,
.stButton > button {{
  background: var(--ink) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 14px 26px !important;
  width: 100% !important;
  transition: transform .15s, box-shadow .15s, background .15s !important;
  letter-spacing: .02em !important;
}}
[data-testid="stDownloadButton"] > button:hover,
.stButton > button:hover {{
  background: var(--orange) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 26px -8px rgba(232,99,44,.4) !important;
}}

/* ── Empty state ──────────────────────────────────────────── */
.empty {{
  text-align: center; padding: 56px 24px;
  background: rgba(255,255,255,.55);
  border: 1px dashed var(--border);
  border-radius: 22px;
  margin: 14px 0 24px;
}}
.empty-ico {{
  width: 64px; height: 64px; margin: 0 auto 16px;
  border-radius: 20px; background: rgba(232,99,44,.1);
  display: grid; place-items: center; color: var(--orange); font-size: 28px;
}}
.empty-title {{
  font-family: var(--serif) !important; font-size: 28px !important;
  font-weight: 400 !important; color: var(--ink) !important; margin: 0 0 8px !important;
}}
.empty-sub {{
  font-size: 13.5px !important; color: var(--muted) !important;
  line-height: 1.7 !important; max-width: 460px; margin: 0 auto !important;
}}

/* ── Footer ───────────────────────────────────────────────── */
.foot {{
  text-align: center; padding: 28px 12px 8px;
  font-size: 12px; color: var(--muted);
  border-top: 1px solid var(--border);
  margin-top: 36px;
  line-height: 1.7;
}}
.foot strong {{ color: var(--orangeD); font-weight: 600; }}

/* ── HR ────────────────────────────────────────────────── */
hr {{ border: 0 !important; border-top: 1px solid var(--border) !important; margin: 24px 0 !important; }}

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] p {{ color: var(--orange) !important; font-weight: 500 !important; }}

/* ── Columns gap ──────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {{ gap: 16px !important; }}

/* ── Tiny utility: feature row ────────────────────────────── */
.feat {{
  display: flex; align-items: flex-start; gap: 12px; padding: 10px 0;
}}
.feat .num {{
  width: 30px; height: 30px; border-radius: 10px;
  background: rgba(232,99,44,.1); color: var(--orange);
  display: grid; place-items: center; font-family: var(--serif);
  font-size: 15px; font-weight: 600; flex-shrink: 0;
}}
.feat .body {{ font-size: 13.5px; color: var(--ink); line-height: 1.55; }}
.feat .body b {{ font-weight: 600; }}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(_css(), unsafe_allow_html=True)
