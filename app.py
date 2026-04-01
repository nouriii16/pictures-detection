"""
=============================================================================
APP.PY — Web App Streamlit
=============================================================================
Sistem Deteksi Manipulasi Citra Digital
Menggunakan Error Level Analysis (ELA) dan Machine Learning

Cara jalankan:
  streamlit run app.py
=============================================================================
"""

import sys, os, io, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

import numpy as np
import matplotlib
matplotlib.use("Agg")

import streamlit as st
from PIL import Image

from ela         import analyze_ela, multi_quality_ela
from ai_detector import analyze_ai_statistical
from fusion      import fuse_full_analysis
from report      import generate_full_report
from visualizer  import (render_ela_panels, render_mask_overlay,
                          render_multi_quality, render_ai_scores)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forensik Citra Digital",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Dark Elegant Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #141414; color: #e8e0d4; }

[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2520;
}

.app-header {
    background: #1a1a1a;
    border: 1px solid #2e2a22;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c, #e8d5a3, #c9a84c, transparent);
}
.app-header::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #c9a84c08, transparent 70%);
    pointer-events: none;
}
.app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #f5f0e8;
    margin: 0 0 .3rem;
    letter-spacing: .02em;
}
.app-header p { color: #5a5248; font-size: .72rem; margin: 0; font-family: 'JetBrains Mono', monospace; letter-spacing: .05em; }
.app-badge {
    display: inline-block;
    background: transparent;
    border: 1px solid #c9a84c50;
    color: #c9a84c;
    font-size: .62rem;
    font-weight: 500;
    padding: 3px 14px;
    border-radius: 20px;
    letter-spacing: .2em;
    text-transform: uppercase;
    margin-bottom: .7rem;
    font-family: 'JetBrains Mono', monospace;
}

.de-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c40, transparent);
    margin: 1rem 0;
}

.vcard { border-radius: 12px; padding: 1.2rem 1.5rem; margin: .8rem 0; border: 1px solid; }
.v-ai   { background: #13101a; border-color: #3d2d5a; }
.v-real { background: #0e1a12; border-color: #2d5a3a; }
.v-manip{ background: #1a0e0e; border-color: #5a2d2d; }
.v-warn { background: #1a1608; border-color: #5a4a1a; }
.v-unc  { background: #1a1608; border-color: #5a4a1a; }
.vcard h2 {
    font-family: 'Playfair Display', serif;
    margin: 0 0 .25rem;
    font-size: 1.25rem;
    font-weight: 600;
}
.v-ai   h2 { color: #bb86fc; }
.v-real h2 { color: #6fcf97; }
.v-manip h2{ color: #eb5757; }
.v-warn h2 { color: #f2c94c; }
.v-unc  h2 { color: #f2c94c; }
.vcard p  { margin: 0; font-size: .78rem; color: #6a6460; }

.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: .6rem; margin: .8rem 0; }
.stat-box {
    background: #111111;
    border: 1px solid #2a2520;
    border-radius: 8px;
    padding: .65rem .9rem;
}
.stat-lbl { font-size: .6rem; color: #4a4540; text-transform: uppercase;
            letter-spacing: .1em; font-family: 'JetBrains Mono', monospace; margin-bottom: .2rem; }
.stat-val { font-size: 1rem; font-weight: 500; color: #c8c0b4; font-family: 'JetBrains Mono', monospace; }

.feat-row { display: flex; align-items: center; gap: .8rem; margin: .35rem 0; }
.feat-name { font-size: .72rem; color: #5a5248; font-family: 'JetBrains Mono', monospace;
             width: 200px; flex-shrink: 0; }
.feat-bar-wrap { flex: 1; background: #111111; border-radius: 2px; height: 5px; overflow: hidden; border: 1px solid #2a2520; }
.feat-bar { height: 5px; border-radius: 2px; transition: width .4s; }
.feat-val { font-size: .72rem; color: #5a5248; font-family: 'JetBrains Mono', monospace; width: 42px; }

.risk-high   { background: #1a0e0e; border: 1px solid #5a2d2d; color: #eb5757; }
.risk-medium { background: #1a1608; border: 1px solid #5a4a1a; color: #f2c94c; }
.risk-low    { background: #0e1a12; border: 1px solid #2d5a3a; color: #6fcf97; }
.risk-badge  { display: inline-block; padding: 2px 12px; border-radius: 4px;
               font-size: .62rem; font-weight: 500; letter-spacing: .1em;
               text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }

.stTabs [data-baseweb="tab-list"] { background: #1a1a1a; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #2a2520; }
.stTabs [data-baseweb="tab"]      { color: #5a5248; border-radius: 6px; font-family: 'Inter', sans-serif; font-size: .82rem; }
.stTabs [aria-selected="true"]    { background: #111111 !important; color: #c9a84c !important; }

.stButton>button {
    background: #1e1a12;
    color: #c9a84c;
    border: 1px solid #c9a84c50;
    border-radius: 8px;
    font-family: 'Playfair Display', serif;
    font-size: .9rem;
    padding: .65rem 1.8rem;
    width: 100%;
    transition: all .2s;
}
.stButton>button:hover { background: #252015; border-color: #c9a84c90; }

[data-testid="stFileUploader"] { background: #1a1a1a !important; border: 1px dashed #2a2520 !important; border-radius: 12px !important; }

.rbox {
    background: #111111; border: 1px solid #2a2520; border-radius: 8px;
    padding: 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: .74rem;
    color: #6a6460; white-space: pre-wrap; line-height: 1.75;
    max-height: 420px; overflow-y: auto;
}
.section-lbl {
    font-size: .62rem; color: #c9a84c; text-transform: uppercase;
    letter-spacing: .2em; font-family: 'JetBrains Mono', monospace;
    font-weight: 500; margin-bottom: .5rem;
}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OVERALL_CSS = {
    "AI_MANIPULATED":   ("v-manip", "Foto AI + Dimanipulasi"),
    "AI_GENERATED":     ("v-ai",    "Foto AI-Generated"),
    "REAL_MANIPULATED": ("v-warn",  "Foto Asli Dimanipulasi"),
    "REAL_AUTHENTIC":   ("v-real",  "Foto Asli & Bersih"),
    "UNCERTAIN":        ("v-unc",   "Tidak Konklusif"),
}

ELA_CSS = {
    "MANIPULATED": ("v-manip", "Terdeteksi Manipulasi"),
    "AUTHENTIC":   ("v-real",  "Tidak Ada Manipulasi"),
    "UNCERTAIN":   ("v-unc",   "Tidak Konklusif"),
}

AI_CSS = {
    "AI_GENERATED": ("v-ai",   "Foto AI-Generated"),
    "REAL_PHOTO":   ("v-real", "Foto Asli (Kamera)"),
    "UNCERTAIN":    ("v-unc",  "Tidak Konklusif"),
}

FEAT_LABELS = {
    "dct_smoothness":      "Noise Frekuensi DCT",
    "color_naturalness":   "Kealamian Warna",
    "noise_pattern":       "Pola Noise Sensor",
    "ela_consistency":     "Konsistensi ELA",
    "sharpness_variation": "Variasi Ketajaman",
}


def feat_bars_html(scores: dict) -> str:
    rows = []
    for k, v in scores.items():
        if k == "ml_raw_probability":
            continue
        label = FEAT_LABELS.get(k, k)
        color = "linear-gradient(90deg,#2d5a3a,#6fcf97)" if v >= 0.5 else "linear-gradient(90deg,#5a2d2d,#eb5757)"
        rows.append(f"""
        <div class="feat-row">
          <div class="feat-name">{label}</div>
          <div class="feat-bar-wrap">
            <div class="feat-bar" style="width:{v*100:.1f}%;background:{color}"></div>
          </div>
          <div class="feat-val">{v:.2f}</div>
        </div>""")
    return "".join(rows)


def stat_box(label, value, color="#c8c0b4"):
    return (f'<div class="stat-box"><div class="stat-lbl">{label}</div>'
            f'<div class="stat-val" style="color:{color}">{value}</div></div>')


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2rem;opacity:.5'>🔬</div>
        <div style='color:#c9a84c;font-size:.62rem;letter-spacing:.2em;
                    text-transform:uppercase;font-weight:500;margin-top:.5rem;
                    font-family:JetBrains Mono,monospace'>
            Forensik Citra
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Parameter ELA</div>', unsafe_allow_html=True)
    quality = st.slider("Kualitas Rekompresi JPEG", 50, 95, 90, 5,
                        help="Kualitas JPEG saat rekompresi. Default 90%.")
    amplify = st.slider("Amplifikasi Visualisasi", 5, 30, 15, 1,
                        help="Penguat error map. Nilai lebih tinggi = perbedaan lebih terlihat.")

    st.markdown('<div class="section-lbl" style="margin-top:1rem">Multi-Kualitas ELA</div>', unsafe_allow_html=True)
    multi_q = st.multiselect("Kualitas untuk perbandingan:",
                              [70, 75, 80, 85, 90, 95], default=[70, 80, 90, 95])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <div class="app-badge">Forensik Citra Digital</div>
  <h1>Sistem Deteksi Manipulasi Citra Digital</h1>
  <p>Error Level Analysis (ELA) &nbsp;·&nbsp; AI-Generated Image Detection &nbsp;·&nbsp; Vision Transformer</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown('<div class="section-lbl">Upload Citra untuk Dianalisis</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload gambar", type=["jpg","jpeg","png","bmp","webp"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:3.5rem 0'>
      <div style='font-size:2.5rem;opacity:.3'>📂</div>
      <div style='color:#4a4540;font-size:.95rem;margin:.6rem 0;font-family:Playfair Display,serif'>
        Upload gambar untuk memulai analisis
      </div>
      <div style='font-size:.72rem;color:#3a3530;font-family:JetBrains Mono,monospace'>
        Format: JPG · PNG · BMP · WEBP
      </div>
      <div style='margin-top:1.5rem;font-size:.7rem;color:#3a3530;background:#1a1a1a;
                  border:1px solid #2a2520;border-radius:10px;padding:.8rem 1.5rem;
                  display:inline-block;font-family:JetBrains Mono,monospace;letter-spacing:.05em'>
        ELA — deteksi editing &amp; manipulasi &nbsp;|&nbsp; AI Detection — deteksi foto AI-generated
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

suffix = os.path.splitext(uploaded.name)[1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

col_info, col_btn = st.columns([3, 1])
with col_info:
    sz = len(uploaded.getvalue()) / 1024
    st.markdown(f'<div style="color:#4a4540;font-size:.78rem;font-family:JetBrains Mono,monospace;padding:.4rem 0">'
                f'<b style="color:#7a7268">{uploaded.name}</b> &nbsp;·&nbsp; {sz:.1f} KB</div>',
                unsafe_allow_html=True)
with col_btn:
    run = st.button("Analisis Sekarang", use_container_width=True)

if not run:
    col_prev, _ = st.columns([1, 1])
    with col_prev:
        st.image(Image.open(tmp_path), caption="Preview", use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Jalankan analisis
# ---------------------------------------------------------------------------
prog = st.progress(0, "Memulai analisis...")

with st.spinner(""):
    prog.progress(15, "Menghitung ELA error map...")
    ela_result = analyze_ela(tmp_path, quality=quality, amplify=amplify)

    prog.progress(45, "Mendeteksi pola foto AI...")
    ai_result  = analyze_ai_statistical(tmp_path)

    prog.progress(70, "Menggabungkan hasil...")
    full       = fuse_full_analysis(ela_result, ai_result)

    prog.progress(88, "Merender visualisasi...")
    img_ela    = render_ela_panels(ela_result)
    img_mask   = render_mask_overlay(ela_result)
    img_mq     = render_multi_quality(tmp_path, multi_q) if multi_q else None
    img_ai     = render_ai_scores(ai_result)

    prog.progress(100, "Selesai.")

prog.empty()

# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------
ov_css, ov_label = OVERALL_CSS.get(full.overall_verdict, ("v-unc", "Tidak Konklusif"))
risk_css = f"risk-{full.risk_level.lower()}"

st.markdown(f"""
<div class="vcard {ov_css}">
  <h2>{ov_label}</h2>
  <p>{full.summary} &nbsp;
    <span class="risk-badge {risk_css}">Risk: {full.risk_level}</span>
  </p>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="de-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dua kolom ringkasan
# ---------------------------------------------------------------------------
col_ela, col_ai = st.columns(2)

with col_ela:
    st.markdown('<div class="section-lbl">ELA — Deteksi Manipulasi</div>', unsafe_allow_html=True)
    e_css, e_lbl = ELA_CSS.get(ela_result.verdict, ("v-unc", "Tidak Konklusif"))
    st.markdown(f"""
    <div class="vcard {e_css}" style="padding:1rem 1.4rem">
      <h2 style="font-size:1rem">{e_lbl}</h2>
      <p>Confidence: <b style="color:#c8c0b4">{ela_result.confidence:.1%}</b></p>
    </div>""", unsafe_allow_html=True)

    ec1 = "#eb5757" if ela_result.verdict == "MANIPULATED" else "#6fcf97"
    st.markdown(f"""
    <div class="stat-row">
      {stat_box("Mean Error",  f"{ela_result.mean_error:.3f}")}
      {stat_box("Std Error",   f"{ela_result.std_error:.3f}")}
      {stat_box("Max Error",   f"{ela_result.max_error:.1f}")}
      {stat_box("Suspicious",  f"{ela_result.suspicious_ratio:.2%}", ec1)}
    </div>""", unsafe_allow_html=True)

with col_ai:
    st.markdown('<div class="section-lbl">AI Detection — Foto Asli vs AI-Generated</div>', unsafe_allow_html=True)
    a_css, a_lbl = AI_CSS.get(ai_result.verdict, ("v-unc", "Tidak Konklusif"))
    st.markdown(f"""
    <div class="vcard {a_css}" style="padding:1rem 1.4rem">
      <h2 style="font-size:1rem">{a_lbl}</h2>
      <p>Confidence: <b style="color:#c8c0b4">{ai_result.confidence:.1%}</b>
         &nbsp;·&nbsp; AI Probability: <b style="color:#c8c0b4">{ai_result.ai_probability:.1%}</b></p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:.65rem;color:#4a4540;font-family:JetBrains Mono,monospace;letter-spacing:.1em;text-transform:uppercase;margin:.6rem 0 .3rem">Skor Fitur &nbsp;·&nbsp; hijau = real &nbsp;|&nbsp; merah = AI</div>', unsafe_allow_html=True)
    st.markdown(feat_bars_html(ai_result.scores), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab detail
# ---------------------------------------------------------------------------
st.markdown('<div class="section-lbl" style="margin-top:1.5rem">Detail Analisis</div>', unsafe_allow_html=True)

tab_ela, tab_mask, tab_mq, tab_ai, tab_report = st.tabs([
    "ELA Map",
    "Area Mencurigakan",
    "Multi-Kualitas",
    "AI Scores",
    "Laporan",
])

with tab_ela:
    st.image(img_ela, use_container_width=True)
    st.caption("Kiri: citra asli · Tengah: ELA map (area terang = inkonsistensi kompresi) · Kanan: heatmap intensitas error")

with tab_mask:
    st.image(img_mask, use_container_width=True)
    st.caption(f"Area merah = piksel dengan error di atas mean + 2σ ({ela_result.suspicious_ratio:.2%} dari total piksel)")

with tab_mq:
    if img_mq:
        st.image(img_mq, use_container_width=True)
        st.caption("Inkonsistensi error antar kualitas = indikasi manipulasi kuat")
    else:
        st.info("Pilih minimal satu kualitas di sidebar.")

with tab_ai:
    st.image(img_ai, use_container_width=True)
    st.markdown("""
    **Panduan membaca skor:**
    - **Noise Frekuensi DCT** — Foto AI sangat halus (minim noise frekuensi tinggi)
    - **Kealamian Warna** — Foto AI sering punya distribusi warna terlalu merata/sempurna
    - **Pola Noise Sensor** — Foto asli dari kamera punya noise sensor yang terstruktur
    - **Konsistensi ELA** — Foto AI tidak punya riwayat kompresi sebelumnya
    - **Variasi Ketajaman** — Foto asli punya depth-of-field alami; foto AI tajam merata
    """)

with tab_report:
    report_text = generate_full_report(full)
    st.markdown(f'<div class="rbox">{report_text}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    base = os.path.splitext(uploaded.name)[0]
    with c1:
        st.download_button("Laporan TXT", report_text.encode(), f"laporan_{base}.txt", "text/plain", use_container_width=True)
    with c2:
        ela_buf = io.BytesIO()
        ela_result.ela_image.save(ela_buf, "PNG"); ela_buf.seek(0)
        st.download_button("ELA Map PNG", ela_buf, f"ela_map_{base}.png", "image/png", use_container_width=True)
    with c3:
        st.download_button("Visualisasi ELA", img_ela, f"visualisasi_{base}.png", "image/png", use_container_width=True)

# Cleanup
try: os.unlink(tmp_path)
except: pass
