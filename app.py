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
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── DASAR ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: #f0f4f8;   /* abu-abu muda, lembut di mata */
    color: #1e293b;        /* teks gelap agar mudah dibaca */
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}

/* ── HEADER UTAMA ── */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 60%, #818cf8 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(79,70,229,0.25);
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.app-header p {
    color: rgba(255,255,255,0.75);
    font-size: 0.82rem;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}
.app-badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.35);
}

/* ── KARTU VERDICT (hasil analisis) ── */
.vcard {
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin: 0.8rem 0;
    border-left: 5px solid;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
/* Warna kartu sesuai verdict */
.v-ai    { border-color: #a855f7; background: #faf5ff; }  /* ungu — foto AI */
.v-real  { border-color: #16a34a; background: #f0fdf4; }  /* hijau — foto asli bersih */
.v-manip { border-color: #dc2626; background: #fef2f2; }  /* merah — dimanipulasi */
.v-warn  { border-color: #ea580c; background: #fff7ed; }  /* oranye — asli tapi diedit */
.v-unc   { border-color: #ca8a04; background: #fefce8; }  /* kuning — tidak konklusif */

.vcard h2 { margin: 0 0 0.25rem; font-size: 1.3rem; font-weight: 700; color: #1e293b; }
.vcard p  { margin: 0; font-size: 0.82rem; color: #475569; }

/* ── KOTAK STATISTIK (Mean Error, Std Error, dll) ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-lbl {
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.2rem;
}
.stat-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    font-family: 'JetBrains Mono', monospace;
}

/* ── BAR SKOR FITUR AI ── */
.feat-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.4rem 0; }
.feat-name {
    font-size: 0.78rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    width: 200px;
    flex-shrink: 0;
}
.feat-bar-wrap {
    flex: 1;
    background: #e2e8f0;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.feat-bar { height: 10px; border-radius: 6px; transition: width 0.4s; }
.feat-val {
    font-size: 0.78rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    width: 42px;
}

/* ── BADGE RISIKO ── */
.risk-high   { background: #dc2626; color: #fff; }
.risk-medium { background: #ea580c; color: #fff; }
.risk-low    { background: #16a34a; color: #fff; }
.risk-badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── TAB NAVIGASI ── */
.stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    border-radius: 8px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

/* ── TOMBOL UTAMA ── */
.stButton > button {
    background: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.8rem;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3);
}
.stButton > button:hover {
    background: #4338ca;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79,70,229,0.4);
}

/* ── AREA UPLOAD FILE ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
}

/* ── KOTAK LAPORAN ── */
.rbox {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #475569;
    white-space: pre-wrap;
    line-height: 1.75;
    max-height: 420px;
    overflow-y: auto;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── LABEL SECTION ── */
.section-lbl {
    font-size: 0.7rem;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

/* ── SEMBUNYIKAN ELEMEN BAWAAN STREAMLIT ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OVERALL_CSS = {
    "AI_MANIPULATED":   ("v-manip", "⛔  FOTO AI + DIMANIPULASI"),
    "AI_GENERATED":     ("v-ai",    "🤖  FOTO AI-GENERATED"),
    "REAL_MANIPULATED": ("v-warn",  "⚠️  FOTO ASLI DIMANIPULASI"),
    "REAL_AUTHENTIC":   ("v-real",  "✅  FOTO ASLI & BERSIH"),
    "UNCERTAIN":        ("v-unc",   "❓  TIDAK KONKLUSIF"),
}

ELA_CSS = {
    "MANIPULATED": ("v-manip", "⚠️  TERDETEKSI MANIPULASI"),
    "AUTHENTIC":   ("v-real",  "✅  TIDAK ADA MANIPULASI"),
    "UNCERTAIN":   ("v-unc",   "❓  TIDAK KONKLUSIF"),
}

AI_CSS = {
    "AI_GENERATED": ("v-ai",   "🤖  FOTO AI-GENERATED"),
    "REAL_PHOTO":   ("v-real", "📷  FOTO ASLI (KAMERA)"),
    "UNCERTAIN":    ("v-unc",  "❓  TIDAK KONKLUSIF"),
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
        color = "#22c55e" if v >= 0.5 else "#ef4444"
        rows.append(f"""
        <div class="feat-row">
          <div class="feat-name">{label}</div>
          <div class="feat-bar-wrap">
            <div class="feat-bar" style="width:{v*100:.1f}%;background:{color}"></div>
          </div>
          <div class="feat-val">{v:.2f}</div>
        </div>""")
    return "".join(rows)


def stat_box(label, value, color="#e2e8f0"):
    return (f'<div class="stat-box"><div class="stat-lbl">{label}</div>'
            f'<div class="stat-val" style="color:{color}">{value}</div></div>')


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.5rem'>🔬</div>
        <div style='color:#6366f1;font-size:.72rem;letter-spacing:.15em;
                    text-transform:uppercase;font-weight:600;margin-top:.3rem'>
            Forensik Citra Digital
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Parameter ELA**")
    quality = st.slider("Kualitas Rekompresi JPEG", 50, 95, 90, 5,
                        help="Kualitas JPEG saat rekompresi. Default 90%.")
    amplify = st.slider("Amplifikasi Visualisasi", 5, 30, 15, 1,
                        help="Penguat error map. Nilai lebih tinggi = perbedaan lebih terlihat.")

    st.markdown("**Multi-Kualitas ELA**")
    multi_q = st.multiselect("Kualitas untuk perbandingan:",
                              [70, 75, 80, 85, 90, 95], default=[70, 80, 90, 95])

    st.divider()
    st.markdown("""
    <div style='font-size:.72rem;color:#1e293b;line-height:1.7'>
    <b style='color:#334155'>Kelompok 7</b><br>
    Ilmu Komputer — UNIMED 2026<br>
    Dosen: Dr. Hermawan Syahputra
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <div class="app-badge">Forensik Citra Digital</div>
  <h1>Sistem Deteksi Manipulasi Citra Digital</h1>
  <p>Error Level Analysis (ELA) · AI-Generated Image Detection</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown('<div class="section-lbl">↑ Upload Citra untuk Dianalisis</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload gambar", type=["jpg","jpeg","png","bmp","webp"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:3.5rem 0;color:#1e293b'>
      <div style='font-size:3rem'>📂</div>
      <div style='color:#334155;font-size:1rem;margin:.6rem 0'>Upload gambar untuk memulai analisis</div>
      <div style='font-size:.8rem;color:#1e293b'>Format: JPG · PNG · BMP · WEBP</div>
      <div style='margin-top:1.5rem;font-size:.78rem;color:#1e3060;background:#080f1e;
                  border:1px solid #1a2540;border-radius:10px;padding:.8rem 1.5rem;
                  display:inline-block;font-family:monospace'>
        Dua analisis dalam satu sistem:<br>
        🔬 ELA — deteksi editing & manipulasi &nbsp;|&nbsp; 🤖 AI Detection — deteksi foto AI-generated
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# Simpan ke temp file
suffix = os.path.splitext(uploaded.name)[1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

col_info, col_btn = st.columns([3, 1])
with col_info:
    sz = len(uploaded.getvalue()) / 1024
    st.markdown(f'<div style="color:#475569;font-size:.8rem;font-family:monospace;padding:.4rem 0">'
                f'📄 <b style="color:#94a3b8">{uploaded.name}</b> &nbsp;·&nbsp; {sz:.1f} KB</div>',
                unsafe_allow_html=True)
with col_btn:
    run = st.button("🔬  Analisis Sekarang", use_container_width=True)

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

    prog.progress(70, "Menggabungkan hasil (fusion)...")
    full       = fuse_full_analysis(ela_result, ai_result)

    prog.progress(88, "Merender visualisasi...")
    img_ela    = render_ela_panels(ela_result)
    img_mask   = render_mask_overlay(ela_result)
    img_mq     = render_multi_quality(tmp_path, multi_q) if multi_q else None
    img_ai     = render_ai_scores(ai_result)

    prog.progress(100, "Selesai!")

prog.empty()

# ---------------------------------------------------------------------------
# Overall verdict banner
# ---------------------------------------------------------------------------
ov_css, ov_label = OVERALL_CSS.get(full.overall_verdict, ("v-unc", "❓ UNCERTAIN"))
risk_css  = f"risk-{full.risk_level.lower()}"

st.markdown(f"""
<div class="vcard {ov_css}">
  <h2>{ov_label}</h2>
  <p>{full.summary} &nbsp;
    <span class="risk-badge {risk_css}">Risk: {full.risk_level}</span>
  </p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dua kolom ringkasan
# ---------------------------------------------------------------------------
col_ela, col_ai = st.columns(2)

with col_ela:
    st.markdown('<div class="section-lbl">🔬 ELA — Deteksi Manipulasi/Editing</div>', unsafe_allow_html=True)
    e_css, e_lbl = ELA_CSS.get(ela_result.verdict, ("v-unc", "❓"))
    st.markdown(f"""
    <div class="vcard {e_css}" style="padding:1rem 1.4rem">
      <h2 style="font-size:1.1rem">{e_lbl}</h2>
      <p>Confidence: <b>{ela_result.confidence:.1%}</b></p>
    </div>""", unsafe_allow_html=True)

    ec1 = "#ef4444" if ela_result.verdict == "MANIPULATED" else "#22c55e"
    st.markdown(f"""
    <div class="stat-row">
      {stat_box("Mean Error",    f"{ela_result.mean_error:.3f}")}
      {stat_box("Std Error",     f"{ela_result.std_error:.3f}")}
      {stat_box("Max Error",     f"{ela_result.max_error:.1f}")}
      {stat_box("Suspicious",    f"{ela_result.suspicious_ratio:.2%}", ec1)}
    </div>""", unsafe_allow_html=True)

with col_ai:
    st.markdown('<div class="section-lbl">🤖 AI Detection — Foto Asli vs AI-Generated</div>', unsafe_allow_html=True)
    a_css, a_lbl = AI_CSS.get(ai_result.verdict, ("v-unc", "❓"))
    st.markdown(f"""
    <div class="vcard {a_css}" style="padding:1rem 1.4rem">
      <h2 style="font-size:1.1rem">{a_lbl}</h2>
      <p>Confidence: <b>{ai_result.confidence:.1%}</b>
         &nbsp;·&nbsp; AI Probability: <b>{ai_result.ai_probability:.1%}</b></p>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Skor Fitur Deteksi AI** *(hijau = cenderung real, merah = cenderung AI)*")
    st.markdown(feat_bars_html(ai_result.scores), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab detail
# ---------------------------------------------------------------------------
st.markdown('<div class="section-lbl" style="margin-top:1.5rem">Detail Analisis</div>', unsafe_allow_html=True)

tab_ela, tab_mask, tab_mq, tab_ai, tab_report = st.tabs([
    "📊 ELA Map",
    "🎯 Area Mencurigakan",
    "🔬 Multi-Kualitas",
    "🤖 AI Scores Chart",
    "📄 Laporan",
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
        st.download_button("⬇ Laporan TXT", report_text.encode(), f"laporan_{base}.txt", "text/plain", use_container_width=True)
    with c2:
        ela_buf = io.BytesIO()
        ela_result.ela_image.save(ela_buf, "PNG"); ela_buf.seek(0)
        st.download_button("⬇ ELA Map PNG", ela_buf, f"ela_map_{base}.png", "image/png", use_container_width=True)
    with c3:
        st.download_button("⬇ Visualisasi ELA", img_ela, f"visualisasi_{base}.png", "image/png", use_container_width=True)

# Cleanup
try: os.unlink(tmp_path)
except: pass
