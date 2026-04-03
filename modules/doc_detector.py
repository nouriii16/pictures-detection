"""
=============================================================================
MODUL DOC DETECTOR — Pendeteksi Jenis Gambar
=============================================================================
Menentukan apakah gambar yang diupload adalah:
  - DOCUMENT  : screenshot, bukti transfer, KTP, ijazah, struk, dll.
  - PHOTO     : foto dari kamera (DSLR, HP, dll.)

Cara kerja:
  Screenshot/dokumen memiliki ciri khas:
    1. Noise sensor sangat rendah (tidak ada grain kamera)
    2. Banyak piksel identik (area background polos)
    3. Proporsi area putih/terang sangat tinggi
    4. Distribusi warna sangat tidak merata (dominated by few colors)
    5. Tepi piksel sangat tajam dan presisi (bukan optical blur)
=============================================================================
"""

import numpy as np
from PIL import Image, ImageFilter
from dataclasses import dataclass


@dataclass
class DocDetectionResult:
    is_document: bool
    image_type: str         # 'DOCUMENT' | 'PHOTO'
    confidence: float       # 0.0 – 1.0
    noise_std: float        # rendah = dokumen
    unique_ratio: float     # rendah = dokumen (banyak piksel identik)
    white_ratio: float      # tinggi = dokumen (banyak area putih)
    color_concentration: float  # tinggi = dokumen (warna sedikit dominan)
    edge_sharpness: float   # tinggi = dokumen (tepi sangat tajam)
    reason: str


def detect_image_type(image_path: str) -> DocDetectionResult:
    """
    Deteksi apakah gambar adalah dokumen/screenshot atau foto kamera.

    Returns DocDetectionResult dengan is_document=True jika dokumen.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize untuk efisiensi (tidak perlu resolusi penuh untuk deteksi ini)
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    total_px = h * w

    # ── 1. NOISE LEVEL ──────────────────────────────────────────────────────
    # Foto kamera selalu punya noise sensor. Screenshot tidak.
    gray = arr.mean(axis=2)
    gray_img = Image.fromarray(gray.astype(np.uint8))
    blurred = np.array(gray_img.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    noise_std = float(np.std(gray - blurred))
    # Dokumen: < 2.5 | Foto: > 4.0

    # ── 2. UNIQUE PIXEL RATIO ───────────────────────────────────────────────
    # Screenshot punya banyak piksel identik (background, teks warna solid)
    # Sample acak untuk efisiensi
    flat = arr.reshape(-1, 3)
    sample_size = min(5000, len(flat))
    idx = np.random.choice(len(flat), sample_size, replace=False)
    sample = flat[idx].astype(np.int32)
    # Kuantisasi ke 16-level untuk mengelompokkan warna mirip
    quantized = (sample // 16)
    unique_colors = len(np.unique(quantized.view(np.dtype((np.void, quantized.dtype.itemsize * 3)))))
    unique_ratio = float(unique_colors / sample_size)
    # Dokumen: < 0.25 | Foto: > 0.45

    # ── 3. WHITE/LIGHT AREA RATIO ───────────────────────────────────────────
    # Dokumen (struk, transfer, KTP) punya background putih dominan
    white_mask = np.all(arr > 220, axis=2)
    white_ratio = float(white_mask.sum() / total_px)
    # Dokumen: > 0.30 | Foto: < 0.15

    # ── 4. COLOR CONCENTRATION ──────────────────────────────────────────────
    # Dokumen punya sedikit warna dominan (putih, hitam, 1-2 accent color)
    hist_r = np.histogram(arr[:, :, 0], bins=16, range=(0, 256))[0]
    hist_g = np.histogram(arr[:, :, 1], bins=16, range=(0, 256))[0]
    hist_b = np.histogram(arr[:, :, 2], bins=16, range=(0, 256))[0]
    # Hitung konsentrasi: proporsi piksel di 3 bin teratas
    top3_r = np.sort(hist_r)[-3:].sum() / total_px
    top3_g = np.sort(hist_g)[-3:].sum() / total_px
    top3_b = np.sort(hist_b)[-3:].sum() / total_px
    color_concentration = float((top3_r + top3_g + top3_b) / 3)
    # Dokumen: > 0.55 | Foto: < 0.35

    # ── 5. EDGE SHARPNESS ───────────────────────────────────────────────────
    # Teks dan border di dokumen punya tepi sangat tajam (bukan optical blur)
    sobel_h = np.abs(np.diff(gray, axis=0))
    sobel_v = np.abs(np.diff(gray, axis=1))
    # Rasio piksel dengan gradient sangat tinggi vs gradient sedang
    strong_edges = float(np.sum(sobel_h > 50) + np.sum(sobel_v > 50))
    medium_edges = float(np.sum(sobel_h > 10) + np.sum(sobel_v > 10)) + 1e-6
    edge_sharpness = min(1.0, strong_edges / medium_edges)
    # Dokumen: > 0.45 | Foto: < 0.25

    # ── SCORING ─────────────────────────────────────────────────────────────
    doc_score = 0.0
    reasons = []

    # Bobot masing-masing fitur
    if noise_std < 2.5:
        doc_score += 0.30
        reasons.append(f"noise sangat rendah ({noise_std:.2f})")
    elif noise_std < 4.0:
        doc_score += 0.12

    if unique_ratio < 0.20:
        doc_score += 0.25
        reasons.append(f"warna sangat seragam (unique={unique_ratio:.2f})")
    elif unique_ratio < 0.30:
        doc_score += 0.10

    if white_ratio > 0.35:
        doc_score += 0.20
        reasons.append(f"area putih dominan ({white_ratio:.1%})")
    elif white_ratio > 0.20:
        doc_score += 0.08

    if color_concentration > 0.60:
        doc_score += 0.15
        reasons.append(f"warna sangat terkonsentrasi ({color_concentration:.2f})")
    elif color_concentration > 0.45:
        doc_score += 0.06

    if edge_sharpness > 0.45:
        doc_score += 0.10
        reasons.append(f"tepi sangat tajam ({edge_sharpness:.2f})")

    doc_score = min(1.0, doc_score)
    is_doc = doc_score >= 0.60

    if is_doc:
        image_type = "DOCUMENT"
        reason = "Terdeteksi sebagai dokumen/screenshot: " + ", ".join(reasons) if reasons else "Karakteristik umum dokumen digital."
    else:
        image_type = "PHOTO"
        reason = "Terdeteksi sebagai foto kamera (ada noise sensor, variasi warna alami)."

    return DocDetectionResult(
        is_document=is_doc,
        image_type=image_type,
        confidence=doc_score if is_doc else (1.0 - doc_score),
        noise_std=round(noise_std, 3),
        unique_ratio=round(unique_ratio, 4),
        white_ratio=round(white_ratio, 4),
        color_concentration=round(color_concentration, 4),
        edge_sharpness=round(edge_sharpness, 4),
        reason=reason,
    )
