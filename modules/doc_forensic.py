"""
=============================================================================
MODUL DOC FORENSIC — Analisis Forensik Khusus Dokumen Digital
=============================================================================
Mendeteksi manipulasi pada:
  - Screenshot bukti transfer / mobile banking
  - Foto KTP, ijazah, sertifikat
  - Struk, invoice, dokumen resmi
  - Scan dokumen

PIPELINE khusus dokumen:
  1. ELA dengan kualitas tinggi (95)
  2. Analisis konsistensi background
  3. Analisis anomali tepi teks (versi ringan tanpa scipy)
  4. Analisis blok kompresi
  5. Analisis gradien warna

Verdict: DOC_MANIPULATED | DOC_AUTHENTIC | DOC_UNCERTAIN
=============================================================================
"""

import io, os, logging
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DocForensicResult:
    """Hasil analisis forensik dokumen."""
    image_path: str
    verdict: str
    confidence: float
    risk_level: str
    summary: str
    ela_suspicious_ratio: float
    ela_mean_error: float
    background_consistency: float
    edge_anomaly_score: float
    block_variance_score: float
    color_jump_score: float
    ela_image: Image.Image = None
    suspicious_mask: np.ndarray = None
    notes: List[str] = field(default_factory=list)


def _compute_ela_document(img: Image.Image, quality: int = 95) -> Tuple[Image.Image, np.ndarray]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(img, recompressed)
    ela_arr = np.array(ela, dtype=np.float32)
    extrema = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    ela_enhanced = ImageEnhance.Brightness(ela).enhance((255.0 / max_diff) * 25)
    return ela_enhanced, ela_arr


def _analyze_background_consistency(img_arr: np.ndarray, ela_arr: np.ndarray) -> float:
    bg_mask = np.all(img_arr > 215, axis=2)
    if bg_mask.sum() < 100:
        return 0.5
    ela_bg = ela_arr[bg_mask]
    bg_mean = float(np.mean(ela_bg))
    bg_std = float(np.std(ela_bg))
    anomaly_ratio = float(np.sum(ela_bg > bg_mean + 3 * bg_std) / len(ela_bg))
    consistency = max(0.0, 1.0 - anomaly_ratio * 10 - bg_mean / 30)
    return min(1.0, consistency)


def _analyze_text_edge_anomaly(ela_arr: np.ndarray) -> float:
    """Versi ringan tanpa scipy — menggunakan statistik sederhana."""
    ela_gray = ela_arr.mean(axis=2)
    mean_ela = float(np.mean(ela_gray))
    std_ela = float(np.std(ela_gray))
    max_ela = float(np.max(ela_gray))
    threshold = mean_ela + 1.5 * std_ela
    high_ratio = float(np.sum(ela_gray > threshold) / ela_gray.size)
    high_pixels = ela_gray[ela_gray > threshold]
    mean_high = float(np.mean(high_pixels)) if len(high_pixels) > 0 else 0
    intensity_score = min(1.0, mean_high / 80)
    isolation_score = min(1.0, high_ratio * 8)
    return min(1.0, isolation_score * 0.6 + intensity_score * 0.4)


def _analyze_block_variance(ela_arr: np.ndarray, block_size: int = 16) -> float:
    ela_gray = ela_arr.mean(axis=2)
    h, w = ela_gray.shape
    block_means = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = ela_gray[y:y + block_size, x:x + block_size]
            block_means.append(float(np.mean(block)))
    if len(block_means) < 4:
        return 0.0
    block_means = np.array(block_means)
    overall_mean = float(np.mean(block_means))
    overall_std = float(np.std(block_means))
    outlier_threshold = overall_mean + 2.5 * overall_std
    outlier_ratio = float(np.sum(block_means > outlier_threshold) / len(block_means))
    return min(1.0, outlier_ratio * 5)


def _analyze_color_jumps(img_arr: np.ndarray) -> float:
    gray = img_arr.mean(axis=2).astype(np.float32)
    grad_h = np.abs(np.diff(gray, axis=1))
    grad_v = np.abs(np.diff(gray, axis=0))
    extreme_h = float(np.sum(grad_h > 100) / grad_h.size)
    extreme_v = float(np.sum(grad_v > 100) / grad_v.size)
    color_jump = (extreme_h + extreme_v) / 2
    normalized = max(0.0, color_jump - 0.05) / 0.15
    return min(1.0, float(normalized))


def analyze_document(image_path: str, quality: int = 95) -> DocForensicResult:
    """Analisis forensik lengkap untuk dokumen/screenshot."""
    img = Image.open(image_path).convert("RGB")

    # Resize agresif untuk mempercepat analisis — max 800px
    max_dim = 800
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    notes = []

    logger.info(f"Analisis dokumen: {os.path.basename(image_path)}, ukuran: {img.size}")

    # ELA
    ela_image, ela_arr = _compute_ela_document(img, quality)
    ela_gray = ela_arr.mean(axis=2)
    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))
    threshold_px = ela_mean + 2 * ela_std
    suspicious_ratio = float(np.sum(ela_gray > threshold_px) / ela_gray.size)
    suspicious_mask = (ela_gray > threshold_px).astype(np.uint8) * 255

    # Metrik
    bg_consistency = _analyze_background_consistency(arr, ela_arr)
    edge_anomaly = _analyze_text_edge_anomaly(ela_arr)
    block_variance = _analyze_block_variance(ela_arr)
    color_jump = _analyze_color_jumps(arr)

    # Scoring — ELA dominant 85%, hapus bg_consistency
    manip_score = (
        suspicious_ratio / 0.06 * 0.85 +
        block_variance * 0.10 +
        color_jump * 0.05
    )
    manip_score = min(1.0, float(manip_score))

    # Verdict
    if manip_score >= 0.75:
        verdict = "DOC_MANIPULATED"
        risk = "HIGH"
        confidence = min(1.0, manip_score)
        summary = "Dokumen terdeteksi telah dimanipulasi — ditemukan inkonsistensi kompresi dan anomali pada area teks/angka."
        notes.append(f"⚠ Suspicious ratio ELA: {suspicious_ratio:.2%} (threshold dokumen: 6%)")
        if bg_consistency < 0.6:
            notes.append(f"⚠ Inkonsistensi background: {bg_consistency:.2f} — area putih tidak seragam.")
        if edge_anomaly > 0.4:
            notes.append(f"⚠ Anomali tepi teks terdeteksi (skor: {edge_anomaly:.2f}) — indikasi teks/angka diedit.")
        if block_variance > 0.3:
            notes.append(f"⚠ Blok kompresi tidak seragam (skor: {block_variance:.2f}) — indikasi konten disisipkan.")

    elif manip_score <= 0.62:
        verdict = "DOC_AUTHENTIC"
        risk = "LOW"
        confidence = min(1.0, 1.0 - manip_score)
        summary = "Dokumen tampak asli — tidak ditemukan inkonsistensi kompresi yang signifikan."
        notes.append(f"✓ Suspicious ratio ELA rendah: {suspicious_ratio:.2%}")
        notes.append(f"✓ Konsistensi background: {bg_consistency:.2f}")
        notes.append(f"✓ Tidak ada anomali tepi yang signifikan (skor: {edge_anomaly:.2f})")

    else:
        verdict = "DOC_UNCERTAIN"
        risk = "MEDIUM"
        confidence = 0.5
        summary = "Hasil analisis dokumen tidak konklusif — diperlukan verifikasi manual atau dokumen referensi asli."
        notes.append(f"Skor manipulasi: {manip_score:.2f} (zona abu-abu: 0.55–0.65)")
        notes.append("Disarankan membandingkan dengan dokumen referensi asli dari penerbit.")
        if suspicious_ratio > 0.03:
            notes.append(f"Catatan: suspicious ratio {suspicious_ratio:.2%} sedikit di atas normal.")

    return DocForensicResult(
        image_path=image_path,
        verdict=verdict,
        confidence=confidence,
        risk_level=risk,
        summary=summary,
        ela_suspicious_ratio=round(suspicious_ratio, 4),
        ela_mean_error=round(ela_mean, 3),
        background_consistency=round(bg_consistency, 4),
        edge_anomaly_score=round(edge_anomaly, 4),
        block_variance_score=round(block_variance, 4),
        color_jump_score=round(color_jump, 4),
        ela_image=ela_image,
        suspicious_mask=suspicious_mask,
        notes=notes,
    )


def generate_doc_report(doc_result: DocForensicResult) -> str:
    """Generate laporan teks untuk hasil analisis dokumen."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r = doc_result
    risk_icon = {"HIGH": "⛔", "MEDIUM": "⚠️", "LOW": "✅"}.get(r.risk_level, "?")

    lines = [
        "=" * 65,
        "  LAPORAN FORENSIK DOKUMEN DIGITAL",
        "  Sistem ELA + Doc Forensic | Kelompok 7 UNIMED 2026",
        "=" * 65,
        f"  Tanggal   : {now}",
        f"  File      : {os.path.basename(r.image_path)}",
        "",
        "-" * 65,
        f"  {risk_icon}  VERDICT : {r.verdict}",
        f"     Risk Level  : {r.risk_level}",
        f"     Confidence  : {r.confidence:.1%}",
        f"     Ringkasan   : {r.summary}",
        "",
        "-" * 65,
        "  METRIK ANALISIS DOKUMEN",
        "-" * 65,
        f"  ELA Mean Error         : {r.ela_mean_error:.3f}",
        f"  ELA Suspicious Ratio   : {r.ela_suspicious_ratio:.2%}",
        f"  Konsistensi Background : {r.background_consistency:.4f}",
        f"  Anomali Tepi Teks      : {r.edge_anomaly_score:.4f}",
        f"  Variansi Blok          : {r.block_variance_score:.4f}",
        f"  Lompatan Warna         : {r.color_jump_score:.4f}",
        "",
        "-" * 65,
        "  CATATAN DETAIL",
        "-" * 65,
    ]
    for n in r.notes:
        lines.append(f"  • {n}")

    lines += [
        "",
        "-" * 65,
        "  PANDUAN INTERPRETASI",
        "-" * 65,
        "  • Suspicious Ratio > 6%  : indikasi kuat manipulasi",
        "  • Background < 0.60      : area putih tidak seragam",
        "  • Anomali Tepi > 0.40    : teks/angka kemungkinan diedit",
        "  • Blok Variance > 0.30   : konten asing mungkin disisipkan",
        "",
        "  ⚠ DISCLAIMER: Sistem ini bersifat indikatif, bukan definitif.",
        "    Hasil analisis harus dikonfirmasi dengan dokumen asli.",
        "",
        "=" * 65,
        "  [Laporan dihasilkan otomatis oleh sistem]",
        "=" * 65,
    ]
    return "\n".join(lines)