"""
=============================================================================
MODUL DOC FORENSIC — Analisis Forensik Khusus Dokumen Digital
=============================================================================
Mendeteksi manipulasi pada:
  - Screenshot bukti transfer / mobile banking
  - Foto KTP, ijazah, sertifikat
  - Struk, invoice, dokumen resmi
  - Scan dokumen

PIPELINE khusus dokumen (berbeda dari pipeline foto biasa):
  1. ELA dengan kualitas tinggi (95) — screenshot belum pernah dikompresi
  2. Analisis konsistensi background — area putih harus seragam
  3. Analisis anomali tepi teks — teks yang diedit punya artefak tepi
  4. Analisis blok — deteksi area dengan inkonsistensi kompresi lokal
  5. Analisis gradien warna — perubahan warna yang tidak wajar di area teks

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
    verdict: str                    # 'DOC_MANIPULATED' | 'DOC_AUTHENTIC' | 'DOC_UNCERTAIN'
    confidence: float               # 0.0 – 1.0
    risk_level: str                 # 'HIGH' | 'MEDIUM' | 'LOW'
    summary: str

    # Metrik detail
    ela_suspicious_ratio: float
    ela_mean_error: float
    background_consistency: float   # 1.0 = sempurna konsisten, 0.0 = sangat tidak konsisten
    edge_anomaly_score: float       # tinggi = ada anomali tepi (indikasi edit teks)
    block_variance_score: float     # tinggi = ada blok dengan kompresi berbeda
    color_jump_score: float         # tinggi = ada lompatan warna tidak wajar

    ela_image: Image.Image = None
    suspicious_mask: np.ndarray = None
    notes: List[str] = field(default_factory=list)


def _compute_ela_document(img: Image.Image, quality: int = 95) -> Tuple[Image.Image, np.ndarray]:
    """
    ELA khusus dokumen menggunakan kualitas tinggi (95).
    Screenshot/scan belum pernah dikompresi JPEG, sehingga
    kualitas 95 memberikan error map yang lebih sensitif.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela = ImageChops.difference(img, recompressed)
    ela_arr = np.array(ela, dtype=np.float32)

    # Amplifikasi lebih tinggi untuk dokumen (15 → 25)
    # karena error map dokumen cenderung lebih halus
    extrema = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    ela_enhanced = ImageEnhance.Brightness(ela).enhance((255.0 / max_diff) * 25)

    return ela_enhanced, ela_arr


def _analyze_background_consistency(img_arr: np.ndarray, ela_arr: np.ndarray) -> float:
    """
    Analisis konsistensi area background (area putih/terang).
    Area yang seharusnya polos/bersih tidak boleh punya ELA error tinggi.
    Return: skor konsistensi 0.0–1.0 (1.0 = sangat konsisten = tidak dimanipulasi)
    """
    # Identifikasi area background (piksel sangat terang)
    bg_mask = np.all(img_arr > 215, axis=2)

    if bg_mask.sum() < 100:
        return 0.5  # tidak cukup area background untuk dianalisis

    # ELA di area background harus sangat rendah
    ela_bg = ela_arr[bg_mask]
    bg_mean = float(np.mean(ela_bg))
    bg_std = float(np.std(ela_bg))

    # Jika ada piksel background dengan ELA sangat tinggi → manipulasi
    anomaly_ratio = float(np.sum(ela_bg > bg_mean + 3 * bg_std) / len(ela_bg))

    # Skor: semakin rendah anomaly, semakin tinggi konsistensi
    consistency = max(0.0, 1.0 - anomaly_ratio * 10 - bg_mean / 30)
    return min(1.0, consistency)


def _analyze_text_edge_anomaly(ela_arr: np.ndarray) -> float:
    """
    Analisis anomali di sekitar area teks.
    Teks yang diedit (angka diubah, nama diganti) meninggalkan
    artefak di batas antara teks baru dan background.

    Return: skor anomali 0.0–1.0 (tinggi = anomali terdeteksi)
    """
    # Konversi ELA ke grayscale
    ela_gray = ela_arr.mean(axis=2)

    # Deteksi area dengan ELA sangat tinggi (kandidat area teks/edit)
    threshold = np.mean(ela_gray) + 1.5 * np.std(ela_gray)
    high_ela = (ela_gray > threshold).astype(np.float32)

    # Analisis apakah area ELA tinggi terlokalisir (edit spot) atau tersebar (kompresi alami)
    # Edit spot biasanya kecil dan terisolir
    from scipy.ndimage import label as scipy_label
    try:
        labeled, num_features = scipy_label(high_ela)
        if num_features == 0:
            return 0.0

        # Hitung ukuran setiap komponen
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        total_area = ela_gray.size

        # Banyak komponen kecil terisolir = lebih mencurigakan
        small_components = sum(1 for s in sizes if s < total_area * 0.001)
        large_components = sum(1 for s in sizes if s > total_area * 0.005)

        # Score berdasarkan rasio small/total
        if num_features == 0:
            return 0.0
        isolation_score = min(1.0, small_components / (num_features + 1))

        # Gabungkan dengan rata-rata ELA di area tinggi
        mean_high = float(np.mean(ela_gray[high_ela > 0.5])) if high_ela.sum() > 0 else 0
        intensity_score = min(1.0, mean_high / 80)

        return min(1.0, isolation_score * 0.6 + intensity_score * 0.4)

    except ImportError:
        # Fallback tanpa scipy
        ela_std = float(np.std(ela_gray))
        ela_max = float(np.max(ela_gray))
        return min(1.0, (ela_max / 255) * 0.5 + (ela_std / 50) * 0.5)


def _analyze_block_variance(ela_arr: np.ndarray, block_size: int = 16) -> float:
    """
    Analisis variansi antar blok ELA.
    Dokumen asli punya ELA yang seragam antar blok.
    Area yang diedit punya blok dengan ELA jauh berbeda dari sekitarnya.

    Return: skor 0.0–1.0 (tinggi = ada blok anomali = mencurigakan)
    """
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

    # Blok yang nilainya jauh di atas rata-rata = mencurigakan
    outlier_threshold = overall_mean + 2.5 * overall_std
    outlier_ratio = float(np.sum(block_means > outlier_threshold) / len(block_means))

    return min(1.0, outlier_ratio * 5)


def _analyze_color_jumps(img_arr: np.ndarray) -> float:
    """
    Deteksi lompatan warna tidak wajar di area yang seharusnya gradual.
    Angka/teks yang diganti sering meninggalkan inkonsistensi warna
    antara objek baru dengan area sekitarnya.

    Return: skor 0.0–1.0 (tinggi = ada lompatan warna mencurigakan)
    """
    gray = img_arr.mean(axis=2).astype(np.float32)

    # Gradien horizontal dan vertikal
    grad_h = np.abs(np.diff(gray, axis=1))
    grad_v = np.abs(np.diff(gray, axis=0))

    # Gradient sangat tinggi (lebih dari 100 per piksel) di area yang tidak seharusnya tajam
    extreme_h = float(np.sum(grad_h > 100) / grad_h.size)
    extreme_v = float(np.sum(grad_v > 100) / grad_v.size)

    # Untuk dokumen, batas teks memang tajam — tapi lompatan ekstrem di
    # area teks internal (bukan di batas karakter) mencurigakan
    color_jump = (extreme_h + extreme_v) / 2

    # Normalkan: dokumen normal punya ~5-10% piksel dengan gradient tinggi
    normalized = max(0.0, color_jump - 0.05) / 0.15
    return min(1.0, float(normalized))


def analyze_document(image_path: str, quality: int = 95) -> DocForensicResult:
    """
    Analisis forensik lengkap untuk dokumen/screenshot.

    Args:
        image_path: Path ke file gambar
        quality: Kualitas JPEG untuk ELA (default 95 untuk dokumen)

    Returns:
        DocForensicResult dengan verdict dan metrik detail
    """
    img = Image.open(image_path).convert("RGB")

    MAX_SIZE = 1500  # maksimal 1500px di sisi terpanjang
    if max(img.size) > MAX_SIZE:
        ratio = MAX_SIZE / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"Gambar diresize dari {img.size} ke {new_size}")
        
    arr = np.array(img, dtype=np.float32)
    notes = []

    logger.info(f"Analisis dokumen: {os.path.basename(image_path)}, ukuran: {img.size}")

    # ── 1. ELA Dokumen ───────────────────────────────────────────────────────
    ela_image, ela_arr = _compute_ela_document(img, quality)
    ela_gray = ela_arr.mean(axis=2)

    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))
    threshold_px = ela_mean + 2 * ela_std
    suspicious_ratio = float(np.sum(ela_gray > threshold_px) / ela_gray.size)

    # Buat mask area mencurigakan
    suspicious_mask = (ela_gray > threshold_px).astype(np.uint8) * 255

    # ── 2. Konsistensi Background ─────────────────────────────────────────────
    bg_consistency = _analyze_background_consistency(arr, ela_arr)

    # ── 3. Anomali Tepi Teks ──────────────────────────────────────────────────
    edge_anomaly = _analyze_text_edge_anomaly(ela_arr)

    # ── 4. Variansi Blok ─────────────────────────────────────────────────────
    block_variance = _analyze_block_variance(ela_arr)

    # ── 5. Lompatan Warna ────────────────────────────────────────────────────
    color_jump = _analyze_color_jumps(arr)

    # ── SCORING GABUNGAN ─────────────────────────────────────────────────────
    # Bobot: ELA(35%) + Background(25%) + Edge(20%) + Block(15%) + Color(5%)
    manip_score = (
    suspicious_ratio / 0.06 * 0.35 +
    (1 - bg_consistency) * 0.30 +
    edge_anomaly * 0.12 +
    block_variance * 0.18 +
    color_jump * 0.05
    )
    manip_score = min(1.0, float(manip_score))

    # ── VERDICT ──────────────────────────────────────────────────────────────
    if manip_score >= 0.55:
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

    elif manip_score <= 0.52:
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
        notes.append(f"Skor manipulasi: {manip_score:.2f} (zona abu-abu: 0.25–0.55)")
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
