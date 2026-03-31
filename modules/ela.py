"""
=============================================================================
MODUL ELA — Error Level Analysis
=============================================================================
Mendeteksi manipulasi citra (copy-move, splicing, retouching) berbasis
inkonsistensi artefak kompresi JPEG.

Formula: ELA(x, y) = |I_original(x,y) - I_recompressed(x,y)|

Verdict logic:
  MANIPULATED : suspicious_ratio tinggi DAN mean_error tinggi
  AUTHENTIC   : suspicious_ratio rendah (< 2%) — tidak peduli mean_error
  UNCERTAIN   : kondisi lainnya
=============================================================================
"""

import os, io, logging
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from dataclasses import dataclass, field
from typing import Tuple, List

logger = logging.getLogger(__name__)

DEFAULT_QUALITY      = 90
AMPLIFY_FACTOR       = 15
THRESHOLD_SUSPICIOUS = 0.08   # rasio piksel mencurigakan
THRESHOLD_MEAN_HIGH  = 180.0  # mean error tinggi (foto kamera modern bisa 100-170)
THRESHOLD_MEAN_LOW   = 175.0  # batas bawah untuk AUTHENTIC


@dataclass
class ELAResult:
    image_path: str
    ela_image: Image.Image
    ela_array: np.ndarray
    mean_error: float
    std_error: float
    max_error: float
    suspicious_ratio: float
    quality_used: int
    verdict: str = "UNCERTAIN"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


def compute_ela(image_path: str, quality: int = DEFAULT_QUALITY,
                amplify: int = AMPLIFY_FACTOR) -> Tuple[Image.Image, np.ndarray]:
    """Hitung error map ELA dari sebuah citra."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela_image = ImageChops.difference(original, recompressed)
    extrema   = ela_image.getextrema()
    max_diff  = max(ex[1] for ex in extrema) or 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance((255.0 / max_diff) * amplify)
    return ela_image, np.array(ela_image, dtype=np.float32)


def analyze_ela(image_path: str, quality: int = DEFAULT_QUALITY,
                amplify: int = AMPLIFY_FACTOR) -> ELAResult:
    """Analisis ELA lengkap + verdict otomatis."""
    ela_image, ela_array = compute_ela(image_path, quality, amplify)

    mean_error       = float(np.mean(ela_array))
    std_error        = float(np.std(ela_array))
    max_error        = float(np.max(ela_array))
    threshold_px     = mean_error + 2 * std_error
    suspicious_ratio = float(np.sum(ela_array > threshold_px) / ela_array.size)

    notes = []

    # MANIPULATED: harus suspicious TINGGI dan mean error TINGGI (keduanya)
    if suspicious_ratio > THRESHOLD_SUSPICIOUS and mean_error > THRESHOLD_MEAN_HIGH:
        verdict    = "MANIPULATED"
        confidence = min(1.0,
                         suspicious_ratio / THRESHOLD_SUSPICIOUS * 0.5 +
                         (mean_error / THRESHOLD_MEAN_HIGH) * 0.5)
        notes.append(f"Piksel mencurigakan: {suspicious_ratio:.2%}")
        notes.append(f"Mean error tinggi ({mean_error:.2f}) — inkonsistensi kompresi.")

    # AUTHENTIC: suspicious RENDAH (tidak peduli mean error)
    elif suspicious_ratio < 0.02:
        verdict    = "AUTHENTIC"
        confidence = min(1.0, (1 - suspicious_ratio / 0.02) * 0.8 + 0.2)
        notes.append(f"Suspicious ratio sangat rendah ({suspicious_ratio:.2%}) — tidak ada manipulasi.")
        if mean_error > 50:
            notes.append(f"Mean error tinggi ({mean_error:.2f}) akibat kompresi kamera, bukan manipulasi.")

    # UNCERTAIN: suspicious sedang atau kondisi ambigu
    else:
        verdict    = "UNCERTAIN"
        confidence = 0.5
        notes.append(f"Suspicious ratio {suspicious_ratio:.2%} — tidak konklusif, disarankan analisis lanjutan.")

    confidence = max(0.0, min(1.0, confidence))

    return ELAResult(
        image_path      = image_path,
        ela_image       = ela_image,
        ela_array       = ela_array,
        mean_error      = mean_error,
        std_error       = std_error,
        max_error       = max_error,
        suspicious_ratio= suspicious_ratio,
        quality_used    = quality,
        verdict         = verdict,
        confidence      = confidence,
        notes           = notes,
    )


def multi_quality_ela(image_path: str, qualities=(70, 80, 90, 95)) -> dict:
    """Jalankan ELA di berbagai kualitas sekaligus."""
    return {q: analyze_ela(image_path, quality=q) for q in qualities}


def extract_ela_features(image_path: str, target_size=(128, 128),
                         quality: int = DEFAULT_QUALITY) -> np.ndarray:
    """Ekstrak fitur ELA siap input ML — shape (H, W, 3), float32 [0,1]."""
    ela_image, _ = compute_ela(image_path, quality)
    return np.array(ela_image.resize(target_size, Image.LANCZOS), dtype=np.float32) / 255.0
