"""
=============================================================================
MODUL FUSION — Integrasi Semua Hasil Analisis
=============================================================================
Menggabungkan tiga sumber analisis:
  1. ELA Result       → apakah ada manipulasi/editing?
  2. AI Detection     → apakah foto dibuat oleh AI?
  3. (Opsional) ML    → prediksi model neural network

Output akhir mencakup dua verdict terpisah yang jelas:
  - manipulation_verdict : MANIPULATED / AUTHENTIC / UNCERTAIN
  - ai_verdict           : AI_GENERATED / REAL_PHOTO / UNCERTAIN
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FullAnalysisResult:
    """Hasil lengkap gabungan semua modul analisis."""
    image_path: str

    # ELA
    ela_verdict: str
    ela_confidence: float
    ela_mean_error: float
    ela_suspicious_ratio: float

    # AI Detection
    ai_verdict: str
    ai_confidence: float
    ai_probability: float
    ai_scores: dict

    # Ringkasan akhir
    summary: str           # satu kalimat ringkasan
    overall_verdict: str   # 'AI_MANIPULATED'|'AI_AUTHENTIC'|'REAL_MANIPULATED'|'REAL_AUTHENTIC'|'UNCERTAIN'
    risk_level: str        # 'HIGH'|'MEDIUM'|'LOW'
    notes: list = field(default_factory=list)


def fuse_full_analysis(ela_result, ai_result) -> FullAnalysisResult:
    """
    Gabungkan hasil ELA dan AI Detection menjadi laporan lengkap.

    Empat kemungkinan kombinasi:
      AI_GENERATED + MANIPULATED → Foto AI yang sudah diedit lagi (sangat mencurigakan)
      AI_GENERATED + AUTHENTIC   → Murni foto AI (belum diedit)
      REAL_PHOTO   + MANIPULATED → Foto asli yang dimanipulasi (forgery klasik)
      REAL_PHOTO   + AUTHENTIC   → Foto asli yang bersih
    """
    ev = ela_result.verdict
    av = ai_result.verdict
    notes = []

    # Gabungkan catatan dari kedua modul
    notes.extend([f"[ELA] {n}" for n in ela_result.notes])
    notes.extend([f"[AI]  {n}" for n in ai_result.notes])

    # Tentukan overall verdict & risk level
    if av == "AI_GENERATED" and ev == "MANIPULATED":
        overall  = "AI_MANIPULATED"
        risk     = "HIGH"
        summary  = "Citra ini kemungkinan besar dibuat oleh AI dan telah mengalami manipulasi tambahan."

    elif av == "AI_GENERATED" and ev in ("AUTHENTIC", "UNCERTAIN"):
        overall  = "AI_GENERATED"
        risk     = "HIGH"
        summary  = "Citra ini terdeteksi sebagai hasil generasi model AI (bukan foto asli)."

    elif av == "REAL_PHOTO" and ev == "MANIPULATED":
        overall  = "REAL_MANIPULATED"
        risk     = "HIGH"
        summary  = "Foto asli yang telah mengalami manipulasi digital (editing, splicing, atau copy-move)."

    elif av == "REAL_PHOTO" and ev == "AUTHENTIC":
        overall  = "REAL_AUTHENTIC"
        risk     = "LOW"
        summary  = "Foto tampak asli dan tidak terdeteksi adanya manipulasi maupun generasi AI."

    else:
        overall  = "UNCERTAIN"
        risk     = "MEDIUM"
        summary  = "Hasil analisis tidak konklusif. Diperlukan pemeriksaan manual lebih lanjut."

    return FullAnalysisResult(
        image_path           = ela_result.image_path,
        ela_verdict          = ev,
        ela_confidence       = ela_result.confidence,
        ela_mean_error       = ela_result.mean_error,
        ela_suspicious_ratio = ela_result.suspicious_ratio,
        ai_verdict           = av,
        ai_confidence        = ai_result.confidence,
        ai_probability       = ai_result.ai_probability,
        ai_scores            = ai_result.scores,
        summary              = summary,
        overall_verdict      = overall,
        risk_level           = risk,
        notes                = notes,
    )
