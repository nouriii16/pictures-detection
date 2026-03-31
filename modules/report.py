"""
=============================================================================
MODUL REPORT — Generator Laporan
=============================================================================
"""

import os, csv, json, logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

RISK_ICONS = {"HIGH": "⛔", "MEDIUM": "⚠️", "LOW": "✅"}


def generate_full_report(full_result, output_path: Optional[str] = None) -> str:
    """Laporan teks lengkap dari FullAnalysisResult."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r   = full_result
    icon = RISK_ICONS.get(r.risk_level, "?")

    lines = [
        "=" * 65,
        "  LAPORAN ANALISIS FORENSIK CITRA DIGITAL",
        "  Sistem ELA + AI Detection | Kelompok 7 UNIMED 2026",
        "=" * 65,
        f"  Tanggal   : {now}",
        f"  File      : {os.path.basename(r.image_path)}",
        "",
        "-" * 65,
        f"  {icon}  VERDICT AKHIR : {r.overall_verdict}",
        f"     Risk Level  : {r.risk_level}",
        f"     Ringkasan   : {r.summary}",
        "",
        "-" * 65,
        "  [1] ANALISIS ELA (Deteksi Manipulasi/Editing)",
        "-" * 65,
        f"  Verdict      : {r.ela_verdict}",
        f"  Confidence   : {r.ela_confidence:.1%}",
        f"  Mean Error   : {r.ela_mean_error:.4f}",
        f"  Suspicious   : {r.ela_suspicious_ratio:.2%}",
        "",
        "-" * 65,
        "  [2] ANALISIS AI DETECTION (Deteksi Foto AI-Generated)",
        "-" * 65,
        f"  Verdict      : {r.ai_verdict}",
        f"  Confidence   : {r.ai_confidence:.1%}",
        f"  AI Prob.     : {r.ai_probability:.1%}",
    ]

    if r.ai_scores:
        lines.append("  Skor fitur :")
        for k, v in r.ai_scores.items():
            lines.append(f"    {k:<25} : {v:.4f}")

    lines += [
        "",
        "-" * 65,
        "  CATATAN DETAIL",
        "-" * 65,
    ]
    for n in r.notes:
        lines.append(f"  • {n}")

    lines += [
        "",
        "=" * 65,
        "  [Laporan dihasilkan otomatis oleh sistem]",
        "=" * 65,
    ]

    report = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report


def generate_csv_report(results: List, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fields = ["filename", "overall_verdict", "risk_level",
              "ela_verdict", "ela_confidence", "ela_mean_error", "ela_suspicious_ratio",
              "ai_verdict", "ai_confidence", "ai_probability", "summary"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "filename": os.path.basename(r.image_path),
                "overall_verdict": r.overall_verdict,
                "risk_level": r.risk_level,
                "ela_verdict": r.ela_verdict,
                "ela_confidence": f"{r.ela_confidence:.4f}",
                "ela_mean_error": f"{r.ela_mean_error:.4f}",
                "ela_suspicious_ratio": f"{r.ela_suspicious_ratio:.4f}",
                "ai_verdict": r.ai_verdict,
                "ai_confidence": f"{r.ai_confidence:.4f}",
                "ai_probability": f"{r.ai_probability:.4f}",
                "summary": r.summary,
            })
    logger.info(f"CSV disimpan: {output_path}")
