"""
=============================================================================
MODUL VISUALISASI
=============================================================================
"""

import os, io, logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter
from typing import Optional

logger = logging.getLogger(__name__)

ELA_CMAP = LinearSegmentedColormap.from_list(
    "ela_heat", ["#0d0d0d", "#7b0000", "#d62728", "#ff7f0e", "#ffff00", "#ffffff"])

VERDICT_COLORS = {
    "MANIPULATED":   "#ef4444",
    "AUTHENTIC":     "#22c55e",
    "UNCERTAIN":     "#eab308",
    "AI_GENERATED":  "#a855f7",
    "REAL_PHOTO":    "#22c55e",
    "AI_MANIPULATED":  "#ef4444",
    "REAL_MANIPULATED": "#f97316",
    "AI_AUTHENTIC":    "#a855f7",
    "REAL_AUTHENTIC":  "#22c55e",
}


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _load_image_matching_ela(image_path: str, ela_array: np.ndarray) -> np.ndarray:
    """
    Load gambar asli dan resize agar ukurannya sama dengan ela_array.
    Ini diperlukan karena ela.py mungkin meresize gambar sebelum analisis.
    """
    img = Image.open(image_path).convert("RGB")
    ela_h, ela_w = ela_array.shape[:2]
    if img.size != (ela_w, ela_h):
        img = img.resize((ela_w, ela_h), Image.LANCZOS)
    return np.array(img)


def render_ela_panels(result, bg="#0a0e1a") -> bytes:
    """3 panel ELA: asli, ELA map, heatmap."""
    # Load original dengan ukuran yang matching ela_array
    original = Image.open(result.image_path).convert("RGB")
    ela_h, ela_w = result.ela_array.shape[:2]
    if original.size != (ela_w, ela_h):
        original = original.resize((ela_w, ela_h), Image.LANCZOS)

    ela_gray = result.ela_array.mean(axis=2)

    fig = plt.figure(figsize=(15, 5), facecolor=bg)
    gs  = gridspec.GridSpec(1, 3, wspace=0.05)
    items = [
        (original,         "Citra Asli", None),
        (result.ela_image, f"ELA Map  (quality={result.quality_used}%)", None),
        (None,             "Heatmap Intensitas Error", ELA_CMAP),
    ]
    for i, (data, title, cmap) in enumerate(items):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(bg); ax.axis("off")
        if cmap:
            im   = ax.imshow(ela_gray, cmap=cmap, vmin=0, vmax=255)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="#64748b")
            cbar.outline.set_edgecolor("#1e2d4a")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)
        else:
            ax.imshow(data)
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=8, fontfamily="monospace")

    plt.tight_layout()
    b = _fig_to_bytes(fig); plt.close(fig); return b


def render_mask_overlay(result, bg="#0a0e1a") -> bytes:
    """Overlay mask area mencurigakan."""
    # Gunakan helper untuk pastikan ukuran sama
    original  = _load_image_matching_ela(result.image_path, result.ela_array)
    ela_gray  = result.ela_array.mean(axis=2)

    # Pastikan ela_gray dan original punya ukuran yang sama
    if ela_gray.shape[:2] != original.shape[:2]:
        ela_h, ela_w = original.shape[:2]
        ela_img = Image.fromarray(ela_gray.astype(np.uint8))
        ela_img = ela_img.resize((ela_w, ela_h), Image.LANCZOS)
        ela_gray = np.array(ela_img, dtype=np.float32)

    mask      = ela_gray > (result.mean_error + 2 * result.std_error)
    overlay   = original.copy().astype(np.float32)
    overlay[mask, 0]  = 255
    overlay[mask, 1] *= 0.25
    overlay[mask, 2] *= 0.25
    blended   = (0.6 * np.clip(overlay, 0, 255).astype(np.uint8)
                 + 0.4 * original).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=bg)
    for ax, img, title, cmap in zip(
        axes,
        [original, mask, blended],
        ["Citra Asli", f"Mask Mencurigakan\n({result.suspicious_ratio:.2%} piksel)", "Overlay"],
        [None, "Reds", None],
    ):
        ax.set_facecolor(bg); ax.axis("off")
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="#94a3b8", fontsize=9, pad=8, fontfamily="monospace")

    plt.tight_layout()
    b = _fig_to_bytes(fig); plt.close(fig); return b


def render_multi_quality(image_path: str, qualities: list, bg="#0a0e1a") -> bytes:
    """Panel ELA di berbagai kualitas."""
    from ela import multi_quality_ela
    results = multi_quality_ela(image_path, qualities)
    fig, axes = plt.subplots(1, len(qualities), figsize=(5 * len(qualities), 5), facecolor=bg)
    if len(qualities) == 1:
        axes = [axes]
    for ax, q in zip(axes, qualities):
        r = results[q]
        ax.set_facecolor(bg); ax.imshow(r.ela_image); ax.axis("off")
        ax.set_title(f"Quality {q}%\nMean={r.mean_error:.1f}",
                     color="#94a3b8", fontsize=9, pad=6, fontfamily="monospace")
        c = VERDICT_COLORS.get(r.verdict, "#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor(c); spine.set_linewidth(2.5)
    plt.tight_layout()
    b = _fig_to_bytes(fig); plt.close(fig); return b


def render_ai_scores(ai_result, bg="#0a0e1a") -> bytes:
    """Bar chart skor fitur AI detection."""
    scores = ai_result.scores
    if "ml_raw_probability" in scores:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=bg)
        ax.set_facecolor(bg)
        prob = scores["ml_raw_probability"]
        ax.barh(["AI Probability"], [prob], color="#a855f7", height=0.4)
        ax.barh(["AI Probability"], [1 - prob], left=[prob], color="#22c55e", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_title("Probabilitas Model ML", color="#94a3b8", fontsize=11, fontfamily="monospace")
        ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1)
        ax.tick_params(colors="#94a3b8")
        plt.tight_layout()
        b = _fig_to_bytes(fig); plt.close(fig); return b

    labels = {
        "dct_smoothness":      "Noise Frekuensi DCT",
        "color_naturalness":   "Kealamian Warna",
        "noise_pattern":       "Pola Noise Sensor",
        "ela_consistency":     "Konsistensi ELA",
        "sharpness_variation": "Variasi Ketajaman",
    }
    keys   = list(scores.keys())
    vals   = [scores[k] for k in keys]
    colors = ["#22c55e" if v >= 0.5 else "#ef4444" for v in vals]
    names  = [labels.get(k, k) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg)
    ax.set_facecolor(bg)
    bars = ax.barh(names, vals, color=colors, height=0.55)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1, label="Batas (0.5)")
    ax.set_title("Skor Fitur AI Detection\n(Hijau = cenderung REAL, Merah = cenderung AI)",
                 color="#94a3b8", fontsize=10, fontfamily="monospace", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color="#94a3b8", fontsize=9, fontfamily="monospace")
    plt.tight_layout()
    b = _fig_to_bytes(fig); plt.close(fig); return b


def render_training_history(history, bg="#0a0e1a") -> bytes:
    """Grafik accuracy & loss training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=bg)
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        for s in ax.spines.values():
            s.set_edgecolor("#1e2d4a")

    ax1.plot(history.history["accuracy"],     color="#38bdf8", lw=2, label="Train")
    ax1.plot(history.history["val_accuracy"], color="#ef4444", lw=2, ls="--", label="Val")
    ax1.set_title("Accuracy", color="#94a3b8", fontfamily="monospace")
    ax1.legend(facecolor="#1e1e1e", labelcolor="#94a3b8", edgecolor="#1e2d4a")
    ax1.grid(color="#1e2d4a", ls="--", lw=0.5)

    ax2.plot(history.history["loss"],     color="#22c55e", lw=2, label="Train")
    ax2.plot(history.history["val_loss"], color="#f39c12", lw=2, ls="--", label="Val")
    ax2.set_title("Loss", color="#94a3b8", fontfamily="monospace")
    ax2.legend(facecolor="#1e1e1e", labelcolor="#94a3b8", edgecolor="#1e2d4a")
    ax2.grid(color="#1e2d4a", ls="--", lw=0.5)

    plt.tight_layout()
    b = _fig_to_bytes(fig); plt.close(fig); return b