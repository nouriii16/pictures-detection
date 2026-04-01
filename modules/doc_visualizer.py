"""
=============================================================================
MODUL DOC VISUALIZER — Visualisasi Hasil Analisis Dokumen
=============================================================================
Menghasilkan gambar visualisasi untuk hasil analisis forensik dokumen:
  1. Panel ELA dokumen (original + ELA map + heatmap)
  2. Overlay area mencurigakan pada dokumen
  3. Chart radar metrik forensik dokumen
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io


def render_doc_ela_panels(doc_result) -> bytes:
    """
    Render panel tiga kolom: original | ELA map | heatmap intensitas.
    Khusus untuk dokumen dengan skema warna berbeda dari foto biasa.
    """
    img_orig = Image.open(doc_result.image_path).convert("RGB")
    ela_img = doc_result.ela_image

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f172a")

    # ── Panel 1: Original ────────────────────────────────────────────────────
    axes[0].imshow(img_orig)
    axes[0].set_title("Dokumen Asli", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)
    axes[0].axis("off")

    # ── Panel 2: ELA Map ─────────────────────────────────────────────────────
    axes[1].imshow(ela_img)
    axes[1].set_title("ELA Map (Kualitas 95)", color="#fbbf24", fontsize=11, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Tambahkan anotasi
    axes[1].text(0.02, 0.02,
                 f"Mean: {doc_result.ela_mean_error:.1f}\nSuspicious: {doc_result.ela_suspicious_ratio:.2%}",
                 transform=axes[1].transAxes,
                 color="white", fontsize=8, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.8))

    # ── Panel 3: Heatmap ─────────────────────────────────────────────────────
    ela_arr = np.array(ela_img, dtype=np.float32).mean(axis=2)
    cmap = LinearSegmentedColormap.from_list("doc_heat",
           ["#0f172a", "#1e3a5f", "#0ea5e9", "#fbbf24", "#ef4444"])
    im = axes[2].imshow(ela_arr, cmap=cmap, vmin=0, vmax=80)
    axes[2].set_title("Intensitas Error", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Warna judul berdasarkan verdict
    verdict_colors = {
        "DOC_MANIPULATED": "#ef4444",
        "DOC_AUTHENTIC": "#22c55e",
        "DOC_UNCERTAIN": "#f59e0b",
    }
    title_color = verdict_colors.get(doc_result.verdict, "#94a3b8")
    fig.suptitle(f"Forensik Dokumen — {doc_result.verdict}  |  Confidence: {doc_result.confidence:.1%}",
                 color=title_color, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_doc_suspicious_overlay(doc_result) -> bytes:
    """
    Render overlay area mencurigakan di atas dokumen asli.
    Area merah = area dengan ELA tinggi (potensi manipulasi).
    """
    img_orig = Image.open(doc_result.image_path).convert("RGB")
    orig_arr = np.array(img_orig)

    # Resize mask ke ukuran gambar asli jika berbeda
    mask = doc_result.suspicious_mask
    if mask is None:
        mask = np.zeros((img_orig.height, img_orig.width), dtype=np.uint8)

    if mask.shape != (img_orig.height, img_orig.width):
        mask_img = Image.fromarray(mask).resize(img_orig.size, Image.NEAREST)
        mask = np.array(mask_img)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")

    # ── Panel kiri: Original ─────────────────────────────────────────────────
    axes[0].imshow(img_orig)
    axes[0].set_title("Dokumen Asli", color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # ── Panel kanan: Overlay ─────────────────────────────────────────────────
    overlay = orig_arr.copy()
    suspicious_px = mask > 128
    # Warnai area mencurigakan dengan merah semi-transparan
    overlay[suspicious_px, 0] = np.clip(overlay[suspicious_px, 0].astype(int) * 0.3 + 180, 0, 255)
    overlay[suspicious_px, 1] = np.clip(overlay[suspicious_px, 1].astype(int) * 0.3, 0, 255)
    overlay[suspicious_px, 2] = np.clip(overlay[suspicious_px, 2].astype(int) * 0.3, 0, 255)

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Area Mencurigakan ({doc_result.ela_suspicious_ratio:.2%} piksel)",
        color="#fca5a5", fontsize=11, fontweight="bold", pad=8
    )
    axes[1].axis("off")

    # Legend
    red_patch = mpatches.Patch(color="#ef4444", label="Area mencurigakan (ELA tinggi)")
    axes[1].legend(handles=[red_patch], loc="lower right",
                   facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="white", fontsize=9)

    # Tambahkan info verdict
    verdict_colors = {
        "DOC_MANIPULATED": "#ef4444",
        "DOC_AUTHENTIC": "#22c55e",
        "DOC_UNCERTAIN": "#f59e0b",
    }
    verdict_color = verdict_colors.get(doc_result.verdict, "#94a3b8")
    fig.suptitle(
        f"Overlay Forensik — {doc_result.verdict}",
        color=verdict_color, fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_doc_metrics_chart(doc_result) -> bytes:
    """
    Render chart radar/bar untuk 5 metrik forensik dokumen.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")

    # ── Panel kiri: Bar chart metrik ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1e293b")

    labels = [
        "ELA Suspicious\nRatio",
        "Background\nInkonsistensi",
        "Anomali Tepi\nTeks",
        "Variansi\nBlok",
        "Lompatan\nWarna",
    ]
    values = [
        min(1.0, doc_result.ela_suspicious_ratio / 0.06),  # normalize ke 0-1
        1.0 - doc_result.background_consistency,
        doc_result.edge_anomaly_score,
        doc_result.block_variance_score,
        doc_result.color_jump_score,
    ]
    thresholds = [1.0, 0.4, 0.4, 0.3, 0.5]  # batas mencurigakan

    colors = []
    for v, t in zip(values, thresholds):
        if v >= t:
            colors.append("#ef4444")      # merah = mencurigakan
        elif v >= t * 0.6:
            colors.append("#f59e0b")      # oranye = waspada
        else:
            colors.append("#22c55e")      # hijau = aman

    bars = ax.barh(labels, values, color=colors, height=0.5, alpha=0.85)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Skor (0 = aman, 1 = mencurigakan)", color="#94a3b8", fontsize=9)
    ax.set_title("Metrik Forensik Dokumen", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Tambahkan nilai di ujung bar
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 1.05), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", ha="left", color="white", fontsize=8)

    # Garis threshold
    ax.axvline(x=0.5, color="#475569", linestyle="--", linewidth=1, alpha=0.7, label="Batas waspada")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8)

    # ── Panel kanan: Ringkasan verdict ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    ax2.axis("off")

    verdict_config = {
        "DOC_MANIPULATED": {"emoji": "⛔", "color": "#ef4444", "bg": "#450a0a", "label": "DOKUMEN DIMANIPULASI"},
        "DOC_AUTHENTIC":   {"emoji": "✅", "color": "#22c55e", "bg": "#052e16", "label": "DOKUMEN ASLI"},
        "DOC_UNCERTAIN":   {"emoji": "⚠️", "color": "#f59e0b", "bg": "#451a03", "label": "TIDAK KONKLUSIF"},
    }
    cfg = verdict_config.get(doc_result.verdict,
                              {"emoji": "❓", "color": "#94a3b8", "bg": "#1e293b", "label": doc_result.verdict})

    # Kotak verdict utama
    ax2.add_patch(plt.Rectangle((0.05, 0.55), 0.9, 0.35,
                                 facecolor=cfg["bg"], edgecolor=cfg["color"],
                                 linewidth=2, transform=ax2.transAxes))
    ax2.text(0.5, 0.78, cfg["emoji"] + "  " + cfg["label"],
             ha="center", va="center", fontsize=14, fontweight="bold",
             color=cfg["color"], transform=ax2.transAxes)
    ax2.text(0.5, 0.63, f"Confidence: {doc_result.confidence:.1%}",
             ha="center", va="center", fontsize=10,
             color="#94a3b8", transform=ax2.transAxes)

    # Detail metrik teks
    detail_lines = [
        f"ELA Suspicious  : {doc_result.ela_suspicious_ratio:.2%}",
        f"ELA Mean Error  : {doc_result.ela_mean_error:.2f}",
        f"Bg Consistency  : {doc_result.background_consistency:.3f}",
        f"Edge Anomaly    : {doc_result.edge_anomaly_score:.3f}",
        f"Block Variance  : {doc_result.block_variance_score:.3f}",
    ]
    ax2.text(0.5, 0.45, "\n".join(detail_lines),
             ha="center", va="top", fontsize=9,
             color="#cbd5e1", transform=ax2.transAxes,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", alpha=0.8))

    # Summary
    summary_wrapped = doc_result.summary[:80] + ("..." if len(doc_result.summary) > 80 else "")
    ax2.text(0.5, 0.10, summary_wrapped,
             ha="center", va="bottom", fontsize=8,
             color="#64748b", transform=ax2.transAxes,
             style="italic", wrap=True)

    ax2.set_title("Ringkasan Hasil", color="#e2e8f0", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
