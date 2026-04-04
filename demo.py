"""
=============================================================================
DEMO.PY — Demo Cepat (tanpa dataset eksternal)
=============================================================================
Jalankan: python demo.py
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))
import matplotlib; matplotlib.use("Agg")

import numpy as np
from PIL import Image, ImageDraw

from ela         import analyze_ela, extract_ela_features
from ai_detector import analyze_ai_statistical
from fusion      import fuse_full_analysis
from report      import generate_full_report

OUT = "output/demo"
os.makedirs(OUT, exist_ok=True)


def make_photo(path="demo_photo.jpg"):
    """Simulasi foto asli: gradien alami + noise sensor."""
    W, H = 480, 360
    arr  = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        arr[y] = [int(60 + y/H*80), int(100 + y/H*90), int(180 - y/H*60)]
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 180, 160], fill=(220, 80, 60), outline=(160, 30, 10), width=2)
    draw.ellipse([220, 60, 400, 240], fill=(80, 200, 160))
    arr2  = np.array(img, dtype=np.float32)
    arr2 += np.random.normal(0, 6, arr2.shape)  # noise sensor kamera
    Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=93)
    print(f"  [+] Foto asli sintetis    → {path}")
    return path


def make_tampered(src, path="demo_tampered.jpg"):
    """Simulasi foto asli yang dimanipulasi."""
    arr = np.array(Image.open(src).convert("RGB"))
    arr[100:220, 180:340] = arr[10:130, 10:170].copy()  # copy-move
    arr[260:310,  30:180] = [255, 60, 60]                # splicing
    Image.fromarray(arr).save(path, "JPEG", quality=82)
    print(f"  [+] Foto manipulasi       → {path}")
    return path


def make_ai_image(path="demo_ai.png"):
    """Simulasi foto AI: terlalu halus, sempurna, noise minimal."""
    W, H = 480, 360
    arr  = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        arr[y] = [int(120 + t*100), int(80 + t*60), int(200 - t*80)]
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 60, 380, 300], fill=(255, 200, 150))
    draw.ellipse([150, 100, 330, 260], fill=(240, 180, 120))
    # Hampir zero noise — ciri AI
    arr2  = np.array(img, dtype=np.float32)
    arr2 += np.random.normal(0, 0.3, arr2.shape)
    Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8)).save(path, "PNG")
    print(f"  [+] Gambar AI sintetis    → {path}")
    return path


def run_demo():
    print("\n" + "="*62)
    print("  DEMO — Sistem Deteksi Manipulasi Citra Digital")
    print("  ELA + AI Detection ")
    print("="*62)

    p_photo   = make_photo()
    p_tamper  = make_tampered(p_photo)
    p_ai      = make_ai_image()

    for label, path in [("Foto Asli", p_photo), ("Foto Dimanipulasi", p_tamper), ("Foto AI", p_ai)]:
        print(f"\n--- {label} ({os.path.basename(path)}) ---")
        ela = analyze_ela(path, quality=90)
        ai  = analyze_ai_statistical(path)
        full = fuse_full_analysis(ela, ai)

        print(f"  ELA Verdict    : {ela.verdict} ({ela.confidence:.1%})")
        print(f"  AI  Verdict    : {ai.verdict}  ({ai.confidence:.1%})")
        print(f"  Overall        : {full.overall_verdict}")
        print(f"  Risk Level     : {full.risk_level}")
        print(f"  Summary        : {full.summary}")

        base = os.path.splitext(os.path.basename(path))[0]
        generate_full_report(full, f"{OUT}/report_{base}.txt")

    print(f"\n{'='*62}")
    print(f"  Selesai! Laporan tersimpan di: {OUT}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run_demo()
