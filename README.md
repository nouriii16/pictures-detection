# Sistem Deteksi Manipulasi Citra Digital
## Menggunakan Error Level Analysis (ELA) dan Machine Learning

**Kelompok 7 — Program Studi Ilmu Komputer, FMIPA**
**Universitas Negeri Medan, 2026**
Dosen Pengampu: Dr. Hermawan Syahputra, S.Si., M.Si.

---

## Dua Fitur Utama Sistem

### 🔬 1. Deteksi Manipulasi (ELA)
Mendeteksi apakah foto telah diedit secara digital:
copy-move, splicing, retouching, penggantian objek, dll.
Menggunakan Error Level Analysis (ELA) berbasis inkonsistensi artefak kompresi JPEG.

### 🤖 2. Deteksi Foto AI-Generated
Mendeteksi apakah gambar dibuat oleh model AI generatif
seperti Midjourney, DALL-E, Stable Diffusion, GAN, dll.
Menggunakan analisis statistik fitur: DCT smoothness, noise sensor,
kealamian warna, konsistensi ELA, dan variasi ketajaman.

---

## Struktur Proyek

```
ela-detection-system-kelompok7/
├── app.py               ← Web App Streamlit (JALANKAN INI)
├── demo.py              ← Demo CLI tanpa dataset
├── requirements.txt
├── README.md
├── modules/
│   ├── ela.py           ← Modul ELA (deteksi manipulasi)
│   ├── ai_detector.py   ← Modul AI Detection (deteksi foto AI)
│   ├── ml_model.py      ← Model CNN & MobileNetV2
│   ├── fusion.py        ← Penggabungan hasil analisis
│   ├── visualizer.py    ← Semua visualisasi
│   └── report.py        ← Generator laporan
├── data/
│   ├── authentic/       ← Foto asli (untuk training ELA model)
│   ├── tampered/        ← Foto manipulasi (untuk training ELA model)
│   ├── real_photos/     ← Foto dari kamera (untuk training AI detector)
│   └── ai_generated/    ← Foto AI (untuk training AI detector)
└── output/
    ├── ela_maps/
    ├── reports/
    └── models/
```

---

## Instalasi & Menjalankan

```bash
# 1. Install dependensi inti (cukup untuk web app)
pip install Pillow numpy matplotlib streamlit

# 2. Jalankan web app
streamlit run app.py
# Buka browser: http://localhost:8501

# 3. (Opsional) Install TensorFlow untuk training model ML
pip install tensorflow scikit-learn
```

---

## Cara Penggunaan Web App

1. Jalankan `streamlit run app.py`
2. Upload gambar yang ingin dianalisis
3. Klik **Analisis Sekarang**
4. Lihat hasil di dua kolom:
   - **ELA** — apakah ada manipulasi/editing?
   - **AI Detection** — apakah dibuat oleh AI?
5. Jelajahi tab detail: ELA Map, Area Mencurigakan, Multi-Kualitas, AI Scores
6. Download laporan TXT atau gambar visualisasi

---

## Output / Verdict Sistem

| Verdict | Artinya |
|---------|---------|
| `REAL_AUTHENTIC` | Foto asli dari kamera, tidak ada manipulasi |
| `REAL_MANIPULATED` | Foto asli yang telah diedit/dimanipulasi |
| `AI_GENERATED` | Foto dibuat murni oleh AI (belum diedit) |
| `AI_MANIPULATED` | Foto AI yang masih diedit lagi |
| `UNCERTAIN` | Tidak konklusif, perlu pemeriksaan manual |

---

## Dataset Rekomendasi

### Untuk training ELA (deteksi manipulasi):
- **CASIA v2.0** — [github.com/namtpham/casia2groundtruth](https://github.com/namtpham/casia2groundtruth)
- **Columbia Splicing Dataset**

### Untuk training AI Detector:
- **CIFAKE** — Real and AI-Generated Image Dataset (Kaggle)
- **ArtiFact** — AI-generated image dataset
- **GenImage** — dataset untuk deteksi foto AI

---

## Referensi

- Baihaqi et al. (2025). *Classification Using ELA and MobileNetV2*. Jurnal Masyarakat Informatika.
- Sudianto & Anwar (2024). *Image Forensics Using ELA and Block Matching*. Mobile and Forensics.
- Bisri & Marzuki (2023). *Forensik Citra Digital Menggunakan ELA*. G-Tech Journal.
- More et al. (2025). *Enhancing Image Forgery Detection with CNN and ELA*. JISEM.
- Hamza et al. (2025). *An Adaptive Compression Factor ELA for Image Forgery*. Science World Journal.
