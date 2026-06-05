# HAPI Dashboard — Panduan Deploy

## Struktur File
```
hapi_project/
├── data/
│   └── clean/
│       ├── model_ready/
│       │   ├── burnout_fatigue_clean.csv
│       │   ├── student_lifestyle_clean.csv
│       │   ├── mbi_questions_clean.csv
│       │   └── curhat_nlp_clean.csv
│       └── supporting/
│           ├── sleep_insomnia_clean.csv
│           └── stress_indicators_clean.csv
├── dashboard/
│   ├── app.py          ← file utama
│   └── requirements.txt
└── notebooks/          ← jalankan 01-04 dulu sebelum dashboard
```

## Cara Menjalankan Lokal
```bash
# 1. Install dependencies
pip install -r dashboard/requirements.txt

# 2. Pastikan data sudah ada (jalankan notebook 01-04 dulu)

# 3. Jalankan dari root folder proyek
streamlit run dashboard/app.py
```

## Catatan Penting
- Jalankan notebook 01, 02, 03, 04 terlebih dahulu untuk menghasilkan file clean CSV
- Dashboard membaca data langsung dari file CSV — tidak perlu file gambar terpisah
- Semua grafik di-generate live saat dashboard diakses (tidak pakai file .png)
- Jika dataset belum ada, halaman akan menampilkan pesan informatif

## 5 Halaman Dashboard
1. **Overview & KPI** — angka ringkasan + distribusi risk level
2. **15 Business Questions** — filter per kategori, grafik interaktif
3. **EDA Dataset** — 4 tab (Burnout / Lifestyle / MBI / NLP)
4. **A/B Testing** — Z-test, CI plot, segmentasi, keputusan bisnis
5. **Rekomendasi** — 5 intervensi + profil mahasiswa berisiko
