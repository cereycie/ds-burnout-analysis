# HAPI: Human Activity Pattern Intelligence

Platform analisis burnout dan fatigue pada mahasiswa berbasis data perilaku harian, instrumen psikometri MBI-SS, analisis sentimen teks curhat, dan A/B testing intervensi. Dibangun sebagai proyek capstone untuk **Coding Camp 2026 powered by DBS Foundation**.

---

## Tim

**CC26-PSU348**

| Nama | ID |
|---|---|
| Greycia Febrina Michelle | CDCC700D6X2644 |
| Khazel Hayfa Yosmi | CDCC308D6X0629 |

---

## Latar Belakang

Burnout akademik pada mahasiswa adalah masalah yang sering tidak terdeteksi hingga sudah parah. Kebanyakan platform kesehatan mental yang ada bersifat generik dan tidak mempertimbangkan pola perilaku individual. HAPI mencoba menjawab ini dengan pendekatan berbasis data: menggabungkan data belajar, tidur, aktivitas fisik, skor MBI, dan ekspresi emosi dari teks curhatan untuk menghasilkan profil fatigue yang personal.

---

## Dataset

### Model-Ready (4 dataset utama)

| Dataset | Baris | Kolom Fitur Utama | Target |
|---|---|---|---|
| `burnout_fatigue_clean.csv` | ~1.000.000 | `study_hours_per_day`, `sleep_hours`, `exam_pressure`, `physical_activity`, `social_support`, `screen_time`, `stress_level`, `academic_performance` | `burnout_score`, `risk_level` |
| `student_lifestyle_clean.csv` | ~2.000 | `Study_Hours_Per_Day`, `Sleep_Hours_Per_Day`, `Physical_Activity_Hours_Per_Day`, `Extracurricular_Hours_Per_Day`, `Social_Hours_Per_Day` | `Stress_Level` |
| `mbi_questions_clean.csv` | ~2.000 | `EX1-EX5`, `CY1-CY4`, `EF1-EF6` | `fatigue_score`, `risk_level` |
| `curhat_nlp_clean.csv` | ~10.000 | `text_curhat` | `emotion` |

Catatan: kolom `Stress_Level` pada dataset lifestyle bertipe string ordinal (`'Low'`, `'Moderate'`, `'High'`). Kolom `dim_exhaustion`, `dim_cynicism`, `dim_efficacy_inv` tidak ada di file clean dan dihitung secara on-the-fly dari item EX/CY/EF.

### Supporting (4 dataset pendukung)

`sleep_insomnia_clean.csv`, `stress_indicators_clean.csv`, `study_behavior_clean.csv`, `sleep_quality_clean.csv`

---

## Struktur Folder

```
hapi_project/
├── data/
│   ├── raw/
│   │   ├── model_ready/          <- 4 file CSV kotor (input wrangling)
│   │   └── supporting/           <- 4 file CSV kotor (input wrangling)
│   └── clean/
│       ├── model_ready/          <- output notebook 01-04
│       └── supporting/           <- output notebook 05-07
├── notebooks/
│   ├── 01_k5_mbi_questions_wrangling.ipynb
│   ├── 02_k3_nlp_curhat_wrangling.ipynb
│   ├── 03_k1_student1m_wrangling.ipynb
│   └── 04_k2_study_habit_wrangling.ipynb
├── notebooks_supporting/
│   ├── 05_support_k1_stress_indicators.ipynb
│   ├── 06_support_k2_study_behavior.ipynb
│   └── 07_support_k3_sleep_insomnia.ipynb
├── nb08/
│   └── 08_eda_model_ready.ipynb
├── nb09/
│   └── 09_eda_explanatory_analysis.ipynb
├── nb10/
│   └── 10_ab_testing.ipynb
├── dashboard/
│   ├── app.py
│   └── requirements.txt
└── docs/
    ├── data_dictionary.md
    ├── fatigue_score_methodology.md
    └── ai_engineer_guide.md
```

---

## Alur Pipeline

```
Raw Data
   |
   v
NB 01-04: Data Wrangling & Cleaning
   |
   v
NB 05-07: EDA Supporting Datasets
   |
   v
NB 08: EDA 4 Dataset Model-Ready
   |
   v
NB 09: EDA Eksplanatif (15 Business Questions)
   |
   v
NB 10: A/B Testing
   |
   v
Streamlit Dashboard
```

**Urutan eksekusi wajib:** NB 01-04 harus dijalankan sebelum NB 08-10 dan dashboard, karena file clean CSV dihasilkan di tahap wrangling.

---

## Notebook

### NB 01-04: Data Wrangling

Membersihkan dan menstandarkan 4 dataset model-ready dari kondisi kotor (duplikat, tipe data salah, nilai hilang, inkonsistensi format).

### NB 05-07: EDA Supporting

EDA pada dataset pendukung yang mencakup indikator stres, kebiasaan belajar, dan pola tidur/insomnia.

### NB 08: EDA 4 Dataset Model-Ready

EDA bertahap pada 4 dataset utama mengikuti kerangka:

- **Data Profiling** - shape, dtypes, missing values, duplikat, statistik deskriptif
- **Univariat** - histogram (satu warna solid), boxplot deteksi outlier, bar distribusi target
- **Bivariat** - scatter plot + trendline, boxplot per kelompok
- **Multivariat** - heatmap korelasi (RdBu_r), uji VIF multikolinearitas, scatter 3 variabel double-encoding

### NB 09: EDA Eksplanatif (15 Business Questions)

Menjawab 15 pertanyaan bisnis yang dibagi ke dalam 4 kategori:

| Kategori | BQ | Pertanyaan |
|---|---|---|
| Deskriptif | BQ1-5 | Distribusi burnout per demografi, profil High Risk, distribusi emosi curhat, pola tidur, dominasi dimensi MBI |
| Diagnostik | BQ6-10 | Pengaruh jam belajar, tidur, stres, aktivitas fisik, dan kata kunci emosi terhadap burnout |
| Prediktif | BQ11-13 | Fitur pembeda stress level, diskriminasi risk level oleh MBI, pola multivariate sinyal burnout |
| Preskriptif | BQ14-15 | Pengaruh social support terhadap burnout, profil mahasiswa berisiko dan rekomendasi intervensi |

### NB 10: A/B Testing

Menguji apakah intervensi personal (berbasis fatigue score) lebih efektif dari intervensi generik dalam meningkatkan penggunaan fitur wellness.

**Desain:**
- Grup A: intervensi generik (notifikasi standar), n = 1.000
- Grup B: intervensi personal (berdasarkan profil fatigue), n = 1.000
- Metrik: conversion rate fitur wellness
- H0: P_A = P_B
- H1: P_A != P_B (two-sided)
- Alpha: 0.05

**Analisis yang dilakukan:**
- Simulasi data binomial (`np.random.seed(42)`)
- Z-test proporsi + Chi-square validasi silang
- 95% Confidence Interval untuk selisih proporsi
- Cohen's h (effect size)
- Analisis sensitivitas alpha (0.01, 0.05, 0.10)
- Segmentasi per tahun akademik

---

## Dashboard Streamlit

5 halaman utama:

| Halaman | Isi |
|---|---|
| Overview & KPI | Angka ringkasan, distribusi risk level, donut chart proporsi, burnout per tahun akademik |
| 15 Business Questions | Filter per kategori, grafik interaktif untuk setiap BQ |
| EDA Dataset | 4 tab (Burnout / Lifestyle / MBI / NLP), profiling, histogram, scatter, heatmap, VIF |
| A/B Testing | Desain eksperimen, hasil Z-test, CI plot, distribusi Z, segmentasi, keputusan bisnis |
| Rekomendasi & Self-Check | 5 kartu intervensi + kuis MBI interaktif untuk estimasi fatigue score personal |

### Cara Menjalankan

```bash
# 1. Clone repo
git clone https://github.com/<username>/hapi.git
cd hapi

# 2. Install dependencies
pip install -r dashboard/requirements.txt

# 3. Jalankan notebook 01-04 terlebih dahulu untuk menghasilkan data clean
# (buka di Jupyter dan run all)

# 4. Jalankan dashboard dari root folder
streamlit run dashboard/app.py
```

---

## Tech Stack

| Kategori | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisasi | `matplotlib`, `seaborn` |
| Statistik | `scipy`, `statsmodels` |
| Dashboard | `streamlit` |
| Notebook | `jupyter` |

Versi Python yang digunakan: **3.10+**

---

## Prinsip Visualisasi

Seluruh visualisasi dalam notebook dan dashboard mengikuti prinsip berikut:

- **Palet warna**: 9 hex biru sekuensial saja (`#00bcff` sampai `#0034c8`), ditambah putih dan abu-abu. Tidak ada warna di luar palet.
- **Darker-is-more**: nilai atau kategori yang lebih tinggi selalu mendapat warna lebih gelap.
- **Histogram**: satu warna solid `#007ff1` dengan `edgecolor='white'`, tidak menggunakan gradasi.
- **Scatter**: `alpha=0.3-0.4`, `color=C_MID` untuk menangani overplotting.
- **Heatmap korelasi**: `cmap='RdBu_r'` dengan `linewidths=1.5, linecolor='white'`.
- **Urutan ordinal**: menggunakan urutan manual (`RISK_ORDER`, `STRESS_ORDER_LS`), bukan `sorted()`.
- **`sns.despine()`** diterapkan di semua grafik.
- Judul grafik dibuat singkat; keterangan panjang ditaruh di `st.caption()`.

---

## Konvensi Nama Kolom

Beberapa hal penting yang perlu diperhatikan saat bekerja dengan dataset ini:

- `burnout_fatigue_clean.csv` menggunakan `snake_case` (contoh: `sleep_hours`, bukan `sleep_hours_per_day`)
- `student_lifestyle_clean.csv` menggunakan `Title_Case` (contoh: `Study_Hours_Per_Day`, `Stress_Level`)
- `mbi_questions_clean.csv` tidak memiliki kolom `dim_*`; semua dimensi dihitung on-the-fly
- `curhat_nlp_clean.csv` tidak memiliki `word_count` atau `stress_keyword_count`; keduanya dihitung dari `text_curhat`

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
```

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik dalam program **Coding Camp 2026 powered by DBS Foundation**. Data yang digunakan adalah data sintetis yang dibuat khusus untuk proyek ini.
