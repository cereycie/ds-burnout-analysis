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

### Supporting (4 dataset pendukung)

`sleep_insomnia_clean.csv`, `stress_indicators_clean.csv`, `study_behavior_clean.csv`, `sleep_quality_clean.csv`

### Catatan Penting Kolom

- `student_lifestyle_clean.csv` menggunakan `Title_Case` (contoh: `Study_Hours_Per_Day`, `Stress_Level`)
- `Stress_Level` bertipe string ordinal: `'Low'`, `'Moderate'`, `'High'` (bukan numerik)
- `mbi_questions_clean.csv` tidak memiliki kolom `dim_*`; semua dimensi dihitung on-the-fly dari item EX/CY/EF
- `curhat_nlp_clean.csv` tidak memiliki `word_count` atau `stress_keyword_count`; keduanya dihitung dari `text_curhat`

---

## Struktur Folder

```
hapi_project/
├── dashboard/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── clean/
│   │   ├── model_ready/              <- output notebook 01-04
│   │   └── supporting/               <- output notebook 05-07
│   ├── preprocessed/
│   │   ├── burnout_fatigue_preproc...
│   │   ├── curhat_nlp_preprocessed...
│   │   ├── mbi_questions_preproces...
│   │   └── student_lifestyle_preproc...
│   └── raw/
│       ├── model_ready/              <- 4 file CSV kotor
│       └── supporting/               <- 4 file CSV kotor
├── notebook/
│   ├── model ready wrangling - p.../
│   │   ├── 01_mbi_questions_wrangling.ipynb
│   │   ├── 02_nlp_curhat_wrangling.ipynb
│   │   ├── 03_student1m_wrangling.ipynb
│   │   └── 04_study_habit_wrangling.ipynb
│   ├── supporting wrangling - EDA/
│   │   ├── 05_stress_indicators_wrangling.ipynb
│   │   ├── 06_study_behavior_wrangling.ipynb
│   │   └── 07_insomnia_wrangling_eda.ipynb
│   ├── 08_eda_model_ready.ipynb
│   ├── 09_eda_explanatory.ipynb
│   └── 10_ab_testing.ipynb
├── .gitattributes
├── README.md
└── url.txt
```

---

## Alur Pipeline

```
data/raw/
   |
   v
notebook/model ready wrangling - p.../  (NB 01-04)
   |
   v
data/clean/model_ready/  +  data/clean/supporting/
   |
   v
notebook/supporting wrangling - EDA/  (NB 05-07)
   |
   v
notebook/08_eda_model_ready.ipynb
   |
   v
notebook/09_eda_explanatory.ipynb
   |
   v
notebook/10_ab_testing.ipynb
   |
   v
dashboard/app.py
```

**Urutan eksekusi wajib:** NB 01-04 harus selesai dijalankan sebelum NB 08-10 dan dashboard, karena file clean CSV dihasilkan di tahap wrangling.

---

## Notebook

### NB 01-04: Data Wrangling (`notebook/model ready wrangling - p.../`)

Membersihkan dan menstandarkan 4 dataset model-ready dari kondisi kotor: duplikat, tipe data salah, nilai hilang, dan inkonsistensi format.

| File | Dataset |
|---|---|
| `01_mbi_questions_wrangling.ipynb` | MBI Questions |
| `02_nlp_curhat_wrangling.ipynb` | Curhat NLP |
| `03_student1m_wrangling.ipynb` | Burnout Fatigue (1 juta baris) |
| `04_study_habit_wrangling.ipynb` | Student Lifestyle |

### NB 05-07: EDA Supporting (`notebook/supporting wrangling - EDA/`)

EDA pada dataset pendukung yang mencakup indikator stres, kebiasaan belajar, dan pola tidur/insomnia.

| File | Dataset |
|---|---|
| `05_stress_indicators_wrangling.ipynb` | Stress Indicators |
| `06_study_behavior_wrangling.ipynb` | Study Behavior |
| `07_insomnia_wrangling_eda.ipynb` | Sleep Insomnia |

### NB 08: EDA 4 Dataset Model-Ready

File: `notebook/08_eda_model_ready.ipynb`

EDA bertahap pada 4 dataset utama dengan kerangka:

- **Data Profiling** - shape, dtypes, missing values, duplikat, statistik deskriptif
- **Univariat** - histogram satu warna solid, boxplot deteksi outlier, bar distribusi target
- **Bivariat** - scatter plot + trendline, boxplot per kelompok
- **Multivariat** - heatmap korelasi (RdBu_r), uji VIF multikolinearitas, scatter 3 variabel double-encoding

### NB 09: EDA Eksplanatif (15 Business Questions)

File: `notebook/09_eda_explanatory.ipynb`

| Kategori | BQ | Pertanyaan |
|---|---|---|
| Deskriptif | BQ1-5 | Distribusi burnout per demografi, profil High Risk, distribusi emosi curhat, pola tidur, dominasi dimensi MBI |
| Diagnostik | BQ6-10 | Pengaruh jam belajar, tidur, stres, aktivitas fisik, dan kata kunci emosi terhadap burnout |
| Prediktif | BQ11-13 | Fitur pembeda stress level, diskriminasi risk level oleh MBI, pola multivariate sinyal burnout |
| Preskriptif | BQ14-15 | Pengaruh social support terhadap burnout, profil mahasiswa berisiko dan rekomendasi intervensi |

### NB 10: A/B Testing

File: `notebook/10_ab_testing.ipynb`

Menguji apakah intervensi personal (berbasis fatigue score) lebih efektif dari intervensi generik dalam meningkatkan penggunaan fitur wellness.

**Desain:**

| Parameter | Nilai |
|---|---|
| Grup A | Intervensi generik (notifikasi standar), n = 1.000 |
| Grup B | Intervensi personal (berdasarkan profil fatigue), n = 1.000 |
| Metrik | Conversion rate fitur wellness |
| H0 | P_A = P_B |
| H1 | P_A != P_B (two-sided) |
| Alpha | 0.05 |

**Analisis yang dilakukan:**

- Simulasi data binomial dengan `np.random.seed(42)`
- Z-test proporsi + Chi-square sebagai validasi silang
- 95% Confidence Interval untuk selisih proporsi
- Cohen's h untuk effect size
- Analisis sensitivitas alpha (0.01, 0.05, 0.10)
- Segmentasi per tahun akademik (Freshman, Sophomore, Senior)

---

## Dashboard Streamlit

File: `dashboard/app.py`

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

# 3. Jalankan notebook 01-04 terlebih dahulu
# Buka Jupyter, masuk ke notebook/model ready wrangling - p.../
# Jalankan dari NB01 sampai NB04 secara berurutan

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

Python: **3.10+**

---

## Prinsip Visualisasi

Seluruh visualisasi dalam notebook dan dashboard mengikuti aturan berikut:

- **Palet warna**: 9 hex biru sekuensial saja (`#00bcff` sampai `#0034c8`), ditambah putih dan abu-abu. Tidak ada warna di luar palet.
- **Darker-is-more**: nilai atau kategori yang lebih tinggi selalu mendapat warna lebih gelap.
- **Histogram**: satu warna solid `#007ff1` dengan `edgecolor='white'`, tidak menggunakan gradasi.
- **Scatter**: `alpha=0.3-0.4`, `color=#007ff1` untuk menangani overplotting.
- **Heatmap korelasi**: `cmap='RdBu_r'` dengan `linewidths=1.5, linecolor='white'`.
- **Urutan ordinal**: menggunakan urutan manual (`RISK_ORDER`, `STRESS_ORDER_LS`), bukan `sorted()`.
- `sns.despine()` diterapkan di semua grafik.
- Judul grafik dibuat singkat; keterangan panjang ditaruh di `st.caption()`.

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
