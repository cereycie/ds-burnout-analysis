import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HAPI Dashboard",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE   = ['#00bcff','#00adfd','#009efa','#008ff6','#007ff1',
             '#006eea','#005ce1','#004ad6','#0034c8']
PALETTE_R = PALETTE[::-1]
C_DARK  = '#0034c8'
C_MID   = '#007ff1'
C_LIGHT = '#00bcff'
C_GREY  = '#d0d8e4'
C_WHITE = '#ffffff'
HEATMAP_DIV = 'RdBu_r'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.20,
    'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

np.random.seed(42)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001a6e 0%, #0034c8 55%, #006eea 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px; padding: 4px 0; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

.metric-card {
    background: white;
    border: 1px solid #e8ecf4;
    border-radius: 10px;
    padding: 16px 18px;
    border-left: 4px solid #0034c8;
    box-shadow: 0 2px 6px rgba(0,52,200,0.06);
    margin-bottom: 2px;
}
.metric-value { font-size: 26px; font-weight: 600; color: #0034c8; line-height: 1; margin-bottom: 4px; }
.metric-label { font-size: 11px; color: #7a8299; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
.metric-delta { font-size: 11px; color: #1D9E75; margin-top: 4px; font-weight: 500; }

.section-hdr {
    font-size: 18px; font-weight: 600; color: #0d1b4b;
    padding-bottom: 6px; border-bottom: 2px solid #0034c8; margin-bottom: 14px;
}
.sub-hdr { font-size: 14px; font-weight: 600; color: #0034c8; margin: 14px 0 6px 0; }

.bq-badge { display: inline-block; font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 20px; margin-right: 4px; }
.badge-desc  { background: #E1F5EE; color: #085041; }
.badge-diag  { background: #EEEDFE; color: #3C3489; }
.badge-pred  { background: #FAEEDA; color: #633806; }
.badge-pres  { background: #FAECE7; color: #712B13; }

.reko-card {
    background: white; border: 1px solid #e8ecf4; border-radius: 10px;
    padding: 14px 16px; border-top: 3px solid #0034c8; height: 100%;
}
.reko-num { font-size: 22px; font-weight: 700; color: #0034c8; opacity: 0.25; }
.reko-ttl { font-size: 13px; font-weight: 600; color: #0d1b4b; margin: 4px 0; }
.reko-dsc { font-size: 11px; color: #7a8299; line-height: 1.5; }
.reko-src { font-size: 10px; font-weight: 600; color: #0034c8; margin-top: 4px; }

.ab-stat { background:#f5f7fd; border:1px solid #e0e5f2; border-radius:8px;
    padding:12px 8px; text-align:center; }
.ab-val { font-size: 20px; font-weight: 600; color: #0034c8; }
.ab-lbl { font-size: 10px; color: #7a8299; margin-top:2px; }

.info-box { background:#f5f7fd; border-left:4px solid #0034c8;
    border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:16px; font-size:13px; }

.self-check { background:white; border:1px solid #e8ecf4; border-radius:10px;
    padding:20px; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
for _ in range(6):
    if (ROOT / 'data').exists(): break
    ROOT = ROOT.parent

@st.cache_data
def load_all():
    MR   = ROOT / 'data' / 'clean' / 'model_ready'
    SUPP = ROOT / 'data' / 'clean' / 'supporting'
    def sl(p): return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return {
        'burnout'   : sl(MR   / 'burnout_fatigue_clean.csv'),
        'lifestyle' : sl(MR   / 'student_lifestyle_clean.csv'),
        'mbi'       : sl(MR   / 'mbi_questions_clean.csv'),
        'nlp'       : sl(MR   / 'curhat_nlp_clean.csv'),
        'sleep'     : sl(SUPP / 'sleep_insomnia_clean.csv'),
        'stress_ind': sl(SUPP / 'stress_indicators_clean.csv'),
    }

DATA = load_all()
df_b = DATA['burnout']
df_l = DATA['lifestyle']
df_m = DATA['mbi']
df_n = DATA['nlp']

if not df_l.empty and 'Stress_Level' in df_l.columns:
    STRESS_ENCODE = {'Low':1,'Moderate':2,'High':3}
    df_l = df_l.copy()
    df_l['Stress_Level_num'] = df_l['Stress_Level'].map(STRESS_ENCODE)

if not df_n.empty and 'text_curhat' in df_n.columns:
    df_n = df_n.copy()
    df_n['word_count']  = df_n['text_curhat'].fillna('').apply(lambda x: len(str(x).split()))
    df_n['char_count']  = df_n['text_curhat'].fillna('').apply(lambda x: len(str(x)))
    STRESS_KW = ['capek','lelah','stres','burnout','deadline','revisi','skripsi',
                 'takut','cemas','panik','nangis','down','menyerah','buntu',
                 'overthinking','gelisah','kewalahan','frustasi','tertekan']
    df_n['stress_keyword_count'] = df_n['text_curhat'].fillna('').apply(
        lambda x: sum(kw in str(x).lower() for kw in STRESS_KW))

RISK_ORDER    = ['Low','Medium','High']
STRESS_LS_ORD = ['Low','Moderate','High']
EMOTION_ORDER = ['Sadness','Anger','Fear','Neutral','Joy']
BURNOUT_FEAT  = [c for c in ['study_hours_per_day','sleep_hours','exam_pressure',
                              'physical_activity','social_support','screen_time',
                              'stress_level','academic_performance'] if c in df_b.columns]
LIFESTYLE_FEAT = [c for c in ['Study_Hours_Per_Day','Sleep_Hours_Per_Day',
                               'Physical_Activity_Hours_Per_Day',
                               'Extracurricular_Hours_Per_Day',
                               'Social_Hours_Per_Day'] if c in df_l.columns]
SHORT_LS = {'Study_Hours_Per_Day':'Study','Sleep_Hours_Per_Day':'Sleep',
            'Physical_Activity_Hours_Per_Day':'Physical',
            'Extracurricular_Hours_Per_Day':'Extra','Social_Hours_Per_Day':'Social',
            'Stress_Level_num':'Stress Level'}
MBI_EX = [f'EX{i}' for i in range(1,6) if f'EX{i}' in df_m.columns]
MBI_CY = [f'CY{i}' for i in range(1,5) if f'CY{i}' in df_m.columns]
MBI_EF = [f'EF{i}' for i in range(1,7) if f'EF{i}' in df_m.columns]

def despine(ax): sns.despine(ax=ax); return ax

def bar_pal(n, reverse=False):
    pal = PALETTE_R if reverse else PALETTE
    return [pal[int(i*(len(pal)-1)/max(n-1,1))] for i in range(n)]

def annotate_bars(ax, bars, fmt='{:.0f}', offset_factor=0.02, fontsize=8):
    ymax = max(b.get_height() for b in bars)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+ymax*offset_factor,
                    fmt.format(h), ha='center', va='bottom', fontsize=fontsize)

def annotate_bars_pct(ax, bars, total, fontsize=8):
    ymax = max(b.get_height() for b in bars)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+ymax*0.02,
                f'{h/total*100:.1f}%', ha='center', va='bottom', fontsize=fontsize)

def fig_show(fig, caption='', key=None):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if caption:
        st.caption(caption)

def shdr(t): st.markdown(f'<div class="section-hdr">{t}</div>', unsafe_allow_html=True)
def sshdr(t): st.markdown(f'<div class="sub-hdr">{t}</div>', unsafe_allow_html=True)

def mcard(col, val, lbl, delta=None):
    d = f'<div class="metric-delta">{delta}</div>' if delta else ''
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div>{d}</div>', unsafe_allow_html=True)

def info_box(txt):
    st.markdown(f'<div class="info-box">{txt}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## HAPI")
    st.markdown("**Human Activity Pattern Intelligence**")
    st.markdown("*Tim CC26-PSU348 | DBS Foundation*")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["Overview & KPI",
         "15 Business Questions",
         "EDA Dataset",
         "A/B Testing",
         "Rekomendasi & Self-Check"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Tim DS:**")
    st.markdown("Greycia Febrina Michelle")
    st.markdown("Khazel Hayfa Yosmi")
    st.markdown("---")
    st.caption("CC26-PSU348")


if page == "Overview & KPI":
    st.markdown("""
    <div style="background:linear-gradient(120deg,#0034c8 0%,#007ff1 60%,#00bcff 100%);
         border-radius:14px;padding:24px 28px;margin-bottom:20px;color:white;">
      <div style="font-size:26px;font-weight:600;margin-bottom:4px;">
        HAPI - Human Activity Pattern Intelligence</div>
      <div style="font-size:13px;opacity:0.88;line-height:1.6;">
        Platform diagnosis mandiri burnout dan fatigue berbasis data perilaku mahasiswa.<br>
        Mengintegrasikan 8 dataset · 15 Business Questions · A/B Testing intervensi.</div>
    </div>""", unsafe_allow_html=True)

    shdr("Key Performance Indicators")
    c1,c2,c3,c4,c5 = st.columns(5)
    n_tot    = f"{len(df_b):,}"   if not df_b.empty else "N/A"
    p_high   = f"{(df_b['risk_level']=='High').mean()*100:.1f}%" if not df_b.empty and 'risk_level'  in df_b.columns else "N/A"
    p_sleep  = f"{(df_b['sleep_hours']<6).mean()*100:.1f}%"     if not df_b.empty and 'sleep_hours'  in df_b.columns else "N/A"
    avg_burn = f"{df_b['burnout_score'].mean():.3f}"             if not df_b.empty and 'burnout_score' in df_b.columns else "N/A"
    mcard(c1, n_tot,    "Total Responden")
    mcard(c2, p_high,   "Mahasiswa High Risk",  "▲ Perlu perhatian")
    mcard(c3, p_sleep,  "Tidur < 6 Jam",       "▲ Zona berisiko")
    mcard(c4, avg_burn, "Burnout Score Rata-rata")
    mcard(c5, "8",      "Dataset Terintegrasi")
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sshdr("Distribusi Risk Level")
        if not df_b.empty and 'risk_level' in df_b.columns:
            ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
            rc = df_b['risk_level'].value_counts().reindex(ra)
            n  = len(rc)
            clrs = [PALETTE[int(i*(len(PALETTE)-1)/max(n-1,1))] for i in range(n)]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(rc.index, rc.values, color=clrs, edgecolor='white', linewidth=0.8, width=0.5)
            annotate_bars_pct(ax, bars, len(df_b))
            ax.set_ylim(0, rc.max()*1.22)
            ax.set_ylabel('Jumlah Mahasiswa')
            ax.set_title('Distribusi Risk Level Burnout')
            despine(ax); fig.tight_layout()
            fig_show(fig, "Low (terang #00bcff) → High (gelap #0034c8) | warna = urutan ordinal")
        else:
            st.info("Jalankan notebook 03 untuk menghasilkan data.")

    with col_b:
        sshdr("Burnout Score per Tahun Akademik")
        if not df_b.empty and 'academic_year' in df_b.columns and 'burnout_score' in df_b.columns:
            yr = df_b.groupby('academic_year')['burnout_score'].mean().sort_index()
            ranks = yr.rank(method='min').astype(int) - 1
            n = len(yr)
            clrs = [PALETTE[int(r*(len(PALETTE)-1)/max(n-1,1))] for r in ranks]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(yr.index.astype(str), yr.values, color=clrs,
                          edgecolor='white', linewidth=0.8, width=0.5)
            for bar, v in zip(bars, yr.values):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
                        f'{v:.3f}', ha='center', fontsize=8)
            ax.set_ylim(0, yr.max()*1.15)
            ax.set_xlabel('Tahun Akademik'); ax.set_ylabel('Rata-rata Burnout Score')
            ax.set_title('Rata-rata Burnout per Tahun')
            despine(ax); fig.tight_layout()
            fig_show(fig, "Warna berdasarkan ranking nilai burnout: gelap = burnout lebih tinggi")

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        sshdr("Proporsi Risk Level (Donut)")
        if not df_b.empty and 'risk_level' in df_b.columns:
            ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
            rc = df_b['risk_level'].value_counts().reindex(ra)
            pie_clrs = [C_LIGHT, C_MID, C_DARK][:len(ra)]
            fig, ax = plt.subplots(figsize=(5, 4))
            wedges, texts, autotexts = ax.pie(
                rc.values, labels=rc.index, autopct='%1.1f%%',
                colors=pie_clrs, startangle=90,
                wedgeprops={'width':0.55, 'edgecolor':'white', 'linewidth':2},
                pctdistance=0.75, labeldistance=1.08
            )
            for at in autotexts: at.set_fontsize(9); at.set_color('white'); at.set_fontweight('bold')
            ax.set_title('Proporsi Risk Level')
            fig.tight_layout()
            fig_show(fig, "Donut chart: kontras lompatan warna (#00bcff, #007ff1, #0034c8)")

    with col_d:
        sshdr("Distribusi Emosi Curhat (Ringkasan)")
        if not df_n.empty and 'emotion' in df_n.columns:
            em_ord = [e for e in EMOTION_ORDER if e in df_n['emotion'].unique()]
            em_ord += [e for e in df_n['emotion'].unique() if e not in em_ord]
            em_c = df_n['emotion'].value_counts().reindex(em_ord)
            n = len(em_c)
            clrs = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(em_c.index[::-1], em_c.values[::-1],
                           color=list(reversed(clrs)), edgecolor='white', linewidth=0.8)
            for bar in bars:
                ax.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2,
                        f'{int(bar.get_width()):,}', va='center', fontsize=8)
            ax.set_xlabel('Jumlah Teks')
            ax.set_title('Distribusi Emosi Curhat')
            ax.set_xlim(0, em_c.max()*1.18)
            despine(ax); fig.tight_layout()
            fig_show(fig, "Sadness (gelap) = emosi terberat | Joy (terang) = paling ringan")

    st.markdown("<br>", unsafe_allow_html=True)
    shdr("Tentang Proyek HAPI")
    c1,c2,c3 = st.columns(3)
    c1.info("**Tema:** Healthy Lives & Well-being\n\n"
            "**Target:** Mahasiswa S1 yang mengalami burnout akademik")
    c2.info("**Model AI:** Fatigue Score Fusion - mengintegrasikan MBI, NLP curhat, "
            "perilaku belajar, dan pola tidur")
    c3.info("**Output:** Skor kelelahan personal, rekomendasi intervensi adaptif, "
            "mood tracker harian")


elif page == "15 Business Questions":
    shdr("15 Business Questions - Analisis Eksplanatif")

    BQ_CAT = {
        "Semua BQ": list(range(1,16)),
        "Deskriptif (BQ 1–5)": list(range(1,6)),
        "Diagnostik (BQ 6–10)": list(range(6,11)),
        "Prediktif & Preskriptif (BQ 11–15)": list(range(11,16)),
    }
    BADGE = {"Deskriptif":"badge-desc","Diagnostik":"badge-diag",
             "Prediktif":"badge-pred","Preskriptif":"badge-pres"}

    BQ_INFO = {
        1: ("Deskriptif","Bagaimana distribusi tingkat burnout mahasiswa berdasarkan jenis kelamin dan tahun akademik?"),
        2: ("Deskriptif","Berapa persentase mahasiswa High Risk dan bagaimana profil perilaku hariannya?"),
        3: ("Deskriptif","Bagaimana distribusi emosi dominan dari teks curhatan mahasiswa?"),
        4: ("Deskriptif","Bagaimana pola jam tidur mahasiswa dan berapa yang tidur < 6 jam?"),
        5: ("Deskriptif","Dimensi MBI mana yang memiliki skor rata-rata tertinggi?"),
        6: ("Diagnostik","Apakah belajar > 8 jam/hari berkorelasi signifikan dengan burnout score?"),
        7: ("Diagnostik","Apakah tidur < 6 jam berhubungan dengan burnout dan kelelahan lebih tinggi?"),
        8: ("Diagnostik","Indikator stres akademik mana yang paling kuat berkorelasi dengan burnout?"),
        9: ("Diagnostik","Apakah mahasiswa yang aktif fisik dan sosial menunjukkan burnout lebih rendah?"),
        10:("Diagnostik","Apakah teks Sadness/Anger mengandung lebih banyak kata kunci stres?"),
        11:("Prediktif","Kombinasi fitur perilaku apa yang paling membedakan mahasiswa stress level tinggi vs rendah?"),
        12:("Prediktif","Apakah fatigue score MBI dapat membedakan kelompok risiko secara konsisten?"),
        13:("Prediktif","Adakah pola multivariate konsisten antara jam belajar, tidur, dan tekanan akademik?"),
        14:("Preskriptif","Seberapa besar perbedaan burnout antara mahasiswa social support tinggi vs rendah?"),
        15:("Preskriptif","Profil mahasiswa paling berisiko dan rekomendasi intervensi berdasarkan data?"),
    }

    cat_sel = st.selectbox("Pilih kategori:", list(BQ_CAT.keys()))
    bq_list = BQ_CAT[cat_sel]

    for bq_num in bq_list:
        kat, pertanyaan = BQ_INFO[bq_num]
        b_cls = BADGE[kat]
        with st.expander(f"BQ{bq_num} - {pertanyaan}",
                         expanded=(bq_num == bq_list[0])):
            st.markdown(f'<span class="bq-badge {b_cls}">{kat}</span>', unsafe_allow_html=True)

            if bq_num == 1 and not df_b.empty:
                col_l, col_r = st.columns(2)
                with col_l:
                    if 'gender' in df_b.columns and 'risk_level' in df_b.columns:
                        ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
                        pv = df_b.groupby(['gender','risk_level']).size().unstack(fill_value=0)
                        pv = pv[[c for c in ra if c in pv.columns]]
                        n  = len(pv.columns)
                        clrs = [PALETTE[int(i*(len(PALETTE)-1)/max(n-1,1))] for i in range(n)]
                        fig, ax = plt.subplots(figsize=(5.5, 3.5))
                        pv.plot(kind='bar', ax=ax, color=clrs, edgecolor='white', linewidth=0.5, width=0.65)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                        ax.set_xlabel('Gender'); ax.set_ylabel('Jumlah')
                        ax.set_title('Risk Level per Gender')
                        ax.legend(title='Risk', fontsize=8, title_fontsize=8)
                        despine(ax); fig.tight_layout()
                        fig_show(fig, "Clustered bar: distribusi risk level berdasarkan jenis kelamin")
                with col_r:
                    if 'academic_year' in df_b.columns and 'burnout_score' in df_b.columns:
                        yr = df_b.groupby('academic_year')['burnout_score'].mean().sort_index()
                        n  = len(yr)
                        clrs = [PALETTE[int(r*(len(PALETTE)-1)/max(n-1,1))] for r in (yr.rank(method='min').astype(int)-1)]
                        fig, ax = plt.subplots(figsize=(5.5, 3.5))
                        bars = ax.bar(yr.index.astype(str), yr.values, color=clrs,
                                      edgecolor='white', linewidth=0.8, width=0.5)
                        for bar, v in zip(bars, yr.values):
                            ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
                                    f'{v:.3f}', ha='center', fontsize=8)
                        ax.set_ylim(0, yr.max()*1.15)
                        ax.set_xlabel('Tahun Akademik'); ax.set_ylabel('Burnout Score Rata-rata')
                        ax.set_title('Burnout per Tahun Akademik')
                        despine(ax); fig.tight_layout()
                        fig_show(fig, "Semakin gelap = burnout rata-rata lebih tinggi")

            elif bq_num == 2 and not df_b.empty and 'risk_level' in df_b.columns:
                pct_h = (df_b['risk_level']=='High').mean()*100
                st.metric("Mahasiswa High Risk", f"{pct_h:.2f}%")
                feat_b2 = [c for c in ['study_hours_per_day','sleep_hours','physical_activity',
                                        'social_support','screen_time'] if c in df_b.columns]
                if feat_b2:
                    ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
                    profile = df_b.groupby('risk_level')[feat_b2].mean().reindex(ra)
                    lc_map  = {'Low': C_LIGHT, 'Medium': C_MID, 'High': C_DARK}
                    n_feat  = len(feat_b2)
                    fig, axes = plt.subplots(1, n_feat, figsize=(2.8*n_feat, 3.5))
                    if n_feat == 1: axes = [axes]
                    for i, col in enumerate(feat_b2):
                        bar_c = [lc_map.get(r, C_MID) for r in profile.index]
                        bars  = axes[i].bar(profile.index, profile[col],
                                            color=bar_c, edgecolor='white', linewidth=0.8, width=0.5)
                        for bar, v in zip(bars, profile[col]):
                            axes[i].text(bar.get_x()+bar.get_width()/2, v+profile[col].max()*0.03,
                                         f'{v:.1f}', ha='center', fontsize=7)
                        axes[i].set_title(col.replace('_',' ').replace('per day','').strip().title(),
                                          fontsize=8, fontweight='bold')
                        axes[i].set_ylim(0, profile[col].max()*1.22)
                        axes[i].tick_params(axis='x', labelsize=7, rotation=15)
                        sns.despine(ax=axes[i])
                    fig.suptitle('Profil Perilaku per Risk Level', fontsize=10, fontweight='bold')
                    fig.tight_layout()
                    fig_show(fig, "Small multiples - sumbu Y mandiri per panel | Low=terang, High=gelap")

            elif bq_num == 3 and not df_n.empty and 'emotion' in df_n.columns:
                em_ord = [e for e in EMOTION_ORDER if e in df_n['emotion'].unique()]
                em_ord += [e for e in df_n['emotion'].unique() if e not in em_ord]
                em_c = df_n['emotion'].value_counts().reindex(em_ord)
                n    = len(em_c)
                clrs = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]

                col_l, col_r = st.columns(2)
                with col_l:
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    bars = ax.bar(em_c.index, em_c.values, color=clrs,
                                  edgecolor='white', linewidth=0.8, width=0.6)
                    annotate_bars(ax, bars, '{:,.0f}', fontsize=8)
                    ax.set_ylim(0, em_c.max()*1.18)
                    ax.set_xlabel('Emosi'); ax.set_ylabel('Jumlah Teks')
                    ax.set_title('Frekuensi Kelas Emosi')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, "Sadness (gelap) = paling berat | Joy (terang) = paling ringan")
                with col_r:
                    pie_clrs_em = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    wedges, texts, autotexts = ax.pie(
                        em_c.values, labels=em_c.index, autopct='%1.1f%%',
                        colors=pie_clrs_em, startangle=90,
                        wedgeprops={'width':0.55,'edgecolor':'white','linewidth':2},
                        pctdistance=0.75, labeldistance=1.1
                    )
                    for at in autotexts:
                        at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
                    ax.set_title('Proporsi Emosi (Donut)')
                    fig.tight_layout()
                    fig_show(fig, "Donut chart: proporsi tiap kelas emosi dari teks curhatan")

            elif bq_num == 4 and not df_b.empty and 'sleep_hours' in df_b.columns:
                pct_u6 = (df_b['sleep_hours'] < 6).mean() * 100
                col_l, col_r = st.columns(2)
                with col_l:
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    _, bins, patches = ax.hist(df_b['sleep_hours'].dropna(), bins=15,
                                               color=C_MID, edgecolor='white', linewidth=1.0)
                    for patch, left in zip(patches, bins[:-1]):
                        patch.set_facecolor(C_DARK if left < 6 else C_GREY)
                    ax.axvline(6, color=C_DARK, linestyle='--', linewidth=1.5, label='Batas 6 jam')
                    ax.set_xlabel('Jam Tidur per Hari')
                    ax.set_ylabel('Jumlah Mahasiswa')
                    ax.set_title('Distribusi Jam Tidur')
                    ax.legend(fontsize=8)
                    despine(ax); fig.tight_layout()
                    fig_show(fig, f"Biru gelap (#0034c8) = zona berisiko (< 6 jam) | {pct_u6:.1f}% mahasiswa dalam zona ini")
                with col_r:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Tidur < 6 Jam", f"{pct_u6:.1f}%")
                    c2.metric("Rata-rata", f"{df_b['sleep_hours'].mean():.2f} jam")
                    c3.metric("Median", f"{df_b['sleep_hours'].median():.2f} jam")

            elif bq_num == 5 and not df_m.empty:
                dim_data = {}
                if MBI_EX: dim_data['Exhaustion']     = df_m[MBI_EX].mean(axis=1)
                if MBI_CY: dim_data['Cynicism']        = df_m[MBI_CY].mean(axis=1)
                if MBI_EF: dim_data['Efficacy (inv)']  = df_m[MBI_EF].mean(axis=1)
                if 'fatigue_score' in df_m.columns:
                    dim_data['Fatigue Score'] = df_m['fatigue_score']
                if dim_data:
                    df_dim5 = pd.DataFrame(dim_data)
                    means = df_dim5.mean().sort_values(ascending=False)
                    std_v = df_dim5.std().reindex(means.index)
                    n = len(means)
                    clrs = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                    bars = axes[0].bar(means.index, means.values, color=clrs,
                                       edgecolor='white', linewidth=0.8, width=0.5)
                    axes[0].errorbar(means.index, means.values, yerr=std_v.values,
                                     fmt='none', color='#334', capsize=4, linewidth=1)
                    for bar, v in zip(bars, means.values):
                        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.12,
                                     f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
                    axes[0].set_ylim(0, (means+std_v).max()+0.5)
                    axes[0].set_ylabel('Rata-rata Skor (0–6)')
                    axes[0].set_title('Rata-rata & Variabilitas Dimensi MBI')
                    sns.despine(ax=axes[0])
                    bp_data = [df_dim5[c].dropna().values for c in means.index]
                    bp = axes[1].boxplot(bp_data, labels=means.index, patch_artist=True,
                                         medianprops=dict(color='white', linewidth=2),
                                         flierprops=dict(marker='o', markersize=2, color='#556'))
                    for i, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(clrs[i]); patch.set_alpha(0.88)
                    axes[1].set_ylabel('Skor (0–6)')
                    axes[1].set_title('Distribusi Dimensi MBI')
                    sns.despine(ax=axes[1])
                    fig.tight_layout()
                    fig_show(fig, "Error bar = standar deviasi | urutan dan warna konsisten antara kedua panel")

            elif bq_num == 6 and not df_b.empty and 'study_hours_per_day' in df_b.columns and 'burnout_score' in df_b.columns:
                smp = df_b.sample(min(2500, len(df_b)), random_state=42)
                r   = smp['study_hours_per_day'].corr(smp['burnout_score'])
                g_h = df_b[df_b['study_hours_per_day']>8]['burnout_score'].dropna()
                g_l = df_b[df_b['study_hours_per_day']<=8]['burnout_score'].dropna()
                _, p_mw = stats.mannwhitneyu(g_h, g_l, alternative='greater')

                col_l, col_r = st.columns([2,1])
                with col_l:
                    fig, ax = plt.subplots(figsize=(6.5, 4))
                    ax.scatter(smp['study_hours_per_day'], smp['burnout_score'],
                               alpha=0.3, color=C_MID, s=8, rasterized=True)
                    m, b_i = np.polyfit(smp['study_hours_per_day'], smp['burnout_score'], 1)
                    x_ = np.linspace(smp['study_hours_per_day'].min(), smp['study_hours_per_day'].max(), 100)
                    ax.plot(x_, m*x_+b_i, color=C_DARK, linewidth=2, label=f'Trendline (r={r:.3f})')
                    ax.axvline(8, color=C_LIGHT, linestyle='--', linewidth=1.2, label='8 jam')
                    ax.set_xlabel('Jam Belajar per Hari')
                    ax.set_ylabel('Burnout Score')
                    ax.set_title('Jam Belajar vs Burnout Score')
                    ax.legend(fontsize=8); despine(ax); fig.tight_layout()
                    fig_show(fig, f"Sample 2.500 | Scatter: alpha=0.3 C_MID #{C_MID[1:]} | Trendline: C_DARK #{C_DARK[1:]}")
                with col_r:
                    st.metric("Korelasi Pearson", f"{r:.4f}")
                    st.metric("Mann-Whitney p", f"{p_mw:.4f}",
                              delta="Signifikan" if p_mw<0.05 else "Tidak signifikan")
                    st.metric(">8 jam: mean burnout", f"{g_h.mean():.4f}")
                    st.metric("≤8 jam: mean burnout", f"{g_l.mean():.4f}")

            elif bq_num == 7 and not df_b.empty and 'sleep_hours' in df_b.columns and 'burnout_score' in df_b.columns:
                df_b7 = df_b.copy()
                df_b7['sg'] = df_b7['sleep_hours'].apply(lambda x: '< 6 jam' if pd.notna(x) and x<6 else '≥ 6 jam')
                g_l7 = df_b7[df_b7['sg']=='< 6 jam']['burnout_score'].dropna()
                g_h7 = df_b7[df_b7['sg']=='≥ 6 jam']['burnout_score'].dropna()
                _, p7 = stats.mannwhitneyu(g_l7, g_h7, alternative='greater')
                pal7 = {'< 6 jam': C_DARK, '≥ 6 jam': C_LIGHT}
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df_b7, x='sg', y='burnout_score', hue='sg',
                            order=['< 6 jam','≥ 6 jam'], hue_order=['< 6 jam','≥ 6 jam'],
                            palette=pal7, legend=False, ax=ax, linewidth=0.8,
                            flierprops=dict(marker='o', markersize=2, markerfacecolor=C_GREY))
                ax.set_xlabel('Kelompok Tidur'); ax.set_ylabel('Burnout Score')
                ax.set_title('Burnout Score per Kelompok Tidur')
                despine(ax); fig.tight_layout()
                fig_show(fig, f"Mann-Whitney U: p = {p7:.4f} | {'Signifikan ✓' if p7<0.05 else 'Tidak signifikan'} (α=0.05)")
                c1,c2 = st.columns(2)
                c1.metric("< 6 jam: mean burnout", f"{g_l7.mean():.4f}")
                c2.metric("≥ 6 jam: mean burnout", f"{g_h7.mean():.4f}")

            elif bq_num == 8 and not df_b.empty:
                feat_corr = [c for c in ['study_hours_per_day','sleep_hours','exam_pressure',
                                          'physical_activity','social_support','screen_time',
                                          'academic_performance'] if c in df_b.columns]
                if feat_corr and 'burnout_score' in df_b.columns:
                    corr_ser = df_b[feat_corr+['burnout_score']].corr()['burnout_score'].drop('burnout_score')
                    corr_ser = corr_ser.reindex(corr_ser.abs().sort_values(ascending=True).index)
                    n = len(corr_ser)
                    clrs = [C_DARK if v > 0 else C_LIGHT for v in corr_ser.values]
                    fig, ax = plt.subplots(figsize=(7, 4))
                    bars = ax.barh(corr_ser.index, corr_ser.values, color=clrs,
                                   edgecolor='white', linewidth=0.8, height=0.55)
                    ax.axvline(0, color='#888', linewidth=0.8)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + (0.005 if w >= 0 else -0.005), bar.get_y()+bar.get_height()/2,
                                f'{w:.3f}', va='center', ha='left' if w>=0 else 'right', fontsize=8)
                    ax.set_xlabel('Korelasi Pearson dengan Burnout Score')
                    ax.set_title('Korelasi Fitur dengan Burnout Score')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, "Biru gelap = korelasi positif (naik bersama burnout) | Biru terang = negatif")

            elif bq_num == 9 and not df_b.empty:
                feat_b9 = [c for c in ['physical_activity','social_support'] if c in df_b.columns]
                if feat_b9 and 'burnout_score' in df_b.columns:
                    n_feat = len(feat_b9)
                    fig, axes = plt.subplots(1, n_feat, figsize=(5.5*n_feat, 4))
                    if n_feat == 1: axes = [axes]
                    for i, col in enumerate(feat_b9):
                        smp9 = df_b[[col,'burnout_score']].dropna().sample(min(2000,len(df_b)), random_state=42)
                        r9,_ = stats.spearmanr(smp9[col], smp9['burnout_score'])
                        axes[i].scatter(smp9[col], smp9['burnout_score'],
                                        alpha=0.3, color=C_MID, s=8, rasterized=True)
                        m, b_i = np.polyfit(smp9[col], smp9['burnout_score'], 1)
                        x_ = np.linspace(smp9[col].min(), smp9[col].max(), 100)
                        axes[i].plot(x_, m*x_+b_i, color=C_DARK, linewidth=2, label=f'Spearman r={r9:.3f}')
                        axes[i].set_xlabel(col.replace('_',' ').title())
                        axes[i].set_ylabel('Burnout Score')
                        axes[i].set_title(col.replace('_',' ').title())
                        axes[i].legend(fontsize=8); sns.despine(ax=axes[i])
                    fig.tight_layout()
                    fig_show(fig, "Korelasi Spearman (non-parametrik) | alpha=0.3, C_MID | trendline C_DARK")

            elif bq_num == 10 and not df_n.empty and 'emotion' in df_n.columns:
                em_ord10 = [e for e in EMOTION_ORDER if e in df_n['emotion'].unique()]
                em_ord10 += [e for e in df_n['emotion'].unique() if e not in em_ord10]
                n10 = len(em_ord10)
                pal10 = {e: PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n10-1,1))] for i,e in enumerate(em_ord10)}
                kw_col = next((c for c in ['stress_keyword_count','word_count'] if c in df_n.columns), None)

                col_l, col_r = st.columns(2)
                with col_l:
                    em_c10 = df_n['emotion'].value_counts().reindex(em_ord10)
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    bars = ax.bar(em_c10.index, em_c10.values,
                                  color=[pal10[e] for e in em_c10.index],
                                  edgecolor='white', linewidth=0.8, width=0.6)
                    annotate_bars(ax, bars, '{:,.0f}', fontsize=8)
                    ax.set_ylim(0, em_c10.max()*1.18)
                    ax.set_xlabel('Emosi'); ax.set_ylabel('Jumlah Teks')
                    ax.set_title('Frekuensi Emosi')
                    despine(ax); fig.tight_layout()
                    fig_show(fig)
                with col_r:
                    if kw_col:
                        kw_m10 = df_n.groupby('emotion')[kw_col].mean().reindex(em_ord10)
                        fig, ax = plt.subplots(figsize=(5.5, 3.8))
                        bars = ax.bar(kw_m10.index, kw_m10.values,
                                      color=[pal10[e] for e in kw_m10.index],
                                      edgecolor='white', linewidth=0.8, width=0.6)
                        for bar, v in zip(bars, kw_m10.values):
                            if v == v:
                                ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                                        f'{v:.2f}', ha='center', fontsize=8)
                        ax.set_ylim(0, kw_m10.max()*1.18)
                        ax.set_xlabel('Emosi'); ax.set_ylabel(kw_col.replace('_',' ').title())
                        ax.set_title(f'Rata-rata {kw_col.replace("_"," ").title()} per Emosi')
                        despine(ax); fig.tight_layout()
                        fig_show(fig, "Sadness & Anger diharapkan lebih tinggi dari Joy & Neutral")
                if kw_col:
                    groups = [df_n[df_n['emotion']==e][kw_col].dropna() for e in em_ord10]
                    valid_g = [g for g in groups if len(g)>1]
                    if len(valid_g)>=2:
                        stat10, p10 = stats.kruskal(*valid_g)
                        st.caption(f"Kruskal-Wallis: H={stat10:.4f}, p={p10:.4f} | {'Signifikan ✓' if p10<0.05 else 'Tidak signifikan'} (α=0.05)")

            elif bq_num == 11 and not df_l.empty and LIFESTYLE_FEAT and 'Stress_Level' in df_l.columns:
                sl_avail = [s for s in STRESS_LS_ORD if s in df_l['Stress_Level'].unique()]
                grouped = df_l.groupby('Stress_Level')[LIFESTYLE_FEAT].mean().reindex(sl_avail)
                lc_map  = {sl_avail[0]: C_LIGHT,
                           sl_avail[len(sl_avail)//2] if len(sl_avail)>2 else sl_avail[0]: C_MID,
                           sl_avail[-1]: C_DARK}
                bar_clrs = [lc_map.get(r, C_MID) for r in sl_avail]
                n_feat = len(LIFESTYLE_FEAT)
                ncols  = 3; nrows = (n_feat+ncols-1)//ncols
                fig = plt.figure(figsize=(14, 3.8*nrows+4))
                for i, col in enumerate(LIFESTYLE_FEAT):
                    ax = fig.add_subplot(nrows+1, ncols, i+1)
                    bars = ax.bar(sl_avail, grouped[col].values, color=bar_clrs,
                                  edgecolor='white', linewidth=0.8, width=0.5)
                    for bar, v in zip(bars, grouped[col].values):
                        ax.text(bar.get_x()+bar.get_width()/2, v+grouped[col].max()*0.04,
                                f'{v:.1f}', ha='center', fontsize=8)
                    ax.set_title(SHORT_LS.get(col, col), fontsize=9, fontweight='bold')
                    ax.set_ylabel('Jam/Hari', fontsize=8)
                    ax.set_ylim(0, grouped[col].max()*1.22)
                    sns.despine(ax=ax)
                ax_h = fig.add_subplot(nrows+1, 1, nrows+1)
                corr_ls = df_l[LIFESTYLE_FEAT].corr()
                short_idx = {c: SHORT_LS.get(c,c) for c in LIFESTYLE_FEAT}
                corr_ls = corr_ls.rename(index=short_idx, columns=short_idx)
                sns.heatmap(corr_ls, annot=True, fmt='.2f', cmap=HEATMAP_DIV,
                            vmin=-1, vmax=1, center=0, linewidths=1.5, linecolor='white',
                            ax=ax_h, annot_kws={'size':9})
                ax_h.set_title('Korelasi Antar Fitur Perilaku (RdBu_r)')
                fig.tight_layout()
                fig_show(fig, "Small multiples per fitur + heatmap korelasi divergen (merah=negatif, biru=positif)")

            elif bq_num == 12 and not df_m.empty and 'risk_level' in df_m.columns:
                dim_data12 = {}
                if MBI_EX: dim_data12['Exhaustion']    = df_m[MBI_EX].mean(axis=1)
                if MBI_CY: dim_data12['Cynicism']       = df_m[MBI_CY].mean(axis=1)
                if MBI_EF: dim_data12['Efficacy(inv)']  = df_m[MBI_EF].mean(axis=1)
                if 'fatigue_score' in df_m.columns: dim_data12['Fatigue Score'] = df_m['fatigue_score']
                if dim_data12:
                    df_dim12 = pd.DataFrame(dim_data12)
                    df_dim12['risk_level'] = df_m['risk_level'].values
                    ra = [r for r in RISK_ORDER if r in df_dim12['risk_level'].unique()]
                    n  = len(ra)
                    pal_r = {r: PALETTE[int(i*(len(PALETTE)-1)/max(n-1,1))] for i,r in enumerate(ra)}
                    dim_cols12 = list(dim_data12.keys())
                    ncols12 = 2; nrows12 = (len(dim_cols12)+ncols12-1)//ncols12
                    fig, axes = plt.subplots(nrows12, ncols12, figsize=(10, 4*nrows12))
                    axes = np.array(axes).flatten()
                    for i, col in enumerate(dim_cols12):
                        sns.boxplot(data=df_dim12, x='risk_level', y=col,
                                    hue='risk_level', order=ra, hue_order=ra,
                                    palette=pal_r, legend=False, ax=axes[i], linewidth=0.8,
                                    flierprops=dict(marker='o', markersize=2))
                        axes[i].set_title(col, fontsize=10, fontweight='bold')
                        axes[i].set_xlabel('Risk Level'); axes[i].set_ylabel('Skor (0–6)')
                        sns.despine(ax=axes[i])
                    for j in range(len(dim_cols12), len(axes)): axes[j].set_visible(False)
                    fig.tight_layout()
                    if 'fatigue_score' in dim_data12:
                        g12 = [df_dim12[df_dim12['risk_level']==r]['fatigue_score'].dropna() for r in ra]
                        _, p12 = stats.kruskal(*[g for g in g12 if len(g)>1])
                        fig_show(fig, f"Kruskal-Wallis fatigue per risk level: p={p12:.4f} | {'Signifikan ✓' if p12<0.05 else 'Tidak signifikan'}")
                    else:
                        fig_show(fig)

            elif bq_num == 13 and not df_b.empty and 'risk_level' in df_b.columns:
                feat_b13 = [c for c in ['study_hours_per_day','sleep_hours'] if c in df_b.columns]
                if len(feat_b13) == 2:
                    smp13 = df_b[feat_b13+['risk_level','burnout_score']].dropna()
                    smp13 = smp13.sample(min(2000, len(smp13)), random_state=42)
                    ra = [r for r in RISK_ORDER if r in smp13['risk_level'].unique()]
                    cmap3 = {'Low': C_LIGHT, 'Medium': C_MID, 'High': C_DARK}
                    mkr3  = {'Low': 'o', 'Medium': 's', 'High': '^'}
                    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
                    for rl in ra:
                        mask = smp13['risk_level'] == rl
                        axes[0].scatter(smp13.loc[mask,'study_hours_per_day'],
                                        smp13.loc[mask,'sleep_hours'],
                                        c=cmap3.get(rl,C_MID), marker=mkr3.get(rl,'o'),
                                        alpha=0.4, s=16, label=rl, rasterized=True)
                    axes[0].axhline(6, color=C_DARK, linestyle='--', linewidth=1.2, label='Tidur 6 jam')
                    axes[0].axvline(8, color=C_GREY, linestyle='--', linewidth=1.2, label='Belajar 8 jam')
                    axes[0].set_xlabel('Jam Belajar per Hari')
                    axes[0].set_ylabel('Jam Tidur per Hari')
                    axes[0].legend(title='Risk Level', fontsize=8, title_fontsize=8)
                    axes[0].set_title('Belajar vs Tidur per Risk Level')
                    sns.despine(ax=axes[0])

                    feat_heat = [c for c in ['study_hours_per_day','sleep_hours','exam_pressure',
                                              'stress_level','burnout_score'] if c in df_b.columns]
                    corr13 = df_b[feat_heat].corr()
                    sns.heatmap(corr13, annot=True, fmt='.2f', cmap=HEATMAP_DIV,
                                vmin=-1, vmax=1, center=0, linewidths=1.5, linecolor='white',
                                ax=axes[1], annot_kws={'size':9})
                    axes[1].set_title('Korelasi Fitur Sinyal Burnout')
                    fig.tight_layout()
                    fig_show(fig, "Double-encoding: warna + marker shape | zona merah = belajar>8 jam & tidur<6 jam")

            elif bq_num == 14 and not df_b.empty and 'social_support' in df_b.columns and 'burnout_score' in df_b.columns:
                df_b14 = df_b[['social_support','burnout_score']].dropna()
                q25 = df_b14['social_support'].quantile(0.25)
                q75 = df_b14['social_support'].quantile(0.75)
                g_low14  = df_b14[df_b14['social_support']<=q25]['burnout_score']
                g_high14 = df_b14[df_b14['social_support']>=q75]['burnout_score']
                _, p14 = stats.mannwhitneyu(g_low14, g_high14, alternative='greater')
                d14 = (g_low14.mean()-g_high14.mean()) / np.sqrt((g_low14.std()**2+g_high14.std()**2)/2)

                col_l, col_r = st.columns([2,1])
                with col_l:
                    df_plot14 = pd.DataFrame({
                        'Social Support': ['Q1 (Rendah)','Q4 (Tinggi)'],
                        'Burnout Mean': [g_low14.mean(), g_high14.mean()],
                    })
                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    bars = ax.bar(df_plot14['Social Support'], df_plot14['Burnout Mean'],
                                  color=[C_DARK, C_LIGHT], edgecolor='white', linewidth=0.8, width=0.45)
                    for bar, v in zip(bars, df_plot14['Burnout Mean']):
                        ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f'{v:.4f}',
                                ha='center', fontsize=9, fontweight='bold')
                    ax.set_ylim(0, df_plot14['Burnout Mean'].max()*1.15)
                    ax.set_ylabel('Burnout Score Rata-rata')
                    ax.set_title('Burnout: Social Support Rendah vs Tinggi')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, f"Mann-Whitney p={p14:.4f} | Cohen's d={d14:.3f} | {'Signifikan ✓' if p14<0.05 else 'Tidak'}")
                with col_r:
                    st.metric("Q1 (Rendah) mean burnout", f"{g_low14.mean():.4f}")
                    st.metric("Q4 (Tinggi) mean burnout", f"{g_high14.mean():.4f}")
                    st.metric("Selisih (absolut)", f"{abs(g_low14.mean()-g_high14.mean()):.4f}")
                    st.metric("Cohen's d", f"{d14:.3f}")
                    st.metric("Mann-Whitney p", f"{p14:.4f}", delta="Signifikan ✓" if p14<0.05 else "Tidak")

            elif bq_num == 15 and not df_b.empty and 'risk_level' in df_b.columns:
                pcols = [c for c in ['study_hours_per_day','sleep_hours','physical_activity',
                                      'social_support','screen_time','exam_pressure'] if c in df_b.columns]
                if pcols:
                    ph = df_b[df_b['risk_level']=='High'][pcols].mean()
                    pl = df_b[df_b['risk_level']=='Low'][pcols].mean()
                    n_feat = len(pcols); ncols = 3; nrows = (n_feat+ncols-1)//ncols
                    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8*nrows))
                    axes = axes.flatten()
                    for i, col in enumerate(pcols):
                        axes[i].bar(['Low Risk','High Risk'], [pl[col], ph[col]],
                                    color=[C_LIGHT, C_DARK], edgecolor='white', linewidth=0.8, width=0.5)
                        ymax = max(pl[col], ph[col])
                        for j, v in enumerate([pl[col], ph[col]]):
                            axes[i].text(j, v+ymax*0.04, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
                        axes[i].set_title(col.replace('_',' ').title(), fontsize=9, fontweight='bold')
                        axes[i].set_ylim(0, max(pl[col], ph[col])*1.22)
                        sns.despine(ax=axes[i])
                    for j in range(n_feat, len(axes)): axes[j].set_visible(False)
                    fig.tight_layout()
                    fig_show(fig, "Biru terang (#00bcff) = Low Risk | Biru gelap (#0034c8) = High Risk")


elif page == "EDA Dataset":
    shdr("EDA 4 Dataset Model-Ready")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Burnout Fatigue", "Student Lifestyle", "MBI Questions", "Curhat NLP"
    ])

    with tab1:
        if df_b.empty:
            st.warning("burnout_fatigue_clean.csv tidak ditemukan.")
        else:
            sshdr("Profiling")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Baris", f"{len(df_b):,}")
            c2.metric("Kolom", len(df_b.columns))
            c3.metric("Missing", int(df_b.isnull().sum().sum()))
            c4.metric("Duplikat", int(df_b.duplicated().sum()))
            with st.expander("Statistik Deskriptif"):
                st.dataframe(df_b[BURNOUT_FEAT+['burnout_score']].describe().round(3) if BURNOUT_FEAT else df_b.describe().round(3))

            sshdr("Univariat - Target")
            col_l, col_r = st.columns(2)
            with col_l:
                if 'risk_level' in df_b.columns:
                    ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
                    rc = df_b['risk_level'].value_counts().reindex(ra)
                    clrs = [PALETTE[int(i*(len(PALETTE)-1)/max(len(rc)-1,1))] for i in range(len(rc))]
                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                    bars = ax.bar(rc.index, rc.values, color=clrs, edgecolor='white', linewidth=0.8, width=0.5)
                    annotate_bars_pct(ax, bars, len(df_b))
                    ax.set_ylim(0, rc.max()*1.22)
                    ax.set_ylabel('Jumlah')
                    ax.set_title('Distribusi Risk Level')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, "Low=terang (#00bcff) | High=gelap (#0034c8) | urutan ordinal Low→High")
            with col_r:
                if 'burnout_score' in df_b.columns:
                    bs = df_b['burnout_score']
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
                    axes[0].hist(bs.dropna(), bins=35, color=C_MID, edgecolor='white', linewidth=0.8)
                    axes[0].set_xlabel('Burnout Score'); axes[0].set_ylabel('Frekuensi')
                    axes[0].set_title(f'Histogram Burnout Score')
                    sns.despine(ax=axes[0])
                    bp = axes[1].boxplot(bs.dropna(), patch_artist=True,
                                         medianprops=dict(color='white', linewidth=2),
                                         flierprops=dict(marker='o', markersize=2, color=C_DARK))
                    bp['boxes'][0].set_facecolor(C_MID); bp['boxes'][0].set_alpha(0.85)
                    axes[1].set_ylabel('Burnout Score'); axes[1].set_xticks([])
                    axes[1].set_title('Boxplot Burnout Score')
                    sns.despine(ax=axes[1])
                    fig.tight_layout()
                    fig_show(fig, f"Satu warna solid C_MID (#007ff1) | skew={bs.skew():.3f} | kurt={bs.kurtosis():.3f}")

            sshdr("Univariat - Semua Fitur")
            if BURNOUT_FEAT:
                view_type = st.radio("Tampilan:", ["Histogram","Boxplot"], horizontal=True, key="b_view")
                n = len(BURNOUT_FEAT); ncols = 4; nrows = (n+ncols-1)//ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5*nrows))
                axes = axes.flatten()
                for i, col in enumerate(BURNOUT_FEAT):
                    if view_type == "Histogram":
                        axes[i].hist(df_b[col].dropna(), bins=25, color=C_MID, edgecolor='white', linewidth=0.8)
                        axes[i].set_ylabel('Frekuensi', fontsize=7)
                    else:
                        bp = axes[i].boxplot(df_b[col].dropna(), patch_artist=True,
                                             medianprops=dict(color='white', linewidth=1.5),
                                             flierprops=dict(marker='o', markersize=2, color=C_DARK))
                        bp['boxes'][0].set_facecolor(C_MID); bp['boxes'][0].set_alpha(0.85)
                        axes[i].set_xticks([])
                    axes[i].set_title(col.replace('_',' ').title(), fontsize=8, fontweight='bold')
                    sns.despine(ax=axes[i])
                for j in range(n, len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                fig_show(fig, "Histogram: 1 warna solid C_MID (#007ff1) dengan edge putih | tidak menggunakan gradasi")

            sshdr("Bivariat - Scatter Fitur vs Burnout Score")
            if BURNOUT_FEAT and 'burnout_score' in df_b.columns:
                smp_b = df_b.sample(min(2000,len(df_b)), random_state=42)
                n = len(BURNOUT_FEAT); ncols = 4; nrows = (n+ncols-1)//ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
                axes = axes.flatten()
                for i, col in enumerate(BURNOUT_FEAT):
                    valid = smp_b[[col,'burnout_score']].dropna()
                    r = valid[col].corr(valid['burnout_score'])
                    axes[i].scatter(valid[col], valid['burnout_score'],
                                    alpha=0.3, color=C_MID, s=8, rasterized=True)
                    m, b_i = np.polyfit(valid[col], valid['burnout_score'], 1)
                    x_ = np.linspace(valid[col].min(), valid[col].max(), 100)
                    axes[i].plot(x_, m*x_+b_i, color=C_DARK, linewidth=1.8, label=f'r={r:.3f}')
                    axes[i].legend(fontsize=7)
                    axes[i].set_title(col.replace('_',' ').title(), fontsize=8, fontweight='bold')
                    axes[i].set_xlabel(col.replace('_',' ').title(), fontsize=7)
                    axes[i].set_ylabel('Burnout', fontsize=7)
                    sns.despine(ax=axes[i])
                for j in range(n, len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                fig_show(fig, "Scatter alpha=0.3 C_MID (#007ff1) | trendline C_DARK (#0034c8) | sample 2.000")

            sshdr("Bivariat - Boxplot Fitur per Risk Level")
            if BURNOUT_FEAT and 'risk_level' in df_b.columns:
                ra = [r for r in RISK_ORDER if r in df_b['risk_level'].unique()]
                pal_r = {r: PALETTE[int(i*(len(PALETTE)-1)/max(len(ra)-1,1))] for i,r in enumerate(ra)}
                n = len(BURNOUT_FEAT); ncols = 4; nrows = (n+ncols-1)//ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
                axes = axes.flatten()
                for i, col in enumerate(BURNOUT_FEAT):
                    sns.boxplot(data=df_b, x='risk_level', y=col,
                                hue='risk_level', order=ra, hue_order=ra,
                                palette=pal_r, legend=False, ax=axes[i], linewidth=0.8,
                                flierprops=dict(marker='o', markersize=2))
                    axes[i].set_title(col.replace('_',' ').title(), fontsize=8, fontweight='bold')
                    axes[i].set_xlabel('Risk Level', fontsize=7)
                    axes[i].set_ylabel('', fontsize=7)
                    sns.despine(ax=axes[i])
                for j in range(n, len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                fig_show(fig, "Low=terang | High=gelap | sumbu Y mandiri per panel")

            sshdr("Multivariat - Heatmap Korelasi & VIF")
            col_l, col_r = st.columns(2)
            with col_l:
                feat_num = [c for c in BURNOUT_FEAT+['burnout_score'] if c in df_b.columns]
                corr = df_b[feat_num].corr()
                fig, ax = plt.subplots(figsize=(7, 5.5))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap=HEATMAP_DIV,
                            vmin=-1, vmax=1, center=0,
                            linewidths=1.5, linecolor='white', ax=ax, annot_kws={'size':8})
                ax.set_title('Heatmap Korelasi')
                fig.tight_layout(); fig_show(fig, "RdBu_r: merah=korelasi negatif | putih=0 | biru=positif | linewidths=1.5 (Tufte: data-ink ratio)")
            with col_r:
                feat_vif = [c for c in BURNOUT_FEAT if c in df_b.columns]
                if len(feat_vif) >= 2:
                    Xv = df_b[feat_vif].dropna().sample(min(5000,len(df_b)), random_state=42)
                    vif_df = pd.DataFrame({
                        'Fitur': feat_vif,
                        'VIF'  : [variance_inflation_factor(Xv.values, i) for i in range(len(feat_vif))]
                    }).sort_values('VIF', ascending=False).reset_index(drop=True)
                    vif_df['Status'] = vif_df['VIF'].apply(lambda v: '🔴 Tinggi' if v>=10 else '🟢 Aman')
                    st.markdown("**Uji Multikolinearitas (VIF)**")
                    st.dataframe(vif_df.round(2), use_container_width=True, height=250)
                    st.caption("VIF < 10 = aman untuk model linear | VIF ≥ 10 = potensi multikolinearitas")

            sshdr("Multivariat - Scatter 3 Variabel")
            if all(c in df_b.columns for c in ['study_hours_per_day','sleep_hours','risk_level']):
                smp3 = df_b.sample(min(2000,len(df_b)), random_state=42)
                ra = [r for r in RISK_ORDER if r in smp3['risk_level'].unique()]
                cmap3 = {'Low':C_LIGHT,'Medium':C_MID,'High':C_DARK}
                mkr3  = {'Low':'o','Medium':'s','High':'^'}
                fig, ax = plt.subplots(figsize=(8, 5))
                for rl in ra:
                    mask = smp3['risk_level'] == rl
                    ax.scatter(smp3.loc[mask,'study_hours_per_day'], smp3.loc[mask,'sleep_hours'],
                               c=cmap3.get(rl,C_MID), marker=mkr3.get(rl,'o'),
                               alpha=0.4, s=18, label=rl, rasterized=True)
                ax.axhline(6, color=C_DARK, linestyle='--', linewidth=1.2, label='Tidur 6 jam')
                ax.axvline(8, color=C_GREY, linestyle='--', linewidth=1.2, label='Belajar 8 jam')
                ax.set_xlabel('Jam Belajar per Hari'); ax.set_ylabel('Jam Tidur')
                ax.legend(title='Risk Level', fontsize=9, title_fontsize=9)
                ax.set_title('Belajar vs Tidur per Risk Level')
                despine(ax); fig.tight_layout()
                fig_show(fig, "Double-encoding: warna + marker shape | Low=○ terang, Medium=□ mid, High=△ gelap")

    with tab2:
        if df_l.empty:
            st.warning("student_lifestyle_clean.csv tidak ditemukan.")
        else:
            sshdr("Profiling")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Baris", f"{len(df_l):,}")
            c2.metric("Kolom", len(df_l.columns))
            c3.metric("Missing", int(df_l.isnull().sum().sum()))
            c4.metric("Duplikat", int(df_l.duplicated().sum()))

            sshdr("Univariat - Target Stress_Level")
            info_box("Kolom <code>Stress_Level</code> adalah string ordinal: <b>'Low', 'Moderate', 'High'</b>. "
                     "Encoded ke numerik (1/2/3) untuk analisis statistik.")
            col_l, col_r = st.columns(2)
            with col_l:
                if 'Stress_Level' in df_l.columns:
                    sl_avail = [s for s in STRESS_LS_ORD if s in df_l['Stress_Level'].unique()]
                    rc_sl = df_l['Stress_Level'].value_counts().reindex(sl_avail)
                    n = len(rc_sl)
                    clrs = [PALETTE[int(i*(len(PALETTE)-1)/max(n-1,1))] for i in range(n)]
                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                    bars = ax.bar(rc_sl.index, rc_sl.values, color=clrs,
                                  edgecolor='white', linewidth=0.8, width=0.5)
                    annotate_bars_pct(ax, bars, len(df_l))
                    ax.set_ylim(0, rc_sl.max()*1.22)
                    ax.set_ylabel('Jumlah Mahasiswa')
                    ax.set_title('Distribusi Stress_Level')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, "Low=terang (#00bcff) | High=gelap (#0034c8) | urutan ordinal manual")
            with col_r:
                if 'Stress_Level_num' in df_l.columns:
                    sl_n = df_l['Stress_Level_num'].dropna()
                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                    ax.hist(sl_n, bins=10, color=C_MID, edgecolor='white', linewidth=0.8)
                    ax.set_xlabel('Stress Level (1=Low, 2=Moderate, 3=High)')
                    ax.set_ylabel('Frekuensi')
                    ax.set_title('Histogram Stress_Level (Encoded)')
                    despine(ax); fig.tight_layout()
                    fig_show(fig, "Setelah encoding ordinal: Low=1, Moderate=2, High=3")

            sshdr("Univariat - Fitur Aktivitas Harian")
            if LIFESTYLE_FEAT:
                n = len(LIFESTYLE_FEAT); ncols = 3; nrows = (n+ncols-1)//ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8*nrows))
                axes = axes.flatten()
                for i, col in enumerate(LIFESTYLE_FEAT):
                    skv = df_l[col].skew()
                    axes[i].hist(df_l[col].dropna(), bins=20, color=C_MID, edgecolor='white', linewidth=1.0)
                    axes[i].set_title(SHORT_LS.get(col,col), fontsize=9, fontweight='bold')
                    axes[i].set_xlabel('Jam per Hari', fontsize=7)
                    axes[i].set_ylabel('Frekuensi', fontsize=7)
                    axes[i].text(0.97, 0.92, f'skew={skv:.2f}', ha='right', va='top',
                                 transform=axes[i].transAxes, fontsize=7, color='#555')
                    sns.despine(ax=axes[i])
                for j in range(n, len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                fig_show(fig, "1 warna solid C_MID per histogram | skewness ditampilkan sebagai anotasi dalam grafik")

            sshdr("Bivariat - Scatter Fitur vs Stress_Level")
            if LIFESTYLE_FEAT and 'Stress_Level_num' in df_l.columns:
                n = len(LIFESTYLE_FEAT); ncols = 3; nrows = (n+ncols-1)//ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4*nrows))
                axes = axes.flatten()
                for i, col in enumerate(LIFESTYLE_FEAT):
                    valid = df_l[[col,'Stress_Level_num']].dropna()
                    r = valid[col].corr(valid['Stress_Level_num'])
                    axes[i].scatter(valid[col], valid['Stress_Level_num'],
                                    alpha=0.4, color=C_MID, s=15)
                    m, b_i = np.polyfit(valid[col], valid['Stress_Level_num'], 1)
                    x_ = np.linspace(valid[col].min(), valid[col].max(), 100)
                    axes[i].plot(x_, m*x_+b_i, color=C_DARK, linewidth=1.8, label=f'r={r:.3f}')
                    axes[i].legend(fontsize=7)
                    axes[i].set_title(SHORT_LS.get(col,col), fontsize=9, fontweight='bold')
                    axes[i].set_xlabel(SHORT_LS.get(col,col), fontsize=7)
                    axes[i].set_ylabel('Stress (1–3)', fontsize=7)
                    axes[i].set_yticks([1,2,3])
                    axes[i].set_yticklabels(['Low','Mod','High'], fontsize=7)
                    sns.despine(ax=axes[i])
                for j in range(n, len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                fig_show(fig, "Menggunakan Stress_Level_num (1=Low, 2=Moderate, 3=High) untuk operasi numerik")

            sshdr("Multivariat - Heatmap Korelasi")
            if LIFESTYLE_FEAT and 'Stress_Level_num' in df_l.columns:
                all_ls = LIFESTYLE_FEAT + ['Stress_Level_num']
                corr_ls = df_l[all_ls].corr()
                corr_ls = corr_ls.rename(index=SHORT_LS, columns=SHORT_LS)
                fig, ax = plt.subplots(figsize=(7.5, 5.5))
                sns.heatmap(corr_ls, annot=True, fmt='.2f', cmap=HEATMAP_DIV,
                            vmin=-1, vmax=1, center=0,
                            linewidths=1.5, linecolor='white', ax=ax, annot_kws={'size':10})
                ax.set_title('Heatmap Korelasi - Student Lifestyle')
                fig.tight_layout()
                fig_show(fig, "RdBu_r divergen | linewidths=1.5 mencegah ilusi optik kontras simultan (Tufte)")

    with tab3:
        if df_m.empty:
            st.warning("mbi_questions_clean.csv tidak ditemukan.")
        else:
            sshdr("Profiling")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Baris", f"{len(df_m):,}")
            c2.metric("Kolom", len(df_m.columns))
            c3.metric("Missing", int(df_m.isnull().sum().sum()))
            c4.metric("Duplikat", int(df_m.duplicated().sum()))
            info_box("Kolom <code>dim_exhaustion / dim_cynicism / dim_efficacy_inv</code> tidak ada di file clean. "
                     "Dimensi dihitung on-the-fly: <code>Exhaustion = mean(EX1–EX5)</code>, dst.")

            sshdr("Univariat - Item MBI per Dimensi")
            groups = [('Exhaustion (EX)',MBI_EX),('Cynicism (CY)',MBI_CY),('Efficacy (EF)',MBI_EF)]
            groups = [(l,c) for l,c in groups if c]
            if groups:
                fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4))
                if len(groups)==1: axes=[axes]
                for ax, (label, cols) in zip(axes, groups):
                    means = df_m[cols].mean().sort_values(ascending=False)
                    n = len(means)
                    clrs = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]
                    bars = ax.bar(means.index, means.values, color=clrs, edgecolor='white', linewidth=0.8, width=0.6)
                    for bar, v in zip(bars, means.values):
                        ax.text(bar.get_x()+bar.get_width()/2, v+0.04, f'{v:.2f}', ha='center', fontsize=7)
                    ax.set_ylim(0, means.max()*1.2)
                    ax.set_title(label, fontsize=9, fontweight='bold')
                    ax.set_ylabel('Rata-rata (0–6)', fontsize=8)
                    sns.despine(ax=ax)
                fig.tight_layout()
                fig_show(fig, "Gelap = skor tertinggi dalam dimensinya | gradasi dalam tiap dimensi")

            sshdr("Univariat - Rata-rata Dimensi MBI")
            dim_data3 = {}
            if MBI_EX: dim_data3['Exhaustion']    = df_m[MBI_EX].mean(axis=1)
            if MBI_CY: dim_data3['Cynicism']       = df_m[MBI_CY].mean(axis=1)
            if MBI_EF: dim_data3['Efficacy(inv)']  = df_m[MBI_EF].mean(axis=1)
            if 'fatigue_score' in df_m.columns: dim_data3['Fatigue Score'] = df_m['fatigue_score']
            if dim_data3:
                df_dim3 = pd.DataFrame(dim_data3)
                means_d = df_dim3.mean().sort_values(ascending=False)
                std_d   = df_dim3.std().reindex(means_d.index)
                n = len(means_d)
                clrs = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n-1,1))] for i in range(n)]
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                bars = axes[0].bar(means_d.index, means_d.values, color=clrs,
                                   edgecolor='white', linewidth=0.8, width=0.5)
                axes[0].errorbar(means_d.index, means_d.values, yerr=std_d.values,
                                 fmt='none', color='#334', capsize=4, linewidth=1)
                for bar, v in zip(bars, means_d.values):
                    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.12, f'{v:.3f}',
                                 ha='center', fontsize=8, fontweight='bold')
                axes[0].set_ylim(0, (means_d+std_d).max()+0.5)
                axes[0].set_ylabel('Rata-rata Skor (0–6)')
                axes[0].set_title('Rata-rata Dimensi MBI')
                sns.despine(ax=axes[0])
                bp_data = [df_dim3[c].dropna().values for c in means_d.index]
                bp = axes[1].boxplot(bp_data, labels=means_d.index, patch_artist=True,
                                     medianprops=dict(color='white', linewidth=2),
                                     flierprops=dict(marker='o', markersize=2, color='#556'))
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(clrs[i]); patch.set_alpha(0.88)
                axes[1].set_ylabel('Skor (0–6)')
                axes[1].set_title('Distribusi Dimensi MBI')
                sns.despine(ax=axes[1])
                fig.tight_layout()
                fig_show(fig, "Error bar = standar deviasi | warna identik dengan panel kiri (konsistensi kognitif)")

            sshdr("Bivariat - Dimensi per Risk Level")
            if dim_data3 and 'risk_level' in df_m.columns:
                df_dim_r = pd.DataFrame(dim_data3)
                df_dim_r['risk_level'] = df_m['risk_level'].values
                ra = [r for r in RISK_ORDER if r in df_dim_r['risk_level'].unique()]
                pal_r = {r: PALETTE[int(i*(len(PALETTE)-1)/max(len(ra)-1,1))] for i,r in enumerate(ra)}
                dim_cols_r = list(dim_data3.keys())
                ncols_r = 2; nrows_r = (len(dim_cols_r)+ncols_r-1)//ncols_r
                fig, axes = plt.subplots(nrows_r, ncols_r, figsize=(10, 4*nrows_r))
                axes = np.array(axes).flatten()
                for i, col in enumerate(dim_cols_r):
                    sns.boxplot(data=df_dim_r, x='risk_level', y=col,
                                hue='risk_level', order=ra, hue_order=ra,
                                palette=pal_r, legend=False, ax=axes[i], linewidth=0.8,
                                flierprops=dict(marker='o', markersize=2))
                    axes[i].set_title(col, fontsize=10, fontweight='bold')
                    axes[i].set_xlabel('Risk Level'); axes[i].set_ylabel('Skor (0–6)')
                    sns.despine(ax=axes[i])
                for j in range(len(dim_cols_r), len(axes)): axes[j].set_visible(False)
                fig.tight_layout()
                if 'fatigue_score' in dim_data3:
                    g_rk = [df_dim_r[df_dim_r['risk_level']==r]['fatigue_score'].dropna() for r in ra]
                    _, p_kw = stats.kruskal(*[g for g in g_rk if len(g)>1])
                    fig_show(fig, f"Kruskal-Wallis fatigue_score per risk level: p={p_kw:.4f} | {'Signifikan ✓' if p_kw<0.05 else 'Tidak'} (α=0.05)")
                else:
                    fig_show(fig)

            sshdr("Multivariat - Heatmap Korelasi Item MBI")
            all_items = [c for c in (MBI_EX+MBI_CY+MBI_EF+
                                     (['fatigue_score'] if 'fatigue_score' in df_m.columns else []))
                         if c in df_m.columns]
            if all_items:
                corr_mbi = df_m[all_items].corr()
                fig, ax = plt.subplots(figsize=(11, 9))
                sns.heatmap(corr_mbi, annot=True, fmt='.2f', cmap=HEATMAP_DIV,
                            vmin=-1, vmax=1, center=0,
                            linewidths=1.5, linecolor='white', ax=ax, annot_kws={'size':7})
                ax.set_title('Heatmap Korelasi Item MBI (EX/CY/EF)')
                fig.tight_layout()
                fig_show(fig, "Item satu dimensi (misal EX1-EX5) seharusnya berkorelasi tinggi satu sama lain → internal consistency")

    with tab4:
        if df_n.empty:
            st.warning("curhat_nlp_clean.csv tidak ditemukan.")
        else:
            sshdr("Profiling")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Baris", f"{len(df_n):,}")
            c2.metric("Kolom", len(df_n.columns))
            c3.metric("Missing", int(df_n.isnull().sum().sum()))
            c4.metric("Duplikat", int(df_n.duplicated().sum()))
            info_box("Kolom <code>word_count, char_count, stress_keyword_count</code> tidak ada di file clean. "
                     "Semua dihitung on-the-fly dari kolom <code>text_curhat</code>.")

            sshdr("Univariat - Distribusi Emosi")
            em_ord4 = [e for e in EMOTION_ORDER if e in df_n['emotion'].unique()]
            em_ord4 += [e for e in df_n['emotion'].unique() if e not in em_ord4]
            em_c4 = df_n['emotion'].value_counts().reindex(em_ord4)
            n4 = len(em_c4)
            clrs4 = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n4-1,1))] for i in range(n4)]

            col_l, col_r = st.columns(2)
            with col_l:
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                bars = ax.bar(em_c4.index, em_c4.values, color=clrs4,
                              edgecolor='white', linewidth=0.8, width=0.6)
                annotate_bars(ax, bars, '{:,.0f}', fontsize=8)
                ax.set_ylim(0, em_c4.max()*1.18)
                ax.set_xlabel('Emosi'); ax.set_ylabel('Jumlah Teks')
                ax.set_title('Distribusi Emosi')
                despine(ax); fig.tight_layout()
                fig_show(fig, "Sadness (gelap #0034c8) = paling berat | Joy (terang #00bcff) = paling ringan")
            with col_r:
                pie_clrs4 = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n4-1,1))] for i in range(n4)]
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                _, _, autotexts = ax.pie(
                    em_c4.values, labels=em_c4.index, autopct='%1.1f%%',
                    colors=pie_clrs4, startangle=90,
                    wedgeprops={'width':0.55,'edgecolor':'white','linewidth':2},
                    pctdistance=0.75, labeldistance=1.1
                )
                for at in autotexts:
                    at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
                ax.set_title('Proporsi Emosi (Donut)')
                fig.tight_layout()
                fig_show(fig, "Donut chart: maks 5 irisan | kontras tinggi antar warna")

            sshdr("Univariat - Histogram Metrik Teks")
            metric_cols4 = [c for c in ['word_count','char_count','stress_keyword_count'] if c in df_n.columns]
            if metric_cols4:
                fig, axes = plt.subplots(1, len(metric_cols4), figsize=(5*len(metric_cols4), 3.8))
                if len(metric_cols4)==1: axes=[axes]
                for i, col in enumerate(metric_cols4):
                    skv = df_n[col].skew()
                    axes[i].hist(df_n[col].dropna(), bins=30, color=C_MID, edgecolor='white', linewidth=0.8)
                    axes[i].set_title(col.replace('_',' ').title(), fontsize=9, fontweight='bold')
                    axes[i].set_xlabel('Nilai', fontsize=7); axes[i].set_ylabel('Frekuensi', fontsize=7)
                    axes[i].text(0.97, 0.92, f'skew={skv:.2f}', ha='right', va='top',
                                 transform=axes[i].transAxes, fontsize=7, color='#555')
                    sns.despine(ax=axes[i])
                fig.tight_layout()
                fig_show(fig, "Dihitung on-the-fly dari text_curhat | 1 warna solid C_MID (#007ff1)")

            sshdr("Bivariat - Rata-rata Metrik per Emosi")
            if metric_cols4:
                pal_em4 = {e: PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n4-1,1))] for i,e in enumerate(em_ord4)}
                fig, axes = plt.subplots(1, len(metric_cols4), figsize=(5*len(metric_cols4), 3.8))
                if len(metric_cols4)==1: axes=[axes]
                for i, col in enumerate(metric_cols4):
                    kw_m = df_n.groupby('emotion')[col].mean().reindex(em_ord4)
                    bars = axes[i].bar(kw_m.index, kw_m.values,
                                       color=[pal_em4[e] for e in kw_m.index],
                                       edgecolor='white', linewidth=0.8, width=0.6)
                    for bar, v in zip(bars, kw_m.values):
                        if v==v:
                            axes[i].text(bar.get_x()+bar.get_width()/2, v+0.01,
                                         f'{v:.1f}', ha='center', fontsize=8)
                    axes[i].set_ylim(0, kw_m.max()*1.18)
                    axes[i].set_xlabel('Emosi')
                    axes[i].set_ylabel(col.replace('_',' ').title())
                    axes[i].set_title(col.replace('_',' ').title(), fontsize=9, fontweight='bold')
                    sns.despine(ax=axes[i])
                fig.tight_layout()
                fig_show(fig, "Bar chart per emosi | Sadness & Anger diharapkan lebih tinggi dari Joy & Neutral")

            sshdr("Multivariat - Korelasi Metrik Teks")
            if len(metric_cols4) >= 2:
                corr_nlp = df_n[metric_cols4].corr()
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.heatmap(corr_nlp, annot=True, fmt='.3f', cmap=HEATMAP_DIV,
                            vmin=-1, vmax=1, center=0,
                            linewidths=1.5, linecolor='white', ax=ax, annot_kws={'size':11})
                ax.set_title('Korelasi Metrik Teks')
                fig.tight_layout()
                fig_show(fig, "word_count & char_count biasanya r>0.99 (redundan) | stress_keyword_count = fitur independen")


elif page == "A/B Testing":
    shdr("A/B Testing - Intervensi Generik vs Personal")
    info_box("<b>Konteks:</b> Menguji apakah notifikasi intervensi personal (berbasis fatigue score) "
             "menghasilkan conversion rate fitur wellness lebih tinggi secara statistik "
             "dibanding intervensi generik standar. Metode: Z-test proporsi two-sided, α=0.05.")

    ALPHA = 0.05
    np.random.seed(42)
    jml_A, jml_B = 1000, 1000
    konv_A = np.random.binomial(n=1, p=0.20, size=jml_A).sum()
    konv_B = np.random.binomial(n=1, p=0.28, size=jml_B).sum()
    r_A = konv_A / jml_A; r_B = konv_B / jml_B
    diff = r_B - r_A
    se   = np.sqrt((r_A*(1-r_A)/jml_A) + (r_B*(1-r_B)/jml_B))
    z_ci = stats.norm.ppf(0.975)
    ci_lo = diff - z_ci*se; ci_hi = diff + z_ci*se
    z_stat, p_val = proportions_ztest([konv_A, konv_B], [jml_A, jml_B], alternative='two-sided')
    chi2, p_chi2, _, _ = stats.chi2_contingency([[konv_A, jml_A-konv_A],[konv_B, jml_B-konv_B]])
    cohen_h = 2*np.arcsin(np.sqrt(r_B)) - 2*np.arcsin(np.sqrt(r_A))
    es_lbl  = 'Kecil' if abs(cohen_h)<0.2 else ('Menengah' if abs(cohen_h)<0.5 else 'Besar')
    rel_up  = (r_B/r_A - 1)*100
    signif  = p_val <= ALPHA

    sshdr("Desain Eksperimen (SMART)")
    with st.expander("Detail Desain A/B Testing"):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            **H₀:** Tidak ada perbedaan conversion rate antara A dan B (P_A = P_B)

            **H₁:** Terdapat perbedaan (P_A ≠ P_B) - *two-sided test*

            | Komponen | Keterangan |
            |---|---|
            | Grup A | Intervensi generik (notifikasi standar) |
            | Grup B | Intervensi personal (berdasarkan fatigue score) |
            | Alokasi | 50:50 - 1.000 mahasiswa per grup |
            | Metrik | Conversion rate fitur wellness |
            | α | 0.05 (tolak H₀ jika p ≤ 0.05) |
            """)
        with col_r:
            st.markdown("""
            **Tujuan SMART:**
            - *Specific*: Uji notifikasi personal vs generik
            - *Measurable*: Conversion rate fitur wellness (%)
            - *Achievable*: 2.000 sampel, 4 minggu
            - *Relevant*: Turunkan burnout mahasiswa HAPI
            - *Time-bound*: Periode eksperimen 28 hari

            **Distribusi data:** Binomial (konversi = 1 atau 0)
            """)

    sshdr("Hasil Statistik Utama")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f'<div class="ab-stat"><div class="ab-val">{r_A*100:.1f}%</div><div class="ab-lbl">Konversi Versi A</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="ab-stat"><div class="ab-val">{r_B*100:.1f}%</div><div class="ab-lbl">Konversi Versi B</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="ab-stat"><div class="ab-val">{p_val:.4f}</div><div class="ab-lbl">P-value (Z-test)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="ab-stat"><div class="ab-val">{cohen_h:.3f}</div><div class="ab-lbl">Cohen\'s h ({es_lbl})</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="ab-stat"><div class="ab-val">{rel_up:+.1f}%</div><div class="ab-lbl">Relative Uplift</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sshdr("Perbandingan Tingkat Konversi")
        labels = ['Versi A\n(Generik)', 'Versi B\n(Personal)']
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(labels, [r_A*100, r_B*100],
                      color=[C_LIGHT, C_DARK], edgecolor='white', linewidth=0.8, width=0.42)
        for bar, rate, count in zip(bars, [r_A, r_B], [konv_A, konv_B]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{rate*100:.1f}%\n({count})', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, max(r_A,r_B)*130)
        ax.set_ylabel('Tingkat Konversi (%)')
        ax.set_title('Konversi A vs B')
        ax.axhline(r_A*100, color=C_GREY, linestyle='--', linewidth=1, label='Baseline A')
        ax.legend(fontsize=8); despine(ax); fig.tight_layout()
        fig_show(fig, f"Versi A (#00bcff terang) = kontrol | Versi B (#0034c8 gelap) = perlakuan | selisih {diff*100:+.2f} pp")

    with col_r:
        sshdr("Distribusi Z & Zona Kritis")
        x_v = np.linspace(-4.5, 4.5, 500)
        y_v = stats.norm.pdf(x_v)
        z_crit = stats.norm.ppf(0.975)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(x_v, y_v, color=C_DARK, linewidth=2)
        x_l = x_v[x_v<=-z_crit]; x_r = x_v[x_v>=z_crit]
        x_m = x_v[(x_v>-z_crit)&(x_v<z_crit)]
        ax.fill_between(x_l, stats.norm.pdf(x_l), color=C_LIGHT, alpha=0.7, label='Zona tolak H₀')
        ax.fill_between(x_r, stats.norm.pdf(x_r), color=C_LIGHT, alpha=0.7)
        ax.fill_between(x_m, stats.norm.pdf(x_m), color=C_GREY, alpha=0.35, label='Zona gagal tolak H₀')
        ax.axvline(z_stat, color=C_DARK, linestyle='--', linewidth=2, label=f'Z = {z_stat:.3f}')
        ax.axvline(-z_crit, color=C_MID, linestyle=':', linewidth=1.2, label=f'±{z_crit:.2f}')
        ax.axvline(z_crit, color=C_MID, linestyle=':', linewidth=1.2)
        ax.set_xlabel('Z-score'); ax.set_ylabel('Densitas Probabilitas')
        ax.set_title('Distribusi Normal Standar')
        ax.legend(fontsize=7, loc='upper right'); despine(ax); fig.tight_layout()
        fig_show(fig, f"Biru terang = zona tolak H₀ | Abu-abu = gagal tolak | Garis gelap = Z observasi ({z_stat:.3f})")

    sshdr("95% Confidence Interval - Selisih Konversi (B − A)")
    fig, ax = plt.subplots(figsize=(10, 3))
    xerr_l = diff*100 - ci_lo*100; xerr_r = ci_hi*100 - diff*100
    ax.errorbar(x=diff*100, y=0, xerr=[[xerr_l],[xerr_r]],
                fmt='o', color=C_DARK, markersize=12,
                capsize=10, capthick=2.5, linewidth=2.5, label=f'Selisih: {diff*100:+.2f} pp')
    ax.fill_betweenx([-0.4,0.4], ci_lo*100, ci_hi*100, color=C_LIGHT, alpha=0.2,
                     label=f'95% CI [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]')
    ax.axvline(0, color=C_MID, linestyle='--', linewidth=1.5, label='H₀: tidak ada perbedaan')
    ax.text(diff*100, 0.22, f'{diff*100:+.2f} pp', ha='center', fontsize=12,
            fontweight='bold', color=C_DARK)
    ax.text(ci_lo*100-0.2, 0, f'{ci_lo*100:.2f}%', ha='right', va='center',
            fontsize=9, color=C_MID)
    ax.text(ci_hi*100+0.2, 0, f'{ci_hi*100:.2f}%', ha='left', va='center',
            fontsize=9, color=C_MID)
    ax.set_xlim(ci_lo*100-4, ci_hi*100+4); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel('Selisih Tingkat Konversi B − A (poin persentase)')
    ax.set_title('95% Confidence Interval untuk Selisih Proporsi')
    ax.legend(fontsize=9, loc='upper left'); despine(ax); fig.tight_layout()
    interp = "CI sepenuhnya positif → perbedaan signifikan" if ci_lo>0 else "CI mencakup 0 → tidak signifikan"
    fig_show(fig, f"{interp} | selisih titik = {diff*100:.2f} pp | CI: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")

    sshdr("Analisis Segmentasi per Tahun Akademik")
    np.random.seed(42)
    n_seg = 300
    segments = {
        'Tahun 1 (Freshman)':    {'A':0.16,'B':0.26},
        'Tahun 2-3 (Sophomore)': {'A':0.20,'B':0.30},
        'Tahun 4+ (Senior)':     {'A':0.24,'B':0.27},
    }
    seg_res = []
    for sn, params in segments.items():
        cA = np.random.binomial(1, params['A'], n_seg).sum()
        cB = np.random.binomial(1, params['B'], n_seg).sum()
        z_s, p_s = proportions_ztest([cA,cB],[n_seg,n_seg], alternative='two-sided')
        seg_res.append({'Segmen':sn,'Rate A (%)':round(cA/n_seg*100,1),
                        'Rate B (%)':round(cB/n_seg*100,1),
                        'Selisih (pp)':round((cB-cA)/n_seg*100,2),
                        'Z-stat':round(z_s,3),'P-value':round(p_s,4),
                        'Signifikan':'✅ Ya' if p_s<=ALPHA else '❌ Tidak'})
    df_seg = pd.DataFrame(seg_res)

    col_l, col_r = st.columns([1,1])
    with col_l:
        st.dataframe(df_seg.set_index('Segmen'), use_container_width=True)
    with col_r:
        seg_names = df_seg['Segmen'].tolist()
        x = np.arange(len(seg_names)); w = 0.32
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        b1 = axes[0].bar(x-w/2, df_seg['Rate A (%)'], w, label='Versi A',
                          color=C_LIGHT, edgecolor='white', linewidth=0.8)
        b2 = axes[0].bar(x+w/2, df_seg['Rate B (%)'], w, label='Versi B',
                          color=C_DARK, edgecolor='white', linewidth=0.8)
        for bar in list(b1)+list(b2):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         f'{bar.get_height():.1f}%', ha='center', fontsize=7)
        axes[0].set_xticks(x); axes[0].set_xticklabels(seg_names, fontsize=7)
        axes[0].set_ylabel('Conversion Rate (%)'); axes[0].legend(fontsize=8)
        axes[0].set_title('Konversi per Segmen'); sns.despine(ax=axes[0])
        clrs_p = [C_DARK if row['Signifikan']=='✅ Ya' else C_GREY for _, row in df_seg.iterrows()]
        bars_p = axes[1].bar(seg_names, df_seg['P-value'], color=clrs_p,
                              edgecolor='white', linewidth=0.8, width=0.5)
        for bar, row in zip(bars_p, seg_res):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                         f"{row['P-value']:.4f}", ha='center', fontsize=7)
        axes[1].axhline(ALPHA, color=C_MID, linestyle='--', linewidth=1.5, label=f'α={ALPHA}')
        axes[1].set_ylabel('P-value'); axes[1].legend(fontsize=8)
        axes[1].set_title('P-value per Segmen'); sns.despine(ax=axes[1])
        fig.tight_layout()
        fig_show(fig, "Gelap (#0034c8) = signifikan | Abu-abu (#d0d8e4) = tidak signifikan")

    sshdr("Analisis Sensitivitas Alpha")
    col_l, col_r = st.columns([1,2])
    with col_l:
        alpha_vals = [0.01, 0.05, 0.10]
        col1, col2, col3 = 'Alpha', 'Keputusan', 'Kesimpulan'
        st.markdown(f"**{col1:>6} | {col2:>18} | {col3}**")
        for a in alpha_vals:
            kep = 'Tolak H₀' if p_val<=a else 'Gagal Tolak H₀'
            kes = 'Signifikan' if p_val<=a else 'Tidak signifikan'
            marker = ' ← digunakan' if a==ALPHA else ''
            st.markdown(f"`{a:.2f}` | `{kep}` | {kes}{marker}")

    sshdr("Keputusan Statistik & Rekomendasi Bisnis")
    if signif:
        st.success(f"""
        **Tolak H₀** - p={p_val:.4f} ≤ α={ALPHA}

        Terdapat perbedaan **signifikan secara statistik**. Versi B (Intervensi Personal)
        lebih efektif dengan uplift **{diff*100:+.2f} pp** ({rel_up:+.1f}% relative).

        **Rekomendasi:** Rollout Versi B secara bertahap, mulai dari segmen dengan efek terbesar.
        """)
    else:
        st.warning(f"""
        **Gagal Menolak H₀** - p={p_val:.4f} > α={ALPHA}

        Tidak ada bukti statistik yang cukup kuat.

        **Rekomendasi:** Pertahankan Versi A. Pertimbangkan tambah sampel, analisis segmentasi,
        atau revisi desain intervensi Versi B.
        """)

    with st.expander("Detail Lengkap Hasil Statistik"):
        c1,c2 = st.columns(2)
        with c1:
            st.write(f"Z-statistik: `{z_stat:.4f}`")
            st.write(f"P-value Z-test: `{p_val:.4f}`")
            st.write(f"P-value Chi-square: `{p_chi2:.4f}`")
            st.write(f"Z-test & Chi-square konsisten: **{'Ya ✅' if (p_val<ALPHA)==(p_chi2<ALPHA) else 'Tidak ⚠️'}**")
        with c2:
            st.write(f"95% CI selisih: `[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]`")
            st.write(f"Cohen's h: `{cohen_h:.4f}` - {es_lbl}")
            st.write(f"Relative uplift: `{rel_up:+.1f}%`")
            st.write(f"Np.random.seed: `42` (reproducible)")


elif page == "Rekomendasi & Self-Check":
    shdr("Rekomendasi Intervensi & Fatigue Self-Check")
    info_box("5 rekomendasi dirumuskan berdasarkan temuan <b>BQ15</b> (profil mahasiswa paling berisiko) "
             "dan <b>A/B Testing</b> (efektivitas intervensi personal vs generik).")

    sshdr("5 Rekomendasi Intervensi Berbasis Data")
    REKOS = [
        ("01","Jam Tidur","Alert Tidur Adaptif","BQ4 + BQ7",
         "Kirimkan notifikasi personal saat mahasiswa terdeteksi tidur <6 jam 3 hari berturut-turut. "
         "Pesan disesuaikan dengan pola tidur mingguan individual."),
        ("02","Jam Belajar","Teknik Belajar Pomodoro","BQ6 + BQ11",
         "Sesi belajar 90 menit + 20 menit istirahat untuk mahasiswa yang belajar >8 jam/hari. "
         "Burnout score mahasiswa dengan pola ini secara statistik lebih tinggi."),
        ("03","Sosial","Koneksi Social Support","BQ14 + BQ15",
         "Fasilitasi sesi peer-support mingguan untuk mahasiswa dengan social_support rendah (Q1). "
         "Perbedaan burnout Q1 vs Q4 terbukti signifikan (Cohen's d > 0.3)."),
        ("04","Fisik","Tantangan Aktivitas Fisik","BQ9 + BQ13",
         "Gamifikasi aktivitas fisik: 30 menit/hari × 5 hari/minggu. "
         "Mahasiswa aktif fisik menunjukkan korelasi negatif dengan burnout score (Spearman signifikan)."),
        ("05","MBI","Kuis MBI Periodik","BQ5 + BQ12 + A/B"),
    ]

    c1, c2, c3 = st.columns(3)
    for col, reko in zip([c1,c2,c3], REKOS[:3]):
        no, icon, title, basis, desc = reko
        col.markdown(f"""
        <div class="reko-card">
          <div class="reko-num">{no}</div>
          <div class="reko-ttl">{title}</div>
          <div class="reko-src">Sumber: {basis}</div>
          <div class="reko-dsc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, _ = st.columns(3)
    no4, icon4, title4, basis4, desc4 = REKOS[3]
    c4.markdown(f"""
    <div class="reko-card">
      <div class="reko-num">{no4}</div>
      <div class="reko-ttl">{title4}</div>
      <div class="reko-src">Sumber: {basis4}</div>
      <div class="reko-dsc">{desc4}</div>
    </div>""", unsafe_allow_html=True)

    np.random.seed(42)
    konv_A_r = np.random.binomial(n=1, p=0.20, size=1000).sum()
    konv_B_r = np.random.binomial(n=1, p=0.28, size=1000).sum()
    _, p_ab_r = proportions_ztest([konv_A_r, konv_B_r], [1000, 1000], alternative='two-sided')
    uplift_r  = (konv_B_r/1000 - konv_A_r/1000) / (konv_A_r/1000) * 100
    c5.markdown(f"""
    <div class="reko-card">
      <div class="reko-num">05</div>
      <div class="reko-ttl">Kuis MBI Periodik</div>
      <div class="reko-src">Sumber: BQ5 + BQ12 + A/B Testing</div>
      <div class="reko-dsc">Kirimkan kuis MBI setiap 4 minggu. Hasil dikompilasi menjadi laporan
      fatigue score personal dibandingkan profil peer-group. Intervensi personal terbukti
      +{uplift_r:.0f}% uplift vs generik (p={p_ab_r:.4f}).</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sshdr("Profil Mahasiswa Paling Berisiko Burnout Berat")
    if not df_b.empty and 'risk_level' in df_b.columns:
        pcols = [c for c in ['study_hours_per_day','sleep_hours','physical_activity',
                              'social_support','screen_time','exam_pressure'] if c in df_b.columns]
        if pcols:
            ph = df_b[df_b['risk_level']=='High'][pcols].mean()
            pl = df_b[df_b['risk_level']=='Low'][pcols].mean()
            n_feat = len(pcols); ncols = 3; nrows = (n_feat+ncols-1)//ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8*nrows))
            axes = axes.flatten()
            for i, col in enumerate(pcols):
                axes[i].bar(['Low Risk','High Risk'], [pl[col], ph[col]],
                            color=[C_LIGHT, C_DARK], edgecolor='white', linewidth=0.8, width=0.5)
                ymax = max(pl[col], ph[col])
                for j, v in enumerate([pl[col], ph[col]]):
                    axes[i].text(j, v+ymax*0.04, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
                axes[i].set_title(col.replace('_',' ').title(), fontsize=8, fontweight='bold')
                axes[i].set_ylim(0, ymax*1.22)
                sns.despine(ax=axes[i])
            for j in range(n_feat, len(axes)): axes[j].set_visible(False)
            fig.tight_layout()
            fig_show(fig, "Terang (#00bcff) = Low Risk | Gelap (#0034c8) = High Risk | semua nilai dianotasi")

    st.markdown("<br>", unsafe_allow_html=True)
    sshdr("Fatigue Self-Check - Kuis MBI Singkat")
    info_box("Isi kuis singkat ini untuk mendapatkan estimasi fatigue score berbasis MBI-SS "
             "(Maslach Burnout Inventory - Student Survey). Skala: 0 = Tidak Pernah, 6 = Selalu.")

    with st.form("mbi_form"):
        st.markdown("**Dimensi Exhaustion (Kelelahan Emosional)**")
        c1,c2 = st.columns(2)
        ex1 = c1.slider("Saya merasa emosionally terkuras akibat studi", 0, 6, 3, key='ex1')
        ex2 = c2.slider("Saya merasa kelelahan di akhir hari kuliah", 0, 6, 3, key='ex2')
        ex3 = c1.slider("Saya merasa lelah ketika bangun pagi dan harus kuliah lagi", 0, 6, 3, key='ex3')
        ex4 = c2.slider("Belajar seharian benar-benar membuat stres", 0, 6, 2, key='ex4')
        ex5 = c1.slider("Saya merasa sudah di titik jenuh dengan studi", 0, 6, 2, key='ex5')

        st.markdown("**Dimensi Cynicism (Depersonalisasi)**")
        c3,c4 = st.columns(2)
        cy1 = c3.slider("Saya menjadi tidak peduli dengan perkuliahan", 0, 6, 2, key='cy1')
        cy2 = c4.slider("Saya meragukan makna dan nilai dari perkuliahan", 0, 6, 2, key='cy2')
        cy3 = c3.slider("Saya kesulitan antusias dengan studi saya", 0, 6, 2, key='cy3')
        cy4 = c4.slider("Saya merasa semakin sinis tentang kegunaan studi saya", 0, 6, 2, key='cy4')

        st.markdown("**Dimensi Efficacy (Rasa Tidak Mampu - dibalik)**")
        c5,c6 = st.columns(2)
        ef1 = c5.slider("Saya dapat memecahkan masalah studi dengan efektif", 0, 6, 4, key='ef1')
        ef2 = c6.slider("Saya memberikan kontribusi positif di lingkungan akademik", 0, 6, 4, key='ef2')
        ef3 = c5.slider("Menurut saya, saya adalah mahasiswa yang baik", 0, 6, 4, key='ef3')
        ef4 = c6.slider("Saya merasa bersemangat ketika mencapai tujuan akademik", 0, 6, 4, key='ef4')
        ef5 = c5.slider("Saya telah menyelesaikan banyak hal berharga dalam studi", 0, 6, 4, key='ef5')
        ef6 = c6.slider("Saya percaya diri dalam menangani tantangan akademik", 0, 6, 4, key='ef6')

        submitted = st.form_submit_button("Hitung Fatigue Score Saya")

    if submitted:
        dim_ex = np.mean([ex1,ex2,ex3,ex4,ex5])
        dim_cy = np.mean([cy1,cy2,cy3,cy4])
        dim_ef_inv = 6 - np.mean([ef1,ef2,ef3,ef4,ef5,ef6])
        fatigue = (dim_ex + dim_cy + dim_ef_inv) / 3

        if fatigue < 2.0:
            level, col_sc, msg = "Rendah", C_LIGHT, "Anda menunjukkan tanda-tanda burnout yang minimal. Pertahankan kebiasaan belajar dan istirahat yang sehat."
        elif fatigue < 3.5:
            level, col_sc, msg = "Sedang", C_MID, "Ada beberapa tanda kelelahan. Pertimbangkan untuk mengurangi beban belajar dan meningkatkan aktivitas sosial."
        else:
            level, col_sc, msg = "Tinggi", C_DARK, "Anda berada di zona berisiko burnout. Sangat disarankan untuk berbicara dengan konselor atau mengurangi beban akademik."

        st.markdown(f"""
        <div style="background:white;border:1px solid #e8ecf4;border-radius:12px;padding:20px;
             border-left:6px solid {col_sc};">
          <div style="font-size:22px;font-weight:600;color:{col_sc};">
            Fatigue Score: {fatigue:.2f} / 6.00</div>
          <div style="font-size:16px;font-weight:600;color:#0d1b4b;margin:6px 0;">
            Tingkat Kelelahan: {level}</div>
          <div style="font-size:13px;color:#7a8299;">{msg}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        dim_vals = [dim_ex, dim_cy, dim_ef_inv]
        dim_lbls = ['Exhaustion\n(Kelelahan)', 'Cynicism\n(Sinisme)', 'Efficacy Inv.\n(Ketidakmampuan)']
        n_dim = len(dim_vals)
        clrs_dim = [PALETTE_R[int(i*(len(PALETTE_R)-1)/max(n_dim-1,1))] for i in range(n_dim)]

        col_l, col_r = st.columns(2)
        with col_l:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars_d = ax.bar(dim_lbls, dim_vals, color=clrs_dim,
                            edgecolor='white', linewidth=0.8, width=0.5)
            for bar, v in zip(bars_d, dim_vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.2f}',
                        ha='center', fontsize=11, fontweight='bold')
            ax.axhline(3.5, color=C_LIGHT, linestyle='--', linewidth=1.2, label='Batas risiko (3.5)')
            ax.set_ylim(0, 7); ax.set_ylabel('Skor Dimensi (0–6)')
            ax.set_title('Hasil Fatigue Self-Check per Dimensi')
            ax.legend(fontsize=8); despine(ax); fig.tight_layout()
            fig_show(fig)

        with col_r:
            st.markdown("**Detail Skor:**")
            st.metric("Exhaustion (Kelelahan)", f"{dim_ex:.2f}")
            st.metric("Cynicism (Sinisme)", f"{dim_cy:.2f}")
            st.metric("Efficacy Inv. (Ketidakmampuan)", f"{dim_ef_inv:.2f}")
            st.metric("Fatigue Score Akhir", f"{fatigue:.2f} / 6.00")

            if level == "Tinggi":
                st.error("Rekomendasi: segera kurangi beban akademik dan hubungi konselor kampus.")
            elif level == "Sedang":
                st.warning("Rekomendasi: terapkan teknik Pomodoro dan perkuat social support.")
            else:
                st.success("Bagus! Pertahankan pola tidur dan aktivitas fisik yang sehat.")

