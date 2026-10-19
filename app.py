"""
Lab Data Preparation · aplikasi simulasi untuk pembelajaran
Jalankan:  streamlit run app.py
"""
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Lab Data Preparation", page_icon="🧪", layout="wide")

WARNA = ["#1f5f8b", "#e8a93a", "#1f7a8c", "#c0503b", "#8da9c4", "#4f8a6b", "#7a5c99", "#5c6b7a"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = WARNA

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #f4f7fb; border-radius: 10px; padding: 10px 14px;}
.catatan {background:#fdf6e7; border-radius:8px; padding:10px 14px; font-size:0.92rem; margin: 4px 0 12px 0;}
</style>""", unsafe_allow_html=True)


# =============================================================================
# DATA CONTOH (fiktif)
# =============================================================================
@st.cache_data
def contoh_wajib_pajak(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    sektor = rng.choice(["Perdagangan", "Jasa", "Manufaktur", "Konstruksi", "Pertambangan"], n, p=[.35, .3, .15, .12, .08])
    wilayah = rng.choice(["Balikpapan", "Samarinda", "Bontang", "Tarakan", "Berau"], n)
    omzet = np.round(np.exp(rng.normal(21, 1.3, n)), -3)
    karyawan = np.maximum(1, rng.poisson(np.clip(omzet / 3e8, 1, 200))).astype(float)
    umur = rng.integers(0, 35, n).astype(float)
    terlambat = rng.poisson(1.2 + 0.8 * (sektor == "Konstruksi"))
    rasio = np.clip(rng.normal(0.85, 0.15, n) - 0.05 * terlambat, 0, 1.2).round(3)
    z = -2.4 + 0.55 * terlambat - 4 * (rasio - 0.85) + 0.5 * (sektor == "Konstruksi") - 0.02 * umur + rng.normal(0, .5, n)
    status = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "tidak_patuh", "patuh")
    df = pd.DataFrame({"npwp_id": [f"WP-{i:05d}" for i in range(1, n + 1)], "sektor": sektor, "wilayah": wilayah,
                       "omzet": omzet, "jumlah_karyawan": karyawan, "umur_usaha": umur,
                       "terlambat_lapor": terlambat, "rasio_bayar": rasio, "status": status})
    for kol, p in [("rasio_bayar", .08), ("jumlah_karyawan", .05), ("sektor", .03), ("umur_usaha", .04)]:
        df.loc[rng.choice(n, int(p * n), replace=False), kol] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "umur_usaha"] = 150
    return pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)


@st.cache_data
def contoh_properti(n=1000, seed=11):
    rng = np.random.default_rng(seed)
    wilayah = rng.choice(["Pusat", "Utara", "Selatan", "Timur", "Barat"], n, p=[.2, .2, .25, .2, .15])
    tipe = rng.choice(["Rumah", "Apartemen", "Ruko"], n, p=[.55, .3, .15])
    luas = np.round(rng.gamma(4, 25, n) + 30)
    kamar = np.clip(np.round(luas / 35 + rng.normal(0, .8, n)), 1, 8)
    jarak = np.round(rng.gamma(2, 3, n), 1)
    umur = rng.integers(0, 40, n).astype(float)
    efek_wil = pd.Series(wilayah).map({"Pusat": 1.5, "Utara": 1.0, "Selatan": 1.2, "Timur": .9, "Barat": .95}).to_numpy()
    efek_tipe = pd.Series(tipe).map({"Rumah": 1.0, "Apartemen": 1.1, "Ruko": 1.4}).to_numpy()
    harga = (8 * luas + 60 * kamar - 15 * jarak - 4 * umur + 200) * efek_wil * efek_tipe * rng.normal(1, .1, n)
    df = pd.DataFrame({"wilayah": wilayah, "tipe": tipe, "luas_m2": luas, "kamar": kamar, "jarak_pusat_km": jarak,
                       "umur_bangunan": umur, "harga_juta": np.round(harga, 1)})
    df.loc[rng.choice(n, 60, replace=False), "luas_m2"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "jarak_pusat_km"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "tipe"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "harga_juta"] *= 12        # outlier ekstrem
    return df


CONTOH = {"Wajib pajak (fiktif, klasifikasi)": contoh_wajib_pajak,
          "Harga properti (fiktif, regresi)": contoh_properti}


# =============================================================================
# FUNGSI DATA PREPARATION
# =============================================================================
def baca_csv(isi: bytes, sep, desimal, ribuan):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            kw = dict(sep=sep, decimal=desimal, encoding=enc)
            if ribuan:
                kw["thousands"] = ribuan
            if sep is None:
                kw["engine"] = "python"
            return pd.read_csv(io.BytesIO(isi), **kw), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Encoding file tidak dikenali.")


def kolom_numerik(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def kolom_kategori(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]


def kolom_tanggal(df):
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def teks_ke_angka(s: pd.Series, format_angka: str) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(r"[^\d,.\-]", "", regex=True)
    if format_angka.startswith("Indonesia"):
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    return pd.to_numeric(t.replace({"": np.nan, "-": np.nan}), errors="coerce")


POLA_ANGKA = re.compile(r"^-?[\d.,]+$")


def kandidat_numerik(df, format_angka, placeholder=("-", "", "NA", "N/A", "null", "?")):
    """Kolom teks yang >= 90% nilainya berbentuk angka (boleh diawali Rp/IDR/$ atau diakhiri %)."""
    hasil = []
    for c in kolom_kategori(df):
        s = df[c].dropna().astype(str).str.strip()
        s = s[~s.isin(placeholder)]
        if len(s) == 0:
            continue
        bersih = (s.str.replace(r"(?i)^(rp|idr|\$)\.?\s*", "", regex=True)
                   .str.replace("%", "", regex=False).str.replace(" ", "", regex=False))
        if bersih.str.match(POLA_ANGKA).mean() >= 0.9:
            hasil.append(c)
    return hasil


def profil_kualitas(df):
    baris = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        num = c in kolom_numerik(df)
        info = {"kolom": c, "tipe": str(s.dtype), "n_kosong": int(s.isna().sum()),
                "pct_kosong": round(100 * s.isna().mean(), 2), "n_unik": int(s.nunique()),
                "pct_unik": round(100 * s.nunique() / max(n, 1), 2)}
        if num and s.notna().any():
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            info.update({"min": s.min(), "median": s.median(), "max": s.max(), "skew": round(s.skew(), 2),
                         "n_outlier_iqr": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                         "n_nol": int((s == 0).sum())})
        else:
            top = s.mode()
            info.update({"nilai_terbanyak": top.iloc[0] if len(top) else None})
        catatan = []
        if info["pct_kosong"] > 30:
            catatan.append("banyak kosong")
        if info["n_unik"] <= 1:
            catatan.append("konstan")
        if not num and info["pct_unik"] > 90 and n > 20:
            catatan.append("mirip ID")
        if num and abs(info.get("skew", 0) or 0) > 1:
            catatan.append("menceng")
        if not num and 20 < info["n_unik"] and info["pct_unik"] <= 90:
            catatan.append("kardinalitas tinggi")
        info["catatan"] = ", ".join(catatan)
        baris.append(info)
    return pd.DataFrame(baris)


def skor_kualitas(df):
    n_sel = df.size or 1
    kelengkapan = 100 * (1 - df.isna().sum().sum() / n_sel)
    keunikan = 100 * (1 - df.duplicated().mean()) if len(df) else 100
    prof = profil_kualitas(df)
    kolom_bermasalah = (prof["catatan"] != "").sum()
    return kelengkapan, keunikan, kolom_bermasalah


def batas_outlier(s, metode, param):
    s = s.dropna()
    if metode == "IQR":
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        return q1 - param * iqr, q3 + param * iqr
    if metode == "Z-score":
        return s.mean() - param * s.std(), s.mean() + param * s.std()
    # Modified Z-score (median dan MAD)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s.min(), s.max()
    return med - param * mad / 0.6745, med + param * mad / 0.6745


def catat(langkah):
    st.session_state.log.append(f"{datetime.now():%H:%M:%S} · {langkah}")


def set_data_kerja(df_baru, langkah):
    st.session_state.df = df_baru.reset_index(drop=True)
    catat(langkah)


# =============================================================================
# FUNGSI PEMODELAN
# =============================================================================
def daftar_model(tugas):
    if tugas == "Regresi":
        return {"Linear Regression": LinearRegression, "Ridge Regression": Ridge,
                "Decision Tree": DecisionTreeRegressor, "Random Forest": RandomForestRegressor,
                "K-Nearest Neighbors": KNeighborsRegressor}
    return {"Logistic Regression": LogisticRegression, "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier, "K-Nearest Neighbors": KNeighborsClassifier,
            "Naive Bayes": GaussianNB}


def buat_estimator(tugas, nama, prm, seimbang):
    kelas = daftar_model(tugas)[nama]
    cw = "balanced" if seimbang else None
    if nama == "Ridge Regression":
        return kelas(alpha=prm["alpha"])
    if nama == "Logistic Regression":
        return kelas(C=prm["C"], max_iter=2000, class_weight=cw)
    if nama == "Decision Tree":
        kw = dict(max_depth=prm["max_depth"] or None, random_state=prm["seed"])
        return kelas(class_weight=cw, **kw) if tugas == "Klasifikasi" else kelas(**kw)
    if nama == "Random Forest":
        kw = dict(n_estimators=prm["n_estimators"], max_depth=prm["max_depth"] or None, random_state=prm["seed"], n_jobs=-1)
        return kelas(class_weight=cw, **kw) if tugas == "Klasifikasi" else kelas(**kw)
    if nama == "K-Nearest Neighbors":
        return kelas(n_neighbors=prm["k"])
    return kelas()


def buat_pipeline(estimator, num, kat, skala, imputasi_num):
    langkah_num = [("imputasi", SimpleImputer(strategy=imputasi_num))]
    if skala == "StandardScaler":
        langkah_num.append(("skala", StandardScaler()))
    elif skala == "MinMaxScaler":
        langkah_num.append(("skala", MinMaxScaler()))
    transformer = []
    if num:
        transformer.append(("num", Pipeline(langkah_num), num))
    if kat:
        transformer.append(("kat", Pipeline([
            ("imputasi", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False)),
        ]), kat))
    prep = ColumnTransformer(transformer, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", estimator)])


def importansi(pipe, X_test, y_test, tugas, seed):
    model = pipe.named_steps["model"]
    nama = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=nama), "Feature importance (model pohon)"
    if hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        return pd.Series(np.abs(coef).mean(axis=0), index=nama), "|Koefisien| (fitur terstandar bila diskalakan)"
    skor = "r2" if tugas == "Regresi" else "accuracy"
    r = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring=skor)
    return pd.Series(r.importances_mean, index=X_test.columns), "Permutation importance"


# =============================================================================
# STATE
# =============================================================================
for k, v in {"df": None, "df_asli": None, "sumber": None, "log": [], "hasil_model": None}.items():
    st.session_state.setdefault(k, v)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🧪 Lab Data Preparation")
    st.caption("Simulasi data preparation dan pemodelan sederhana")
    sumber = st.radio("Sumber data", ["Unggah CSV", "Data contoh"], horizontal=True)
    if sumber == "Unggah CSV":
        berkas = st.file_uploader("Pilih file CSV", type=["csv", "txt"])
        with st.expander("Opsi pembacaan"):
            pilihan_sep = {"Deteksi otomatis": None, "Koma (,)": ",", "Titik koma (;)": ";", "Pipa (|)": "|", "Tab": "\t"}
            sep = pilihan_sep[st.selectbox("Pemisah kolom", list(pilihan_sep))]
            desimal = st.selectbox("Pemisah desimal", [".", ","])
            ribuan = st.selectbox("Pemisah ribuan", ["(tidak ada)", ".", ","])
            ribuan = None if ribuan == "(tidak ada)" else ribuan
        if berkas is not None:
            kunci = f"upload:{berkas.name}:{berkas.size}:{sep}:{desimal}:{ribuan}"
            if st.session_state.sumber != kunci:
                try:
                    data, enc = baca_csv(berkas.getvalue(), sep, desimal, ribuan)
                    st.session_state.update(df=data, df_asli=data.copy(), sumber=kunci, log=[], hasil_model=None)
                    catat(f"Memuat {berkas.name} ({data.shape[0]} baris × {data.shape[1]} kolom, encoding {enc})")
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
    else:
        nama_contoh = st.selectbox("Pilih data contoh", list(CONTOH))
        kunci = f"contoh:{nama_contoh}"
        if st.session_state.sumber != kunci:
            data = CONTOH[nama_contoh]()
            st.session_state.update(df=data.copy(), df_asli=data.copy(), sumber=kunci, log=[], hasil_model=None)
            catat(f"Memuat data contoh: {nama_contoh}")

    st.divider()
    menu = st.radio("Menu", ["1 · Data & Tipe", "2 · Missing Value", "3 · Imputasi", "4 · EDA & Kualitas",
                             "5 · Outlier", "6 · Pemodelan", "7 · Riwayat & Unduh"])
    if st.session_state.df is not None:
        st.divider()
        d = st.session_state.df
        st.caption(f"Data kerja: **{d.shape[0]:,} baris × {d.shape[1]} kolom**")
        st.caption(f"Langkah tercatat: {max(len(st.session_state.log) - 1, 0)}")
        if st.button("↺ Kembalikan ke data asli", width="stretch"):
            st.session_state.df = st.session_state.df_asli.copy()
            st.session_state.hasil_model = None
            catat("Data dikembalikan ke kondisi asli")
            st.rerun()

df = st.session_state.df
if df is None:
    st.title("Lab Data Preparation")
    st.info("Unggah file CSV di panel kiri, atau pilih **Data contoh** untuk mulai bersimulasi.")
    st.stop()

NUM, KAT, TGL = kolom_numerik(df), kolom_kategori(df), kolom_tanggal(df)

# =============================================================================
# 1. DATA & TIPE
# =============================================================================
if menu.startswith("1"):
    st.title("Data & Tipe Kolom")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Baris", f"{len(df):,}")
    k2.metric("Kolom", df.shape[1])
    k3.metric("Numerik", len(NUM))
    k4.metric("Kategorikal / teks", len(KAT))
    k5.metric("Baris duplikat", int(df.duplicated().sum()))

    st.subheader("Cuplikan data")
    st.dataframe(df.head(50), width="stretch", height=300)

    st.subheader("Tipe data per kolom")
    tipe = pd.DataFrame({"tipe": df.dtypes.astype(str), "contoh nilai": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
                         "jenis": ["numerik" if c in NUM else "tanggal" if c in TGL else "kategorikal/teks" for c in df.columns]})
    st.dataframe(tipe.astype(str), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Teks angka → numerik")
        st.markdown('<div class="catatan">Kolom angka yang tersimpan sebagai teks (misalnya "Rp 1.250.000") tidak bisa dihitung. Pilih format penulisan angkanya lalu konversi.</div>', unsafe_allow_html=True)
        fmt = st.radio("Format angka", ["Indonesia (1.234,56)", "Internasional (1,234.56)"], horizontal=True)
        kandidat = kandidat_numerik(df, fmt)
        pilih = st.multiselect("Kolom yang dikonversi", KAT, default=kandidat,
                               help="Kolom yang terdeteksi ≥90% bisa dibaca sebagai angka sudah dipilih otomatis.")
        if pilih:
            pratinjau = pd.DataFrame({f"{c} (asli)": df[c].head(5).astype(str) for c in pilih[:2]} |
                                     {f"{c} (hasil)": teks_ke_angka(df[c].head(5), fmt) for c in pilih[:2]})
            st.dataframe(pratinjau, width="stretch")
        if st.button("Konversi ke numerik", disabled=not pilih):
            baru = df.copy()
            gagal = {}
            for c in pilih:
                hasil = teks_ke_angka(baru[c], fmt)
                gagal[c] = int(hasil.isna().sum() - baru[c].isna().sum())
                baru[c] = hasil
            set_data_kerja(baru, f"Konversi teks → numerik ({fmt.split()[0]}): {', '.join(pilih)}")
            st.success("Konversi selesai. Nilai yang gagal dibaca menjadi NaN: " + ", ".join(f"{k}={v}" for k, v in gagal.items()))
            st.rerun()

    with c2:
        st.subheader("Bersihkan nilai placeholder dan duplikat")
        placeholder = st.text_input("Teks yang dianggap kosong (pisahkan koma)", "-, ?, NA, N/A, null, kosong")
        if st.button("Ganti placeholder menjadi NaN"):
            daftar = [p.strip() for p in placeholder.split(",") if p.strip()] + [""]
            baru = df.copy()
            sebelum = int(baru.isna().sum().sum())
            for c in kolom_kategori(baru):
                baru[c] = baru[c].where(~baru[c].astype(str).str.strip().isin(daftar), np.nan)
            set_data_kerja(baru, f"Placeholder {daftar[:-1]} diganti NaN (+{int(baru.isna().sum().sum()) - sebelum} sel)")
            st.rerun()
        if st.button(f"Hapus {int(df.duplicated().sum())} baris duplikat", disabled=not df.duplicated().any()):
            n0 = len(df)
            set_data_kerja(df.drop_duplicates(), f"Hapus baris duplikat ({n0 - len(df.drop_duplicates())} baris)")
            st.rerun()

        st.subheader("Kolom tanggal")
        kol_tgl = st.selectbox("Konversi kolom teks ke tanggal", ["(pilih)"] + KAT)
        ekstrak = st.checkbox("Sekaligus buat fitur tahun, bulan, dan hari", value=True)
        if st.button("Konversi tanggal", disabled=kol_tgl == "(pilih)"):
            baru = df.copy()
            baru[kol_tgl] = pd.to_datetime(baru[kol_tgl], errors="coerce", dayfirst=True, format="mixed")
            if ekstrak:
                baru[f"{kol_tgl}_tahun"] = baru[kol_tgl].dt.year
                baru[f"{kol_tgl}_bulan"] = baru[kol_tgl].dt.month
                baru[f"{kol_tgl}_hari"] = baru[kol_tgl].dt.dayofweek
            set_data_kerja(baru, f"Konversi {kol_tgl} ke tanggal" + (" + ekstrak fitur" if ekstrak else ""))
            st.rerun()

        st.subheader("Hapus kolom")
        hapus = st.multiselect("Kolom yang tidak dipakai", list(df.columns))
        if st.button("Hapus kolom terpilih", disabled=not hapus):
            set_data_kerja(df.drop(columns=hapus), f"Hapus kolom: {', '.join(hapus)}")
            st.rerun()

# =============================================================================
# 2. MISSING VALUE
# =============================================================================
elif menu.startswith("2"):
    st.title("Identifikasi Missing Value")
    total = int(df.isna().sum().sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sel kosong", f"{total:,}")
    k2.metric("% sel kosong", f"{100 * total / max(df.size, 1):.2f}%")
    k3.metric("Kolom berisi NaN", int((df.isna().sum() > 0).sum()))
    k4.metric("Baris berisi NaN", f"{int(df.isna().any(axis=1).sum()):,}")

    if total == 0:
        st.success("Tidak ada nilai kosong. Cek menu **Data & Tipe** bila ada placeholder seperti '-' atau 'NA' yang belum dikenali.")
    else:
        ringkas = (pd.DataFrame({"n_kosong": df.isna().sum(), "pct_kosong": (100 * df.isna().mean()).round(2),
                                 "tipe": df.dtypes.astype(str)}).query("n_kosong > 0").sort_values("pct_kosong", ascending=False))
        c1, c2 = st.columns([1, 1.3])
        c1.dataframe(ringkas, width="stretch")
        c2.plotly_chart(px.bar(ringkas.reset_index(names="kolom"), x="pct_kosong", y="kolom", orientation="h",
                               title="Persentase kosong per kolom", labels={"pct_kosong": "% kosong", "kolom": ""}),
                        width="stretch")

        st.subheader("Peta nilai kosong")
        n_tampil = st.slider("Jumlah baris yang ditampilkan", 50, min(len(df), 2000), min(len(df), 300), 50)
        peta = df.head(n_tampil).isna().astype(int)
        fig = px.imshow(peta.T, aspect="auto", color_continuous_scale=[[0, "#eef2f7"], [1, "#c0503b"]],
                        labels={"x": "baris", "y": "kolom", "color": "kosong"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pola kosong bersamaan")
        kol_na = ringkas.index.tolist()
        if len(kol_na) >= 2:
            korelasi = df[kol_na].isna().astype(int).corr().round(2)
            st.plotly_chart(px.imshow(korelasi, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                      title="Korelasi indikator kosong (nilai tinggi = sering kosong bersamaan)"),
                            width="stretch")
        pola = df[kol_na].isna().value_counts().head(10).rename("jumlah_baris").reset_index()
        pola[kol_na] = pola[kol_na].astype(object).replace({True: "kosong", False: "·"})
        st.caption("10 pola kombinasi kosong yang paling sering")
        st.dataframe(pola, width="stretch")

        st.markdown('<div class="catatan"><b>Cara membaca.</b> Kosong yang terjadi bersamaan di beberapa kolom sering menandakan masalah sumber data yang sama. Kosong yang acak (MCAR) relatif aman diimputasi, sedangkan kosong yang bergantung pada nilai lain (MAR/MNAR) perlu diselidiki sebelum diisi.</div>', unsafe_allow_html=True)

        if NUM and KAT:
            st.subheader("Apakah kosongnya bergantung pada kategori?")
            a, b = st.columns(2)
            kol_cek = a.selectbox("Kolom yang kosong", kol_na)
            kol_grup = b.selectbox("Dikelompokkan menurut", [c for c in KAT if c != kol_cek])
            grup = df.groupby(kol_grup, dropna=False)[kol_cek].apply(lambda s: 100 * s.isna().mean()).reset_index(name="pct_kosong")
            st.plotly_chart(px.bar(grup, x=kol_grup, y="pct_kosong", title=f"% {kol_cek} kosong per {kol_grup}"), width="stretch")

# =============================================================================
# 3. IMPUTASI
# =============================================================================
elif menu.startswith("3"):
    st.title("Imputasi dan Penanganan Missing Value")
    kol_na = [c for c in df.columns if df[c].isna().any()]
    if not kol_na:
        st.success("Data kerja sudah tidak memiliki nilai kosong.")
    else:
        st.markdown('<div class="catatan">Imputasi di menu ini mengubah <b>data kerja</b> untuk eksplorasi. Pada menu Pemodelan, imputasi dilakukan ulang <b>di dalam pipeline</b> dan hanya dipelajari dari data latih, sehingga tidak terjadi kebocoran data.</div>', unsafe_allow_html=True)
        tab_a, tab_b = st.tabs(["Isi nilai kosong", "Hapus baris / kolom"])
        with tab_a:
            kol = st.multiselect("Kolom yang diimputasi", kol_na, default=kol_na[:1])
            num_dipilih = [c for c in kol if c in NUM]
            kat_dipilih = [c for c in kol if c not in NUM]
            c1, c2 = st.columns(2)
            metode_num = c1.selectbox("Metode untuk kolom numerik", ["Median", "Mean", "Konstanta", "KNN Imputer",
                                                                     "Interpolasi linear", "Forward fill", "Backward fill",
                                                                     "Median per kelompok"], disabled=not num_dipilih)
            metode_kat = c2.selectbox("Metode untuk kolom kategorikal", ["Modus", "Konstanta 'Tidak diketahui'",
                                                                         "Forward fill", "Backward fill"], disabled=not kat_dipilih)
            konstanta = c1.number_input("Nilai konstanta", value=0.0) if metode_num == "Konstanta" else None
            k_knn = c1.slider("Jumlah tetangga (k)", 1, 20, 5) if metode_num == "KNN Imputer" else None
            grup = c1.selectbox("Kelompok", KAT) if metode_num == "Median per kelompok" and KAT else None
            flag = st.checkbox("Tambahkan kolom penanda *_diimputasi", value=True)

            def jalankan(d):
                d = d.copy()
                for c in kol:
                    if flag:
                        d[f"{c}_diimputasi"] = d[c].isna().astype(int)
                if num_dipilih:
                    if metode_num == "Median":
                        d[num_dipilih] = d[num_dipilih].fillna(d[num_dipilih].median())
                    elif metode_num == "Mean":
                        d[num_dipilih] = d[num_dipilih].fillna(d[num_dipilih].mean())
                    elif metode_num == "Konstanta":
                        d[num_dipilih] = d[num_dipilih].fillna(konstanta)
                    elif metode_num == "KNN Imputer":
                        basis = [c for c in kolom_numerik(d) if not c.endswith("_diimputasi")]
                        skala = d[basis].std().replace(0, 1)
                        hasil = KNNImputer(n_neighbors=k_knn).fit_transform(d[basis] / skala) * skala.to_numpy()
                        d[num_dipilih] = pd.DataFrame(hasil, columns=basis, index=d.index)[num_dipilih]
                    elif metode_num == "Interpolasi linear":
                        d[num_dipilih] = d[num_dipilih].interpolate(limit_direction="both")
                    elif metode_num == "Forward fill":
                        d[num_dipilih] = d[num_dipilih].ffill().bfill()
                    elif metode_num == "Backward fill":
                        d[num_dipilih] = d[num_dipilih].bfill().ffill()
                    elif metode_num == "Median per kelompok" and grup:
                        for c in num_dipilih:
                            d[c] = d[c].fillna(d.groupby(grup)[c].transform("median")).fillna(d[c].median())
                for c in kat_dipilih:
                    if metode_kat == "Modus":
                        d[c] = d[c].fillna(d[c].mode().iloc[0])
                    elif metode_kat.startswith("Konstanta"):
                        d[c] = d[c].fillna("Tidak diketahui")
                    elif metode_kat == "Forward fill":
                        d[c] = d[c].ffill().bfill()
                    else:
                        d[c] = d[c].bfill().ffill()
                return d

            if kol:
                hasil = jalankan(df)
                st.subheader("Pratinjau dampak imputasi")
                if num_dipilih:
                    c = st.selectbox("Lihat kolom", num_dipilih)
                    a, b = st.columns([1.4, 1])
                    gabung = pd.concat([df[c].dropna().to_frame("nilai").assign(versi="sebelum (tanpa NaN)"),
                                        hasil[c].to_frame("nilai").assign(versi="sesudah")])
                    a.plotly_chart(px.histogram(gabung, x="nilai", color="versi", barmode="overlay", opacity=.6, nbins=40,
                                                title=f"Distribusi {c}"), width="stretch")
                    banding = pd.DataFrame({"sebelum": df[c].describe(), "sesudah": hasil[c].describe()}).round(3)
                    b.dataframe(banding, width="stretch")
                    if metode_num in ("Mean", "Median", "Konstanta"):
                        turun = 100 * (1 - hasil[c].std() / df[c].std()) if df[c].std() else 0
                        b.caption(f"Simpangan baku turun {turun:.1f}%. Imputasi dengan satu nilai membuat sebaran menyempit.")
                for c in kat_dipilih:
                    st.dataframe(pd.DataFrame({"sebelum": df[c].value_counts(dropna=False), "sesudah": hasil[c].value_counts(dropna=False)}).fillna(0).astype(int),
                                 width="stretch")
                if st.button("Terapkan imputasi", type="primary"):
                    teks = []
                    if num_dipilih:
                        teks.append(f"{metode_num} → {', '.join(num_dipilih)}")
                    if kat_dipilih:
                        teks.append(f"{metode_kat} → {', '.join(kat_dipilih)}")
                    set_data_kerja(hasil, "Imputasi: " + "; ".join(teks))
                    st.rerun()
        with tab_b:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Hapus baris** yang kosong pada kolom tertentu")
                kol_drop = st.multiselect("Kolom acuan", kol_na, key="drop_baris")
                n_hilang = int(df[kol_drop].isna().any(axis=1).sum()) if kol_drop else 0
                st.caption(f"Akan menghapus {n_hilang:,} baris ({100 * n_hilang / len(df):.1f}%).")
                if st.button("Hapus baris", disabled=not kol_drop):
                    set_data_kerja(df.dropna(subset=kol_drop), f"Hapus {n_hilang} baris kosong pada {', '.join(kol_drop)}")
                    st.rerun()
            with c2:
                st.markdown("**Hapus kolom** dengan persentase kosong di atas ambang")
                ambang = st.slider("Ambang % kosong", 5, 95, 50, 5)
                kol_buang = [c for c in df.columns if 100 * df[c].isna().mean() > ambang]
                st.caption(f"Kolom terdampak: {', '.join(kol_buang) or 'tidak ada'}")
                if st.button("Hapus kolom", disabled=not kol_buang):
                    set_data_kerja(df.drop(columns=kol_buang), f"Hapus kolom kosong >{ambang}%: {', '.join(kol_buang)}")
                    st.rerun()

# =============================================================================
# 4. EDA & KUALITAS
# =============================================================================
elif menu.startswith("4"):
    st.title("EDA: Sebaran dan Kualitas Data")
    lengkap, unik, bermasalah = skor_kualitas(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Kelengkapan sel", f"{lengkap:.1f}%")
    k2.metric("Baris unik", f"{unik:.1f}%")
    k3.metric("Kolom perlu perhatian", int(bermasalah))
    k4.metric("Skor kualitas sederhana", f"{(lengkap + unik) / 2:.1f}")
    st.caption("Skor kualitas sederhana = rata-rata kelengkapan sel dan persentase baris unik. Ukuran ini kasar dan hanya untuk perbandingan sebelum dan sesudah pembersihan.")

    t1, t2, t3, t4 = st.tabs(["Profil kualitas", "Distribusi", "Hubungan antarvariabel", "Perbandingan kelompok"])
    with t1:
        prof = profil_kualitas(df)
        st.dataframe(prof, width="stretch", height=380)
        st.download_button("Unduh profil kualitas (CSV)", prof.to_csv(index=False), "profil_kualitas.csv")
        if NUM:
            st.subheader("Statistik deskriptif")
            st.dataframe(df[NUM].describe().T.assign(skew=df[NUM].skew(), kurtosis=df[NUM].kurt()).round(3), width="stretch")
    with t2:
        jenis = st.radio("Jenis kolom", ["Numerik", "Kategorikal"], horizontal=True)
        if jenis == "Numerik" and NUM:
            c = st.selectbox("Kolom", NUM)
            a, b = st.columns([3, 1])
            log = b.checkbox("Skala log (nilai > 0)")
            warna = b.selectbox("Warna menurut", ["(tidak ada)"] + KAT)
            nbins = b.slider("Jumlah bin", 10, 100, 40)
            data_plot = df[df[c] > 0] if log else df
            x = data_plot[c].pipe(np.log10) if log else data_plot[c]
            fig = px.histogram(data_plot.assign(_x=x), x="_x", nbins=nbins, marginal="box",
                               color=None if warna == "(tidak ada)" else warna, barmode="overlay",
                               labels={"_x": f"log10({c})" if log else c}, title=f"Distribusi {c}")
            a.plotly_chart(fig, width="stretch")
            s = df[c].dropna()
            st.write(f"Rata-rata **{s.mean():,.3f}**, median **{s.median():,.3f}**, simpangan baku **{s.std():,.3f}**, skewness **{s.skew():.2f}**.")
            if abs(s.skew()) > 1:
                st.info("Skewness di atas 1 menandakan distribusi sangat menceng. Pertimbangkan transformasi log, terutama untuk model linear.")
        elif KAT:
            c = st.selectbox("Kolom", KAT)
            top = st.slider("Tampilkan N kategori teratas", 5, 50, 15)
            vc = df[c].value_counts(dropna=False).head(top).rename_axis(c).reset_index(name="jumlah")
            vc[c] = vc[c].astype(str)
            st.plotly_chart(px.bar(vc, x="jumlah", y=c, orientation="h", title=f"Frekuensi {c}").update_yaxes(autorange="reversed"),
                            width="stretch")
            st.caption(f"{df[c].nunique()} kategori unik. Kategori terbanyak mencakup {100 * df[c].value_counts(normalize=True).iloc[0]:.1f}% data.")
    with t3:
        if len(NUM) >= 2:
            metode = st.radio("Metode korelasi", ["pearson", "spearman"], horizontal=True)
            kor = df[NUM].corr(method=metode).round(2)
            st.plotly_chart(px.imshow(kor, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                      title=f"Matriks korelasi {metode}", height=520), width="stretch")
            a, b, c = st.columns(3)
            x = a.selectbox("Sumbu X", NUM, index=0)
            y = b.selectbox("Sumbu Y", NUM, index=1)
            w = c.selectbox("Warna", ["(tidak ada)"] + KAT, key="w_scatter")
            sampel = df.sample(min(len(df), 3000), random_state=0)
            st.plotly_chart(px.scatter(sampel, x=x, y=y, color=None if w == "(tidak ada)" else w, opacity=.6,
                                       trendline="ols" if w == "(tidak ada)" else None, title=f"{y} vs {x}"), width="stretch")
        else:
            st.info("Butuh minimal dua kolom numerik.")
    with t4:
        if NUM and KAT:
            a, b = st.columns(2)
            y = a.selectbox("Variabel numerik", NUM, key="grp_y")
            g = b.selectbox("Kelompok", KAT, key="grp_g")
            if df[g].nunique() > 30:
                st.warning("Kategori lebih dari 30. Hanya 30 kategori terbanyak yang ditampilkan.")
            atas = df[g].value_counts().head(30).index
            sub = df[df[g].isin(atas)]
            st.plotly_chart(px.box(sub, x=g, y=y, points="outliers", title=f"{y} menurut {g}"), width="stretch")
            st.dataframe(sub.groupby(g)[y].agg(["count", "mean", "median", "std", "min", "max"]).round(3), width="stretch")
        else:
            st.info("Butuh minimal satu kolom numerik dan satu kolom kategorikal.")

# =============================================================================
# 5. OUTLIER
# =============================================================================
elif menu.startswith("5"):
    st.title("Analisis Outlier")
    if not NUM:
        st.info("Tidak ada kolom numerik.")
        st.stop()
    t1, t2 = st.tabs(["Univariat (per kolom)", "Multivariat (Isolation Forest)"])
    with t1:
        a, b = st.columns(2)
        metode = a.selectbox("Metode deteksi", ["IQR", "Z-score", "Modified Z-score (MAD)"])
        if metode == "IQR":
            param = b.slider("Pengali IQR (k)", 1.0, 3.0, 1.5, 0.1, help="Batas = Q1 − k·IQR dan Q3 + k·IQR")
        elif metode == "Z-score":
            param = b.slider("Ambang |z|", 2.0, 4.0, 3.0, 0.1)
        else:
            param = b.slider("Ambang modified z", 2.0, 5.0, 3.5, 0.1)
        kol_tanpa_flag = [c for c in NUM if not c.endswith("_diimputasi")]
        baris = []
        for c in kol_tanpa_flag:
            lo, hi = batas_outlier(df[c], metode, param)
            m = (df[c] < lo) | (df[c] > hi)
            baris.append({"kolom": c, "batas_bawah": lo, "batas_atas": hi, "n_outlier": int(m.sum()),
                          "pct_outlier": round(100 * m.mean(), 2), "min": df[c].min(), "max": df[c].max()})
        ringkas = pd.DataFrame(baris).sort_values("pct_outlier", ascending=False)
        st.dataframe(ringkas.round(3), width="stretch")

        pk1, pk2 = st.columns([3, 1])
        c = pk1.selectbox("Periksa kolom", ringkas["kolom"])
        sumbu_log = pk2.checkbox("Sumbu log", value=bool(df[c].min() > 0 and abs(df[c].skew()) > 2),
                                 help="Membantu bila rentang nilai sangat lebar. Batas yang ≤ 0 tidak ditampilkan pada sumbu log.")
        lo, hi = batas_outlier(df[c], metode, param)
        m = (df[c] < lo) | (df[c] > hi)
        fig = go.Figure()
        fig.add_trace(go.Box(x=df[c], name=c, boxpoints="outliers", marker_color=WARNA[0]))
        for v in (lo, hi):
            if not (sumbu_log and v <= 0):
                fig.add_vline(x=v, line_dash="dash", line_color=WARNA[3])
        fig.update_layout(height=260, title=f"Boxplot {c} dan batas {metode} (garis putus-putus)", showlegend=False,
                          margin=dict(t=50, b=30))
        if sumbu_log:
            fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Batas bawah **{lo:,.3f}** · batas atas **{hi:,.3f}** · {int(m.sum())} outlier ({100 * m.mean():.2f}%)")
        with st.expander(f"Lihat {int(m.sum())} baris outlier pada {c}"):
            st.dataframe(df[m].sort_values(c), width="stretch")

        st.subheader("Penanganan")
        st.markdown('<div class="catatan">Outlier belum tentu salah. Periksa dulu apakah nilainya salah input atau kejadian ekstrem yang sah. Membuang outlier yang sah menghilangkan informasi penting.</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        kol_tangani = p1.multiselect("Kolom yang ditangani", kol_tanpa_flag, default=[c])
        cara = p2.selectbox("Cara penanganan", ["Capping ke batas (winsorizing)", "Hapus baris outlier",
                                                "Ganti dengan NaN (lalu imputasi)", "Transformasi log1p (kolom baru)"])
        if st.button("Terapkan penanganan", type="primary", disabled=not kol_tangani):
            baru = df.copy()
            ringkasan = []
            if cara.startswith("Hapus"):
                masker = pd.Series(False, index=baru.index)
                for k in kol_tangani:
                    lo_, hi_ = batas_outlier(baru[k], metode, param)
                    masker |= (baru[k] < lo_) | (baru[k] > hi_)
                baru = baru[~masker]
                ringkasan.append(f"{int(masker.sum())} baris dihapus")
            else:
                for k in kol_tangani:
                    lo_, hi_ = batas_outlier(baru[k], metode, param)
                    mk = (baru[k] < lo_) | (baru[k] > hi_)
                    if cara.startswith("Capping"):
                        baru[k] = baru[k].clip(lo_, hi_)
                    elif cara.startswith("Ganti"):
                        baru.loc[mk, k] = np.nan
                    else:
                        if (baru[k] < 0).any():
                            st.warning(f"{k} memiliki nilai negatif, log1p dilewati.")
                            continue
                        baru[f"log_{k}"] = np.log1p(baru[k])
                    ringkasan.append(f"{k}: {int(mk.sum())} nilai")
            set_data_kerja(baru, f"Outlier {metode} (param {param}) · {cara} · {'; '.join(ringkasan)}")
            st.rerun()
    with t2:
        st.markdown("Isolation Forest mendeteksi baris yang **tidak biasa sebagai kombinasi** beberapa kolom, bukan per kolom.")
        kol_if = st.multiselect("Kolom numerik", NUM, default=NUM[:min(4, len(NUM))])
        kontam = st.slider("Perkiraan proporsi outlier", 0.01, 0.2, 0.05, 0.01)
        if len(kol_if) >= 2:
            X = df[kol_if]
            X = X.fillna(X.median())
            pred = IsolationForest(contamination=kontam, random_state=0).fit_predict(X)
            hasil = df.assign(outlier_multivariat=np.where(pred == -1, "outlier", "normal"))
            a, b = st.columns(2)
            x = a.selectbox("Sumbu X", kol_if, index=0, key="if_x")
            y = b.selectbox("Sumbu Y", kol_if, index=1, key="if_y")
            st.plotly_chart(px.scatter(hasil, x=x, y=y, color="outlier_multivariat", opacity=.7,
                                       color_discrete_map={"normal": "#8da9c4", "outlier": "#c0503b"},
                                       title=f"{int((pred == -1).sum())} baris terdeteksi outlier multivariat"), width="stretch")
            st.caption("Nilai kosong diisi median sementara, hanya untuk deteksi ini.")
            if st.button("Tambahkan kolom penanda outlier_multivariat ke data kerja"):
                set_data_kerja(hasil.assign(outlier_multivariat=(pred == -1).astype(int)),
                               f"Isolation Forest ({kontam:.0%}) pada {', '.join(kol_if)}")
                st.rerun()
        else:
            st.info("Pilih minimal dua kolom.")

# =============================================================================
# 6. PEMODELAN
# =============================================================================
elif menu.startswith("6"):
    st.title("Simulasi Regresi dan Klasifikasi")
    st.markdown('<div class="catatan">Data dibagi menjadi data latih dan data uji. Imputasi, penskalaan, dan one-hot encoding dilakukan <b>di dalam pipeline</b> yang hanya di-fit pada data latih.</div>', unsafe_allow_html=True)

    s1, s2 = st.columns([1, 2])
    with s1:
        target = st.selectbox("🎯 Target (y)", list(df.columns), index=len(df.columns) - 1)
        y_mentah = df[target]
        tebakan = "Regresi" if target in NUM and y_mentah.nunique() > 15 else "Klasifikasi"
        tugas = st.radio("Jenis tugas", ["Regresi", "Klasifikasi"], index=["Regresi", "Klasifikasi"].index(tebakan), horizontal=True,
                         help=f"Ditebak otomatis: {tebakan}")
        if tugas == "Regresi" and target not in NUM:
            st.error("Target regresi harus numerik.")
            st.stop()
    with s2:
        kandidat = [c for c in df.columns if c != target and c not in TGL]
        bawaan = [c for c in kandidat if not (c in KAT and df[c].nunique() > 0.9 * len(df))]
        prediktor = st.multiselect("Prediktor (X)", kandidat, default=bawaan,
                                   help="Kolom mirip ID (hampir semua nilainya unik) tidak dipilih otomatis.")
    if not prediktor:
        st.warning("Pilih minimal satu prediktor.")
        st.stop()

    data = df[prediktor + [target]].dropna(subset=[target])
    n_buang = len(df) - len(data)
    if tugas == "Klasifikasi":
        y_all = data[target].astype(str)
        if y_all.nunique() > 20:
            st.error(f"Target memiliki {y_all.nunique()} kelas. Terlalu banyak untuk klasifikasi sederhana. Pilih target lain atau gunakan regresi.")
            st.stop()
        dist = y_all.value_counts()
        a, b = st.columns([1, 2])
        a.dataframe(dist.rename("jumlah").to_frame().assign(persen=lambda d: (100 * d["jumlah"] / d["jumlah"].sum()).round(1)), width="stretch")
        if dist.min() / dist.max() < 0.4:
            b.warning("Kelas tidak seimbang. Akurasi bisa menipu, jadi perhatikan recall dan F1, atau aktifkan class_weight='balanced'.")
    else:
        y_all = data[target]
    if n_buang:
        st.caption(f"{n_buang} baris dengan target kosong tidak diikutkan.")

    algo_tersedia = list(daftar_model(tugas))
    with st.expander("⚙️ Pengaturan model", expanded=True):
        algoritma = st.multiselect("Algoritma", algo_tersedia, default=algo_tersedia[:3])
        c1, c2, c3, c4 = st.columns(4)
        uji = c1.slider("Proporsi data uji", 0.1, 0.5, 0.25, 0.05)
        cv = c2.selectbox("Cross-validation pada data latih", [0, 3, 5, 10], index=2, format_func=lambda v: "tidak" if v == 0 else f"{v}-fold")
        skala = c3.selectbox("Penskalaan numerik", ["StandardScaler", "MinMaxScaler", "Tanpa"])
        imp = c3.selectbox("Imputasi numerik (di pipeline)", ["median", "mean"])
        seed = c4.number_input("Random state", 0, 9999, 42)
        seimbang = c4.checkbox("class_weight='balanced'", disabled=tugas == "Regresi")
        st.markdown("**Hyperparameter dasar**")
        h1, h2, h3, h4, h5 = st.columns(5)
        prm = {"seed": int(seed),
               "max_depth": h1.number_input("max_depth (0 = bebas)", 0, 50, 6),
               "n_estimators": h2.number_input("n_estimators (RF)", 10, 500, 200, 10),
               "k": h3.number_input("k (KNN)", 1, 50, 5),
               "C": h4.number_input("C (Logistic)", 0.01, 100.0, 1.0),
               "alpha": h5.number_input("alpha (Ridge)", 0.0, 100.0, 1.0)}

    if st.button("🚀 Latih dan bandingkan model", type="primary", disabled=not algoritma):
        X = data[prediktor]
        num = [c for c in prediktor if c in NUM]
        kat = [c for c in prediktor if c not in NUM]
        strat = y_all if tugas == "Klasifikasi" and y_all.value_counts().min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=uji, random_state=int(seed), stratify=strat)
        hasil, pipes = [], {}
        prog = st.progress(0.0)
        for i, nama in enumerate(algoritma):
            pipe = buat_pipeline(buat_estimator(tugas, nama, prm, seimbang), num, kat, skala, imp)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            baris = {"Algoritma": nama}
            if tugas == "Regresi":
                baris.update({"R² uji": r2_score(y_te, pred), "MAE": mean_absolute_error(y_te, pred),
                              "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                              "R² latih": r2_score(y_tr, pipe.predict(X_tr))})
                skor_cv = "r2"
                kf = KFold(cv, shuffle=True, random_state=int(seed)) if cv else None
            else:
                rata = "binary" if y_all.nunique() == 2 else "macro"
                pos = sorted(y_all.unique())[-1] if rata == "binary" else None
                kw = {"average": rata, "zero_division": 0}
                if pos is not None:
                    kw["pos_label"] = pos
                baris.update({"Akurasi uji": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred, **kw),
                              "Recall": recall_score(y_te, pred, **kw), "F1": f1_score(y_te, pred, **kw),
                              "Akurasi latih": accuracy_score(y_tr, pipe.predict(X_tr))})
                if rata == "binary" and hasattr(pipe, "predict_proba"):
                    idx = list(pipe.classes_).index(pos)
                    baris["ROC AUC"] = roc_auc_score((y_te == pos).astype(int), pipe.predict_proba(X_te)[:, idx])
                skor_cv = "f1_macro"
                kf = StratifiedKFold(cv, shuffle=True, random_state=int(seed)) if cv and strat is not None else (KFold(cv, shuffle=True, random_state=int(seed)) if cv else None)
            if kf is not None:
                s = cross_val_score(buat_pipeline(buat_estimator(tugas, nama, prm, seimbang), num, kat, skala, imp),
                                    X_tr, y_tr, cv=kf, scoring=skor_cv)
                baris[f"CV {skor_cv} (mean ± sd)"] = f"{s.mean():.3f} ± {s.std():.3f}"
            hasil.append(baris)
            pipes[nama] = pipe
            prog.progress((i + 1) / len(algoritma))
        prog.empty()
        st.session_state.hasil_model = {"tugas": tugas, "target": target, "prediktor": prediktor, "tabel": pd.DataFrame(hasil),
                                        "pipes": pipes, "X_te": X_te, "y_te": y_te, "X_tr": X_tr, "seed": int(seed),
                                        "data": data}
        catat(f"Melatih {tugas.lower()} · target {target} · {len(prediktor)} prediktor · {', '.join(algoritma)}")

    hm = st.session_state.hasil_model
    if hm and hm["target"] == target and hm["tugas"] == tugas:
        st.subheader("Perbandingan model pada data uji")
        utama = "R² uji" if tugas == "Regresi" else ("F1")
        tabel = hm["tabel"].sort_values(utama, ascending=False).reset_index(drop=True)
        st.dataframe(tabel.style.format(precision=4).highlight_max(subset=[utama], color="#d8efe4"), width="stretch")
        selisih_kol = ("R² latih", "R² uji") if tugas == "Regresi" else ("Akurasi latih", "Akurasi uji")
        for _, r in tabel.iterrows():
            if r[selisih_kol[0]] - r[selisih_kol[1]] > 0.15:
                st.warning(f"{r['Algoritma']}: skor latih jauh di atas skor uji, indikasi overfitting.")
        st.plotly_chart(px.bar(tabel, x="Algoritma", y=utama, text_auto=".3f", title=f"{utama} per algoritma"), width="stretch")

        pilih = st.selectbox("Periksa model", tabel["Algoritma"])
        pipe = hm["pipes"][pilih]
        X_te, y_te = hm["X_te"], hm["y_te"]
        pred = pipe.predict(X_te)
        a, b = st.columns(2)
        if tugas == "Regresi":
            lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
            fig = px.scatter(x=y_te, y=pred, opacity=.6, labels={"x": "aktual", "y": "prediksi"}, title="Aktual vs prediksi")
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#c0503b"), name="garis ideal"))
            a.plotly_chart(fig, width="stretch")
            b.plotly_chart(px.scatter(x=pred, y=y_te - pred, opacity=.6, labels={"x": "prediksi", "y": "residual"},
                                      title="Residual vs prediksi").add_hline(y=0, line_dash="dash"), width="stretch")
        else:
            label = list(pipe.classes_)
            cm = confusion_matrix(y_te, pred, labels=label)
            a.plotly_chart(px.imshow(cm, x=label, y=label, text_auto=True, color_continuous_scale="Blues",
                                     labels={"x": "prediksi", "y": "aktual", "color": "jumlah"}, title="Confusion matrix"),
                           width="stretch")
            lap = pd.DataFrame({"kelas": label,
                                "precision": precision_score(y_te, pred, labels=label, average=None, zero_division=0),
                                "recall": recall_score(y_te, pred, labels=label, average=None, zero_division=0),
                                "f1": f1_score(y_te, pred, labels=label, average=None, zero_division=0),
                                "support": [int((y_te == l).sum()) for l in label]})
            b.dataframe(lap.round(3), width="stretch")
        try:
            imp_s, judul = importansi(pipe, X_te, y_te, tugas, hm["seed"])
            imp_s = imp_s.sort_values(ascending=False).head(15).iloc[::-1]
            st.plotly_chart(px.bar(x=imp_s.values, y=imp_s.index, orientation="h", title=f"Fitur paling berpengaruh · {judul}",
                                   labels={"x": "", "y": ""}), width="stretch")
        except Exception as e:
            st.caption(f"Importansi fitur tidak tersedia: {e}")

        st.subheader("🔮 Simulasi prediksi satu data baru")
        with st.form("simulasi"):
            kolom_form = st.columns(3)
            masukan = {}
            for i, c in enumerate(hm["prediktor"]):
                s = hm["data"][c]
                with kolom_form[i % 3]:
                    if c in NUM:
                        masukan[c] = st.number_input(c, value=float(s.median()) if s.notna().any() else 0.0)
                    else:
                        opsi = s.dropna().astype(str).value_counts().index.tolist()[:50] or [""]
                        masukan[c] = st.selectbox(c, opsi)
            kirim = st.form_submit_button("Prediksi")
        if kirim:
            baris = pd.DataFrame([masukan])
            hasil_pred = pipe.predict(baris)[0]
            if tugas == "Regresi":
                st.success(f"Prediksi **{target}** dengan {pilih}: **{hasil_pred:,.3f}**")
            else:
                teks = f"Prediksi kelas **{target}** dengan {pilih}: **{hasil_pred}**"
                if hasattr(pipe, "predict_proba"):
                    prob = pd.Series(pipe.predict_proba(baris)[0], index=pipe.classes_).sort_values(ascending=False)
                    teks += " · " + ", ".join(f"{k}: {v:.1%}" for k, v in prob.items())
                st.success(teks)

        keluaran = X_te.assign(**{f"{target}_aktual": y_te.values, f"{target}_prediksi": pred})
        st.download_button("Unduh prediksi data uji (CSV)", keluaran.to_csv(index=False), f"prediksi_{pilih.replace(' ', '_')}.csv")

# =============================================================================
# 7. RIWAYAT & UNDUH
# =============================================================================
elif menu.startswith("7"):
    st.title("Riwayat Langkah dan Unduh Data")
    a, b = st.columns(2)
    asli, kerja = st.session_state.df_asli, df
    with a:
        st.subheader("Riwayat")
        for i, l in enumerate(st.session_state.log):
            st.markdown(f"{i}. {l}")
        st.download_button("Unduh riwayat (TXT)", "\n".join(st.session_state.log), "riwayat_langkah.txt")
    with b:
        st.subheader("Sebelum vs sesudah")
        l0, u0, _ = skor_kualitas(asli)
        l1, u1, _ = skor_kualitas(kerja)
        banding = pd.DataFrame({"data asli": [asli.shape[0], asli.shape[1], int(asli.isna().sum().sum()), int(asli.duplicated().sum()), round(l0, 2), round(u0, 2)],
                                "data kerja": [kerja.shape[0], kerja.shape[1], int(kerja.isna().sum().sum()), int(kerja.duplicated().sum()), round(l1, 2), round(u1, 2)]},
                               index=["baris", "kolom", "sel kosong", "baris duplikat", "kelengkapan %", "baris unik %"])
        st.dataframe(banding, width="stretch")
        st.download_button("⬇️ Unduh data kerja (CSV)", kerja.to_csv(index=False), "data_bersih.csv", type="primary")
    st.subheader("Cuplikan data kerja")
    st.dataframe(kerja.head(30), width="stretch")
