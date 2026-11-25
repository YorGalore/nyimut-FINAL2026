"""
fusion_common.py
================
Modul bersama untuk tahap A (stage_a_cv.py), early fusion (fusion_early_cv.py),
dan late fusion (fusion_late_cv.py). Spesifikasi: blok "KONTEKS BERSAMA"
bagian 3 dokumen "Prompt Early & Late Fusion -- Revisi".

ATURAN
- Script lain IMPORT dari sini; jangan menyalin atau mengubah fungsi di sini.
- Semua fitur per fold HANYA lewat fold_blocks(). Script lain tidak boleh
  menghitung scaler, PCA, probabilitas H, atau agregasi sendiri.
- Tidak ada baris test (tanggal > 2025-12-09) yang masuk ke data apa pun.
- Perubahan file ini harus dicatat di dokumen aturan (bagian 13).
"""

from __future__ import annotations

import hashlib
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as C  # noqa: E402
from dataset_split import PRICE_FEATURES, TRAIN_FRAC, canonical_split  # noqa: E402

# ---------------------------------------------------------------------------
# KONSTANTA (dikunci)
# ---------------------------------------------------------------------------
SEED = 42
CLASS_ORDER = ["low", "medium", "high"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
N_DEV = 937
N_TRAIN, N_VALID = 772, 165
DEV_END = pd.Timestamp("2025-12-09")
TRAIN_END = pd.Timestamp("2025-03-25")
N_SPLITS = 5
PCA_EMB = 16
PCA_TOPIC = 8
PROBA_DECIMALS = 6

XGB_PARAMS = dict(
    max_depth=2,
    learning_rate=0.03,
    n_estimators=100,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=10,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=1,
)

KNOWN4 = ["lag1_log_return", "lag2_log_return", "lag3_log_return", "lag4_log_return"]

# ---------------------------------------------------------------------------
# PATH FITUR TEKS -- diisi setelah tahap A (bagian 3B) dikunci.
# Selama masih None, fold_blocks() hanya membuat blok yang path-nya diberikan
# eksplisit lewat argumen `paths` (dipakai stage_a_cv.py).
# ---------------------------------------------------------------------------
# Dikunci 25 Nov 2026 dari hasil tahap A (data/processed/stage_a_cv/, run_meta.json):
#   langkah 1a -> roberta_headline (macro-F1 CV 0,7952), langkah 1b -> lda_train_only,
#   langkah 2  -> A0 (agregasi lain tidak ada yang lebih baik dari A0).
PATHS = {
    "embedding": ROOT / "data/processed/head_vs_full/headline/roberta_headline_features.csv",
    "embedding_articles": None,  # tidak dipakai (EMB_AGG = A0)
    "topik": ROOT / "data/processed/topic_modelling/lda_train_only/daily_topic_features.csv",
    "ner": ROOT / "data/processed/ner/daily_entity_features.csv",  # hasil 3A
}
EMB_AGG = "A0"  # "A0" | "A1-3" | "A1-7" | "A1-14" | "A2-hl1" | "A2-hl3" | "A3" | "A4"
N_EMB = 16      # 16, atau 16 + K kalau A4

# ---------------------------------------------------------------------------
# NER (poin 7 blok konteks bersama; hasil bagian 3A, rebuild_report.json)
# ---------------------------------------------------------------------------
NER_HAS_COLS = ["has_country", "has_central_bank", "has_economic_event"]
NER_CONSTANT_COLS = [  # konstan di 937 hari dev, dikunci sekali
    "central_bank_reserve_bank_of_australia_mentions",
    "central_bank_bank_indonesia_mentions",
    "central_bank_bank_of_canada_mentions",
]
NER_KEEP_AS_IS = ["mean_entities_per_article"]
N_NER = 48

TOPIC_COL_RE = re.compile(r"^topic_\d+_mean$")
EMB_COL_RE = re.compile(r"^[a-z0-9_]+_\d{3}$")   # vektor: w2v_000, dim_000, trade_war__dim_000
EMB_COUNT_PREFIX = "count_"                       # A4: jumlah artikel per kategori (tanpa PCA)


# ---------------------------------------------------------------------------
# DATA PENGEMBANGAN DAN LABEL
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def volatility_edges() -> np.ndarray:
    """Batas tertile label, dihitung ulang persis seperti build_dataset()."""
    df = pd.read_csv(C.ALIGNED_DAILY_CSV, usecols=["date", "realized_vol"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n_train_all = int(len(df) * TRAIN_FRAC)
    train_vol = df["realized_vol"].iloc[:n_train_all].dropna()
    _, edges = pd.qcut(train_vol, 3, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    assert len(edges) == 4, edges
    return edges


def classify_vol(values) -> np.ndarray:
    """Kelas 0/1/2 dengan aturan pd.cut(right=True): edges[i] < x <= edges[i+1]."""
    return np.searchsorted(volatility_edges(), np.asarray(values, dtype=float), side="left") - 1


@lru_cache(maxsize=1)
def _dev_frame() -> pd.DataFrame:
    full = canonical_split()
    dev = full.loc[full["split"] != "test"].copy()
    del full  # baris test tidak disimpan di mana pun
    dev = dev.sort_values("date").reset_index(drop=True)
    assert len(dev) == N_DEV, len(dev)
    assert (dev["split"] == "train").sum() == N_TRAIN
    assert (dev["split"] == "valid").sum() == N_VALID
    assert dev["date"].max() == DEV_END, dev["date"].max()
    assert dev["date"].is_monotonic_increasing and not dev["date"].duplicated().any()
    dev["y"] = dev["volatility_class"].map(CLASS_TO_INT).astype(int)
    # label dari volatility_edges() harus sama dengan label resmi
    assert (classify_vol(dev["realized_vol"]) == dev["y"].to_numpy()).all()
    assert not dev[PRICE_FEATURES + KNOWN4 + ["log_return", "n_news"]].isna().any().any()
    keep = ["date", "split", "y", "volatility_class", "realized_vol", "log_return",
            "n_news", *dict.fromkeys(PRICE_FEATURES + KNOWN4)]
    return dev[keep]


def load_dev_data() -> pd.DataFrame:
    """937 hari pengembangan (train + valid), urut tanggal, kolom y = 0/1/2."""
    return _dev_frame().copy()


def get_folds(n: int, n_splits: int = N_SPLITS):
    """list (train_idx, eval_idx) dari TimeSeriesSplit (expanding, tanpa gap)."""
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n)))


# ---------------------------------------------------------------------------
# PROBABILITAS H
# ---------------------------------------------------------------------------
def h_proba(rows: pd.DataFrame, train_rows: pd.DataFrame) -> np.ndarray:
    """
    Untuk setiap baris: 4 return yang sudah diketahui + SETIAP log_return di
    train_rows sebagai calon r[t] -> std (ddof=1) -> kelas -> proporsi.
    """
    known = rows[KNOWN4].to_numpy(float)              # (n, 4)
    cand = train_rows["log_return"].to_numpy(float)   # (m,)
    n, m = len(known), len(cand)
    s1 = known.sum(1, keepdims=True) + cand[None, :]
    s2 = (known ** 2).sum(1, keepdims=True) + cand[None, :] ** 2
    var = (s2 - s1 ** 2 / 5) / 4
    std = np.sqrt(np.clip(var, 0, None))              # (n, m)
    cls = classify_vol(std.ravel()).reshape(n, m)
    proba = np.stack([(cls == k).mean(1) for k in range(3)], axis=1)
    return proba


# ---------------------------------------------------------------------------
# FITUR TEKS
# ---------------------------------------------------------------------------
def _read_daily(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    assert not df["date"].duplicated().any(), f"tanggal ganda di {path}"
    return df.set_index("date")


def embedding_columns(df: pd.DataFrame) -> list[str]:
    """Kolom vektor + (A4) kolom count_* . Kolom lain (token_count, coverage) dibuang."""
    cols = [c for c in df.columns if EMB_COL_RE.match(c)]
    assert cols, "tidak ada kolom vektor embedding"
    return cols + [c for c in df.columns if c.startswith(EMB_COUNT_PREFIX)]


def topic_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if TOPIC_COL_RE.match(c)]
    assert cols, "tidak ada kolom topic_XX_mean"
    return cols


def ner_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if c != "date" and c not in NER_HAS_COLS and c not in NER_CONSTANT_COLS]
    assert len(cols) == N_NER, (len(cols), N_NER)
    return cols


def load_text_features(kind: str, dates, path=None) -> pd.DataFrame:
    """
    Fitur teks harian sesuai poin 7, diindeks `dates` (urutan dipertahankan).
    kind = "embedding" | "topik" | "ner". path=None -> PATHS[kind].
    Tanggal tanpa baris di file -> NaN (diisi 0 setelah PCA di fold_blocks).
    """
    path = path if path is not None else PATHS[kind]
    if path is None:
        raise ValueError(f"PATH '{kind}' belum diisi (tahap A belum dikunci).")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)).dt.normalize())
    assert dates.max() <= DEV_END, "tanggal test tidak boleh dimuat"
    raw = _read_daily(path)
    if kind == "embedding":
        cols = embedding_columns(raw)
        out = raw.reindex(dates)[cols]
    elif kind == "topik":
        cols = topic_columns(raw)
        out = raw.reindex(dates)[cols]
    elif kind == "ner":
        cols = ner_columns(raw.reset_index())
        out = raw.reindex(dates)[cols].astype(float)
        n_news = _dev_frame().set_index("date")["n_news"].reindex(dates).to_numpy(float)
        assert np.isfinite(n_news).all() and (n_news > 0).all()
        divide = [c for c in cols if c not in NER_KEEP_AS_IS]
        out[divide] = out[divide].to_numpy() / n_news[:, None]
    else:
        raise ValueError(kind)
    out.index.name = "date"
    return out


def fit_text_pca(X_train: np.ndarray, X_eval: np.ndarray, n_components: int):
    """StandardScaler + PCA di-fit di X_train saja. Baris NaN -> 0 setelah PCA."""
    X_train = np.asarray(X_train, float)
    X_eval = np.asarray(X_eval, float)
    ok_tr = ~np.isnan(X_train).any(1)
    ok_ev = ~np.isnan(X_eval).any(1)
    scaler = StandardScaler().fit(X_train[ok_tr])
    pca = PCA(n_components=n_components, random_state=SEED).fit(scaler.transform(X_train[ok_tr]))
    Z_tr = np.zeros((len(X_train), n_components))
    Z_ev = np.zeros((len(X_eval), n_components))
    Z_tr[ok_tr] = pca.transform(scaler.transform(X_train[ok_tr]))
    if ok_ev.any():
        Z_ev[ok_ev] = pca.transform(scaler.transform(X_eval[ok_ev]))
    return Z_tr, Z_ev, int(ok_tr.sum())


def block_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.round(np.asarray(arr, float), 10))
    return hashlib.md5(a.tobytes()).hexdigest()


def fold_blocks(train_idx, eval_idx, paths: dict | None = None, emb_agg: str | None = None,
                data: pd.DataFrame | None = None, y_override: np.ndarray | None = None) -> dict:
    """
    Satu-satunya jalur fitur per fold (poin 8). Mengembalikan
      {"train": {"H","PRICE","EMB","TOPIC","NER"}, "eval": {...},
       "hash": {...}, "fit_rows": {...}, "dates": {"train","eval"}}
    Blok yang sumbernya tidak tersedia bernilai None.

    paths   : override PATHS (tahap A memberi path eksplisit per varian).
    emb_agg : override EMB_AGG. "A3" melatih attention di baris train fold saja.
    data    : DataFrame indeks posisi (default load_dev_data()).
    y_override : label train untuk A3 (dipakai uji label acak G2).
    """
    data = load_dev_data() if data is None else data
    train_idx = np.asarray(train_idx)
    eval_idx = np.asarray(eval_idx)
    assert len(np.intersect1d(train_idx, eval_idx)) == 0
    tr, ev = data.iloc[train_idx], data.iloc[eval_idx]
    assert tr["date"].max() < ev["date"].min(), "tanggal eval ada di train fold"
    assert ev["date"].max() <= DEV_END

    p = dict(PATHS)
    if paths is not None:
        p = {k: None for k in PATHS}
        p.update(paths)
    agg = emb_agg if emb_agg is not None else (EMB_AGG or "A0")

    out = {"train": {}, "eval": {}, "fit_rows": {}, "hash": {},
           "dates": {"train": tr["date"].to_numpy(), "eval": ev["date"].to_numpy()}}
    # H dan PRICE
    out["train"]["H"] = h_proba(tr, tr)
    out["eval"]["H"] = h_proba(ev, tr)
    out["fit_rows"]["H"] = len(tr)
    out["train"]["PRICE"] = tr[PRICE_FEATURES].to_numpy(float)
    out["eval"]["PRICE"] = ev[PRICE_FEATURES].to_numpy(float)

    # EMBEDDING
    if agg == "A3" and p.get("embedding_articles") is not None:
        import embedding_aggregation as EA
        y_tr = tr["y"].to_numpy() if y_override is None else np.asarray(y_override)
        X_tr, X_ev = EA.a3_fold_vectors(p["embedding_articles"], tr["date"], y_tr, ev["date"])
        z_tr, z_ev, n_fit = fit_text_pca(X_tr, X_ev, PCA_EMB)
        out["train"]["EMB"], out["eval"]["EMB"] = z_tr, z_ev
        out["fit_rows"]["EMB"] = n_fit
    elif p.get("embedding") is not None:
        emb = load_text_features("embedding", data["date"], p["embedding"])
        count_cols = [c for c in emb.columns if c.startswith(EMB_COUNT_PREFIX)]  # A4: tanpa PCA
        vec_cols = [c for c in emb.columns if c not in count_cols]
        z_tr, z_ev, n_fit = fit_text_pca(emb[vec_cols].to_numpy()[train_idx],
                                         emb[vec_cols].to_numpy()[eval_idx], PCA_EMB)
        if count_cols:
            c = emb[count_cols].fillna(0).to_numpy(float)
            z_tr, z_ev = np.hstack([z_tr, c[train_idx]]), np.hstack([z_ev, c[eval_idx]])
        out["train"]["EMB"], out["eval"]["EMB"] = z_tr, z_ev
        out["fit_rows"]["EMB"] = n_fit
    else:
        out["train"]["EMB"] = out["eval"]["EMB"] = None

    # TOPIK
    if p.get("topik") is not None:
        top = load_text_features("topik", data["date"], p["topik"]).to_numpy()
        z_tr, z_ev, n_fit = fit_text_pca(top[train_idx], top[eval_idx], PCA_TOPIC)
        out["train"]["TOPIC"], out["eval"]["TOPIC"] = z_tr, z_ev
        out["fit_rows"]["TOPIC"] = n_fit
    else:
        out["train"]["TOPIC"] = out["eval"]["TOPIC"] = None

    # NER (tanpa PCA; hari tanpa fitur -> 0)
    if p.get("ner") is not None:
        ner = np.nan_to_num(load_text_features("ner", data["date"], p["ner"]).to_numpy(), nan=0.0)
        out["train"]["NER"], out["eval"]["NER"] = ner[train_idx], ner[eval_idx]
    else:
        out["train"]["NER"] = out["eval"]["NER"] = None

    for part in ("train", "eval"):
        for name, arr in out[part].items():
            if arr is not None:
                assert np.isfinite(arr).all(), f"NaN/inf di blok {name} ({part})"
                out["hash"][f"{part}_{name}"] = block_hash(arr)
    return out


def locked_columns() -> dict:
    """EMB_COLS, TOPIC_COLS, NER_COLS dari file PATHS (header saja, untuk run_meta)."""
    head = {k: pd.read_csv(v, nrows=0).columns for k, v in PATHS.items()
            if v is not None and k != "embedding_articles"}
    cols = {
        "EMB_COLS": embedding_columns(pd.DataFrame(columns=head["embedding"])),
        "TOPIC_COLS": topic_columns(pd.DataFrame(columns=head["topik"])),
        "NER_COLS": ner_columns(pd.DataFrame(columns=head["ner"])),
    }
    assert len(cols["TOPIC_COLS"]) == 15 and len(cols["NER_COLS"]) == N_NER
    return cols


def stack_blocks(blocks: dict, part: str, names: list[str]) -> np.ndarray:
    arrs = [blocks[part][n] for n in names]
    assert all(a is not None for a in arrs), f"blok kosong di {names}"
    return np.hstack(arrs)


# ---------------------------------------------------------------------------
# MODEL, METRIK, BASELINE
# ---------------------------------------------------------------------------
def make_xgb() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def fit_predict(X_train, y_train, X_eval):
    """Latih make_xgb() -> (proba_train, proba_eval)."""
    clf = make_xgb()
    clf.fit(np.asarray(X_train, float), np.asarray(y_train, int))
    return clf.predict_proba(np.asarray(X_train, float)), clf.predict_proba(np.asarray(X_eval, float))


def score(y, proba) -> dict:
    y = np.asarray(y, int)
    proba = np.clip(np.asarray(proba, float), 1e-15, 1)
    proba = proba / proba.sum(1, keepdims=True)
    pred = proba.argmax(1)
    f1s = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=[0, 1, 2]),
        "f1_low": f1s[0], "f1_medium": f1s[1], "f1_high": f1s[2],
    }


def naive_pred(rows: pd.DataFrame) -> np.ndarray:
    """Kelas kemarin (lag1_volatility_class_enc) sebagai probabilitas one-hot."""
    cls = rows["lag1_volatility_class_enc"].to_numpy().astype(int)
    return np.eye(3)[cls]


def baseline_a(blocks: dict, y_train):
    """Baseline A: XGBoost di PRICE_FEATURES saja -> (proba_train, proba_eval)."""
    return fit_predict(blocks["train"]["PRICE"], y_train, blocks["eval"]["PRICE"])


# ---------------------------------------------------------------------------
# GERBANG OVERFIT (bagian 6) -- dipakai early dan late dengan kode yang sama
# ---------------------------------------------------------------------------
G1_MAX_GAP_POINTS = 10.0
G2_MAX_MACRO_F1 = 0.40
W1_MAX_EXTRA_GAP = 5.0
W2_MAX_SD = 0.08
W3_MAX_WEIGHT_SHIFT = 0.5


def shuffled_labels(y_train, k: int) -> np.ndarray:
    """Label train fold k diacak dengan default_rng(42 + k) (sama di early/late)."""
    return np.random.default_rng(SEED + k).permutation(np.asarray(y_train))


def overfit_gate(pipeline: str, folds: pd.DataFrame, g2_macro_f1: list[float],
                 g3_ok: bool, g3_note: str = "", weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    folds : cv_folds.csv (kolom model, fold, part, accuracy, macro_f1).
    g2_macro_f1 : macro-F1 eval per fold untuk E4 dengan label train acak.
    g3_ok : hasil semua assert G3 (assert yang gagal sudah menghentikan script).
    Mengembalikan baris overfit_check.csv. G4 dijalankan terpisah (equality_check).
    """
    rows = []
    cands = [f"E{i}" for i in range(5)]
    acc = folds.pivot_table(index=["model", "fold"], columns="part", values="accuracy")
    gap = ((acc["train"] - acc["eval"]) * 100).groupby("model").mean()
    for m in cands:
        if m in gap:
            rows.append((m, "G1", gap[m], G1_MAX_GAP_POINTS,
                         "LOLOS" if gap[m] <= G1_MAX_GAP_POINTS else "GAGAL",
                         "rata-rata gap akurasi train - eval (poin)"))
    g2 = float(np.mean(g2_macro_f1))
    rows.append(("E4", "G2", g2, G2_MAX_MACRO_F1, "LOLOS" if g2 <= G2_MAX_MACRO_F1 else "GAGAL",
                 "macro-F1 eval dengan label train acak"))
    rows.append(("all", "G3", float(g3_ok), 1.0, "LOLOS" if g3_ok else "GAGAL", g3_note))
    if "E4" in gap and "E0" in gap:
        w1 = gap["E4"] - gap["E0"]
        rows.append(("E4", "W1", w1, W1_MAX_EXTRA_GAP,
                     "PERINGATAN" if w1 > W1_MAX_EXTRA_GAP else "LOLOS", "gap E4 - gap E0"))
    ev = folds[folds["part"] == "eval"]
    sd = ev.groupby("model")["macro_f1"].std(ddof=1)
    mean = ev.groupby("model")["macro_f1"].mean()
    for m in cands:
        if m in sd:
            rows.append((m, "W2", sd[m], W2_MAX_SD,
                         "PERINGATAN" if sd[m] > W2_MAX_SD else "LOLOS", "SD macro-F1 antar fold"))
            if "H" in mean:
                rows.append((m, "W4", mean[m] - mean["H"], 0.0,
                             "PERINGATAN" if mean[m] < mean["H"] else "LOLOS",
                             "macro-F1 kandidat - H"))
    if weights is not None and len(weights):
        for m, g in weights.groupby("model"):
            for w in ("w_embedding", "w_topik", "w_ner"):
                shift = g[w].max() - g[w].min()
                rows.append((m, "W3", shift, W3_MAX_WEIGHT_SHIFT,
                             "PERINGATAN" if shift > W3_MAX_WEIGHT_SHIFT else "LOLOS",
                             f"rentang {w} antar fold"))
    out = pd.DataFrame(rows, columns=["model", "check", "value", "threshold", "status", "note"])
    out.insert(0, "pipeline", pipeline)
    return out


def equality_check(early_dir, late_dir) -> pd.DataFrame:
    """G4: dijalankan setelah early dan late selesai."""
    import json
    early_dir, late_dir = Path(early_dir), Path(late_dir)
    rows = []
    oe = pd.read_csv(early_dir / "oof_predictions.csv")
    ol = pd.read_csv(late_dir / "oof_predictions.csv")
    cols = ["p_low", "p_medium", "p_high"]
    e0 = oe[oe.model == "E0"].sort_values("date")[cols].to_numpy()
    l0 = ol[ol.model == "E0"].sort_values("date")[cols].to_numpy()
    d = float(np.abs(e0 - l0).max()) if e0.shape == l0.shape else np.inf
    rows.append(("E0", "G4", d, 1e-6, "LOLOS" if d <= 1e-6 else "GAGAL", "selisih maks proba E0"))
    fe = pd.read_csv(early_dir / "cv_folds.csv")
    fl = pd.read_csv(late_dir / "cv_folds.csv")
    key = ["model", "fold", "part"]
    met = ["n_rows", "macro_f1", "accuracy", "balanced_accuracy", "log_loss"]
    be = fe[fe.model.isin(["naive", "H", "A"])].sort_values(key)[key + met].reset_index(drop=True)
    bl = fl[fl.model.isin(["naive", "H", "A"])].sort_values(key)[key + met].reset_index(drop=True)
    same = be.equals(bl)
    rows.append(("baseline", "G4", float(same), 1.0, "LOLOS" if same else "GAGAL", "naive/H/A identik"))
    me = json.loads((early_dir / "run_meta.json").read_text())
    ml = json.loads((late_dir / "run_meta.json").read_text())
    for k in ["XGB_PARAMS", "EMB_AGG", "N_EMB", "N_NER", "EMB_COLS", "TOPIC_COLS", "NER_COLS",
              "fold_sizes", "block_hashes", "feature_files_md5", "volatility_edges", "versions"]:
        ok = me.get(k) == ml.get(k)
        rows.append((k, "G4", float(ok), 1.0, "LOLOS" if ok else "GAGAL", "run_meta sama"))
    out = pd.DataFrame(rows, columns=["model", "check", "value", "threshold", "status", "note"])
    out.insert(0, "pipeline", "early_vs_late")
    return out


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict:
    import platform

    import sklearn
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__}
