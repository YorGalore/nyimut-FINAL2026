"""
ablation_test.py
================
Tahap 2 Tugas 3: STUDI ABLASI dan EVALUASI AKHIR DI TEST.

PERINGATAN: skrip ini satu-satunya yang membuka data test (167 hari,
2025-12-11 .. 2026-09-01). Jalankan SETELAH arsitektur (Model C = E4) dan
hyperparameter (src/tune_model_c.py) terkunci. Jangan dipakai untuk memilih
model atau parameter.

Protokol -- sama persis dengan Tugas 2 (Keputusan 41):
  latih di train (772 hari) -> evaluasi di test (167 hari), sekali.
  Parameter tiap model diambil dari data/processed/tuning/selected_params.json.

Fitur dibentuk dengan logika yang SAMA dengan fusion_common.fold_blocks():
  H      : proporsi kelas dari 4 return yang sudah diketahui, kandidat r[t]
           hanya dari log_return train.
  PRICE  : PRICE_FEATURES (semua lag, tanpa data hari t).
  EMB    : embedding harian -> StandardScaler + PCA(16), di-fit di train saja.
  TOPIC  : topic_XX_mean   -> StandardScaler + PCA(8),  di-fit di train saja.
  NER    : 48 kolom, dibagi n_news hari itu, tanpa PCA.
Model B (baseline NLP Tugas 2) dibangun ulang di sini dengan resep
train_combined.py, supaya tangga ablasi A -> B -> C memakai satu protokol.
Bedanya hanya satu: baris test ikut di-transform (bukan di-fit).

Cara pakai (dari root proyek):
    ./.venv311/bin/python src/ablation_test.py

Output: data/processed/ablation/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fusion_common as F  # noqa: E402
from dataset_split import LM_FEATURES, XGB_GRID, canonical_split  # noqa: E402

OUT = F.ROOT / "data" / "processed" / "ablation"
TUNING = F.ROOT / "data" / "processed" / "tuning" / "selected_params.json"

LADDER = {
    "A": ["PRICE"],
    "E0": ["H", "PRICE"],
    "E1": ["H", "PRICE", "EMB"],
    "E2": ["H", "PRICE", "EMB", "TOPIC"],
    "E3": ["H", "PRICE", "EMB", "NER"],
    "E4": ["H", "PRICE", "EMB", "TOPIC", "NER"],
}
MODEL_C = "E4"

# Model B Tugas 2: harga + Loughran-McDonald + TF-IDF yang dikompres SVD.
# Setelan disalin persis dari train_combined.py supaya angkanya sebanding.
MIN_DF, MAX_DF, NGRAM_RANGE, MAX_FEATURES, N_SVD_COMPONENTS = 5, 0.8, (1, 2), 5000, 20

PAIRS = [("B", "A"), ("E0", "A"), ("E1", "E0"), ("E2", "E1"), ("E3", "E1"),
         ("E4", "E0"), ("E4", "E1"), ("E4", "E3"), ("E4", "B"), ("E4", "naive")]


def model_b_predict(full, train_idx, valid_idx, test_idx, y):
    """
    Model B (baseline NLP Tugas 2), dibangun ulang di sini supaya seluruh
    tangga ablasi memakai satu protokol yang sama. TF-IDF dan SVD di-fit di
    train saja; hyperparameter dipilih dengan grid dan aturan yang sama.
    """
    text = full["text_norm_concat"].fillna("")
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF,
                                 ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    tfidf_train = vectorizer.fit_transform(text.iloc[train_idx])
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=F.SEED)
    z_train = svd.fit_transform(tfidf_train)

    base = F.PRICE_FEATURES + LM_FEATURES

    def design(idx, z):
        return np.hstack([full.iloc[idx][base].fillna(0).to_numpy(float), z])

    X_train = design(train_idx, z_train)
    X_valid = design(valid_idx, svd.transform(vectorizer.transform(text.iloc[valid_idx])))
    X_test = design(test_idx, svd.transform(vectorizer.transform(text.iloc[test_idx])))

    best_acc, best_params = -1.0, None
    for params in XGB_GRID:
        clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=F.SEED)
        clf.fit(X_train, y[train_idx])
        acc = accuracy_score(y[valid_idx], clf.predict(X_valid))
        if acc > best_acc:
            best_acc, best_params = acc, params

    clf = XGBClassifier(**best_params, eval_metric="mlogloss", random_state=F.SEED)
    clf.fit(X_train, y[train_idx])
    return clf.predict(X_test).astype(int), X_train.shape[1], best_params, best_acc


def load_text_features_full(kind: str, dates, data: pd.DataFrame, path=None) -> pd.DataFrame:
    """Sama dengan fusion_common.load_text_features(), tanpa batas DEV_END."""
    path = path if path is not None else F.PATHS[kind]
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)).dt.normalize())
    raw = F._read_daily(path)
    if kind == "embedding":
        out = raw.reindex(dates)[F.embedding_columns(raw)]
    elif kind == "topik":
        out = raw.reindex(dates)[F.topic_columns(raw)]
    elif kind == "ner":
        cols = F.ner_columns(raw.reset_index())
        out = raw.reindex(dates)[cols].astype(float)
        n_news = data.set_index("date")["n_news"].reindex(dates).to_numpy(float)
        assert np.isfinite(n_news).all() and (n_news > 0).all()
        divide = [c for c in cols if c not in F.NER_KEEP_AS_IS]
        out[divide] = out[divide].to_numpy() / n_news[:, None]
    else:
        raise ValueError(kind)
    out.index.name = "date"
    return out


def build_blocks(data: pd.DataFrame, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict:
    tr, ev = data.iloc[train_idx], data.iloc[eval_idx]
    assert tr["date"].max() < ev["date"].min(), "tanggal eval harus setelah train"
    out = {"train": {}, "eval": {}}

    out["train"]["H"] = F.h_proba(tr, tr)
    out["eval"]["H"] = F.h_proba(ev, tr)
    out["train"]["PRICE"] = tr[F.PRICE_FEATURES].to_numpy(float)
    out["eval"]["PRICE"] = ev[F.PRICE_FEATURES].to_numpy(float)

    emb = load_text_features_full("embedding", data["date"], data)
    count_cols = [c for c in emb.columns if c.startswith(F.EMB_COUNT_PREFIX)]
    vec = emb[[c for c in emb.columns if c not in count_cols]].to_numpy()
    z_tr, z_ev, n_fit = F.fit_text_pca(vec[train_idx], vec[eval_idx], F.PCA_EMB)
    assert n_fit == len(train_idx), "PCA embedding tidak di-fit di seluruh train"
    out["train"]["EMB"], out["eval"]["EMB"] = z_tr, z_ev

    top = load_text_features_full("topik", data["date"], data).to_numpy()
    z_tr, z_ev, n_fit = F.fit_text_pca(top[train_idx], top[eval_idx], F.PCA_TOPIC)
    assert n_fit == len(train_idx), "PCA topik tidak di-fit di seluruh train"
    out["train"]["TOPIC"], out["eval"]["TOPIC"] = z_tr, z_ev

    ner = np.nan_to_num(load_text_features_full("ner", data["date"], data).to_numpy(), nan=0.0)
    out["train"]["NER"], out["eval"]["NER"] = ner[train_idx], ner[eval_idx]

    for part in ("train", "eval"):
        for name, arr in out[part].items():
            assert np.isfinite(arr).all(), f"NaN/inf di blok {name} ({part})"
    return out


def metrics(y_true, y_pred) -> dict:
    f1s = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_low": f1s[0], "f1_medium": f1s[1], "f1_high": f1s[2],
    }


def main() -> None:
    print("=" * 68)
    print("EVALUASI AKHIR DI TEST -- dijalankan sekali, setelah model terkunci")
    print("=" * 68)

    tuned = json.loads(TUNING.read_text())["selected"]
    for model in LADDER:
        assert model in tuned, f"parameter {model} belum ada di {TUNING}"

    full = canonical_split()
    full["date"] = pd.to_datetime(full["date"]).dt.normalize()
    full = full.sort_values("date").reset_index(drop=True)
    full["y"] = full["volatility_class"].map(F.CLASS_TO_INT).astype(int)
    assert (F.classify_vol(full["realized_vol"]) == full["y"].to_numpy()).all()
    assert not full["date"].duplicated().any()

    train_idx = np.flatnonzero((full["split"] == "train").to_numpy())
    valid_idx = np.flatnonzero((full["split"] == "valid").to_numpy())
    test_idx = np.flatnonzero((full["split"] == "test").to_numpy())
    assert len(train_idx) == F.N_TRAIN and len(test_idx) == 167
    assert full["date"].iloc[train_idx].max() == F.TRAIN_END
    assert full["date"].iloc[test_idx].min() > F.DEV_END

    y = full["y"].to_numpy()
    y_test = y[test_idx]
    dates_test = full["date"].iloc[test_idx].dt.strftime("%Y-%m-%d").to_numpy()
    print(f"Train: {len(train_idx)} hari (s.d. {F.TRAIN_END.date()})")
    print(f"Test : {len(test_idx)} hari ({dates_test[0]} .. {dates_test[-1]})")
    print(f"Sebaran kelas test: {dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    blocks = build_blocks(full, train_idx, test_idx)

    rows, preds = [], {}
    naive = full["lag1_volatility_class_enc"].to_numpy().astype(int)[test_idx]
    preds["naive"] = naive
    rows.append({"model": "naive", "n_cols": 1, "params": "-", **metrics(y_test, naive)})

    pred_b, n_cols_b, params_b, valid_acc_b = model_b_predict(
        full, train_idx, valid_idx, test_idx, y)
    preds["B"] = pred_b
    print(f"Model B: {n_cols_b} kolom, parameter terpilih {params_b} (akurasi valid {valid_acc_b:.4f})")
    rows.append({
        "model": "B", "n_cols": n_cols_b,
        "params": f"n={params_b['n_estimators']},d={params_b['max_depth']},lr={params_b['learning_rate']}",
        **metrics(y_test, pred_b),
    })

    for model, names in LADDER.items():
        X_train = F.stack_blocks(blocks, "train", names)
        X_test = F.stack_blocks(blocks, "eval", names)
        assert X_train.shape[1] == tuned[model]["n_cols"], (
            model, X_train.shape[1], tuned[model]["n_cols"])
        params = tuned[model]["params"]
        clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=F.SEED)
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test).astype(int)
        preds[model] = pred
        rows.append({
            "model": model, "n_cols": X_train.shape[1],
            "params": f"n={params['n_estimators']},d={params['max_depth']},lr={params['learning_rate']}",
            **metrics(y_test, pred),
        })

    results = pd.DataFrame(rows)

    mc = []
    for a, b in PAIRS:
        ca, cb = preds[a] == y_test, preds[b] == y_test
        a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0
        mc.append({"model_a": a, "model_b": b, "a_benar_saja": a_only, "b_benar_saja": b_only,
                   "p_value": p, "signifikan_0.05": p < 0.05})
    mcnemar = pd.DataFrame(mc)

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "test_metrics.csv", index=False)
    mcnemar.to_csv(OUT / "mcnemar_test.csv", index=False)
    pd.DataFrame({"date": dates_test, "y_true": y_test,
                  **{f"pred_{m}": p for m, p in preds.items()}}
                 ).to_csv(OUT / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_test, preds[MODEL_C], labels=[0, 1, 2])
    pd.DataFrame(cm, index=["aktual_low", "aktual_medium", "aktual_high"],
                 columns=["pred_low", "pred_medium", "pred_high"]
                 ).to_csv(OUT / f"confusion_matrix_{MODEL_C}.csv")

    (OUT / "run_meta.json").write_text(json.dumps({
        "catatan": "Evaluasi test dijalankan sekali setelah arsitektur dan hyperparameter terkunci.",
        "model_c": MODEL_C,
        "protokol": "latih di train 772 hari, evaluasi di test 167 hari (sama dengan Tugas 2)",
        "sumber_hyperparameter": "data/processed/tuning/selected_params.json",
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "test_range": [str(dates_test[0]), str(dates_test[-1])],
        "feature_files": {k: str(Path(v).relative_to(F.ROOT))
                          for k, v in F.PATHS.items() if v is not None},
        "versions": F.library_versions(),
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("=== HASIL TEST ===")
    print(results.round(4).to_string(index=False))
    print("\n=== UJI McNEMAR (test) ===")
    print(mcnemar.round(4).to_string(index=False))
    print(f"\n=== CONFUSION MATRIX {MODEL_C} (baris=aktual, kolom=prediksi) ===")
    print(cm)
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()