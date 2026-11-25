"""
analisis_fitur.py
=================
Tahap 3 Tugas 3: analisis eksploratif fitur (poin 3c).

Menghasilkan tiga keluaran, semuanya ANALISIS atas model yang sudah terkunci.
Tidak ada pemilihan model atau hyperparameter di sini.

  3.1  Visualisasi ruang embedding : proyeksi t-SNE (dan UMAP bila terpasang)
       atas embedding judul harian, diwarnai kelas volatilitas.
  3.2  Tingkat kepentingan fitur   : feature importance Model C (E4) per blok
       dan 20 fitur teratas.
  2.3  Verifikasi gap overfit      : akurasi latih dan evaluasi pada CV dev
       untuk setelan fusion terkunci versus setelan hasil tuning.

Cara pakai (dari root proyek):
    ./.venv311/bin/python src/analisis_fitur.py

Output: data/processed/analisis_fitur/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fusion_common as F  # noqa: E402
from ablation_test import build_blocks, load_text_features_full  # noqa: E402
from dataset_split import canonical_split  # noqa: E402

OUT = F.ROOT / "data" / "processed" / "analisis_fitur"
TUNING = F.ROOT / "data" / "processed" / "tuning" / "selected_params.json"

CLASS_NAMES = {0: "rendah", 1: "sedang", 2: "tinggi"}
BLOCK_ORDER = ["H", "PRICE", "EMB", "TOPIC", "NER"]
RANDOM_STATE = 42


def feature_names() -> dict[str, list[str]]:
    """Nama kolom per blok, urutannya sama dengan stack_blocks()."""
    ner_raw = load_text_features_full("ner", pd.Series([F.TRAIN_END]), None_safe_frame())
    return {
        "H": ["h_rendah", "h_sedang", "h_tinggi"],
        "PRICE": list(F.PRICE_FEATURES),
        "EMB": [f"emb_pca_{i:02d}" for i in range(F.PCA_EMB)],
        "TOPIC": [f"topic_pca_{i}" for i in range(F.PCA_TOPIC)],
        "NER": list(ner_raw.columns),
    }


def None_safe_frame() -> pd.DataFrame:
    """Kerangka minimal berisi kolom n_news, dipakai hanya untuk membaca nama kolom NER."""
    full = canonical_split()
    full["date"] = pd.to_datetime(full["date"]).dt.normalize()
    return full[["date", "n_news"]]


# ============================================================
# 3.1 VISUALISASI RUANG EMBEDDING
# ============================================================

def visualisasi_embedding(full: pd.DataFrame) -> None:
    emb = load_text_features_full("embedding", full["date"], full)
    X = emb.to_numpy(float)
    ok = ~np.isnan(X).any(axis=1)
    X, y, dates = X[ok], full["y"].to_numpy()[ok], full["date"].to_numpy()[ok]
    split = full["split"].to_numpy()[ok]

    Z = StandardScaler().fit_transform(X)
    proj = {}
    proj["tsne"] = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca",
                        perplexity=30, max_iter=1000).fit_transform(Z)
    try:
        from umap import UMAP
        proj["umap"] = UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
        print("UMAP terpasang, proyeksi UMAP ikut dibuat.")
    except ImportError:
        print("UMAP tidak terpasang; hanya t-SNE yang dibuat.")

    rows = pd.DataFrame({"date": dates, "split": split,
                         "volatility_class": [CLASS_NAMES[v] for v in y]})
    for name, P in proj.items():
        rows[f"{name}_x"], rows[f"{name}_y"] = P[:, 0], P[:, 1]
    rows.to_csv(OUT / "embedding_projection.csv", index=False)

    # Plot: hanya kelas rendah dan tinggi, sesuai permintaan tugas.
    for name, P in proj.items():
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for cls, warna, label in ((0, "#2c7fb8", "volatilitas rendah"),
                                  (2, "#d95f02", "volatilitas tinggi")):
            m = y == cls
            ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.65, c=warna, label=label,
                       edgecolors="none")
        ax.set_title(f"Ruang embedding judul berita ({name.upper()})")
        ax.set_xlabel(f"{name} dimensi 1")
        ax.set_ylabel(f"{name} dimensi 2")
        ax.legend(frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"embedding_{name}.png", dpi=160)
        plt.close(fig)

    # Ukuran pemisahan: jarak antar pusat kelas dibanding sebaran dalam kelas.
    ringkas = []
    for name, P in proj.items():
        pusat = {c: P[y == c].mean(axis=0) for c in (0, 1, 2)}
        sebar = {c: np.linalg.norm(P[y == c] - pusat[c], axis=1).mean() for c in (0, 1, 2)}
        jarak = float(np.linalg.norm(pusat[0] - pusat[2]))
        ringkas.append({"proyeksi": name, "jarak_pusat_rendah_tinggi": round(jarak, 3),
                        "sebaran_rendah": round(sebar[0], 3),
                        "sebaran_tinggi": round(sebar[2], 3),
                        "rasio_pemisahan": round(jarak / ((sebar[0] + sebar[2]) / 2), 3)})
    pd.DataFrame(ringkas).to_csv(OUT / "embedding_separasi.csv", index=False)
    print("\n3.1 Pemisahan kelas di ruang proyeksi:")
    print(pd.DataFrame(ringkas).to_string(index=False))


# ============================================================
# 3.2 TINGKAT KEPENTINGAN FITUR
# ============================================================

def kepentingan_fitur(full: pd.DataFrame, train_idx, test_idx) -> None:
    tuned = json.loads(TUNING.read_text())["selected"]["E4"]
    blocks = build_blocks(full, train_idx, test_idx)
    X_train = F.stack_blocks(blocks, "train", BLOCK_ORDER)
    y = full["y"].to_numpy()

    names_per_block = feature_names()
    names = [n for b in BLOCK_ORDER for n in names_per_block[b]]
    asal = [b for b in BLOCK_ORDER for _ in names_per_block[b]]
    assert len(names) == X_train.shape[1], (len(names), X_train.shape[1])

    clf = XGBClassifier(**tuned["params"], eval_metric="mlogloss", random_state=F.SEED)
    clf.fit(X_train, y[train_idx])
    imp = clf.feature_importances_

    per_fitur = pd.DataFrame({"fitur": names, "blok": asal, "importance": imp})
    per_fitur = per_fitur.sort_values("importance", ascending=False)
    per_fitur.head(20).to_csv(OUT / "feature_importance_top20.csv", index=False)

    per_blok = (per_fitur.groupby("blok")
                .agg(n_kolom=("fitur", "size"), total=("importance", "sum"))
                .reindex(BLOCK_ORDER))
    per_blok["per_kolom"] = per_blok["total"] / per_blok["n_kolom"]
    per_blok = per_blok.round(4)
    per_blok.to_csv(OUT / "feature_importance_blok.csv")

    print("\n3.2 Kepentingan fitur Model C per blok:")
    print(per_blok.to_string())
    print("\n3.2 Sepuluh fitur teratas:")
    print(per_fitur.head(10).round(4).to_string(index=False))


# ============================================================
# VERIFIKASI GAP OVERFIT (untuk bagian 2.3)
# ============================================================

def gap_overfit() -> None:
    data = F.load_dev_data()
    y = data["y"].to_numpy()
    tuned = json.loads(TUNING.read_text())["selected"]
    setelan = {
        "fusion terkunci": dict(F.XGB_PARAMS),
        "E3 hasil tuning": {**tuned["E3"]["params"], "eval_metric": "mlogloss",
                            "random_state": F.SEED},
        "E4 hasil tuning": {**tuned["E4"]["params"], "eval_metric": "mlogloss",
                            "random_state": F.SEED},
    }
    blok = {"fusion terkunci": BLOCK_ORDER, "E3 hasil tuning": ["H", "PRICE", "EMB", "NER"],
            "E4 hasil tuning": BLOCK_ORDER}

    rows = []
    for label, params in setelan.items():
        latih, evaluasi = [], []
        for tr, ev in F.get_folds(len(data)):
            b = F.fold_blocks(tr, ev, data=data)
            X_tr = F.stack_blocks(b, "train", blok[label])
            X_ev = F.stack_blocks(b, "eval", blok[label])
            clf = XGBClassifier(**params).fit(X_tr, y[tr])
            latih.append(accuracy_score(y[tr], clf.predict(X_tr)))
            evaluasi.append(accuracy_score(y[ev], clf.predict(X_ev)))
        rows.append({"setelan": label, "akurasi_latih": round(float(np.mean(latih)), 4),
                     "akurasi_evaluasi": round(float(np.mean(evaluasi)), 4),
                     "gap_poin": round((np.mean(latih) - np.mean(evaluasi)) * 100, 1)})
    pd.DataFrame(rows).to_csv(OUT / "overfit_gap.csv", index=False)
    print("\n2.3 Verifikasi gap latih terhadap evaluasi (CV dev):")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    full = canonical_split()
    full["date"] = pd.to_datetime(full["date"]).dt.normalize()
    full = full.sort_values("date").reset_index(drop=True)
    full["y"] = full["volatility_class"].map(F.CLASS_TO_INT).astype(int)
    train_idx = np.flatnonzero((full["split"] == "train").to_numpy())
    test_idx = np.flatnonzero((full["split"] == "test").to_numpy())

    print(f"Total hari: {len(full)} | train {len(train_idx)} | test {len(test_idx)}")
    visualisasi_embedding(full)
    kepentingan_fitur(full, train_idx, test_idx)
    gap_overfit()
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()