"""
tune_model_c.py
===============
Tahap 1 Tugas 3: penyesuaian hyperparameter untuk seluruh tangga ablasi.

Semua kandidat (A, E0, E1, E2, E3, E4) di-tuning dengan grid YANG SAMA
(XGB_GRID di dataset_split.py, 27 kombinasi) dan aturan seleksi yang sama
(akurasi valid tertinggi), sesuai Keputusan 41. Dengan begitu selisih antar
model mencerminkan kontribusi fitur, bukan perbedaan upaya tuning.

Tidak ada baris test yang dimuat. Fitur hanya diambil lewat
fusion_common.fold_blocks(), yang menolak tanggal > 2025-12-09.

  Protokol utama        P1 : latih di train (772 hari), pilih di valid (165 hari).
  Protokol sensitivitas P2 : walk-forward CV 5 fold di 937 hari dev.

Cara pakai (dari root proyek):
    ./.venv311/bin/python src/tune_model_c.py

Output: data/processed/tuning/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import fusion_common as F  # noqa: E402
from dataset_split import XGB_GRID  # noqa: E402

OUT = F.ROOT / "data" / "processed" / "tuning"

# Tangga ablasi: tiap anak tangga menambah satu sumber fitur.
LADDER = {
    "A": ["PRICE"],
    "E0": ["H", "PRICE"],
    "E1": ["H", "PRICE", "EMB"],
    "E2": ["H", "PRICE", "EMB", "TOPIC"],
    "E3": ["H", "PRICE", "EMB", "NER"],
    "E4": ["H", "PRICE", "EMB", "TOPIC", "NER"],
}


def main() -> None:
    data = F.load_dev_data()
    y = data["y"].to_numpy()
    assert len(data) == F.N_DEV and data["date"].max() == F.DEV_END

    protocols = {
        "P1_valid": [(np.arange(F.N_TRAIN), np.arange(F.N_TRAIN, F.N_DEV))],
        "P2_wfcv": F.get_folds(len(data)),
    }

    cache: dict = {}

    def blocks(tr, ev):
        key = (tr[0], tr[-1], ev[0], ev[-1])
        if key not in cache:
            cache[key] = F.fold_blocks(tr, ev, data=data)
        return cache[key]

    rows = []
    for protocol, splits in protocols.items():
        for model, names in LADDER.items():
            first = blocks(*splits[0])
            n_cols = F.stack_blocks(first, "train", names).shape[1]
            for grid_id, params in enumerate(XGB_GRID):
                accuracies, macro_f1s = [], []
                for train_idx, eval_idx in splits:
                    b = blocks(train_idx, eval_idx)
                    X_train = F.stack_blocks(b, "train", names)
                    X_eval = F.stack_blocks(b, "eval", names)
                    clf = XGBClassifier(**params, eval_metric="mlogloss", random_state=F.SEED)
                    clf.fit(X_train, y[train_idx])
                    pred = clf.predict(X_eval)
                    accuracies.append(accuracy_score(y[eval_idx], pred))
                    macro_f1s.append(f1_score(y[eval_idx], pred, average="macro"))
                rows.append({
                    "protocol": protocol, "model": model, "grid_id": grid_id, **params,
                    "n_cols": n_cols,
                    "accuracy": float(np.mean(accuracies)),
                    "macro_f1": float(np.mean(macro_f1s)),
                })
            print(f"{protocol} {model}: {len(XGB_GRID)} kombinasi selesai", flush=True)

    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "grid_results.csv", index=False)

    # Seleksi: akurasi valid tertinggi; kalau seri, kombinasi pertama di grid
    # (aturan yang sama dengan tune_xgb() di dataset_split.py).
    selected = {}
    for model in LADDER:
        candidates = results[(results.protocol == "P1_valid") & (results.model == model)]
        best = candidates.sort_values(["accuracy", "grid_id"], ascending=[False, True]).iloc[0]
        same = results[(results.protocol == "P2_wfcv") & (results.model == model)
                       & (results.grid_id == best.grid_id)]
        selected[model] = {
            "blocks": LADDER[model],
            "n_cols": int(best.n_cols),
            "params": {
                "n_estimators": int(best.n_estimators),
                "max_depth": int(best.max_depth),
                "learning_rate": float(best.learning_rate),
            },
            "valid_accuracy": float(best.accuracy),
            "valid_macro_f1": float(best.macro_f1),
            "cv_accuracy_same_params": float(same.accuracy.iloc[0]),
        }

    meta = {
        "catatan": "Tuning memakai data dev saja. Test belum dibuka.",
        "grid_size": len(XGB_GRID),
        "seleksi": "akurasi valid tertinggi; seri -> kombinasi pertama (sama dengan tune_xgb)",
        "n_train": F.N_TRAIN, "n_valid": F.N_VALID, "dev_end": str(F.DEV_END.date()),
        "feature_files": {k: str(Path(v).relative_to(F.ROOT))
                          for k, v in F.PATHS.items() if v is not None},
        "selected": selected,
    }
    (OUT / "selected_params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Kombinasi terpilih (seleksi di valid) ===")
    for model, info in selected.items():
        p = info["params"]
        print(f"{model:3s} n_cols={info['n_cols']:3d}  "
              f"n={p['n_estimators']:3d} d={p['max_depth']} lr={p['learning_rate']}  "
              f"valid acc {info['valid_accuracy']:.4f}  "
              f"cv acc {info['cv_accuracy_same_params']:.4f}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()