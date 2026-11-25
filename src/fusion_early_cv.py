"""
fusion_early_cv.py
==================
Bagian 4 dokumen "Prompt Early & Late Fusion -- Revisi": EARLY FUSION.

Semua fitur digabung jadi SATU tabel lalu masuk SATU XGBoost (setelan tetap),
dievaluasi dengan walk-forward CV 5 fold di 937 hari pengembangan.
Semua fitur per fold HANYA lewat fusion_common.fold_blocks(). Tidak ada baris test.

Kandidat (semua dilaporkan, tidak ada yang dipilih):
  E0 = H + PRICE                       = 13 kolom
  E1 = E0 + EMB                        = 13 + N_EMB
  E2 = E1 + TOPIC                      = 21 + N_EMB
  E3 = E1 + NER                        = 13 + N_EMB + N_NER
  E4 = E1 + TOPIC + NER                = 21 + N_EMB + N_NER
Baseline: naive (kelas kemarin), H (argmax probabilitas H), A (XGBoost PRICE saja).

Output: data/processed/fusion_cv/early/  (format bagian 7)
Cara pakai: python src/fusion_early_cv.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import fusion_common as F  # noqa: E402

PIPELINE = "early"
OUT = F.ROOT / "data/processed/fusion_cv" / PIPELINE

CANDIDATES = {
    "E0": ["H", "PRICE"],
    "E1": ["H", "PRICE", "EMB"],
    "E2": ["H", "PRICE", "EMB", "TOPIC"],
    "E3": ["H", "PRICE", "EMB", "NER"],
    "E4": ["H", "PRICE", "EMB", "TOPIC", "NER"],
}
EXPECTED_COLS = {
    "E0": 13,
    "E1": 13 + F.N_EMB,
    "E2": 21 + F.N_EMB,
    "E3": 13 + F.N_EMB + F.N_NER,
    "E4": 21 + F.N_EMB + F.N_NER,
}
BASELINE_COLS = {"naive": 1, "H": 3, "A": 10}

FOLD_COLS = ["pipeline", "model", "fold", "part", "n_rows", "date_start", "date_end",
             "macro_f1", "accuracy", "balanced_accuracy", "log_loss",
             "f1_low", "f1_medium", "f1_high"]
OOF_COLS = ["pipeline", "model", "fold", "date", "y_true", "p_low", "p_medium", "p_high", "y_pred"]
WEIGHT_COLS = ["pipeline", "model", "fold", "w_kurs", "w_embedding", "w_topik", "w_ner",
               "inner_oof_rows", "inner_log_loss"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(F.ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> list[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(F.ROOT), "status", "--short"], text=True)
        return [l for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def fold_row(model, k, part, dates, y_true, proba) -> dict:
    return {"pipeline": PIPELINE, "model": model, "fold": k, "part": part,
            "n_rows": len(y_true), "date_start": pd.Timestamp(dates.min()).date(),
            "date_end": pd.Timestamp(dates.max()).date(), **F.score(y_true, proba)}


def oof_rows(model, k, dates, y_true, proba) -> pd.DataFrame:
    p = np.round(np.asarray(proba, float), F.PROBA_DECIMALS)
    return pd.DataFrame({
        "pipeline": PIPELINE, "model": model, "fold": k,
        "date": pd.to_datetime(dates).strftime("%Y-%m-%d"), "y_true": y_true,
        "p_low": p[:, 0], "p_medium": p[:, 1], "p_high": p[:, 2],
        "y_pred": np.asarray(proba).argmax(1),
    })


def g3_global_checks(data: pd.DataFrame) -> list[str]:
    """Cek G3 yang tidak per fold. Assert yang gagal menghentikan script."""
    notes = []
    assert data["date"].max() <= F.DEV_END
    assert (F.classify_vol(data["realized_vol"]) == data["y"].to_numpy()).all()
    notes.append("label volatility_edges() = volatility_class (937 hari)")
    ner = F.load_text_features("ner", data["date"])
    const = [c for c in ner.columns if ner[c].nunique(dropna=False) <= 1]
    assert not const, f"kolom NER konstan di 937 hari dev: {const}"
    assert ner.shape[1] == F.N_NER
    notes.append(f"NER {F.N_NER} kolom, tidak ada yang konstan")
    return notes


def summarize(folds: pd.DataFrame, n_cols: dict) -> pd.DataFrame:
    ev = folds[folds.part == "eval"].pivot(index="fold", columns="model", values="macro_f1")
    acc_ev = folds[folds.part == "eval"].pivot(index="fold", columns="model", values="accuracy")
    acc_tr = folds[folds.part == "train"].pivot(index="fold", columns="model", values="accuracy")
    rows = []
    for m in list(CANDIDATES) + list(BASELINE_COLS):
        f1 = ev[m]
        rows.append({
            "pipeline": PIPELINE, "model": m, "n_cols": n_cols[m],
            "cv_macro_f1_mean": f1.mean(), "cv_macro_f1_sd": f1.std(ddof=1),
            "cv_acc_mean": acc_ev[m].mean(), "train_acc_mean": acc_tr[m].mean(),
            "gap_acc_points": ((acc_tr[m] - acc_ev[m]) * 100).mean(),
            "delta_vs_H_mean": (f1 - ev["H"]).mean(), "folds_beat_H": int((f1 > ev["H"]).sum()),
            "delta_vs_E0_mean": (f1 - ev["E0"]).mean(), "folds_beat_E0": int((f1 > ev["E0"]).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = datetime.now()
    data = F.load_dev_data()
    y = data["y"].to_numpy()
    dates = data["date"].to_numpy()
    folds = F.get_folds(len(data))
    g3_notes = g3_global_checks(data)

    fold_records, oof_frames, g2_scores = [], [], []
    block_hashes, fit_rows = {}, {}
    for k, (tr, ev) in enumerate(folds, start=1):
        b = F.fold_blocks(tr, ev)
        # ---- G3 per fold ----
        assert set(dates[tr]).isdisjoint(set(dates[ev])), "tanggal eval ada di train fold"
        assert dates[ev].max() <= np.datetime64(F.DEV_END)
        for name in ("H", "EMB", "TOPIC"):
            assert b["fit_rows"][name] == len(tr), (name, b["fit_rows"][name], len(tr))
        assert b["train"]["EMB"].shape[1] == F.N_EMB and b["train"]["NER"].shape[1] == F.N_NER
        block_hashes[f"fold{k}"] = b["hash"]
        fit_rows[f"fold{k}"] = b["fit_rows"]

        # ---- kandidat E0..E4 ----
        for m, names in CANDIDATES.items():
            X_tr, X_ev = F.stack_blocks(b, "train", names), F.stack_blocks(b, "eval", names)
            assert X_tr.shape[1] == EXPECTED_COLS[m], (m, X_tr.shape[1], EXPECTED_COLS[m])
            assert np.isfinite(X_tr).all() and np.isfinite(X_ev).all()
            p_tr, p_ev = F.fit_predict(X_tr, y[tr], X_ev)
            fold_records += [fold_row(m, k, "train", dates[tr], y[tr], p_tr),
                             fold_row(m, k, "eval", dates[ev], y[ev], p_ev)]
            oof_frames.append(oof_rows(m, k, dates[ev], y[ev], p_ev))
            if m == "E4":  # G2: label train acak, setelan sama, evaluasi dengan label asli
                _, p_sh = F.fit_predict(X_tr, F.shuffled_labels(y[tr], k), X_ev)
                g2_scores.append(F.score(y[ev], p_sh)["macro_f1"])

        # ---- baseline naive, H, A ----
        base = {
            "naive": (F.naive_pred(data.iloc[tr]), F.naive_pred(data.iloc[ev])),
            "H": (b["train"]["H"], b["eval"]["H"]),
            "A": F.baseline_a(b, y[tr]),
        }
        for m, (p_tr, p_ev) in base.items():
            fold_records += [fold_row(m, k, "train", dates[tr], y[tr], p_tr),
                             fold_row(m, k, "eval", dates[ev], y[ev], p_ev)]
            oof_frames.append(oof_rows(m, k, dates[ev], y[ev], p_ev))
        print(f"fold {k}/5 selesai | train {len(tr)} eval {len(ev)}", flush=True)

    cv_folds = pd.DataFrame(fold_records)[FOLD_COLS]
    oof = pd.concat(oof_frames, ignore_index=True)[OOF_COLS]
    assert pd.to_datetime(oof["date"]).max() <= F.DEV_END
    summary = summarize(cv_folds, {**EXPECTED_COLS, **BASELINE_COLS})
    g3_notes.append("per fold: fit_rows H/EMB/TOPIC = ukuran train fold; tanggal train < eval; "
                    "tanpa NaN; tanggal maks <= 2025-12-09")
    gate = F.overfit_gate(PIPELINE, cv_folds, g2_scores, g3_ok=True, g3_note="; ".join(g3_notes))
    gate = pd.concat([gate, pd.DataFrame([{
        "pipeline": PIPELINE, "model": "E0", "check": "G4", "value": np.nan, "threshold": 1e-6,
        "status": "PERINGATAN", "note": "belum dijalankan: menunggu output late (equality_check)"}])],
        ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cv_folds.to_csv(OUT / "cv_folds.csv", index=False)
    summary.to_csv(OUT / "cv_summary.csv", index=False)
    oof.to_csv(OUT / "oof_predictions.csv", index=False, float_format="%.6f")
    gate.to_csv(OUT / "overfit_check.csv", index=False)
    pd.DataFrame(columns=WEIGHT_COLS).to_csv(OUT / "weights.csv", index=False)

    cols = F.locked_columns()
    feature_files = {k: v for k, v in F.PATHS.items() if v is not None}
    meta = {
        "pipeline": PIPELINE,
        "git_commit": git_commit(),
        "git_uncommitted_changes": git_dirty(),
        "run_at": t0.isoformat(timespec="seconds"),
        "runtime_seconds": round((datetime.now() - t0).total_seconds(), 1),
        "versions": F.library_versions(),
        "XGB_PARAMS": F.XGB_PARAMS, "PCA_EMB": F.PCA_EMB, "PCA_TOPIC": F.PCA_TOPIC,
        "EMB_AGG": F.EMB_AGG, "N_EMB": F.N_EMB, "N_NER": F.N_NER,
        "feature_files": {k: str(Path(v).relative_to(F.ROOT)) for k, v in feature_files.items()},
        "feature_files_md5": {k: F.file_md5(v) for k, v in feature_files.items()},
        "fold_sizes": [[int(len(a)), int(len(b))] for a, b in folds],
        "fit_rows": fit_rows,
        "block_hashes": block_hashes,
        "volatility_edges": [float(x) for x in F.volatility_edges()],
        **cols,
        "ner_dropped": {"has_cols": F.NER_HAS_COLS, "constant_cols": F.NER_CONSTANT_COLS},
        "candidates": CANDIDATES, "n_cols": EXPECTED_COLS,
        "g2_macro_f1_per_fold": g2_scores,
        "catatan": "fusion_common.py belum di-review orang kedua saat run ini",
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\nRINGKASAN CV (eval fold)")
    print(summary.round(4).to_string(index=False))
    print("\nMACRO-F1 PER FOLD (eval)")
    print(cv_folds[cv_folds.part == "eval"].pivot(index="model", columns="fold", values="macro_f1")
          .loc[list(CANDIDATES) + list(BASELINE_COLS)].round(4).to_string())
    print("\nGERBANG OVERFIT")
    print(gate.round(4).to_string(index=False))
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
