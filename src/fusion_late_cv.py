"""
fusion_late_cv.py
=================
Bagian 5 dokumen "Prompt Early & Late Fusion -- Revisi": late fusion dengan
walk-forward CV 5 fold di 937 hari pengembangan (train + valid). Tidak ada baris
atau label test (tanggal > 2025-12-09) yang dimuat.

Late fusion = setiap sumber punya XGBoost sendiri (make_xgb(), setelan tetap),
lalu probabilitasnya digabung dengan weighted average. Bobot dicari HANYA dari
OOF fold dalam (TimeSeriesSplit(3) di dalam train fold luar), tidak pernah dari
eval fold luar.

  base model : kurs = H + PRICE (13) | embedding = EMB (N_EMB)
               topik = TOPIC (8)     | ner = NER (N_NER)
  kandidat   : E0 = kurs (tanpa bobot), E1 = kurs + embedding,
               E2 = E1 + topik, E3 = E1 + ner, E4 = E1 + topik + ner

Semua fitur per fold (luar dan dalam) HANYA lewat fusion_common.fold_blocks().
Script ini tidak menghitung scaler, PCA, H, agregasi, atau batas tertile sendiri.

Cara pakai (dari root project):
    python src/fusion_late_cv.py          # CV + gerbang overfit G1-G3, W1-W4
    python src/fusion_late_cv.py --g4     # G4, setelah output early juga ada

Output: data/processed/fusion_cv/late/  (format bagian 7)
"""

from __future__ import annotations

import argparse
import itertools
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

ROOT = F.ROOT
PIPELINE = "late"
OUT = ROOT / "data/processed/fusion_cv" / PIPELINE
EARLY_OUT = ROOT / "data/processed/fusion_cv/early"
G4_OUT = ROOT / "data/processed/fusion_cv/g4_equality_check.csv"

# ---------------------------------------------------------------------------
# DEFINISI LATE (bagian 5)
# ---------------------------------------------------------------------------
SOURCES = ["kurs", "embedding", "topik", "ner"]           # urutan = urutan tie-break
SOURCE_BLOCKS = {"kurs": ["H", "PRICE"], "embedding": ["EMB"], "topik": ["TOPIC"], "ner": ["NER"]}
SOURCE_NCOLS = {"kurs": 13, "embedding": F.N_EMB, "topik": F.PCA_TOPIC, "ner": F.N_NER}
CANDIDATES = {
    "E0": ["kurs"],
    "E1": ["kurs", "embedding"],
    "E2": ["kurs", "embedding", "topik"],
    "E3": ["kurs", "embedding", "ner"],
    "E4": ["kurs", "embedding", "topik", "ner"],
}
WEIGHTED = ["E1", "E2", "E3", "E4"]
INNER_SPLITS = 3
GRID_STEPS = 20          # bobot 0.00, 0.05, ..., 1.00
TIE_TOL = 1e-12
CLIP = 1e-15
G2_CANDIDATE = "E4"

FOLD_COLS = ["pipeline", "model", "fold", "part", "n_rows", "date_start", "date_end",
             "macro_f1", "accuracy", "balanced_accuracy", "log_loss",
             "f1_low", "f1_medium", "f1_high"]
SUMMARY_COLS = ["pipeline", "model", "n_cols", "cv_macro_f1_mean", "cv_macro_f1_sd", "cv_acc_mean",
                "train_acc_mean", "gap_acc_points", "delta_vs_H_mean", "folds_beat_H",
                "delta_vs_E0_mean", "folds_beat_E0"]
OOF_COLS = ["pipeline", "model", "fold", "date", "y_true", "p_low", "p_medium", "p_high", "y_pred"]
WEIGHT_COLS = ["pipeline", "model", "fold", "w_kurs", "w_embedding", "w_topik", "w_ner",
               "inner_oof_rows", "inner_log_loss"]

N_COLS = {
    **{m: sum(SOURCE_NCOLS[s] for s in src) for m, src in CANDIDATES.items()},
    **{f"simple_avg_{m}": sum(SOURCE_NCOLS[s] for s in CANDIDATES[m]) for m in WEIGHTED},
    **{f"base_{s}": n for s, n in SOURCE_NCOLS.items()},
    "naive": 1, "H": 3, "A": len(F.PRICE_FEATURES),
}
# jumlah kolom sesuai bagian 4/5 (sama dengan early kandidat yang bernama sama)
assert N_COLS["E0"] == 13
assert N_COLS["E1"] == 13 + F.N_EMB
assert N_COLS["E2"] == 21 + F.N_EMB
assert N_COLS["E3"] == 13 + F.N_EMB + F.N_NER
assert N_COLS["E4"] == 21 + F.N_EMB + F.N_NER


# ---------------------------------------------------------------------------
# BASE MODEL DAN PENGGABUNGAN
# ---------------------------------------------------------------------------
def source_matrix(blocks: dict, part: str, source: str) -> np.ndarray:
    X = F.stack_blocks(blocks, part, SOURCE_BLOCKS[source])
    assert X.shape[1] == SOURCE_NCOLS[source], (source, X.shape)
    assert np.isfinite(X).all(), f"NaN/inf di matriks {source} ({part})"
    return X


def fit_bases(blocks: dict, y_train) -> dict:
    """4 base model make_xgb() -> {source: (proba_train, proba_eval)}."""
    return {s: F.fit_predict(source_matrix(blocks, "train", s), y_train,
                             source_matrix(blocks, "eval", s))
            for s in SOURCES}


def combine(probas: dict, weights: dict) -> np.ndarray:
    """Weighted average, clip ke [1e-15, 1], normalisasi ulang per baris."""
    p = sum(weights[s] * probas[s] for s in weights if weights[s] > 0)
    p = np.clip(p, CLIP, 1.0)
    return p / p.sum(1, keepdims=True)


def weight_grid(n_sources: int) -> np.ndarray:
    """Semua bobot integer (satuan 1/GRID_STEPS) dengan jumlah = GRID_STEPS."""
    rows = [c + (GRID_STEPS - sum(c),)
            for c in itertools.product(range(GRID_STEPS + 1), repeat=n_sources - 1)
            if sum(c) <= GRID_STEPS]
    return np.array(rows, dtype=int)


def search_weights(oof: dict, y: np.ndarray, sources: list[str]) -> tuple[dict, float]:
    """
    Grid bobot di OOF fold dalam -> log-loss terkecil.
    Tie (selisih < 1e-12): w_kurs terbesar, lalu w_embedding, lalu w_topik.
    """
    grid = weight_grid(len(sources))                             # (m, s) integer
    P = np.stack([oof[s] for s in sources])                      # (s, n, 3)
    comb = np.einsum("ms,snc->mnc", grid / GRID_STEPS, P)
    comb = np.clip(comb, CLIP, 1.0)
    comb = comb / comb.sum(2, keepdims=True)
    ll = -np.log(comb[:, np.arange(len(y)), y]).mean(1)          # (m,)
    full = np.zeros((len(grid), len(SOURCES)), dtype=int)        # kolom urut SOURCES
    for j, s in enumerate(sources):
        full[:, SOURCES.index(s)] = grid[:, j]
    tied = np.flatnonzero(ll - ll.min() < TIE_TOL)
    # lexsort: kunci terakhir = kunci utama; negatif -> terbesar dulu
    order = np.lexsort((-full[tied, 2], -full[tied, 1], -full[tied, 0]))
    best = tied[order[0]]
    weights = {s: full[best, i] / GRID_STEPS for i, s in enumerate(SOURCES)}
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    return weights, float(ll[best])


def late_procedure(blocks: dict, inner: list, y_train: np.ndarray, candidates: list[str]) -> dict:
    """
    Prosedur late lengkap untuk satu fold luar.
    blocks : fold_blocks(train fold luar, eval fold luar)
    inner  : [(pos_train, pos_eval, fold_blocks dalam)], pos relatif ke train fold luar
    y_train: label train fold luar (asli, atau acak untuk G2)
    """
    oof = {s: [] for s in SOURCES}
    y_oof = []
    for pos_tr, pos_ev, ib in inner:
        bases = fit_bases(ib, y_train[pos_tr])
        for s in SOURCES:
            oof[s].append(bases[s][1])
        y_oof.append(y_train[pos_ev])
    oof = {s: np.vstack(v) for s, v in oof.items()}
    y_oof = np.concatenate(y_oof)

    weights = {}
    for m in candidates:
        w, ll = search_weights(oof, y_oof, CANDIDATES[m])
        weights[m] = {**w, "inner_oof_rows": int(len(y_oof)), "inner_log_loss": ll}

    bases = fit_bases(blocks, y_train)  # seluruh train fold luar
    out = {"weights": weights, "bases": bases, "cand": {}, "simple": {}}
    for m in candidates:
        w = {s: weights[m][s] for s in SOURCES}
        eq = {s: (1.0 / len(CANDIDATES[m]) if s in CANDIDATES[m] else 0.0) for s in SOURCES}
        out["cand"][m] = tuple(combine({s: bases[s][i] for s in SOURCES}, w) for i in (0, 1))
        out["simple"][m] = tuple(combine({s: bases[s][i] for s in SOURCES}, eq) for i in (0, 1))
    return out


# ---------------------------------------------------------------------------
# G3: CEK BOCOR DAN DATA
# ---------------------------------------------------------------------------
class G3Log:
    def __init__(self):
        self.n = 0

    def check(self, cond: bool, msg: str):
        assert cond, f"G3 GAGAL: {msg}"
        self.n += 1


def g3_blocks(g3: G3Log, blocks: dict, train_idx, eval_idx, tag: str):
    d_tr = pd.DatetimeIndex(blocks["dates"]["train"])
    d_ev = pd.DatetimeIndex(blocks["dates"]["eval"])
    g3.check(max(d_tr.max(), d_ev.max()) <= F.DEV_END, f"{tag}: tanggal > {F.DEV_END.date()}")
    g3.check(len(d_tr.intersection(d_ev)) == 0 and d_tr.max() < d_ev.min(),
             f"{tag}: tanggal eval ada di train fold")
    for name in ("H", "EMB", "TOPIC"):
        g3.check(blocks["fit_rows"][name] == len(train_idx),
                 f"{tag}: {name} di-fit dengan {blocks['fit_rows'][name]} baris, train fold {len(train_idx)}")
    widths = {"H": 3, "PRICE": len(F.PRICE_FEATURES), "EMB": F.N_EMB, "TOPIC": F.PCA_TOPIC, "NER": F.N_NER}
    for part, idx in (("train", train_idx), ("eval", eval_idx)):
        for name, w in widths.items():
            arr = blocks[part][name]
            g3.check(arr is not None and arr.shape == (len(idx), w), f"{tag}: bentuk blok {name} {part}")
            g3.check(np.isfinite(arr).all(), f"{tag}: NaN di blok {name} {part}")


def g3_global(g3: G3Log, data: pd.DataFrame):
    g3.check(data["date"].max() <= F.DEV_END and len(data) == F.N_DEV, "data dev bukan 937 hari")
    g3.check((F.classify_vol(data["realized_vol"]) == data["y"].to_numpy()).all(),
             "label volatility_edges() != volatility_class")
    cols = F.locked_columns()
    g3.check(len(cols["NER_COLS"]) == F.N_NER, "len(NER_COLS) != N_NER")
    ner = F.load_text_features("ner", data["date"])
    const = [c for c in ner.columns if ner[c].nunique(dropna=False) <= 1]
    g3.check(not const, f"kolom NER konstan di 937 hari dev: {const}")


# ---------------------------------------------------------------------------
# METRIK DAN RINGKASAN
# ---------------------------------------------------------------------------
def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files": [l[3:] for l in run("status", "--porcelain").splitlines()]}


def normalize(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1.0)
    return p / p.sum(1, keepdims=True)


class Recorder:
    def __init__(self, data: pd.DataFrame):
        self.data, self.y = data, data["y"].to_numpy()
        self.folds, self.oof = [], []

    def add(self, model: str, k: int, part: str, idx, proba):
        proba = np.asarray(proba, float)
        assert proba.shape == (len(idx), 3) and np.isfinite(proba).all(), model
        d = self.data["date"].iloc[idx]
        self.folds.append({"pipeline": PIPELINE, "model": model, "fold": k, "part": part,
                           "n_rows": len(idx), "date_start": d.min().strftime("%Y-%m-%d"),
                           "date_end": d.max().strftime("%Y-%m-%d"), **F.score(self.y[idx], proba)})
        if part == "eval":
            self.oof.append(pd.DataFrame({
                "pipeline": PIPELINE, "model": model, "fold": k,
                "date": d.dt.strftime("%Y-%m-%d").to_numpy(), "y_true": self.y[idx],
                "p_low": proba[:, 0], "p_medium": proba[:, 1], "p_high": proba[:, 2],
                "y_pred": normalize(proba).argmax(1)}))


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    ev = folds[folds.part == "eval"].set_index(["model", "fold"])
    tr = folds[folds.part == "train"].set_index(["model", "fold"])
    f1 = ev["macro_f1"].unstack("fold")
    acc_ev, acc_tr = ev["accuracy"].unstack("fold"), tr["accuracy"].unstack("fold")
    rows = []
    for m in folds["model"].drop_duplicates():
        d_h, d_e0 = f1.loc[m] - f1.loc["H"], f1.loc[m] - f1.loc["E0"]
        rows.append({
            "pipeline": PIPELINE, "model": m, "n_cols": N_COLS[m],
            "cv_macro_f1_mean": f1.loc[m].mean(), "cv_macro_f1_sd": f1.loc[m].std(ddof=1),
            "cv_acc_mean": acc_ev.loc[m].mean(), "train_acc_mean": acc_tr.loc[m].mean(),
            "gap_acc_points": ((acc_tr.loc[m] - acc_ev.loc[m]) * 100).mean(),
            "delta_vs_H_mean": d_h.mean(), "folds_beat_H": int((d_h > 0).sum()),
            "delta_vs_E0_mean": d_e0.mean(), "folds_beat_E0": int((d_e0 > 0).sum()),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_cv():
    t0 = datetime.now()
    data = F.load_dev_data()
    y = data["y"].to_numpy()
    rec = Recorder(data)
    g3 = G3Log()
    g3_global(g3, data)

    outer = F.get_folds(len(data))
    weight_rows, g2_f1, g2_weights = [], [], {}
    block_hashes, inner_sizes = {}, {}
    for k, (tr, ev) in enumerate(outer, start=1):
        print(f"[fold {k}/5] train {len(tr)}, eval {len(ev)} ({datetime.now():%H:%M:%S})", flush=True)
        blocks = F.fold_blocks(tr, ev, data=data)
        g3_blocks(g3, blocks, tr, ev, f"fold {k}")
        block_hashes[f"fold{k}"] = blocks["hash"]

        # fold dalam: fitur di-fit ulang di train fold dalam (dipakai juga oleh G2)
        inner = []
        for pos_tr, pos_ev in F.get_folds(len(tr), n_splits=INNER_SPLITS):
            ib = F.fold_blocks(tr[pos_tr], tr[pos_ev], data=data)
            g3_blocks(g3, ib, pos_tr, pos_ev, f"fold {k} dalam")
            inner.append((pos_tr, pos_ev, ib))
        inner_sizes[f"fold{k}"] = [[len(a), len(b)] for a, b, _ in inner]

        # prosedur late dengan label asli
        res = late_procedure(blocks, inner, y[tr], WEIGHTED)
        bases = res["bases"]
        rec.add("E0", k, "train", tr, bases["kurs"][0])
        rec.add("E0", k, "eval", ev, bases["kurs"][1])
        for m in WEIGHTED:
            for i, (part, idx) in enumerate((("train", tr), ("eval", ev))):
                rec.add(m, k, part, idx, res["cand"][m][i])
            weight_rows.append({"pipeline": PIPELINE, "model": m, "fold": k,
                                **{f"w_{s}": res["weights"][m][s] for s in SOURCES},
                                "inner_oof_rows": res["weights"][m]["inner_oof_rows"],
                                "inner_log_loss": res["weights"][m]["inner_log_loss"]})
        for m in WEIGHTED:
            for i, (part, idx) in enumerate((("train", tr), ("eval", ev))):
                rec.add(f"simple_avg_{m}", k, part, idx, res["simple"][m][i])
        for s in SOURCES:
            for i, (part, idx) in enumerate((("train", tr), ("eval", ev))):
                rec.add(f"base_{s}", k, part, idx, bases[s][i])

        # baseline di fold yang sama
        a_tr, a_ev = F.baseline_a(blocks, y[tr])
        for part, idx, p_naive, p_h, p_a in (
                ("train", tr, F.naive_pred(data.iloc[tr]), blocks["train"]["H"], a_tr),
                ("eval", ev, F.naive_pred(data.iloc[ev]), blocks["eval"]["H"], a_ev)):
            rec.add("naive", k, part, idx, p_naive)
            rec.add("H", k, part, idx, p_h)
            rec.add("A", k, part, idx, p_a)

        # G2: seluruh prosedur E4 (cari bobot + 4 base model) dengan label train acak
        y_shuf = F.shuffled_labels(y[tr], k)
        res2 = late_procedure(blocks, inner, y_shuf, [G2_CANDIDATE])
        g2_f1.append(F.score(y[ev], res2["cand"][G2_CANDIDATE][1])["macro_f1"])
        g2_weights[f"fold{k}"] = {s: res2["weights"][G2_CANDIDATE][s] for s in SOURCES}

    folds = pd.DataFrame(rec.folds, columns=FOLD_COLS)
    oof = pd.concat(rec.oof, ignore_index=True)[OOF_COLS]
    weights = pd.DataFrame(weight_rows, columns=WEIGHT_COLS)
    summary = summarize(folds)

    # sanity: E0 late = base kurs persis
    e0 = oof[oof.model == "E0"][["p_low", "p_medium", "p_high"]].to_numpy()
    bk = oof[oof.model == "base_kurs"][["p_low", "p_medium", "p_high"]].to_numpy()
    assert np.array_equal(e0, bk)
    g3.check(pd.to_datetime(oof["date"]).max() <= F.DEV_END, "OOF berisi tanggal test")

    g3_note = (f"{g3.n} assert lolos: tanggal <= {F.DEV_END.date()}, fit_rows H/EMB/TOPIC = ukuran "
               f"train fold (luar + dalam), tidak ada tanggal eval di train, label edges = "
               f"volatility_class, tidak ada NaN, NER tidak konstan, N_EMB={F.N_EMB}, N_NER={F.N_NER}")
    gate = F.overfit_gate(PIPELINE, folds, g2_f1, True, g3_note, weights)

    # ---- tulis output ----
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "cv_folds.csv", index=False)
    summary.to_csv(OUT / "cv_summary.csv", index=False)
    oof.round(F.PROBA_DECIMALS).to_csv(OUT / "oof_predictions.csv", index=False,
                                       float_format=f"%.{F.PROBA_DECIMALS}f")
    gate.to_csv(OUT / "overfit_check.csv", index=False)
    weights.to_csv(OUT / "weights.csv", index=False)

    cols = F.locked_columns()
    feature_files = {k: v for k, v in F.PATHS.items() if v is not None}
    meta = {
        "pipeline": PIPELINE,
        "git_commit": git_info()["commit"], "git": git_info(),
        "run_at": t0.isoformat(timespec="seconds"),
        "runtime_seconds": round((datetime.now() - t0).total_seconds(), 1),
        "versions": F.library_versions(),
        "XGB_PARAMS": F.XGB_PARAMS, "PCA_EMB": F.PCA_EMB, "PCA_TOPIC": F.PCA_TOPIC,
        "EMB_AGG": F.EMB_AGG, "N_EMB": F.N_EMB, "N_NER": F.N_NER,
        "feature_files": {k: {"path": Path(v).relative_to(ROOT).as_posix(), "md5": F.file_md5(v)}
                          for k, v in feature_files.items()},
        "feature_files_md5": {k: F.file_md5(v) for k, v in feature_files.items()},
        "fold_sizes": [[int(len(a)), int(len(b))] for a, b in outer],
        "block_hashes": block_hashes,
        "volatility_edges": [float(x) for x in F.volatility_edges()],
        **cols,
        "NER_DROPPED": {"has": F.NER_HAS_COLS, "constant_937_dev": F.NER_CONSTANT_COLS},
        "late": {
            "sources": SOURCE_BLOCKS, "source_ncols": SOURCE_NCOLS, "candidates": CANDIDATES,
            "inner_splits": INNER_SPLITS, "inner_fold_sizes": inner_sizes,
            "weight_grid": f"0..1 langkah 1/{GRID_STEPS}, jumlah = 1",
            "objective": "log-loss OOF fold dalam (clip 1e-15, normalisasi ulang)",
            "tie_break": "selisih < 1e-12: w_kurs terbesar, lalu w_embedding, lalu w_topik",
            "simple_average": "bobot sama antar sumber kandidat, pembanding, bukan kandidat",
        },
        "G2": {"candidate": G2_CANDIDATE, "seed": "default_rng(42 + k)",
               "macro_f1_per_fold": g2_f1, "weights_per_fold": g2_weights},
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return folds, summary, weights, gate


def report(summary: pd.DataFrame, weights: pd.DataFrame, gate: pd.DataFrame):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = summary.drop(columns="pipeline").copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].round(4)
    print("\n=== cv_summary (late) ===")
    print(show.to_string(index=False))
    print("\n=== bobot per fold ===")
    print(weights.drop(columns="pipeline").round(4).to_string(index=False))
    print("\n=== gerbang overfit ===")
    print(gate.drop(columns="pipeline").round(4).to_string(index=False))
    print(f"\nOutput: {OUT.relative_to(ROOT)}/")


def run_g4():
    assert (EARLY_OUT / "run_meta.json").exists(), f"output early belum ada di {EARLY_OUT}"
    g4 = F.equality_check(EARLY_OUT, OUT)
    g4.to_csv(G4_OUT, index=False)
    print(g4.to_string(index=False))
    print(f"\nG4: {'LOLOS' if (g4.status == 'LOLOS').all() else 'GAGAL'} -> {G4_OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", action="store_true", help="jalankan G4 (butuh output early dan late)")
    args = ap.parse_args()
    if args.g4:
        run_g4()
        return
    _, summary, weights, gate = run_cv()
    report(summary, weights, gate)


if __name__ == "__main__":
    main()
