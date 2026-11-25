"""
stage_a_cv.py
=============
Bagian 3B dokumen "Prompt Early & Late Fusion -- Revisi": tahap A
(pemilihan varian embedding + topik, lalu cara agregasi untuk pemenang).

Semua fitur lewat fusion_common.fold_blocks(); model = make_xgb(); 5 fold
TimeSeriesSplit di 937 hari dev. Tidak ada baris/label test.

Cara pakai (dari root project):
    python src/stage_a_cv.py --step inventory
    python src/stage_a_cv.py --step 1
    python src/stage_a_cv.py --step 2 --winner <variant> --articles <path per artikel>

Output: data/processed/stage_a_cv/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import embedding_aggregation as EA  # noqa: E402
import fusion_common as F  # noqa: E402

ROOT = F.ROOT
OUT = ROOT / "data/processed/stage_a_cv"
AGG_DIR = OUT / "embedding_aggregations"
HV = ROOT / "data/processed/head_vs_full"

EMB_VARIANTS = {
    # word2vec_headline menang di putaran 2 langkah 1 dengan file lama (rata-rata token
    # per hari, model tidak ada di repo). Dilatih ulang dengan setelan sama (reproduksi
    # cosine 1,0000), A0 diganti ke rata-rata vektor per judul supaya cek 2a bisa lolos.
    # Keputusan pengguna, 25 Nov 2026.
    "word2vec_headline": HV / "headline_per_article/word2vec_headline_A0_daily.csv",
    "fasttext_headline": HV / "headline/fasttext_headline_features.csv",
    "doc2vec_headline": HV / "headline/doc2vec_headline_features.csv",
    "finbert_headline": HV / "headline/finbert_headline_features.csv",
    "roberta_headline": HV / "headline/roberta_headline_features.csv",
    "qwen_headline": HV / "headline/qwen_headline_features.csv",
    "word2vec_fulltext": HV / "fulltext/word2vec_fulltext_features.csv",
    "fasttext_fulltext": HV / "fulltext/fasttext_fulltext_features.csv",
    "doc2vec_fulltext": HV / "fulltext/doc2vec_fulltext_features.csv",
    # Fulltext contextual: file eccc8f1 (raw text, digabung per hari, overlap 64).
    # Ekstraksi ulang per artikel (overlap 50) untuk FinBERT memberi hasil praktis sama
    # (cosine harian rata-rata 0,992; macro-F1 CV 0,789 vs 0,796), jadi RoBERTa/Qwen
    # tidak diekstraksi ulang. Keputusan pengguna, 25 Nov 2026; dicatat di run_meta.
    # finbert_fulltext menang putaran 3 dengan file eccc8f1; diganti ke A0 per artikel
    # (overlap 50) yang sudah diekstraksi. Keputusan pengguna, 25 Nov 2026.
    "finbert_fulltext": HV / "fulltext_per_article/finbert_fulltext_A0_daily.csv",
    "roberta_fulltext": HV / "fulltext/roberta_fulltext_features.csv",
    # Qwen fulltext menang di langkah 1 versi file eccc8f1; langkah 2 butuh file per
    # artikel yang konsisten dengan A0 (cek 2a), jadi A0-nya diganti ke ekstraksi per
    # artikel (overlap 50) dan langkah 1 diulang. Keputusan pengguna, 25 Nov 2026.
    "qwen_fulltext": HV / "fulltext_per_article/qwen_fulltext_A0_daily.csv",
}
FULLTEXT_NOTE = {
    "keputusan": "fulltext contextual memakai file eccc8f1 (per hari, overlap 64), tanpa ekstraksi ulang",
    "bukti_finbert": {"cosine_harian_mean": 0.9916, "cosine_harian_p5": 0.9669,
                      "macro_f1_cv_lama": 0.7957, "macro_f1_cv_per_artikel": 0.7894},
    "file_per_artikel_finbert": "data/processed/head_vs_full/fulltext_per_article/finbert_fulltext_articles.csv",
    "pengecualian_qwen": ("qwen_fulltext menang langkah 1 dengan file eccc8f1 (macro-F1 0,7978); "
                          "A0-nya diganti ke ekstraksi per artikel overlap 50 supaya langkah 2 "
                          "(cek 2a) bisa jalan, lalu langkah 1 diulang"),
    "langkah1_awal_qwen_eccc8f1": {"cv_macro_f1_mean": 0.7978, "cv_log_loss_mean": 0.5473,
                                   "folds_beat_H": 4},
    "langkah1_putaran2_word2vec_headline_lama": {"cv_macro_f1_mean": 0.7961, "cv_log_loss_mean": 0.5451,
                                                 "folds_beat_H": 2,
                                                 "catatan": "A0 lama = rata-rata token per hari; "
                                                            "diganti ke rata-rata per judul (putaran 3)"},
    "qwen_fulltext_per_artikel_putaran2": {"cv_macro_f1_mean": 0.7940, "cv_log_loss_mean": 0.5481},
}
TOPIC_VARIANTS = {
    "lda_train_only": ROOT / "data/processed/topic_modelling/lda_train_only/daily_topic_features.csv",
}
TOPIC_EXCLUDED = {
    "bertopic": "BERTopic.fit_transform(documents) di semua dokumen (topic_models.py run_bertopic)",
    "dtm": "LdaSeqModel di semua time slice termasuk valid/test (topic_models.py run_dtm)",
    "top2vec": "Top2Vec(documents) di semua dokumen (topic_models.py run_top2vec)",
}

FOLD_COLS = ["step", "variant", "fold", "part", "n_rows", "macro_f1", "accuracy",
             "balanced_accuracy", "log_loss", "f1_low", "f1_medium", "f1_high"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CV SATU VARIAN
# ---------------------------------------------------------------------------
def run_variant(step: str, variant: str, paths: dict, block: str, emb_agg: str | None = None):
    data = F.load_dev_data()
    y = data["y"].to_numpy()
    rows, h_rows, n_cols = [], [], None
    for k, (tr, ev) in enumerate(F.get_folds(len(data)), start=1):
        b = F.fold_blocks(tr, ev, paths=paths, emb_agg=emb_agg, data=data)
        names = ["H", "PRICE", block]
        X_tr, X_ev = F.stack_blocks(b, "train", names), F.stack_blocks(b, "eval", names)
        n_cols = X_tr.shape[1]
        assert b["fit_rows"]["H"] == len(tr)
        p_tr, p_ev = F.fit_predict(X_tr, y[tr], X_ev)
        for part, idx, p in (("train", tr, p_tr), ("eval", ev, p_ev)):
            rows.append({"step": step, "variant": variant, "fold": k, "part": part,
                         "n_rows": len(idx), **F.score(y[idx], p)})
        if emb_agg == "A3":
            print(f"    {variant} fold {k}/5 selesai ({datetime.now():%H:%M:%S})", flush=True)
        h_rows.append({"step": step, "variant": "H", "fold": k, "part": "eval",
                       "n_rows": len(ev), **F.score(y[ev], b["eval"]["H"])})
    return pd.DataFrame(rows), pd.DataFrame(h_rows), n_cols


def summarize(folds: pd.DataFrame, h: pd.DataFrame, n_cols: dict) -> pd.DataFrame:
    ev = folds[folds.part == "eval"]
    h_f1 = h.drop_duplicates("fold").set_index("fold")["macro_f1"]
    out = []
    for (step, variant), g in ev.groupby(["step", "variant"], sort=False):
        f1 = g.set_index("fold")["macro_f1"]
        out.append({
            "step": step, "variant": variant, "n_cols": n_cols.get(variant),
            "cv_macro_f1_mean": f1.mean(), "cv_macro_f1_sd": f1.std(ddof=1),
            "cv_macro_f1_median": f1.median(), "cv_log_loss_mean": g["log_loss"].mean(),
            "delta_vs_H_mean": (f1 - h_f1).mean(), "folds_beat_H": int((f1 > h_f1).sum()),
        })
    return pd.DataFrame(out)


def pick_winner(summary: pd.DataFrame) -> str:
    s = summary.sort_values(["cv_macro_f1_mean", "cv_log_loss_mean"], ascending=[False, True])
    return s.iloc[0]["variant"]


# ---------------------------------------------------------------------------
# LANGKAH 0d: INVENTARIS
# ---------------------------------------------------------------------------
def inventory() -> pd.DataFrame:
    dev_dates = set(F.load_dev_data()["date"])
    rows = []
    files = [("embedding", k, v) for k, v in EMB_VARIANTS.items()]
    files += [("topik", k, v) for k, v in TOPIC_VARIANTS.items()]
    art_files = {
        "finbert_headline": ROOT / "data/processed/temporal_aggregation/article_embeddings/finbert_headline_articles.csv",
        "finbert_fulltext": HV / "fulltext_per_article/finbert_fulltext_articles.csv",
        "roberta_fulltext": HV / "fulltext_per_article/roberta_fulltext_articles.csv",
        "qwen_fulltext": HV / "fulltext_per_article/qwen_fulltext_articles.csv",
    }
    files += [("per_artikel", k, v) for k, v in art_files.items()]
    for kind, name, path in files:
        if not Path(path).exists():
            rows.append({"kind": kind, "variant": name, "path": str(path.relative_to(ROOT)),
                         "exists": False})
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        if kind == "embedding":
            used = F.embedding_columns(df)
        elif kind == "topik":
            used = F.topic_columns(df)
        else:
            used = [c for c in df.columns if F.EMB_COL_RE.match(c)]
        covers = dev_dates.issubset(set(df["date"].dt.normalize()))
        rows.append({
            "kind": kind, "variant": name, "path": str(Path(path).relative_to(ROOT)), "exists": True,
            "rows": len(df), "cols_total": df.shape[1], "cols_used": len(used),
            "date_min": df["date"].min().date(), "date_max": df["date"].max().date(),
            "covers_937_dev": covers, "md5": F.file_md5(path),
        })
        if kind != "per_artikel":
            assert covers, f"{name} tidak menutup 937 hari dev"
    inv = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT / "stage_a_inventory.csv", index=False)
    return inv


# ---------------------------------------------------------------------------
# LANGKAH 1
# ---------------------------------------------------------------------------
def step1():
    all_folds, n_cols, h = [], {}, None
    for name, path in EMB_VARIANTS.items():
        print(f"[1a] {name}", flush=True)
        f, h, n_cols[name] = run_variant("1a", name, {"embedding": path}, "EMB")
        all_folds.append(f)
    for name, path in TOPIC_VARIANTS.items():
        print(f"[1b] {name}", flush=True)
        f, h, n_cols[name] = run_variant("1b", name, {"topik": path}, "TOPIC")
        all_folds.append(f)
    folds = pd.concat(all_folds, ignore_index=True)
    summary = summarize(folds, h, n_cols)
    for step in ("1a", "1b"):
        win = pick_winner(summary[summary.step == step])
        summary.loc[summary.step == step, "selected"] = summary.loc[summary.step == step, "variant"] == win
    h_summary = summarize(h.assign(step="ref"), h, {"H": 3}).assign(selected=False)
    write_outputs(folds, h, pd.concat([summary, h_summary], ignore_index=True), step="1")
    return summary


# ---------------------------------------------------------------------------
# LANGKAH 2
# ---------------------------------------------------------------------------
def trailing_test(method: str, art_path, dates) -> float:
    """Ubah acak semua artikel setelah t; vektor hari t tidak boleh berubah."""
    art, cols = EA.load_articles(art_path)
    rng = np.random.default_rng(F.SEED)
    t = pd.DatetimeIndex(dates)[len(dates) // 2]
    base = EA.daily_features(method, art_path, [t])
    pert = art.copy()
    after = pert["date"] > t
    pert.loc[after, cols] = rng.normal(size=(int(after.sum()), len(cols)))
    fn = {"A1": EA.a1, "A2": EA.a2}[method[:2]]
    arg = int(method.split("-")[1]) if method.startswith("A1") else float(method.split("hl")[1])
    new = fn(pert, cols, [t], arg)
    diff = float(np.nanmax(np.abs(base.to_numpy() - new.to_numpy())))
    assert diff == 0.0, f"{method} tidak trailing (selisih {diff})"
    return diff


def step2(winner: str, art_path: Path):
    data = F.load_dev_data()
    a0_path = EMB_VARIANTS[winner]
    # 2a. konsistensi file per artikel vs file A0 langkah 1
    a0_file = F.load_text_features("embedding", data["date"], a0_path)
    art, art_cols = EA.load_articles(art_path)
    a0_art = EA.a0(art, art_cols, data["date"])
    a, b = a0_file.to_numpy(float), a0_art.to_numpy(float)
    assert a.shape == b.shape, (a.shape, b.shape)
    diff_2a = float(np.nanmax(np.abs(a - b)))
    print(f"[2a] selisih maks A0 file vs rata-rata per artikel: {diff_2a:.2e}", flush=True)
    assert diff_2a <= 1e-4, "A0 langkah 1 dan file per artikel tidak konsisten -- berhenti."

    # 2c. uji trailing
    trailing = {m: trailing_test(m, art_path, data["date"]) for m in ["A1-3", "A1-7", "A1-14", "A2-hl1", "A2-hl3"]}
    print(f"[2c] uji trailing lolos: {trailing}", flush=True)

    # file harian semua cara (kecuali A3) untuk semua tanggal kalender (tanpa label)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT / "data/processed/kurs_clean.csv",
                                                      usecols=["date"])["date"]).dt.normalize()
                           ).drop_duplicates().sort_values()
    paths = {}
    for m in EA.METHODS:
        if m == "A0":
            paths[m] = a0_path  # A0 = file langkah 1 persis
        elif m != "A3":
            df = EA.daily_features(m, art_path, cal)
            df.index.name = "date"
            p = AGG_DIR / f"{winner}_{m}.csv"
            df.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(
                p, index=False, float_format="%.7g")
            paths[m] = p

    # 2d. CV 8 cara
    all_folds, n_cols, h = [], {}, None
    for m in EA.METHODS:
        print(f"[2d] {winner} {m}", flush=True)
        if m == "A3":
            f, h, n_cols[m] = run_variant("2", m, {"embedding_articles": art_path}, "EMB", emb_agg="A3")
        else:
            f, h, n_cols[m] = run_variant("2", m, {"embedding": paths[m]}, "EMB", emb_agg=m)
        all_folds.append(f)
    folds = pd.concat(all_folds, ignore_index=True)
    summary = summarize(folds, h, n_cols)
    # 2e. aturan pilih: pemenang dipakai hanya kalau >= 1 poin di atas A0
    best = pick_winner(summary)
    f1 = summary.set_index("variant")["cv_macro_f1_mean"]
    chosen = best if (f1[best] - f1["A0"]) >= 0.01 else "A0"
    summary["selected"] = summary["variant"] == chosen
    write_outputs(folds, h, summary, step="2", extra={
        "winner_1a": winner, "articles_path": str(Path(art_path).relative_to(ROOT)),
        "diff_2a": diff_2a, "trailing_test": trailing, "best_raw": best, "chosen": chosen,
        "daily_paths": {m: str(Path(p).relative_to(ROOT)) for m, p in paths.items()},
        "a3_settings": {"epochs": EA.A3_EPOCHS, "lr": EA.A3_LR, "weight_decay": EA.A3_WEIGHT_DECAY,
                        "seed": EA.A3_SEED, "device": "cpu", "trained": "per fold, train fold saja"},
        "a4_categories": EA.EVENT_CATEGORIES,
    })
    return summary, chosen


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def write_outputs(folds, h, summary, step: str, extra: dict | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    fpath, spath = OUT / "stage_a_folds.csv", OUT / "stage_a_summary.csv"
    folds = pd.concat([folds, h.assign(step=folds["step"].iloc[0])], ignore_index=True)[FOLD_COLS]
    if fpath.exists():
        old = pd.read_csv(fpath, dtype={"step": str})
        folds = pd.concat([old[~old.step.isin(folds.step.unique())], folds], ignore_index=True)
    if spath.exists():
        old = pd.read_csv(spath, dtype={"step": str})
        drop = {"1": ["1a", "1b", "ref"], "2": ["2"]}[step]
        summary = pd.concat([old[~old.step.isin(drop)], summary], ignore_index=True)
    folds.to_csv(fpath, index=False)
    summary.to_csv(spath, index=False)

    mpath = OUT / "run_meta.json"
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    data = F.load_dev_data()
    meta.update({
        "git_commit": git_commit(), "versions": F.library_versions(),
        "XGB_PARAMS": F.XGB_PARAMS, "PCA_EMB": F.PCA_EMB, "PCA_TOPIC": F.PCA_TOPIC,
        "fold_sizes": [[int(len(a)), int(len(b))] for a, b in F.get_folds(len(data))],
        "volatility_edges": [float(x) for x in F.volatility_edges()],
    })
    meta[f"step_{step}"] = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "variant_files": ({k: {"path": str(v.relative_to(ROOT)), "md5": F.file_md5(v)}
                           for k, v in {**EMB_VARIANTS, **TOPIC_VARIANTS}.items()} if step == "1" else None),
        "topic_excluded": TOPIC_EXCLUDED if step == "1" else None,
        "fulltext_contextual": FULLTEXT_NOTE if step == "1" else None,
        "selection_rule": ("macro-F1 CV tertinggi; tie log-loss terkecil" if step == "1" else
                           "macro-F1 CV tertinggi; dipakai hanya jika >= 1 poin di atas A0"),
        **(extra or {}),
    }
    mpath.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["inventory", "1", "2"], required=True)
    ap.add_argument("--winner")
    ap.add_argument("--articles")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.step == "inventory":
        print(inventory().to_string(index=False))
    elif args.step == "1":
        print(step1().round(4).to_string(index=False))
    else:
        art = Path(args.articles)
        art = (art if art.is_absolute() else ROOT / art).resolve()
        summary, chosen = step2(args.winner, art)
        print(summary.round(4).to_string(index=False))
        print(f"Dipilih: {chosen}")


if __name__ == "__main__":
    main()
