"""
word2vec_per_article.py (headline)
==================================
Bagian 3B langkah 2a untuk pemenang langkah 1 = word2vec_headline.

File harian lama (word2vec_headline_features.csv) = rata-rata SEMUA token known
dari semua judul hari itu, dan modelnya (models/word2vec_headline.model) tidak
ada di repo. Jadi file per judul yang konsisten dengan A0 lama tidak mungkin dibuat.
Keputusan pengguna (25 Nov 2026): latih ulang word2vec dengan setelan yang sama,
buat vektor PER JUDUL, dan A0 baru = rata-rata vektor judul per tanggal.

Setelan sama dengan src/head_vs_full/headline/word2vec.py:
  Word2Vec(sg=1, vector_size=100, window=5, min_count=2, negative=5, epochs=30,
           seed=42, workers=1), dilatih di judul tanggal <= TRAIN_END.
Jalankan dengan PYTHONHASHSEED=0 supaya deterministik.

Output: data/processed/head_vs_full/headline_per_article/
    word2vec_headline_articles.csv   (date, article_id, w2v_000..w2v_099)
    word2vec_headline_A0_daily.csv   (date, w2v_000..; tanggal tanpa vektor = kosong)
    word2vec_headline_run_metadata.json
Model: models/word2vec_headline_retrained.model
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import embedding_aggregation as EA  # noqa: E402
from dataset_split import split_boundaries  # noqa: E402

DAILY_CSV = ROOT / "data/processed/head_vs_full/headline/cnbc_headline_daily.csv"
ARTICLE_CSV = ROOT / "data/processed/head_vs_full/headline/cnbc_headline_preprocessed.csv"
OLD_A0 = ROOT / "data/processed/head_vs_full/headline/word2vec_headline_features.csv"
CALENDAR_CSV = ROOT / "data/processed/kurs_clean.csv"
OUT_DIR = ROOT / "data/processed/head_vs_full/headline_per_article"
MODEL_PATH = ROOT / "models/word2vec_headline_retrained.model"
PARAMS = dict(vector_size=100, window=5, min_count=2, workers=1, sg=1, negative=5, epochs=30, seed=42)


def parse(value) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    return json.loads(value)


def mean_vector(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    return (np.mean(vecs, axis=0), len(vecs)) if vecs else (None, 0)


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "jalankan dengan PYTHONHASHSEED=0"
    train_end, _ = split_boundaries()

    # ---- latih ulang, sama persis dengan word2vec.py -------------------------
    daily = pd.read_csv(DAILY_CSV)
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("date").reset_index(drop=True)
    docs = daily["headline_tokens"].apply(parse).tolist()
    docs = [[[str(t).strip() for t in h if str(t).strip()] for h in day if isinstance(h, list)]
            for day in docs]
    docs = [[h for h in day if h] for day in docs]
    split_idx = min(max(1, int((pd.to_datetime(daily["date"]) <= train_end).sum())), len(daily))
    sentences = [h for day in docs[:split_idx] for h in day]
    model = Word2Vec(sentences=sentences, **PARAMS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"Word2Vec dilatih ulang: {len(sentences):,} judul train, vocab {len(model.wv):,}")

    # ---- cek reproduksi: rata-rata token per hari vs file lama --------------
    old = pd.read_csv(OLD_A0).set_index("date")
    vec_cols = [f"w2v_{i:03d}" for i in range(100)]
    repro = []
    for date, day in zip(daily["date"], docs):
        v, _ = mean_vector([t for h in day for t in h], model)
        if v is not None and date in old.index:
            o = old.loc[date, vec_cols].to_numpy(float)
            repro.append(float(v @ o / (np.linalg.norm(v) * np.linalg.norm(o) + 1e-12)))
    print(f"Reproduksi A0 lama (cosine harian): mean {np.mean(repro):.4f}, min {np.min(repro):.4f}")

    # ---- vektor per judul -------------------------------------------------------
    art = pd.read_csv(ARTICLE_CSV)
    art["date"] = pd.to_datetime(art["target_date"]).dt.normalize()
    art["tokens"] = art["headline_tokens"].apply(parse)
    rows, vecs, n_known = [], [], []
    for _, r in art.iterrows():
        v, k = mean_vector(r["tokens"], model)
        if v is not None:
            rows.append((r["date"], r["url"]))
            vecs.append(v)
            n_known.append(k)
    per_article = pd.concat([pd.DataFrame(rows, columns=["date", "article_id"]),
                             pd.DataFrame(np.vstack(vecs), columns=vec_cols)], axis=1)
    per_article = per_article.sort_values(["date", "article_id"]).reset_index(drop=True)
    dropped = len(art) - len(per_article)
    print(f"Judul dengan vektor: {len(per_article):,} dari {len(art):,} (tanpa token known: {dropped})")

    # judul per hari harus sama dengan file harian lama
    cnt_daily = pd.Series([len(d) for d in docs], index=pd.to_datetime(daily["date"]))
    cnt_art = art.loc[art["tokens"].map(len) > 0].groupby("date").size()
    mismatch = int((cnt_daily[cnt_daily > 0].reindex(cnt_art.index).fillna(-1) != cnt_art).sum())
    print(f"Tanggal dengan jumlah judul berbeda dari file harian: {mismatch}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    art_path = OUT_DIR / "word2vec_headline_articles.csv"
    per_article.assign(date=per_article["date"].dt.strftime("%Y-%m-%d")).to_csv(
        art_path, index=False, float_format="%.8f")

    # ---- A0 baru = rata-rata judul per hari (fungsi yang sama dengan langkah 2) ----
    EA._load_articles_cached.cache_clear()
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(CALENDAR_CSV, usecols=["date"])["date"])
                                ).normalize().drop_duplicates().sort_values()
    a_df, cols = EA.load_articles(art_path)
    a0 = EA.a0(a_df, cols, calendar)
    a0.index.name = "date"
    a0_path = OUT_DIR / "word2vec_headline_A0_daily.csv"
    a0.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(
        a0_path, index=False, float_format="%.8f")

    meta = {"params": PARAMS, "train_end": str(train_end.date()), "train_headlines": len(sentences),
            "vocab": len(model.wv), "repro_cosine_mean": float(np.mean(repro)),
            "repro_cosine_min": float(np.min(repro)), "articles_with_vector": len(per_article),
            "articles_without_known_tokens": dropped, "dates_count_mismatch": mismatch,
            "article_vector": "mean token known per judul", "daily_A0": "mean vektor judul per target_date",
            "calendar_dates_without_vector": int(a0.isna().any(axis=1).sum())}
    (OUT_DIR / "word2vec_headline_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
