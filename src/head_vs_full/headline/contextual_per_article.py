"""
contextual_per_article.py (headline)
====================================
Bagian 3B langkah 2a (item #6 bagian 2 dokumen revisi), hanya untuk pemenang
langkah 1 kalau pemenangnya varian HEADLINE contextual.

Menyimpan embedding PER JUDUL (= per artikel) dengan prosedur yang sama persis
dengan headline/contextual.py (raw title dari cnbc_raw.csv, tanpa truncation,
mean pooling token konten). File harian lama = rata-rata judul per tanggal,
jadi rata-rata harian file ini harus sama dengan file A0 (dicek di stage_a_cv 2a).

Output: data/processed/head_vs_full/headline_per_article/{model}_headline_articles.csv
Cara pakai: python src/head_vs_full/headline/contextual_per_article.py --model roberta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contextual as HC  # noqa: E402

ROOT = HERE.parents[2]
OUT_DIR = ROOT / "data/processed/head_vs_full/headline_per_article"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(HC.MODELS), required=True)
    args = ap.parse_args()
    name, model_id = args.model, HC.MODELS[args.model]

    raw = pd.read_csv(HC.RAW_NEWS_CSV, usecols=["url", "title"], keep_default_na=False)
    raw = raw.drop_duplicates(subset="url", keep="first")
    dates = pd.read_csv(HC.ARTICLE_DATES_CSV, usecols=["url", "target_date"], parse_dates=["target_date"])
    art = dates.merge(raw, on="url", how="left", validate="one_to_one")
    art["title"] = art["title"].str.strip()
    art = art.loc[art["title"].ne("")].copy()
    art["date"] = art["target_date"].dt.normalize()
    art = art.sort_values(["date", "url"]).reset_index(drop=True)

    tokenizer = HC.load_tokenizer(model_id)
    model = AutoModel.from_pretrained(model_id).to(HC.DEVICE).eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    hidden = int(model.config.hidden_size)
    emb = HC.extract_headline_embeddings(model, tokenizer, art["title"].tolist(), hidden)

    cols = [f"{name}_{i:03d}" for i in range(hidden)]
    out = pd.concat([art[["date", "url"]].rename(columns={"url": "article_id"}),
                     pd.DataFrame(emb, columns=cols)], axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_headline_articles.csv"
    out.assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False, float_format="%.8f")
    print(f"Tersimpan: {path} | {out.shape}")


if __name__ == "__main__":
    main()
