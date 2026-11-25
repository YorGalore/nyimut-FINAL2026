"""
contextual_per_article.py
=========================
Bagian 3B langkah 0c (item #5 bagian 2 dokumen revisi).

Ekstraksi ulang embedding contextual FULLTEXT PER ARTIKEL dari raw text
(title + description + body di cnbc_raw.csv), sesuai aturan:
  - chunk 512 token (termasuk special token), overlap 50 token;
  - mean pooling token konten per chunk (fungsi mean_pooling di contextual.py);
  - vektor artikel = rata-rata (mean) vektor chunk-nya;
  - file harian A0 = rata-rata vektor artikel per target_date
    (kalender JISDOR, tanggal tanpa berita = 0, sama seperti file lama).

Seleksi artikel dan target_date sama dengan contextual.py (news_tfidf_clean.csv).
File lama di data/processed/head_vs_full/fulltext/ TIDAK disentuh.

Output: data/processed/head_vs_full/fulltext_per_article/
    {model}_fulltext_articles.csv   (date, article_id, dim_000..)
    {model}_fulltext_A0_daily.csv   (date, dim_000..)
    {model}_fulltext_run_metadata.json

Cara pakai:  python src/head_vs_full/fulltext/contextual_per_article.py --model all
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contextual as CX  # noqa: E402  (fungsi tokenisasi/pooling yang sama)

ROOT = HERE.parents[2]
RAW_NEWS_CSV = ROOT / "data/raw/cnbc_raw.csv"
ARTICLE_DATES_CSV = ROOT / "data/processed/news_tfidf_clean.csv"
CALENDAR_CSV = ROOT / "data/processed/kurs_clean.csv"
OUT_DIR = ROOT / "data/processed/head_vs_full/fulltext_per_article"

MODELS = CX.MODELS  # finbert, roberta, qwen -- sama dengan contextual.py
MAX_LENGTH = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = {"finbert": 16, "roberta": 16, "qwen": 4}
DEVICE = CX.DEVICE


def load_articles() -> pd.DataFrame:
    text_cols = ["title", "description", "body"]
    raw = pd.read_csv(RAW_NEWS_CSV, usecols=["url", *text_cols], keep_default_na=False)
    raw = raw.drop_duplicates(subset="url", keep="first")
    dates = pd.read_csv(ARTICLE_DATES_CSV, usecols=["url", "target_date"], parse_dates=["target_date"])
    art = dates.merge(raw, on="url", how="left", validate="one_to_one", indicator=True)
    if art["_merge"].eq("left_only").any():
        raise ValueError("Ada URL terseleksi yang tidak ada di cnbc_raw.csv")
    art["text"] = art[text_cols].apply(
        lambda parts: "\n".join(p for p in map(str.strip, parts) if p), axis=1
    )
    art["date"] = art["target_date"].dt.normalize()
    art = art.rename(columns={"url": "article_id"})
    art = art.sort_values(["date", "article_id"]).reset_index(drop=True)
    assert not art["article_id"].duplicated().any()
    return art[["date", "article_id", "text"]]


def make_chunks(texts, tokenizer):
    tokenizer.model_max_length = 10**9
    prefix, suffix = CX.infer_special_token_wrapper(tokenizer)
    content_len = MAX_LENGTH - len(prefix) - len(suffix)
    step = content_len - CHUNK_OVERLAP
    chunks, owner = [], []
    for doc_idx, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False) if text else []
        start = 0
        while start < len(ids):
            end = min(start + content_len, len(ids))
            body = ids[start:end]
            chunks.append({
                "input_ids": prefix + body + suffix,
                "special_tokens_mask": [1] * len(prefix) + [0] * len(body) + [1] * len(suffix),
            })
            owner.append(doc_idx)
            if end >= len(ids):
                break
            start += step
    return chunks, np.asarray(owner), content_len, step


@torch.inference_mode()
def embed_chunks(model, tokenizer, chunks, hidden, batch_size):
    order = np.argsort([len(c["input_ids"]) for c in chunks])  # kurangi padding
    out = np.zeros((len(chunks), hidden), dtype=np.float32)
    t0 = time.time()
    n_batches = (len(order) + batch_size - 1) // batch_size
    for b, s in enumerate(range(0, len(order), batch_size)):
        idx = order[s:s + batch_size]
        ids, att, spec = CX.prepare_batch([chunks[i] for i in idx], tokenizer)
        ids, att, spec = ids.to(DEVICE), att.to(DEVICE), spec.to(DEVICE)
        hs = model(input_ids=ids, attention_mask=att, return_dict=True).last_hidden_state
        out[idx] = CX.mean_pooling(hs, att, spec).float().cpu().numpy()
        if b % 100 == 0 or b == n_batches - 1:
            mem = torch.cuda.max_memory_allocated() // 2**20 if torch.cuda.is_available() else 0
            print(f"  batch {b:,}/{n_batches:,} | {time.time() - t0:.0f}s | peak {mem} MB", flush=True)
    return out


def process(name: str, model_id: str, art: pd.DataFrame, calendar: pd.Series) -> None:
    t0 = time.time()
    print(f"\n=== {name} ({model_id}) | device {DEVICE} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(DEVICE).eval()
    hidden = int(model.config.hidden_size)

    chunks, owner, content_len, step = make_chunks(art["text"].tolist(), tok)
    print(f"Artikel {len(art):,} | chunk {len(chunks):,} | content_len {content_len} | step {step}",
          flush=True)
    emb = embed_chunks(model, tok, chunks, hidden, BATCH_SIZE[name])

    # vektor artikel = mean vektor chunk
    sums = np.zeros((len(art), hidden), dtype=np.float64)
    np.add.at(sums, owner, emb)
    counts = np.bincount(owner, minlength=len(art))
    assert (counts > 0).all(), "ada artikel tanpa chunk"
    art_vec = (sums / counts[:, None]).astype(np.float32)
    assert np.isfinite(art_vec).all()

    cols = [f"dim_{i:03d}" for i in range(hidden)]
    per_article = pd.concat(
        [art[["date", "article_id"]].reset_index(drop=True),
         pd.DataFrame(art_vec, columns=cols)], axis=1)
    daily = per_article.groupby("date")[cols].mean()
    daily = daily.reindex(calendar).fillna(0.0)
    daily.index.name = "date"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_article.assign(date=per_article["date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUT_DIR / f"{name}_fulltext_articles.csv", index=False, float_format="%.7g")
    daily.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUT_DIR / f"{name}_fulltext_A0_daily.csv", index=False, float_format="%.7g")
    meta = {
        "model": name, "model_id": model_id, "source": "cnbc_raw.csv title+description+body",
        "max_length": MAX_LENGTH, "chunk_overlap": CHUNK_OVERLAP, "content_len": content_len,
        "chunk_pooling": "mean token konten", "article_pooling": "mean antar chunk",
        "daily_pooling": "mean antar artikel per target_date",
        "dtype": "float32", "n_articles": int(len(art)), "n_chunks": int(len(chunks)),
        "chunks_per_article_mean": float(counts.mean()), "hidden_size": hidden,
        "calendar_dates": int(len(daily)), "seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / f"{name}_fulltext_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    del model, tok, emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["all", *MODELS], default="all")
    args = ap.parse_args()
    art = load_articles()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(CALENDAR_CSV, usecols=["date"])["date"]).dt.normalize()
    ).drop_duplicates().sort_values()
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        process(name, MODELS[name], art, calendar)


if __name__ == "__main__":
    main()
