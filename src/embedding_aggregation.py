"""
embedding_aggregation.py
========================
Bagian 3B langkah 2b: SATU modul untuk semua cara agregasi temporal embedding.
Di-import oleh stage_a_cv.py dan fusion_common.py (fold_blocks untuk A3).

Input: file embedding PER ARTIKEL dengan kolom date, article_id, <vektor>.
Satuan "hari" = HARI KALENDER. Semua cara hanya memakai artikel bertanggal <= t.

  A0      rata-rata vektor artikel bertanggal t.
  A1-w    rata-rata A0 di tanggal berita d dengan t-w+1 <= d <= t (w = 3, 7, 14).
          Tanggal tanpa berita TIDAK dihitung sebagai vektor nol.
  A2-hl   rata-rata tertimbang per artikel, bobot 0.5 ** (umur_hari / hl), hl = 1, 3.
  A3      attention pooling (arsitektur + setelan tetap, seed 42), dilatih ULANG per
          fold hanya dengan artikel + label baris train fold (a3_fold_vectors).
  A4      rata-rata vektor per kategori kejadian + count_<kategori> (jumlah artikel).
          Kategori multi-label dari build_article_embeddings.category_labels().

Hari tanpa artikel (A0/A3/A4) atau tanpa artikel di jendela (A1/A2) -> NaN
(fusion_common.fit_text_pca mengisinya 0 SETELAH PCA).

Beda dengan src/temporal_aggregation/temporal_aggregation.py (kode lama):
  - kode lama menghitung A1/A2 per BARIS kalender JISDOR (hari bursa) dan
    memasukkan vektor nol untuk hari bursa tanpa berita;
  - kode lama melatih A3 sekali di periode train, bukan per fold.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

VECTOR_COL_RE = re.compile(r"^[a-z0-9]+_\d{3}$")  # finbert_000, dim_000, ...
METHODS = ["A0", "A1-3", "A1-7", "A1-14", "A2-hl1", "A2-hl3", "A3", "A4"]
EVENT_CATEGORIES = [
    "military_conflict", "economic_sanctions", "trade_war",
    "diplomatic_relations", "central_bank_policy", "other",
]
EVENT_CATEGORIES_CSV = ROOT / "data/processed/stage_a_cv/article_event_categories.csv"

# A3: arsitektur dan setelan tetap (sama dengan temporal_aggregation.py lama)
A3_EPOCHS = 100
A3_LR = 1e-3
A3_WEIGHT_DECAY = 1e-4
A3_SEED = 42


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_articles_cached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["article_id"] = df["article_id"].astype(str)
    assert not df["article_id"].duplicated().any(), "article_id harus unik"
    return df.sort_values(["date", "article_id"]).reset_index(drop=True)


def load_articles(path) -> tuple[pd.DataFrame, list[str]]:
    df = _load_articles_cached(str(path))
    cols = [c for c in df.columns if VECTOR_COL_RE.match(c)]
    assert cols, f"tidak ada kolom vektor di {path}"
    return df, cols


def build_event_categories() -> pd.DataFrame:
    """Kategori kejadian per artikel (article_id = url), aturan build_article_embeddings."""
    sys.path.insert(0, str(SRC / "temporal_aggregation"))
    import build_article_embeddings as BAE

    frame = BAE.load_articles()  # sudah menghitung event_category
    out = frame[["article_id", "date", "event_category"]].copy()
    EVENT_CATEGORIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EVENT_CATEGORIES_CSV, index=False)
    return out


@lru_cache(maxsize=1)
def event_categories() -> pd.Series:
    if not EVENT_CATEGORIES_CSV.exists():
        build_event_categories()
    df = pd.read_csv(EVENT_CATEGORIES_CSV)
    return df.set_index(df["article_id"].astype(str))["event_category"].fillna("other")


# ---------------------------------------------------------------------------
# AGREGASI HARIAN (A0, A1, A2, A4)
# ---------------------------------------------------------------------------
def _daily_sums(art: pd.DataFrame, cols: list[str]):
    g = art.groupby("date")
    return g[cols].sum(), g.size()


def a0(art, cols, dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    return art.groupby("date")[cols].mean().reindex(dates)


def a1(art, cols, dates, window: int) -> pd.DataFrame:
    """Rata-rata A0 di tanggal berita dalam jendela kalender [t-w+1, t]."""
    dates = pd.DatetimeIndex(dates)
    daily = art.groupby("date")[cols].mean().sort_index()
    news_dates = daily.index.to_numpy()
    values = daily.to_numpy(float)
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    out = np.full((len(dates), len(cols)), np.nan)
    t = dates.to_numpy()
    lo = np.searchsorted(news_dates, t - np.timedelta64(window - 1, "D"), side="left")
    hi = np.searchsorted(news_dates, t, side="right")
    n = hi - lo
    ok = n > 0
    out[ok] = (csum[hi[ok]] - csum[lo[ok]]) / n[ok, None]
    return pd.DataFrame(out, index=dates, columns=cols)


def a2(art, cols, dates, half_life: float) -> pd.DataFrame:
    """Rata-rata per artikel dengan bobot 0.5 ** (umur hari kalender / half_life)."""
    dates = pd.DatetimeIndex(dates)
    sums, counts = _daily_sums(art, cols)
    start = min(sums.index.min(), dates.min())
    end = max(dates.max(), start)
    cal = pd.date_range(start, end, freq="D")
    S = sums.reindex(cal, fill_value=0.0).to_numpy(float)
    N = counts.reindex(cal, fill_value=0).to_numpy(float)
    decay = 0.5 ** (1.0 / half_life)
    num = np.zeros_like(S)
    den = np.zeros_like(N)
    acc_num = np.zeros(S.shape[1])
    acc_den = 0.0
    for i in range(len(cal)):
        acc_num = S[i] + decay * acc_num
        acc_den = N[i] + decay * acc_den
        num[i], den[i] = acc_num, acc_den
    out = pd.DataFrame(np.nan, index=cal, columns=cols)
    ok = den > 1e-12
    out.loc[ok] = num[ok] / den[ok, None]
    return out.reindex(dates)


def a4(art, cols, dates) -> pd.DataFrame:
    """Vektor rata-rata per kategori + count_<kategori>. Hari tanpa berita -> NaN."""
    dates = pd.DatetimeIndex(dates)
    cats = event_categories()
    a = art[["date", "article_id", *cols]].copy()
    a["event_category"] = a["article_id"].map(cats)
    missing = a["event_category"].isna().sum()
    assert missing == 0, f"{missing} artikel tanpa kategori kejadian"
    a["event_category"] = a["event_category"].str.split("|")
    a = a.explode("event_category")
    has_news = pd.Index(art["date"].unique())
    pieces, counts = [], []
    for cat in EVENT_CATEGORIES:
        sub = a[a["event_category"] == cat]
        m = sub.groupby("date")[cols].mean().reindex(dates).fillna(0.0)
        m.columns = [f"{cat}__{c}" for c in cols]
        pieces.append(m)
        counts.append(sub.groupby("date").size().reindex(dates, fill_value=0).rename(f"count_{cat}"))
    out = pd.concat(pieces + counts, axis=1).astype(float)
    out.loc[~out.index.isin(has_news)] = np.nan
    return out


def daily_features(method: str, path, dates) -> pd.DataFrame:
    """Fitur harian untuk satu cara agregasi (bukan A3). Kolom vektor ber-suffix _NNN."""
    art, cols = load_articles(path)
    if method == "A0":
        return a0(art, cols, dates)
    if method.startswith("A1-"):
        return a1(art, cols, dates, int(method.split("-")[1]))
    if method.startswith("A2-hl"):
        return a2(art, cols, dates, float(method.split("hl")[1]))
    if method == "A4":
        return a4(art, cols, dates)
    raise ValueError(f"{method}: A3 dihitung per fold lewat a3_fold_vectors()")


# ---------------------------------------------------------------------------
# A3: ATTENTION, DILATIH ULANG PER FOLD
# ---------------------------------------------------------------------------
def a3_fold_vectors(path, train_dates, y_train, eval_dates):
    """
    Latih attention pooling HANYA dengan artikel + label tanggal train fold,
    lalu pool artikel untuk tanggal train dan eval. Deterministik (CPU, seed 42).
    Return (X_train, X_eval); tanggal tanpa artikel -> baris NaN.
    """
    import torch
    from torch import nn

    art, cols = load_articles(path)
    by_date = {d: g[cols].to_numpy(np.float32) for d, g in art.groupby("date")}
    train_dates = pd.DatetimeIndex(train_dates)
    eval_dates = pd.DatetimeIndex(eval_dates)
    assert train_dates.max() < eval_dates.min()
    y_train = np.asarray(y_train, int)

    torch.manual_seed(A3_SEED)
    rng = np.random.default_rng(A3_SEED)
    torch.set_num_threads(1)
    dim = len(cols)

    class AttentionPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(dim, dim)
            self.query = nn.Parameter(torch.randn(dim) * 0.02)
            self.classifier = nn.Linear(dim, 3)

        def pool(self, x):
            w = torch.softmax(torch.tanh(self.projection(x)) @ self.query, dim=0)
            return (w.unsqueeze(1) * x).sum(0)

        def forward(self, x):
            return self.classifier(self.pool(x))

    model = AttentionPool()
    opt = torch.optim.AdamW(model.parameters(), lr=A3_LR, weight_decay=A3_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    train_items = [(torch.from_numpy(by_date[d]), int(y)) for d, y in zip(train_dates, y_train)
                   if d in by_date]
    assert len(train_items) >= 10
    model.train()
    for _ in range(A3_EPOCHS):
        for i in rng.permutation(len(train_items)):
            x, y = train_items[i]
            opt.zero_grad()
            loss = loss_fn(model(x).unsqueeze(0), torch.tensor([y]))
            loss.backward()
            opt.step()

    model.eval()

    def pool_dates(dates):
        out = np.full((len(dates), dim), np.nan)
        with torch.no_grad():
            for i, d in enumerate(dates):
                if d in by_date:
                    out[i] = model.pool(torch.from_numpy(by_date[d])).numpy()
        return out

    return pool_dates(train_dates), pool_dates(eval_dates)
