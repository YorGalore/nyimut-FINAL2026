
import html
import json
import re
from pathlib import Path

import pandas as pd


# =========================================================
# PATH
# =========================================================

INDEX_PATH = Path("data/raw/cnbc_index.csv")
SELECTED_PATH = Path("data/processed/news_tfidf_clean.csv")
DAILY_PATH = Path("data/processed/aligned_daily.csv")

ARTICLE_OUTPUT = Path("data/processed/cnbc_headline_preprocessed.csv")
DAILY_OUTPUT = Path("data/processed/cnbc_headline_daily.csv")


# =========================================================
# URL NORMALIZATION
# =========================================================

def normalize_url(url):
    if pd.isna(url):
        return None

    url = str(url).strip().lower()

    # Hapus query parameter dan fragment
    url = url.split("?")[0].split("#")[0]

    # Samakan trailing slash
    return url.rstrip("/")


# =========================================================
# HEADLINE PREPROCESSING
# =========================================================

def clean_headline(text):
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    # Hapus HTML tag dan URL
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalisasi istilah
    text = re.sub(r"\bu\.s\.?\b", "us", text, flags=re.IGNORECASE)

    # Lowercase
    text = text.lower()

    # Hapus tanda baca, pertahankan apostrof dan angka
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Rapikan whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 1. LOAD DATA
# =========================================================

index = pd.read_csv(INDEX_PATH)
selected = pd.read_csv(SELECTED_PATH)

required_index = {"title", "url"}
required_selected = {"url", "target_date"}

if not required_index.issubset(index.columns):
    raise ValueError("cnbc_index.csv harus memiliki kolom title dan url.")

if not required_selected.issubset(selected.columns):
    raise ValueError(
        "File seleksi harus memiliki kolom url dan target_date. "
        "Gunakan file artikel-level, bukan aligned_daily.csv."
    )


# =========================================================
# 2. NORMALIZE URL
# =========================================================

index["url_key"] = index["url"].apply(normalize_url)
selected["url_key"] = selected["url"].apply(normalize_url)

# Hilangkan duplikasi URL pada katalog CNBC
index = index.drop_duplicates(subset="url_key", keep="first")


# =========================================================
# 3. MATCH SELECTED ARTICLES WITH ORIGINAL HEADLINES
# =========================================================

headline_lookup = index[["url_key", "title"]].rename(
    columns={"title": "headline_raw"}
)

merged = selected.merge(
    headline_lookup,
    on="url_key",
    how="left",
    validate="many_to_one",
    indicator=True
)

matched = merged["_merge"] == "both"

print(f"Jumlah berita terseleksi : {len(selected)}")
print(f"Headline ditemukan       : {matched.sum()}")
print(f"Headline tidak ditemukan : {(~matched).sum()}")

if (~matched).any():
    print("\nContoh URL yang tidak ditemukan:")
    print(merged.loc[~matched, "url"].head(5).to_string(index=False))


# Hanya gunakan berita yang headline aslinya ditemukan
articles = merged.loc[matched].copy()

# Gunakan target_date dari dataset seleksi, bukan date dari index
articles["target_date"] = pd.to_datetime(
    articles["target_date"]
).dt.strftime("%Y-%m-%d")


# =========================================================
# 4. PREPROCESS HEADLINES
# =========================================================

articles["headline_clean"] = articles["headline_raw"].apply(
    clean_headline
)

articles["headline_tokens"] = articles["headline_clean"].apply(
    lambda text: text.split()
)

# Simpan artikel-level
article_output = articles[
    [
        "url",
        "target_date",
        "headline_raw",
        "headline_clean",
        "headline_tokens"
    ]
].copy()

article_output["headline_tokens"] = article_output[
    "headline_tokens"
].apply(json.dumps)

ARTICLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
article_output.to_csv(ARTICLE_OUTPUT, index=False)


# =========================================================
# 5. AGGREGATE HEADLINES BY TARGET DATE
# =========================================================

calendar = pd.read_csv(DAILY_PATH, usecols=["date"])
calendar["date"] = pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d")

# Satu tanggal berisi kumpulan headline, masing-masing berupa token
daily_lookup = (
    articles.groupby("target_date")["headline_tokens"]
    .apply(list)
    .to_dict()
)

daily = pd.DataFrame({"date": calendar["date"]})

daily["headline_tokens"] = daily["date"].map(
    lambda date: daily_lookup.get(date, [])
)

daily["headline_count"] = daily["headline_tokens"].apply(len)

daily["headline_tokens"] = daily["headline_tokens"].apply(
    json.dumps
)

daily.to_csv(DAILY_OUTPUT, index=False)


# =========================================================
# 6. SUMMARY
# =========================================================

print("\nPreprocessing selesai.")
print(f"Artikel-level : {ARTICLE_OUTPUT}")
print(f"Daily-level   : {DAILY_OUTPUT}")
print(f"Jumlah artikel dengan headline: {len(articles)}")
print(f"Jumlah tanggal observasi      : {len(daily)}")
