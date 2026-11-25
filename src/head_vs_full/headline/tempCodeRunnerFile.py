
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText


# ==================================================
# 1. CONFIGURATION
# ==================================================

DAILY_CSV = "data/processed/aligned_daily.csv"
HEADLINE_CSV = "data/processed/cnbc_clean.csv"

OUTPUT_FEATURES = (
    "data/processed/fasttext_headline_features.csv"
)
MODEL_PATH = "models/fasttext_headline.model"

DATE_COLUMN = "target_date"
HEADLINE_CANDIDATES = [
    "title_norm",
    "headline_norm",
    "title",
    "headline"
]

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10
TRAIN_RATIO = 0.80
SEED = 42
WORKERS = 4

MIN_N = 3
MAX_N = 6


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)


# ==================================================
# 3. LOAD AND ALIGN HEADLINES
# ==================================================

daily_df = pd.read_csv(DAILY_CSV)
news_df = pd.read_csv(HEADLINE_CSV)

if DATE_COLUMN not in news_df.columns:
    raise ValueError(
        f"Kolom {DATE_COLUMN} tidak ditemukan. "
        f"Kolom tersedia: {news_df.columns.tolist()}"
    )

HEADLINE_COLUMN = next(
    (
        col for col in HEADLINE_CANDIDATES
        if col in news_df.columns
    ),
    None
)

if HEADLINE_COLUMN is None:
    raise ValueError(
        "Kolom headline tidak ditemukan. "
        f"Kolom tersedia: {news_df.columns.tolist()}"
    )

print("Headline column:", HEADLINE_COLUMN)

if HEADLINE_COLUMN in ["title", "headline"]:
    print(
        "WARNING: Headline yang digunakan belum "
        "terdeteksi sebagai kolom hasil preprocessing."
    )

daily_df["date"] = pd.to_datetime(
    daily_df["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

news_df["_date"] = pd.to_datetime(
    news_df[DATE_COLUMN], errors="coerce"
).dt.strftime("%Y-%m-%d")

news_df[HEADLINE_COLUMN] = (
    news_df[HEADLINE_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)

news_df = news_df[
    news_df["_date"].notna()
    & news_df[HEADLINE_COLUMN].ne("")
].copy()

daily_headlines = (
    news_df.groupby("_date")[HEADLINE_COLUMN]
    .apply(lambda x: " ".join(x))
    .rename("headline_concat")
)

daily_df["headline_concat"] = (
    daily_df["date"]
    .map(daily_headlines)
    .fillna("")
)

print("\nJumlah tanggal:", len(daily_df))
print("Jumlah berita:", len(news_df))
print(
    "Tanggal dengan berita:",
    (daily_df["headline_concat"] != "").sum()
)


# ==================================================
# 4. TOKENIZATION
# ==================================================

def tokenize_text(text):
    if not text.strip():
        return []

    return text.lower().split()


tokenized_docs = (
    daily_df["headline_concat"]
    .apply(tokenize_text)
    .tolist()
)


# ==================================================
# 5. TEMPORAL TRAINING SPLIT
# ==================================================

split_idx = int(len(daily_df) * TRAIN_RATIO)

train_sentences = [
    tokens
    for tokens in tokenized_docs[:split_idx]
    if tokens
]

if not train_sentences:
    raise ValueError(
        "Tidak ada dokumen headline untuk training."
    )

print("\nTraining documents:", len(train_sentences))
print(
    "Training period:",
    daily_df.loc[0, "date"],
    "to",
    daily_df.loc[split_idx - 1, "date"]
)


# ==================================================
# 6. TRAIN FASTTEXT
# ==================================================

ft_model = FastText(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=1,
    negative=5,
    min_n=MIN_N,
    max_n=MAX_N,
    epochs=EPOCHS,
    seed=SEED
)

ft_model.build_vocab(
    corpus_iterable=train_sentences
)

print("\nVocabulary size:", len(ft_model.wv))

ft_model.train(
    corpus_iterable=train_sentences,
    total_examples=len(train_sentences),
    epochs=EPOCHS
)

Path(MODEL_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

ft_model.save(MODEL_PATH)


# ==================================================
# 7. DOCUMENT EMBEDDING - MEAN POOLING
# ==================================================

def get_document_embedding(tokens, model):

    vectors = []

    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue

    if not vectors:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    document_vector = np.mean(vectors, axis=0)

    represented_count = len(vectors)

    known_count = sum(
        1 for token in tokens
        if token in model.wv.key_to_index
    )

    coverage = (
        known_count / len(tokens)
        if tokens else 0.0
    )

    return document_vector, represented_count, coverage


embeddings = []
token_counts = []
vocab_coverages = []

for tokens in tokenized_docs:

    vector, count, coverage = get_document_embedding(
        tokens, ft_model
    )

    embeddings.append(vector)
    token_counts.append(count)
    vocab_coverages.append(coverage)


# ==================================================
# 8. CREATE FEATURE DATAFRAME
# ==================================================

embedding_columns = [
    f"ft_{i:03d}"
    for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["ft_token_count"] = token_counts
embedding_df["ft_vocab_coverage"] = vocab_coverages

feature_result = pd.concat(
    [
        daily_df[["date"]].reset_index(drop=True),
        embedding_df
    ],
    axis=1
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

Path(OUTPUT_FEATURES).parent.mkdir(
    parents=True,
    exist_ok=True
)

feature_result.to_csv(
    OUTPUT_FEATURES,
    index=False
)

print("\nFASTTEXT HEADLINE COMPLETED")
print("Output shape:", feature_result.shape)
print("Output:", OUTPUT_FEATURES)
print("Model:", MODEL_PATH)
print(feature_result.head())
