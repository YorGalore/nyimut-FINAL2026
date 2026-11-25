
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# split resmi 70/15/15 (sama dengan Tugas 2) -- lihat dataset_split.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_split import split_boundaries  # noqa: E402


# ==================================================
# 1. CONFIGURATION
# ==================================================

HEADLINE_CSV = "data/processed/head_vs_full/headline/cnbc_headline_daily.csv"

OUTPUT_FEATURES = (
    "data/processed/head_vs_full/headline/doc2vec_headline_features.csv"
)

MODEL_PATH = "models/doc2vec_headline.model"

DATE_COLUMN = "date"
TOKEN_COLUMN = "headline_tokens"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 20
INFER_EPOCHS = 30

# Doc2Vec hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()
SEED = 42
WORKERS = 4


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)


# ==================================================
# 3. LOAD PREPROCESSED HEADLINES
# ==================================================

news_df = pd.read_csv(HEADLINE_CSV)

required_columns = {
    DATE_COLUMN,
    TOKEN_COLUMN
}

if not required_columns.issubset(news_df.columns):
    raise ValueError(
        f"Kolom yang dibutuhkan tidak ditemukan. "
        f"Kolom tersedia: {news_df.columns.tolist()}"
    )

news_df[DATE_COLUMN] = pd.to_datetime(
    news_df[DATE_COLUMN],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


# ==================================================
# 4. PARSE HEADLINE TOKENS
# ==================================================

def parse_tokens(value):

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)

        if not isinstance(parsed, list):
            return []

        # Setiap tanggal berisi list headline,
        # dan setiap headline berisi list token.
        tokens = []

        for headline in parsed:
            if isinstance(headline, list):
                tokens.extend(
                    str(token)
                    for token in headline
                    if str(token).strip()
                )

        return tokens

    except (json.JSONDecodeError, TypeError):
        return []


news_df["tokens"] = news_df[TOKEN_COLUMN].apply(
    parse_tokens
)

news_df = news_df[
    news_df[DATE_COLUMN].notna()
].copy()

news_df = news_df.sort_values(
    DATE_COLUMN
).reset_index(drop=True)

tokenized_docs = news_df["tokens"].tolist()

print("Jumlah tanggal:", len(news_df))
print(
    "Tanggal dengan headline:",
    sum(bool(tokens) for tokens in tokenized_docs)
)
print(
    "Total token:",
    sum(len(tokens) for tokens in tokenized_docs)
)


# ==================================================
# 5. TEMPORAL TRAINING SPLIT
# ==================================================

# periode TRAIN resmi (split 70/15/15); news_df sudah urut tanggal
split_idx = int((pd.to_datetime(news_df[DATE_COLUMN]) <= TRAIN_END).sum())

train_tagged_docs = [
    TaggedDocument(
        words=tokens,
        tags=[str(i)]
    )
    for i, tokens in enumerate(tokenized_docs[:split_idx])
    if tokens
]

if not train_tagged_docs:
    raise ValueError(
        "Tidak ada dokumen headline untuk training."
    )

print("\nTraining documents:", len(train_tagged_docs))
print(
    "Training period:",
    news_df.loc[0, DATE_COLUMN],
    "to",
    news_df.loc[split_idx - 1, DATE_COLUMN]
)


# ==================================================
# 6. TRAIN DOC2VEC
# ==================================================

d2v_model = Doc2Vec(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    dm=1,
    negative=5,
    epochs=EPOCHS,
    seed=SEED
)

d2v_model.build_vocab(
    train_tagged_docs
)

print("\nVocabulary size:", len(d2v_model.wv))

d2v_model.train(
    train_tagged_docs,
    total_examples=d2v_model.corpus_count,
    epochs=EPOCHS
)

print("\nDoc2Vec training completed!")

Path(MODEL_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

d2v_model.save(MODEL_PATH)


# ==================================================
# 7. DOCUMENT VECTOR INFERENCE
# ==================================================

def get_document_embedding(tokens, model):

    if not tokens:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    known_tokens = [
        token
        for token in tokens
        if token in model.wv.key_to_index
    ]

    known_count = len(known_tokens)

    if known_count == 0:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    coverage = known_count / len(tokens)

    document_vector = model.infer_vector(
        tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=INFER_EPOCHS
    )

    return document_vector, known_count, coverage


# ==================================================
# 8. GENERATE DAILY EMBEDDINGS
# ==================================================

embeddings = []
token_counts = []
vocab_coverages = []

for i, tokens in enumerate(tokenized_docs):

    vector, count, coverage = get_document_embedding(
        tokens,
        d2v_model
    )

    embeddings.append(vector)
    token_counts.append(count)
    vocab_coverages.append(coverage)

    if (i + 1) % 100 == 0 or (i + 1) == len(tokenized_docs):
        print(
            f"Embedding progress: "
            f"{i + 1}/{len(tokenized_docs)}"
        )


# ==================================================
# 9. CREATE FEATURE DATAFRAME
# ==================================================

embedding_columns = [
    f"d2v_{i:03d}"
    for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["d2v_token_count"] = token_counts
embedding_df["d2v_vocab_coverage"] = vocab_coverages

feature_result = pd.concat(
    [
        news_df[[DATE_COLUMN]].reset_index(drop=True),
        embedding_df
    ],
    axis=1
)

feature_result = feature_result.rename(
    columns={DATE_COLUMN: "date"}
)


# ==================================================
# 10. SAVE RESULTS
# ==================================================

Path(OUTPUT_FEATURES).parent.mkdir(
    parents=True,
    exist_ok=True
)

feature_result.to_csv(
    OUTPUT_FEATURES,
    index=False
)

print("\nDOC2VEC HEADLINE COMPLETED")
print("Output shape:", feature_result.shape)
print("Output:", OUTPUT_FEATURES)
print("Model:", MODEL_PATH)
print(feature_result.head())
