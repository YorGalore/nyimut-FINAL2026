
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText

# split resmi 70/15/15 (sama dengan Tugas 2) -- lihat dataset_split.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_split import split_boundaries  # noqa: E402


# ==================================================
# 1. CONFIGURATION
# ==================================================

HEADLINE_CSV = Path(
    "data/processed/head_vs_full/headline/cnbc_headline_daily.csv"
)

OUTPUT_FEATURES = Path(
    "data/processed/head_vs_full/headline/fasttext_headline_features.csv"
)

MODEL_PATH = Path(
    "models/fasttext_headline.model"
)

DATE_COLUMN = "date"
TOKEN_COLUMN = "headline_tokens"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10

# FastText hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()
SEED = 42
WORKERS = 4

MIN_N = 3
MAX_N = 6


# ==================================================
# 2. LOAD PREPROCESSED HEADLINES
# ==================================================

if not HEADLINE_CSV.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {HEADLINE_CSV}\n"
        "Jalankan src/headline/preprocess_headline.py terlebih dahulu."
    )

news_df = pd.read_csv(HEADLINE_CSV)

required_columns = {
    DATE_COLUMN,
    TOKEN_COLUMN
}

if not required_columns.issubset(news_df.columns):
    raise ValueError(
        f"Kolom yang dibutuhkan tidak ditemukan.\n"
        f"Kolom tersedia: {news_df.columns.tolist()}"
    )

news_df[DATE_COLUMN] = pd.to_datetime(
    news_df[DATE_COLUMN],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

news_df = news_df.dropna(
    subset=[DATE_COLUMN]
).copy()

news_df = news_df.sort_values(
    DATE_COLUMN
).reset_index(drop=True)

if news_df[DATE_COLUMN].duplicated().any():
    raise ValueError(
        "Terdapat tanggal duplikat dalam cnbc_headline_daily.csv."
    )

print("Dataset berhasil dimuat.")
print("Jumlah tanggal:", len(news_df))


# ==================================================
# 3. PARSE HEADLINE TOKENS
# ==================================================

def parse_headline_tokens(value):
    """
    Mengubah JSON headline_tokens menjadi list of lists.

    Format input:
    [
        ["headline", "pertama"],
        ["headline", "kedua"]
    ]

    Format output:
    List berisi token dari setiap headline.
    """

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)

    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    # Antisipasi apabila data hanya berupa satu list token
    if parsed and all(
        isinstance(item, str) for item in parsed
    ):
        parsed = [parsed]

    headlines = []

    for headline in parsed:

        if not isinstance(headline, list):
            continue

        tokens = [
            str(token).strip().lower()
            for token in headline
            if str(token).strip()
        ]

        if tokens:
            headlines.append(tokens)

    return headlines


news_df["parsed_headlines"] = news_df[
    TOKEN_COLUMN
].apply(parse_headline_tokens)

headline_docs = news_df["parsed_headlines"].tolist()

total_headlines = sum(
    len(headlines)
    for headlines in headline_docs
)

total_tokens = sum(
    len(headline)
    for headlines in headline_docs
    for headline in headlines
)

print("Total headline:", total_headlines)
print("Total token:", total_tokens)

print(
    "Tanggal dengan headline:",
    sum(bool(headlines) for headlines in headline_docs)
)


# ==================================================
# 4. TEMPORAL TRAINING SPLIT
# ==================================================

# periode TRAIN resmi (split 70/15/15); news_df sudah urut tanggal
split_idx = max(
    1,
    int((pd.to_datetime(news_df[DATE_COLUMN]) <= TRAIN_END).sum())
)

split_idx = min(
    split_idx,
    len(news_df)
)

# Training menggunakan headline individual,
# bukan gabungan seluruh headline dalam satu tanggal.
train_sentences = [
    headline
    for daily_headlines in headline_docs[:split_idx]
    for headline in daily_headlines
]

if not train_sentences:
    raise ValueError(
        "Tidak ada headline untuk training. "
        "Periksa isi kolom headline_tokens."
    )

print("\nTraining headlines:", len(train_sentences))

print(
    "Training period:",
    news_df.loc[0, DATE_COLUMN],
    "to",
    news_df.loc[split_idx - 1, DATE_COLUMN]
)


# ==================================================
# 5. TRAIN FASTTEXT
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

print("\nFastText training completed!")


# ==================================================
# 6. SAVE FASTTEXT MODEL
# ==================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

ft_model.save(str(MODEL_PATH))

print("Model saved:", MODEL_PATH)


# ==================================================
# 7. DOCUMENT EMBEDDING - MEAN POOLING
# ==================================================

def get_document_embedding(headlines, model):
    """
    Menghasilkan satu vektor representasi untuk satu tanggal.

    Seluruh token headline dalam tanggal tersebut
    direpresentasikan menggunakan FastText, kemudian
    dilakukan mean pooling.

    FastText dapat menghasilkan vektor untuk token OOV
    melalui character n-grams.
    """

    tokens = [
        token
        for headline in headlines
        for token in headline
    ]

    if not tokens:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    vectors = []
    known_count = 0

    for token in tokens:

        if token in model.wv.key_to_index:
            known_count += 1

        try:
            vector = model.wv[token]
            vectors.append(vector)

        except KeyError:
            continue

    if not vectors:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    document_vector = np.mean(
        vectors,
        axis=0
    )

    represented_count = len(vectors)

    vocabulary_coverage = (
        known_count / len(tokens)
    )

    return (
        document_vector,
        represented_count,
        vocabulary_coverage
    )


# ==================================================
# 8. GENERATE DAILY EMBEDDINGS
# ==================================================

embeddings = []
token_counts = []
vocab_coverages = []
headline_counts = []

for i, headlines in enumerate(headline_docs):

    vector, count, coverage = get_document_embedding(
        headlines,
        ft_model
    )

    embeddings.append(vector)
    token_counts.append(count)
    vocab_coverages.append(coverage)
    headline_counts.append(len(headlines))

    if (i + 1) % 100 == 0 or (i + 1) == len(headline_docs):
        print(
            f"Embedding progress: "
            f"{i + 1}/{len(headline_docs)}"
        )


# ==================================================
# 9. CREATE FEATURE DATAFRAME
# ==================================================

embedding_columns = [
    f"ft_{i:03d}"
    for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["ft_headline_count"] = headline_counts
embedding_df["ft_token_count"] = token_counts
embedding_df["ft_vocab_coverage"] = vocab_coverages

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

OUTPUT_FEATURES.parent.mkdir(
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

print("\nFeature preview:")
print(feature_result.head())

print("\nEmbedding dimension:", VECTOR_SIZE)
