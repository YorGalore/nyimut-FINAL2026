
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText

# split resmi 70/15/15 (sama dengan Tugas 2) -- lihat dataset_split.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_split import split_boundaries  # noqa: E402


# ==================================================
# 1. KONFIGURASI
# ==================================================

INPUT_CSV = "data/processed/aligned_daily.csv"

OUTPUT_FEATURES = "data/processed/head_vs_full/fulltext/fasttext_fulltext_features.csv"

MODEL_PATH = "models/fasttext_fulltext.model"

TEXT_COLUMN = "text_norm_concat"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10
SEED = 42
# FastText hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()

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
# 3. MEMBACA DATASET
# ==================================================

df = pd.read_csv(INPUT_CSV)

df["date"] = pd.to_datetime(df["date"])

df = (
    df.sort_values("date")
    .reset_index(drop=True)
)

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Kolom {TEXT_COLUMN} tidak ditemukan!"
    )

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
)

print("Jumlah data:", len(df))
print("Periode:", df["date"].min(), "sampai", df["date"].max())


# ==================================================
# 4. TOKENISASI TEKS
# ==================================================

def tokenize_text(text):
    """
    Tokenisasi menggunakan whitespace.
    Teks diasumsikan sudah melalui preprocessing.
    """

    if not text.strip():
        return []

    return text.lower().split()


tokenized_docs = df[TEXT_COLUMN].apply(
    tokenize_text
).tolist()

print("\nContoh token:")
print(tokenized_docs[1][:20])


# ==================================================
# 5. TEMPORAL TRAIN-TEST SPLIT
# ==================================================

# tanggal periode TRAIN resmi (split 70/15/15); df sudah urut tanggal
split_idx = int((pd.to_datetime(df["date"]) <= TRAIN_END).sum())

train_sentences = [
    tokens
    for tokens in tokenized_docs[:split_idx]
    if len(tokens) > 0
]

if not train_sentences:
    raise ValueError(
        "Tidak ada dokumen untuk training FastText."
    )

print("\nJumlah data training:", split_idx)
print("Dokumen training:", len(train_sentences))
print("Tanggal batas training:", df.loc[split_idx - 1, "date"])


# ==================================================
# 6. TRAINING FASTTEXT
# ==================================================

fasttext_model = FastText(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=1,                   # 1 = Skip-Gram, 0 = CBOW
    negative=5,
    min_n=MIN_N,
    max_n=MAX_N,
    epochs=EPOCHS,
    seed=SEED
)

# Membangun vocabulary dari data training
fasttext_model.build_vocab(
    corpus_iterable=train_sentences
)

print("\nVocabulary size:", len(fasttext_model.wv))

# Training model
fasttext_model.train(
    corpus_iterable=train_sentences,
    total_examples=len(train_sentences),
    epochs=EPOCHS
)

print("\nFastText berhasil dilatih!")
print("Vector dimension:", fasttext_model.wv.vector_size)

# Menyimpan model
Path(MODEL_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

fasttext_model.save(MODEL_PATH)


# ==================================================
# 7. DAILY DOCUMENT EMBEDDING
#    MEAN POOLING
# ==================================================

def get_document_embedding(tokens, model):
    """
    Menghasilkan representasi dokumen harian
    dengan menghitung rata-rata embedding kata.

    FastText dapat menghasilkan embedding untuk
    kata OOV melalui informasi subword.
    """

    if len(tokens) == 0:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    # Mengambil embedding setiap token
    vectors = [
        model.wv[token]
        for token in tokens
    ]

    # Mean Pooling
    document_vector = np.mean(
        vectors,
        axis=0
    )

    token_count = len(vectors)

    # FastText dapat merepresentasikan OOV melalui subword.
    # Coverage di sini mengukur token yang masuk vocabulary
    # eksplisit, bukan seluruh token yang dapat direpresentasikan.
    known_token_count = sum(
        1 for token in tokens
        if token in model.wv.key_to_index
    )

    vocab_coverage = (
        known_token_count / token_count
        if token_count > 0
        else 0.0
    )

    return (
        document_vector,
        token_count,
        vocab_coverage
    )


# Menghasilkan embedding setiap tanggal
embeddings = []
token_counts = []
vocab_coverages = []

for tokens in tokenized_docs:

    vector, token_count, coverage = (
        get_document_embedding(
            tokens,
            fasttext_model
        )
    )

    embeddings.append(vector)
    token_counts.append(token_count)
    vocab_coverages.append(coverage)


# ==================================================
# 8. MEMBUAT DATAFRAME EMBEDDING
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


# ==================================================
# 9. MENYIMPAN HASIL EMBEDDING
# ==================================================

feature_result = pd.concat(
    [
        df[["date"]].reset_index(drop=True),
        embedding_df.reset_index(drop=True)
    ],
    axis=1
)

feature_result["date"] = (
    feature_result["date"]
    .dt.strftime("%Y-%m-%d")
)

Path(OUTPUT_FEATURES).parent.mkdir(
    parents=True,
    exist_ok=True
)

feature_result.to_csv(
    OUTPUT_FEATURES,
    index=False
)


# ==================================================
# 10. HASIL
# ==================================================

print("\n===================================")
print("FASTTEXT FULL-TEXT SELESAI")
print("===================================")

print("Jumlah tanggal:", len(feature_result))
print("Dimensi embedding:", VECTOR_SIZE)

print("\nUkuran hasil:", feature_result.shape)

print("\nContoh hasil:")
print(feature_result.head())

print("\nFile berhasil disimpan:")
print(OUTPUT_FEATURES)
print(MODEL_PATH)
