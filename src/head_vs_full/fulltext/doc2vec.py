
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
# 1. KONFIGURASI
# ==================================================

INPUT_CSV = "data/processed/aligned_daily.csv"

OUTPUT_FEATURES = "data/processed/head_vs_full/fulltext/doc2vec_fulltext_features.csv"

MODEL_PATH = "models/doc2vec_fulltext.model"

TEXT_COLUMN = "text_norm_concat"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 20
INFER_EPOCHS = 30

SEED = 42
# Doc2Vec hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()
WORKERS = 4


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
print(
    "Periode:",
    df["date"].min(),
    "sampai",
    df["date"].max()
)


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
if len(tokenized_docs) > 1:
    print(tokenized_docs[1][:20])


# ==================================================
# 5. TEMPORAL TRAIN-TEST SPLIT
# ==================================================

# Menggunakan tanggal periode TRAIN resmi (split 70/15/15)
# untuk training Doc2Vec. df sudah urut tanggal.

split_idx = int((pd.to_datetime(df["date"]) <= TRAIN_END).sum())

if split_idx == 0 or split_idx >= len(df):
    raise ValueError("Pembagian data training tidak valid.")

train_tagged_docs = [
    TaggedDocument(
        words=tokens,
        tags=[str(i)]
    )
    for i, tokens in enumerate(tokenized_docs[:split_idx])
    if len(tokens) > 0
]

if not train_tagged_docs:
    raise ValueError(
        "Tidak ada dokumen untuk training Doc2Vec."
    )

print("\nJumlah data training:", split_idx)
print("Dokumen training:", len(train_tagged_docs))
print(
    "Tanggal batas training:",
    df.loc[split_idx - 1, "date"]
)


# ==================================================
# 6. TRAINING DOC2VEC
# ==================================================

doc2vec_model = Doc2Vec(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    dm=1,                  # 1 = PV-DM, 0 = PV-DBOW
    epochs=EPOCHS,
    seed=SEED,
    negative=5
)

# Membangun vocabulary dari data training
doc2vec_model.build_vocab(
    train_tagged_docs
)

print("\nVocabulary size:", len(doc2vec_model.wv))

# Training model
doc2vec_model.train(
    train_tagged_docs,
    total_examples=doc2vec_model.corpus_count,
    epochs=EPOCHS
)

print("\nDoc2Vec berhasil dilatih!")
print(
    "Vector dimension:",
    doc2vec_model.vector_size
)

# Menyimpan model
Path(MODEL_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

doc2vec_model.save(MODEL_PATH)


# ==================================================
# 7. DOCUMENT VECTOR INFERENCE
# ==================================================

def get_document_embedding(tokens, model):
    """
    Menghasilkan representasi dokumen menggunakan
    infer_vector dari model Doc2Vec.

    Token yang tidak terdapat dalam vocabulary
    training akan diabaikan.
    """

    if len(tokens) == 0:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    # Mengambil token yang dikenal model
    known_tokens = [
        token
        for token in tokens
        if token in model.wv.key_to_index
    ]

    known_token_count = len(known_tokens)

    # Jika tidak ada token yang dikenal
    if known_token_count == 0:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    # Menghitung vocabulary coverage
    coverage = known_token_count / len(tokens)

    # Infer document vector
    document_vector = model.infer_vector(
        known_tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=INFER_EPOCHS
    )

    return (
        document_vector,
        known_token_count,
        coverage
    )


# ==================================================
# 8. MENGHASILKAN EMBEDDING SETIAP TANGGAL
# ==================================================

embeddings = []
known_token_counts = []
vocab_coverages = []

for i, tokens in enumerate(tokenized_docs):

    vector, token_count, coverage = (
        get_document_embedding(
            tokens,
            doc2vec_model
        )
    )

    embeddings.append(vector)
    known_token_counts.append(token_count)
    vocab_coverages.append(coverage)

    # Menampilkan progress setiap 100 dokumen
    if (i + 1) % 100 == 0 or (i + 1) == len(tokenized_docs):
        print(
            f"Embedding progress: "
            f"{i + 1}/{len(tokenized_docs)}"
        )


# ==================================================
# 9. MEMBUAT DATAFRAME EMBEDDING
# ==================================================

embedding_columns = [
    f"d2v_{i:03d}"
    for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["d2v_token_count"] = known_token_counts

embedding_df["d2v_vocab_coverage"] = vocab_coverages


# ==================================================
# 10. MENYIMPAN HASIL
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
# 11. HASIL
# ==================================================

print("\n===================================")
print("DOC2VEC FULL-TEXT SELESAI")
print("===================================")

print("Jumlah tanggal:", len(feature_result))
print("Dimensi embedding:", VECTOR_SIZE)
print("Ukuran hasil:", feature_result.shape)

print("\nContoh hasil:")
print(feature_result.head())

print("\nFile berhasil disimpan:")
print(OUTPUT_FEATURES)
print(MODEL_PATH)
