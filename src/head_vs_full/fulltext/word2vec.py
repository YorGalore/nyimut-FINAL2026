
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from gensim.models import Word2Vec

# split resmi 70/15/15 (sama dengan Tugas 2) -- lihat dataset_split.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_split import split_boundaries  # noqa: E402

# ==================================================
# 1. KONFIGURASI
# ==================================================

INPUT_CSV = "data/processed/aligned_daily.csv"

OUTPUT_FEATURES = "data/processed/head_vs_full/fulltext/word2vec_fulltext_features.csv"

MODEL_PATH = "models/word2vec_model.model"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 30
SEED = 42
# Word2Vec hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()


# ==================================================
# 2. MEMBACA DATASET
# ==================================================

df = pd.read_csv(INPUT_CSV)

# Mengurutkan data berdasarkan tanggal
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Memastikan kolom teks tersedia
TEXT_COLUMN = "text_norm_concat"

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Kolom {TEXT_COLUMN} tidak ditemukan!")

# Mengubah nilai kosong menjadi string kosong
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

print("Jumlah data:", len(df))
print("Periode:", df["date"].min(), "sampai", df["date"].max())


# ==================================================
# 3. TOKENISASI TEKS
# ==================================================

def tokenize_text(text):
    """
    Mengubah teks menjadi kumpulan token.
    Menggunakan whitespace tokenization karena
    text_norm_concat sudah melalui preprocessing.
    """

    if not text.strip():
        return []

    tokens = text.lower().split()

    return tokens


# Tokenisasi setiap dokumen harian
tokenized_docs = df[TEXT_COLUMN].apply(tokenize_text).tolist()

print("\nContoh token:")
print(tokenized_docs[1][:20])


# ==================================================
# 4. TRAIN-TEST TEMPORAL SPLIT
# ==================================================

# Word2Vec hanya dilatih menggunakan data training
# agar informasi dari periode masa depan tidak masuk
# ke proses pembentukan vocabulary dan embedding.

# tanggal periode TRAIN resmi (split 70/15/15); df sudah urut tanggal
split_idx = int((df["date"] <= TRAIN_END).sum())

train_sentences = tokenized_docs[:split_idx]

# Menghapus dokumen kosong dari corpus training
train_sentences = [
    tokens for tokens in train_sentences if len(tokens) > 0
]

if not train_sentences:
    raise ValueError("Tidak ada dokumen teks untuk training Word2Vec.")

print("\nJumlah data training:", split_idx)
print("Jumlah dokumen untuk training Word2Vec:", len(train_sentences))
print("Tanggal batas training:", df.loc[split_idx - 1, "date"])


# ==================================================
# 5. TRAINING WORD2VEC
# ==================================================

w2v_model = Word2Vec(
    sentences=train_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=1,
    sg=1,              # 1 = Skip-Gram, 0 = CBOW
    negative=5,
    epochs=EPOCHS,
    seed=SEED
)

# Menampilkan informasi model
print("\nWord2Vec berhasil dilatih!")
print("Vocabulary size:", len(w2v_model.wv))
print("Vector dimension:", w2v_model.wv.vector_size)

# Menyimpan model
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
w2v_model.save(MODEL_PATH)


# ==================================================
# 6. DAILY DOCUMENT EMBEDDING (MEAN POOLING)
# ==================================================

def get_document_embedding(tokens, model):
    """
    Menghasilkan representasi satu dokumen harian
    dengan menghitung rata-rata embedding setiap kata.

    Kata yang tidak terdapat dalam vocabulary
    tidak digunakan dalam perhitungan.
    """

    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    # Mean pooling
    document_vector = np.mean(vectors, axis=0)

    # Jumlah token yang berhasil direpresentasikan
    known_token_count = len(vectors)

    # Persentase token yang terdapat dalam vocabulary
    coverage = known_token_count / len(tokens)

    return document_vector, known_token_count, coverage


# Menghasilkan embedding untuk setiap tanggal
embeddings = []
known_token_counts = []
vocab_coverages = []

for tokens in tokenized_docs:

    vector, token_count, coverage = get_document_embedding(
        tokens,
        w2v_model
    )

    embeddings.append(vector)
    known_token_counts.append(token_count)
    vocab_coverages.append(coverage)


# Mengubah embedding menjadi DataFrame
embedding_columns = [
    f"w2v_{i:03d}" for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["w2v_token_count"] = known_token_counts
embedding_df["w2v_vocab_coverage"] = vocab_coverages


# ==================================================
# 7. MENGGABUNGKAN EMBEDDING DENGAN DATASET ASLI
# ==================================================

result = pd.concat(
    [
        df.reset_index(drop=True),
        embedding_df.reset_index(drop=True)
    ],
    axis=1
)

# Mengubah tanggal kembali menjadi format YYYY-MM-DD
result["date"] = result["date"].dt.strftime("%Y-%m-%d")

# (Dulu dataset lengkap juga disimpan ke OUTPUT_CSV, tapi variabel itu tidak
# pernah didefinisikan -> NameError. Tidak dipakai tahap mana pun, jadi dihapus.)

# Menyimpan dataset fitur embedding saja
Path(OUTPUT_FEATURES).parent.mkdir(parents=True, exist_ok=True)
feature_result = pd.concat(
    [
        result[["date"]],
        result[embedding_columns],
        result[[
            "w2v_token_count",
            "w2v_vocab_coverage"
        ]]
    ],
    axis=1
)

feature_result.to_csv(OUTPUT_FEATURES, index=False)


# ==================================================
# 8. HASIL
# ==================================================

print("\n===================================")
print("WORD2VEC FEATURE EXTRACTION SELESAI")
print("===================================")

print("\nUkuran dataset awal:", df.shape)
print("Ukuran dataset hasil:", result.shape)

print("\nContoh embedding:")
print(feature_result.head())

print("\nFile berhasil disimpan:")
print(OUTPUT_FEATURES)
print(MODEL_PATH)
