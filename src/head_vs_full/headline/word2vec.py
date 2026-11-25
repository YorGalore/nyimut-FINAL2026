
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

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
    "data/processed/head_vs_full/headline/word2vec_headline_features.csv"
)

MODEL_PATH = Path(
    "models/word2vec_headline.model"
)

DATE_COLUMN = "date"
TOKEN_COLUMN = "headline_tokens"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 30

# Word2Vec hanya dilatih pada periode TRAIN resmi (s.d. tanggal terakhir train)
TRAIN_END, _ = split_boundaries()
SEED = 42
WORKERS = 1


# ==================================================
# 2. LOAD PREPROCESSED HEADLINES
# ==================================================

if not HEADLINE_CSV.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {HEADLINE_CSV}\n"
        "Jalankan src/headline/preprocess_headline.py terlebih dahulu."
    )

df = pd.read_csv(HEADLINE_CSV)

required_columns = {
    DATE_COLUMN,
    TOKEN_COLUMN
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Kolom yang dibutuhkan tidak ditemukan.\n"
        f"Kolom tersedia: {df.columns.tolist()}"
    )

# Mengubah tanggal ke format YYYY-MM-DD
df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

# Menghapus tanggal yang tidak valid
df = df.dropna(
    subset=[DATE_COLUMN]
).copy()

# Mengurutkan data berdasarkan tanggal
df = df.sort_values(
    DATE_COLUMN
).reset_index(drop=True)

# Memastikan satu baris mewakili satu tanggal
if df[DATE_COLUMN].duplicated().any():
    raise ValueError(
        "Terdapat tanggal duplikat dalam "
        "cnbc_headline_daily.csv."
    )

if df.empty:
    raise ValueError("Dataset headline kosong.")

print("Jumlah tanggal:", len(df))
print(
    "Periode:",
    df[DATE_COLUMN].min(),
    "sampai",
    df[DATE_COLUMN].max()
)


# ==================================================
# 3. PARSE HEADLINE TOKENS
# ==================================================

def parse_headline_tokens(value):
    """
    Mengubah headline_tokens dari JSON menjadi
    list of lists.

    Contoh input:
    [
        ["headline", "pertama"],
        ["headline", "kedua"]
    ]

    Output:
    [
        ["headline", "pertama"],
        ["headline", "kedua"]
    ]
    """

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)

    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, list):
        return []

    # Mengantisipasi format satu headline berupa
    # list token, bukan kumpulan list headline.
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


df["parsed_headlines"] = df[
    TOKEN_COLUMN
].apply(parse_headline_tokens)

# Setiap tanggal berisi kumpulan headline.
headline_docs = df["parsed_headlines"].tolist()

total_headlines = sum(
    len(headlines)
    for headlines in headline_docs
)

total_tokens = sum(
    len(headline)
    for headlines in headline_docs
    for headline in headlines
)

print("\nTotal headline:", total_headlines)
print("Total token:", total_tokens)

print(
    "Tanggal dengan headline:",
    sum(bool(headlines) for headlines in headline_docs)
)

if total_headlines == 0:
    raise ValueError(
        "Tidak ada headline yang berhasil dibaca. "
        "Periksa isi kolom headline_tokens."
    )


# ==================================================
# 4. TEMPORAL TRAIN-TEST SPLIT
# ==================================================

# Model hanya dilatih menggunakan headline dari
# periode TRAIN resmi (split 70/15/15). df sudah urut tanggal.

split_idx = max(
    1,
    int((pd.to_datetime(df[DATE_COLUMN]) <= TRAIN_END).sum())
)

split_idx = min(
    split_idx,
    len(df)
)

# Menggunakan setiap headline sebagai satu kalimat
# untuk training Word2Vec.
train_sentences = [
    headline
    for daily_headlines in headline_docs[:split_idx]
    for headline in daily_headlines
]

if not train_sentences:
    raise ValueError(
        "Tidak ada headline untuk training Word2Vec."
    )

print("\nJumlah tanggal training:", split_idx)
print("Jumlah headline training:", len(train_sentences))

print(
    "Periode training:",
    df.loc[0, DATE_COLUMN],
    "sampai",
    df.loc[split_idx - 1, DATE_COLUMN]
)


# ==================================================
# 5. TRAINING WORD2VEC
# ==================================================

w2v_model = Word2Vec(
    sentences=train_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=1,                  # Skip-Gram
    negative=5,
    epochs=EPOCHS,
    seed=SEED
)

print("\nWord2Vec training completed!")
print("Vocabulary size:", len(w2v_model.wv))
print("Vector dimension:", w2v_model.wv.vector_size)

# Menyimpan model
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

w2v_model.save(str(MODEL_PATH))

print("Model saved:", MODEL_PATH)


# ==================================================
# 6. DAILY DOCUMENT EMBEDDING - MEAN POOLING
# ==================================================

def get_document_embedding(headlines, model):
    """
    Menghasilkan satu representasi vektor untuk
    seluruh headline dalam satu tanggal.

    Tahapan:
    1. Menggabungkan token dari seluruh headline.
    2. Mengambil embedding token yang terdapat
       dalam vocabulary Word2Vec.
    3. Menghitung rata-rata embedding (mean pooling).

    Kata OOV tidak digunakan dalam perhitungan.
    """

    tokens = [
        token
        for headline in headlines
        for token in headline
    ]

    if not tokens:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    known_vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv.key_to_index
    ]

    if not known_vectors:
        return np.zeros(VECTOR_SIZE), 0, 0.0

    # Mean pooling
    document_vector = np.mean(
        known_vectors,
        axis=0
    )

    # Jumlah token yang berhasil direpresentasikan
    known_token_count = len(known_vectors)

    # Vocabulary coverage
    coverage = (
        known_token_count / len(tokens)
    )

    return (
        document_vector,
        known_token_count,
        coverage
    )


# ==================================================
# 7. GENERATE DAILY EMBEDDINGS
# ==================================================

embeddings = []
known_token_counts = []
vocab_coverages = []
headline_counts = []

for i, headlines in enumerate(headline_docs):

    vector, token_count, coverage = get_document_embedding(
        headlines,
        w2v_model
    )

    embeddings.append(vector)
    known_token_counts.append(token_count)
    vocab_coverages.append(coverage)
    headline_counts.append(len(headlines))

    if (i + 1) % 100 == 0 or (i + 1) == len(headline_docs):
        print(
            f"Embedding progress: "
            f"{i + 1}/{len(headline_docs)}"
        )


# ==================================================
# 8. CREATE FEATURE DATAFRAME
# ==================================================

embedding_columns = [
    f"w2v_{i:03d}"
    for i in range(VECTOR_SIZE)
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)

embedding_df["w2v_headline_count"] = headline_counts
embedding_df["w2v_token_count"] = known_token_counts
embedding_df["w2v_vocab_coverage"] = vocab_coverages

feature_result = pd.concat(
    [
        df[[DATE_COLUMN]].reset_index(drop=True),
        embedding_df
    ],
    axis=1
)

feature_result = feature_result.rename(
    columns={
        DATE_COLUMN: "date"
    }
)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

OUTPUT_FEATURES.parent.mkdir(
    parents=True,
    exist_ok=True
)

feature_result.to_csv(
    OUTPUT_FEATURES,
    index=False
)


# ==================================================
# 10. RESULTS
# ==================================================

print("\n======================================")
print("WORD2VEC HEADLINE FEATURE EXTRACTION SELESAI")
print("======================================")

print("\nJumlah tanggal:", len(feature_result))
print("Jumlah headline:", total_headlines)
print("Dimensi embedding:", VECTOR_SIZE)
print("Output shape:", feature_result.shape)

print("\nContoh hasil:")
print(feature_result.head())

print("\nFile berhasil disimpan:")
print("Features:", OUTPUT_FEATURES)
print("Model:", MODEL_PATH)
