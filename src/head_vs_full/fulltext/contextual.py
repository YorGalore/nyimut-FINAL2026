
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT_DIR
    / "data"
    / "processed"
    / "aligned_daily.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

DATE_COLUMN = "date"
TEXT_COLUMN = "text_norm_concat"

MODELS = {
    "finbert": "ProsusAI/finbert",
    "roberta": "FacebookAI/roberta-base",
    "qwen": "Qwen/Qwen2.5-0.5B",
}

MAX_LENGTH = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 2
LOG_EVERY = 20


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")


# ============================================================
# LOAD DAILY DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING ALIGNED DAILY DATA")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False
    )

    required_columns = [
        DATE_COLUMN,
        TEXT_COLUMN
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    if df[DATE_COLUMN].isna().any():
        raise ValueError(
            "Terdapat tanggal tidak valid."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise ValueError(
            "Terdapat tanggal duplikat."
        )

    df = (
        df.sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    texts = df[TEXT_COLUMN].tolist()

    empty_count = sum(
        1 for text in texts
        if not text
    )

    print(f"Input file       : {INPUT_CSV}")
    print(f"Jumlah baris     : {len(df):,}")
    print(f"Jumlah dokumen   : {len(texts):,}")
    print(f"Dokumen kosong   : {empty_count:,}")

    print(
        f"Rentang tanggal  : "
        f"{df[DATE_COLUMN].min().date()} "
        f"hingga "
        f"{df[DATE_COLUMN].max().date()}"
    )

    return df, texts


# ============================================================
# FIND SUBSEQUENCE
# ============================================================

def find_subsequence(sequence, subsequence):

    """
    Mencari posisi subsequence di dalam sequence.

    Mengembalikan posisi awal jika ditemukan,
    atau -1 jika tidak ditemukan.
    """

    if not subsequence:
        return -1

    seq_length = len(sequence)
    sub_length = len(subsequence)

    for start in range(seq_length - sub_length + 1):

        end = start + sub_length

        if sequence[start:end] == subsequence:
            return start

    return -1


# ============================================================
# AUTOMATIC SPECIAL TOKEN DETECTION
# ============================================================

def infer_special_token_wrapper(tokenizer):

    """
    Mendeteksi prefix dan suffix special tokens
    tanpa menggunakan:

    - prepare_for_model()
    - build_inputs_with_special_tokens()
    - get_special_tokens_mask()

    Deteksi menggunakan perbandingan hasil encode
    dengan add_special_tokens=False dan True.
    """

    probe_text = (
        "contextual embedding special token "
        "detection probe 7391 alpha beta"
    )

    # Tokenisasi tanpa special tokens
    content_ids = tokenizer.encode(
        probe_text,
        add_special_tokens=False,
        truncation=False
    )

    # Tokenisasi dengan special tokens
    wrapped_ids = tokenizer.encode(
        probe_text,
        add_special_tokens=True,
        truncation=False
    )

    # Cari posisi content tokens di dalam wrapped sequence
    content_start = find_subsequence(
        wrapped_ids,
        content_ids
    )

    if content_start == -1:

        raise ValueError(
            f"Gagal mendeteksi special token wrapper "
            f"untuk tokenizer {tokenizer.__class__.__name__}.\n"
            "Content token sequence tidak ditemukan "
            "di dalam sequence dengan special tokens."
        )

    content_end = (
        content_start + len(content_ids)
    )

    prefix_ids = wrapped_ids[:content_start]

    suffix_ids = wrapped_ids[content_end:]

    # Validasi panjang wrapper
    expected_special_count = (
        len(prefix_ids) + len(suffix_ids)
    )

    if expected_special_count == 0:

        print(
            "Warning: tokenizer tidak menambahkan "
            "special tokens."
        )

    return prefix_ids, suffix_ids


# ============================================================
# TOKENIZATION AND CHUNKING
# ============================================================

def create_chunks(
    texts,
    tokenizer
):

    print("\nTOKENIZATION AND CHUNKING")
    print("-" * 70)

    # Dokumentasi model mungkin membatasi max length
    # default tokenizer menjadi 512.
    # Kita melakukan chunking secara manual.
    tokenizer.model_max_length = 10**9

    tokenizer.padding_side = "right"

    # Deteksi special tokens secara otomatis
    prefix_ids, suffix_ids = (
        infer_special_token_wrapper(tokenizer)
    )

    special_tokens_count = (
        len(prefix_ids) + len(suffix_ids)
    )

    content_length = (
        MAX_LENGTH - special_tokens_count
    )

    if content_length <= CHUNK_OVERLAP:

        raise ValueError(
            "CHUNK_OVERLAP harus lebih kecil "
            "daripada content_length."
        )

    step_size = (
        content_length - CHUNK_OVERLAP
    )

    print(
        f"Tokenizer        : {tokenizer.__class__.__name__}"
    )

    print(
        f"Prefix tokens    : {prefix_ids}"
    )

    print(
        f"Suffix tokens    : {suffix_ids}"
    )

    print(
        f"Content length   : {content_length}"
    )

    print(
        f"Step size        : {step_size}"
    )

    all_chunks = []

    # Menyimpan hubungan dokumen dengan chunk
    doc_chunk_map = [
        [] for _ in range(len(texts))
    ]

    total_tokens = 0

    for doc_idx, text in enumerate(texts):

        # Pertahankan tanggal tanpa berita.
        # Dokumen kosong tidak menghasilkan chunk.
        if not text:
            continue

        # Tokenisasi seluruh dokumen tanpa special tokens
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        if not token_ids:
            continue

        total_tokens += len(token_ids)

        previous_end = 0
        start = 0

        while start < len(token_ids):

            end = min(
                start + content_length,
                len(token_ids)
            )

            chunk_token_ids = token_ids[start:end]

            if not chunk_token_ids:
                break

            # Hitung token baru yang belum tercakup
            # oleh chunk sebelumnya.
            unique_weight = (
                end - max(start, previous_end)
            )

            # Tambahkan special tokens secara manual
            input_ids = (
                prefix_ids
                + chunk_token_ids
                + suffix_ids
            )

            # Mask:
            # 0 = content token
            # 1 = special token
            special_mask = (
                [1] * len(prefix_ids)
                + [0] * len(chunk_token_ids)
                + [1] * len(suffix_ids)
            )

            if len(input_ids) > MAX_LENGTH:

                raise ValueError(
                    f"Chunk melebihi MAX_LENGTH: "
                    f"{len(input_ids)} tokens."
                )

            chunk_idx = len(all_chunks)

            all_chunks.append({
                "input_ids": input_ids,
                "special_tokens_mask": special_mask
            })

            doc_chunk_map[doc_idx].append(
                (
                    chunk_idx,
                    max(unique_weight, 1)
                )
            )

            previous_end = end

            if end >= len(token_ids):
                break

            start += step_size

    print(f"Total token      : {total_tokens:,}")
    print(f"Total chunks     : {len(all_chunks):,}")
    print(f"Max sequence     : {MAX_LENGTH}")
    print(f"Chunk overlap    : {CHUNK_OVERLAP}")

    return all_chunks, doc_chunk_map


# ============================================================
# MANUAL BATCH PADDING
# ============================================================

def prepare_batch(
    batch_chunks,
    tokenizer
):

    """
    Melakukan dynamic padding secara manual.
    Tidak menggunakan tokenizer.pad().
    """

    batch_size = len(batch_chunks)

    max_seq_length = max(
        len(chunk["input_ids"])
        for chunk in batch_chunks
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.unk_token_id

    if pad_token_id is None:
        pad_token_id = 0

    input_ids = torch.full(
        (batch_size, max_seq_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (batch_size, max_seq_length),
        dtype=torch.long
    )

    special_tokens_mask = torch.ones(
        (batch_size, max_seq_length),
        dtype=torch.long
    )

    for i, chunk in enumerate(batch_chunks):

        ids = torch.tensor(
            chunk["input_ids"],
            dtype=torch.long
        )

        special_mask = torch.tensor(
            chunk["special_tokens_mask"],
            dtype=torch.long
        )

        seq_length = len(ids)

        # Right padding
        input_ids[i, :seq_length] = ids

        attention_mask[i, :seq_length] = 1

        special_tokens_mask[i, :seq_length] = (
            special_mask
        )

    return (
        input_ids,
        attention_mask,
        special_tokens_mask
    )


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    last_hidden_state,
    attention_mask,
    special_tokens_mask
):

    """
    Menghitung mean pooling hanya pada content tokens.

    Padding dan special tokens tidak ikut dihitung.
    """

    content_mask = (
        attention_mask
        * (1 - special_tokens_mask)
    )

    content_mask = content_mask.unsqueeze(-1).to(
        last_hidden_state.dtype
    )

    summed_embeddings = (
        last_hidden_state * content_mask
    ).sum(dim=1)

    token_counts = (
        content_mask.sum(dim=1).clamp(min=1.0)
    )

    embeddings = (
        summed_embeddings / token_counts
    )

    return embeddings


# ============================================================
# EXTRACT CHUNK EMBEDDINGS
# ============================================================

def extract_chunk_embeddings(
    model,
    tokenizer,
    chunks,
    hidden_size
):

    total_chunks = len(chunks)

    chunk_embeddings = np.zeros(
        (total_chunks, hidden_size),
        dtype=np.float32
    )

    if total_chunks == 0:
        return chunk_embeddings

    total_batches = (
        total_chunks + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\nEXTRACTING CONTEXTUAL EMBEDDINGS")
    print("-" * 70)

    model.eval()

    for batch_idx, start_idx in enumerate(
        range(0, total_chunks, BATCH_SIZE),
        start=1
    ):

        end_idx = min(
            start_idx + BATCH_SIZE,
            total_chunks
        )

        batch_chunks = chunks[start_idx:end_idx]

        (
            input_ids,
            attention_mask,
            special_tokens_mask
        ) = prepare_batch(
            batch_chunks,
            tokenizer
        )

        input_ids = input_ids.to(DEVICE)

        attention_mask = attention_mask.to(DEVICE)

        special_tokens_mask = (
            special_tokens_mask.to(DEVICE)
        )

        with torch.inference_mode():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            last_hidden_state = (
                outputs.last_hidden_state
            )

            embeddings = mean_pooling(
                last_hidden_state,
                attention_mask,
                special_tokens_mask
            )

            embeddings = (
                embeddings
                .float()
                .cpu()
                .numpy()
            )

        chunk_embeddings[start_idx:end_idx] = (
            embeddings
        )

        if (
            batch_idx % LOG_EVERY == 0
            or batch_idx == total_batches
        ):

            print(
                f"Batch {batch_idx:,}/{total_batches:,} "
                f"| Chunks {end_idx:,}/{total_chunks:,}"
            )

        del (
            input_ids,
            attention_mask,
            special_tokens_mask,
            outputs,
            last_hidden_state,
            embeddings
        )

    return chunk_embeddings


# ============================================================
# AGGREGATE CHUNK EMBEDDINGS INTO DAILY EMBEDDINGS
# ============================================================

def aggregate_daily_embeddings(
    chunk_embeddings,
    doc_chunk_map,
    hidden_size
):

    n_documents = len(doc_chunk_map)

    daily_embeddings = np.zeros(
        (n_documents, hidden_size),
        dtype=np.float32
    )

    for doc_idx, chunk_info in enumerate(
        doc_chunk_map
    ):

        # Jika tidak ada berita, embedding tetap nol
        if not chunk_info:
            continue

        chunk_indices = [
            item[0] for item in chunk_info
        ]

        chunk_weights = np.array(
            [item[1] for item in chunk_info],
            dtype=np.float32
        )

        embeddings = (
            chunk_embeddings[chunk_indices]
        )

        # Weighted average berdasarkan token unik
        daily_embedding = np.average(
            embeddings,
            axis=0,
            weights=chunk_weights
        )

        daily_embeddings[doc_idx] = (
            daily_embedding
        )

    return daily_embeddings


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def process_model(
    model_name,
    model_id,
    df,
    texts
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING MODEL: {model_name.upper()}")
    print("=" * 70)

    print(f"Model checkpoint : {model_id}")
    print(f"Device           : {DEVICE}")

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModel.from_pretrained(
        model_id
    )

    model = model.to(DEVICE)

    model.eval()

    hidden_size = getattr(
        model.config,
        "hidden_size",
        None
    )

    if hidden_size is None:

        raise ValueError(
            f"hidden_size tidak ditemukan pada {model_id}"
        )

    print(f"Hidden size      : {hidden_size}")

    # --------------------------------------------------------
    # TOKENIZATION AND CHUNKING
    # --------------------------------------------------------

    chunks, doc_chunk_map = create_chunks(
        texts=texts,
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # EXTRACT CHUNK EMBEDDINGS
    # --------------------------------------------------------

    chunk_embeddings = extract_chunk_embeddings(
        model=model,
        tokenizer=tokenizer,
        chunks=chunks,
        hidden_size=hidden_size
    )

    # --------------------------------------------------------
    # AGGREGATE DAILY EMBEDDINGS
    # --------------------------------------------------------

    print("\nAGGREGATING DAILY EMBEDDINGS")
    print("-" * 70)

    daily_embeddings = aggregate_daily_embeddings(
        chunk_embeddings=chunk_embeddings,
        doc_chunk_map=doc_chunk_map,
        hidden_size=hidden_size
    )

    # --------------------------------------------------------
    # BUILD OUTPUT DATAFRAME
    # --------------------------------------------------------

    feature_names = [
        f"{model_name}_{i:03d}"
        for i in range(hidden_size)
    ]

    features_df = pd.DataFrame(
        daily_embeddings,
        columns=feature_names
    )

    features_df.insert(
        0,
        DATE_COLUMN,
        df[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(features_df) != len(df):

        raise ValueError(
            "Jumlah baris feature tidak sama "
            "dengan aligned_daily.csv."
        )

    if features_df[DATE_COLUMN].duplicated().any():

        raise ValueError(
            "Terdapat tanggal duplikat."
        )

    if features_df[feature_names].isna().any().any():

        raise ValueError(
            "Terdapat NaN pada embedding."
        )

    if not np.isfinite(
        features_df[feature_names].to_numpy()
    ).all():

        raise ValueError(
            "Terdapat inf atau nilai non-finite "
            "pada embedding."
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{model_name}_fulltext_features.csv"
    )

    features_df.to_csv(
        output_path,
        index=False,
        float_format="%.8f"
    )

    empty_documents = sum(
        1 for chunk_list in doc_chunk_map
        if not chunk_list
    )

    print("\nMODEL SUMMARY")
    print("-" * 70)

    print(f"Model             : {model_name}")
    print(f"Embedding dim     : {hidden_size}")
    print(f"Total documents   : {len(features_df):,}")
    print(f"Empty documents   : {empty_documents:,}")
    print(f"Total chunks      : {len(chunks):,}")
    print(f"Output shape      : {features_df.shape}")
    print(f"Saved             : {output_path}")

    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del model
    del tokenizer
    del chunks
    del doc_chunk_map
    del chunk_embeddings
    del daily_embeddings
    del features_df

    gc.collect()

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    elif DEVICE.type == "mps":

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL TEXT CONTEXTUAL EMBEDDING EXTRACTION")
    print("=" * 70)

    print(f"Root directory : {ROOT_DIR}")
    print(f"Input file     : {INPUT_CSV}")
    print(f"Device         : {DEVICE}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load daily dataset once
    df, texts = load_data()

    # Process models sequentially
    for model_name, model_id in MODELS.items():

        process_model(
            model_name=model_name,
            model_id=model_id,
            df=df,
            texts=texts
        )

    print("\n")
    print("=" * 70)
    print("ALL MODELS COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    for model_name in MODELS:

        output_path = (
            OUTPUT_DIR
            / f"{model_name}_fulltext_features.csv"
        )

        print(f"  - {output_path}")

    print("\nSemua contextual embeddings berhasil disimpan.")


if __name__ == "__main__":
    main()
