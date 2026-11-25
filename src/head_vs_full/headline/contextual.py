
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

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
RAW_NEWS_CSV = ROOT_DIR / "data" / "raw" / "cnbc_raw.csv"
ARTICLE_DATES_CSV = OUTPUT_DIR / "news_tfidf_clean.csv"
CALENDAR_CSV = OUTPUT_DIR / "kurs_clean.csv"

DATE_COLUMN = "date"

MODELS = {
    "finbert": "ProsusAI/finbert",
    "roberta": "FacebookAI/roberta-base",
    "qwen": "Qwen/Qwen2.5-0.5B",
}

# Maximum sequence length, including special tokens
MAX_LENGTH = 512

# Batch size for headline embedding extraction
BATCH_SIZE = 8

# Print progress every N batches
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
# LOAD DAILY HEADLINE DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING DAILY HEADLINE DATA")
    print("=" * 70)

    df = pd.read_csv(CALENDAR_CSV, usecols=[DATE_COLUMN])

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    if df[DATE_COLUMN].isna().any():

        raise ValueError(
            "Terdapat tanggal yang tidak valid."
        )

    if df[DATE_COLUMN].duplicated().any():

        raise ValueError(
            "Terdapat tanggal duplikat pada dataset."
        )

    df = (
        df.sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    raw_articles = pd.read_csv(
        RAW_NEWS_CSV, usecols=["url", "title"], keep_default_na=False
    ).drop_duplicates(subset="url", keep="first")
    # Pertahankan seleksi artikel dan tanggal target dari pipeline alignment.
    article_dates = pd.read_csv(
        ARTICLE_DATES_CSV, usecols=["url", "target_date"], parse_dates=["target_date"]
    )
    articles = article_dates.merge(
        raw_articles, on="url", how="left", validate="one_to_one", indicator=True
    )
    unmatched = articles["_merge"].eq("left_only").sum()
    if unmatched:
        raise ValueError(
            f"{unmatched:,} URL artikel terseleksi tidak ditemukan di {RAW_NEWS_CSV}."
        )

    articles["title"] = articles["title"].str.strip()
    daily_lookup = (
        articles.loc[articles["title"].ne("")]
        .groupby("target_date")["title"]
        .agg(list)
        .to_dict()
    )
    daily_headlines = [daily_lookup.get(date, []) for date in df[DATE_COLUMN]]

    # --------------------------------------------------------
    # FLATTEN HEADLINES
    # --------------------------------------------------------

    all_headline_texts = []

    # Menyimpan hubungan antara tanggal dan headline
    # doc_headline_map[date_idx] = daftar indeks headline
    doc_headline_map = [
        [] for _ in range(len(df))
    ]

    for date_idx, headlines in enumerate(daily_headlines):

        for headline in headlines:

            headline_idx = len(all_headline_texts)

            all_headline_texts.append(headline)

            doc_headline_map[date_idx].append(
                headline_idx
            )

    empty_dates = sum(
        1 for headlines in daily_headlines
        if not headlines
    )

    total_headlines = len(all_headline_texts)

    print(f"Kalender JISDOR     : {CALENDAR_CSV}")
    print(f"Jumlah tanggal      : {len(df):,}")
    print(f"Jumlah headline     : {total_headlines:,}")
    print(f"Tanggal tanpa berita: {empty_dates:,}")

    print(
        f"Rentang tanggal     : "
        f"{df[DATE_COLUMN].min().date()} "
        f"hingga "
        f"{df[DATE_COLUMN].max().date()}"
    )

    if total_headlines > 0:

        print(
            f"Rata-rata headline  : "
            f"{total_headlines / len(df):.2f} per tanggal"
        )

    return (
        df,
        all_headline_texts,
        doc_headline_map
    )


# ============================================================
# PREPARE TOKENIZER
# ============================================================

def load_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True
    )

    tokenizer.padding_side = "right"

    # Qwen biasanya tidak memiliki pad token.
    # EOS digunakan sebagai pad token tanpa menambah
    # vocabulary baru atau mengubah bobot model.
    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token is not None:

            tokenizer.pad_token = tokenizer.eos_token

        else:

            raise ValueError(
                f"Tokenizer {model_id} tidak memiliki "
                "pad_token maupun eos_token."
            )

    return tokenizer


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    last_hidden_state,
    attention_mask,
    special_tokens_mask
):

    """
    Menghitung mean pooling dari token embedding.

    Token yang diabaikan:
    - Padding tokens
    - Special tokens seperti CLS, SEP, BOS, EOS
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
# EXTRACT HEADLINE EMBEDDINGS
# ============================================================

def extract_headline_embeddings(
    model,
    tokenizer,
    headline_texts,
    hidden_size
):

    """
    Menghasilkan satu embedding untuk setiap headline.

    Proses:
    1. Tokenisasi headline.
    2. Dynamic padding.
    3. Forward pass.
    4. Last hidden state.
    5. Mean pooling.
    """

    total_headlines = len(headline_texts)

    headline_embeddings = np.zeros(
        (total_headlines, hidden_size),
        dtype=np.float32
    )

    if total_headlines == 0:

        return headline_embeddings

    total_batches = (
        total_headlines + BATCH_SIZE - 1
    ) // BATCH_SIZE

    model.eval()

    print("\nEXTRACTING HEADLINE EMBEDDINGS")
    print("-" * 70)

    for batch_idx, start_idx in enumerate(
        range(0, total_headlines, BATCH_SIZE),
        start=1
    ):

        end_idx = min(
            start_idx + BATCH_SIZE,
            total_headlines
        )

        batch_texts = headline_texts[
            start_idx:end_idx
        ]

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_special_tokens_mask=True
        )

        # Jangan diam-diam memotong headline.
        # Jika ada yang melebihi batas, hentikan proses
        # agar data tidak terpotong tanpa diketahui.
        if encoded["input_ids"].shape[1] > MAX_LENGTH:

            longest_length = int(
                encoded["attention_mask"].sum(dim=1).max()
            )

            raise ValueError(
                f"Headline melebihi MAX_LENGTH={MAX_LENGTH}. "
                f"Panjang sequence: {longest_length} token. "
                "Periksa headline tersebut sebelum melanjutkan."
            )

        special_tokens_mask = (
            encoded.pop("special_tokens_mask")
        )

        # ----------------------------------------------------
        # MOVE TO DEVICE
        # ----------------------------------------------------

        model_inputs = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        special_tokens_mask = (
            special_tokens_mask.to(DEVICE)
        )

        attention_mask = model_inputs["attention_mask"]

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                **model_inputs,
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

        headline_embeddings[start_idx:end_idx] = (
            embeddings
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_idx % LOG_EVERY == 0
            or batch_idx == total_batches
        ):

            print(
                f"Batch {batch_idx:,}/{total_batches:,} "
                f"| Headlines {end_idx:,}/{total_headlines:,}"
            )

        del (
            encoded,
            model_inputs,
            special_tokens_mask,
            attention_mask,
            outputs,
            last_hidden_state,
            embeddings
        )

    return headline_embeddings


# ============================================================
# AGGREGATE HEADLINE EMBEDDINGS INTO DAILY EMBEDDINGS
# ============================================================

def aggregate_daily_embeddings(
    headline_embeddings,
    doc_headline_map,
    hidden_size
):

    """
    Menggabungkan embedding seluruh headline
    menjadi satu embedding per tanggal.

    Digunakan mean pooling antar-headline.

    Tanggal tanpa headline diberi zero vector.
    """

    n_dates = len(doc_headline_map)

    daily_embeddings = np.zeros(
        (n_dates, hidden_size),
        dtype=np.float32
    )

    for date_idx, headline_indices in enumerate(
        doc_headline_map
    ):

        # Tidak ada headline pada tanggal tersebut
        if not headline_indices:

            continue

        embeddings = headline_embeddings[
            headline_indices
        ]

        daily_embeddings[date_idx] = (
            np.mean(embeddings, axis=0)
        )

    return daily_embeddings


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def process_model(
    model_name,
    model_id,
    df,
    headline_texts,
    doc_headline_map
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

    tokenizer = load_tokenizer(model_id)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModel.from_pretrained(
        model_id
    )

    model = model.to(DEVICE)

    model.eval()

    # Set pad token ID in model config
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

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
    # EXTRACT HEADLINE EMBEDDINGS
    # --------------------------------------------------------

    headline_embeddings = extract_headline_embeddings(
        model=model,
        tokenizer=tokenizer,
        headline_texts=headline_texts,
        hidden_size=hidden_size
    )

    # --------------------------------------------------------
    # AGGREGATE DAILY EMBEDDINGS
    # --------------------------------------------------------

    print("\nAGGREGATING DAILY EMBEDDINGS")
    print("-" * 70)

    daily_embeddings = aggregate_daily_embeddings(
        headline_embeddings=headline_embeddings,
        doc_headline_map=doc_headline_map,
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
            "dengan jumlah tanggal pada input."
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
        / f"{model_name}_headline_features.csv"
    )

    features_df.to_csv(
        output_path,
        index=False,
        float_format="%.8f"
    )

    empty_dates = sum(
        1 for indices in doc_headline_map
        if not indices
    )

    print("\nMODEL SUMMARY")
    print("-" * 70)

    print(f"Model              : {model_name}")
    print(f"Embedding dim      : {hidden_size}")
    print(f"Total dates        : {len(features_df):,}")
    print(f"Total headlines    : {len(headline_texts):,}")
    print(f"Dates without news : {empty_dates:,}")
    print(f"Output shape       : {features_df.shape}")
    print(f"Saved              : {output_path}")

    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del model
    del tokenizer
    del headline_embeddings
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
    print("HEADLINE CONTEXTUAL EMBEDDING EXTRACTION")
    print("=" * 70)

    print(f"Root directory : {ROOT_DIR}")
    print(f"Input artikel  : {RAW_NEWS_CSV}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Device         : {DEVICE}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        df,
        headline_texts,
        doc_headline_map
    ) = load_data()

    # --------------------------------------------------------
    # PROCESS ALL MODELS
    # --------------------------------------------------------

    for model_name, model_id in MODELS.items():

        process_model(
            model_name=model_name,
            model_id=model_id,
            df=df,
            headline_texts=headline_texts,
            doc_headline_map=doc_headline_map
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALL MODELS COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    for model_name in MODELS:

        output_path = (
            OUTPUT_DIR
            / f"{model_name}_headline_features.csv"
        )

        print(f"  - {output_path}")

    print("\nSemua headline contextual embeddings berhasil disimpan.")


if __name__ == "__main__":

    main()
