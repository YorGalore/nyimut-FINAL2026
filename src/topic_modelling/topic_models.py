
from __future__ import annotations

import logging
import re
import traceback
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# Input data
INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "news_tfidf_clean.csv"
)

ALIGNED_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "aligned_daily.csv"
)

# CSV outputs: data/processed/topic_modelling/
OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "topic_modelling"
)

# Trained models: models/topic_modelling/
MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "topic_modelling"
)

MODEL_NAMES = [
    "bertopic",
    "lda",
    "dtm",
    "top2vec",
]

# General settings
RANDOM_STATE = 42
TOP_WORDS = 15
MIN_DOCUMENT_TOKENS = 5

# LDA
N_TOPICS = 15
LDA_MAX_FEATURES = 20000
LDA_MIN_DF = 5
LDA_MAX_DF = 0.95
LDA_MAX_ITER = 30

# BERTopic
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BERTOPIC_MIN_TOPIC_SIZE = 15

# DTM
DTM_ITERATIONS = 500
DTM_WORKERS = 4

# Top2Vec
TOP2VEC_SPEED = "learn"
TOP2VEC_WORKERS = 4


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. DIRECTORY SETUP
# ============================================================

def create_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name in MODEL_NAMES:

        (OUTPUT_DIR / model_name).mkdir(
            parents=True,
            exist_ok=True,
        )

        (MODEL_DIR / model_name).mkdir(
            parents=True,
            exist_ok=True,
        )

    (OUTPUT_DIR / "comparison").mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Model directory: %s", MODEL_DIR)


# ============================================================
# 4. CUSTOM STOPWORDS
# ============================================================

CUSTOM_STOPWORDS = {
    "said",
    "says",
    "say",
    "according",
    "report",
    "reports",
    "reported",
    "reuters",
    "cnbc",
    "news",
    "article",
    "update",
    "updated",
    "today",
    "yesterday",
    "week",
    "month",
    "year",
    "years",
    "new",
    "one",
    "two",
    "also",
    "could",
    "would",
    "may",
    "might",
    "will",
    "now",
    "still",
    "much",
    "many",
    "make",
    "made",
    "get",
    "got",
    "take",
    "took",
    "go",
    "going",
    "come",
    "came",
    "know",
    "known",
    "see",
    "seen",
    "look",
    "looking",
    "first",
    "last",
    "next",
    "back",
    "well",
    "however",
    "including",
    "amid",
    "amidst",
    "via",
    "per",
    "among",
    "around",
    "across",
    "near",
    "far",
    "just",
    "even",
    "rather",
    "whether",
    "since",
    "while",
    "though",
    "despite",
    "within",
    "without",
    "toward",
    "towards",
}


def get_stopwords():

    from sklearn.feature_extraction.text import (
        ENGLISH_STOP_WORDS,
    )

    return CUSTOM_STOPWORDS.union(
        set(ENGLISH_STOP_WORDS)
    )


STOPWORDS = get_stopwords()


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():

    logger.info("=" * 70)
    logger.info("LOADING ARTICLE-LEVEL NEWS DATA")
    logger.info("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file tidak ditemukan:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    logger.info("Input: %s", INPUT_FILE)
    logger.info("Original shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))

    required_columns = [
        "target_date",
        "text_norm",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_columns}"
        )

    df = df.copy()

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    df["text_norm"] = (
        df["text_norm"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["target_date"]
    )

    df = df[
        df["text_norm"].str.len() > 0
    ].copy()

    if "url" in df.columns:

        df = df.drop_duplicates(
            subset=["url"],
            keep="first",
        )

    df = df.sort_values(
        "target_date"
    ).reset_index(drop=True)

    df["document_id"] = [
        f"article_{i:06d}"
        for i in range(len(df))
    ]

    logger.info("After cleaning: %s", df.shape)

    logger.info(
        "Date range: %s to %s",
        df["target_date"].min(),
        df["target_date"].max(),
    )

    return df


# ============================================================
# 6. TEXT PREPROCESSING
# ============================================================

def tokenize_text(text):

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z][a-z0-9\-']+|[0-9]+",
        text,
    )

    tokens = [
        token.strip("-'")
        for token in tokens
    ]

    tokens = [
        token
        for token in tokens
        if len(token) >= 2
        and token not in STOPWORDS
    ]

    return tokens


def prepare_documents(df):

    logger.info("=" * 70)
    logger.info("PREPARING DOCUMENTS")
    logger.info("=" * 70)

    df = df.copy()

    df["tokens"] = df["text_norm"].apply(
        tokenize_text
    )

    df["n_tokens_topic"] = df["tokens"].apply(
        len
    )

    df = df[
        df["n_tokens_topic"] >= MIN_DOCUMENT_TOKENS
    ].copy()

    df = df.reset_index(drop=True)

    logger.info(
        "Documents after token filtering: %s",
        len(df),
    )

    logger.info(
        "Average tokens per document: %.2f",
        df["n_tokens_topic"].mean(),
    )

    if len(df) < 20:

        raise ValueError(
            "Jumlah dokumen terlalu sedikit untuk topic modeling."
        )

    return df


# ============================================================
# 7. COMMON UTILITIES
# ============================================================

def save_csv(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    logger.info("Saved CSV: %s", path)


def normalize_distribution(matrix):

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    matrix = np.maximum(
        matrix,
        0.0,
    )

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    row_sums[row_sums == 0] = 1.0

    return matrix / row_sums


def build_document_topic_dataframe(
    df,
    probabilities,
    topic_ids,
    model_name,
):

    probabilities = normalize_distribution(
        probabilities
    )

    topic_columns = [
        f"topic_{topic_id:02d}_prob"
        for topic_id in topic_ids
    ]

    dominant_indices = probabilities.argmax(
        axis=1
    )

    dominant_topic_ids = np.asarray(
        topic_ids
    )[dominant_indices]

    result = df[
        [
            "document_id",
            "target_date",
            "text_norm",
        ]
    ].copy()

    result["model"] = model_name

    result["dominant_topic"] = (
        dominant_topic_ids
    )

    result["topic_confidence"] = (
        probabilities.max(axis=1)
    )

    probability_df = pd.DataFrame(
        probabilities,
        columns=topic_columns,
        index=result.index,
    )

    result = pd.concat(
        [
            result,
            probability_df,
        ],
        axis=1,
    )

    return result


# ============================================================
# 8. DAILY FEATURE AGGREGATION
# ============================================================

def aggregate_daily_features(
    document_topics,
    topic_ids,
    model_name,
):

    topic_columns = [
        f"topic_{topic_id:02d}_prob"
        for topic_id in topic_ids
    ]

    records = []

    for date, group in document_topics.groupby(
        "target_date"
    ):

        probabilities = group[
            topic_columns
        ].to_numpy(dtype=float)

        mean_probabilities = probabilities.mean(
            axis=0
        )

        dominant_topics = group[
            "dominant_topic"
        ].to_numpy()

        dominant_counts = pd.Series(
            dominant_topics
        ).value_counts()

        record = {
            "date": pd.Timestamp(date).normalize(),
            "n_articles": len(group),
        }

        # Mean probability per topic.
        for i, topic_id in enumerate(topic_ids):

            record[
                f"topic_{topic_id:02d}_mean"
            ] = mean_probabilities[i]

        # Dominant-topic article counts.
        for topic_id in topic_ids:

            record[
                f"topic_{topic_id:02d}_count"
            ] = int(
                dominant_counts.get(
                    topic_id,
                    0,
                )
            )

        # Topic entropy.
        p = mean_probabilities[
            mean_probabilities > 0
        ]

        record["topic_entropy"] = float(
            -np.sum(p * np.log(p))
        )

        record["n_topics_present"] = int(
            (mean_probabilities > 0).sum()
        )

        # Use actual topic ID, not matrix column index.
        dominant_index = int(
            mean_probabilities.argmax()
        )

        record["dominant_topic"] = int(
            topic_ids[dominant_index]
        )

        record["dominant_topic_share"] = float(
            mean_probabilities[dominant_index]
        )

        record["model"] = model_name

        records.append(record)

    daily = pd.DataFrame(records)

    if daily.empty:
        return daily

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    # Preserve all JISDOR dates, including dates without news.
    if ALIGNED_FILE.exists():

        aligned = pd.read_csv(
            ALIGNED_FILE,
            usecols=["date"],
        )

        aligned["date"] = pd.to_datetime(
            aligned["date"],
            errors="coerce",
        ).dt.normalize()

        all_dates = (
            aligned["date"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        daily = pd.DataFrame({
            "date": all_dates
        }).merge(
            daily,
            on="date",
            how="left",
        )

    feature_columns = [
        col
        for col in daily.columns
        if (
            col.startswith("topic_")
            and (
                col.endswith("_mean")
                or col.endswith("_count")
            )
        )
    ]

    daily[feature_columns] = (
        daily[feature_columns].fillna(0)
    )

    daily["n_articles"] = (
        daily["n_articles"]
        .fillna(0)
        .astype(int)
    )

    daily["topic_entropy"] = (
        daily["topic_entropy"]
        .fillna(0)
    )

    daily["n_topics_present"] = (
        daily["n_topics_present"]
        .fillna(0)
        .astype(int)
    )

    daily["dominant_topic_share"] = (
        daily["dominant_topic_share"]
        .fillna(0)
    )

    daily["dominant_topic"] = (
        daily["dominant_topic"]
        .fillna(-1)
        .astype(int)
    )

    daily["model"] = model_name

    return daily


# ============================================================
# 9. SAVE COMMON MODEL OUTPUTS
# ============================================================

def save_model_outputs(
    df,
    probabilities,
    topic_ids,
    topic_summary,
    model_name,
):

    output_model_dir = (
        OUTPUT_DIR / model_name
    )

    document_topics = build_document_topic_dataframe(
        df=df,
        probabilities=probabilities,
        topic_ids=topic_ids,
        model_name=model_name,
    )

    daily_features = aggregate_daily_features(
        document_topics=document_topics,
        topic_ids=topic_ids,
        model_name=model_name,
    )

    save_csv(
        document_topics,
        output_model_dir / "document_topics.csv",
    )

    save_csv(
        topic_summary,
        output_model_dir / "topic_summary.csv",
    )

    save_csv(
        daily_features,
        output_model_dir / "daily_topic_features.csv",
    )

    return document_topics, daily_features


# ============================================================
# 10. MODEL 1: BERTOPIC
# ============================================================

def run_bertopic(df):

    logger.info("=" * 70)
    logger.info("RUNNING BERTOPIC")
    logger.info("=" * 70)

    from bertopic import BERTopic

    from sentence_transformers import (
        SentenceTransformer,
    )

    from umap import UMAP

    from hdbscan import HDBSCAN

    documents = df[
        "text_norm"
    ].tolist()

    embedding_model = SentenceTransformer(
        BERTOPIC_EMBEDDING_MODEL
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=BERTOPIC_MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        top_n_words=TOP_WORDS,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(
        documents
    )

    # BERTopic uses hard topic assignments.
    # Each document gets one-hot topic probability.
    topic_ids = sorted(
        topic_model.get_topics().keys()
    )

    probabilities = np.zeros(
        (
            len(documents),
            len(topic_ids),
        ),
        dtype=float,
    )

    topic_to_col = {
        topic_id: i
        for i, topic_id in enumerate(topic_ids)
    }

    for row, topic_id in enumerate(topics):

        if topic_id in topic_to_col:

            probabilities[
                row,
                topic_to_col[topic_id],
            ] = 1.0

    topic_info = topic_model.get_topic_info()

    summary_records = []

    for topic_id in topic_ids:

        topic_words = topic_model.get_topic(
            topic_id
        )

        words = (
            [
                word
                for word, score in topic_words
            ]
            if topic_words is not None
            else []
        )

        info_row = topic_info[
            topic_info["Topic"] == topic_id
        ]

        topic_name = (
            info_row["Name"].iloc[0]
            if len(info_row) > 0
            else f"Topic {topic_id}"
        )

        topic_count = (
            int(info_row["Count"].iloc[0])
            if len(info_row) > 0
            else 0
        )

        summary_records.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_words": ", ".join(words),
            "document_count": topic_count,
        })

    topic_summary = pd.DataFrame(
        summary_records
    )

    # Save trained model to models/, not data/.
    model_path = (
        MODEL_DIR
        / "bertopic"
        / "bertopic_model"
    )

    topic_model.save(
        str(model_path),
        serialization="safetensors",
        save_ctfidf=True,
    )

    save_model_outputs(
        df,
        probabilities,
        topic_ids,
        topic_summary,
        "bertopic",
    )

    return {
        "model": "bertopic",
        "n_topics": len(topic_ids),
        "n_documents": len(df),
        "n_outliers": int(
            np.sum(np.array(topics) == -1)
        ),
    }


# ============================================================
# 11. MODEL 2: LDA
# ============================================================

def run_lda(df):

    logger.info("=" * 70)
    logger.info("RUNNING LDA")
    logger.info("=" * 70)

    from sklearn.feature_extraction.text import (
        CountVectorizer,
    )

    from sklearn.decomposition import (
        LatentDirichletAllocation,
    )

    import joblib

    documents = df[
        "text_norm"
    ].tolist()

    vectorizer = CountVectorizer(
        tokenizer=tokenize_text,
        token_pattern=None,
        lowercase=True,
        max_features=LDA_MAX_FEATURES,
        min_df=LDA_MIN_DF,
        max_df=LDA_MAX_DF,
    )

    document_term_matrix = vectorizer.fit_transform(
        documents
    )

    logger.info(
        "Document-term matrix shape: %s",
        document_term_matrix.shape,
    )

    if document_term_matrix.shape[1] == 0:

        raise ValueError(
            "Vocabulary LDA kosong. Periksa preprocessing."
        )

    lda_model = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=LDA_MAX_ITER,
        learning_method="batch",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        evaluate_every=-1,
        verbose=1,
    )

    probabilities = lda_model.fit_transform(
        document_term_matrix
    )

    feature_names = vectorizer.get_feature_names_out()

    summary_records = []

    for topic_id, component in enumerate(
        lda_model.components_
    ):

        top_indices = component.argsort()[
            -TOP_WORDS:
        ][::-1]

        top_words = feature_names[
            top_indices
        ]

        summary_records.append({
            "topic_id": topic_id,
            "topic_name": f"Topic {topic_id}",
            "topic_words": ", ".join(top_words),
            "document_count": int(
                (
                    probabilities.argmax(axis=1)
                    == topic_id
                ).sum()
            ),
        })

    topic_summary = pd.DataFrame(
        summary_records
    )

    # Save model artifacts to models/lda/.
    model_path = (
        MODEL_DIR / "lda"
    )

    joblib.dump(
        lda_model,
        model_path / "lda_model.joblib",
    )

    joblib.dump(
        vectorizer,
        model_path / "count_vectorizer.joblib",
    )

    save_model_outputs(
        df,
        probabilities,
        list(range(N_TOPICS)),
        topic_summary,
        "lda",
    )

    return {
        "model": "lda",
        "n_topics": N_TOPICS,
        "n_documents": len(df),
        "vocabulary_size": len(feature_names),
    }


# ============================================================
# 12. MODEL 3: DYNAMIC TOPIC MODELING
# ============================================================

def run_dtm(df):

    logger.info("=" * 70)
    logger.info("RUNNING DYNAMIC TOPIC MODELING")
    logger.info("=" * 70)

    import tomotopy as tp

    data = df.copy()

    # Create monthly time slices.
    data["month"] = (
        data["target_date"]
        .dt.to_period("M")
    )

    data = data.sort_values(
        [
            "month",
            "target_date",
        ]
    ).reset_index(drop=True)

    months = sorted(
        data["month"].unique()
    )

    month_to_timepoint = {
        month: i
        for i, month in enumerate(months)
    }

    data["timepoint"] = data["month"].map(
        month_to_timepoint
    )

    logger.info(
        "Number of monthly time slices: %s",
        len(months),
    )

    dtm_model = tp.DTModel(
        k=N_TOPICS,
        t=len(months),
        seed=RANDOM_STATE,
    )

    for row in data.itertuples():

        dtm_model.add_doc(
            words=row.tokens,
            timepoint=int(row.timepoint),
        )

    logger.info(
        "Documents added: %s",
        len(dtm_model.docs),
    )

    dtm_model.train(
        DTM_ITERATIONS,
        workers=DTM_WORKERS,
    )

    # Document-topic distributions.
    probabilities = np.array([
        doc.get_topic_dist()
        for doc in dtm_model.docs
    ])

    probabilities = normalize_distribution(
        probabilities
    )

    topic_ids = list(
        range(N_TOPICS)
    )

    # Extract topic words for each month.
    summary_records = []

    for timepoint, month in enumerate(months):

        for topic_id in topic_ids:

            topic_words = dtm_model.get_topic_words(
                topic_id,
                timepoint,
                top_n=TOP_WORDS,
            )

            words = [
                word
                for word, score in topic_words
            ]

            summary_records.append({
                "timepoint": timepoint,
                "month": str(month),
                "topic_id": topic_id,
                "topic_name": f"Topic {topic_id}",
                "topic_words": ", ".join(words),
            })

    topic_summary = pd.DataFrame(
        summary_records
    )

    save_csv(
        topic_summary,
        OUTPUT_DIR
        / "dtm"
        / "topic_summary_by_month.csv",
    )

    # Save trained model to models/dtm/.
    dtm_model.save(
        str(
            MODEL_DIR
            / "dtm"
            / "dtm_model.bin"
        )
    )

    # Keep the original document order aligned with probabilities.
    output_df = data.drop(
        columns=[
            "month",
            "timepoint",
            "tokens",
            "n_tokens_topic",
        ],
        errors="ignore",
    )

    save_model_outputs(
        output_df,
        probabilities,
        topic_ids,
        topic_summary,
        "dtm",
    )

    return {
        "model": "dtm",
        "n_topics": N_TOPICS,
        "n_documents": len(data),
        "n_timepoints": len(months),
    }


# ============================================================
# 13. MODEL 4: TOP2VEC
# ============================================================

def run_top2vec(df):

    logger.info("=" * 70)
    logger.info("RUNNING TOP2VEC")
    logger.info("=" * 70)

    from top2vec import Top2Vec

    documents = df[
        "text_norm"
    ].tolist()

    top2vec_model = Top2Vec(
        documents=documents,
        speed=TOP2VEC_SPEED,
        workers=TOP2VEC_WORKERS,
        ngram_vocab=True,
        min_count=5,
        keep_documents=True,
    )

    n_topics = top2vec_model.get_num_topics()

    logger.info(
        "Top2Vec detected %s topics.",
        n_topics,
    )

    if n_topics < 1:

        raise ValueError(
            "Top2Vec tidak berhasil menemukan topik."
        )

    topic_words, word_scores, topic_nums = (
        top2vec_model.get_topics(
            n_topics
        )
    )

    topic_ids = [
        int(topic_id)
        for topic_id in topic_nums
    ]

    summary_records = []

    for i, topic_id in enumerate(topic_nums):

        words = topic_words[i]

        summary_records.append({
            "topic_id": int(topic_id),
            "topic_name": f"Topic {topic_id}",
            "topic_words": ", ".join(
                list(words[:TOP_WORDS])
            ),
        })

    topic_summary = pd.DataFrame(
        summary_records
    )

    # Get topic scores for each document.
    (
        document_topic_nums,
        document_topic_scores,
        _,
        _,
    ) = top2vec_model.get_documents_topics(
        doc_ids=list(range(len(documents))),
        num_topics=n_topics,
    )

    topic_to_col = {
        topic_id: i
        for i, topic_id in enumerate(topic_ids)
    }

    probabilities = np.zeros(
        (
            len(documents),
            len(topic_ids),
        ),
        dtype=float,
    )

    for doc_index in range(len(documents)):

        current_topics = np.asarray(
            document_topic_nums[doc_index]
        ).reshape(-1)

        current_scores = np.asarray(
            document_topic_scores[doc_index],
            dtype=float,
        ).reshape(-1)

        for topic_id, score in zip(
            current_topics,
            current_scores,
        ):

            topic_id = int(topic_id)

            if topic_id in topic_to_col:

                probabilities[
                    doc_index,
                    topic_to_col[topic_id],
                ] = max(
                    float(score),
                    0.0,
                )

    probabilities = normalize_distribution(
        probabilities
    )

    dominant_topics = np.asarray(
        topic_ids
    )[probabilities.argmax(axis=1)]

    topic_counts = pd.Series(
        dominant_topics
    ).value_counts()

    topic_summary["document_count"] = (
        topic_summary["topic_id"]
        .map(topic_counts)
        .fillna(0)
        .astype(int)
    )

    # Save trained model to models/top2vec/.
    top2vec_model.save(
        str(
            MODEL_DIR
            / "top2vec"
            / "top2vec_model"
        )
    )

    save_model_outputs(
        df,
        probabilities,
        topic_ids,
        topic_summary,
        "top2vec",
    )

    return {
        "model": "top2vec",
        "n_topics": n_topics,
        "n_documents": len(df),
    }


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

def save_comparison(results):

    if not results:

        logger.warning(
            "No successful models to compare."
        )

        return

    comparison = pd.DataFrame(
        results
    )

    save_csv(
        comparison,
        OUTPUT_DIR
        / "comparison"
        / "topic_modeling_overview.csv",
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main():

    create_directories()

    logger.info("=" * 70)
    logger.info("GEOPOLITICAL NEWS TOPIC MODELING")
    logger.info("=" * 70)

    logger.info(
        "Project root: %s",
        ROOT_DIR,
    )

    logger.info(
        "CSV output: %s",
        OUTPUT_DIR,
    )

    logger.info(
        "Model output: %s",
        MODEL_DIR,
    )

    df = load_data()

    df = prepare_documents(df)

    # Save exact input corpus used for experiments.
    save_csv(
        df.drop(
            columns=["tokens"],
            errors="ignore",
        ),
        OUTPUT_DIR / "input_documents.csv",
    )

    model_functions = {
        "bertopic": run_bertopic,
        "lda": run_lda,
        "dtm": run_dtm,
        "top2vec": run_top2vec,
    }

    results = []
    errors = []

    for model_name in MODEL_NAMES:

        logger.info("\n")
        logger.info("#" * 70)
        logger.info(
            "STARTING MODEL: %s",
            model_name.upper(),
        )
        logger.info("#" * 70)

        try:

            result = model_functions[
                model_name
            ](df)

            results.append(result)

            logger.info(
                "MODEL %s COMPLETED SUCCESSFULLY",
                model_name.upper(),
            )

        except Exception as error:

            logger.exception(
                "MODEL %s FAILED",
                model_name.upper(),
            )

            errors.append({
                "model": model_name,
                "error": str(error),
                "traceback": traceback.format_exc(),
            })

    save_comparison(results)

    if errors:

        save_csv(
            pd.DataFrame(errors),
            OUTPUT_DIR
            / "comparison"
            / "model_errors.csv",
        )

    logger.info("\n")
    logger.info("=" * 70)
    logger.info("TOPIC MODELING PIPELINE FINISHED")
    logger.info("=" * 70)

    logger.info(
        "Successful models: %s",
        len(results),
    )

    logger.info(
        "Failed models: %s",
        len(errors),
    )

    logger.info(
        "CSV outputs: %s",
        OUTPUT_DIR,
    )

    logger.info(
        "Trained models: %s",
        MODEL_DIR,
    )

    for result in results:
        logger.info("SUCCESS: %s", result)

    for error in errors:
        logger.error(
            "FAILED: %s | %s",
            error["model"],
            error["error"],
        )


if __name__ == "__main__":
    main()
