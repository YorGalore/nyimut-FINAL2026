from __future__ import annotations

import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TOPIC_DIR = ROOT_DIR / "data" / "processed" / "topic_modelling"
INPUT_DOCUMENTS_FILE = TOPIC_DIR / "input_documents.csv"
NEWS_INPUT_FILE = ROOT_DIR / "data" / "processed" / "news_tfidf_clean.csv"
ALIGNED_FILE = ROOT_DIR / "data" / "processed" / "aligned_daily.csv"

EVALUATION_DIR = TOPIC_DIR / "evaluation"
FEATURES_DIR = TOPIC_DIR / "features"

MODEL_NAMES = ["bertopic", "lda", "dtm", "top2vec"]

# Maximum representative words used in coherence and diversity evaluation.
TOP_WORDS = 10

# Number of example articles saved per topic for manual interpretation.
N_SAMPLE_ARTICLES = 3
SAMPLE_TEXT_MAX_CHARS = 450

# Gensim is deliberately restricted to one process to avoid joblib-like
# temporary storage pressure on a Mac with limited free disk space.
GENSIM_PROCESSES = 1

# The model with the highest weighted score is only a quantitative candidate.
# Set one of: "bertopic", "lda", "dtm", "top2vec" to manually override it.
# Leave as None to select the highest-scoring candidate automatically.
# Final choice: LDA retains full document coverage and is the most stable,
# interpretable candidate under the shared forecasting evaluation protocol.
SELECTED_MODEL_OVERRIDE: str | None = "lda"

# Provisional model-selection weights. The script renormalizes weights when
# a metric is unavailable. These weights are a transparent heuristic,
# not a universal scientific standard.
SCORE_WEIGHTS = {
    "coherence_cv_normalized": 0.40,
    "coherence_npmi_normalized": 0.30,
    "topic_diversity": 0.20,
    "document_coverage": 0.10,
}

# For the evaluation tokenizer. This aims to be consistent with the English
# normalized CNBC corpus used in the topic-modeling experiments.
CUSTOM_STOPWORDS = {
    "said", "says", "say", "according", "report", "reports", "reported",
    "reuters", "cnbc", "news", "article", "update", "updated", "today",
    "yesterday", "week", "month", "year", "years", "new", "one", "two",
    "also", "could", "would", "may", "might", "will", "now", "still",
    "much", "many", "make", "made", "get", "got", "take", "took", "go",
    "going", "come", "came", "know", "known", "see", "seen", "look",
    "looking", "first", "last", "next", "back", "well", "however",
    "including", "amid", "amidst", "via", "per", "among", "around",
    "across", "near", "far", "just", "even", "rather", "whether",
    "since", "while", "though", "despite", "within", "without", "toward",
    "towards",
}


# ============================================================
# 2. LOGGING AND DIRECTORIES
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("topic_evaluation")


def create_directories() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved: %s", path)


# ============================================================
# 3. TOKENIZATION AND TOPIC WORD PARSING
# ============================================================

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9'-]*|[0-9]+", re.IGNORECASE)


def get_stopwords() -> set[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)
    except Exception:
        return CUSTOM_STOPWORDS.copy()


STOPWORDS = get_stopwords()


def tokenize(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    tokens = _TOKEN_PATTERN.findall(str(text).lower())
    return [t.strip("-' ") for t in tokens if len(t.strip("-' ")) >= 2
            and t.strip("-' ") not in STOPWORDS]


def parse_topic_words(value: Any) -> list[str]:
    """Parse the comma-separated topic_words column saved by topic_models.py."""
    if pd.isna(value):
        return []
    raw_words = [w.strip().lower() for w in str(value).split(",")]
    result: list[str] = []
    for raw in raw_words:
        # Topic words in the current pipeline are normally unigrams.
        # Keep a token only if it resolves to exactly one token after cleaning.
        tokens = tokenize(raw)
        if len(tokens) == 1 and tokens[0] not in result:
            result.append(tokens[0])
    return result[:TOP_WORDS]


def safe_topic_id(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def topic_probability_columns(df: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for col in df.columns:
        match = re.fullmatch(r"topic_(-?\d+)_prob", str(col))
        if match:
            mapping[int(match.group(1))] = col
    return mapping


# ============================================================
# 4. LOAD CORPUS AND MODEL OUTPUTS
# ============================================================

def choose_text_column(df: pd.DataFrame) -> str | None:
    for col in ["text_norm", "text_clean", "article_text", "full_text", "text"]:
        if col in df.columns:
            return col
    return None


def load_evaluation_corpus() -> tuple[list[list[str]], pd.DataFrame]:
    """Load the exact document corpus saved by the topic-modeling run."""
    source = None
    if INPUT_DOCUMENTS_FILE.exists():
        source = INPUT_DOCUMENTS_FILE
    elif NEWS_INPUT_FILE.exists():
        source = NEWS_INPUT_FILE
    else:
        raise FileNotFoundError(
            "Tidak menemukan input_documents.csv maupun news_tfidf_clean.csv. "
            "Pastikan hasil run_topic_models.py masih tersedia."
        )

    df = pd.read_csv(source, low_memory=False)
    text_col = choose_text_column(df)
    if text_col is None:
        raise ValueError(f"Tidak ada kolom teks yang dikenali pada {source}")

    logger.info("Evaluation corpus: %s | text column: %s", source, text_col)
    texts = [tokenize(x) for x in df[text_col].fillna("").astype(str)]
    texts = [x for x in texts if len(x) >= 2]
    if not texts:
        raise ValueError("Tidak ada dokumen dengan minimal dua token untuk coherence.")
    logger.info("Documents used for coherence: %d", len(texts))
    return texts, df


def read_model_outputs(model_name: str) -> dict[str, Any] | None:
    model_dir = TOPIC_DIR / model_name
    docs_path = model_dir / "document_topics.csv"
    summary_month_path = model_dir / "topic_summary_by_month.csv"
    summary_path = model_dir / "topic_summary.csv"
    daily_path = model_dir / "daily_topic_features.csv"

    missing = [p.name for p in [docs_path, summary_path, daily_path] if not p.exists()]
    if missing:
        logger.warning("Skip %s: missing files %s", model_name, missing)
        return None

    docs = pd.read_csv(docs_path, low_memory=False)
    summary_source = summary_month_path if model_name == "dtm" and summary_month_path.exists() else summary_path
    summary = pd.read_csv(summary_source, low_memory=False)
    daily = pd.read_csv(daily_path, low_memory=False)

    if "target_date" in docs.columns:
        docs["target_date"] = pd.to_datetime(docs["target_date"], errors="coerce")
        docs["month"] = docs["target_date"].dt.to_period("M").astype(str)

    if "dominant_topic" not in docs.columns:
        prob_cols = topic_probability_columns(docs)
        if prob_cols:
            ids = list(prob_cols.keys())
            values = docs[[prob_cols[i] for i in ids]].to_numpy(dtype=float)
            docs["dominant_topic"] = np.asarray(ids)[np.nan_to_num(values).argmax(axis=1)]
        else:
            docs["dominant_topic"] = np.nan

    docs["dominant_topic"] = pd.to_numeric(docs["dominant_topic"], errors="coerce")

    if "topic_id" not in summary.columns:
        raise ValueError(f"{summary_source} tidak mempunyai kolom topic_id")
    if "topic_words" not in summary.columns:
        raise ValueError(f"{summary_source} tidak mempunyai kolom topic_words")

    summary["topic_id"] = summary["topic_id"].apply(safe_topic_id)
    summary["parsed_words"] = summary["topic_words"].apply(parse_topic_words)

    if "month" in summary.columns:
        summary["month"] = summary["month"].astype(str)

    return {
        "model": model_name,
        "docs": docs,
        "summary": summary,
        "daily": daily,
        "docs_path": docs_path,
        "summary_path": summary_source,
        "daily_path": daily_path,
    }


# ============================================================
# 5. GENSIM COHERENCE EVALUATION
# ============================================================

def calculate_coherence(
    topic_rows: pd.DataFrame,
    texts: list[list[str]],
) -> tuple[dict[int, tuple[float, float]], str | None]:
    """Calculate C_V and NPMI per topic unit; gensim runs in one process."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except Exception as exc:
        raise RuntimeError(
            "Tidak bisa mengimpor gensim CoherenceModel. "
            "Coba periksa kompatibilitas gensim, scipy, dan numpy pada environment. "
            "Instal/upgrade gensim jika diperlukan. Detail: " + str(exc)
        ) from exc

    dictionary = Dictionary(texts)
    if len(dictionary) == 0:
        return {}, "Dictionary kosong; coherence tidak dihitung."

    valid_rows: list[int] = []
    valid_topics: list[list[str]] = []
    for idx, row in topic_rows.iterrows():
        words = [w for w in row["parsed_words"] if w in dictionary.token2id]
        # Coherence requires at least two in-vocabulary words.
        if len(words) >= 2:
            valid_rows.append(idx)
            valid_topics.append(words[:TOP_WORDS])

    if not valid_topics:
        return {}, "Tidak ada topik dengan minimal dua kata yang ada di vocabulary corpus."

    scores: dict[int, tuple[float, float]] = {}
    warning_text = None

    try:
        cv_model = CoherenceModel(
            topics=valid_topics,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            topn=TOP_WORDS,
            processes=GENSIM_PROCESSES,
        )
        cv_values = cv_model.get_coherence_per_topic()
    except Exception as exc:
        cv_values = [float("nan")] * len(valid_topics)
        warning_text = f"C_V gagal dihitung: {exc}"

    try:
        npmi_model = CoherenceModel(
            topics=valid_topics,
            texts=texts,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=TOP_WORDS,
            processes=GENSIM_PROCESSES,
        )
        npmi_values = npmi_model.get_coherence_per_topic()
    except Exception as exc:
        npmi_values = [float("nan")] * len(valid_topics)
        msg = f"NPMI gagal dihitung: {exc}"
        warning_text = f"{warning_text}; {msg}" if warning_text else msg

    for row_idx, cv, npmi in zip(valid_rows, cv_values, npmi_values):
        scores[row_idx] = (float(cv), float(npmi))

    return scores, warning_text


# ============================================================
# 6. TOPIC QUALITY METRICS
# ============================================================

def jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a.union(b)
    return len(a.intersection(b)) / len(union) if union else float("nan")


def prepare_topic_rows(model_data: dict[str, Any]) -> pd.DataFrame:
    summary = model_data["summary"].copy()
    model_name = model_data["model"]

    if "topic_name" not in summary.columns:
        summary["topic_name"] = summary["topic_id"].apply(lambda x: f"Topic {x}")

    if "month" not in summary.columns:
        summary["month"] = ""
    else:
        summary["month"] = summary["month"].fillna("").astype(str)

    summary["model"] = model_name
    summary["is_outlier_topic"] = summary["topic_id"].eq(-1)
    summary["n_top_words"] = summary["parsed_words"].apply(len)

    docs = model_data["docs"]
    if "dominant_topic" in docs.columns:
        if model_name == "dtm" and "month" in docs.columns and summary["month"].ne("").any():
            counts = (
                docs.dropna(subset=["dominant_topic"])
                .groupby(["month", "dominant_topic"])
                .size()
                .to_dict()
            )
            summary["document_count_eval"] = summary.apply(
                lambda r: int(counts.get((r["month"], r["topic_id"]), 0)), axis=1
            )
        else:
            counts = docs["dominant_topic"].value_counts().to_dict()
            summary["document_count_eval"] = summary["topic_id"].map(counts).fillna(0).astype(int)
    else:
        summary["document_count_eval"] = 0

    return summary


def compute_topic_diversity(topic_rows: pd.DataFrame) -> float:
    valid = topic_rows[~topic_rows["is_outlier_topic"]]
    all_words = [w for words in valid["parsed_words"] for w in words]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def compute_mean_topic_overlap(topic_rows: pd.DataFrame) -> float:
    """Average pairwise Jaccard overlap between topics (lower means less redundancy)."""
    valid = topic_rows[~topic_rows["is_outlier_topic"]]
    sets = [set(x) for x in valid["parsed_words"] if len(x) >= 2]
    if len(sets) < 2:
        return float("nan")
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            values.append(jaccard_similarity(sets[i], sets[j]))
    return float(np.nanmean(values)) if values else float("nan")


def compute_dtm_temporal_stability(topic_rows: pd.DataFrame) -> pd.DataFrame:
    """Adjacent-month top-word Jaccard; this is temporal continuity, not rerun stability."""
    rows = topic_rows[topic_rows["month"].astype(str).str.len() > 0].copy()
    if rows.empty:
        return pd.DataFrame(columns=[
            "topic_id", "month_from", "month_to", "top_words_jaccard", "shared_word_count"
        ])

    rows["month_period"] = pd.PeriodIndex(rows["month"], freq="M")
    rows = rows.sort_values(["topic_id", "month_period"])
    output = []

    for topic_id, group in rows.groupby("topic_id"):
        group = group.sort_values("month_period")
        records = list(group.iterrows())
        for (idx_a, a), (idx_b, b) in zip(records, records[1:]):
            if b["month_period"].ordinal - a["month_period"].ordinal != 1:
                continue
            words_a = set(a["parsed_words"])
            words_b = set(b["parsed_words"])
            output.append({
                "topic_id": int(topic_id),
                "month_from": str(a["month"]),
                "month_to": str(b["month"]),
                "top_words_jaccard": jaccard_similarity(words_a, words_b),
                "shared_word_count": len(words_a.intersection(words_b)),
            })

    return pd.DataFrame(output)


def evaluate_model(
    model_data: dict[str, Any],
    texts: list[list[str]],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    model_name = model_data["model"]
    topic_rows = prepare_topic_rows(model_data)

    quality_rows = topic_rows.copy()
    quality_rows["coherence_cv"] = np.nan
    quality_rows["coherence_npmi"] = np.nan

    valid_for_coherence = quality_rows[
        (~quality_rows["is_outlier_topic"]) & (quality_rows["n_top_words"] >= 2)
    ]
    coherence_scores, coherence_warning = calculate_coherence(
        valid_for_coherence, texts
    )

    for row_idx, (cv, npmi) in coherence_scores.items():
        quality_rows.loc[row_idx, "coherence_cv"] = cv
        quality_rows.loc[row_idx, "coherence_npmi"] = npmi

    docs = model_data["docs"]
    n_docs = len(docs)
    if "dominant_topic" in docs.columns:
        assigned = pd.to_numeric(docs["dominant_topic"], errors="coerce")
        outlier_count = int((assigned == -1).sum())
    else:
        outlier_count = 0
    outlier_ratio = outlier_count / n_docs if n_docs else float("nan")
    coverage = 1.0 - outlier_ratio if n_docs else float("nan")

    non_outlier = quality_rows[~quality_rows["is_outlier_topic"]]
    c_v = float(non_outlier["coherence_cv"].mean()) if non_outlier["coherence_cv"].notna().any() else float("nan")
    npmi = float(non_outlier["coherence_npmi"].mean()) if non_outlier["coherence_npmi"].notna().any() else float("nan")
    diversity = compute_topic_diversity(quality_rows)
    overlap = compute_mean_topic_overlap(quality_rows)

    temporal = pd.DataFrame()
    temporal_mean = float("nan")
    if model_name == "dtm":
        temporal = compute_dtm_temporal_stability(quality_rows)
        if not temporal.empty and temporal["top_words_jaccard"].notna().any():
            temporal_mean = float(temporal["top_words_jaccard"].mean())

    # Exclude BERTopic's -1 outlier bucket from the topic count.
    unique_topic_count = quality_rows.loc[
        ~quality_rows["is_outlier_topic"], "topic_id"
    ].nunique()

    summary = {
        "model": model_name,
        "n_documents": int(n_docs),
        "n_topics": int(unique_topic_count),
        "n_topic_units_evaluated": int(len(non_outlier)),
        "outlier_count": int(outlier_count),
        "outlier_ratio": outlier_ratio,
        "document_coverage": coverage,
        "coherence_cv_mean": c_v,
        "coherence_npmi_mean": npmi,
        "topic_diversity": diversity,
        "mean_pairwise_topic_word_jaccard": overlap,
        "dtm_adjacent_month_word_jaccard": temporal_mean,
        "coherence_warning": coherence_warning or "",
    }

    # Add topic count and quality fields to an interpretation-ready table.
    quality_columns = [
        "model", "topic_id", "month", "topic_name", "topic_words",
        "n_top_words", "document_count_eval", "coherence_cv", "coherence_npmi",
        "is_outlier_topic",
    ]
    quality = quality_rows[[c for c in quality_columns if c in quality_rows.columns]].copy()

    return quality, summary, temporal


# ============================================================
# 7. REPRESENTATIVE DOCUMENTS AND MANUAL INTERPRETATION TEMPLATE
# ============================================================

def get_sample_documents(
    docs: pd.DataFrame,
    topic_id: int,
    month: str,
    model_name: str,
    n_samples: int = N_SAMPLE_ARTICLES,
) -> list[str]:
    subset = docs.copy()
    subset = subset[pd.to_numeric(subset["dominant_topic"], errors="coerce") == topic_id]

    if model_name == "dtm" and month and "month" in subset.columns:
        subset = subset[subset["month"].astype(str) == str(month)]

    if subset.empty:
        return [""] * n_samples

    # Prefer documents with the largest probability for this topic.
    prob_map = topic_probability_columns(subset)
    if topic_id in prob_map:
        subset["_topic_score"] = pd.to_numeric(subset[prob_map[topic_id]], errors="coerce").fillna(0)
    elif "topic_confidence" in subset.columns:
        subset["_topic_score"] = pd.to_numeric(subset["topic_confidence"], errors="coerce").fillna(0)
    else:
        subset["_topic_score"] = 0.0

    subset = subset.sort_values("_topic_score", ascending=False)

    text_col = choose_text_column(subset)
    if text_col is None:
        return [""] * n_samples

    samples = []
    for text in subset[text_col].fillna("").astype(str).head(n_samples):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > SAMPLE_TEXT_MAX_CHARS:
            text = text[:SAMPLE_TEXT_MAX_CHARS].rsplit(" ", 1)[0] + "..."
        samples.append(text)

    return samples + [""] * (n_samples - len(samples))


def create_interpretation_template(
    model_data: dict[str, Any],
    quality: pd.DataFrame,
) -> pd.DataFrame:
    docs = model_data["docs"]
    model_name = model_data["model"]
    rows = []

    for _, row in quality.iterrows():
        topic_id = safe_topic_id(row["topic_id"])
        month = str(row.get("month", "") or "")
        if topic_id is None:
            continue
        if bool(row.get("is_outlier_topic", False)):
            continue

        samples = get_sample_documents(
            docs=docs,
            topic_id=topic_id,
            month=month,
            model_name=model_name,
        )

        record = {
            "model": model_name,
            "topic_id": topic_id,
            "month": month,
            "topic_name_auto": row.get("topic_name", f"Topic {topic_id}"),
            "top_words": row.get("topic_words", ""),
            "document_count": row.get("document_count_eval", 0),
            "coherence_cv": row.get("coherence_cv", np.nan),
            "coherence_npmi": row.get("coherence_npmi", np.nan),
            "sample_article_1": samples[0],
            "sample_article_2": samples[1],
            "sample_article_3": samples[2],
            # Fill these columns manually after reading the top words and samples.
            "manual_topic_label": "",
            "geopolitical_theme": "",
            "is_geopolitical_topic": "",
            "interpretation_notes": "",
        }
        rows.append(record)

    return pd.DataFrame(rows)


# ============================================================
# 8. PROVISIONAL MODEL RANKING
# ============================================================

def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def add_model_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    result = summary_df.copy()

    # Convert each component to a 0..1 score for transparent weighted ranking.
    result["coherence_cv_normalized"] = result["coherence_cv_mean"].apply(
        lambda x: float(np.clip(safe_float(x), 0.0, 1.0)) if np.isfinite(safe_float(x)) else np.nan
    )
    result["coherence_npmi_normalized"] = result["coherence_npmi_mean"].apply(
        lambda x: float(np.clip((safe_float(x) + 1.0) / 2.0, 0.0, 1.0))
        if np.isfinite(safe_float(x)) else np.nan
    )
    result["topic_diversity_normalized"] = result["topic_diversity"].apply(
        lambda x: float(np.clip(safe_float(x), 0.0, 1.0)) if np.isfinite(safe_float(x)) else np.nan
    )
    result["document_coverage_normalized"] = result["document_coverage"].apply(
        lambda x: float(np.clip(safe_float(x), 0.0, 1.0)) if np.isfinite(safe_float(x)) else np.nan
    )

    components = list(SCORE_WEIGHTS.keys())

    def weighted_score(row: pd.Series) -> float:
        numerator = 0.0
        denominator = 0.0
        for component in components:
            value = safe_float(row.get(component, np.nan))
            if np.isfinite(value):
                weight = SCORE_WEIGHTS[component]
                numerator += value * weight
                denominator += weight
        return numerator / denominator if denominator > 0 else float("nan")

    result["provisional_weighted_score"] = result.apply(weighted_score, axis=1)
    result = result.sort_values(
        "provisional_weighted_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    result["quantitative_rank"] = np.arange(1, len(result) + 1)
    result["selection_note"] = (
        "Kandidat kuantitatif; keputusan akhir perlu mempertimbangkan interpretasi manual "
        "dan performa forecasting di validation/test set."
    )
    return result


# ============================================================
# 9. DAILY FEATURE EXTRACTION AND ALIGNMENT
# ============================================================

def load_aligned_dates() -> pd.DataFrame | None:
    if not ALIGNED_FILE.exists():
        logger.warning("aligned_daily.csv tidak ditemukan; fitur tidak akan dipaksa ke kalender JISDOR.")
        return None
    aligned = pd.read_csv(ALIGNED_FILE, usecols=["date"])
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce").dt.normalize()
    return (
        aligned.dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def align_daily_features(daily: pd.DataFrame, aligned_dates: pd.DataFrame | None) -> pd.DataFrame:
    daily = daily.copy()
    if "date" not in daily.columns:
        raise ValueError("Kolom date tidak ditemukan pada daily_topic_features.csv")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")

    if aligned_dates is not None:
        daily = aligned_dates.merge(daily, on="date", how="left")

    numeric_cols = daily.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # A dominant topic of -1 means no topic/news for the date.
        if col == "dominant_topic":
            daily[col] = daily[col].fillna(-1)
        else:
            daily[col] = daily[col].fillna(0)

    if "model" in daily.columns:
        daily["model"] = daily["model"].fillna("")

    return daily.sort_values("date").reset_index(drop=True)


def extract_daily_feature_files(
    loaded_models: dict[str, dict[str, Any]],
    selected_model: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    aligned_dates = load_aligned_dates()
    wide_frames = []
    selected_daily = None

    for model_name, data in loaded_models.items():
        daily = align_daily_features(data["daily"], aligned_dates)
        out_path = FEATURES_DIR / f"{model_name}_daily_topic_features.csv"
        save_csv(daily, out_path)

        feature_cols = [c for c in daily.columns if c not in {"date", "model"}]
        renamed = daily[["date"] + feature_cols].copy()
        renamed = renamed.rename(columns={c: f"{model_name}__{c}" for c in feature_cols})
        wide_frames.append(renamed)

        if selected_model == model_name:
            selected_daily = daily.copy()

    if wide_frames:
        all_features = wide_frames[0]
        for frame in wide_frames[1:]:
            all_features = all_features.merge(frame, on="date", how="outer")
        all_features = all_features.sort_values("date").reset_index(drop=True)
        value_cols = [c for c in all_features.columns if c != "date"]
        all_features[value_cols] = all_features[value_cols].fillna(0)
        save_csv(all_features, FEATURES_DIR / "all_models_daily_topic_features.csv")
    else:
        all_features = pd.DataFrame()

    if selected_daily is not None:
        save_csv(selected_daily, FEATURES_DIR / "selected_model_daily_topic_features.csv")

    return all_features, selected_daily


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:
    create_directories()

    logger.info("=" * 75)
    logger.info("TOPIC MODELING EVALUATION AND FEATURE EXTRACTION")
    logger.info("=" * 75)
    logger.info("Topic output directory: %s", TOPIC_DIR)
    logger.info("Evaluation output: %s", EVALUATION_DIR)
    logger.info("Feature output: %s", FEATURES_DIR)

    # Load a common corpus so that coherence scores are computed consistently.
    texts, _ = load_evaluation_corpus()

    loaded_models: dict[str, dict[str, Any]] = {}
    all_quality = []
    all_interpretation = []
    all_summaries = []
    all_temporal = []
    errors = []

    for model_name in MODEL_NAMES:
        logger.info("\n%s", "=" * 65)
        logger.info("EVALUATING MODEL: %s", model_name.upper())
        logger.info("%s", "=" * 65)

        try:
            model_data = read_model_outputs(model_name)
            if model_data is None:
                continue

            quality, summary, temporal = evaluate_model(model_data, texts)
            interpretation = create_interpretation_template(model_data, quality)

            loaded_models[model_name] = model_data
            all_quality.append(quality)
            all_summaries.append(summary)
            if not interpretation.empty:
                all_interpretation.append(interpretation)
            if not temporal.empty:
                temporal.insert(0, "model", model_name)
                all_temporal.append(temporal)

            logger.info(
                "%s | topics=%s | C_V=%s | NPMI=%s | diversity=%s | outlier_ratio=%s",
                model_name,
                summary["n_topics"],
                round(summary["coherence_cv_mean"], 4) if np.isfinite(summary["coherence_cv_mean"]) else "NA",
                round(summary["coherence_npmi_mean"], 4) if np.isfinite(summary["coherence_npmi_mean"]) else "NA",
                round(summary["topic_diversity"], 4) if np.isfinite(summary["topic_diversity"]) else "NA",
                round(summary["outlier_ratio"], 4) if np.isfinite(summary["outlier_ratio"]) else "NA",
            )

        except Exception as exc:
            logger.exception("Evaluation failed for %s", model_name)
            errors.append({"model": model_name, "error": str(exc)})

    if not all_summaries:
        raise RuntimeError(
            "Tidak ada model yang berhasil dievaluasi. Periksa struktur folder "
            "data/processed/topic_modelling/ dan file output tiap model."
        )

    quality_df = pd.concat(all_quality, ignore_index=True) if all_quality else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries)
    ranking_df = add_model_ranking(summary_df)
    interpretation_df = pd.concat(all_interpretation, ignore_index=True) if all_interpretation else pd.DataFrame()

    save_csv(quality_df, EVALUATION_DIR / "topic_quality_by_topic.csv")
    save_csv(summary_df, EVALUATION_DIR / "model_evaluation_summary.csv")
    save_csv(ranking_df, EVALUATION_DIR / "model_ranking.csv")
    save_csv(interpretation_df, EVALUATION_DIR / "topic_interpretation_template.csv")

    if all_temporal:
        temporal_df = pd.concat(all_temporal, ignore_index=True)
    else:
        temporal_df = pd.DataFrame(columns=[
            "model", "topic_id", "month_from", "month_to", "top_words_jaccard", "shared_word_count"
        ])
    save_csv(temporal_df, EVALUATION_DIR / "dtm_temporal_word_stability.csv")

    if errors:
        save_csv(pd.DataFrame(errors), EVALUATION_DIR / "evaluation_errors.csv")

    # Choose a quantitative candidate unless a manual override is configured.
    available_models = set(loaded_models.keys())
    selected_model = SELECTED_MODEL_OVERRIDE

    if selected_model is not None and selected_model not in available_models:
        raise ValueError(
            f"SELECTED_MODEL_OVERRIDE='{selected_model}' tidak tersedia. "
            f"Model yang berhasil dibaca: {sorted(available_models)}"
        )

    if selected_model is None:
        candidates = ranking_df[
            ranking_df["model"].isin(available_models)
            & ranking_df["provisional_weighted_score"].notna()
        ]
        if not candidates.empty:
            selected_model = str(candidates.iloc[0]["model"])

    all_features, selected_daily = extract_daily_feature_files(
        loaded_models,
        selected_model,
    )

    selection = {
        "selected_model": selected_model,
        "selection_type": "manual_override" if SELECTED_MODEL_OVERRIDE else "provisional_quantitative_candidate",
        "score_weights": SCORE_WEIGHTS,
        "selected_model_is_final": selected_model == "lda",
        "note": (
            "LDA ditetapkan sebagai model final: cakupan dokumen penuh, interpretabilitas "
            "stabil, dan evaluasi forecasting memakai split kronologis pusat 70/15/15."
            if selected_model == "lda"
            else "Model dipilih melalui override manual; periksa interpretasi dan evaluasi forecasting."
        ),
        "available_models": sorted(available_models),
    }
    save_json(selection, EVALUATION_DIR / "selected_model.json")

    config = {
        "top_words": TOP_WORDS,
        "gensim_processes": GENSIM_PROCESSES,
        "score_weights": SCORE_WEIGHTS,
        "selected_model_override": SELECTED_MODEL_OVERRIDE,
        "coherence_corpus": str(INPUT_DOCUMENTS_FILE if INPUT_DOCUMENTS_FILE.exists() else NEWS_INPUT_FILE),
        "coherence_scope": "All documents from the saved exploratory topic-modeling corpus",
        "warning": "Exploratory corpus-wide evaluation is not a leakage-free forecasting evaluation. Refit topic models on training data only for final time-series experiments.",
    }
    save_json(config, EVALUATION_DIR / "evaluation_config.json")

    logger.info("\n%s", "=" * 75)
    logger.info("TOPIC MODEL EVALUATION FINISHED")
    logger.info("%s", "=" * 75)
    logger.info("Models evaluated: %s", sorted(available_models))
    logger.info("Provisional selected model: %s", selected_model)
    logger.info("Evaluation outputs: %s", EVALUATION_DIR)
    logger.info("Feature outputs: %s", FEATURES_DIR)
    if not ranking_df.empty:
        logger.info("\nMODEL RANKING:\n%s", ranking_df[[
            "quantitative_rank", "model", "n_topics", "coherence_cv_mean",
            "coherence_npmi_mean", "topic_diversity", "document_coverage",
            "provisional_weighted_score",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
