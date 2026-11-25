
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "news_tfidf_clean.csv"
)

OPTIONAL_TEXT_SOURCE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "cnbc_clean.csv"
)

NER_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "ner"
)

OUTPUT_DIR = NER_OUTPUT_DIR / "evaluation"

# Number of articles to annotate manually.
SAMPLE_SIZE = 200

# Reproducible sampling.
RANDOM_STATE = 42

# Same text-column candidates as ner.py.
TEXT_COLUMN_CANDIDATES = [
    "text_clean_original",
    "text_clean",
    "article_text",
    "full_text",
    "body_clean",
    "body",
    "text",
    "text_norm",
]

# Gold annotation labels must match ner.py.
ENTITY_LABELS = [
    "country",
    "central bank",
    "economic event",
]

# Annotation schema.
GOLD_COLUMNS = [
    "document_id",
    "target_date",
    "url",
    "entity_text",
    "entity_label",
    "start_char",
    "end_char",
    "annotation_status",
    "annotator",
    "notes",
]


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("prepare_annotation")


# ============================================================
# 3. HELPERS
# ============================================================

def create_directories() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_stable_document_id(
    row: pd.Series,
    row_number: int,
) -> str:

    url = str(row.get("url", "")).strip()

    if url and url.lower() != "nan":
        digest = hashlib.sha1(
            url.encode("utf-8")
        ).hexdigest()[:16]

        return f"news_{digest}"

    return f"news_row_{row_number:07d}"


def choose_text_column(
    df: pd.DataFrame,
) -> str:

    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Tidak menemukan kolom teks yang sesuai. "
        f"Kolom tersedia: {list(df.columns)}"
    )


# ============================================================
# 4. LOAD ARTICLES
# ============================================================

def load_articles() -> tuple[pd.DataFrame, str]:

    logger.info("Loading input: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if "target_date" not in df.columns:
        raise ValueError(
            "Kolom target_date tidak ditemukan."
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    ).dt.normalize()

    df = df.dropna(
        subset=["target_date"]
    ).copy()

    # --------------------------------------------------------
    # Enrich original article text.
    # Keep the same procedure as ner.py.
    # --------------------------------------------------------

    if (
        OPTIONAL_TEXT_SOURCE.exists()
        and "url" in df.columns
    ):

        try:
            clean_df = pd.read_csv(
                OPTIONAL_TEXT_SOURCE
            )

            if "url" in clean_df.columns:

                candidate_columns = [
                    column
                    for column in [
                        "url",
                        "text_clean",
                        "body_clean",
                    ]
                    if column in clean_df.columns
                ]

                clean_df = clean_df[
                    candidate_columns
                ].drop_duplicates(
                    subset=["url"],
                    keep="first",
                )

                rename_map = {}

                if "text_clean" in clean_df.columns:
                    rename_map[
                        "text_clean"
                    ] = "text_clean_original"

                elif "body_clean" in clean_df.columns:
                    rename_map[
                        "body_clean"
                    ] = "text_clean_original"

                clean_df = clean_df.rename(
                    columns=rename_map
                )

                if "text_clean_original" in clean_df.columns:

                    df = df.merge(
                        clean_df[
                            ["url", "text_clean_original"]
                        ],
                        on="url",
                        how="left",
                    )

                    logger.info(
                        "Original text enriched: %s/%s",
                        int(
                            df["text_clean_original"]
                            .notna()
                            .sum()
                        ),
                        len(df),
                    )

        except Exception as exc:
            logger.warning(
                "Optional enrichment skipped: %s",
                exc,
            )

    # Same fallback behavior as ner.py.
    if (
        "text_clean_original" in df.columns
        and "text_norm" in df.columns
    ):

        df["text_clean_original"] = (
            df["text_clean_original"].where(
                df["text_clean_original"].notna()
                & df["text_clean_original"]
                .astype(str)
                .str.strip()
                .ne(""),
                df["text_norm"],
            )
        )

    text_column = choose_text_column(df)

    df["ner_text"] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["ner_text"].str.len() > 0
    ].copy()

    # Same URL deduplication as ner.py.
    if "url" in df.columns:

        df = df.drop_duplicates(
            subset=["url"],
            keep="first",
        )

    # Same sorting as ner.py.
    sort_columns = ["target_date"]

    if "url" in df.columns:
        sort_columns.append("url")

    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    # IMPORTANT:
    # Preserve exactly the document_id logic in ner.py.
    df["document_id"] = [
        get_stable_document_id(row, i)
        for i, (_, row) in enumerate(df.iterrows())
    ]

    logger.info("Selected text column: %s", text_column)
    logger.info("Available articles: %s", len(df))
    logger.info(
        "Date range: %s to %s",
        df["target_date"].min(),
        df["target_date"].max(),
    )

    return df, text_column


# ============================================================
# 5. STRATIFIED SAMPLING BY QUARTER
# ============================================================

def allocate_samples(
    group_sizes: pd.Series,
    sample_size: int,
) -> dict[str, int]:

    total = int(group_sizes.sum())

    if sample_size >= total:
        return {
            str(key): int(value)
            for key, value in group_sizes.items()
        }

    # Proportional allocation.
    raw_allocations = (
        group_sizes / total
    ) * sample_size

    allocations = np.floor(
        raw_allocations
    ).astype(int)

    remaining = (
        sample_size - int(allocations.sum())
    )

    # Distribute remaining samples according to
    # the largest fractional remainders.
    remainders = (
        raw_allocations - allocations
    ).sort_values(
        ascending=False,
        kind="stable",
    )

    for key in remainders.index[:remaining]:
        allocations.loc[key] += 1

    return {
        str(key): int(value)
        for key, value in allocations.items()
    }


def stratified_quarter_sampling(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Performing stratified sampling by quarter."
    )

    df = df.copy()

    df["sampling_period"] = (
        df["target_date"]
        .dt.to_period("Q")
        .astype(str)
    )

    group_sizes = (
        df.groupby("sampling_period")
        .size()
        .sort_index()
    )

    actual_sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    allocations = allocate_samples(
        group_sizes,
        actual_sample_size,
    )

    sampled_groups = []

    for period, group in df.groupby(
        "sampling_period",
        sort=True,
    ):

        n = allocations.get(period, 0)

        if n == 0:
            continue

        sampled = group.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

        sampled_groups.append(sampled)

    sample_df = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    sample_df = sample_df.sort_values(
        ["target_date", "document_id"]
    ).reset_index(drop=True)

    logger.info(
        "Sampled articles: %s",
        len(sample_df),
    )

    return sample_df


# ============================================================
# 6. PREPARE ANNOTATION ARTICLES
# ============================================================

def make_annotation_articles(
    sample_df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "document_id",
        "target_date",
        "url",
        "section",
        "article_text",
        "text_length",
        "sampling_period",
    ]

    result = pd.DataFrame()

    result["document_id"] = (
        sample_df["document_id"]
    )

    result["target_date"] = (
        sample_df["target_date"]
        .dt.strftime("%Y-%m-%d")
    )

    result["url"] = (
        sample_df["url"]
        if "url" in sample_df.columns
        else ""
    )

    result["section"] = (
        sample_df["section"]
        if "section" in sample_df.columns
        else ""
    )

    result["article_text"] = (
        sample_df["ner_text"]
    )

    result["text_length"] = (
        sample_df["ner_text"].str.len()
    )

    result["sampling_period"] = (
        sample_df["sampling_period"]
    )

    return result[columns]


# ============================================================
# 7. PREPARE GOLD ANNOTATION TEMPLATE
# ============================================================

def make_gold_template(
    sample_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    One initial row per article.

    If an article has no entity:
        Keep entity fields empty.
        Set annotation_status = ANNOTATED.

    If an article has entities:
        Duplicate the article row once per entity.
        Fill entity_text, entity_label, start_char, end_char.
        Set annotation_status = ANNOTATED.
    """

    gold = pd.DataFrame({
        "document_id": sample_df["document_id"],
        "target_date": sample_df["target_date"].dt.strftime(
            "%Y-%m-%d"
        ),
        "url": (
            sample_df["url"]
            if "url" in sample_df.columns
            else ""
        ),
        "entity_text": "",
        "entity_label": "",
        "start_char": "",
        "end_char": "",
        "annotation_status": "NOT_ANNOTATED",
        "annotator": "",
        "notes": "",
    })

    return gold[GOLD_COLUMNS]


# ============================================================
# 8. SAMPLING SUMMARY
# ============================================================

def make_sampling_summary(
    original_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:

    original_counts = (
        original_df.assign(
            sampling_period=original_df[
                "target_date"
            ].dt.to_period("Q").astype(str)
        )
        .groupby("sampling_period")
        .size()
        .rename("population_count")
    )

    sample_counts = (
        sample_df.groupby("sampling_period")
        .size()
        .rename("sample_count")
    )

    summary = pd.concat(
        [original_counts, sample_counts],
        axis=1,
    ).fillna(0)

    summary["population_count"] = (
        summary["population_count"].astype(int)
    )

    summary["sample_count"] = (
        summary["sample_count"].astype(int)
    )

    summary["sampling_fraction"] = (
        summary["sample_count"]
        / summary["population_count"]
    )

    summary = summary.reset_index()

    return summary


# ============================================================
# 9. ANNOTATION GUIDELINES
# ============================================================

def save_annotation_guidelines() -> None:

    content = """
NER GOLD ANNOTATION GUIDELINES
================================

PROJECT:
Geopolitical News Named Entity Recognition

MODEL:
GLiNER - urchade/gliner_small-v2.1

ENTITY LABELS:
1. country
2. central bank
3. economic event

ANNOTATION FILE:
gold_entities.csv

ARTICLE TEXT:
annotation_articles.csv

--------------------------------
1. COUNTRY
--------------------------------

Annotate explicit mentions of countries
or geopolitical territories that function
as country references in the article.

Examples:
- United States
- U.S.
- China
- Russia
- Iran
- Ukraine
- Japan

Do not annotate:
- Nationalities used only as adjectives
  unless the annotation protocol explicitly
  treats them as country mentions.
- City names.
- Company names.

--------------------------------
2. CENTRAL BANK
--------------------------------

Annotate explicit mentions of central banks
and their commonly used abbreviations.

Examples:
- Federal Reserve
- Fed
- European Central Bank
- ECB
- People's Bank of China
- PBOC
- Bank of Japan
- BOJ
- Bank Indonesia

Annotate the full entity span present in the text.

--------------------------------
3. ECONOMIC EVENT
--------------------------------

Annotate explicit economic events or
economic phenomena relevant to the research.

Examples:
- inflation
- interest rate hike
- monetary tightening
- economic sanctions
- trade war
- tariffs
- currency devaluation
- sovereign debt crisis
- recession
- banking crisis
- oil embargo

Annotate the shortest complete phrase
that expresses the economic event.

Do not annotate generic words such as
"economy", "market", or "government"
unless they form part of a specific
economic event expression.

--------------------------------
4. SPAN ANNOTATION
--------------------------------

start_char:
Zero-based character position where the
entity starts.

end_char:
Exclusive character position immediately
after the entity ends.

Example:

Text:
"The Federal Reserve raised interest rates."

Entity:
"Federal Reserve"

start_char = 4
end_char = 19

IMPORTANT:
The extracted substring must satisfy:

article_text[start_char:end_char] == entity_text

Use exact text from the original article.
Do not use normalized entity names here.

--------------------------------
5. LABEL VALUES
--------------------------------

Use exactly:

country
central bank
economic event

Do not change capitalization or spelling.

--------------------------------
6. ANNOTATION STATUS
--------------------------------

NOT_ANNOTATED:
Article has not been reviewed.

ANNOTATED:
Article has been reviewed.

If no relevant entity is found:
- Keep entity_text empty.
- Keep entity_label empty.
- Keep start_char and end_char empty.
- Change annotation_status to ANNOTATED.

If multiple entities are found:
- Create one row per entity.
- Repeat document_id and article metadata.
- Set annotation_status to ANNOTATED.

--------------------------------
7. IMPORTANT
--------------------------------

Annotate based on article_text.

Do not copy GLiNER predictions as gold labels.

The gold annotations will be used to calculate:
- Precision
- Recall
- F1-score
- Micro F1
- Macro F1
- Per-class metrics
- Exact span match

For research reliability, ideally use two
independent annotators and adjudicate
disagreements before final evaluation.
"""

    path = OUTPUT_DIR / "ANNOTATION_GUIDELINES.txt"

    path.write_text(
        content.strip(),
        encoding="utf-8",
    )

    logger.info("Saved: %s", path)


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

def save_outputs(
    original_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    text_column: str,
) -> None:

    articles = make_annotation_articles(
        sample_df
    )

    gold_template = make_gold_template(
        sample_df
    )

    sampling_summary = make_sampling_summary(
        original_df,
        sample_df,
    )

    # Article text for manual annotation.
    articles.to_csv(
        OUTPUT_DIR / "annotation_articles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Gold annotation template.
    gold_template.to_csv(
        OUTPUT_DIR / "gold_entities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Sampling distribution by quarter.
    sampling_summary.to_csv(
        OUTPUT_DIR / "annotation_sampling_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_annotation_guidelines()

    metadata = {
        "input_file": str(INPUT_FILE),
        "optional_text_source": (
            str(OPTIONAL_TEXT_SOURCE)
            if OPTIONAL_TEXT_SOURCE.exists()
            else None
        ),
        "selected_text_column": text_column,
        "sampling_method": "Proportional stratified random sampling by calendar quarter",
        "random_state": RANDOM_STATE,
        "requested_sample_size": SAMPLE_SIZE,
        "actual_sample_size": len(sample_df),
        "population_size": len(original_df),
        "date_start": str(
            original_df["target_date"].min().date()
        ),
        "date_end": str(
            original_df["target_date"].max().date()
        ),
        "entity_labels": ENTITY_LABELS,
        "gold_annotation_file": "gold_entities.csv",
        "span_convention": "Zero-based start, exclusive end",
    }

    with open(
        OUTPUT_DIR / "annotation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("=" * 65)
    logger.info("ANNOTATION PREPARATION COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "Population: %s articles",
        len(original_df),
    )

    logger.info(
        "Annotation sample: %s articles",
        len(sample_df),
    )

    logger.info(
        "Selected text column: %s",
        text_column,
    )

    logger.info(
        "Output directory: %s",
        OUTPUT_DIR,
    )

    logger.info("Generated files:")

    for filename in [
        "annotation_articles.csv",
        "gold_entities.csv",
        "annotation_sampling_summary.csv",
        "ANNOTATION_GUIDELINES.txt",
        "annotation_metadata.json",
    ]:

        logger.info(
            "  - %s",
            OUTPUT_DIR / filename,
        )


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:

    logger.info("=" * 65)
    logger.info("NER GOLD ANNOTATION PREPARATION")
    logger.info("=" * 65)

    create_directories()

    original_df, text_column = load_articles()

    if original_df.empty:
        raise ValueError(
            "Tidak ada artikel yang dapat digunakan."
        )

    sample_df = stratified_quarter_sampling(
        original_df
    )

    if sample_df.empty:
        raise ValueError(
            "Sampling menghasilkan dataset kosong."
        )

    save_outputs(
        original_df=original_df,
        sample_df=sample_df,
        text_column=text_column,
    )


if __name__ == "__main__":
    main()
