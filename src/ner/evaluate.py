from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# 1. PROJECT PATHS AND DEFAULT CONFIGURATION
# ============================================================================

# Intended location in the project: src/ner/evaluate_ner.py
ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "news_tfidf_clean.csv"
OPTIONAL_TEXT_SOURCE = ROOT_DIR / "data" / "processed" / "cnbc_clean.csv"
NER_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "ner"

PREDICTION_FILE = NER_OUTPUT_DIR / "entity_mentions.csv"
GOLD_DOCUMENTS_FILE = NER_OUTPUT_DIR / "gold_documents.csv"
GOLD_ENTITIES_FILE = NER_OUTPUT_DIR / "gold_entities.csv"
EVALUATION_OUTPUT_DIR = NER_OUTPUT_DIR / "evaluation"
METADATA_FILE = NER_OUTPUT_DIR / "run_metadata.json"

ENTITY_LABELS = ["country", "central bank", "economic event"]
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

# All thresholds in this list are >= the standard NER.py threshold (0.45).
# The prediction file cannot recover predictions discarded by GLiNER at inference.
DEFAULT_CONFIDENCE_THRESHOLDS = [
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ner_evaluation")


# ============================================================================
# 2. INPUT DATA PREPARATION (MUST MATCH ner.py DOCUMENT IDs AND TEXT)
# ============================================================================

def stable_document_id(row: pd.Series, row_number: int) -> str:
    url = str(row.get("url", "")).strip()
    if url and url.lower() != "nan":
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return f"news_{digest}"
    return f"news_row_{row_number:07d}"


def load_source_articles() -> pd.DataFrame:
    """Rebuild article IDs/text using the same logic as the supplied ner.py."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File input NER tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "target_date" not in df.columns:
        raise ValueError(
            "Kolom target_date tidak ada di news_tfidf_clean.csv. "
            "Script ini mengikuti persyaratan dataset pada ner.py."
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"], errors="coerce"
    ).dt.normalize()
    df = df.dropna(subset=["target_date"]).copy()

    # Same optional enrichment logic as ner.py.
    if OPTIONAL_TEXT_SOURCE.exists() and "url" in df.columns:
        try:
            clean_df = pd.read_csv(OPTIONAL_TEXT_SOURCE)
            if "url" in clean_df.columns:
                candidate_columns = [
                    col for col in ["url", "text_clean", "body_clean"]
                    if col in clean_df.columns
                ]
                clean_df = clean_df[candidate_columns].drop_duplicates(
                    subset=["url"], keep="first"
                )

                rename_map: dict[str, str] = {}
                if "text_clean" in clean_df.columns:
                    rename_map["text_clean"] = "text_clean_original"
                elif "body_clean" in clean_df.columns:
                    rename_map["body_clean"] = "text_clean_original"
                clean_df = clean_df.rename(columns=rename_map)

                if "text_clean_original" in clean_df.columns:
                    df = df.merge(
                        clean_df[["url", "text_clean_original"]],
                        on="url",
                        how="left",
                    )
        except Exception as exc:
            logger.warning("Enrichment cnbc_clean.csv dilewati: %s", exc)

    if "text_clean_original" in df.columns and "text_norm" in df.columns:
        df["text_clean_original"] = df["text_clean_original"].where(
            df["text_clean_original"].notna()
            & df["text_clean_original"].astype(str).str.strip().ne(""),
            df["text_norm"],
        )

    text_column = next(
        (col for col in TEXT_COLUMN_CANDIDATES if col in df.columns), None
    )
    if text_column is None:
        raise ValueError(
            "Kolom teks tidak ditemukan. Kolom yang dicari: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
        )

    df["ner_text"] = df[text_column].fillna("").astype(str).str.strip()
    df = df[df["ner_text"].str.len() > 0].copy()

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")

    sort_columns = ["target_date"] + (["url"] if "url" in df.columns else [])
    df = df.sort_values(sort_columns).reset_index(drop=True)
    df["document_id"] = [
        stable_document_id(row, i) for i, (_, row) in enumerate(df.iterrows())
    ]

    keep_columns = ["document_id", "target_date", "ner_text"]
    for optional in ["url", "section"]:
        if optional in df.columns:
            keep_columns.append(optional)

    logger.info(
        "Source articles loaded: %s | text column: %s | date range: %s - %s",
        len(df), text_column, df["target_date"].min(), df["target_date"].max()
    )
    return df[keep_columns].copy()


# ============================================================================
# 3. GOLD ANNOTATION TEMPLATE
# ============================================================================

def prepare_gold_templates(sample_size: int, random_state: int) -> None:
    """Create blank, human-annotation templates; does NOT invent gold labels."""
    NER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    articles = load_source_articles()

    # Do not include documents that the NER pipeline explicitly failed to process.
    errors_file = NER_OUTPUT_DIR / "ner_errors.csv"
    if errors_file.exists():
        try:
            errors_df = pd.read_csv(errors_file, usecols=["document_id"])
            failed_ids = set(errors_df["document_id"].dropna().astype(str))
            if failed_ids:
                before = len(articles)
                articles = articles[~articles["document_id"].isin(failed_ids)].copy()
                logger.warning(
                    "Excluded %s articles listed in ner_errors.csv from annotation sampling.",
                    before - len(articles),
                )
        except Exception as exc:
            logger.warning("Could not apply ner_errors.csv exclusion: %s", exc)

    if sample_size <= 0:
        sample = articles.copy()
    else:
        n = min(sample_size, len(articles))
        sample = articles.sample(n=n, random_state=random_state).copy()
        sample = sample.sort_values("target_date").reset_index(drop=True)

    docs = sample.rename(columns={"ner_text": "text"})
    docs["annotation_complete"] = 0
    docs["annotator"] = ""
    docs["annotation_notes"] = ""

    # One row per manually identified entity mention. Empty file is intentional.
    gold_entities = pd.DataFrame(columns=[
        "document_id", "entity_text", "entity_label", "start_char", "end_char"
    ])

    docs.to_csv(GOLD_DOCUMENTS_FILE, index=False)
    gold_entities.to_csv(GOLD_ENTITIES_FILE, index=False)

    logger.info("Gold annotation templates created.")
    logger.info("Documents: %s", len(docs))
    logger.info("Documents template: %s", GOLD_DOCUMENTS_FILE)
    logger.info("Entity template: %s", GOLD_ENTITIES_FILE)
    logger.info("No gold labels were generated automatically; annotate them manually.")


# ============================================================================
# 4. LOAD / VALIDATE GOLD AND PREDICTED ENTITIES
# ============================================================================

def normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def load_gold_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GOLD_DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Gold document file tidak ditemukan: {GOLD_DOCUMENTS_FILE}\n"
            "Jalankan terlebih dahulu dengan --prepare-gold."
        )
    if not GOLD_ENTITIES_FILE.exists():
        raise FileNotFoundError(
            f"Gold entity file tidak ditemukan: {GOLD_ENTITIES_FILE}\n"
            "Jalankan terlebih dahulu dengan --prepare-gold."
        )

    docs = pd.read_csv(GOLD_DOCUMENTS_FILE, keep_default_na=False)
    required_doc_cols = {"document_id", "text", "annotation_complete"}
    missing = required_doc_cols - set(docs.columns)
    if missing:
        raise ValueError(f"Kolom wajib di gold_documents.csv tidak ada: {sorted(missing)}")

    docs["document_id"] = docs["document_id"].astype(str)
    docs["text"] = docs["text"].fillna("").astype(str)
    complete_values = docs["annotation_complete"].astype(str).str.strip().str.lower()
    complete_mask = complete_values.isin({"1", "true", "yes", "y", "complete", "completed"})
    incomplete_count = int((~complete_mask).sum())
    docs = docs.loc[complete_mask].copy()

    if docs.empty:
        raise ValueError(
            "Tidak ada dokumen dengan annotation_complete=1. "
            "Setelah seluruh entitas dalam suatu artikel diperiksa, ubah kolom "
            "annotation_complete menjadi 1. Untuk artikel tanpa entitas, tetap tandai 1."
        )

    if docs["document_id"].duplicated().any():
        duplicates = docs.loc[docs["document_id"].duplicated(), "document_id"].head(10).tolist()
        raise ValueError(f"document_id duplikat di gold_documents.csv: {duplicates}")

    gold = pd.read_csv(GOLD_ENTITIES_FILE, keep_default_na=False)
    required_gold_cols = {
        "document_id", "entity_text", "entity_label", "start_char", "end_char"
    }
    missing = required_gold_cols - set(gold.columns)
    if missing:
        raise ValueError(f"Kolom wajib di gold_entities.csv tidak ada: {sorted(missing)}")

    gold["document_id"] = gold["document_id"].astype(str)
    gold["entity_text"] = gold["entity_text"].astype(str)
    gold["entity_label"] = gold["entity_label"].map(normalize_label)

    # Ignore fully blank lines often added by spreadsheet editors.
    gold = gold.loc[
        gold["entity_text"].str.strip().ne("")
        | gold["entity_label"].str.strip().ne("")
    ].copy()

    unknown_docs = sorted(set(gold["document_id"]) - set(docs["document_id"]))
    if unknown_docs:
        raise ValueError(
            "Ada gold entity yang document_id-nya tidak termasuk dokumen "
            f"yang annotation_complete=1. Contoh: {unknown_docs[:10]}"
        )

    for col in ["start_char", "end_char"]:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")
    if gold[["start_char", "end_char"]].isna().any().any():
        bad_rows = gold.index[gold[["start_char", "end_char"]].isna().any(axis=1)].tolist()[:10]
        raise ValueError(f"start_char/end_char tidak valid pada baris gold: {bad_rows}")

    gold["start_char"] = gold["start_char"].astype(int)
    gold["end_char"] = gold["end_char"].astype(int)

    text_by_doc = docs.set_index("document_id")["text"].to_dict()
    invalid_spans = []
    invalid_labels = []
    for idx, row in gold.iterrows():
        label = row["entity_label"]
        doc_text = text_by_doc[row["document_id"]]
        start, end = int(row["start_char"]), int(row["end_char"])
        if label not in ENTITY_LABELS:
            invalid_labels.append((idx, label))
        if start < 0 or end <= start or end > len(doc_text):
            invalid_spans.append((idx, start, end, len(doc_text)))
            continue
        actual_text = doc_text[start:end]
        if actual_text != row["entity_text"]:
            invalid_spans.append((idx, start, end, f"substring={actual_text!r}, entity_text={row['entity_text']!r}"))

    if invalid_labels:
        raise ValueError(
            "Label gold tidak dikenal. Gunakan persis: "
            + ", ".join(ENTITY_LABELS)
            + f". Contoh baris bermasalah: {invalid_labels[:10]}"
        )
    if invalid_spans:
        raise ValueError(
            "Ada offset gold yang tidak sesuai dengan text. start_char harus 0-based, "
            "end_char harus exclusive, dan text[start_char:end_char] harus sama persis "
            f"dengan entity_text. Contoh: {invalid_spans[:10]}"
        )

    logger.info(
        "Gold loaded: %s fully annotated documents | %s entity mentions | %s incomplete docs ignored",
        len(docs), len(gold), incomplete_count
    )
    return docs, gold


def load_predictions(annotated_doc_ids: set[str], prediction_file: Path) -> pd.DataFrame:
    if not prediction_file.exists():
        raise FileNotFoundError(f"File prediksi NER tidak ditemukan: {prediction_file}")

    pred = pd.read_csv(prediction_file, keep_default_na=False)
    required_cols = {
        "document_id", "entity_text", "entity_label", "start_char", "end_char"
    }
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"Kolom wajib di file prediksi tidak ada: {sorted(missing)}")

    pred["document_id"] = pred["document_id"].astype(str)
    pred["entity_text"] = pred["entity_text"].astype(str)
    pred["entity_label"] = pred["entity_label"].map(normalize_label)

    pred = pred[pred["document_id"].isin(annotated_doc_ids)].copy()
    for col in ["start_char", "end_char"]:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    pred = pred.dropna(subset=["start_char", "end_char"]).copy()
    pred["start_char"] = pred["start_char"].astype(int)
    pred["end_char"] = pred["end_char"].astype(int)

    if "confidence" in pred.columns:
        pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0)
    else:
        pred["confidence"] = 1.0

    invalid = pred[
        (pred["start_char"] < 0)
        | (pred["end_char"] <= pred["start_char"])
    ]
    if not invalid.empty:
        logger.warning("Menghapus %s prediksi dengan offset invalid.", len(invalid))
        pred = pred.drop(index=invalid.index)

    unknown_labels = sorted(set(pred["entity_label"]) - set(ENTITY_LABELS))
    if unknown_labels:
        logger.warning("Prediksi memiliki label di luar skema: %s", unknown_labels)

    logger.info(
        "Predictions loaded for annotated docs: %s mentions",
        len(pred)
    )
    return pred.reset_index(drop=True)


# ============================================================================
# 5. ENTITY-LEVEL MATCHING AND METRICS
# ============================================================================

def safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def exact_span_match(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict match: document + exact character boundaries + entity label."""
    key_cols = ["document_id", "entity_label", "start_char", "end_char"]

    gold_records = gold.to_dict("records")
    pred_records = pred.to_dict("records")

    gold_map: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    pred_map: dict[tuple[Any, ...], list[int]] = defaultdict(list)

    for i, row in enumerate(gold_records):
        key = tuple(row[col] for col in key_cols)
        gold_map[key].append(i)
    for i, row in enumerate(pred_records):
        key = tuple(row[col] for col in key_cols)
        pred_map[key].append(i)

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    for key in set(gold_map) & set(pred_map):
        count = min(len(gold_map[key]), len(pred_map[key]))
        matched_gold.update(gold_map[key][:count])
        matched_pred.update(pred_map[key][:count])

    fp_rows = []
    fn_rows = []
    for i, row in enumerate(pred_records):
        if i not in matched_pred:
            fp_rows.append({**row, "error_type": "FALSE_POSITIVE"})
    for i, row in enumerate(gold_records):
        if i not in matched_gold:
            fn_rows.append({**row, "error_type": "FALSE_NEGATIVE"})

    counts: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in labels
    }
    for i in matched_gold:
        label = normalize_label(gold_records[i]["entity_label"])
        if label in counts:
            counts[label]["tp"] += 1
    for row in fp_rows:
        label = normalize_label(row["entity_label"])
        if label in counts:
            counts[label]["fp"] += 1
    for row in fn_rows:
        label = normalize_label(row["entity_label"])
        if label in counts:
            counts[label]["fn"] += 1

    metrics = metrics_table(counts, labels, "strict_exact")
    errors = pd.DataFrame(fp_rows + fn_rows)
    return metrics, errors, pd.DataFrame({"gold_index": list(matched_gold), "pred_index": list(matched_pred)})


def overlap_match(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relaxed diagnostic: one-to-one match if spans overlap and labels agree."""
    candidates: list[tuple[float, int, int]] = []
    gold_by_doc_label: dict[tuple[str, str], list[int]] = defaultdict(list)
    pred_by_doc_label: dict[tuple[str, str], list[int]] = defaultdict(list)

    for gi, row in gold.iterrows():
        gold_by_doc_label[(str(row.document_id), normalize_label(row.entity_label))].append(gi)
    for pi, row in pred.iterrows():
        pred_by_doc_label[(str(row.document_id), normalize_label(row.entity_label))].append(pi)

    for key in set(gold_by_doc_label) & set(pred_by_doc_label):
        for gi in gold_by_doc_label[key]:
            g = gold.loc[gi]
            for pi in pred_by_doc_label[key]:
                p = pred.loc[pi]
                intersection = max(
                    0,
                    min(int(g.end_char), int(p.end_char))
                    - max(int(g.start_char), int(p.start_char)),
                )
                if intersection <= 0:
                    continue
                union = max(int(g.end_char), int(p.end_char)) - min(
                    int(g.start_char), int(p.start_char)
                )
                iou = intersection / union if union else 0.0
                candidates.append((iou, gi, pi))

    # Greedy maximum-IoU matching prevents one prediction matching multiple gold spans.
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi not in used_gold and pi not in used_pred:
            used_gold.add(gi)
            used_pred.add(pi)
            matched_pairs.append((gi, pi, iou))

    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for gi, _, _ in matched_pairs:
        label = normalize_label(gold.loc[gi, "entity_label"])
        if label in counts:
            counts[label]["tp"] += 1

    for pi, row in pred.iterrows():
        if pi not in used_pred:
            label = normalize_label(row.entity_label)
            if label in counts:
                counts[label]["fp"] += 1

    for gi, row in gold.iterrows():
        if gi not in used_gold:
            label = normalize_label(row.entity_label)
            if label in counts:
                counts[label]["fn"] += 1

    match_details = pd.DataFrame([
        {
            "document_id": str(gold.loc[gi, "document_id"]),
            "gold_entity": gold.loc[gi, "entity_text"],
            "predicted_entity": pred.loc[pi, "entity_text"],
            "entity_label": gold.loc[gi, "entity_label"],
            "gold_start": int(gold.loc[gi, "start_char"]),
            "gold_end": int(gold.loc[gi, "end_char"]),
            "pred_start": int(pred.loc[pi, "start_char"]),
            "pred_end": int(pred.loc[pi, "end_char"]),
            "intersection_over_union": iou,
        }
        for gi, pi, iou in matched_pairs
    ])
    return metrics_table(counts, labels, "relaxed_overlap"), match_details


def metrics_table(
    counts: dict[str, dict[str, int]],
    labels: list[str],
    evaluation_mode: str,
) -> pd.DataFrame:
    rows = []
    total_tp = total_fp = total_fn = 0
    per_label_values = []

    for label in labels:
        tp = int(counts[label]["tp"])
        fp = int(counts[label]["fp"])
        fn = int(counts[label]["fn"])
        support = tp + fn
        predicted_count = tp + fp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # A class absent from the gold sample is not evaluable for recall.
        # Report NaN rather than a misleading 0.0 and exclude it from macro/weighted averages.
        if support == 0:
            precision = 0.0 if predicted_count > 0 else np.nan
            recall = np.nan
            f1 = 0.0 if predicted_count > 0 else np.nan
        else:
            precision, recall, f1 = safe_prf(tp, fp, fn)
            per_label_values.append((precision, recall, f1, support))

        rows.append({
            "evaluation_mode": evaluation_mode,
            "entity_label": label,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "predicted_count": predicted_count,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    micro_p, micro_r, micro_f1 = safe_prf(total_tp, total_fp, total_fn)
    rows.append({
        "evaluation_mode": evaluation_mode,
        "entity_label": "micro_avg",
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "predicted_count": total_tp + total_fp,
        "support": total_tp + total_fn,
        "precision": micro_p,
        "recall": micro_r,
        "f1_score": micro_f1,
    })

    macro_p = float(np.mean([item[0] for item in per_label_values])) if per_label_values else 0.0
    macro_r = float(np.mean([item[1] for item in per_label_values])) if per_label_values else 0.0
    macro_f1 = float(np.mean([item[2] for item in per_label_values])) if per_label_values else 0.0
    rows.append({
        "evaluation_mode": evaluation_mode,
        "entity_label": "macro_avg",
        "true_positive": np.nan,
        "false_positive": np.nan,
        "false_negative": np.nan,
        "predicted_count": np.nan,
        "support": total_tp + total_fn,
        "precision": macro_p,
        "recall": macro_r,
        "f1_score": macro_f1,
    })

    total_support = sum(item[3] for item in per_label_values)
    if total_support:
        weighted_p = sum(item[0] * item[3] for item in per_label_values) / total_support
        weighted_r = sum(item[1] * item[3] for item in per_label_values) / total_support
        weighted_f1 = sum(item[2] * item[3] for item in per_label_values) / total_support
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0
    rows.append({
        "evaluation_mode": evaluation_mode,
        "entity_label": "weighted_avg",
        "true_positive": np.nan,
        "false_positive": np.nan,
        "false_negative": np.nan,
        "predicted_count": np.nan,
        "support": total_support,
        "precision": weighted_p,
        "recall": weighted_r,
        "f1_score": weighted_f1,
    })

    return pd.DataFrame(rows)


# ============================================================================
# 6. ADDITIONAL ARTICLE-LEVEL EVALUATION
# ============================================================================

def article_presence_metrics(
    annotated_docs: pd.DataFrame,
    gold: pd.DataFrame,
    pred: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate whether each article contains >=1 entity of each class."""
    doc_ids = annotated_docs["document_id"].astype(str).tolist()
    rows = []

    for label in ENTITY_LABELS:
        gold_docs = set(gold.loc[gold.entity_label == label, "document_id"].astype(str))
        pred_docs = set(pred.loc[pred.entity_label == label, "document_id"].astype(str))
        tp = len(gold_docs & pred_docs)
        fp = len(pred_docs - gold_docs)
        fn = len(gold_docs - pred_docs)
        tn = len(set(doc_ids) - gold_docs - pred_docs)
        precision, recall, f1 = safe_prf(tp, fp, fn)
        accuracy = (tp + tn) / len(doc_ids) if doc_ids else 0.0
        rows.append({
            "entity_label": label,
            "true_positive_documents": tp,
            "false_positive_documents": fp,
            "false_negative_documents": fn,
            "true_negative_documents": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "n_documents": len(doc_ids),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 7. CONFIDENCE THRESHOLD ANALYSIS
# ============================================================================

def get_default_model_threshold() -> float:
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("threshold", 0.45))
        except Exception:
            pass
    return 0.45


def threshold_sweep(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    model_threshold = get_default_model_threshold()
    thresholds = [t for t in DEFAULT_CONFIDENCE_THRESHOLDS if t >= model_threshold]
    if not thresholds:
        thresholds = [model_threshold]

    rows = []
    gold_counts = gold["entity_label"].value_counts().to_dict()

    for threshold in thresholds:
        filtered_pred = pred[pred["confidence"] >= threshold].copy()
        # Compute strict exact micro metrics at this confidence cut-off.
        _, errors, matched = exact_span_match(gold, filtered_pred, labels)
        tp = len(matched)
        fp = int((errors["error_type"] == "FALSE_POSITIVE").sum()) if not errors.empty else 0
        fn = int((errors["error_type"] == "FALSE_NEGATIVE").sum()) if not errors.empty else 0
        precision, recall, f1 = safe_prf(tp, fp, fn)
        rows.append({
            "confidence_threshold": threshold,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "predicted_count": len(filtered_pred),
            "gold_support": len(gold),
            "precision_micro": precision,
            "recall_micro": recall,
            "f1_micro": f1,
        })
    return pd.DataFrame(rows)


# ============================================================================
# 8. EVALUATION DRIVER
# ============================================================================

def run_evaluation(prediction_file: Path) -> None:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotated_docs, gold = load_gold_annotations()
    annotated_ids = set(annotated_docs["document_id"].astype(str))
    pred = load_predictions(annotated_ids, prediction_file)

    # Restrict evaluation to the manually completed gold subset only.
    gold = gold[gold["document_id"].isin(annotated_ids)].reset_index(drop=True)
    pred = pred[pred["document_id"].isin(annotated_ids)].reset_index(drop=True)

    # Primary metric used by most NER evaluations: exact entity span + label.
    exact_metrics, exact_errors, exact_matches = exact_span_match(
        gold, pred, ENTITY_LABELS
    )

    # Additional relaxed boundary diagnostic. Do not replace strict exact F1 with it.
    overlap_metrics, overlap_matches = overlap_match(
        gold, pred, ENTITY_LABELS
    )

    all_metrics = pd.concat([exact_metrics, overlap_metrics], ignore_index=True)
    presence_metrics = article_presence_metrics(annotated_docs, gold, pred)
    sweep = threshold_sweep(gold, pred, ENTITY_LABELS)

    # Include all evaluated docs in a compact coverage report.
    docs_with_gold = int(gold["document_id"].nunique()) if not gold.empty else 0
    docs_with_pred = int(pred["document_id"].nunique()) if not pred.empty else 0
    docs_with_neither = len(annotated_docs) - len(set(gold.document_id) | set(pred.document_id))
    coverage = pd.DataFrame([{
        "n_annotated_documents": len(annotated_docs),
        "n_documents_with_gold_entities": docs_with_gold,
        "n_documents_with_predicted_entities": docs_with_pred,
        "n_documents_with_neither_gold_nor_prediction": docs_with_neither,
        "n_gold_mentions": len(gold),
        "n_predicted_mentions": len(pred),
        "n_entity_labels_in_schema": len(ENTITY_LABELS),
    }])

    # Save clear separate outputs for reporting and manual error analysis.
    all_metrics.to_csv(EVALUATION_OUTPUT_DIR / "ner_metrics.csv", index=False)
    exact_errors.to_csv(EVALUATION_OUTPUT_DIR / "ner_exact_match_errors.csv", index=False)
    overlap_matches.to_csv(EVALUATION_OUTPUT_DIR / "ner_overlap_matches.csv", index=False)
    presence_metrics.to_csv(EVALUATION_OUTPUT_DIR / "ner_article_presence_metrics.csv", index=False)
    sweep.to_csv(EVALUATION_OUTPUT_DIR / "ner_confidence_threshold_sweep.csv", index=False)
    coverage.to_csv(EVALUATION_OUTPUT_DIR / "ner_evaluation_coverage.csv", index=False)

    # Store run details for reproducibility.
    metadata = {
        "prediction_file": str(prediction_file),
        "gold_documents_file": str(GOLD_DOCUMENTS_FILE),
        "gold_entities_file": str(GOLD_ENTITIES_FILE),
        "n_annotated_documents": int(len(annotated_docs)),
        "n_gold_mentions": int(len(gold)),
        "n_predicted_mentions": int(len(pred)),
        "entity_labels": ENTITY_LABELS,
        "primary_metric": "strict exact entity-level micro F1",
        "matching_definition": "same document_id + exact start_char + exact end_char + exact entity_label",
        "offset_definition": "zero-based character offsets; end_char is exclusive",
        "confidence_threshold_from_ner_metadata": get_default_model_threshold(),
    }
    with open(EVALUATION_OUTPUT_DIR / "evaluation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 78)
    logger.info("NER EVALUATION COMPLETED")
    logger.info("=" * 78)
    logger.info("Annotated documents: %s", len(annotated_docs))
    logger.info("Gold entity mentions: %s", len(gold))
    logger.info("Predicted entity mentions: %s", len(pred))
    logger.info("\nSTRICT EXACT ENTITY-LEVEL RESULTS:\n%s", exact_metrics.to_string(index=False))
    logger.info("\nRELAXED OVERLAP RESULTS (DIAGNOSTIC ONLY):\n%s", overlap_metrics.to_string(index=False))
    logger.info("\nARTICLE PRESENCE RESULTS:\n%s", presence_metrics.to_string(index=False))
    logger.info("\nCONFIDENCE THRESHOLD SWEEP:\n%s", sweep.to_string(index=False))
    logger.info("Evaluation outputs saved to: %s", EVALUATION_OUTPUT_DIR)


# ============================================================================
# 9. CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GLiNER NER outputs using manually annotated gold-standard entities. "
            "Supports gold-template preparation and exact/overlap span evaluation."
        )
    )
    parser.add_argument(
        "--prepare-gold",
        action="store_true",
        help="Create manual annotation templates; does not create gold labels automatically.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Number of source articles in the annotation sample; use 0 for all articles. Default: 250.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible annotation sampling.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation using completed gold annotations.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTION_FILE,
        help=f"Prediction CSV path. Default: {PREDICTION_FILE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare_gold and args.evaluate:
        raise ValueError("Pilih salah satu mode: --prepare-gold atau --evaluate.")

    if args.prepare_gold:
        prepare_gold_templates(args.sample_size, args.random_state)
        return

    # Default behavior: evaluate if --evaluate is provided; require explicit
    # evaluation mode so the user cannot accidentally run before annotating gold.
    if args.evaluate:
        run_evaluation(args.predictions)
        return

    print("Pilih salah satu perintah:")
    print("  1) Buat template anotasi manual:")
    print("     python -u src/ner/evaluate_ner.py --prepare-gold --sample-size 250")
    print("  2) Evaluasi setelah gold selesai dianotasi:")
    print("     python -u src/ner/evaluate_ner.py --evaluate")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("NER evaluation failed: %s", exc)
        sys.exit(1)
