from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# Main article-level corpus used by the existing project.
INPUT_FILE = ROOT_DIR / "data" / "processed" / "news_tfidf_clean.csv"

# Optional richer text source. If it contains text_clean and url,
# it will enrich the same article sample without changing target_date.
OPTIONAL_TEXT_SOURCE = ROOT_DIR / "data" / "processed" / "cnbc_clean.csv"

ALIGNED_FILE = ROOT_DIR / "data" / "processed" / "aligned_daily.csv"

# Results and model cache are kept separately.
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "ner"
MODEL_DIR = ROOT_DIR / "models" / "ner"
MODEL_CACHE_DIR = MODEL_DIR / "hf_cache"

# CNBC articles are English. The small English model is more practical
# for a MacBook than the larger multilingual checkpoint.
MODEL_NAME = "urchade/gliner_small-v2.1"

# Set to "auto", "cpu", or "mps".
DEVICE = "mps"

# Number of chunks processed per GPU batch.
# Start with 4 for MacBook Air.
INFERENCE_BATCH_SIZE = 4

# Lower threshold increases recall but may add false positives.
NER_THRESHOLD = 0.45

# Chunking keeps long articles within a manageable input length.
MAX_WORDS_PER_CHUNK = 150
CHUNK_OVERLAP_WORDS = 25
FLAT_NER = True

# Checkpointing allows the script to resume after an interruption.
RESUME_FROM_CHECKPOINT = True
SAVE_EVERY_N_ARTICLES = 100

# Optional manual text column. Leave None for automatic selection.
TEXT_COLUMN = None
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

# The three requested NER classes.
ENTITY_LABELS = [
    "country",
    "central bank",
    "economic event",
]

RELEVANT_LABELS = set(ENTITY_LABELS)
CONTEXT_WINDOW_CHARS = 140


# ============================================================
# 2. LOGGING AND OUTPUT SCHEMA
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("geopolitical_ner")

MENTION_COLUMNS = [
    "document_id",
    "target_date",
    "url",
    "section",
    "entity_text",
    "normalized_entity",
    "entity_label",
    "confidence",
    "start_char",
    "end_char",
    "chunk_id",
    "context",
    "event_category",
]


# ============================================================
# 3. ENTITY NORMALIZATION DICTIONARIES
# ============================================================

COUNTRY_ALIASES = {
    "us": "United States",
    "u.s": "United States",
    "u.s.": "United States",
    "usa": "United States",
    "u.s.a": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "prc": "China",
    "people's republic of china": "China",
    "peoples republic of china": "China",
    "mainland china": "China",
    "russian federation": "Russia",
    "republic of korea": "South Korea",
    "dprk": "North Korea",
    "uae": "United Arab Emirates",
    "u.a.e": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "islamic republic of iran": "Iran",
    "turkiye": "Turkey",
    "republic of turkey": "Turkey",
    "republic of indonesia": "Indonesia",
    "taiwan": "Taiwan",
}

CENTRAL_BANK_ALIASES = {
    "fed": "Federal Reserve",
    "the fed": "Federal Reserve",
    "federal reserve": "Federal Reserve",
    "federal reserve system": "Federal Reserve",
    "us federal reserve": "Federal Reserve",
    "u.s. federal reserve": "Federal Reserve",
    "ecb": "European Central Bank",
    "european central bank": "European Central Bank",
    "pboc": "People's Bank of China",
    "people's bank of china": "People's Bank of China",
    "peoples bank of china": "People's Bank of China",
    "boj": "Bank of Japan",
    "bank of japan": "Bank of Japan",
    "boe": "Bank of England",
    "bank of england": "Bank of England",
    "snb": "Swiss National Bank",
    "swiss national bank": "Swiss National Bank",
    "rba": "Reserve Bank of Australia",
    "reserve bank of australia": "Reserve Bank of Australia",
    "rbi": "Reserve Bank of India",
    "reserve bank of india": "Reserve Bank of India",
    "bank indonesia": "Bank Indonesia",
    "bangko sentral ng pilipinas": "Bangko Sentral ng Pilipinas",
    "central bank of brazil": "Central Bank of Brazil",
    "bank of canada": "Bank of Canada",
    "bank of korea": "Bank of Korea",
    "bank negara malaysia": "Bank Negara Malaysia",
    "monetary authority of singapore": "Monetary Authority of Singapore",
}

# Event phrase patterns are used to subcategorize GLiNER-detected event spans.
# They do not replace the model's event-span recognition.
EVENT_PATTERNS = {
    "MONETARY_POLICY": [
        r"\binterest[- ]rate (?:hike|hikes|increase|increases|rise|rises|cut|cuts|reduction|reductions)\b",
        r"\brate (?:hike|hikes|increase|increases|cut|cuts|decision|decisions)\b",
        r"\bmonetary (?:policy|tightening|easing)\b",
        r"\bquantitative (?:easing|tightening)\b",
        r"\bpolicy rate\b",
        r"\brate decision\b",
        r"\brate-setting\b",
    ],
    "INFLATION": [
        r"\binflation\b",
        r"\bdeflation\b",
        r"\bdisinflation\b",
        r"\bconsumer prices?\b",
        r"\bprice pressures?\b",
        r"\bcpi\b",
        r"\bpce\b",
    ],
    "SANCTIONS": [
        r"\bsanctions?\b",
        r"\bembargo(?:es)?\b",
        r"\basset freeze\b",
        r"\bfinancial restrictions\b",
    ],
    "TRADE_POLICY": [
        r"\btariffs?\b",
        r"\btrade war\b",
        r"\btrade restrictions?\b",
        r"\bexport controls?\b",
        r"\bexport restrictions?\b",
        r"\bimport restrictions?\b",
        r"\btrade barriers?\b",
    ],
    "CURRENCY_FX": [
        r"\bcurrency (?:crisis|collapse|devaluation|depreciation|plunge)\b",
        r"\bexchange[- ]rate crisis\b",
        r"\bforeign[- ]exchange crisis\b",
        r"\bdevalu(?:e|ation)\b",
        r"\bdepreciat(?:e|ion)\b",
    ],
    "SOVEREIGN_DEBT": [
        r"\bsovereign debt\b",
        r"\bdebt default\b",
        r"\bsovereign default\b",
        r"\bdebt restructuring\b",
        r"\bdebt crisis\b",
    ],
    "FINANCIAL_STABILITY": [
        r"\bfinancial crisis\b",
        r"\bbanking crisis\b",
        r"\bbank failure\b",
        r"\bbank collapse\b",
        r"\bbank bailout\b",
        r"\bliquidity crisis\b",
        r"\bcapital shortfall\b",
    ],
    "RECESSION_GROWTH": [
        r"\brecession\b",
        r"\beconomic contraction\b",
        r"\beconomic slowdown\b",
        r"\bgdp contraction\b",
        r"\bgrowth slowdown\b",
    ],
    "ENERGY_COMMODITY": [
        r"\boil supply\b",
        r"\boil production\b",
        r"\boil embargo\b",
        r"\benergy crisis\b",
        r"\bnatural gas\b",
        r"\benergy prices?\b",
        r"\bopec\b",
    ],
    "FISCAL_POLICY": [
        r"\bfiscal policy\b",
        r"\bbudget deficit\b",
        r"\bbudget surplus\b",
        r"\bgovernment spending\b",
        r"\bpublic spending\b",
        r"\btax increase\b",
        r"\btax cut\b",
    ],
}

EVENT_CANONICAL_NAMES = {
    "MONETARY_POLICY": "Monetary policy",
    "INFLATION": "Inflation and prices",
    "SANCTIONS": "Economic sanctions",
    "TRADE_POLICY": "Trade restrictions and tariffs",
    "CURRENCY_FX": "Currency and exchange-rate event",
    "SOVEREIGN_DEBT": "Sovereign debt event",
    "FINANCIAL_STABILITY": "Financial stability event",
    "RECESSION_GROWTH": "Economic growth and recession",
    "ENERGY_COMMODITY": "Energy and commodity event",
    "FISCAL_POLICY": "Fiscal policy",
    "OTHER_ECONOMIC_EVENT": "Other economic event",
}

COUNTRIES_OF_INTEREST = [
    "United States",
    "China",
    "Russia",
    "Iran",
    "Israel",
    "Ukraine",
    "Taiwan",
    "Japan",
    "South Korea",
    "United Kingdom",
    "Germany",
    "Indonesia",
    "Saudi Arabia",
    "Turkey",
    "India",
]

CENTRAL_BANKS_OF_INTEREST = [
    "Federal Reserve",
    "European Central Bank",
    "People's Bank of China",
    "Bank of Japan",
    "Bank of England",
    "Swiss National Bank",
    "Reserve Bank of Australia",
    "Reserve Bank of India",
    "Bank Indonesia",
    "Bank of Canada",
]

COMPILED_EVENT_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in EVENT_PATTERNS.items()
}


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def create_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_key(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.,]+$", "", value)
    return value


def slugify(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def choose_device() -> str:
    import torch

    requested = DEVICE.lower().strip()

    cuda_available = torch.cuda.is_available()

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = (
        mps_backend is not None
        and mps_backend.is_available()
    )

    if requested == "auto":
        if cuda_available:
            return "cuda"

        if mps_available:
            return "mps"

        logger.warning(
            "No GPU detected. Falling back to CPU."
        )
        return "cpu"

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA GPU tidak tersedia. "
                "Periksa instalasi PyTorch dan NVIDIA GPU."
            )

        return "cuda"

    if requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "Apple MPS GPU tidak tersedia. "
                "Periksa dukungan Apple Silicon dan PyTorch."
            )

        return "mps"

    if requested == "cpu":
        return "cpu"

    raise ValueError(
        f"DEVICE tidak valid: {DEVICE}. "
        "Gunakan 'auto', 'mps', 'cuda', atau 'cpu'."
    )


def get_stable_document_id(row: pd.Series, row_number: int) -> str:
    url = str(row.get("url", "")).strip()
    if url and url.lower() != "nan":
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return f"news_{digest}"
    return f"news_row_{row_number:07d}"


def choose_text_column(df: pd.DataFrame) -> str:
    if TEXT_COLUMN:
        if TEXT_COLUMN not in df.columns:
            raise ValueError(f"TEXT_COLUMN '{TEXT_COLUMN}' tidak ditemukan. Kolom: {list(df.columns)}")
        return TEXT_COLUMN

    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Tidak menemukan kolom teks. Kolom yang dicari: "
        + ", ".join(TEXT_COLUMN_CANDIDATES)
    )


# ============================================================
# 5. LOAD AND PREPARE ARTICLE DATA
# ============================================================

def load_articles() -> tuple[pd.DataFrame, str]:
    logger.info("Loading input: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "target_date" not in df.columns:
        raise ValueError(
            "Kolom 'target_date' tidak ditemukan. Untuk eksperimen USD/IDR, "
            "gunakan dataset yang target_date-nya sudah ditetapkan oleh preprocessing "
            "dan cutoff fixing, bukan membuat tanggal baru dari tanggal publikasi secara otomatis."
        )

    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["target_date"]).copy()

    # Enrich the same filtered article sample with original/clean text if possible.
    if OPTIONAL_TEXT_SOURCE.exists() and "url" in df.columns:
        try:
            clean_df = pd.read_csv(OPTIONAL_TEXT_SOURCE)
            if "url" in clean_df.columns:
                candidate_columns = [c for c in ["url", "text_clean", "body_clean"] if c in clean_df.columns]
                clean_df = clean_df[candidate_columns].drop_duplicates(subset=["url"], keep="first")
                rename_map = {}
                if "text_clean" in clean_df.columns:
                    rename_map["text_clean"] = "text_clean_original"
                elif "body_clean" in clean_df.columns:
                    rename_map["body_clean"] = "text_clean_original"
                clean_df = clean_df.rename(columns=rename_map)
                if "text_clean_original" in clean_df.columns:
                    df = df.merge(clean_df[["url", "text_clean_original"]], on="url", how="left")
                    logger.info(
                        "Enriched original clean text for %s/%s articles",
                        int(df["text_clean_original"].notna().sum()),
                        len(df),
                    )
        except Exception as exc:
            logger.warning("Optional clean-text enrichment skipped: %s", exc)

    # If only some URLs were enriched from cnbc_clean.csv, fall back to
    # the text_norm column instead of dropping those articles.
    if "text_clean_original" in df.columns and "text_norm" in df.columns:
        df["text_clean_original"] = df["text_clean_original"].where(
            df["text_clean_original"].notna()
            & df["text_clean_original"].astype(str).str.strip().ne(""),
            df["text_norm"],
        )

    text_column = choose_text_column(df)
    df["ner_text"] = df[text_column].fillna("").astype(str).str.strip()
    df = df[df["ner_text"].str.len() > 0].copy()

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")

    df = df.sort_values(["target_date"] + (["url"] if "url" in df.columns else [])).reset_index(drop=True)
    df["document_id"] = [get_stable_document_id(row, i) for i, (_, row) in enumerate(df.iterrows())]

    logger.info("Selected text column: %s", text_column)
    logger.info("Articles: %s", len(df))
    logger.info("Date range: %s to %s", df["target_date"].min(), df["target_date"].max())
    logger.info("Mean characters/article: %.1f", df["ner_text"].str.len().mean())

    return df, text_column


# ============================================================
# 6. MODEL LOADING
# ============================================================

def load_gliner_model():
    import torch
    from gliner import GLiNER

    device = choose_device()

    logger.info("=" * 65)
    logger.info("LOADING GLiNER MODEL")
    logger.info("=" * 65)

    logger.info("Model: %s", MODEL_NAME)
    logger.info("Selected device: %s", device)
    logger.info("Model cache: %s", MODEL_CACHE_DIR)

    model = GLiNER.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    # Move the complete GLiNER model to the selected device.
    model = model.to(device)

    # Set inference mode.
    model.eval()

    # Verify the actual model device.
    try:
        actual_device = str(model.device)
    except (AttributeError, StopIteration):
        actual_device = str(
            next(model.parameters()).device
        )

    logger.info("Actual model device: %s", actual_device)

    if not actual_device.startswith(device):
        raise RuntimeError(
            f"Model gagal dipindahkan ke {device}. "
            f"Device aktual: {actual_device}"
        )

    # Test a small tensor operation on the selected device.
    test_tensor = torch.tensor(
        [1.0, 2.0],
        device=device
    )

    _ = test_tensor * 2

    if device == "mps":
        torch.mps.synchronize()

    elif device == "cuda":
        torch.cuda.synchronize()

    del test_tensor

    logger.info("GPU verification successful.")
    logger.info("GLiNER is ready for inference on %s.", device)
    logger.info("=" * 65)

    return model, device


# ============================================================
# 7. CHUNKING AND CONTEXT
# ============================================================

def split_text_into_chunks(
    text: str,
    max_words: int = MAX_WORDS_PER_CHUNK,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    word_matches = list(re.finditer(r"\S+", text))
    if not word_matches:
        return []

    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("MAX_WORDS_PER_CHUNK harus lebih besar daripada CHUNK_OVERLAP_WORDS.")

    chunks = []
    for start_index in range(0, len(word_matches), step):
        end_index = min(start_index + max_words, len(word_matches))
        char_start = word_matches[start_index].start()
        char_end = word_matches[end_index - 1].end()
        chunks.append({
            "chunk_id": len(chunks),
            "text": text[char_start:char_end],
            "char_start": char_start,
            "char_end": char_end,
        })
        if end_index >= len(word_matches):
            break

    return chunks


def get_context(text: str, start: int, end: int, window: int = CONTEXT_WINDOW_CHARS) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


# ============================================================
# 8. ENTITY NORMALIZATION AND EVENT SUBCATEGORIZATION
# ============================================================

# Lookup tanpa peka huruf besar/kecil untuk nama kanonik negara.
# GLiNER berjalan di text_norm (huruf kecil), jadi span seperti "ukraine"
# harus dipetakan ke "Ukraine" supaya masuk kolom country_*_mentions.
# Perbaikan minimal: TIDAK menambah alias baru (demonim / ibu kota tetap tidak cocok).
COUNTRY_CANONICAL_CASEFOLD = {
    name.strip().casefold(): name for name in COUNTRIES_OF_INTEREST
}


def normalize_country(entity_text: str) -> str:
    key = normalize_key(entity_text)
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    canonical = COUNTRY_CANONICAL_CASEFOLD.get(entity_text.strip().casefold())
    if canonical is not None:
        return canonical
    return entity_text.strip()


def normalize_central_bank(entity_text: str) -> str:
    key = normalize_key(entity_text)
    if key in CENTRAL_BANK_ALIASES:
        return CENTRAL_BANK_ALIASES[key]
    return entity_text.strip()


def classify_economic_event(entity_text: str, context: str) -> list[str]:
    search_text = f"{entity_text} {context}".lower()
    found = []

    for category, patterns in COMPILED_EVENT_PATTERNS.items():
        if any(pattern.search(search_text) for pattern in patterns):
            found.append(category)

    return found if found else ["OTHER_ECONOMIC_EVENT"]


def normalize_event(entity_text: str, event_categories: list[str]) -> str:
    if not event_categories:
        return entity_text.strip()
    # If a phrase matches multiple categories, preserve all canonical categories.
    names = [EVENT_CANONICAL_NAMES.get(cat, cat) for cat in event_categories]
    return " | ".join(names)


def normalize_entity(entity_text: str, label: str, context: str) -> tuple[str, list[str]]:
    label = label.lower().strip()

    if label == "country":
        return normalize_country(entity_text), []

    if label == "central bank":
        return normalize_central_bank(entity_text), []

    if label == "economic event":
        event_categories = classify_economic_event(entity_text, context)
        return normalize_event(entity_text, event_categories), event_categories

    return entity_text.strip(), []


# ============================================================
# 9. ARTICLE-LEVEL ENTITY EXTRACTION
# ============================================================

def extract_entities_for_article(
    model,
    row: pd.Series
) -> list[dict[str, Any]]:

    text = row["ner_text"]
    document_id = row["document_id"]
    target_date = row["target_date"]

    url = row.get("url", "")
    section = row.get("section", "")

    mentions = []

    # Split article into overlapping chunks.
    chunks = split_text_into_chunks(text)

    if not chunks:
        return mentions

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # ---------------------------------------------------------
    # BATCHED GPU INFERENCE
    # ---------------------------------------------------------

    predictions_by_chunk = model.inference(
        chunk_texts,
        ENTITY_LABELS,
        flat_ner=FLAT_NER,
        threshold=NER_THRESHOLD,
        batch_size=INFERENCE_BATCH_SIZE,
    )

    if len(predictions_by_chunk) != len(chunks):
        raise RuntimeError(
            f"Prediction count mismatch for {document_id}: "
            f"{len(predictions_by_chunk)} predictions "
            f"for {len(chunks)} chunks."
        )

    # ---------------------------------------------------------
    # PROCESS ENTITY PREDICTIONS
    # ---------------------------------------------------------

    for chunk, predictions in zip(chunks, predictions_by_chunk):

        for prediction in predictions:

            label = str(
                prediction.get("label", "")
            ).lower().strip()

            entity_text = str(
                prediction.get("text", "")
            ).strip()

            if label not in RELEVANT_LABELS or not entity_text:
                continue

            local_start = prediction.get("start")
            local_end = prediction.get("end")

            if local_start is None or local_end is None:
                continue

            try:
                local_start = int(local_start)
                local_end = int(local_end)

            except (TypeError, ValueError):
                continue

            if (
                local_start < 0
                or local_end <= local_start
                or local_end > len(chunk["text"])
            ):
                continue

            # Convert chunk offsets into original article offsets.
            global_start = (
                chunk["char_start"] + local_start
            )

            global_end = (
                chunk["char_start"] + local_end
            )

            context = get_context(
                text,
                global_start,
                global_end
            )

            normalized_entity, event_categories = normalize_entity(
                entity_text,
                label,
                context,
            )

            try:
                confidence = float(
                    prediction.get("score", 0.0)
                )

            except (TypeError, ValueError):
                confidence = 0.0

            mentions.append({
                "document_id": document_id,
                "target_date": target_date,
                "url": url,
                "section": section,
                "entity_text": entity_text,
                "normalized_entity": normalized_entity,
                "entity_label": label,
                "confidence": confidence,
                "start_char": global_start,
                "end_char": global_end,
                "chunk_id": chunk["chunk_id"],
                "context": context,
                "event_category": (
                    "|".join(event_categories)
                    if event_categories
                    else ""
                ),
            })

    return mentions


def deduplicate_mentions(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return pd.DataFrame(columns=MENTION_COLUMNS)

    mentions_df = mentions_df.copy()
    mentions_df["confidence"] = pd.to_numeric(mentions_df["confidence"], errors="coerce").fillna(0.0)
    mentions_df = mentions_df.sort_values("confidence", ascending=False)

    # Overlapping chunks can produce the same span more than once.
    mentions_df = mentions_df.drop_duplicates(
        subset=["document_id", "entity_label", "start_char", "end_char", "normalized_entity"],
        keep="first",
    )

    # Deduplicate repeated exact mentions with identical context.
    mentions_df = mentions_df.drop_duplicates(
        subset=["document_id", "entity_label", "normalized_entity", "context"],
        keep="first",
    )

    return mentions_df.reindex(columns=MENTION_COLUMNS).reset_index(drop=True)


# ============================================================
# 10. CHECKPOINTING / RESUME
# ============================================================

CHECKPOINT_FILE = OUTPUT_DIR / "entity_mentions_checkpoint.csv"
PROGRESS_FILE = OUTPUT_DIR / "ner_progress.json"
ERRORS_FILE = OUTPUT_DIR / "ner_errors.csv"


def make_run_signature(df: pd.DataFrame, text_column: str) -> dict[str, Any]:
    stat = INPUT_FILE.stat()
    return {
        "model_name": MODEL_NAME,
        "input_file": str(INPUT_FILE),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "n_articles": len(df),
        "text_column": text_column,
        "threshold": NER_THRESHOLD,
        "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "entity_labels": ENTITY_LABELS,
    }


def load_checkpoint(signature: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    if not RESUME_FROM_CHECKPOINT or not PROGRESS_FILE.exists() or not CHECKPOINT_FILE.exists():
        return [], set()

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)

        if state.get("signature") != signature:
            logger.warning("Checkpoint signature differs from current input/config. Starting a fresh run.")
            return [], set()

        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        if checkpoint_df.empty:
            mentions = []
        else:
            mentions = checkpoint_df.to_dict(orient="records")

        processed_ids = set(state.get("processed_document_ids", []))
        logger.info("Resuming: %s articles already processed, %s saved entity mentions", len(processed_ids), len(mentions))
        return mentions, processed_ids

    except Exception as exc:
        logger.warning("Could not load checkpoint; starting fresh: %s", exc)
        return [], set()


def save_checkpoint(
    mentions: list[dict[str, Any]],
    processed_ids: set[str],
    signature: dict[str, Any],
) -> None:
    mention_df = pd.DataFrame(mentions, columns=MENTION_COLUMNS)
    mention_df.to_csv(CHECKPOINT_FILE, index=False)

    state = {
        "signature": signature,
        "processed_document_ids": sorted(processed_ids),
        "processed_count": len(processed_ids),
        "entity_mention_count": len(mentions),
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)


def run_ner(model, df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    signature = make_run_signature(df, text_column)
    mention_records, processed_ids = load_checkpoint(signature)
    errors = []

    new_successes = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="NER articles"):
        document_id = row["document_id"]
        if document_id in processed_ids:
            continue

        try:
            article_mentions = extract_entities_for_article(model, row)
            mention_records.extend(article_mentions)
            processed_ids.add(document_id)
            new_successes += 1

        except Exception as exc:
            logger.exception("NER failed for document %s", document_id)
            errors.append({
                "document_id": document_id,
                "url": row.get("url", ""),
                "error": str(exc),
            })

        if new_successes > 0 and new_successes % SAVE_EVERY_N_ARTICLES == 0:
            save_checkpoint(mention_records, processed_ids, signature)
            logger.info("Checkpoint saved: %s/%s articles", len(processed_ids), len(df))

    save_checkpoint(mention_records, processed_ids, signature)

    mentions_df = deduplicate_mentions(pd.DataFrame(mention_records, columns=MENTION_COLUMNS))
    if not mentions_df.empty:
        mentions_df["target_date"] = pd.to_datetime(mentions_df["target_date"], errors="coerce").dt.normalize()
    errors_df = pd.DataFrame(errors, columns=["document_id", "url", "error"])

    return mentions_df, errors_df


# ============================================================
# 11. ARTICLE-LEVEL FEATURES
# ============================================================

def make_article_features(df: pd.DataFrame, mentions_df: pd.DataFrame) -> pd.DataFrame:
    features = df[["document_id", "target_date"]].copy()

    if "url" in df.columns:
        features["url"] = df["url"].values

    if mentions_df.empty:
        features["total_entity_mentions"] = 0
        features["n_unique_countries"] = 0
        features["n_unique_central_banks"] = 0
        features["n_unique_economic_events"] = 0
        features["country_entities"] = ""
        features["central_bank_entities"] = ""
        features["economic_event_entities"] = ""
        features["has_country"] = 0
        features["has_central_bank"] = 0
        features["has_economic_event"] = 0
        for country in COUNTRIES_OF_INTEREST:
            features[f"country_{slugify(country)}_mentions"] = 0
        for bank in CENTRAL_BANKS_OF_INTEREST:
            features[f"central_bank_{slugify(bank)}_mentions"] = 0
        for category in list(EVENT_PATTERNS) + ["OTHER_ECONOMIC_EVENT"]:
            features[f"event_{category.lower()}_count"] = 0
        return features

    grouped = {doc_id: group for doc_id, group in mentions_df.groupby("document_id")}
    rows = []

    for document_id in features["document_id"]:
        group = grouped.get(document_id)
        if group is None:
            rows.append({
                "document_id": document_id,
                "total_entity_mentions": 0,
                "n_unique_countries": 0,
                "n_unique_central_banks": 0,
                "n_unique_economic_events": 0,
                "country_entities": "",
                "central_bank_entities": "",
                "economic_event_entities": "",
                "has_country": 0,
                "has_central_bank": 0,
                "has_economic_event": 0,
            })
            continue

        def unique_values(label: str) -> list[str]:
            return sorted(group.loc[group["entity_label"] == label, "normalized_entity"].dropna().astype(str).unique())

        countries = unique_values("country")
        banks = unique_values("central bank")
        events = unique_values("economic event")

        rows.append({
            "document_id": document_id,
            "total_entity_mentions": len(group),
            "n_unique_countries": len(countries),
            "n_unique_central_banks": len(banks),
            "n_unique_economic_events": len(events),
            "country_entities": " | ".join(countries),
            "central_bank_entities": " | ".join(banks),
            "economic_event_entities": " | ".join(events),
            "has_country": int(bool(countries)),
            "has_central_bank": int(bool(banks)),
            "has_economic_event": int(bool(events)),
        })

    features = features.merge(pd.DataFrame(rows), on="document_id", how="left")

    # Counts for a selected set of geopolitically relevant countries.
    country_mentions = mentions_df[mentions_df["entity_label"] == "country"]
    if not country_mentions.empty:
        counts = pd.crosstab(country_mentions["document_id"], country_mentions["normalized_entity"])
    else:
        counts = pd.DataFrame()

    for country in COUNTRIES_OF_INTEREST:
        column = f"country_{slugify(country)}_mentions"
        if country in counts.columns:
            features[column] = features["document_id"].map(counts[country]).fillna(0).astype(int)
        else:
            features[column] = 0

    # Counts for selected central banks.
    bank_mentions = mentions_df[mentions_df["entity_label"] == "central bank"]
    if not bank_mentions.empty:
        bank_counts = pd.crosstab(bank_mentions["document_id"], bank_mentions["normalized_entity"])
    else:
        bank_counts = pd.DataFrame()

    for bank in CENTRAL_BANKS_OF_INTEREST:
        column = f"central_bank_{slugify(bank)}_mentions"
        if bank in bank_counts.columns:
            features[column] = features["document_id"].map(bank_counts[bank]).fillna(0).astype(int)
        else:
            features[column] = 0

    # Number of detected economic-event mentions by rule-based subcategory.
    event_mentions = mentions_df[mentions_df["entity_label"] == "economic event"]
    for category in EVENT_PATTERNS:
        column = f"event_{category.lower()}_count"
        if event_mentions.empty:
            features[column] = 0
            continue
        selected = event_mentions[
            event_mentions["event_category"].fillna("").str.split("|").apply(lambda values: category in values)
        ]
        event_counts = selected.groupby("document_id").size()
        features[column] = features["document_id"].map(event_counts).fillna(0).astype(int)

    other_events = event_mentions[
        event_mentions["event_category"].fillna("").str.contains("OTHER_ECONOMIC_EVENT", regex=False)
    ]
    other_counts = other_events.groupby("document_id").size() if not other_events.empty else pd.Series(dtype=int)
    features["event_other_economic_event_count"] = features["document_id"].map(other_counts).fillna(0).astype(int)

    return features


# ============================================================
# 12. FILTER ARTICLES WITH RELEVANT ENTITIES
# ============================================================

def make_filtered_articles(df: pd.DataFrame, article_features: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        article_features.drop(columns=["url"], errors="ignore"),
        on=["document_id", "target_date"],
        how="left",
    )

    condition = (
        (merged["has_country"].fillna(0) > 0)
        | (merged["has_central_bank"].fillna(0) > 0)
        | (merged["has_economic_event"].fillna(0) > 0)
    )
    filtered = merged.loc[condition].copy()
    filtered["is_ner_relevant"] = 1
    return filtered


# ============================================================
# 13. DAILY FEATURE AGGREGATION
# ============================================================

def make_daily_features(
    df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    article_features: pd.DataFrame,
) -> pd.DataFrame:
    # Sum article-level numeric features per target_date.
    id_columns = {"document_id", "target_date", "url", "country_entities", "central_bank_entities", "economic_event_entities"}
    numeric_columns = [
        col for col in article_features.columns
        if col not in id_columns and pd.api.types.is_numeric_dtype(article_features[col])
    ]

    daily = (
        article_features.groupby("target_date")[numeric_columns]
        .sum()
        .reset_index()
        .rename(columns={"target_date": "date"})
    )

    article_stats = article_features.groupby("target_date").agg(
        n_articles_with_ner=("total_entity_mentions", lambda values: int((values > 0).sum())),
        n_articles_with_country=("has_country", "sum"),
        n_articles_with_central_bank=("has_central_bank", "sum"),
        n_articles_with_economic_event=("has_economic_event", "sum"),
        mean_entities_per_article=("total_entity_mentions", "mean"),
    ).reset_index().rename(columns={"target_date": "date"})

    daily = daily.merge(article_stats, on="date", how="outer")

    if not mentions_df.empty:
        # Distinct entity count per date, not the sum of article-level unique counts.
        for label, output_column in [
            ("country", "daily_n_unique_countries"),
            ("central bank", "daily_n_unique_central_banks"),
            ("economic event", "daily_n_unique_economic_event_types"),
        ]:
            subset = mentions_df[mentions_df["entity_label"] == label]
            if label == "economic event":
                # Count distinct event categories for the day.
                pairs = []
                for date, group in subset.groupby("target_date"):
                    categories = set()
                    for value in group["event_category"].fillna(""):
                        categories.update(item for item in str(value).split("|") if item)
                    pairs.append({"date": date, output_column: len(categories)})
                distinct = pd.DataFrame(pairs, columns=["date", output_column])
            else:
                distinct = (
                    subset.groupby("target_date")["normalized_entity"]
                    .nunique()
                    .reset_index(name=output_column)
                    .rename(columns={"target_date": "date"})
                )
            daily = daily.merge(distinct, on="date", how="left")

        # Daily counts by detected label.
        label_daily = pd.crosstab(mentions_df["target_date"], mentions_df["entity_label"]).reset_index()
        label_daily = label_daily.rename(columns={"target_date": "date"})
        label_daily.columns = [
            "date" if column == "date" else f"daily_{slugify(column)}_mentions"
            for column in label_daily.columns
        ]
        daily = daily.merge(label_daily, on="date", how="left")

    # Preserve every date in aligned JISDOR, including days without news.
    if ALIGNED_FILE.exists():
        aligned_dates = pd.read_csv(ALIGNED_FILE, usecols=["date"])
        aligned_dates["date"] = pd.to_datetime(aligned_dates["date"], errors="coerce").dt.normalize()
        all_dates = aligned_dates["date"].dropna().drop_duplicates().sort_values()
        daily = pd.DataFrame({"date": all_dates}).merge(daily, on="date", how="left")
    else:
        logger.warning("aligned_daily.csv not found. Daily features contain only dates with news.")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.sort_values("date").reset_index(drop=True)

    numeric = daily.select_dtypes(include=[np.number]).columns
    daily[numeric] = daily[numeric].fillna(0)

    return daily


# ============================================================
# 14. ENTITY FREQUENCY AND CONFIG OUTPUTS
# ============================================================

def make_entity_frequency(mentions_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["entity_label", "normalized_entity", "mention_count", "article_count", "mean_confidence"]
    if mentions_df.empty:
        return pd.DataFrame(columns=columns)

    result = (
        mentions_df.groupby(["entity_label", "normalized_entity"])
        .agg(
            mention_count=("document_id", "size"),
            article_count=("document_id", "nunique"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["mention_count", "article_count"], ascending=False)
    )
    return result


def save_configuration(text_column: str, device: str, df: pd.DataFrame) -> None:
    aliases = {
        "country_aliases": COUNTRY_ALIASES,
        "central_bank_aliases": CENTRAL_BANK_ALIASES,
        "event_patterns": EVENT_PATTERNS,
        "event_canonical_names": EVENT_CANONICAL_NAMES,
    }
    with open(MODEL_DIR / "entity_normalization_rules.json", "w", encoding="utf-8") as file:
        json.dump(aliases, file, indent=2, ensure_ascii=False)

    metadata = {
        "model_name": MODEL_NAME,
        "device": device,
        "input_file": str(INPUT_FILE),
        "optional_text_source": str(OPTIONAL_TEXT_SOURCE) if OPTIONAL_TEXT_SOURCE.exists() else None,
        "selected_text_column": text_column,
        "n_articles": int(len(df)),
        "date_start": str(df["target_date"].min()),
        "date_end": str(df["target_date"].max()),
        "entity_labels": ENTITY_LABELS,
        "threshold": NER_THRESHOLD,
        "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "event_categories": list(EVENT_PATTERNS.keys()) + ["OTHER_ECONOMIC_EVENT"],
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

def save_results(
    mentions_df: pd.DataFrame,
    article_features: pd.DataFrame,
    filtered_articles: pd.DataFrame,
    daily_features: pd.DataFrame,
    entity_frequency: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    outputs = {
        "entity_mentions.csv": mentions_df,
        "article_entity_features.csv": article_features,
        "filtered_articles.csv": filtered_articles,
        "daily_entity_features.csv": daily_features,
        "entity_frequency.csv": entity_frequency,
        "ner_errors.csv": errors_df,
    }

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        logger.info("Saved: %s | shape=%s", path, frame.shape)

    # Remove checkpoints only when every article completed without an error.
    # If some articles failed, keep checkpoints so a later run can retry them.
    if errors_df.empty:
        for path in [CHECKPOINT_FILE, PROGRESS_FILE]:
            if path.exists():
                path.unlink()
    else:
        logger.warning(
            "Some articles failed. Checkpoint files are retained so failed articles can be retried."
        )


# ============================================================
# 16. MAIN
# ============================================================

def main() -> None:
    create_directories()

    logger.info("=" * 75)
    logger.info("CNBC GEOPOLITICAL NEWS - NAMED ENTITY RECOGNITION")
    logger.info("=" * 75)

    df, text_column = load_articles()
    model, device = load_gliner_model()
    save_configuration(text_column, device, df)

    mentions_df, errors_df = run_ner(model, df, text_column)

    logger.info("Detected entity mentions: %s", len(mentions_df))
    if not mentions_df.empty:
        logger.info("Entity label distribution:\n%s", mentions_df["entity_label"].value_counts().to_string())

    article_features = make_article_features(df, mentions_df)
    filtered_articles = make_filtered_articles(df, article_features)
    daily_features = make_daily_features(df, mentions_df, article_features)
    entity_frequency = make_entity_frequency(mentions_df)

    save_results(
        mentions_df=mentions_df,
        article_features=article_features,
        filtered_articles=filtered_articles,
        daily_features=daily_features,
        entity_frequency=entity_frequency,
        errors_df=errors_df,
    )

    logger.info("=" * 75)
    logger.info("NER PIPELINE COMPLETED")
    logger.info("Articles processed: %s", len(df))
    logger.info("Articles containing relevant entities: %s", int((article_features["total_entity_mentions"] > 0).sum()))
    logger.info("Filtered articles: %s", len(filtered_articles))
    logger.info("Entity mentions: %s", len(mentions_df))
    logger.info("Failed articles: %s", len(errors_df))
    logger.info("Daily feature rows: %s", len(daily_features))
    logger.info("CSV outputs: %s", OUTPUT_DIR)
    logger.info("Model cache/rules: %s", MODEL_DIR)

    if not entity_frequency.empty:
        logger.info("Top 20 detected entities:\n%s", entity_frequency.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
