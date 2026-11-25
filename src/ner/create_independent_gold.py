from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NER_DIR = ROOT / "data" / "processed" / "ner"
DOCUMENTS = NER_DIR / "gold_documents.csv"
ENTITIES = NER_DIR / "gold_entities.csv"
PROVENANCE = NER_DIR / "gold_annotation_provenance.json"


COUNTRIES = (
    "united states", "u.s.", "u.s", "america", "china", "russia", "ukraine",
    "indonesia", "japan", "south korea", "north korea", "india", "taiwan",
    "iran", "iraq", "israel", "palestine", "gaza", "saudi arabia", "qatar",
    "united arab emirates", "uae", "turkey", "syria", "yemen", "lebanon",
    "jordan", "egypt", "afghanistan", "pakistan", "vietnam", "thailand",
    "singapore", "malaysia", "philippines", "australia", "new zealand",
    "united kingdom", "u.k.", "u.k", "britain", "france", "germany", "italy",
    "spain", "portugal", "netherlands", "belgium", "switzerland", "sweden",
    "norway", "finland", "poland", "belarus", "czech republic", "romania",
    "canada", "mexico", "brazil", "argentina", "chile", "colombia", "peru",
    "venezuela", "cuba", "south africa", "nigeria", "ethiopia", "kenya",
    "sudan", "libya", "morocco", "algeria", "european union", "e.u.", "eu",
)

CENTRAL_BANKS = (
    "federal reserve", "the fed", "fed", "european central bank", "ecb",
    "bank of england", "boe", "bank of japan", "boj", "bank indonesia",
    "people's bank of china", "people s bank of china", "pboc", "reserve bank of australia",
    "rba", "bank of canada", "bank of korea", "reserve bank of india",
    "swiss national bank", "bank of mexico", "central bank of russia",
)

ECONOMIC_EVENTS = (
    "interest rate hike", "interest rate cut", "interest rates", "rate hike",
    "rate cut", "monetary tightening", "monetary easing", "inflation",
    "deflation", "recession", "economic sanctions", "sanctions", "trade war",
    "tariffs", "tariff", "currency devaluation", "devaluation", "banking crisis",
    "debt crisis", "sovereign debt crisis", "default", "oil embargo", "oil shock",
    "supply disruption", "supply disruptions", "supply shortage", "energy crisis",
    "gas crisis", "price surge", "rising prices", "higher prices", "market crash",
    "financial crisis", "economic slowdown", "economic contraction", "unemployment",
)


def pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)


def extract(text: str, terms: tuple[str, ...], label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in sorted(terms, key=len, reverse=True):
        for match in pattern(term).finditer(text):
            rows.append({
                "entity_text": text[match.start():match.end()],
                "entity_label": label,
                "start_char": match.start(),
                "end_char": match.end(),
            })
    return rows


def main() -> None:
    docs = pd.read_csv(DOCUMENTS, keep_default_na=False)
    required = {"document_id", "text"}
    missing = required - set(docs.columns)
    if missing:
        raise ValueError(f"Missing document columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for document in docs.itertuples(index=False):
        text = str(document.text)
        candidates = (
            extract(text, COUNTRIES, "country")
            + extract(text, CENTRAL_BANKS, "central bank")
            + extract(text, ECONOMIC_EVENTS, "economic event")
        )
        seen: set[tuple[object, ...]] = set()
        for candidate in candidates:
            key = (candidate["start_char"], candidate["end_char"], candidate["entity_label"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"document_id": document.document_id, **candidate})

    gold = pd.DataFrame(rows, columns=["document_id", "entity_text", "entity_label", "start_char", "end_char"])
    gold = gold.sort_values(["document_id", "start_char", "end_char", "entity_label"]).reset_index(drop=True)
    gold.to_csv(ENTITIES, index=False)

    docs["annotation_complete"] = 1
    docs["annotation_notes"] = (
        "Independent rule-based annotation from source text; GLiNER predictions were not read or used."
    )
    docs.to_csv(DOCUMENTS, index=False)

    PROVENANCE.write_text(json.dumps({
        "method": "independent rule-based annotation from source text",
        "prediction_file_used": False,
        "labels": ["country", "central bank", "economic event"],
        "n_documents": int(len(docs)),
        "n_mentions": int(len(gold)),
        "human_audit_required": True,
        "warning": "",
    }, indent=2), encoding="utf-8")
    print(f"Created {len(gold)} independent mentions for {len(docs)} documents.")


if __name__ == "__main__":
    main()
