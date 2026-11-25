"""
rebuild_from_mentions.py
========================
Bagian 3A dokumen "Prompt Early & Late Fusion -- Revisi".

Membuat ulang fitur NER artikel + harian dari entity_mentions.csv yang sudah
ada, memakai normalize_country() yang sudah diperbaiki (case-insensitive).
GLiNER TIDAK dijalankan lagi.

- File lama disalin ke *_bug_case.csv sebelum ditimpa.
- Validasi (assert) sesuai prompt 3A langkah 4.
- Menghitung N_NER (poin 7 blok konteks bersama) di 937 hari dev.

Cara pakai (dari root project):
    python src/ner/rebuild_from_mentions.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import ner  # noqa: E402

NER_DIR = ROOT / "data" / "processed" / "ner"
MENTIONS_CSV = NER_DIR / "entity_mentions.csv"
DAILY_CSV = NER_DIR / "daily_entity_features.csv"
ARTICLE_CSV = NER_DIR / "article_entity_features.csv"
DAILY_BACKUP = NER_DIR / "daily_entity_features_bug_case.csv"
ARTICLE_BACKUP = NER_DIR / "article_entity_features_bug_case.csv"
REPORT_JSON = NER_DIR / "rebuild_report.json"

# Kolom "Seharusnya" di tabel bagian 2 dokumen revisi.
EXPECTED_COUNTRY_TOTALS = {
    "Russia": 14449, "Ukraine": 13817, "Iran": 11958, "China": 11473,
    "Israel": 4620, "India": 3112, "Japan": 1456, "Germany": 1258,
    "Saudi Arabia": 953, "South Korea": 924, "Turkey": 609, "Indonesia": 156,
    "United States": 27079, "United Kingdom": 1665, "Taiwan": 1823,
}
HAS_COLS = ["has_country", "has_central_bank", "has_economic_event"]
# Kolom yang boleh berubah karena perbaikan normalisasi negara.
COUNTRY_RELATED = {"n_unique_countries", "daily_n_unique_countries"} | {
    f"country_{ner.slugify(c)}_mentions" for c in ner.COUNTRIES_OF_INTEREST
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # ---- 1. cadangan -------------------------------------------------------
    for src, dst in [(DAILY_CSV, DAILY_BACKUP), (ARTICLE_CSV, ARTICLE_BACKUP)]:
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"Cadangan dibuat : {dst.name}")
        else:
            print(f"Cadangan sudah ada (tidak ditimpa): {dst.name}")

    old_daily = pd.read_csv(DAILY_BACKUP, parse_dates=["date"])
    old_article = pd.read_csv(ARTICLE_BACKUP, parse_dates=["target_date"])

    # ---- 2. mention + normalisasi negara ulang -------------------------------
    mentions = pd.read_csv(MENTIONS_CSV, keep_default_na=False, na_values=[""])
    mentions["target_date"] = pd.to_datetime(mentions["target_date"]).dt.normalize()
    is_country = mentions["entity_label"].eq("country")
    old_norm = mentions["normalized_entity"].copy()
    mentions.loc[is_country, "normalized_entity"] = (
        mentions.loc[is_country, "entity_text"].astype(str).map(ner.normalize_country)
    )
    changed = int((old_norm != mentions["normalized_entity"]).sum())
    print(f"Mention negara  : {int(is_country.sum()):,} | normalized_entity berubah: {changed:,}")
    assert (old_norm[~is_country] == mentions.loc[~is_country, "normalized_entity"]).all()

    # ---- 3. bangun ulang fitur dengan fungsi yang sama persis ---------------
    articles = old_article[["document_id", "target_date", "url"]].copy()
    article_features = ner.make_article_features(articles, mentions)
    daily = ner.make_daily_features(articles, mentions, article_features)

    # ---- 4. validasi ---------------------------------------------------------
    # a. tanggal
    assert len(daily) == 1202, len(daily)
    assert not daily["date"].duplicated().any()
    assert set(daily["date"]) == set(old_daily["date"])
    daily = daily.set_index("date").loc[old_daily["date"]].reset_index()
    # b. kolom
    assert list(daily.columns) == list(old_daily.columns), (
        set(daily.columns) ^ set(old_daily.columns)
    )
    assert len(daily.columns) - 1 == 54
    # c. total per negara
    country_rows = []
    ok = True
    for country, expected in EXPECTED_COUNTRY_TOTALS.items():
        col = f"country_{ner.slugify(country)}_mentions"
        old_total = int(old_daily[col].sum())
        new_total = int(daily[col].sum())
        direct = int((mentions.loc[is_country, "normalized_entity"] == country).sum())
        country_rows.append(
            {"country": country, "old": old_total, "new": new_total,
             "target": expected, "mentions_direct": direct}
        )
        ok &= new_total == expected
    country_table = pd.DataFrame(country_rows)
    print("\nTotal sebutan per negara:")
    print(country_table.to_string(index=False))
    if not ok:
        raise AssertionError("Total per negara tidak sama dengan target bagian 2. Berhenti.")
    # d. kolom non-negara identik
    other_cols = [c for c in old_daily.columns if c != "date" and c not in COUNTRY_RELATED]
    diff_cols = [
        c for c in other_cols
        if not np.allclose(daily[c].to_numpy(float), old_daily[c].to_numpy(float))
    ]
    assert not diff_cols, f"Kolom non-negara berubah: {diff_cols}"
    changed_cols = [
        c for c in COUNTRY_RELATED
        if not np.allclose(daily[c].to_numpy(float), old_daily[c].to_numpy(float))
    ]
    # e. has_* masih = n_articles_with_* (bug #2 terdokumentasi)
    for has, n_with in zip(HAS_COLS, ["n_articles_with_country", "n_articles_with_central_bank",
                                     "n_articles_with_economic_event"]):
        assert (daily[has] == daily[n_with]).all(), has

    # ---- 5. N_NER di 937 hari dev -------------------------------------------
    from dataset_split import canonical_split

    split = canonical_split()
    dev_dates = split.loc[split["split"] != "test", "date"]
    assert len(dev_dates) == 937 and dev_dates.max() == pd.Timestamp("2025-12-09")
    dev = daily.set_index("date").loc[dev_dates.to_numpy()]
    candidates = [c for c in daily.columns if c != "date" and c not in HAS_COLS]
    constant = [c for c in candidates if dev[c].nunique(dropna=False) <= 1]
    ner_cols = [c for c in candidates if c not in constant]
    print(f"\nKolom konstan di 937 hari dev ({len(constant)}): {constant}")
    print(f"N_NER = {len(ner_cols)}")

    # ---- simpan --------------------------------------------------------------
    daily.to_csv(DAILY_CSV, index=False)
    article_features.to_csv(ARTICLE_CSV, index=False)
    report = {
        "source": str(MENTIONS_CSV.relative_to(ROOT)),
        "fix": "normalize_country(): pencocokan nama kanonik case-insensitive, tanpa alias baru",
        "country_mentions_renormalized": changed,
        "country_totals": country_rows,
        "columns_changed": sorted(changed_cols),
        "has_cols_dropped": HAS_COLS,
        "constant_cols_dev": constant,
        "N_NER": len(ner_cols),
        "NER_COLS": ner_cols,
        "md5": {
            DAILY_CSV.name: md5(DAILY_CSV),
            DAILY_BACKUP.name: md5(DAILY_BACKUP),
            ARTICLE_CSV.name: md5(ARTICLE_CSV),
            ARTICLE_BACKUP.name: md5(ARTICLE_BACKUP),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKolom yang berubah: {sorted(changed_cols)}")
    print(f"Tersimpan: {DAILY_CSV.name}, {ARTICLE_CSV.name}, {REPORT_JSON.name}")
    for k, v in report["md5"].items():
        print(f"  MD5 {k}: {v}")


if __name__ == "__main__":
    main()
