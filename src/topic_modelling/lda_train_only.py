"""
lda_train_only.py
=================
Bagian 3B langkah 0b (item #4 bagian 2 dokumen revisi).

CountVectorizer + LDA di-FIT hanya di dokumen bertanggal <= TRAIN_END
(2025-03-25), dengan hyperparameter yang sama persis dengan run_lda() di
topic_models.py, lalu TRANSFORM semua dokumen. Output disimpan di folder baru
(file lama data/processed/topic_modelling/lda/ dan features/ tidak disentuh):

    data/processed/topic_modelling/lda_train_only/document_topics.csv
    data/processed/topic_modelling/lda_train_only/topic_summary.csv
    data/processed/topic_modelling/lda_train_only/daily_topic_features.csv
    data/processed/topic_modelling/lda_train_only/run_metadata.json

Tidak ada label yang dipakai. Transform dokumen setelah TRAIN_END hanya
memakai model yang sudah di-fit, jadi topik tidak "tahu" berita valid/test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topic_models as TM  # noqa: E402

TRAIN_END = pd.Timestamp("2025-03-25")
OUT_DIR = TM.OUTPUT_DIR / "lda_train_only"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = TM.prepare_documents(TM.load_data())
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    is_train = df["target_date"] <= TRAIN_END
    print(f"Dokumen total: {len(df):,} | fit (<= {TRAIN_END.date()}): {int(is_train.sum()):,}")

    vectorizer = CountVectorizer(
        tokenizer=TM.tokenize_text,
        token_pattern=None,
        lowercase=True,
        max_features=TM.LDA_MAX_FEATURES,
        min_df=TM.LDA_MIN_DF,
        max_df=TM.LDA_MAX_DF,
    )
    dtm_train = vectorizer.fit_transform(df.loc[is_train, "text_norm"].tolist())
    lda = LatentDirichletAllocation(
        n_components=TM.N_TOPICS,
        max_iter=TM.LDA_MAX_ITER,
        learning_method="batch",
        random_state=TM.RANDOM_STATE,
        n_jobs=-1,
        evaluate_every=-1,
    )
    lda.fit(dtm_train)
    probabilities = lda.transform(vectorizer.transform(df["text_norm"].tolist()))

    feature_names = vectorizer.get_feature_names_out()
    summary = pd.DataFrame([
        {
            "topic_id": k,
            "topic_name": f"Topic {k}",
            "topic_words": ", ".join(feature_names[comp.argsort()[-TM.TOP_WORDS:][::-1]]),
            "document_count_train": int((probabilities[is_train.to_numpy()].argmax(1) == k).sum()),
        }
        for k, comp in enumerate(lda.components_)
    ])

    topic_ids = list(range(TM.N_TOPICS))
    document_topics = TM.build_document_topic_dataframe(df, probabilities, topic_ids, "lda_train_only")
    daily = TM.aggregate_daily_features(
        document_topics=document_topics, topic_ids=topic_ids, model_name="lda_train_only"
    )
    TM.save_csv(document_topics, OUT_DIR / "document_topics.csv")
    TM.save_csv(summary, OUT_DIR / "topic_summary.csv")
    TM.save_csv(daily, OUT_DIR / "daily_topic_features.csv")

    meta = {
        "fit_period_end": str(TRAIN_END.date()),
        "n_documents_total": int(len(df)),
        "n_documents_fit": int(is_train.sum()),
        "vocabulary_size": int(len(feature_names)),
        "hyperparameters": {
            "n_topics": TM.N_TOPICS, "max_features": TM.LDA_MAX_FEATURES,
            "min_df": TM.LDA_MIN_DF, "max_df": TM.LDA_MAX_DF,
            "max_iter": TM.LDA_MAX_ITER, "random_state": TM.RANDOM_STATE,
        },
        "daily_rows": int(len(daily)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(summary[["topic_id", "topic_words"]].to_string(index=False))


if __name__ == "__main__":
    main()
