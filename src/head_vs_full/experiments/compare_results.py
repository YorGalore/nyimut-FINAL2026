
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Root repository: nyimut-NLP26/
ROOT_DIR = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "head_vs_full"
)

RESULTS_PATH = OUTPUT_DIR / "experiment_results.csv"

METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]


# ============================================================
# LOAD RESULTS
# ============================================================

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {RESULTS_PATH}\n"
        "Jalankan train_evaluate.py terlebih dahulu."
    )

results = pd.read_csv(RESULTS_PATH)

if results.empty:
    raise ValueError("File experiment_results.csv kosong.")

# Do not mix the single JISDOR-only baseline into the headline/fulltext pivot.
# Every remaining row has the same JISDOR history plus one text representation.
if "feature_set" in results.columns:
    price_baseline = results[results["feature_set"] == "jisdor_price_only"].copy()
    results = results[results["feature_set"] == "jisdor_plus_embedding"].copy()
else:
    price_baseline = pd.DataFrame()

# Select representations by validation externally; this report presents the
# held-out 15% test rows only, so validation and test are never mixed.
if "evaluation_split" in results.columns:
    results = results[results["evaluation_split"] == "test"].copy()
    if not price_baseline.empty:
        price_baseline = price_baseline[price_baseline["evaluation_split"] == "test"].copy()

print("=" * 70)
print("HEADLINE VS FULL TEXT COMPARISON")
print("=" * 70)


# ============================================================
# SORT RESULTS
# ============================================================

results = results.sort_values(
    by=["classifier", "model", "text_type"] if "classifier" in results.columns else ["model", "text_type"]
).reset_index(drop=True)


# ============================================================
# PIVOT COMPARISON
# ============================================================

comparison = results.pivot(
    index=["classifier", "model"] if "classifier" in results.columns else "model",
    columns="text_type",
    values=METRICS,
)

comparison.columns = [
    f"{metric}_{text_type}"
    for metric, text_type in comparison.columns
]

comparison = comparison.reset_index()


# ============================================================
# CALCULATE DIFFERENCES
# Positive difference = Headline performs better
# Negative difference = Full text performs better
# ============================================================

for metric in METRICS:

    headline_col = f"{metric}_headline"
    fulltext_col = f"{metric}_fulltext"

    if headline_col in comparison.columns and fulltext_col in comparison.columns:

        comparison[f"{metric}_difference"] = (
            comparison[headline_col]
            - comparison[fulltext_col]
        )


# ============================================================
# DETERMINE WINNER
# ============================================================

def determine_winner(row, metric):

    headline = row.get(f"{metric}_headline", np.nan)
    fulltext = row.get(f"{metric}_fulltext", np.nan)

    if pd.isna(headline) or pd.isna(fulltext):
        return "Incomplete"

    if np.isclose(headline, fulltext):
        return "Tie"

    return "Headline" if headline > fulltext else "Full Text"


comparison["roc_auc_winner"] = comparison.apply(
    lambda row: determine_winner(row, "roc_auc"),
    axis=1,
)

comparison["pr_auc_winner"] = comparison.apply(
    lambda row: determine_winner(row, "pr_auc"),
    axis=1,
)


# ============================================================
# SAVE DETAILED COMPARISON
# ============================================================

comparison_path = OUTPUT_DIR / "headline_vs_fulltext_comparison.csv"

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = (
    results
    .groupby(["classifier", "text_type"] if "classifier" in results.columns else "text_type")[METRICS]
    .mean()
    .reset_index()
)

summary_path = OUTPUT_DIR / "overall_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n1. INDIVIDUAL MODEL COMPARISON")
print("-" * 70)

display_columns = [
    "classifier",
    "model",
    "roc_auc_headline",
    "roc_auc_fulltext",
    "roc_auc_difference",
    "roc_auc_winner",
]

available_columns = [
    col for col in display_columns
    if col in comparison.columns
]

print(
    comparison[available_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n2. OVERALL AVERAGE PERFORMANCE")
print("-" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n3. ROC-AUC WINNER COUNT")
print("-" * 70)

print(
    comparison["roc_auc_winner"].value_counts().to_string()
)

if not price_baseline.empty:
    print("\n4. JISDOR PRICE-ONLY BASELINE")
    print("-" * 70)
    print(
        price_baseline[
            [c for c in ["classifier", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "pr_auc"] if c in price_baseline.columns]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print(f"Detailed comparison: {comparison_path}")
print(f"Overall summary    : {summary_path}")
