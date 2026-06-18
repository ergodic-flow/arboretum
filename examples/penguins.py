from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from cart import build_classification_tree, predict
from forest import build_classification_forest, predict_classification

DATA_PATH = ROOT / "data" / "penguins.csv"
SUMMARY_LABELS = {"macro avg", "weighted avg"}


def print_average_classification_report(reports):
    labels = [label for label, scores in reports[0].items() if isinstance(scores, dict)]
    averaged_scores = {}

    for label in labels:
        averaged_scores[label] = {
            "precision": mean(report[label]["precision"] for report in reports),
            "recall": mean(report[label]["recall"] for report in reports),
            "f1-score": mean(report[label]["f1-score"] for report in reports),
            "support": sum(report[label]["support"] for report in reports),
        }

    accuracy = mean(report["accuracy"] for report in reports)
    total_support = sum(report["weighted avg"]["support"] for report in reports)
    label_width = max(12, max(len(label) for label in labels))

    print(
        f"{'':>{label_width}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    )
    print()

    for label in labels:
        if label in SUMMARY_LABELS:
            continue

        scores = averaged_scores[label]
        print(
            f"{label:>{label_width}} "
            f"{scores['precision']:>10.2f} "
            f"{scores['recall']:>10.2f} "
            f"{scores['f1-score']:>10.2f} "
            f"{scores['support']:>10.0f}"
        )

    print(
        f"{'accuracy':>{label_width}} "
        f"{'':>10} "
        f"{'':>10} "
        f"{accuracy:>10.2f} "
        f"{total_support:>10.0f}"
    )

    for label in labels:
        if label not in SUMMARY_LABELS:
            continue

        scores = averaged_scores[label]
        print(
            f"{label:>{label_width}} "
            f"{scores['precision']:>10.2f} "
            f"{scores['recall']:>10.2f} "
            f"{scores['f1-score']:>10.2f} "
            f"{scores['support']:>10.0f}"
        )


def main():
    df = pl.read_csv(DATA_PATH)
    tree_reports = []
    forest_reports = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kfold.split(df, df["species"].to_list()):
        train_df = df[train_idx.tolist()]
        test_df = df[test_idx.tolist()]

        tree = build_classification_tree(
            train_df, target_column="species", max_depth=5, min_samples_split=2
        )
        forest = build_classification_forest(
            train_df,
            target_column="species",
            n_trees=5,
            max_depth=5,
            min_samples_split=2,
            random_state=42,
        )

        y_true = test_df["species"].to_list()
        tree_pred = predict(tree, test_df)
        forest_pred = predict_classification(forest, test_df)

        tree_reports.append(
            classification_report(y_true, tree_pred, output_dict=True, zero_division=0)
        )
        forest_reports.append(
            classification_report(
                y_true, forest_pred, output_dict=True, zero_division=0
            )
        )

    print("Single Tree - Average 5-Fold Cross-Validation Report")
    print_average_classification_report(tree_reports)
    print()
    print("Random Forest - Average 5-Fold Cross-Validation Report")
    print_average_classification_report(forest_reports)


if __name__ == "__main__":
    main()
