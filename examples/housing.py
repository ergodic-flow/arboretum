from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold

from cart import build_regression_tree, compute_variance_reduction, mean_label, predict
from forest import build_regression_forest, predict_regression


DATA_PATH = ROOT / "data" / "housing.csv"
SAMPLE_SIZE = 500


def print_average_metrics(metrics):
    print(f"MAE: {mean(metric['mae'] for metric in metrics):.2f}")
    print(f"RMSE: {mean(metric['rmse'] for metric in metrics):.2f}")
    print(f"R2: {mean(metric['r2'] for metric in metrics):.3f}")


def main():
    df = pl.read_csv(DATA_PATH).drop_nulls().sample(n=SAMPLE_SIZE, seed=42)
    tree_metrics = []
    forest_metrics = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kfold.split(df):
        train_df = df[train_idx.tolist()]
        test_df = df[test_idx.tolist()]

        tree = build_regression_tree(
            train_df,
            target_column="median_house_value",
            max_depth=3,
            min_samples_split=20,
        )
        forest = build_regression_forest(
            train_df,
            target_column="median_house_value",
            n_trees=5,
            max_depth=3,
            min_samples_split=20,
            random_state=42,
        )

        y_true = test_df["median_house_value"].to_list()
        tree_pred = predict(tree, test_df)
        forest_pred = predict_regression(forest, test_df)

        for metrics, y_pred in (
            (tree_metrics, tree_pred),
            (forest_metrics, forest_pred),
        ):
            metrics.append(
                {
                    "mae": mean_absolute_error(y_true, y_pred),
                    "rmse": root_mean_squared_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                }
            )

    print("Single Tree - Average 5-Fold Cross-Validation Report")
    print_average_metrics(tree_metrics)
    print()
    print("Random Forest - Average 5-Fold Cross-Validation Report")
    print_average_metrics(forest_metrics)


if __name__ == "__main__":
    main()
