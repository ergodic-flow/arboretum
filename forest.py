import random
from collections import Counter
from polars import DataFrame
from statistics import mean

from cart import (
    build_classification_tree,
    build_regression_tree,
    predict as predict_tree,
)


def bootstrap_sample(df, rng=random):
    indices = [rng.randint(0, len(df) - 1) for _ in range(len(df))]
    return df[indices]


def build_classification_forest(
    df: DataFrame,
    target_column: str,
    n_trees=100,
    max_depth=5,
    min_samples_split=2,
    max_features="sqrt",
    random_state=None,
):
    if random_state is not None:
        random.seed(random_state)

    trees = []
    for _ in range(n_trees):
        sample_df = bootstrap_sample(df)
        tree = build_classification_tree(
            sample_df,
            target_column=target_column,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        trees.append(tree)

    return trees


def build_regression_forest(
    df,
    target_column="target",
    n_trees=100,
    max_depth=5,
    min_samples_split=2,
    max_features="sqrt",
    random_state=None,
):
    if random_state is not None:
        random.seed(random_state)

    trees = []
    for _ in range(n_trees):
        sample_df = bootstrap_sample(df)
        tree = build_regression_tree(
            sample_df,
            target_column=target_column,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        trees.append(tree)

    return trees


def predict_classification(forest, df):
    all_preds = [predict_tree(tree, df) for tree in forest]
    return [Counter(row_preds).most_common(1)[0][0] for row_preds in zip(*all_preds)]


def predict_regression(forest, df):
    all_preds = [predict_tree(tree, df) for tree in forest]
    return [mean(row_preds) for row_preds in zip(*all_preds)]
