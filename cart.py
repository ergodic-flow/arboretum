import math
import random
from dataclasses import dataclass
from itertools import combinations
from measures import *
from polars import Series, DataFrame
from typing import Any, Optional


def is_numeric(feat) -> bool:
    return feat.dtype.is_numeric()


def most_common_label(labels: Series) -> int:
    return labels.mode()[0]


def mean_label(labels: Series) -> float:
    return labels.mean()


def find_best_feature_split(df: DataFrame, feature: str, target_column: str, score_split):
    best_score = float("-inf")
    best_split_criteria = None

    if is_numeric(df[feature]):
        values = sorted(value for value in df[feature].unique() if value is not None)
        midpoints = [(v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:])]

        for m in midpoints:
            left_target = df.filter(df[feature] <= m)[target_column]
            right_target = df.filter(df[feature] > m)[target_column]

            score = score_split(df[target_column], left_target, right_target)

            if score > best_score:
                best_score = score
                best_split_criteria = m

        return best_score, best_split_criteria

    else:
        categories = df[feature].unique()
        n = len(categories)

        # We only need to check subsets up to size n // 2.
        for r in range(1, (n // 2) + 1):
            for subset in combinations(categories, r):
                # Partition based on whether the category is in our target subset
                left_mask = df[feature].is_in(subset)
                left_target = df.filter(left_mask)[target_column]
                right_target = df.filter(~left_mask)[target_column]

                score = score_split(df[target_column], left_target, right_target)

                if score > best_score:
                    best_score = score
                    best_split_criteria = subset

        return best_score, best_split_criteria


def split(df: DataFrame, target_column: str, score_split, max_features=None):
    best_value = float("-inf")
    best_split = None
    best_feature = None

    features = [col for col in df.columns if col != target_column]

    # feature subsampling: used for random forests
    if max_features is not None:
        if max_features == "sqrt":
            k = max(1, int(math.sqrt(len(features))))
        elif isinstance(max_features, int):
            k = min(max_features, len(features))
        else:
            k = len(features)
        features = random.sample(features, k)

    for feature in features:
        split_value, feature_split = find_best_feature_split(
            df,
            feature,
            target_column,
            score_split,
        )

        if split_value > best_value:
            best_split = feature_split
            best_value = split_value
            best_feature = feature

    return best_feature, best_split, best_value


@dataclass(eq=False)
class TreeNode:
    feature: str | None = None
    split_criteria: Any = None
    left: Optional[TreeNode] = None
    right: Optional[TreeNode] = None
    value: Any = None
    is_leaf: bool = False
    is_categorical: bool = False
    n_samples: int = 0
    impurity: float = 0.0
    weighted_impurity: float = 0.0
    subtree_risk: float = 0.0
    leaf_count: int = 1


def build_tree(
    df: DataFrame,
    target_column: str,
    depth: int = 0,
    max_depth: int = 5,
    min_samples_split: int = 2,
    score_split=compute_gini_gain,
    compute_impurity=compute_gini_impurity,
    make_leaf_value=most_common_label,
    max_features=None,
    ccp_alpha=0.0,
):
    """
    Recursively builds a binary decision tree using greedy splitting logic.
    """
    total_samples = len(df)

    def make_leaf_node(labels: Series):
        impurity = compute_impurity(labels)
        weighted_impurity = impurity * len(labels) / total_samples
        return TreeNode(
            value=make_leaf_value(labels),
            is_leaf=True,
            n_samples=len(labels),
            impurity=impurity,
            weighted_impurity=weighted_impurity,
            subtree_risk=weighted_impurity,
            leaf_count=1,
        )

    def build_node(node_df: DataFrame, current_depth: int):
        labels = node_df[target_column]

        # --- Stopping Criteria ---

        if len(labels.unique()) == 1:
            return make_leaf_node(labels)

        if len(node_df) < min_samples_split:
            return make_leaf_node(labels)

        if current_depth >= max_depth:
            return make_leaf_node(labels)

        best_feature, best_split, best_gain = split(
            node_df, target_column, score_split, max_features=max_features
        )

        # If no split yields any improvement, stop and make a leaf
        if best_gain <= 0.0 or best_feature is None:
            return make_leaf_node(labels)

        # partition the data
        best_is_categorical = not is_numeric(node_df[best_feature])
        if best_is_categorical:
            # Categorical splitting
            left_mask = node_df[best_feature].is_in(best_split)
        else:
            # Numerical splitting
            left_mask = node_df[best_feature] <= best_split

        left_df = node_df.filter(left_mask)
        right_df = node_df.filter(~left_mask)

        # If a split ends up with an empty dataframe on either side
        if len(left_df) == 0 or len(right_df) == 0:
            return make_leaf_node(labels)

        left_child = build_node(left_df, current_depth + 1)
        right_child = build_node(right_df, current_depth + 1)

        impurity = compute_impurity(labels)
        weighted_impurity = impurity * len(labels) / total_samples
        leaf_count = left_child.leaf_count + right_child.leaf_count
        subtree_risk = left_child.subtree_risk + right_child.subtree_risk

        node = TreeNode(
            feature=best_feature,
            split_criteria=best_split,
            left=left_child,
            right=right_child,
            is_categorical=best_is_categorical,
            n_samples=len(labels),
            impurity=impurity,
            weighted_impurity=weighted_impurity,
            subtree_risk=subtree_risk,
            leaf_count=leaf_count,
        )

        if ccp_alpha > 0.0 and leaf_count > 1:
            effective_alpha = (weighted_impurity - subtree_risk) / (leaf_count - 1)
            if effective_alpha <= ccp_alpha:
                return make_leaf_node(labels)

        return node

    return build_node(df, depth)


def build_classification_tree(
    df: DataFrame,
    target_column: str,
    depth: int = 0,
    max_depth: int = 5,
    min_samples_split: int = 2,
    max_features=None,
    ccp_alpha=0.0,
):
    return build_tree(
        df,
        target_column=target_column,
        depth=depth,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        score_split=compute_gini_gain,
        compute_impurity=compute_gini_impurity,
        make_leaf_value=most_common_label,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
    )


def build_regression_tree(
    df: DataFrame,
    target_column: str,
    depth: int = 0,
    max_depth: int = 5,
    min_samples_split: int = 2,
    max_features=None,
    ccp_alpha=0.0,
):
    return build_tree(
        df,
        target_column=target_column,
        depth=depth,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        score_split=compute_variance_reduction,
        compute_impurity=compute_variance_impurity,
        make_leaf_value=mean_label,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
    )


def predict_row(node: TreeNode, row: dict) -> float | int:
    """
    Traverses the tree recursively for a single data point to find its prediction.
    """
    if node.is_leaf:
        return node.value

    feature_val = row[node.feature]

    if node.is_categorical:
        go_left = feature_val in node.split_criteria
    elif feature_val is None:
        go_left = False
    else:
        go_left = feature_val <= node.split_criteria

    if go_left:
        return predict_row(node.left, row)
    else:
        return predict_row(node.right, row)


def predict(tree_root: TreeNode, df: DataFrame):
    """
    Predicts the target labels for an entire DataFrame.
    """
    return [predict_row(tree_root, row) for row in df.to_dicts()]
