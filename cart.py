import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

from measures import (
    compute_gini_gain,
    compute_gini_impurity,
    compute_variance_impurity,
    compute_variance_reduction,
)
from polars import DataFrame, Series


def is_numeric(feat) -> bool:
    return feat.dtype.is_numeric()


def most_common_label(labels: Series) -> int:
    return labels.mode()[0]


def mean_label(labels: Series) -> float:
    return labels.mean()


def _gini_from_counts(counts: Counter, n: int) -> float:
    """Compute Gini impurity without rebuilding a target Series."""
    if n == 0:
        return 0.0
    return 1.0 - sum(count * count for count in counts.values()) / (n * n)


def _sse(n: int, total: float, total_sq: float) -> float:
    """Sum of squared errors represented by count, sum, and sum of squares."""
    if n == 0:
        return 0.0
    # Roundoff can make this a tiny negative number for nearly constant targets.
    return max(0.0, total_sq - total * total / n)


def _classification_gain(parent_counts, left_counts, n_left, n_total):
    n_right = n_total - n_left
    right_counts = parent_counts - left_counts
    parent_impurity = _gini_from_counts(parent_counts, n_total)
    children_impurity = (
        n_left * _gini_from_counts(left_counts, n_left)
        + n_right * _gini_from_counts(right_counts, n_right)
    ) / n_total
    return parent_impurity - children_impurity


def _best_numeric_classification_split(values, labels):
    """Sort once, then move class counts from the right child to the left."""
    pairs = sorted(
        (value, label)
        for value, label in zip(values, labels)
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    )
    if len(pairs) < 2:
        return float("-inf"), None

    parent_counts = Counter(labels)
    left_counts = Counter()
    n_total = len(labels)
    n_left = 0
    best_gain = float("-inf")
    best_threshold = None
    i = 0

    while i < len(pairs):
        value = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == value:
            left_counts[pairs[i][1]] += 1
            n_left += 1
            i += 1

        if i == len(pairs):
            break

        gain = _classification_gain(parent_counts, left_counts, n_left, n_total)
        if gain > best_gain:
            best_gain = gain
            best_threshold = (value + pairs[i][0]) / 2

    return best_gain, best_threshold


def _best_numeric_regression_split(values, labels):
    """Sort once, then scan count/sum/sum-of-squares statistics."""
    pairs = sorted(
        (value, label)
        for value, label in zip(values, labels)
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    )
    if len(pairs) < 2:
        return float("-inf"), None

    n_total = len(labels)
    total = sum(labels)
    total_sq = sum(label * label for label in labels)
    parent_sse = _sse(n_total, total, total_sq)
    left_n = 0
    left_sum = 0.0
    left_sum_sq = 0.0
    best_gain = float("-inf")
    best_threshold = None
    i = 0

    while i < len(pairs):
        value = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == value:
            label = pairs[i][1]
            left_n += 1
            left_sum += label
            left_sum_sq += label * label
            i += 1

        if i == len(pairs):
            break

        right_n = n_total - left_n
        right_sum = total - left_sum
        right_sum_sq = total_sq - left_sum_sq
        gain = (
            parent_sse
            - _sse(left_n, left_sum, left_sum_sq)
            - _sse(right_n, right_sum, right_sum_sq)
        ) / n_total
        if gain > best_gain:
            best_gain = gain
            best_threshold = (value + pairs[i][0]) / 2

    return best_gain, best_threshold


def _best_categorical_classification_split(values, labels):
    """Aggregate class counts by category, then score each category subset."""
    category_counts = defaultdict(Counter)
    for category, label in zip(values, labels):
        category_counts[category][label] += 1

    categories = list(category_counts)
    parent_counts = Counter(labels)
    n_total = len(labels)
    best_gain = float("-inf")
    best_subset = None

    for r in range(1, (len(categories) // 2) + 1):
        for subset in combinations(categories, r):
            left_counts = Counter()
            for category in subset:
                left_counts.update(category_counts[category])
            n_left = sum(left_counts.values())
            gain = _classification_gain(parent_counts, left_counts, n_left, n_total)
            if gain > best_gain:
                best_gain = gain
                best_subset = subset

    return best_gain, best_subset


def _best_categorical_regression_split(values, labels):
    """Aggregate regression statistics by category, then score each subset."""
    category_stats = defaultdict(lambda: [0, 0.0, 0.0])
    for category, label in zip(values, labels):
        stats = category_stats[category]
        stats[0] += 1
        stats[1] += label
        stats[2] += label * label

    categories = list(category_stats)
    n_total = len(labels)
    total = sum(labels)
    total_sq = sum(label * label for label in labels)
    parent_sse = _sse(n_total, total, total_sq)
    best_gain = float("-inf")
    best_subset = None

    for r in range(1, (len(categories) // 2) + 1):
        for subset in combinations(categories, r):
            left_n = sum(category_stats[c][0] for c in subset)
            left_sum = sum(category_stats[c][1] for c in subset)
            left_sum_sq = sum(category_stats[c][2] for c in subset)
            gain = (
                parent_sse
                - _sse(left_n, left_sum, left_sum_sq)
                - _sse(n_total - left_n, total - left_sum, total_sq - left_sum_sq)
            ) / n_total
            if gain > best_gain:
                best_gain = gain
                best_subset = subset

    return best_gain, best_subset


def _categorical_left_mask(values: Series, subset):
    """Create a two-valued mask and treat null as an ordinary category."""
    non_null_categories = [category for category in subset if category is not None]
    mask = values.is_in(non_null_categories).fill_null(False)
    if None in subset:
        mask = mask | values.is_null()
    return mask


def _find_best_feature_split_fallback(
    df: DataFrame, feature: str, target_column: str, score_split
):
    """Readable fallback for callers supplying a custom split score."""
    best_score = float("-inf")
    best_split_criteria = None

    if is_numeric(df[feature]):
        values = sorted(value for value in df[feature].unique() if value is not None)
        candidates = ((v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:]))
    else:
        categories = df[feature].unique().to_list()
        candidates = (
            subset
            for r in range(1, (len(categories) // 2) + 1)
            for subset in combinations(categories, r)
        )

    for candidate in candidates:
        if is_numeric(df[feature]):
            left_mask = df[feature].is_not_null() & (df[feature] <= candidate)
        else:
            left_mask = _categorical_left_mask(df[feature], candidate)
        left_target = df.filter(left_mask)[target_column]
        right_target = df.filter(~left_mask)[target_column]
        score = score_split(df[target_column], left_target, right_target)
        if score > best_score:
            best_score = score
            best_split_criteria = candidate

    return best_score, best_split_criteria


def find_best_feature_split(df: DataFrame, feature: str, target_column: str, score_split):
    """Find a CART split, using sufficient statistics for the built-in scores."""
    values = df[feature].to_list()
    labels = df[target_column].to_list()
    numeric = is_numeric(df[feature])

    if score_split is compute_gini_gain:
        finder = (
            _best_numeric_classification_split
            if numeric
            else _best_categorical_classification_split
        )
        return finder(values, labels)

    if score_split is compute_variance_reduction:
        finder = (
            _best_numeric_regression_split
            if numeric
            else _best_categorical_regression_split
        )
        return finder(values, labels)

    return _find_best_feature_split_fallback(
        df, feature, target_column, score_split
    )


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
            left_mask = _categorical_left_mask(node_df[best_feature], best_split)
        else:
            # Numerical splitting. Missing values follow the same right-hand
            # branch during both training and prediction.
            left_mask = node_df[best_feature].is_not_null() & (
                node_df[best_feature] <= best_split
            )

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


def _predict_row_at_index(node: TreeNode, columns: dict, index: int):
    """Traverse a tree without constructing a dictionary for every row."""
    while not node.is_leaf:
        feature_val = columns[node.feature][index]

        if node.is_categorical:
            go_left = feature_val in node.split_criteria
        elif feature_val is None:
            go_left = False
        else:
            go_left = feature_val <= node.split_criteria

        node = node.left if go_left else node.right

    return node.value


def predict(tree_root: TreeNode, df: DataFrame):
    """Predict all rows using column-oriented data access."""
    columns = df.to_dict(as_series=False)
    return [_predict_row_at_index(tree_root, columns, i) for i in range(len(df))]
