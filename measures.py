import math

from polars import Series


def compute_gini_impurity(labels: Series) -> float:
    """
    Calculates the Gini impurity for a list/Series of class labels.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return 0.0

    counts = labels.value_counts()["count"]
    sum_squared_probs = sum((count / total_samples) ** 2 for count in counts)

    return 1.0 - sum_squared_probs


def compute_entropy_impurity(labels: Series) -> float:
    """
    Calculates the entropy impurity for a list/Series of class labels.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return 0.0

    counts = labels.value_counts()["count"]
    entropy = 0.0

    for count in counts:
        probability = count / total_samples
        entropy -= probability * math.log2(probability)

    return entropy


def compute_variance_impurity(labels: Series) -> float:
    """
    Calculates the mean squared deviation from the mean for regression targets.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return 0.0

    mean = labels.mean()
    return ((labels - mean) ** 2).mean()


def compute_impurity_reduction(
    parent_labels: Series,
    left_labels: Series,
    right_labels: Series,
    compute_impurity,
) -> float:
    """
    Calculates the reduction in impurity achieved by a split.
    """
    n_parent = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if n_left == 0 or n_right == 0:
        return 0.0

    parent_impurity = compute_impurity(parent_labels)
    left_impurity = compute_impurity(left_labels)
    right_impurity = compute_impurity(right_labels)

    weighted_children_impurity = (n_left / n_parent) * left_impurity + (
        n_right / n_parent
    ) * right_impurity

    return parent_impurity - weighted_children_impurity


def compute_gini_gain(
    parent_labels: Series, left_labels: Series, right_labels: Series
) -> float:
    """
    Calculates the Gini Gain achieved by a classification split.
    """
    return compute_impurity_reduction(
        parent_labels,
        left_labels,
        right_labels,
        compute_gini_impurity,
    )


def compute_information_gain(
    parent_labels: Series, left_labels: Series, right_labels: Series
) -> float:
    """
    Calculates the Information Gain achieved by a classification split.
    """
    return compute_impurity_reduction(
        parent_labels,
        left_labels,
        right_labels,
        compute_entropy_impurity,
    )


def compute_variance_reduction(
    parent_labels: Series, left_labels: Series, right_labels: Series
) -> float:
    """
    Calculates the reduction in target variance achieved by a regression split.
    """
    return compute_impurity_reduction(
        parent_labels,
        left_labels,
        right_labels,
        compute_variance_impurity,
    )
