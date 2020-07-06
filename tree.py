import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class _Node:
    """The fundamental data structure representing a binary decision tree.

    Parameters
    ----------
    min_samples_split : int
        The minimum number of samples required to split a node.

    Attributes
    ----------
    left : _Node or None
        The left node of the tree or None if the current node is a leaf.
    right : _Node or None
        The right node of the tree or None if the current node is a leaf.
    feature_index: int
        The column index of the feature to split on in the current node.
    split_value : float or None
        The feature value to split by or None if the node is a leaf.
    prediction : float or None
        The prediction value if the node is a leaf or None.
    """

    def __init__(self, min_samples_split):
        self._min_samples_split = min_samples_split

        self.left = None
        self.right = None
        self.feature_index = None
        self.split_value = None
        self.prediction = None

    @staticmethod
    def _find_best_split(x, y):
        (num_samples,) = x.shape

        best_score = np.inf
        best_split_value = None
        best_partition = None
        for i in np.arange(num_samples):
            # Use average of two consecutive feature values as split value.
            # split_value = (x[i] + x[i+1]) / 2
            split_value = x[i]

            # Obtain binary masks for all samples whose feature values are
            # below (left) or above (right) the split value.
            mask_left = x < split_value
            mask_right = x >= split_value

            # If we can't split the samples based on 'split_value', move on.
            if not mask_left.any() or not mask_right.any():
                continue

            y_left = y[mask_left]
            y_right = y[mask_right]

            # Score the candidate split.
            score = (((y_left - y_left.mean()) ** 2).sum() +
                     ((y_right - y_right.mean()) ** 2).sum())

            if score < best_score:
                best_score = score
                best_split_value = split_value
                best_partition = (mask_left, mask_right)

        return {
            "score": best_score,
            "split_value": best_split_value,
            "partition": best_partition
        }

    def construct_tree(self, X, y):
        X, y = map(np.array, (X, y))
        num_samples, num_features = X.shape

        # Too few samples to split, so turn node into a leaf.
        if num_samples < self._min_samples_split:
            self.prediction = y.mean()
            return

        # For each feature, compute the best split.
        feature_scores = {}
        for feature_index in np.arange(num_features):
            x = X[:, feature_index]
            feature_scores[feature_index] = self._find_best_split(x, y)

        # Retrieve the split configuration for the best (lowest) score.
        feature_index = min(feature_scores,
                            key=lambda key: feature_scores[key]["score"])
        split = feature_scores[feature_index]
        split_value = split["split_value"]
        if split_value is None:
            self.prediction = y.mean()
            return

        self.feature_index = feature_index
        self.split_value = split_value
        mask_left, mask_right = split["partition"]

        self.left = _Node(self._min_samples_split)
        self.left.construct_tree(X[mask_left, :], y[mask_left])

        self.right = _Node(self._min_samples_split)
        self.right.construct_tree(X[mask_right, :], y[mask_right])

    def _apply_tree_to_sample(self, x):
        node = self
        while True:
            if node.split_value is None:
                return node.prediction
            if x[node.feature_index] < node.split_value:
                node = node.left
            else:
                node = node.right
        raise RuntimeError("No leaf node reached")

    def apply(self, X):
        X = np.array(X)
        return np.array([self._apply_tree_to_sample(row) for row in X])


class DecisionTree(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_split=2):
        self.min_samples_split_ = min_samples_split

        self._tree = None

    def fit(self, X, y):
        self._tree = _Node(self.min_samples_split_)
        self._tree.construct_tree(X, y)
        return self

    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("Tree needs to be fitted first")
        return self._tree.apply(X)
