import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Tree:
    """The fundamental data structure representing a binary decision tree.

    Parameters
    ----------
    min_samples_split : int
        The minimum number of samples required to split an internal node.

    Attributes
    ----------
    left : Tree or None
        The left node of the tree or None if the current node is a leaf.
    right : Tree or None
        The right node of the tree or None if the current node is a leaf.
    feature_index: int
        The column index of the feature to split on in the current node.
    threshold : float or None
        The feature value to split by or None if the node is a leaf.
    prediction : float or None
        The prediction value if the node is a leaf or None.
    """

    def __init__(self, min_samples_split):
        self._min_samples_split = min_samples_split

        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.prediction = None

    def _score_split(self, y, partition):
        """
        Return the mean squared error of a potential split partition.

        Parameters
        ----------
        y : (num_samples,) ndarray
            The vector of targets in the current node.

        partition : tuple
            A 2-tuple of boolean masks to index left and right samples in
            ``y``.
        """
        mask_left, mask_right = partition
        y_left = y[mask_left]
        y_right = y[mask_right]
        return (np.sum((y_left - y_left.mean()) ** 2) +
                np.sum((y_right - y_right.mean()) ** 2))

    def _find_best_split(self, x, y):
        """Find the best split for a vector of samples and the corresponding
        target vector.

        Determine the threshold in `x` which optimizes the split score by
        minimizing the mean squared error of the target.

        Parameters
        ----------
        x : (num_samples,) ndarray
            The vector of observations of a particular feature.
        y : (num_samples,) ndarray
            The vector of targets.

        Returns
        -------
        split_configuration : dict
            A dictionary with the best `score`, `threshold` and `partition`.
        """
        best_score = np.inf
        best_threshold = None
        best_partition = None
        for threshold in x:
            # Obtain binary masks for all samples whose feature values are
            # below (left) or above (right) the split threshold.
            mask_left = x < threshold
            mask_right = x >= threshold

            # If we can't split the samples based on `threshold', move on.
            if not mask_left.any() or not mask_right.any():
                continue

            # Score the candidate split.
            partition = (mask_left, mask_right)
            score = self._score_split(y, partition)

            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_partition = partition

        return {
            "score": best_score,
            "threshold": best_threshold,
            "partition": best_partition
        }

    def construct_tree(self, X, y):
        """Construct the binary decision tree via recursive splitting.

        Parameters
        ----------
        X : (num_samples, num_features) ndarray
            The matrix of observations.
        y : (num_samples,) ndarray
            The vector of targets corresponding to the observations `X`.
        """
        num_samples, num_features = X.shape

        # Too few samples to split, so turn the node into a leaf.
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
        threshold = split["threshold"]
        if threshold is None:
            self.prediction = y.mean()
            return

        self.feature_index = feature_index
        self.threshold = threshold
        mask_left, mask_right = split["partition"]

        self.left = Tree(self._min_samples_split)
        self.left.construct_tree(X[mask_left, :], y[mask_left])

        self.right = Tree(self._min_samples_split)
        self.right.construct_tree(X[mask_right, :], y[mask_right])

    def apply_tree_to_sample(self, x):
        """Perform regression on a single observation.

        Parameters
        ----------
        x : (num_features,) ndarray
            The vector of observations.

        Returns
        -------
        prediction : float
        """
        if self.threshold is None:
            return self.prediction
        if x[self.feature_index] < self.threshold:
            node = self.left
        else:
            node = self.right
        return node.apply_tree_to_sample(x)

    def apply(self, X):
        """Perform regression on a matrix of observations.

        Parameters
        ----------
        X : (num_samples, num_features) ndarray
            The matrix of observations.

        Returns
        -------
        predictions : (num_samples,) ndarray
            A vector of predictions for each individual observation.
        """
        return np.vectorize(self.apply_tree_to_sample, signature="(m)->()")(X)


class DecisionTree(BaseEstimator, RegressorMixin):
    """A binary decision tree regressor class.

    Parameters
    ----------
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    """

    def __init__(self, min_samples_split=2):
        self.min_samples_split_ = min_samples_split

        self._tree = None

    def fit(self, X, y):
        """Fit the decision tree to a given set of observations and targets.

        Parameters
        ----------
        X : (num_samples, num_features) array_like
            The matrix of observations.
        y : (num_samples,) array_like
            The vector of targets correponding to the rows of `X`.

        Returns
        -------
        self : DecisionTree
        """
        self._tree = Tree(self.min_samples_split_)
        self._tree.construct_tree(*map(np.array, (X, y)))
        return self

    def predict(self, X):
        """Perform prediction on a matrix of observations.

        Parameters
        ----------
        X : (num_samples, num_features) array_like
            The matrix of observations.

        Returns
        -------
        predictions : (num_samples,) ndarray
            A vector of predictions for each individual observation.
        """
        if self._tree is None:
            raise RuntimeError("Tree needs to be fitted first")
        return self._tree.apply(np.array(X))
