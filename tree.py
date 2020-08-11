import numba
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


njit_cached = numba.njit(cache=True)


@njit_cached
def _score_split(y, partition):
    """Return the mean squared error of a potential split partition.

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


@njit_cached
def _find_best_split(x, y):
    """Find the best split for a vector of samples.

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
        score = _score_split(y, partition)

        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_partition = partition

    return best_score, best_threshold, best_partition


class Tree:
    """The fundamental data structure representing a binary decision tree.

    Parameters
    ----------
    max_depth : int or None
        The maximum allowed tree depth. In general, this requires pruning the
        tree to select the best subtree configuration. For simplicity, we only
        allow `max_depth=1`.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    max_features : int or None
        The size of the randomly selected subset of features to consider when
        splitting an internal node.
    random_state : numpy.random.Generator or None
        A pseudo random number generator to allow for reproducible tree
        construction when `max_features` is not None.

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

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None,
                 random_state=None):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_features = max_features
        self._random_state = random_state

        if self._max_depth is not None:
            assert self._max_depth == 1, "Only 'max_depth=1' allowed"
        if self._max_features is not None:
            assert self._random_state is not None, "No random state provided"

        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.prediction = None

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
        if num_samples < self._min_samples_split or self._max_depth == 1:
            self.prediction = y.mean()
            return

        if self._max_features is not None:
            feature_indices = self._random_state.integers(
                num_features, size=min(self._max_features, num_features))
        else:
            feature_indices = np.arange(num_features)

        # For each feature, compute the best split.
        feature_scores = {}
        for feature_index in feature_indices:
            x = X[:, feature_index]
            score, threshold, partition = _find_best_split(x, y)
            feature_scores[feature_index] = {
                "score": score,
                "threshold": threshold,
                "partition": partition
            }

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

        self.left = Tree(
            min_samples_split=self._min_samples_split,
            max_features=self._max_features,
            random_state=self._random_state)
        self.left.construct_tree(X[mask_left, :], y[mask_left])

        self.right = Tree(
            min_samples_split=self._min_samples_split,
            max_features=self._max_features,
            random_state=self._random_state)
        self.right.construct_tree(X[mask_right, :], y[mask_right])

    def apply_to_sample(self, x):
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
        return node.apply_to_sample(x)

    def apply(self, X):
        """Perform prediction on a matrix of observations.

        Parameters
        ----------
        X : (num_samples, num_features) ndarray
            The matrix of observations.

        Returns
        -------
        predictions : (num_samples,) ndarray
            A vector of predictions for each individual observation.
        """
        return np.array([self.apply_to_sample(row) for row in X])


class DecisionTree(BaseEstimator, RegressorMixin):
    """A binary decision tree regressor class.

    Parameters
    ----------
    max_depth : int or None
        The maximum allowed tree depth. In general, this requires pruning the
        tree to select the best subtree configuration. For simplicity, we only
        allow `max_depth=1`.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    max_features : int or None
        The size of the randomly selected subset of features to consider when
        splitting an internal node.
    random_state : int or None
        The random state of the estimator. This parameter is currently ignored;
        it is only here for compatibility with scikit-learn.
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None

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
        self.tree_ = Tree(min_samples_split=self.min_samples_split)
        self.tree_.construct_tree(*map(np.array, (X, y)))
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
        if self.tree_ is None:
            raise RuntimeError("Estimator needs to be fitted first")
        return self.tree_.apply(np.array(X))
