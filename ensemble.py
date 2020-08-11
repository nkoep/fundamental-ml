import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from random_state import ensure_random_state
from tree import DecisionTree


class RandomForest(BaseEstimator, RegressorMixin):
    """A random forest ensemble regressor.

    This class fits a number of randomized decision tree regressors, and
    averages the output of every tree to form its prediction result.

    Parameters
    ----------
    n_estimators : int
        The number of estimators to consider in the ensemble.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    max_features : int or None
        The size of the randomly selected subset of features to consider when
        splitting an internal node.
    random_state : int or None
        The random state of the estimator. Since each tree is fitted on a
        randomly subsampled collection of training examples, specifying a
        'random_state' explicitly allows for deterministic estimator training.
    """

    def __init__(self, n_estimators=100, min_samples_split=2,
                 max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []

    def fit(self, X, y):
        """Fit the random forest to a given set of observations and targets.

        Parameters
        ----------
        X : (num_samples, num_features) array_like
            The matrix of observations.
        y : (num_samples,) array_like
            The vector of targets correponding to the rows of `X`.

        Returns
        -------
        self : RandomForest
        """
        X, y = map(np.array, (X, y))

        random_state = ensure_random_state(self.random_state)
        num_samples = X.shape[0]

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=random_state)
            indices = random_state.integers(num_samples, size=num_samples)
            tree.fit(X[indices, :], y[indices])
            self.trees_.append(tree)

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
        if not self.trees_:
            raise RuntimeError("Estimator needs to be fitted first")

        predictions = 0
        for tree in self.trees_:
            predictions += tree.predict(X)
        return predictions / self.n_estimators
