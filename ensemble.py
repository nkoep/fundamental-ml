import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from tree import Tree


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
        self.n_estimators_ = n_estimators
        self.min_samples_split_ = min_samples_split
        self.max_features_ = max_features
        self.random_state_ = random_state

        self._trees = []

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

        rng = np.random.default_rng(self.random_state_)
        num_samples = X.shape[0]

        for _ in range(self.n_estimators_):
            tree = Tree(
                min_samples_split=self.min_samples_split_,
                max_features=self.max_features_,
                rng=rng)
            indices = rng.integers(num_samples, size=num_samples)
            tree.construct_tree(X[indices, :], y[indices])
            self._trees.append(tree)

        return self

    def _predict_sample(self, x):
        """Perform prediction on a single sample.

        Parameters
        ----------
        x : (num_samples,) ndarray
            A vector representing a single observation.

        Returns
        -------
        prediction : float
            The prediction result for the observation `x`.
        """
        return np.array(
            [tree.apply_to_sample(x) for tree in self._trees]).mean()

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
        if not self._trees:
            raise RuntimeError("Estimator needs to be fitted first")
        return np.array([self._predict_sample(row) for row in np.array(X)])
