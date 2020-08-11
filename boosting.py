import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from tree import DecisionTree


class Stump(DecisionTree):
    def __init__(self, **kwargs):
        super().__init__(max_depth=1, **kwargs)


class AdaBoost(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators_ = n_estimators
        self.random_state_ = random_state

        self._reset_stumps()

    def _reset_stumps(self):
        self._stump_weights = np.zeros(self.n_estimators_)
        self._stumps = []

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
        self : AdaBoost
        """
        X, y = map(np.array, (X, y))
        num_samples = X.shape[0]
        sample_weights = np.ones(num_samples) / num_samples

        self._reset_stumps()

        rng = np.random.default_rng(self.random_state_)
        X_resampled, y_resampled = X, y

        for i in range(self.n_estimators_):
            stump = Stump(random_state=self.random_state_)
            stump.fit(X_resampled, y_resampled)
            predictions = stump.predict(X)

            # TODO: Update sample weights.

            # TODO: Determine weight of the stump.
            self._stumps.append(stump)

            # Resample training set for the next iteration.
            indices = rng.choice(
                np.arange(num_samples), size=num_samples, replace=True,
                p=sample_weights)
            X_resampled = X[indices, :]
            y_resampled = y[indices, :]

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
        if not self._stumps:
            raise RuntimeError("Estimator needs to be fitted first")

        y = 0
        for weight, stump in zip(self._stump_weights, self._stumps):
            y += weight * stump.predict(X)
        return y
