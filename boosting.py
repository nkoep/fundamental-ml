import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from random_state import ensure_random_state
from tree import DecisionTree


class Sprout(DecisionTree):
    """A shallow decision tree regressor.

    The default estimator in sklearn's AdaBoostRegressor is not a decision tree
    stump (that is, a tree of depth 1) but a decision tree regressor of depth
    3. We refer to this as a 'sprout' rather than a 'stump'.
    """

    def __init__(self, **kwargs):
        super().__init__(max_depth=3, **kwargs)


class AdaBoost(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._reset_sprouts()

    def _reset_sprouts(self):
        self.sprout_weights_ = np.zeros(self.n_estimators)
        self.sprouts_ = []

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

        self._reset_sprouts()

        random_state = ensure_random_state(self.random_state)

        for i in range(self.n_estimators):
            # Resample the training set.
            indices = random_state.choice(
                np.arange(num_samples), size=num_samples, replace=True,
                p=sample_weights)
            X_resampled = X[indices, :]
            y_resampled = y[indices]

            # Train a weak learner on the resampled training data.
            sprout = Sprout(random_state=random_state)
            sprout.fit(X_resampled, y_resampled)

            predictions = sprout.predict(X)
            prediction_errors = np.abs(y - predictions)
            average_loss = np.inner(
                prediction_errors / prediction_errors.max(), sample_weights)

            beta = average_loss / (1 - average_loss)
            self.sprout_weights_[i] = np.log(1 / beta)
            self.sprouts_.append(sprout)

            # Update sample weights.
            weights = sample_weights * beta ** (1 - average_loss)
            sample_weights = weights / weights.sum()

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
        if not self.sprouts_:
            raise RuntimeError("Estimator needs to be fitted first")

        predictions = np.array([
            sprout.predict(X) for sprout in self.sprouts_])
        return np.median(predictions, axis=0)