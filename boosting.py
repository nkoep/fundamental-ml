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
    """An AdaBoost regressor based on shallow decision trees.

    While AdaBoost supports any regression algorithm, this implementation uses
    shallow decision trees of depth 3, which we refer to as 'sprouts'.

    Parameters
    ----------
    n_estimators : int
        The number of estimators in the ensemble.
    random_state : numpy.random.Generator or int or None
        The random state of the estimator to allow reproducible training.

    Attributes
    ----------
    sprout_weights_ : (n_estimators,) ndarray
        The estimator weights.
    sprouts_ : list of `Sprout` instances
        The trained estimators of the ensemble.
    """

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

            # Compute normalized losses and average loss.
            predictions = sprout.predict(X)
            prediction_errors = np.abs(y - predictions)
            prediction_errors /= prediction_errors.max()
            average_loss = np.inner(prediction_errors, sample_weights)

            # Early termination if loss is too bad.
            if average_loss >= 0.5:
                if len(self.sprouts_) == 0:
                    self.sprouts_.append(sprout)
                break

            # Update estimator weights.
            beta = average_loss / (1 - average_loss)
            self.sprout_weights_[i] = np.log(1 / beta)
            self.sprouts_.append(sprout)

            # Update sample weights.
            weights = sample_weights * beta ** (1 - prediction_errors)
            sample_weights = weights / weights.sum()

        return self

    def _weighted_median(self, weights, elements):
        sort_indices = np.argsort(elements)
        sorted_weights = weights[sort_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        index = (cumulative_weights >= 0.5 * np.sum(weights)).argmax()
        return elements[sort_indices[index]]

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

        predictions = np.array([sprout.predict(X)
                                for sprout in self.sprouts_]).T
        return np.array([self._weighted_median(self.sprout_weights_, row)
                         for row in predictions])

    def _predict(self, X):
        # This is an alternative but slightly less readable implementation of
        # weighted median for prediction that computes all predictions in one
        # go along the lines of sklearn's implementation.
        predictions = np.array([
            sprout.predict(X) for sprout in self.sprouts_])

        sort_indices = predictions.argsort(axis=0)
        sorted_weights = self.sprout_weights_[:, np.newaxis][
            sort_indices].squeeze(axis=-1)
        cumulative_weights = np.cumsum(sorted_weights, axis=0)
        weight_sums = cumulative_weights[-1, :]
        predictor_indices = (cumulative_weights >=
               0.5 * weight_sums[np.newaxis, :]).argmax(axis=0)

        num_samples = X.shape[0]
        selectors = np.arange(num_samples)
        return predictions[sort_indices[predictor_indices, selectors],
                           selectors]
