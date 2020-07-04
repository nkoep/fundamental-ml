import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class _Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_index = None
        self.value = None

    def fit_tree(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape

        features = np.arange(num_features)
        samples = np.arange(samples)

        raise NotImplementedError

    def _apply_tree_to_sample(self, x):
        node = self
        while True:
            if node.left is None:
                assert node.right is None, "invalid tree"
                return node.value
            feature_value = x[node.feature_index]
            if feature_value <= node.left.value:
                node = node.left
            else:
                node = node.right
        raise RuntimeError("This mustn't happen")

    _apply_tree_to_samples = np.vectorize(_apply_tree_to_sample,
                                          signature="(m,n)->(m)")

    def apply(self, X):
        X = np.array(X)
        return self._apply_tree_to_samples(X)


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self._tree = None

    def fit(self, X, y):
        self._tree = _Node()
        self._tree.fit_tree(X, y)

    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("Tree needs to be fitted first")
        return self._tree.apply(X)
