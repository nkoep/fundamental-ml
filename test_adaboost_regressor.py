from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor

from example_runner import run_example
from boosting import AdaBoost


class AdaBoostRegressor(_AdaBoostRegressor):
    """Convenience wrapper class to run sklearn's AdaBoostRegressor with
    decision tree stumps of depth 1.
    """

    def __init__(self, **kwargs):
        kwargs["base_estimator"] = DecisionTreeRegressor(max_depth=1)
        super().__init__(**kwargs)


regressors = [("sklearn", AdaBoostRegressor),
              ("naive", AdaBoost)]
run_example(regressors, n_estimators=25, random_state=0)
