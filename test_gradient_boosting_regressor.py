from sklearn.ensemble import GradientBoostingRegressor

from example_runner import run_example
from boosting import GradientBoosting


regressors = [("sklearn", GradientBoostingRegressor),
              ("naive", GradientBoosting)]
run_example(regressors, n_estimators=50, random_state=0)
