from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from example_runner import run_example
from boosting import AdaBoost


regressors = [("sklearn", AdaBoostRegressor),
              ("naive", AdaBoost)]
run_example(regressors, n_estimators=25, random_state=0)
