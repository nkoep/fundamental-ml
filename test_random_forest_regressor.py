from sklearn.ensemble import RandomForestRegressor

from example_runner import run_example
from ensemble import RandomForest


regressors = [("sklearn", RandomForestRegressor),
              ("naive", RandomForest)]
run_example(regressors, n_estimators=25, max_features=5, random_state=0)
