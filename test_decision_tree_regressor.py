from sklearn.tree import DecisionTreeRegressor

from example_runner import run_example
from tree import DecisionTree


regressors = [("sklearn", DecisionTreeRegressor),
              ("naive", DecisionTree)]
run_example(regressors, random_state=0)
