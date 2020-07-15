import time

from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class Timer:
    def __enter__(self, *args, **kwargs):
        self._start_time = time.time()

    def __exit__(self, *args, **kwargs):
        print(f"Time elapsed: {time.time() - self._start_time :.6f} seconds")
        print()


def run_example(regressors, **parameters):
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for implementation, Regressor in regressors:
        regressor = Regressor(**parameters)
        with Timer():
            regressor.fit(X_train, y_train)
            prediction = regressor.predict(X_test)
            print(implementation)
            print("-" * len(implementation))
            print(f"MAE: {mean_absolute_error(prediction, y_test)}")
            print(f"R^2 score: {r2_score(prediction, y_test)}")
