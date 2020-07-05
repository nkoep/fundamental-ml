from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from tree import DecisionTree


def main():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    sklearn_regressor = DecisionTreeRegressor()
    naive_regressor = DecisionTree()

    for implementation, regressor in [("sklearn", sklearn_regressor),
                                      ("naive", naive_regressor)]:
        regressor.fit(X_train, y_train)
        prediction = regressor.predict(X_test)
        print(f"Mean absolute error ({implementation}): ",
              mean_absolute_error(prediction, y_test))


if __name__ == "__main__":
    main()
