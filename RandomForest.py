import numpy as np
import pandas as pd
import matplotlib as mtl


def train_random_forest(training_set, params):
    # TODO: train the random forest
    pass


def predict_using_trained_forest(trained_forest, test_set):
    # TODO: Prediction using the trained forest
    pass


def run_random_forest(df):
    # TODO: Implement random forest algorithm
    size = df.shape[0]
    test_size = int(size / 5)

    test_set = df.iloc[:test_size, :]
    training_set = df.iloc[test_size:, :]

    params = {
        "n_estimators": 0,
        "max_depth": 0,
        "max_features": 0,
        "min_samples_leaf": 0,
        "bootstrap": False,
        "criterion": "Null",
    }

    trained_forest = train_random_forest(training_set, params)
    result = predict_using_trained_forest(trained_forest, test_set)

    return result


def format_properly_and_print(result):
    # TODO: Print output properly based on result
    pass


def main():
    df = pd.read_csv("./database/All_Pokemon.csv")

    result = run_random_forest(df)
    format_properly_and_print(result)


if __name__ == '__main__':
    main()
